"""
llm_router.py — single LLM routing layer: Groq primary -> Groq sibling -> Gemini model ladder.

Why this exists (prod log 2026-10-17):
  * Groq free tier is 8K TPM per model, sliding window, input + output counted. A 12K-char
    tool observation + tool schemas + max_tokens blows the bucket, so the old chain burned
    429, 429, models.list, 400, 429 (~10s, 5 wasted calls) before Gemini answered.
  * gpt-oss and Gemini (3.x) are reasoning models: thinking tokens are billed against
    max_tokens, so a 1024 cap truncated visible replies mid-list (finish_reason="length"),
    and nothing ever looked at finish_reason.
  * print() diagnostics (including the provider's 400 body) never reached the log file.

What it does:
  * Per-model TPM governor (real usage.total_tokens) -> skips a model that cannot fit the
    request instead of provoking a 429; honours retry-after via a per-model cooldown.
  * reasoning_effort="low" on synthesis turns (no tools) for gpt-oss and Gemini; Gemini gets a
    larger max_tokens because thinking is billed as output.
  * finish_reason="length" -> bounded continuation, so a long answer is completed, not cut.
  * Every call logs provider/model/finish/tokens/latency and failures log the response body.

Gemini resilience (prod log 2026-11-03 12:00: two Gemini 503s ended the whole turn):
  * Gemini capacity is per model, so a 503 "high demand" on one model fails over to the next
    model in a ladder (GEMINI_FALLBACK_MODELS) instead of re-hitting the saturated one.
  * SDK-level retries are disabled for these calls (with_options(max_retries=0)) so retries are
    only ours: the old stack was SDK retry x router retry = 4 POSTs on one overloaded model.
  * Transient failures (408/429/5xx, connection/timeout errors) fail over immediately; the last
    model in the ladder gets one jittered-backoff retry that honours retry-after.
  * A failed model is put on a short cooldown so the next user message skips it instead of
    paying the failure latency again; 404 (retired model id) cools it for an hour.
  * A wall-clock deadline (LLM_GEMINI_DEADLINE_S) bounds total time spent walking the ladder.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from collections import deque
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("mojo.llm")

GROQ_TPM_LIMIT = int(os.getenv("GROQ_TPM_LIMIT", "8000"))  # free tier gpt-oss-*: 8K TPM per model
_TPM_SAFETY = 0.92
_WINDOW_S = 60.0
_MAX_CONTINUATIONS = 2

# Gemini model ladder (capacity is per model). Override with GEMINI_FALLBACK_MODELS=a,b,c
_DEFAULT_GEMINI_FALLBACKS = "gemini-3.8-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite"
_GEMINI_DEADLINE_S = float(os.getenv("LLM_GEMINI_DEADLINE_S", "25"))
_GEMINI_COOLDOWN_S = float(os.getenv("LLM_GEMINI_COOLDOWN_S", "30"))
_RETIRED_COOLDOWN_S = 3600.0
_BACKOFF_BASE_S = 0.8
_BACKOFF_CAP_S = 4.0
_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_NET_ERRORS = frozenset({"APIConnectionError", "APITimeoutError", "ReadTimeout", "ConnectTimeout",
                         "ConnectError", "RemoteProtocolError", "TimeoutException"})

_lock = threading.Lock()
_used: Dict[str, deque] = {}
_cooldown: Dict[str, float] = {}


# --------------------------------------------------------------------------- estimation / governor
def est_tokens(messages: List[dict], tools: Optional[list] = None, max_out: int = 0) -> int:
    chars = 0
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, str):
            chars += len(c)
        elif c:
            chars += len(json.dumps(c, default=str)[:20000])
        if m.get("tool_calls"):
            chars += len(json.dumps(m["tool_calls"], default=str))
    if tools:
        chars += len(json.dumps(tools, default=str))
    return int(chars / 3.3) + 60 + int(max_out or 0)


def _headroom(model: str) -> int:
    now = time.time()
    with _lock:
        q = _used.setdefault(model, deque())
        while q and now - q[0][0] > _WINDOW_S:
            q.popleft()
        used = sum(t for _, t in q)
    return int(GROQ_TPM_LIMIT * _TPM_SAFETY) - used


def _record(model: str, tokens: int) -> None:
    with _lock:
        _used.setdefault(model, deque()).append((time.time(), int(tokens or 0)))


def _cooled(model: str) -> float:
    return max(0.0, _cooldown.get(model, 0.0) - time.time())


def _cool(model: str, seconds: float, cap: float = 60.0) -> None:
    with _lock:
        _cooldown[model] = time.time() + max(1.0, min(float(seconds), float(cap)))


def reset_state() -> None:  # tests
    with _lock:
        _used.clear()
        _cooldown.clear()


# --------------------------------------------------------------------------- error helpers
def _status(e: Exception) -> Optional[int]:
    st = getattr(e, "status_code", None)
    if isinstance(st, int):
        return st
    m = re.search(r"\b(4\d\d|5\d\d)\b", str(e)[:80])
    return int(m.group(1)) if m else None


def _retry_after(e: Exception) -> Optional[float]:
    try:
        h = getattr(getattr(e, "response", None), "headers", None) or {}
        v = h.get("retry-after") or h.get("Retry-After")
        return float(v) if v is not None else None
    except Exception:
        return None


def err_detail(e: Exception) -> str:
    body = getattr(e, "body", None)
    if not body:
        resp = getattr(e, "response", None)
        body = getattr(resp, "text", "") if resp is not None else ""
    return f"{str(e)[:200]} | body={str(body)[:500]}"


def _retryable(e: Exception, st: Optional[int]) -> bool:
    return st in _RETRY_STATUS or type(e).__name__ in _NET_ERRORS


def _backoff(attempt: int, ra: Optional[float]) -> float:
    """Full-jitter exponential backoff, never shorter than a provider-supplied retry-after (capped)."""
    top = min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * (2 ** attempt))
    d = random.uniform(top * 0.5, top)
    return max(d, min(float(ra), _BACKOFF_CAP_S)) if ra else d


def _no_sdk_retry(client: Any) -> Any:
    """OpenAI SDK retries on its own (default 2, or whatever the client was built with). Turn that off so
    every retry is a router decision and logged. Fakes/clients without with_options pass through."""
    try:
        wo = getattr(client, "with_options", None)
        return wo(max_retries=0) if callable(wo) else client
    except Exception:  # noqa: BLE001
        return client


def _gemini_models(primary: Optional[str], fallbacks: Optional[List[str]]) -> List[str]:
    """Ordered, de-duplicated Gemini ladder with cooled-down models removed. If every model is cooling
    down, probe the one that recovers soonest rather than failing without trying."""
    if not primary:
        return []
    if fallbacks is None:
        fallbacks = [x for x in re.split(r"[,\s]+", os.getenv("GEMINI_FALLBACK_MODELS", _DEFAULT_GEMINI_FALLBACKS)) if x]
    ladder: List[str] = []
    for m in [primary, *fallbacks]:
        m = (m or "").strip()
        if m and m not in ladder:
            ladder.append(m)
    live = [m for m in ladder if _cooled(m) <= 0]
    if live:
        for m in ladder:
            if m not in live:
                log.info("LLM_SKIP model=%s reason=cooldown %.0fs", m, _cooled(m))
        return live
    return [min(ladder, key=_cooled)]


# --------------------------------------------------------------------------- core
def _build_kwargs(kind: str, model: str, messages: List[dict], tools: Optional[list],
                  temperature: float, max_tokens: int) -> Dict[str, Any]:
    kw: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
    if kind == "gemini":
        # thinking is billed as output -> give it room, and keep it cheap
        kw["max_tokens"] = min(8192, max(int(max_tokens) * 2, 2048))
        kw["extra_body"] = {"reasoning_effort": "low"}
        return kw
    kw["max_tokens"] = int(max_tokens)
    if tools:
        kw["tools"] = tools
        kw["tool_choice"] = "auto"
    elif "gpt-oss" in model:
        kw["extra_body"] = {"reasoning_effort": "low"}  # synthesis turn: don't burn the cap on thinking
    return kw


def complete(
    messages: List[dict],
    tools: Optional[list],
    temperature: float,
    max_tokens: int,
    *,
    groq_client: Any,
    groq_model: Optional[str],
    groq_sibling: Optional[str],
    gemini_client: Any,
    gemini_model: Optional[str],
    shrink: Callable[[List[dict], int], List[dict]],
    flatten: Callable[[List[dict]], List[dict]],
    is_usable: Optional[Callable[[Any, str], bool]] = None,
    gemini_fallbacks: Optional[List[str]] = None,
) -> Tuple[Any, str, str]:
    """Returns (message, finish_reason, provider_tag). Raises the last error if every provider fails.
    gemini_fallbacks=None -> GEMINI_FALLBACK_MODELS env (or the built-in ladder); [] -> primary only."""
    est = est_tokens(messages, tools, max_tokens)
    chain: List[Tuple[str, Any, str]] = []
    for m in (groq_model, groq_sibling):
        if groq_client and m and all(m != c[2] for c in chain):
            try:
                if is_usable is None or is_usable(groq_client, m):
                    chain.append(("groq", groq_client, m))
            except Exception:
                chain.append(("groq", groq_client, m))
    if gemini_client and gemini_model:
        for gm in _gemini_models(gemini_model, gemini_fallbacks):
            chain.append(("gemini", gemini_client, gm))

    last_err: Optional[Exception] = None
    working, work_tools = messages, tools
    gem_deadline: Optional[float] = None
    prev_gem: Optional[str] = None
    for idx, (kind, client, model) in enumerate(chain):
        has_next_gemini = any(c[0] == "gemini" for c in chain[idx + 1:])
        if kind == "groq":
            cd = _cooled(model)
            if cd > 0:
                log.info("LLM_SKIP model=%s reason=cooldown %.0fs", model, cd)
                continue
            hr = _headroom(model)
            if hr < est:
                log.info("LLM_SKIP model=%s reason=tpm est=%s headroom=%s", model, est, hr)
                continue
        else:
            now = time.time()
            if gem_deadline is None:
                gem_deadline = now + _GEMINI_DEADLINE_S
            elif now >= gem_deadline:
                log.warning("LLM_SKIP model=%s reason=deadline %.0fs", model, _GEMINI_DEADLINE_S)
                break
            if prev_gem:
                log.info("LLM_FALLBACK from=%s to=%s", prev_gem, model)
            prev_gem = model
            client = _no_sdk_retry(client)
        flat = (kind == "gemini") or (model != groq_model and not work_tools)
        drop_reasoning = False
        for attempt in range(2):
            if attempt > 0 and kind == "groq" and _headroom(model) < est_tokens(working, work_tools, max_tokens):
                log.info("LLM_SKIP model=%s reason=tpm-after-failure", model)
                break
            kw: Dict[str, Any] = {}
            try:
                msgs = flatten(working) if flat else working
                kw = _build_kwargs(kind, model, msgs, work_tools, temperature, max_tokens)
                if drop_reasoning:
                    kw.pop("extra_body", None)
                t0 = time.time()
                resp = client.chat.completions.create(**kw)
                choice = resp.choices[0]
                msg = choice.message
                finish = (getattr(choice, "finish_reason", None) or "stop").lower()
                usage = getattr(resp, "usage", None)
                total = getattr(usage, "total_tokens", None) or est
                if kind == "groq":
                    _record(model, total)
                log.info("LLM_OK provider=%s model=%s finish=%s tokens=%s est=%s ms=%d",
                         kind, model, finish, total, est, int((time.time() - t0) * 1000))
                return msg, finish, kind
            except Exception as e:  # noqa: BLE001
                last_err = e
                st, ra = _status(e), _retry_after(e)
                log.warning("LLM_FAIL provider=%s model=%s attempt=%d status=%s retry_after=%s %s",
                            kind, model, attempt + 1, st, ra, err_detail(e))
                if kind == "gemini":
                    if st == 404:  # retired / mistyped model id: stop paying for it on every request
                        _cool(model, _RETIRED_COOLDOWN_S, cap=_RETIRED_COOLDOWN_S)
                        break
                    if st == 400 and attempt == 0 and "extra_body" in kw and not drop_reasoning:
                        drop_reasoning = True  # this model rejects reasoning_effort: same model, without it
                        continue
                    if _retryable(e, st):
                        if has_next_gemini:
                            # capacity is per model: don't re-hit the saturated one, go to the next
                            _cool(model, ra if ra else _GEMINI_COOLDOWN_S)
                            break
                        delay = _backoff(attempt, ra)
                        if attempt == 0 and gem_deadline is not None and time.time() + delay < gem_deadline:
                            time.sleep(delay)
                            continue
                        _cool(model, ra if ra else _GEMINI_COOLDOWN_S)
                    break
                if st == 413 and kind == "groq" and attempt == 0:
                    working, work_tools = shrink(working, 4), None
                    continue
                if st == 429 and kind == "groq":
                    _cool(model, ra if ra is not None else 20)
                    break  # next model has its own TPM bucket — don't sleep on this one
                if st == 400 and kind == "groq" and "tool_use_failed" in str(e):
                    # model emitted a tool call we did not offer (prod: forget_knowledge with tools=None).
                    # Flattening won't help and just burns TPM on the same model — go to the next provider.
                    break
                if st == 400 and kind == "groq" and attempt == 0 and not flat:
                    flat, work_tools = True, None  # malformed tool turn: retry flattened, no tools
                    continue
                break
    raise last_err or RuntimeError("no LLM provider available")


def complete_full(messages: List[dict], tools: Optional[list], temperature: float, max_tokens: int,
                  **provider_kw: Any) -> Any:
    """complete() + bounded continuation when the provider stopped on the token cap."""
    msg, finish, _ = complete(messages, tools, temperature, max_tokens, **provider_kw)
    if finish != "length" or getattr(msg, "tool_calls", None):
        return msg
    text = getattr(msg, "content", None) or ""   # raw: a trailing newline is the item boundary
    if not text.strip():
        return msg
    for i in range(_MAX_CONTINUATIONS):
        cont_msgs = list(messages) + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": "Your previous message was cut off by the length limit. Continue "
                                        "exactly where it stopped. No preamble, no repetition, same format."},
        ]
        try:
            m2, f2, _ = complete(cont_msgs, None, temperature, max_tokens, **provider_kw)
        except Exception as e:  # noqa: BLE001
            log.warning("LLM_CONTINUE_FAIL i=%d %s", i + 1, err_detail(e))
            break
        extra = getattr(m2, "content", None) or ""
        if not extra.strip():
            break
        text += extra
        log.info("LLM_CONTINUED i=%d added=%d finish=%s", i + 1, len(extra), f2)
        if f2 != "length":
            break
    return SimpleNamespace(content=text.strip(), tool_calls=None, role="assistant")
