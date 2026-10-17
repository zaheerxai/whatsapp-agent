"""
llm_router.py — single LLM routing layer: Groq primary -> Groq sibling -> Gemini.

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
"""
from __future__ import annotations

import json
import logging
import os
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


def _cool(model: str, seconds: float) -> None:
    with _lock:
        _cooldown[model] = time.time() + max(1.0, min(float(seconds), 60.0))


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
) -> Tuple[Any, str, str]:
    """Returns (message, finish_reason, provider_tag). Raises the last error if every provider fails."""
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
        chain.append(("gemini", gemini_client, gemini_model))

    last_err: Optional[Exception] = None
    working, work_tools = messages, tools
    for kind, client, model in chain:
        if kind == "groq":
            cd = _cooled(model)
            if cd > 0:
                log.info("LLM_SKIP model=%s reason=cooldown %.0fs", model, cd)
                continue
            hr = _headroom(model)
            if hr < est:
                log.info("LLM_SKIP model=%s reason=tpm est=%s headroom=%s", model, est, hr)
                continue
        flat = (kind == "gemini") or (model != groq_model and not work_tools)
        for attempt in range(2):
            try:
                msgs = flatten(working) if flat else working
                kw = _build_kwargs(kind, model, msgs, work_tools, temperature, max_tokens)
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
                if st == 413 and kind == "groq" and attempt == 0:
                    working, work_tools = shrink(working, 4), None
                    continue
                if st == 429 and kind == "groq":
                    _cool(model, ra if ra is not None else 20)
                    break  # next model has its own TPM bucket — don't sleep on this one
                if st == 400 and kind == "groq" and attempt == 0 and not flat:
                    flat, work_tools = True, None  # malformed tool turn: retry flattened, no tools
                    continue
                if kind == "gemini" and attempt == 0 and (st in (429, 500, 502, 503, 504)):
                    time.sleep(min(ra, 3.0) if ra else 1.5)
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
