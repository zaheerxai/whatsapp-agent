"""
llm_json.py — one tiny, resilient "ask an LLM for a JSON object" helper.

Why it exists:
  Groq rate limits are PER MODEL (free tier: gpt-oss-120b = 8K TPM / 1K RPD,
  gpt-oss-20b = 8K TPM / 1K RPD).
  NOTE: llama-3.1-8b-instant / llama-3.3-70b-versatile were shut down on
  2026-08-16 and qwen3-32b / llama-4-scout on 2026-07-17 (Groq deprecations page).
  Model ids are therefore verified against GET /models at runtime and any model
  that 404s is benched for an hour — a retired id can never silently break a chain.
  The main agent loop already saturates the 120b bucket (big system prompt +
  tool schemas per call), so every *side* LLM job (voice cleanup, reminder
  picking, ...) must live on a sibling model with its own bucket. Otherwise a
  single voice note causes 429s on the main reply (seen in prod logs).

Chain per call:  Groq model #1 -> Groq model #2 -> ... -> Gemini -> None
Returns a parsed dict or None. Never raises.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Dict, Optional, Sequence

log = logging.getLogger("mojo.llm_json")

# Side-job chain (voice cleanup, reminder picking, ...): NOT the main agent model.
SIDE_MODELS = [
    m.strip()
    for m in (os.getenv("SIDE_LLM_MODELS") or "openai/gpt-oss-20b,qwen/qwen3.6-27b").split(",")
    if m.strip()
]

_lock = threading.Lock()
_avail_cache: Dict[int, tuple] = {}   # id(client) -> (set[str] | None, expires)
_dead: Dict[str, float] = {}          # model -> benched-until


def available_models(client) -> Optional[set]:
    """Live model ids for this Groq client (cached 1h). None = unknown (don't filter)."""
    key, now = id(client), time.time()
    with _lock:
        ent = _avail_cache.get(key)
        if ent and ent[1] > now:
            return ent[0]
    ids = None
    try:
        ids = {m.id for m in client.models.list().data}
    except Exception as e:  # fakes / network: stay permissive
        log.debug("models.list unavailable: %s", str(e)[:120])
    with _lock:
        _avail_cache[key] = (ids, now + (3600 if ids else 120))
    return ids


def is_usable(client, model: str) -> bool:
    if _dead.get(model, 0) > time.time():
        return False
    ids = available_models(client)
    return ids is None or model in ids


def _bench_if_dead(model: str, err: Exception) -> None:
    s = str(err).lower()
    if "404" in s or "does not exist" in s or "model_not_found" in s or "decommission" in s:
        with _lock:
            _dead[model] = time.time() + 3600
        log.warning("model %s benched for 1h (%s)", model, str(err)[:100])


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I | re.M)


def parse_json_loose(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from model output; tolerate fences / stray prose."""
    s = (text or "").strip()
    if not s:
        return None
    s = _FENCE_RE.sub("", s).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _call(client, model: str, messages: list, max_tokens: int, timeout: float,
          json_mode: bool = True) -> str:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    # gpt-oss are reasoning models: keep thinking cheap for extraction jobs.
    if model.startswith("openai/gpt-oss"):
        kwargs["extra_body"] = {"reasoning_effort": "low"}
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def json_chat(
    system: str,
    user: str,
    *,
    groq_client=None,
    groq_models: Sequence[str] = (),
    gemini_client=None,
    gemini_model: Optional[str] = None,
    max_tokens: int = 500,
    timeout: float = 25.0,
    validator: Optional[Callable[[Dict[str, Any]], bool]] = None,
    tag: str = "llm_json",
) -> Optional[Dict[str, Any]]:
    """`system` MUST mention the word JSON (Groq json_object requirement)."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    def _accept(raw: str, who: str) -> Optional[Dict[str, Any]]:
        obj = parse_json_loose(raw)
        if obj is None:
            log.warning("%s: %s returned non-JSON", tag, who)
            return None
        if validator is not None:
            try:
                ok = bool(validator(obj))
            except Exception:
                ok = False
            if not ok:
                log.warning("%s: %s failed validation", tag, who)
                return None
        return obj

    if groq_client is not None:
        for model in groq_models:
            model = (model or "").strip()
            if not model or not is_usable(groq_client, model):
                continue
            try:
                out = _call(groq_client, model, messages, max_tokens, timeout)
                obj = _accept(out, model)
                if obj is not None:
                    return obj
            except Exception as e:
                # 429 / 5xx / unsupported param -> just try the next bucket
                _bench_if_dead(model, e)
                log.warning("%s: groq %s failed: %s", tag, model, str(e)[:160])

    if gemini_client is not None and gemini_model:
        for json_mode in (True, False):
            try:
                out = _call(gemini_client, gemini_model, messages, max_tokens,
                            timeout, json_mode=json_mode)
                obj = _accept(out, f"gemini(json_mode={json_mode})")
                if obj is not None:
                    return obj
                break
            except Exception as e:
                log.warning("%s: gemini failed (json_mode=%s): %s",
                            tag, json_mode, str(e)[:160])
                if "400" not in str(e):
                    break  # 503/429: a retry without json_mode won't help, only adds latency
    return None


def text_chat(
    system: str,
    user: str,
    *,
    groq_client=None,
    groq_models: Sequence[str] = (),
    gemini_client=None,
    gemini_model: Optional[str] = None,
    max_tokens: int = 1200,
    timeout: float = 40.0,
    prefer_gemini: bool = False,
    temperature: float = 0.2,
    tag: str = "text_chat",
) -> Optional[str]:
    """Plain-text sibling of json_chat (same chain). Use prefer_gemini for long inputs:
    Groq free tiers cap at ~8K tokens/min, Gemini takes the whole transcript."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _one(client, model) -> Optional[str]:
        kw: Dict[str, Any] = {"model": model, "messages": messages,
                              "temperature": temperature, "max_tokens": max_tokens,
                              "timeout": timeout}
        if model.startswith("openai/gpt-oss"):
            kw["extra_body"] = {"reasoning_effort": "low"}
        out = (client.chat.completions.create(**kw).choices[0].message.content or "").strip()
        return out or None

    def _groq() -> Optional[str]:
        if groq_client is None:
            return None
        for m in groq_models:
            m = (m or "").strip()
            if not m or not is_usable(groq_client, m):
                continue
            try:
                r = _one(groq_client, m)
                if r:
                    return r
            except Exception as e:
                _bench_if_dead(m, e)
                log.warning("%s: groq %s failed: %s", tag, m, str(e)[:160])
        return None

    def _gem() -> Optional[str]:
        if gemini_client is None or not gemini_model:
            return None
        try:
            return _one(gemini_client, gemini_model)
        except Exception as e:
            log.warning("%s: gemini failed: %s", tag, str(e)[:160])
            return None

    for step in ((_gem, _groq) if prefer_gemini else (_groq, _gem)):
        r = step()
        if r:
            return r
    return None
