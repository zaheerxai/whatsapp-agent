"""
llm_json.py — one tiny, resilient "ask an LLM for a JSON object" helper.

Why it exists:
  Groq rate limits are PER MODEL (free tier: gpt-oss-120b = 8K TPM / 1K RPD,
  gpt-oss-20b = 8K TPM / 1K RPD, llama-3.1-8b-instant = 6K TPM / 14.4K RPD).
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
from typing import Any, Callable, Dict, Optional, Sequence

log = logging.getLogger("mojo.llm_json")

# Side-job chain (voice cleanup, reminder picking, ...): NOT the main agent model.
SIDE_MODELS = [
    m.strip()
    for m in (os.getenv("SIDE_LLM_MODELS") or "openai/gpt-oss-20b,llama-3.1-8b-instant").split(",")
    if m.strip()
]

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
            if not model:
                continue
            try:
                out = _call(groq_client, model, messages, max_tokens, timeout)
                obj = _accept(out, model)
                if obj is not None:
                    return obj
            except Exception as e:
                # 429 / 5xx / unsupported param -> just try the next bucket
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
    return None
