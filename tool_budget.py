"""
tool_budget.py — offer the model only the tools a turn can plausibly need, when the full set will not fit.

Why (prod log 2026-11-03): the owner chat has every tool enabled. 24 schemas cost ~5.3k tokens and the
system prompt ~3.3k, so every owner turn was estimated at 11-12k tokens against Groq's 7.36k single-request
ceiling (8K TPM free tier x 0.92). Groq was skipped 100% of the time and Gemini (text-only here) carried it.

Design:
  * The decision is made by an LLM returning clean JSON ({"tools": [...]}) through llm_json.json_chat —
    no keyword/regex intent rules to maintain, and new tools are picked up automatically because the
    catalog is derived from the live TOOL_SCHEMAS (name + opening of the description).
  * It runs ONLY when the full tool set does not fit the Groq budget. Turns that already fit are untouched.
  * Fail-open: planner unavailable/invalid -> caller keeps the full tool set (previous behaviour).
  * Admin/owner enablement is unchanged. This narrows what is OFFERED for one turn, never what is allowed.
  * Planner models are tried in order and share the router's per-model cooldowns, so a Gemini incident
    costs one failed call, not one per turn.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

log = logging.getLogger("mojo.tools")

DEFAULT_MAX_TOOLS = 8
_SUMMARY_CHARS = 170


def tool_name(schema: Dict[str, Any]) -> str:
    return str(((schema or {}).get("function") or {}).get("name") or "")


def catalog(schemas: Sequence[Dict[str, Any]], summary_chars: int = _SUMMARY_CHARS) -> List[Dict[str, str]]:
    """Compact, always-current tool menu: name + the start of the description (which carries 'use when…')."""
    out: List[Dict[str, str]] = []
    for s in schemas or []:
        name = tool_name(s)
        if not name:
            continue
        desc = re.sub(r"\s+", " ", str(((s.get("function") or {}).get("description")) or "")).strip()
        if len(desc) > summary_chars:
            desc = desc[:summary_chars].rsplit(" ", 1)[0] + "…"
        out.append({"name": name, "use": desc})
    return out


def subset(schemas: Sequence[Dict[str, Any]], names: Sequence[str]) -> List[Dict[str, Any]]:
    """Schemas for `names`, in the ORIGINAL order (stable order keeps provider prompt-caching prefixes stable)."""
    want = set(names or [])
    return [s for s in schemas or [] if tool_name(s) in want]


_PLANNER_SYSTEM = (
    "You are a tool router for a WhatsApp assistant. Given the tool catalog and the user's latest message "
    "(with brief recent context), choose which tools the assistant may need to complete THIS turn. "
    'Reply with JSON only: {"tools": ["tool_name", ...]}.\n'
    "Rules:\n"
    "- Choose only names that appear in the catalog.\n"
    "- Include every tool plausibly needed, including natural companions: if a tool's description says to "
    "call another tool first or after, include both (e.g. look something up before deleting it; draft before "
    "send); include the matching tool for any link, reminder, memory, e-mail, file, weather or research ask.\n"
    "- Greetings, thanks and pure conversation need no tools: return {\"tools\": []}.\n"
    "- At most {max_tools} tools. When genuinely unsure whether a tool is needed, include it."
)


def plan_tools(
    schemas: Sequence[Dict[str, Any]],
    latest_user_text: str,
    recent_lines: Sequence[str] = (),
    *,
    gemini_client: Any,
    models: Sequence[str],
    json_chat: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
    is_cooled: Optional[Callable[[str], bool]] = None,
    mark_failed: Optional[Callable[[str, float], None]] = None,
    max_tools: int = DEFAULT_MAX_TOOLS,
    timeout: float = 6.0,
) -> Optional[List[str]]:
    """Returns validated tool names (possibly empty = no tools needed) or None if no planner answered."""
    if not schemas or gemini_client is None:
        return None
    if json_chat is None:
        import llm_json
        json_chat = llm_json.json_chat
    cat = catalog(schemas)
    known = {c["name"] for c in cat}
    user = json.dumps(
        {
            "catalog": cat,
            "recent": [str(x)[:300] for x in list(recent_lines)[-2:]],
            "latest_message": (latest_user_text or "")[:1500],
        },
        ensure_ascii=False,
    )

    def _valid(obj: Dict[str, Any]) -> bool:
        t = obj.get("tools")
        return isinstance(t, list) and all(isinstance(x, str) for x in t)

    system = _PLANNER_SYSTEM.replace("{max_tools}", str(int(max_tools)))
    for model in models:
        model = (model or "").strip()
        if not model:
            continue
        if is_cooled and is_cooled(model):
            continue
        obj = json_chat(system, user, gemini_client=gemini_client, gemini_model=model,
                        max_tokens=300, timeout=timeout, validator=_valid, tag="tool_planner")
        if obj is not None:
            names: List[str] = []
            for n in obj.get("tools") or []:
                n = n.strip()
                if n in known and n not in names:
                    names.append(n)
            return names[: int(max_tools)]
        if mark_failed:
            mark_failed(model, 30.0)
    return None
