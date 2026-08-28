"""
agent_loop.py — ReAct-style tool-calling agent for Mojo.

Replaces the old single-shot get_ai_response + keyword reminder routing.
Hardened for Groq / Gemini tool-call format quirks.
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from agent_tools import TOOL_SCHEMAS, execute_tool

# Injected
_client_ai = None
_client_gemini = None
_MODEL_NAME = None
_GEMINI_MODEL = None
_BUSINESS_KNOWLEDGE = ""
_DEFAULT_TIMEZONE = "Asia/Karachi"
_get_user_timezone = None
_get_world_clocks = None
_fetch_chat_history = None
_get_contacts_maps = None
_get_group_memory = None
_get_tzinfo = None

MAX_TOOL_STEPS = 6


def init_agent(
    client_ai,
    client_gemini,
    model_name,
    gemini_model,
    business_knowledge,
    default_timezone,
    get_user_timezone,
    get_world_clocks,
    fetch_chat_history,
    get_contacts_maps,
    get_group_memory,
    get_tzinfo,
):
    global _client_ai, _client_gemini, _MODEL_NAME, _GEMINI_MODEL
    global _BUSINESS_KNOWLEDGE, _DEFAULT_TIMEZONE
    global _get_user_timezone, _get_world_clocks, _fetch_chat_history
    global _get_contacts_maps, _get_group_memory, _get_tzinfo

    _client_ai = client_ai
    _client_gemini = client_gemini
    _MODEL_NAME = model_name
    _GEMINI_MODEL = gemini_model
    _BUSINESS_KNOWLEDGE = business_knowledge or ""
    _DEFAULT_TIMEZONE = default_timezone
    _get_user_timezone = get_user_timezone
    _get_world_clocks = get_world_clocks
    _fetch_chat_history = fetch_chat_history
    _get_contacts_maps = get_contacts_maps
    _get_group_memory = get_group_memory
    _get_tzinfo = get_tzinfo


def _build_system_prompt(
    chat_id: str,
    sender_id: str,
    is_group: bool,
    time_context: str,
    tag_block: str,
    memory_block: str,
) -> str:
    env = (
        "ENVIRONMENT: GROUP CHAT.\n"
        "BEHAVIOR: Informal, intelligent, friendly, natural. Never rigid sales mode."
        if is_group
        else "ENVIRONMENT: PRIVATE CHAT.\n"
        "BEHAVIOR: Professional, warm, helpful."
    )

    return f"""You are Mojo, the official AI Assistant for Mojo AI Agency (Founder: Muhammad Zaheer).

{env}

{time_context}

{tag_block}

{memory_block}

=== LANGUAGE POLICY (CRITICAL) ===
- Default reply language for Urdu / Hindi / mixed users = **Roman Urdu** (Latin script), e.g. "Theek hai, 1 minute baad paani peene ka reminder set kar diya."
- Use full Urdu script (نستعلیق / Arabic letters) **ONLY** if the user explicitly asks for it (e.g. "Urdu mein likho", "اردو میں جواب دو").
- If the user writes pure English, reply in natural English.
- Match the user's vibe: casual when they are casual.

=== CORE RULES (SPEED + ACCURACY) ===
1. Keep replies SHORT — WhatsApp friendly (1–2 short lines max). No walls of text. No bio dumps.
2. Do NOT invent timestamps, brackets around names, or internal IDs in your final reply.
3. GREETINGS / SMALL TALK ("hi", "hello", "hows it going", "kya haal", "salam") → reply naturally in ONE short line. Do NOT call any tool. Do NOT dump agency stats or founder bio.
4. Call search_knowledge ONLY when the user actually asks about agency services, portfolio, founder background, pricing, or "what can you do". Never on a plain greeting.
5. For ANY reminder create / list / cancel intent — including Roman Urdu like "remind karna", "1 min me paani", "list reminder", "cancel karo", "paani wale cancel" — you MUST call set_reminder / list_reminders / cancel_reminders. Never pretend you set a reminder without the tool. Never suggest "set a phone timer instead".
6. cancel_reminders understands keywords like "paani", "water", "debug", or "all"/"sab". Prefer calling it over asking clarifying questions when intent is clear.
7. After tools finish, give a natural confirmation or answer. If the tool observation is already a clean confirmation, you may lightly polish or just relay it — do not call more tools.
8. You have tools. Use them when they help accuracy. Prefer tools over guessing times, facts, or contacts. Prefer ZERO tools when the answer is pure conversation.

=== URL / WEB FACTS (NO HALLUCINATION) ===
9. If the CURRENT message has a URL (or force_urls / priority system note lists one), or the user says "details", "iska", "ye", "batao", "fetch", "open", "latest repo" about a link → you MUST call browse_url on THAT URL BEFORE answering. Never invent page content.
10. When the user says "fetch latest repo" after a GitHub profile link, call browse_url on that exact github.com/username URL.
11. If a tool returns an error or empty data, say so honestly. Do not fabricate fallback facts.
12. For weather / temperature / mausam (e.g. Islamabad kitna garam hai), ALWAYS call get_weather — not web_search.

Agency knowledge is available via the search_knowledge tool (only when asked).
Brief agency summary:
{_BUSINESS_KNOWLEDGE[:800]}
"""


def _normalize_tool_calls(msg) -> List[Any]:
    """Return a list of tool-call-like objects with .id, .function.name, .function.arguments."""
    raw = getattr(msg, "tool_calls", None)
    if not raw:
        # Some providers put function_call (singular, legacy)
        fc = getattr(msg, "function_call", None)
        if fc:
            class _FC:
                pass
            o = _FC()
            o.id = "call_legacy_0"
            o.function = fc
            return [o]
        return []

    normalized = []
    for i, tc in enumerate(raw):
        # Already object style
        if hasattr(tc, "function") and hasattr(tc.function, "name"):
            if not getattr(tc, "id", None):
                try:
                    tc.id = f"call_{i}"
                except Exception:
                    pass
            normalized.append(tc)
            continue
        # Dict style
        if isinstance(tc, dict):
            class _Fn:
                pass
            class _Tc:
                pass
            fn = _Fn()
            fn.name = (tc.get("function") or {}).get("name") or tc.get("name") or ""
            fn.arguments = (tc.get("function") or {}).get("arguments") or tc.get("arguments") or "{}"
            o = _Tc()
            o.id = tc.get("id") or f"call_{i}"
            o.function = fn
            normalized.append(o)
    return normalized


def _sanitize_messages_for_gemini(messages: List[dict]) -> List[dict]:
    """Gemini OpenAI-compat is picky about content=None and long tool histories.
    Convert pure tool turns into plain text so the fallback can still answer."""
    out: List[dict] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "tool":
            # Fold tool observation into a user-visible note
            out.append({
                "role": "user",
                "content": f"[Tool result]\n{content or ''}",
            })
            continue
        if role == "assistant" and m.get("tool_calls"):
            # Keep any text the model already produced; drop raw tool_calls for Gemini
            text = (content or "").strip()
            names = []
            for tc in m.get("tool_calls") or []:
                try:
                    names.append((tc.get("function") or {}).get("name") or "tool")
                except Exception:
                    names.append("tool")
            if not text:
                text = f"(called {', '.join(names)})"
            out.append({"role": "assistant", "content": text})
            continue
        # Ensure content is always a string (never None)
        out.append({
            "role": role if role in ("system", "user", "assistant") else "user",
            "content": content if isinstance(content, str) else (content or ""),
        })
    return out


def _chat_completion(messages: List[dict], tools: Optional[List] = None, temperature: float = 0.4):
    """Primary Groq (with 429 retry), fallback Gemini. Returns the message object."""
    import time as _time

    kwargs: Dict[str, Any] = {
        "model": _MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    last_err = None
    for attempt in range(3):
        try:
            resp = _client_ai.chat.completions.create(**kwargs)
            return resp.choices[0].message
        except Exception as primary_err:
            last_err = primary_err
            err_s = str(primary_err).lower()
            is_rate = "429" in err_s or "rate" in err_s or "too many" in err_s
            if is_rate and attempt < 2:
                sleep_s = 1.2 * (attempt + 1)
                print(f"[AGENT] Groq 429/rate — retry in {sleep_s:.1f}s (attempt {attempt+1})")
                _time.sleep(sleep_s)
                continue
            print(f"[AGENT] Primary model failed: {primary_err}. Trying Gemini...")
            break

    # Gemini fallback — sanitize tool messages so we don't get 400s
    try:
        clean = _sanitize_messages_for_gemini(messages)
        gkwargs: Dict[str, Any] = {
            "model": _GEMINI_MODEL,
            "messages": clean,
            "temperature": temperature,
        }
        # Prefer no tools on Gemini; we already have observations folded in
        resp = _client_gemini.chat.completions.create(**gkwargs)
        return resp.choices[0].message
    except Exception as e2:
        print(f"[AGENT] Gemini fallback failed: {e2}")
        # Last resort: tiny prompt with last user + last tool obs only
        try:
            slim = [m for m in messages if m.get("role") == "system"][:1]
            for m in reversed(messages):
                if m.get("role") in ("user", "tool", "assistant") and m.get("content"):
                    slim.append({"role": "user" if m["role"] == "tool" else m["role"], "content": str(m["content"])[:2000]})
                    if len(slim) >= 4:
                        break
            slim = list(reversed(slim[1:])) + slim[:1]  # rough order restore
            slim = _sanitize_messages_for_gemini(slim)
            resp = _client_gemini.chat.completions.create(
                model=_GEMINI_MODEL,
                messages=slim or [{"role": "user", "content": "Reply briefly that there was a temporary issue."}],
                temperature=temperature,
            )
            return resp.choices[0].message
        except Exception as e3:
            print(f"[AGENT] All models failed: {e3}")
            raise last_err or e3


def run_agent(
    chat_id: str,
    sender_id: str,
    sender_num: Optional[str] = None,
    history_limit: int = 20,
    is_group: bool = False,
    msg_time: Optional[float] = None,
    extra_user_note: Optional[str] = None,
    force_urls: Optional[List[str]] = None,
    latest_user_text: Optional[str] = None,
) -> str:
    """
    Full agentic turn. Returns final natural-language reply for WhatsApp.
    """
    try:
        now_utc = datetime.now(timezone.utc)
        utc_time_str = now_utc.strftime("%A, %Y-%m-%d %I:%M %p UTC")
        world_clocks_str = _get_world_clocks() if _get_world_clocks else ""

        user_tz = _get_user_timezone(sender_id) if _get_user_timezone else None
        if user_tz:
            try:
                tz = ZoneInfo(str(user_tz)) if not str(user_tz).replace(".", "").replace("-", "").replace("+", "").isdigit() else None
                if tz is None:
                    # numeric offset stored historically
                    from datetime import timedelta
                    offset = float(user_tz)
                    time_obj = now_utc + timedelta(hours=offset)
                    formatted_time = time_obj.strftime("%A, %I:%M %p")
                    time_context = (
                        f"CURRENT UTC TIME: {utc_time_str}\n"
                        f"THIS SENDER'S LOCAL TIME (approx UTC{offset:+g}): {formatted_time}.\n\n"
                        f"REAL-TIME WORLD CLOCKS:\n{world_clocks_str}"
                    )
                else:
                    time_obj = now_utc.astimezone(tz)
                    formatted_time = time_obj.strftime("%A, %I:%M %p")
                    time_context = (
                        f"CURRENT UTC TIME: {utc_time_str}\n"
                        f"THIS SENDER'S LOCAL TIME: {formatted_time} ({user_tz}).\n\n"
                        f"REAL-TIME WORLD CLOCKS:\n{world_clocks_str}\n\n"
                        "For time queries use the world clocks above — do not invent offsets."
                    )
            except Exception:
                time_context = f"CURRENT UTC TIME: {utc_time_str}\n\nWORLD CLOCKS:\n{world_clocks_str}"
        else:
            time_context = (
                f"CURRENT UTC TIME: {utc_time_str}\n\n"
                f"WORLD CLOCKS:\n{world_clocks_str}\n"
                "(Sender timezone unknown — if they set a reminder for the first time, ask for city/country once.)"
            )

        raw_history = _fetch_chat_history(chat_id, history_limit) or []
        contacts_map, reverse_map = _get_contacts_maps() if _get_contacts_maps else ({}, {})
        memory_notes = _get_group_memory(chat_id) if _get_group_memory else []
        memory_block = ""
        if memory_notes:
            memory_block = "PERMANENT NOTES FOR THIS CHAT:\n" + "\n".join(
                f"- {n}" for n in memory_notes
            )

        tag_block = ""
        if is_group:
            active = {
                contacts_map.get(m["sender_id"], m["sender_id"])
                for m in raw_history
                if m.get("sender_id") and m["sender_id"] != "mojo_agent"
            }
            if active:
                tag_block = (
                    "PEOPLE RECENTLY ACTIVE: "
                    + ", ".join(sorted(str(x) for x in active))
                    + ".\n"
                    "When tagging, use exact @Name from this list."
                )

        system = _build_system_prompt(
            chat_id, sender_id, is_group, time_context, tag_block, memory_block
        )

        messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]

        # Only replay clean user/assistant turns (never stale tool messages from DB)
        for msg in raw_history:
            role = msg.get("role") or "user"
            if role not in ("user", "assistant"):
                continue
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                name = contacts_map.get(msg.get("sender_id"), msg.get("sender_id", "?"))
                content = f"{name}: {content}"
            messages.append({"role": role, "content": content})

        if extra_user_note:
            messages.append({"role": "user", "content": extra_user_note})

        # Force-notice URLs from the CURRENT WhatsApp message (including quoted links)
        import re as _re
        urls_in_last = list(force_urls or [])
        if not urls_in_last:
            last_user = latest_user_text or ""
            if not last_user:
                for m in reversed(messages):
                    if m.get("role") == "user":
                        last_user = m.get("content") or ""
                        break
            urls_in_last = _re.findall(r"https?://[^\s<>\"\'\]\)]+", last_user)
        # de-dupe
        _seen = set()
        urls_in_last = [u for u in urls_in_last if not (u in _seen or _seen.add(u))]
        if urls_in_last:
            messages.append({
                "role": "system",
                "content": (
                    "PRIORITY: The user's LATEST message is about this URL(s): "
                    + ", ".join(urls_in_last)
                    + ". You MUST call browse_url on THIS URL as your FIRST (and usually only) tool. "
                    "Do NOT call list_reminders, set_reminder, or search_knowledge for this turn. "
                    "Ignore older links from chat history."
                ),
            })
            print(f"[AGENT] force browse_url for: {urls_in_last}")

        tool_ctx = {
            "chat_id": chat_id,
            "sender_id": sender_id,
            "sender_num": sender_num,
            "msg_time": msg_time,
            "is_group": is_group,
        }

        # --- Agentic loop ---
        for step in range(MAX_TOOL_STEPS):
            try:
                msg = _chat_completion(messages, tools=TOOL_SCHEMAS, temperature=0.3)
            except Exception as e:
                print(f"[AGENT] completion failed at step {step}: {e}")
                traceback.print_exc()
                # If we already have observations, try a no-tool close
                if step > 0:
                    try:
                        final = _chat_completion(messages, tools=None, temperature=0.4)
                        return (final.content or "").strip() or "Ho gaya."
                    except Exception:
                        pass
                return "Thori si technical issue aa gayi — ek second baad dobara try karo."

            tool_calls = _normalize_tool_calls(msg)

            # Build assistant message carefully (content=None when pure tool call)
            content = msg.content
            if content is not None:
                content = content.strip() or None

            assistant_entry: Dict[str, Any] = {"role": "assistant"}
            if content:
                assistant_entry["content"] = content
            else:
                assistant_entry["content"] = None

            if tool_calls:
                assistant_entry["tool_calls"] = []
                for tc in tool_calls:
                    fn_name = getattr(getattr(tc, "function", None), "name", None) or ""
                    fn_args = getattr(getattr(tc, "function", None), "arguments", None) or "{}"
                    if isinstance(fn_args, dict):
                        fn_args = json.dumps(fn_args)
                    assistant_entry["tool_calls"].append(
                        {
                            "id": getattr(tc, "id", None) or f"call_{step}",
                            "type": "function",
                            "function": {"name": fn_name, "arguments": fn_args},
                        }
                    )
            messages.append(assistant_entry)

            if not tool_calls:
                answer = (content or "").strip()
                return answer or "Theek hai 👍"

            # Execute tools
            for tc in tool_calls:
                name = getattr(getattr(tc, "function", None), "name", None) or ""
                raw_args = getattr(getattr(tc, "function", None), "arguments", None) or "{}"
                try:
                    if isinstance(raw_args, dict):
                        args = raw_args
                    else:
                        args = json.loads(raw_args or "{}")
                except json.JSONDecodeError:
                    args = {}
                print(f"[AGENT TOOL] step={step+1} {name}({args})")
                logging.getLogger("mojo.agent").info("TOOL %s %s", name, args)
                try:
                    observation = execute_tool(name, args, tool_ctx)
                except Exception as te:
                    traceback.print_exc()
                    observation = f"Tool error: {te}"
                logging.getLogger("mojo.agent").info("OBS %s", str(observation)[:500])
                print(f"[AGENT OBS] {str(observation)[:400]}{'…' if len(str(observation)) > 400 else ''}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": getattr(tc, "id", None) or f"call_{step}",
                        "content": str(observation),
                    }
                )

        # Max steps — force close without tools
        messages.append(
            {
                "role": "user",
                "content": "Give the best final short answer now from the tool results above. No more tools.",
            }
        )
        try:
            final = _chat_completion(messages, tools=None, temperature=0.4)
            return (final.content or "").strip() or "Ho gaya."
        except Exception as e:
            print(f"[AGENT] final close failed: {e}")
            return "Kaam almost complete ho gaya — list reminders dobara try kar lo."

    except Exception as e:
        traceback.print_exc()
        print(f"[AGENT ERROR] {e}")
        logging.getLogger("mojo.agent").exception("run_agent failed: %s", e)
        return "Thori si technical issue aa gayi — ek second baad dobara try karo."
