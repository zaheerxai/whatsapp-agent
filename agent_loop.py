"""
agent_loop.py — ReAct-style tool-calling agent for Mojo.

Replaces the old single-shot get_ai_response + keyword reminder routing.
"""

from __future__ import annotations

import json
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

=== CORE RULES ===
1. LANGUAGE MATCHING: If the user writes in Roman Urdu / Hindi / mixed Urdu-English (e.g. "kya haal hai", "paani peena hai", "cancel karo"), you MUST reply in the same natural casual style. Never force formal Urdu or pure English when they mixed.
2. Keep replies SHORT — WhatsApp friendly (1-3 short paragraphs max). No walls of text.
3. Do NOT invent timestamps, brackets around names, or internal IDs in your final reply.
4. When the user asks about agency services, portfolio, founder, or "what can you do", ALWAYS call search_knowledge first.
5. For any reminder create / list / cancel intent (including "remind me", "reminder set karo", "cancel karo", "list reminders", "paani wale reminders cancel", "1 min me ..."), you MUST use the set_reminder / list_reminders / cancel_reminders tools. Never pretend you set a reminder without calling the tool.
6. cancel_reminders understands keywords like "paani", "water", "debug", or "all"/"sab". Prefer calling it over asking clarifying questions when the intent is clear.
7. After tools finish, give a natural confirmation or answer. Do not dump raw JSON.
8. You have tools. Use them when they help accuracy. Prefer tools over guessing times, facts, or contacts.

Agency knowledge is available via the search_knowledge tool (do not rely only on the short summary below).
Brief agency summary (call tool for details):
{_BUSINESS_KNOWLEDGE[:1200]}
"""


def _chat_completion(messages: List[dict], tools: Optional[List] = None, temperature: float = 0.4):
    """Primary Groq, fallback Gemini. Returns the message object."""
    kwargs = {
        "model": _MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    try:
        resp = _client_ai.chat.completions.create(**kwargs)
        return resp.choices[0].message
    except Exception as primary_err:
        print(f"[AGENT] Primary model failed: {primary_err}. Trying Gemini...")
        # Gemini OpenAI-compat may not support tools the same way on all models;
        # try with tools first, then without.
        try:
            gkwargs = {
                "model": _GEMINI_MODEL,
                "messages": messages,
                "temperature": temperature,
            }
            if tools:
                gkwargs["tools"] = tools
                gkwargs["tool_choice"] = "auto"
            resp = _client_gemini.chat.completions.create(**gkwargs)
            return resp.choices[0].message
        except Exception as e2:
            print(f"[AGENT] Gemini with tools failed: {e2}")
            # Last resort: no tools
            try:
                resp = _client_gemini.chat.completions.create(
                    model=_GEMINI_MODEL,
                    messages=messages,
                    temperature=temperature,
                )
                return resp.choices[0].message
            except Exception as e3:
                print(f"[AGENT] All models failed: {e3}")
                raise


def run_agent(
    chat_id: str,
    sender_id: str,
    sender_num: Optional[str] = None,
    history_limit: int = 20,
    is_group: bool = False,
    msg_time: Optional[float] = None,
    extra_user_note: Optional[str] = None,
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
                tz = ZoneInfo(user_tz)
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
                "(Sender timezone unknown — if they set a reminder, ask for city/country once.)"
            )

        raw_history = _fetch_chat_history(chat_id, history_limit) or []
        contacts_map, reverse_map = _get_contacts_maps() if _get_contacts_maps else ({}, {})
        memory_notes = _get_group_memory(chat_id) if _get_group_memory else []
        memory_block = ""
        if memory_notes:
            memory_block = "PERMANENT NOTES FOR THIS CHAT:\n" + "\n".join(
                f"- {n}" for n in memory_notes
            )

        # Tag block (lightweight — full group cache stays in main file)
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

        for msg in raw_history:
            content = msg.get("content") or ""
            role = msg.get("role") or "user"
            if role == "user":
                name = contacts_map.get(msg.get("sender_id"), msg.get("sender_id", "?"))
                content = f"{name}: {content}"
            messages.append({"role": role, "content": content})

        if extra_user_note:
            messages.append({"role": "user", "content": extra_user_note})

        tool_ctx = {
            "chat_id": chat_id,
            "sender_id": sender_id,
            "sender_num": sender_num,
            "msg_time": msg_time,
            "is_group": is_group,
        }

        # --- Agentic loop ---
        for step in range(MAX_TOOL_STEPS):
            msg = _chat_completion(messages, tools=TOOL_SCHEMAS, temperature=0.35)
            # Normalize tool_calls across providers
            tool_calls = getattr(msg, "tool_calls", None) or []

            # Append assistant message (with tool_calls if any)
            assistant_entry: Dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }
            if tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_entry)

            if not tool_calls:
                # Final natural answer
                answer = (msg.content or "").strip()
                return answer or "Got that 👍"

            # Execute each tool
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                print(f"[AGENT TOOL] step={step+1} {name}({args})")
                observation = execute_tool(name, args, tool_ctx)
                print(f"[AGENT OBS] {observation[:300]}{'…' if len(observation) > 300 else ''}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": observation,
                    }
                )

        # Max steps reached — force a closing answer without tools
        messages.append(
            {
                "role": "system",
                "content": "You have used the maximum number of tool steps. Give the best final answer now from the observations you already have. No more tools.",
            }
        )
        final = _chat_completion(messages, tools=None, temperature=0.4)
        return (final.content or "").strip() or "Done."

    except Exception as e:
        traceback.print_exc()
        print(f"[AGENT ERROR] {e}")
        return "I'm having a little trouble right now — try again in a moment."
