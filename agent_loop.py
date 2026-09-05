"""
agent_loop.py — ReAct-style tool-calling agent for Mojo.

Replaces the old single-shot get_ai_response + keyword reminder routing.
Hardened for Groq / Gemini tool-call format quirks.
"""

from __future__ import annotations

import json
import logging
import re
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from agent_tools import TOOL_SCHEMAS, execute_tool, LANGUAGE_POLICY

try:
    import admin_commands as _admin_commands
except ImportError:  # pragma: no cover — always present in production package
    _admin_commands = None

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

MAX_TOOL_STEPS = 3  # was 5 — each step is a full Groq call + tool schemas
# Keep payloads under Groq limits (413 Payload Too Large was hitting group chats)
MAX_HISTORY_MSGS = 12
MAX_MSG_CHARS = 600
MAX_OBS_CHARS = 3500
# Transcript tool now returns a compact refined summary (~2k), not raw 50k text
MAX_OBS_CHARS_TRANSCRIPT = 3500
# Cap completion size — WhatsApp replies are short; unbounded drafts burn quota
MAX_COMPLETION_TOKENS = 1024
MAX_COMPLETION_TOKENS_SHORT = 512


def _tool_allowed_for_chat(chat_id: str, tool_name: str) -> bool:
    """True only if admin enabled this tool for the chat (default OFF)."""
    if not tool_name:
        return False
    if _admin_commands is None:
        # Fail closed if control plane missing
        return False
    try:
        return bool(_admin_commands.is_tool_enabled(chat_id, tool_name))
    except Exception as e:
        print(f"[AGENT] is_tool_enabled failed for {tool_name}: {e}")
        return False


def _schemas_for_chat(chat_id: str) -> List[Dict[str, Any]]:
    """Subset of TOOL_SCHEMAS that are enabled for this chat. Empty = pure chat."""
    if _admin_commands is None:
        return []
    try:
        enabled = set(_admin_commands.get_enabled_tools(chat_id) or [])
    except Exception as e:
        print(f"[AGENT] get_enabled_tools failed: {e}")
        return []
    if not enabled:
        return []
    out: List[Dict[str, Any]] = []
    for schema in TOOL_SCHEMAS:
        name = (
            (schema.get("function") or {}).get("name")
            if isinstance(schema, dict)
            else None
        )
        if name and name in enabled:
            out.append(schema)
    return out

_VAGUE_ACK_RE = re.compile(
    r"^(theek hai|ok|okay|done|ho gaya|sure|haan|ji|alright|got it)"
    r"(\s*[.!.👍✅🙏]*)?$",
    re.I,
)


def _is_vague_ack(text: str) -> bool:
    """True for empty / pure acknowledgement replies that discard tool results."""
    t = (text or "").strip()
    if not t:
        return True
    if len(t) <= 14 and _VAGUE_ACK_RE.match(t):
        return True
    return False


def _is_bad_post_tool_reply(text: str) -> bool:
    """
    Catch replies that discard good tool OBS: multi-part promises, 'part 1',
    invented rate-limit excuses, pure meta without the actual content.
    """
    t = (text or "").strip()
    if _is_vague_ack(t):
        return True
    low = t.lower()
    bad_markers = (
        "part 1",
        "part\u202f1",
        "here's part",
        "here is part",
        "few short messages",
        "i'll send",
        "i will send",
        "sending in parts",
        "technical issue",
        "thori si technical",
        "dobara try",
        "try again later",
        "ek second baad",
        "thodi der baad",
        "thori der baad",
        "phir try",
        "request limit",
        "rate limit",
        "rate-limit",
        "rate‑limit",  # unicode hyphen variant models invent
        "couldn't pull",
        "could not pull",
        "due to a request",
        "fetch nahi",
        "fetch karte",
        "content abhi",
        "content dekh nahi",
        "abhi tak fetch",
        "exact details nahi",
        "fetch nahi ho paaya",
        "fetch nahi hua",
    )
    if any(m in low for m in bad_markers):
        return True
    # Very short after tools usually means the model bailed
    if len(t) < 40:
        return True
    return False


def _last_tool_obs_for_user(messages: List[dict], max_len: int = 3500) -> Optional[str]:
    """Strip Source header from last tool OBS so it can be sent to the user as-is."""
    for m in reversed(messages or []):
        if m.get("role") != "tool":
            continue
        obs = str(m.get("content") or "").strip()
        if not obs:
            continue
        # Drop "Source: … | mode=…" first line if present
        lines = obs.split("\n")
        if lines and lines[0].lower().startswith("source:"):
            obs = "\n".join(lines[1:]).lstrip("\n")
        obs = obs.strip()
        if not obs:
            continue
        if len(obs) > max_len:
            obs = obs[:max_len].rstrip() + "…"
        return obs
    return None


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
{LANGUAGE_POLICY.strip()}

=== CORE RULES (SPEED + ACCURACY) ===
1. Keep replies SHORT — WhatsApp friendly (1–3 short lines max). No walls of text. No markdown headers (###). No bullet essays. No bio dumps.
2. Never repeat the same sentence twice in one reply.
3. Do NOT invent timestamps, brackets around names, or internal IDs in your final reply.
4. GREETINGS / SMALL TALK ("hi", "hello", "hows it going", "kya haal", "salam") → reply naturally in ONE short line. Do NOT call any tool. Do NOT dump agency stats or founder bio.
5. PURE @mention only (message is just "@mojo" / "@aimojo" / a number tag with no real question) → reply "Haan, boliye?" Do NOT call any tool. Do NOT continue a previous website topic from history.
6. Call search_knowledge ONLY when the user actually asks about agency services, portfolio, founder background, pricing, or "what can you do". Never on a plain greeting or pure mention.
7. For ANY reminder create / list / cancel intent — including Roman Urdu like "remind karna", "1 min me paani", "list reminder", "cancel karo", "paani wale cancel" — you MUST call set_reminder / list_reminders / cancel_reminders. Never pretend you set a reminder without the tool. Never suggest "set a phone timer instead".
8. cancel_reminders understands keywords like "paani", "water", "debug", or "all"/"sab". Prefer calling it over asking clarifying questions when intent is clear.
9. After tools finish, give a natural confirmation or answer in 1–3 lines. Prefer ZERO tools when the answer is pure conversation.
10. When summarizing a website from browse_url: 2–3 plain lines max. No numbered sections, no markdown.
11. VOICE: If the turn includes "[Voice note transcript]" or "[Cached recent voice-note transcript]", answer from that text. For "kya bola" / "voice note me kya" / "what did I say" use the transcript — never browse a website and never claim no voice exists when a transcript is present.
11b. NOTE DOWN + QUOTE: If the user says "note down" / "note kar lo" / "ye cheez note" (voice or text) AND a "[Quoted Message]:" block is present in the same turn, call note_down with that quoted content immediately. Do NOT ask "kis cheez ko note karna hai?" when the quoted text is already provided.

=== URL / WEB FACTS (NO HALLUCINATION) ===
12. Call browse_url ONLY when the CURRENT message has a URL (force_urls / priority note) OR the user clearly asks about a link/site ("details iska", "what is this about" with a link context, "fetch latest repo"). Never browse just because the last topic was a website.
13. When the user says "fetch latest repo" after a GitHub profile link, call browse_url on that exact github.com/username URL.
14. If a tool returns an error or empty data, say so honestly. Do not fabricate fallback facts.
15. For weather / temperature / mausam (e.g. Islamabad kitna garam hai), ALWAYS call get_weather — not web_search.
16. VIDEO CONTENT: When the CURRENT message has a video link (YouTube / TikTok / Instagram reel / Vimeo / FB) and the user asks what it is about — including Roman Urdu "ye kya hai", "isme kya hai", "summarize karo", "what is this about" — you MUST use transcribe_video (never browse_url, never invent).
   - mode=transcript → "transcript", "poora transcript", "likh ke do", "kya bola"
   - mode=summary → "summary", "khulasa", "ye kya hai", "isme kya hai", "what is this", "what is this about"
   - mode=key_points → "points", "key points", "main baatein", "bullets"
   If user asks for N words, set target_words=N and mode=summary.
   ALWAYS call the tool again when mode/length changes — never invent from a prior short summary.
   Prefer this over browse_url. Do NOT invent rate-limit / "fetch nahi hua" / "thodi der baad" excuses.
16b. TRANSCRIPT REPLY (tool already refined for the chosen mode):
   - Present the tool result almost as-is in ONE WhatsApp message.
   - Do NOT wrap in code fences, do NOT promise "part 1 / more messages later", do NOT re-translate into Devanagari.
   - Match LANGUAGE POLICY (Roman Urdu if user wrote Roman Urdu).

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
    had_tool_result = False
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "tool":
            had_tool_result = True
            # Fold tool observation into a user-visible note
            out.append({
                "role": "user",
                "content": f"[Tool result — use this to answer the user]\n{content or ''}",
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
    # After tools ran, Gemini often replies with a vague ack ("Theek hai") unless
    # told explicitly to synthesize facts/links from the tool results.
    if had_tool_result:
        out.append({
            "role": "user",
            "content": (
                "Using ONLY the tool results above, give the user a short direct answer "
                "with any concrete facts and full URLs/links found. Do NOT reply with only "
                "'Theek hai' or a vague acknowledgement — include the useful information."
            ),
        })
    return out


def _shrink_messages(messages: List[dict], keep_last: int = 6) -> List[dict]:
    """Drop older turns to recover from 413 Payload Too Large.

    Preserve the latest user message more aggressively — it often carries a
    full document extract (xlsx distinct values). Truncating that to 1200
    chars is what made cluster lists incomplete.
    """
    if not messages:
        return messages
    system = [m for m in messages if m.get("role") == "system"][:1]
    rest = [m for m in messages if m.get("role") != "system"]
    rest = rest[-keep_last:]
    out = []
    for idx, m in enumerate(system + rest):
        c = m.get("content")
        if not isinstance(c, str):
            out.append(m)
            continue
        is_last_user = (
            m.get("role") == "user"
            and idx == len(system + rest) - 1
        )
        is_doc = "[Document:" in c or "### Distinct values" in c
        # Keep document / latest user extracts much larger
        if is_last_user or is_doc:
            cap = 50000
        elif m.get("role") == "tool":
            cap = 3500
        else:
            cap = 1200
        if len(c) > cap:
            m = {**m, "content": c[:cap] + "…"}
        out.append(m)
    return out


def _chat_completion(
    messages: List[dict],
    tools: Optional[List] = None,
    temperature: float = 0.4,
    max_tokens: Optional[int] = None,
):
    """Primary Groq (429 + 413 aware), fallback Gemini. Returns the message object."""
    import time as _time

    working = messages
    tok = max_tokens if max_tokens is not None else MAX_COMPLETION_TOKENS
    kwargs: Dict[str, Any] = {
        "model": _MODEL_NAME,
        "messages": working,
        "temperature": temperature,
        "max_tokens": tok,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    last_err = None
    for attempt in range(2):  # only 2 attempts — avoid retry storms that worsen 429
        try:
            kwargs["messages"] = working
            resp = _client_ai.chat.completions.create(**kwargs)
            return resp.choices[0].message
        except Exception as primary_err:
            last_err = primary_err
            err_s = str(primary_err).lower()
            is_rate = "429" in err_s or "rate" in err_s or "too many" in err_s
            is_payload = "413" in err_s or "payload" in err_s or "too large" in err_s
            if is_payload:
                print(f"[AGENT] Groq 413 payload too large — shrinking history (attempt {attempt+1})")
                working = _shrink_messages(working, keep_last=5 if attempt == 0 else 3)
                # Drop tools on the second shrink to minimize payload further
                if attempt >= 1 and "tools" in kwargs:
                    kwargs.pop("tools", None)
                    kwargs.pop("tool_choice", None)
                continue
            if is_rate and attempt < 1:
                sleep_s = 1.5
                print(f"[AGENT] Groq 429/rate — retry in {sleep_s:.1f}s")
                _time.sleep(sleep_s)
                continue
            print(f"[AGENT] Primary model failed: {primary_err}. Trying Gemini...")
            break

    # Gemini fallback — sanitize tool messages so we don't get 400s.
    # Single attempt only (no nested retry storms on 503/429).
    try:
        clean = _sanitize_messages_for_gemini(working)
        gkwargs: Dict[str, Any] = {
            "model": _GEMINI_MODEL,
            "messages": clean,
            "temperature": temperature,
            "max_tokens": tok,
        }
        resp = _client_gemini.chat.completions.create(**gkwargs)
        return resp.choices[0].message
    except Exception as e2:
        print(f"[AGENT] Gemini fallback failed: {e2}")
        # Do NOT burn more time on Gemini 503 loops. Caller uses tool OBS.
        raise last_err or e2


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

        # Cap history to avoid 413 Payload Too Large on Groq (esp. group chats
        # that accumulated long agent essays about websites).
        effective_limit = min(history_limit or MAX_HISTORY_MSGS, MAX_HISTORY_MSGS)
        raw_history = _fetch_chat_history(chat_id, effective_limit) or []
        contacts_map, reverse_map = _get_contacts_maps() if _get_contacts_maps else ({}, {})
        memory_notes = _get_group_memory(chat_id) if _get_group_memory else []
        memory_block = ""
        if memory_notes:
            # Cap memory block size too
            clipped = [str(n)[:200] for n in memory_notes[:12]]
            memory_block = "PERMANENT NOTES FOR THIS CHAT:\n" + "\n".join(
                f"- {n}" for n in clipped
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
                    + ", ".join(sorted(str(x) for x in list(active)[:20]))
                    + ".\n"
                    "When tagging, use exact @Name from this list."
                )

        system = _build_system_prompt(
            chat_id, sender_id, is_group, time_context, tag_block, memory_block
        )

        messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]

        # Only replay clean user/assistant turns (never stale tool messages from DB)
        # Truncate each message so one long past essay cannot blow the payload.
        for msg in raw_history:
            role = msg.get("role") or "user"
            if role not in ("user", "assistant"):
                continue
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if len(content) > MAX_MSG_CHARS:
                content = content[:MAX_MSG_CHARS] + "…"
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

        # Resolve the user text used for intent detection
        _intent_text = (latest_user_text or "")
        if not _intent_text:
            for m in reversed(messages):
                if m.get("role") == "user":
                    _intent_text = m.get("content") or ""
                    break
        _intent_low = _intent_text.lower()

        def _is_video_url(u: str) -> bool:
            ul = (u or "").lower()
            return any(
                h in ul
                for h in (
                    "youtube.com/",
                    "youtu.be/",
                    "tiktok.com/",
                    "vm.tiktok.com/",
                    "vt.tiktok.com/",
                    "vimeo.com/",
                    "facebook.com/watch",
                    "facebook.com/reel",
                    "facebook.com/reels",
                    "facebook.com/share/r/",
                    "facebook.com/share/v/",
                    "fb.watch/",
                    "fb.gg/",
                    "instagram.com/reel",
                    "instagram.com/reels",
                    "instagram.com/p/",
                    "instagram.com/tv/",
                    "instagr.am/",
                )
            )

        def _video_spoken_intent(text_low: str) -> Optional[str]:
            """Return transcribe_video mode if user wants spoken/video content, else None.

            Covers English + Roman Urdu. Vague "what is this / ye kya hai" on a
            video link is treated as summary — never leave it to browse_url
            (browse cannot hear spoken content and models invent rate-limits).
            """
            # Strip bot mentions / quoted-url noise so phrase match is reliable
            t = text_low or ""
            t = re.sub(r"@\d[\d\s]*", " ", t)
            t = re.sub(r"\[quoted message\]:.*", " ", t, flags=re.I | re.S)
            t = re.sub(r"https?://\S+", " ", t)
            t = re.sub(r"\s+", " ", t).strip()

            if any(
                w in t
                for w in (
                    "transcript",
                    "poora transcript",
                    "whole transcript",
                    "full transcript",
                    "likh ke do",
                    "likh do",
                    "kya bola",
                    "kya kaha",
                    "kya keh raha",
                    "is video me kya",
                    "video me kya",
                    "poora sunao",
                    "word for word",
                )
            ):
                return "transcript"
            if any(
                w in t
                for w in (
                    "key points",
                    "keypoints",
                    "main baatein",
                    "main baate",
                    "main points",
                    "bullet",
                    "points nikal",
                    "key takeaway",
                )
            ):
                return "key_points"
            if any(
                w in t
                for w in (
                    "summary",
                    "summarise",
                    "summarize",
                    "summarise karo",
                    "summarize karo",
                    "khulasa",
                    "short me batao",
                    "short mein batao",
                    "mukhtasir",
                    "kis bare me",
                    "kis baare",
                    "kis bare",
                    "kiske bare",
                    "what is this video",
                    "what's this video",
                    "video about",
                    "is video ka",
                    "ye video",
                    "this video",
                    "this reel",
                    "ye reel",
                    "is reel",
                    # Generic "what is this / ye kya hai" — with a video URL present
                    # the force-path caller only invokes us when URL is video.
                    "what is this about",
                    "what's this about",
                    "what is this",
                    "whats this",
                    "what's this",
                    "ye kya hai",
                    "ye kia hai",
                    "ye kya he",
                    "ye kia he",
                    "isme kya hai",
                    "isme kia hai",
                    "isme kya he",
                    "is mein kya",
                    "is me kya",
                    "iska matlab",
                    "ye about",
                    "batao iske",
                    "iske bare",
                    "iske baare",
                    "explain this",
                    "explain karo",
                    "samjhao",
                    "samjha do",
                    "tell me about",
                    "about this",
                )
            ):
                return "summary"
            # Very short residual text after stripping URL/mention ("?", "ye?", "ye")
            # still means "tell me about the linked video".
            residual = re.sub(r"[^\w\s]", "", t).strip()
            if residual in ("", "ye", "yeh", "this", "bro", "bhai", "ji", "pls", "please"):
                return "summary"
            return None

        def _parse_target_words(text_low: str) -> Optional[int]:
            m = _re.search(r"(\d{2,4})\s*[- ]?\s*words?", text_low)
            if not m:
                m = _re.search(r"(\d{2,4})\s*word", text_low)
            if not m:
                return None
            try:
                n = int(m.group(1))
                return max(80, min(800, n))
            except ValueError:
                return None

        # Only tools the admin enabled for this chat are offered to the model.
        # Default is empty → pure conversational reply (ai_chat), no tools.
        tools_for_next: Optional[List] = _schemas_for_chat(chat_id)
        if tools_for_next:
            print(
                f"[AGENT] tools enabled for {chat_id}: "
                + ", ".join(
                    (s.get("function") or {}).get("name", "?") for s in tools_for_next
                )
            )
        else:
            print(f"[AGENT] no tools enabled for {chat_id} — chat-only mode")

        if urls_in_last:
            # Deterministic pre-fetch: tool_choice=auto is not a guarantee the model
            # will call the right tool. Inject the tool result ourselves — but ONLY
            # when the corresponding tool flag is ON for this chat.
            primary_url = urls_in_last[0]
            spoken_mode = _video_spoken_intent(_intent_low)
            is_video = _is_video_url(primary_url)
            # Video links: never fall through to browse_url for content questions.
            # If intent matched → use that mode; if intent missed but URL is clearly
            # a video and the user is asking about the linked message, default summary.
            if is_video and not spoken_mode:
                # Link-intent / "about the quoted video" without exact phrase match
                _li = _intent_low
                if any(
                    w in _li
                    for w in (
                        "kya",
                        "what",
                        "about",
                        "summary",
                        "summar",
                        "batao",
                        "batain",
                        "explain",
                        "samjha",
                        "content",
                        "video",
                        "reel",
                    )
                ):
                    spoken_mode = "summary"
            use_video = bool(spoken_mode and is_video)
            can_transcribe = _tool_allowed_for_chat(chat_id, "transcribe_video")
            can_browse = _tool_allowed_for_chat(chat_id, "browse_url")

            if use_video and can_transcribe:
                tw = _parse_target_words(_intent_low)
                tool_args: Dict[str, Any] = {
                    "url": primary_url,
                    "mode": spoken_mode,
                    "language": "auto",
                    "timestamps": False,
                }
                if tw and spoken_mode == "summary":
                    tool_args["target_words"] = tw
                print(
                    f"[AGENT] force transcribe_video mode={spoken_mode} "
                    f"tw={tw} for: {primary_url}"
                )
                try:
                    _forced_observation = execute_tool(
                        "transcribe_video", tool_args, tool_ctx
                    )
                except Exception as _fe:
                    _forced_observation = f"Tool error: {_fe}"
                _forced_obs_str = str(_forced_observation)
                _obs_cap = MAX_OBS_CHARS_TRANSCRIPT
                if len(_forced_obs_str) > _obs_cap:
                    _forced_obs_str = _forced_obs_str[:_obs_cap] + "…"

                _forced_call_id = "call_forced_transcribe_0"
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": _forced_call_id,
                        "type": "function",
                        "function": {
                            "name": "transcribe_video",
                            "arguments": json.dumps(tool_args),
                        },
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": _forced_call_id,
                    "content": _forced_obs_str,
                })
                # Always prefer the tool observation as the user-facing reply —
                # both success (refined summary) and honest failures (private IG,
                # bot-check, etc.). Avoids a second LLM call inventing rate-limits.
                direct = _last_tool_obs_for_user(messages)
                if direct and len(direct) >= 30:
                    print(
                        "[AGENT] force transcribe_video → direct OBS reply "
                        f"(chars={len(direct)}, starts={direct[:40]!r})"
                    )
                    logging.getLogger("mojo.agent").info(
                        "DIRECT_OBS_REPLY mode=%s chars=%s",
                        spoken_mode,
                        len(direct),
                    )
                    return direct
                messages.append({
                    "role": "system",
                    "content": (
                        f"You already ran transcribe_video on {primary_url} "
                        f"(mode={spoken_mode}). Answer from that tool result only in "
                        "ONE message. Present the content almost as-is. If the tool "
                        "reported an error (private video, download failed, etc.), "
                        "relay that honestly in 1–2 lines. Do NOT invent a rate-limit / "
                        "request-limit / 'fetch nahi hua' / 'thodi der baad' excuse. "
                        "Do NOT call transcribe_video or browse_url again this turn."
                    ),
                })
                tools_for_next = None  # synthesis only — no tool schema payload
            elif use_video and not can_transcribe:
                print(
                    f"[AGENT] SKIP force transcribe_video — tool disabled for {chat_id}"
                )
                logging.getLogger("mojo.agent").info(
                    "TOOL_BLOCKED transcribe_video chat=%s", chat_id
                )
                messages.append({
                    "role": "system",
                    "content": (
                        "Video transcript/summary tool is turned OFF for this chat. "
                        "Politely tell the user you cannot transcribe or summarize "
                        "video links here. Do NOT invent a transcript or summary."
                    ),
                })
            elif can_browse:
                print(f"[AGENT] force browse_url for: {urls_in_last}")
                try:
                    _forced_observation = execute_tool(
                        "browse_url", {"url": primary_url}, tool_ctx
                    )
                except Exception as _fe:
                    _forced_observation = f"Tool error: {_fe}"
                _forced_obs_str = str(_forced_observation)
                if len(_forced_obs_str) > MAX_OBS_CHARS:
                    _forced_obs_str = _forced_obs_str[:MAX_OBS_CHARS] + "…"

                _forced_call_id = "call_forced_browse_0"
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": _forced_call_id,
                        "type": "function",
                        "function": {
                            "name": "browse_url",
                            "arguments": json.dumps({"url": primary_url}),
                        },
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": _forced_call_id,
                    "content": _forced_obs_str,
                })
                messages.append({
                    "role": "system",
                    "content": (
                        f"You already fetched {primary_url} above via browse_url — that IS the "
                        "user's latest link. Answer from that tool result only. Do NOT call "
                        "browse_url again this turn, and ignore any older/different links from "
                        "chat history. Do NOT invent rate-limit errors."
                    ),
                })
                tools_for_next = None  # synthesis only — no tool schema payload
            else:
                # URL present but neither transcribe nor browse is allowed
                print(
                    f"[AGENT] SKIP force URL tools — disabled for {chat_id} url={primary_url}"
                )
                logging.getLogger("mojo.agent").info(
                    "TOOL_BLOCKED browse_url/transcribe chat=%s", chat_id
                )

        # --- Agentic loop ---
        for step in range(MAX_TOOL_STEPS):
            try:
                msg = _chat_completion(
                    messages,
                    tools=tools_for_next,
                    temperature=0.3,
                    max_tokens=MAX_COMPLETION_TOKENS,
                )
            except Exception as e:
                print(f"[AGENT] completion failed at step {step}: {e}")
                traceback.print_exc()
                # Prefer already-fetched tool OBS over spinning on rate limits
                obs = _last_tool_obs_for_user(messages)
                if obs:
                    logging.getLogger("mojo.agent").info(
                        "RETURNING_TOOL_OBS after completion failure (step=%s)", step
                    )
                    return obs
                if step > 0:
                    try:
                        final = _chat_completion(
                            messages,
                            tools=None,
                            temperature=0.4,
                            max_tokens=MAX_COMPLETION_TOKENS_SHORT,
                        )
                        ans = (final.content or "").strip()
                        if ans and not _is_bad_post_tool_reply(ans):
                            return ans
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
                # After tools (including force-injected ones before the loop): reject
                # vague acks, multi-part promises, invented rate-limit excuses.
                had_tools = any(m.get("role") == "tool" for m in messages)
                if had_tools and _is_bad_post_tool_reply(answer):
                    logging.getLogger("mojo.agent").info(
                        "BAD_AFTER_TOOLS answer=%r — forcing synthesis or OBS", answer
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            "Tool results are already above. Reply with the useful content "
                            "from those results in ONE message. Do NOT invent rate-limit or "
                            "request-limit errors. Do NOT say you will send parts later. "
                            "Prefer the tool text as-is if it is already a clean "
                            "transcript/summary."
                        ),
                    })
                    try:
                        final = _chat_completion(
                            messages,
                            tools=None,
                            temperature=0.3,
                            max_tokens=MAX_COMPLETION_TOKENS_SHORT,
                        )
                        forced = (final.content or "").strip()
                        if forced and not _is_bad_post_tool_reply(forced):
                            return forced
                    except Exception as fe:
                        print(f"[AGENT] forced synthesis failed: {fe}")
                    obs = _last_tool_obs_for_user(messages)
                    if obs:
                        return obs
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
                # Defense in depth: refuse tools not enabled for this chat even if
                # the model hallucinated a call (or schemas leaked from history).
                if not _tool_allowed_for_chat(chat_id, name):
                    observation = (
                        f"Tool '{name}' is disabled for this chat. "
                        "Do not claim you ran it. Tell the user it is unavailable."
                    )
                    logging.getLogger("mojo.agent").info(
                        "TOOL_BLOCKED %s chat=%s", name, chat_id
                    )
                else:
                    try:
                        observation = execute_tool(name, args, tool_ctx)
                    except Exception as te:
                        traceback.print_exc()
                        observation = f"Tool error: {te}"
                obs_str = str(observation)
                # Transcripts need a higher cap so the model can refine/summarize
                # a long video instead of only seeing the first ~3k chars.
                obs_cap = (
                    MAX_OBS_CHARS_TRANSCRIPT
                    if name == "transcribe_video"
                    else MAX_OBS_CHARS
                )
                if len(obs_str) > obs_cap:
                    obs_str = (
                        obs_str[:obs_cap]
                        + "\n… [truncated for context — video may be longer; "
                        "summarize what you have, note if incomplete]"
                    )
                logging.getLogger("mojo.agent").info("OBS %s", obs_str[:500])
                print(f"[AGENT OBS] {obs_str[:400]}{'…' if len(obs_str) > 400 else ''}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": getattr(tc, "id", None) or f"call_{step}",
                        "content": obs_str,
                    }
                )
            # After tools ran this step → next step is synthesis only (no schema payload)
            tools_for_next = None

        # Max steps — force close without tools
        messages.append(
            {
                "role": "user",
                "content": (
                    "Give the best final answer now from the tool results above in ONE message. "
                    "No more tools. Do not promise parts later."
                ),
            }
        )
        try:
            final = _chat_completion(
                messages,
                tools=None,
                temperature=0.4,
                max_tokens=MAX_COMPLETION_TOKENS_SHORT,
            )
            ans = (final.content or "").strip()
            if ans and not _is_bad_post_tool_reply(ans):
                return ans
        except Exception as e:
            print(f"[AGENT] final close failed: {e}")
        obs = _last_tool_obs_for_user(messages)
        if obs:
            return obs
        return "Kaam almost complete ho gaya — dobara try kar lo."

    except Exception as e:
        traceback.print_exc()
        print(f"[AGENT ERROR] {e}")
        logging.getLogger("mojo.agent").exception("run_agent failed: %s", e)
        return "Thori si technical issue aa gayi — ek second baad dobara try karo."