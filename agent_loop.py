"""
agent_loop.py — ReAct-style tool-calling agent for Mojo.

Replaces the old single-shot get_ai_response + keyword reminder routing.
Hardened for Groq / Gemini too
l-call format quirks.
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
# Full lecture transcript delivery (was 3500 → cut mid-sentence after speech_render)
MAX_OBS_CHARS_TRANSCRIPT = 14000
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


# ---------------------------------------------------------------------------
# Video / link intent (module-level so unit tests can import without run_agent)
# ---------------------------------------------------------------------------

_VIDEO_URL_HOST_MARKERS = (
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

_TRANSCRIPT_PHRASES = frozenset({
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
    "word-to-word",
    "what was said",
    "what is said",
    "dont miss",
    "don't miss",
    "do not miss",
    "each point",
    "har point",
    "every point",
    "detailed what",
    "full detail",
    "sab kuch batao",
    "everything covered",
    "covered in the",
    "covered in this",
})

_KEY_POINTS_PHRASES = frozenset({
    "key points",
    "keypoints",
    "main baatein",
    "main baate",
    "main points",
    "bullet",
    "points nikal",
    "key takeaway",
})

_EXPLICIT_SUMMARY_PHRASES = frozenset({
    "summary",
    "summarise",
    "summarize",
    "summarise karo",
    "summarize karo",
    "khulasa",
    "mukhtasir",
    "short me batao",
    "short mein batao",
    "1.5k words",
    "in 1.5k",
    "in 1500",
    # analysis / explain — NOT raw transcript dumps
    "explain",
    "explain this",
    "explain karo",
    "analysis",
    "analyse",
    "analyze",
    "point by point",
    "action by action",
    "step by step",
    "break down",
    "breakdown",
    "in depth analysis",
    "indepth analysis",
    "deep dive",
    "walk through",
    "walk me through",
    "samjhao",
    "samjha do",
})

_SUMMARY_PHRASES = frozenset({
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
    "tell me in depth",
    "about this",
    "about all",
    "ideas and",
    "implementation",
    "batana",
    "bata na",
    "bata do",
    "bata dena",
    "btao",
    "btana",
    "ye bata",
    "ye batana",
    "mujhe bata",
    "sunao",
    "suna do",
    "suna dena",
    "dekho ye",
    "check karo",
    "dekhna",
    "dekho isko",
})

# Tokens that signal the residual is about video *content* (not a side command)
_VIDEO_CONTENT_TOKENS = frozenset({
    "video", "reel", "youtube", "yt", "clip", "lecture", "session",
    "episode", "tutorial", "webinar", "day", "ideas", "implementation",
    "said", "bola", "kaha", "spoken", "caption", "transcript",
})

# Vague "about" — prefer link_preview (caption) over full ASR when possible
_VAGUE_ABOUT_PHRASES = frozenset({
    "ye kya hai",
    "ye kia hai",
    "ye kya he",
    "ye kia he",
    "isme kya hai",
    "isme kia hai",
    "what is this",
    "whats this",
    "what's this",
    "what is this about",
    "what's this about",
    "ye about",
    "iska matlab",
})

_GREETING_TOKENS = frozenset({
    "hi", "hello", "hey", "salam", "salaam", "assalam", "asalam",
    "assalamualaikum", "assalamu", "alaikum",
    "thanks", "shukriya", "ok", "okay", "theek", "haan", "han",
    "ji", "bro", "bhai", "yaar", "yar", "boss", "pls", "please",
    "ye", "yeh", "this",
})

# Commands / non-content intents — never force-transcribe/browse even if a URL is present
_BLOCK_FORCE_VIDEO_PHRASES = frozenset({
    # reminders
    "reminder", "remind", "reminders", "yaad", "alarm",
    "set karo", "set kar", "list reminder", "cancel reminder",
    "cancel karo", "delete reminder", "sab cancel",
    # permanent memory / standing notes (must not force-browse incidental links)
    "always remember", "remember this", "remember that",
    "yaad rakh", "yaad rakho", "yaad rakhna", "yaad kar lo", "yaad karlo",
    "hamesha yaad", "permanent note", "save this permanently",
    "from now on remember",
    # outbound email / job apply (URLs inside JD must not force-browse)
    "apply to", "apply for", "send my resume", "send resume", "cover letter",
    "email this", "email them", "draft email", "send email", "bhej do email",
    "job application", "invite to collaborate", "offer a demo",

    # admin / ops
    "enable", "disable", "status", "/enable", "/disable", "/status",
    "/help", "/chats", "feature",
    # messaging / group actions
    "bhejo", "bhej do", "bhej dena", "send karo", "forward",
    "delete", "remove", "milte", "milte hain", "meeting",
    "call karo", "phone karo",
    # pure admin-ish
    "group ka status", "is group", "members",
})

_BLOCK_FORCE_VIDEO_TOKENS = frozenset({
    "reminder", "remind", "reminders", "yaad", "alarm",
    "enable", "disable", "bhejo", "bhej", "delete", "remove",
    "milte", "meeting", "cancel", "forward",
    # memory tokens (after strip, "remember" alone should still block force URL tools)
    "remember", "rakh", "rakho", "rakhna",
})


def _strip_intent_noise(text: str) -> str:
    t = text or ""
    t = re.sub(r"@\d[\d\s]*", " ", t)
    t = re.sub(r"\[quoted message\]:.*", " ", t, flags=re.I | re.S)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_video_url(u: str) -> bool:
    ul = (u or "").lower()
    return any(h in ul for h in _VIDEO_URL_HOST_MARKERS)


def _is_blocked_force_command(text_low: str) -> bool:
    """True when residual is a reminder/memory/admin/imperative — do not force
    video or browse tools (incidental links inside a 'remember this' quote must
    not hijack the turn into a GitHub profile listing)."""
    t = _strip_intent_noise(text_low).lower()
    if not t:
        return False
    if any(p in t for p in _BLOCK_FORCE_VIDEO_PHRASES):
        return True
    tokens = re.sub(r"[^\w\s]", "", t).split()
    if any(tok in _BLOCK_FORCE_VIDEO_TOKENS for tok in tokens):
        return True
    return False


def _is_vague_about_ask(text_low: str) -> bool:
    t = _strip_intent_noise(text_low).lower()
    if any(p in t for p in _VAGUE_ABOUT_PHRASES):
        return True
    # bare residual like "ye" / "this" after strip
    residual = re.sub(r"[^\w\s]", "", t).strip()
    return residual in ("", "ye", "yeh", "this", "ye kya", "kya hai")


def video_spoken_intent(text_low: str) -> Optional[str]:
    """
    Return transcribe mode if user wants spoken/video content, else None.

    Priority:
      1. Explicit summary / key_points (even with "in depth" / long word counts)
      2. Explicit transcript / word-for-word
      3. Detail language without "summarize" → transcript
      4. Short residual → summary
    """
    t = _strip_intent_noise(text_low).lower()
    if not t:
        return "summary"

    # Explicit summary / analysis / explain wins over "in depth"
    # so "explain point by point in depth analysis" → summary, not transcript dump.
    if any(w in t for w in _EXPLICIT_SUMMARY_PHRASES):
        return "summary"
    if any(w in t for w in _KEY_POINTS_PHRASES):
        return "key_points"

    # Word-for-word / spoken-text only (not analysis)
    if any(w in t for w in _TRANSCRIPT_PHRASES):
        # "each point" / "har point" alone can mean analysis — if explain/analysis
        # already handled above. Pure transcript phrases still win here.
        return "transcript"

    # Detail language WITHOUT explain/analysis/summary → still prefer summary
    # for "in depth" / "detailed" (users almost always want structured analysis,
    # not a raw Hindi dump). Only force transcript when they clearly ask for
    # spoken wording (kya bola / word for word / transcript).
    if any(
        m in t
        for m in (
            "in depth",
            "indepth",
            "in-depth",
            "detailed",
            "in detail",
            "each point",
            "har point",
            "every point",
            "full detail",
            "sab kuch",
            "everything covered",
            "point by point",
            "action by action",
        )
    ):
        return "summary"

    if any(w in t for w in _SUMMARY_PHRASES):
        return "summary"

    residual = re.sub(r"[^\w\s]", "", t).strip()
    tokens = residual.split()
    if not tokens:
        return "summary"
    if all(tok in _GREETING_TOKENS for tok in tokens):
        return None
    if _is_blocked_force_command(t):
        return None
    # Short non-greeting residual → summary
    if len(tokens) <= 6:
        return "summary"
    # Longer residual: still force when it looks like a content ask about the video
    # (otherwise force path falls through to browse_url → title only).
    if any(tok in _VIDEO_CONTENT_TOKENS for tok in tokens):
        if any(m in t for m in ("bola", "kaha", "said", "word for word", "transcript")):
            return "transcript"
        return "summary"
    # Substantial non-blocked ask with a video URL present (caller gates URL) —
    # prefer summary over doing nothing / title-only browse.
    if len(tokens) >= 8:
        return "summary"
    return None


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
        "koi detail ya content",
        "content yahan available nahi",
        "summarize nahi kar sakta",
        "detail ya content yahan",
        "available nahi hai, isliye",
    )
    if any(m in low for m in bad_markers):
        return True
    # Very short after tools usually means the model bailed
    if len(t) < 40:
        return True
    return False


def _last_tool_obs_for_user(messages: List[dict], max_len: int = 3500) -> Optional[str]:
    """Strip Source header from last tool OBS so it can be sent to the user as-is.

    Never dump raw permanent-memory / chat-note blobs (common after a failed
    job-apply turn that mistakenly called search_knowledge).
    """
    for m in reversed(messages or []):
        if m.get("role") != "tool":
            continue
        obs = str(m.get("content") or "").strip()
        if not obs:
            continue
        low = obs[:200].lower()
        # Memory / knowledge dumps are not user-facing replies
        if (
            low.startswith("[chat note]")
            or "always remember" in low
            or obs.count("[Chat note]") >= 2
            or (low.startswith("permission denied"))
        ):
            continue
        # Drop "Source: … | mode=…" first line if present
        lines = obs.split("\n")
        if lines and lines[0].lower().startswith("source:"):
            obs = "\n".join(lines[1:]).lstrip("\n")
        obs = obs.strip()
        if not obs:
            continue
        use_len = max_len
        if max_len <= 3500 and len(obs) > 3500:
            use_len = max(max_len, MAX_OBS_CHARS_TRANSCRIPT)
        if len(obs) > use_len:
            obs = obs[:use_len].rstrip() + "…"
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



# ---------------------------------------------------------------------------
# WhatsApp reply formatting (native syntax only)
# ---------------------------------------------------------------------------
# Long guide is injected only when the turn is likely structured (list /
# transcript / summary / tool OBS). Greetings and pure chat keep the short
# system prompt to save tokens on every Groq turn.

_WHATSAPP_FORMAT_GUIDE = """=== WHATSAPP RESPONSE FORMATTING (USE WITH AWARENESS) ===
WhatsApp supports a small native style set. Use it only when it makes the reply clearer, more scannable, or more professional. Never decorate every sentence. Prefer plain text for short casual replies.

Supported syntax (exact — no spaces between marker and text):
- *bold* → highlight the single most important word/phrase (names, times, status, prices, key answer).
- _italic_ → soft emphasis, titles, gentle nuance, or a short aside.
- ~strikethrough~ → corrections, old price / superseded info, light sarcasm.
- `inline code` (single backtick) → short codes, IDs, commands, order numbers, file names.
- ```monospace block``` (three backticks on their own lines or around a short block) → multi-line code, aligned data, or a short pasted snippet that must keep spacing.
- Bulleted list: start a line with "- " (hyphen + space). Prefer bullets for most multi-item replies.
- Numbered list: ONLY for real sequential steps. Each item on its OWN line with increasing numbers: "1. " then "2. " then "3. ". NEVER repeat "1." on every row. Hard max 5 items. If unsure, use "- " bullets instead.
- Labels (To / Subject / Body / etc.): each on its own line — never glue *To:*value*Subject:*value on one line.
- Block quote: start a line with "> " to set off a short quoted line or key takeaway.
- Combinations work when nested cleanly, e.g. *_bold italic_*, *~strike bold~*. Close in reverse order of opening. Do not over-nest.
- Line breaks: one blank line between short paragraphs for readability. Avoid huge gaps.

Smart usage rules:
- Default = plain text. Add style only when it earns its place (emphasis, structure, or technical clarity).
- One primary emphasis per short reply is usually enough (*bold* the answer or the deadline).
- Lists: prefer 2–5 tight bullets over a dense paragraph. Never turn a 1-line answer into a list.
- For Roman Urdu / mixed replies the same markers work; keep markers ASCII and text natural.
- Never use Markdown headers (###), tables, labeled links [text](url), or HTML. WhatsApp does not render them.
- Do not wrap entire replies in monospace or code fences. Do not promise "part 1" or multi-message dumps.
- Transcript / long tool output: present cleanly; light *key points* or a short - list is fine if it improves scanability. Still one WhatsApp message.
- Models: emit WhatsApp *bold* / _italic_ — NEVER GitHub-flavored **bold** or __italic__ (those show as literal asterisks on WhatsApp).
- When in doubt, stay plain and short.
"""

# Cheap tokens that mean "this turn may need structure" (list/summary/transcript)
_STRUCTURE_HINT_RE = re.compile(
    r"\b("
    r"list|bullet|steps?|summary|summarize|khulasa|key\s*points?|points?|"
    r"transcript|transcribe|full\s*text|detail|details|compare|pros|cons|"
    r"options?|features?|checklist|todo|plan|agenda|ingredients?|"
    r"how\s+to|kaise|steps\s+to|numbered|points\s+mein"
    r")\b",
    re.I,
)

_GREETING_ONLY_RE = re.compile(
    r"^[\s@]*(hi|hello|hey|salam|salaam|assalam|asalam|aoa|hola|yo|"
    r"kya\s*haal|kais[ae]|hows?\s*it\s*going|what'?s\s*up|sup|"
    r"good\s*(morning|evening|night|afternoon)|gm|gn|"
    r"mojo|aimojo)[\s!?.❤️💛🙏]*$",
    re.I,
)


def _needs_format_guide(
    latest_user_text: Optional[str] = None,
    force_urls: Optional[List[str]] = None,
    extra_user_note: Optional[str] = None,
) -> bool:
    """True when the long format block is worth the tokens this turn."""
    if force_urls:
        return True
    blob = " ".join(
        x for x in ((latest_user_text or ""), (extra_user_note or "")) if x
    ).strip()
    if not blob:
        return False
    if _GREETING_ONLY_RE.match(blob.strip()):
        return False
    # Transcript / document / quoted long content already in the note
    low = blob.lower()
    if any(
        m in low
        for m in (
            "[voice note transcript]",
            "[cached recent voice",
            "[quoted message]",
            "transcript",
            "summary",
            "key points",
            "key_points",
        )
    ):
        return True
    if _STRUCTURE_HINT_RE.search(blob):
        return True
    # Longer asks often benefit from light structure
    if len(blob) > 180:
        return True
    return False


def _format_whatsapp_reply(text: str) -> str:
    """Post-synthesis sanitizer: GFM → clean WhatsApp-native text.

    WhatsApp supports: *bold*  _italic_  ~strike~  ```mono```
    Does NOT support: ** **, ## headers, pipe tables, nested emphasis.
    """
    if not text or not isinstance(text, str):
        return text or ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # Horizontal rules / section dividers → blank line
    t = re.sub(r"(?m)^\s*[-*_]{3,}\s*$", "", t)
    t = re.sub(r"\s*---+\s*", "\n\n", t)

    # Strip fenced code language tags
    t = re.sub(r"```[\w+-]*\n", "```\n", t)

    # Convert **bold** / __bold__ → *bold* (repeat until stable for nested leftovers)
    for _ in range(3):
        t2_ = re.sub(r"\*\*(?!\s)(.+?)(?<!\s)\*\*", r"*\1*", t, flags=re.S)
        t2_ = re.sub(r"__(?!\s)(.+?)(?<!\s)__", r"_\1_", t2_, flags=re.S)
        if t2_ == t:
            break
        t = t2_
    # Stray leftover multi-asterisks
    t = re.sub(r"\*{3,}", "*", t)
    t = re.sub(r"_{3,}", "_", t)
    t = t.replace("****", "").replace("____", "")

    # ATX headers → bold line on its own
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", r"*\1*", t)

    # --- Glued field labels (classic email-draft collapse) ---
    # To:*a@b.com*Subject:*Foo*Body:text → one field per line, clean values
    _FIELD_RE = (
        r"To|Cc|Bcc|From|Subject|Body|Attach(?:\s+resume)?|Purpose"
    )
    if re.search(rf"(?:\*)?\b(?:{_FIELD_RE})\s*:", t, flags=re.I):
        parts = re.split(
            rf"(?:\*)?\b({_FIELD_RE})\s*:\s*\*?",
            t,
            flags=re.I,
        )
        # parts: [preamble, label1, val1, label2, val2, ...]
        rebuilt: List[str] = []
        preamble = (parts[0] or "").strip(" \t*")
        if preamble:
            rebuilt.append(preamble)
        i = 1
        while i < len(parts):
            label = parts[i].strip()
            raw_val = parts[i + 1] if i + 1 < len(parts) else ""
            # value runs until next label split; strip leftover bold stars
            val = (raw_val or "").strip()
            val = re.sub(r"^\*+\s*", "", val)
            val = re.sub(r"\s*\*+\s*$", "", val)
            val = val.strip(" \t*")
            # Keep internal newlines in Body
            if label.lower() == "body":
                val = val.strip()
                rebuilt.append(f"*{label}:*")
                if val:
                    rebuilt.append(val)
            else:
                # single-line fields: first line only if multi
                first = val.split("\n", 1)[0].strip().strip("*").strip()
                rebuilt.append(f"*{label}:* {first}".rstrip())
            i += 2
        t = "\n".join(rebuilt)

    # Glue fix: *Section**3.1 Title* → *Section*\n\n*3.1 Title*
    t = re.sub(r"\*([^*\n]{2,80})\*\*+(\d+\.\d+[^*\n]*)\*", r"*\1*\n\n*\2*", t)
    t = re.sub(r"\*([^*\n]{2,80})\*\*+([A-Z][^*\n]{2,60})\*", r"*\1*\n\n*\2*", t)
    # *text**next without space
    t = re.sub(r"\*([^*\n]+)\*\*([A-Za-z0-9])", r"*\1*\n\n*\2", t)

    # Ensure blank line before numbered section headings like *3.1 Foo*
    t = re.sub(r"(?<!\n)\n?(\*\d+\.\d+[^*\n]*\*)", r"\n\n\1", t)
    t = re.sub(r"(?<!\n)\n?(\*\d+\.\s+[^*\n]+\*)", r"\n\n\1", t)
    # "sentence.*Section title*" glued → break
    t = re.sub(
        r"([.!?])\*([A-Z0-9][^*\n]{2,60})\*",
        r"\1\n\n*\2*",
        t,
    )

    # word.*N. Section* (missing space after period)
    t = re.sub(
        r"(\w)\.(\*\d+[^*\n]*\*)",
        r"\1.\n\n\2",
        t,
    )

    # Inline "1. foo 1. bar 1. baz" → real multiline list
    def _split_inline_numbered(line: str) -> List[str]:
        if not re.search(r"\d{1,2}\.\s+\S.+\d{1,2}\.\s+\S", line):
            return [line]
        # Don't touch lines that are already a single clean list item
        if re.match(r"^\s*\d{1,2}\.\s+\S.*$", line) and line.count(". ") <= 1:
            return [line]
        parts = re.split(r"(?=\b\d{1,2}\.\s+)", line)
        out: List[str] = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if re.match(r"^\d{1,2}\.\s+", p):
                out.append(p)
            else:
                out.append(p)
        return out if len(out) >= 2 else [line]

    # Pipe tables → plain lines
    lines_out: List[str] = []
    for line in t.split("\n"):
        s = line.strip()
        if re.match(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", s):
            continue
        if s.count("|") >= 2 and (s.startswith("|") or s.endswith("|")):
            cells = [c.strip() for c in s.strip("|").split("|")]
            lines_out.append(" · ".join(c for c in cells if c))
            continue
        lines_out.extend(_split_inline_numbered(line))
    t = "\n".join(lines_out)

    # Bare HTML
    t = re.sub(
        r"</?(?:b|strong|i|em|u|s|strike|code|pre|p|br|div|span)[^>]*>",
        "",
        t,
        flags=re.I,
    )
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 (\2)", t)

    # Tighten * bold * → *bold* (same-line only)
    def _tight(m: re.Match) -> str:
        return f"{m.group(1)}{m.group(2).strip()}{m.group(1)}"

    t = re.sub(r"([*_~])\s+([^*\n_~]+?)\s+\1", _tight, t)

    # Bullet lines starting with · mid-paragraph → own line with dash
    t = re.sub(r"(?<!\n)·\s+", "\n- ", t)

    # Numbered list: renumber consecutive items so 1./1./1. becomes 1./2./3.
    # Prefer bullets when the model stamped every row as "1."
    capped: List[str] = []
    num_count = 0
    for line in t.split("\n"):
        m = re.match(r"^(\s*)(\d{1,2})([.)])\s+(.*)$", line)
        if m:
            num_count += 1
            body = m.group(4)
            indent = m.group(1)
            if num_count > 12:
                capped.append(f"{indent}- {body}")
            else:
                capped.append(f"{indent}{num_count}. {body}")
        else:
            if not line.strip():
                num_count = 0
            elif not re.match(r"^\s*([-*]|\d{1,2}[.)])\s+", line):
                num_count = 0
            capped.append(line)
    t = "\n".join(capped)

    # Collapse 3+ blank lines → 2; trim trailing spaces per line
    t = "\n".join(ln.rstrip() for ln in t.split("\n"))
    t = re.sub(r"\n{3,}", "\n\n", t)
    # Leading newline from field-split cleanup
    t = t.lstrip("\n")
    # Drop trailing ellipsis-only cut markers that look unfinished mid-word
    t = re.sub(r"(\w)\.\.\.\s*$", r"\1.", t)
    return t.strip()




def _reply(text: str) -> str:
    """User-facing exit: always run the WhatsApp sanitizer."""
    return _format_whatsapp_reply(text) if text else text


def _build_system_prompt(
    chat_id: str,
    sender_id: str,
    is_group: bool,
    time_context: str,
    tag_block: str,
    memory_block: str,
    include_format_guide: bool = False,
) -> str:
    env = (
        "ENVIRONMENT: GROUP CHAT.\n"
        "BEHAVIOR: Informal, intelligent, friendly, natural. Never rigid sales mode."
        if is_group
        else "ENVIRONMENT: PRIVATE CHAT.\n"
        "BEHAVIOR: Professional, warm, helpful."
    )

    # Short always-on style hint (cheap). Full guide only when structured reply likely.
    style_hint = (
        "STYLE: WhatsApp-native only — *bold* _italic_ ~strike~ `code`; "
        "never **GFM** or ### headers. Short replies."
    )
    format_block = ("\n" + _WHATSAPP_FORMAT_GUIDE) if include_format_guide else ""

    return f"""You are Mojo, the official AI Assistant for Mojo AI Agency (Founder: Muhammad Zaheer).

{env}

{time_context}

{tag_block}

{memory_block}

=== LANGUAGE POLICY (CRITICAL) ===
{LANGUAGE_POLICY.strip()}

=== CORE RULES (SPEED + ACCURACY) ===
1. Keep replies SHORT — WhatsApp friendly (1–3 short lines max for simple answers; up to ~6–8 lines only when a tight list or structured answer truly helps). No walls of text. No Markdown headers (### / ##). No long bullet essays. No bio dumps.
2. Never repeat the same sentence twice in one reply.
3. Do NOT invent timestamps, brackets around names, or internal IDs in your final reply.
4. GREETINGS / SMALL TALK ("hi", "hello", "hows it going", "kya haal", "salam") → reply naturally in ONE short line. Do NOT call any tool. Do NOT dump agency stats or founder bio.
5. PURE @mention only (message is just "@mojo" / "@aimojo" / a number tag with no real question) → reply "Haan, boliye?" Do NOT call any tool. Do NOT continue a previous website topic from history.
6. Call search_knowledge ONLY when the user actually asks about agency services, portfolio, founder background, pricing, or "what can you do". Never on a plain greeting or pure mention.
7. For ANY reminder create / list / cancel intent — including Roman Urdu like "remind karna", "1 min me paani", "list reminder", "cancel karo", "paani wale cancel" — you MUST call set_reminder / list_reminders / cancel_reminders. Never pretend you set a reminder without the tool. Never suggest "set a phone timer instead".
8. cancel_reminders understands keywords like "paani", "water", "debug", or "all"/"sab". Prefer calling it over asking clarifying questions when intent is clear.
9. After tools finish, give a natural confirmation or answer in 1–3 lines. Prefer ZERO tools when the answer is pure conversation.
10. When summarizing a website from browse_url: 2–3 plain lines max. Light formatting (*bold* key phrase or a short list of ≤5 items) is fine; no numbered essay sections.
11. VOICE: If the turn includes "[Voice note transcript]" or "[Cached recent voice-note transcript]", answer from that text. For "kya bola" / "voice note me kya" / "what did I say" use the transcript — never browse a website and never claim no voice exists when a transcript is present.
11b. NOTE DOWN vs IMAGE DESCRIBE:
   - note down / note karlo / onedrive me note / save this → note_down with FULL content WORD-FOR-WORD (quoted text, transcript, or image OCR). No "…" truncation. Summarize ONLY if user asked summary/khulasa/key points.
   - ye photo kya hai / what is in this image / describe → describe only (image_describe / vision). Do NOT write OneDrive.
   - Never claim "saved"/"noted" unless note_down returned success.
   - Do not ask "kis cheez ko note karna hai?" when quoted/OCR content is already present.
11c. PERMANENT MEMORY (save_memory / "always remember" / "yaad rakh"):
   - When the user says always remember / remember this / yaad rakh / permanent note AND a quoted body or long profile/resume is present → call save_memory with the FULL quoted/profile content (not a 1-line stub like "Remember @bot").
   - Do NOT browse_url or transcribe incidental links inside that quoted body (GitHub, LinkedIn, portfolio URLs are part of the note, not the task).
   - Confirm in 1 short line after save_memory succeeds. Do not re-dump the whole note back to chat.
11d. OUTBOUND EMAIL / JOB APPLY (owner only — draft_email + send_email):
   - Triggers: apply to this / apply karo / send resume / cover letter / email this to X / send email X ko / invite to collaborate / offer demo / pitch (any language).
   - EXACT BODY DEFAULT: When the user gives the email text after "email to X:", put that text WORD-FOR-WORD in draft_email body. Do NOT polish unless asked.
   - REWRITE ONLY ON REQUEST: make it professional / rewrite / polish / formal bana do / in a X tone → rewrite; never leave the instruction in the body.
   - JOB APPLY: Extract HR/apply email from the quoted JD (e.g. "Send CV to hr@…"). NEVER ask the user for an email that is already in the JD. Subject from JD "Subject:" line or "Application – <Position>". attach_resume=true. Cover letter from permanent notes + JD via draft_email.
   - Flow: draft_email → show draft (real newlines) → edit → send_email only after send it / bhej do.
   - Never claim sent unless send_email returned success.

{style_hint}
{format_block}
=== URL / WEB FACTS (NO HALLUCINATION) ===
12. Call browse_url ONLY when the CURRENT message has a URL (force_urls / priority note) OR the user clearly asks about a link/site ("details iska", "what is this about" with a link context, "fetch latest repo"). Never browse just because the last topic was a website.
13. When the user says "fetch latest repo" after a GitHub profile link, call browse_url on that exact github.com/username URL.
14. If a tool returns an error or empty data, say so honestly. Do not fabricate fallback facts.
15. For weather / temperature / mausam (e.g. Islamabad kitna garam hai), ALWAYS call get_weather — not web_search.
16. VIDEO / LINK CONTENT: When the CURRENT message has a video or social link and the user asks what it is about:
   - Vague "ye kya hai" / "what is this" on Instagram/Facebook/TikTok → prefer link_preview (title/caption/author only, no ASR).
   - Explicit transcript/summary/key points / "kya bola" / "summarize karo" → transcribe_video.
   - mode=transcript | summary | key_points as appropriate; target_words when user asks for N words.
   - Never browse_url for spoken video content. Never invent rate-limit / "fetch nahi hua" excuses.
   - If a tool says private/restricted/unavailable, relay that honestly in 1–2 lines.
   - Do NOT force video tools for reminder/admin/cancel/send commands even if a link is in the quote.
16b. TRANSCRIPT REPLY (tool already refined for the chosen mode):
   - Present the tool result almost as-is in ONE WhatsApp message.
   - Do NOT wrap the whole reply in code fences, do NOT promise "part 1 / more messages later", do NOT re-translate into Devanagari.
   - Light *emphasis* or a short - list (≤5) is allowed if it improves readability.
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
            # Prefer longer clips for profile/resume notes so later turns
            # (e.g. "write intro for this job") still see name, skills, exp.
            # Cap total injected chars so we never blow the context budget.
            clipped = []
            budget = 3500
            for n in memory_notes[:16]:
                s = str(n or "").strip()
                if not s:
                    continue
                # Profile-like notes get up to 1200 chars; short rules stay short
                per = 1200 if len(s) > 400 else 280
                take = s[:per]
                if len(take) > budget:
                    take = take[: max(0, budget)]
                if not take:
                    break
                clipped.append(take)
                budget -= len(take)
                if budget <= 0:
                    break
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

        _include_fmt = _needs_format_guide(
            latest_user_text=latest_user_text,
            force_urls=force_urls,
            extra_user_note=extra_user_note,
        )
        system = _build_system_prompt(
            chat_id,
            sender_id,
            is_group,
            time_context,
            tag_block,
            memory_block,
            include_format_guide=_include_fmt,
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

        # --- Force note_down (verbatim) when user asks to note + content is present ---
        import re as _re

        def _extract_note_body(text: str) -> Optional[str]:
            """Pull quoted message / document / image-OCR block for note_down."""
            if not text:
                return None
            m = _re.search(
                r"\[Quoted Message\]:\s*(.+)$",
                text,
                flags=_re.I | _re.S,
            )
            if m and m.group(1).strip():
                return m.group(1).strip()
            m = _re.search(
                r"\[(?:Document|Image content|Voice note transcript|Cached recent voice-note transcript)[^\]]*\]:\s*(.+)$",
                text,
                flags=_re.I | _re.S,
            )
            if m and m.group(1).strip():
                return m.group(1).strip()
            return None

        from note_intent import (
            is_note_intent as _is_note_intent,
            wants_condensed_note as _user_wants_condensed_note,
            content_fingerprint as _note_fp,
        )

        _intent_src = latest_user_text or extra_user_note or ""
        if not _intent_src:
            for m in reversed(messages):
                if m.get("role") == "user":
                    _intent_src = m.get("content") or ""
                    break
        if (
            _is_note_intent(_intent_src)
            and _tool_allowed_for_chat(chat_id, "note_down")
        ):
            body = _extract_note_body(_intent_src)
            if body and not _user_wants_condensed_note(_intent_src):
                # Verbatim force — no LLM rewrite that truncates
                print(
                    f"[AGENT] force note_down verbatim chars={len(body)}"
                )
                try:
                    obs = execute_tool(
                        "note_down", {"content": body}, tool_ctx
                    )
                except Exception as _ne:
                    obs = f"Tool error: {_ne}"
                obs_s = str(obs)
                if "Successfully saved" in obs_s:
                    logging.getLogger("mojo.agent").info(
                        "DIRECT_NOTE_DOWN chars=%s", len(body)
                    )
                    return _reply(
                        f"OneDrive (MojoAgent) mein note save ho gaya ✅ "
                        f"({len(body)} chars)."
                    )
                if "Permission denied" in obs_s:
                    return _reply("Note down sirf bot owner ke liye available hai.")
                # Soft failure: fingerprint only in history (never sliced body)
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_forced_note_0",
                        "type": "function",
                        "function": {
                            "name": "note_down",
                            "arguments": json.dumps(
                                {
                                    "content_ref": _note_fp(body),
                                    "status": "attempted",
                                }
                            ),
                        },
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": "call_forced_note_0",
                    "content": obs_s,
                })
                messages.append({
                    "role": "system",
                    "content": (
                        "note_down already ran. Tell the user the result in 1 short line. "
                        "Do not invent a successful save if the tool failed. "
                        "Do not invent or truncate note body text."
                    ),
                })

        # --- Force email draft / send (owner) — never invent a draft without the tool ---
        # Live bug: model replied "I've created a draft" with zero draft_email call,
        # so send_email later returned "No draft to send".
        try:
            from agent_tools import (
                parse_direct_email_command as _parse_email_cmd,
                parse_job_apply_command as _parse_job_apply,
                is_email_send_confirm as _is_email_confirm,
                _get_draft as _email_get_draft,
            )
        except Exception:
            _parse_email_cmd = None  # type: ignore
            _parse_job_apply = None  # type: ignore
            _is_email_confirm = None  # type: ignore
            _email_get_draft = None  # type: ignore

        _email_src = latest_user_text or extra_user_note or _intent_src or ""

        # Job apply: hard-force draft (no multi-tool loop — avoids search_knowledge
        # distraction, 413 from huge memory dumps, and RETURNING_TOOL_OBS of notes).
        if _parse_job_apply and _tool_allowed_for_chat(chat_id, "draft_email"):
            apply = _parse_job_apply(_email_src)
            if apply is not None:
                if not apply.get("to"):
                    print("[AGENT] apply intent but no email in JD")
                    return _reply(
                        "Is JD mein apply email nahi mili. HR/recruiter ka email "
                        "bhej do, phir apply kar deta hoon."
                    )
                print(
                    f"[AGENT] force job-apply HARD to={apply['to']!r} "
                    f"subj={apply.get('subject', '')[:50]!r}"
                )
                logging.getLogger("mojo.agent").info(
                    "FORCE_JOB_APPLY to=%s subject=%r",
                    apply["to"],
                    (apply.get("subject") or "")[:80],
                )
                # Compact profile for cover letter (cap hard — full notes already in prompt)
                try:
                    mem_obs = str(execute_tool("get_memory", {}, tool_ctx) or "")
                except Exception as _me:
                    mem_obs = ""
                # Prefer the always-on memory_block already injected if get_memory is noisy
                profile_clip = mem_obs
                if "[Chat note]" in profile_clip or len(profile_clip) > 1800:
                    # Strip repeated chat-note wrappers; keep substance
                    profile_clip = re.sub(
                        r"\[Chat note\][^\n]*\n?", " ", profile_clip
                    )
                    profile_clip = re.sub(
                        r"\[Quoted Message\]:\s*", "", profile_clip
                    )
                    profile_clip = re.sub(r"\s{2,}", " ", profile_clip).strip()
                profile_clip = (profile_clip or memory_block or "")[:1600]
                jd_clip = (apply.get("jd_excerpt") or "")[:1800]
                subject = apply.get("subject") or "Job Application"

                cover = ""
                try:
                    cover_msg = _chat_completion(
                        [
                            {
                                "role": "system",
                                "content": (
                                    "Write a concise plain-text job application email body "
                                    "(cover letter). 120–180 words. No markdown, no subject "
                                    "line, no 'Dear Sir/Madam' generic fluff if a name is "
                                    "unknown — use Hello Hiring Team. Tie 2–3 concrete "
                                    "candidate strengths to the role. End with a short close "
                                    "and the candidate name if present in the profile. "
                                    "Output ONLY the email body."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Role / subject: {subject}\n\n"
                                    f"JOB DESCRIPTION:\n{jd_clip}\n\n"
                                    f"CANDIDATE PROFILE:\n{profile_clip or '(see notes)'}"
                                ),
                            },
                        ],
                        tools=None,
                        temperature=0.4,
                        max_tokens=500,
                    )
                    cover = (getattr(cover_msg, "content", None) or "").strip()
                except Exception as _ce:
                    print(f"[AGENT] cover letter LLM failed: {_ce}")
                    cover = ""

                if not cover or len(cover) < 40:
                    # Deterministic fallback so apply never dies on 429/503
                    name = "Muhammad Zaheeruddin"
                    nm = re.search(
                        r"(?i)\*?Muhammad\s+Zaheer[^\n*]*",
                        profile_clip or "",
                    )
                    if nm:
                        name = re.sub(r"[*]", "", nm.group(0)).strip()
                    cover = (
                        f"Hello Hiring Team,\n\n"
                        f"I am writing to apply for the {subject} role. "
                        f"I bring hands-on experience in AI engineering, automation, "
                        f"and data-driven operations — including multi-agent systems, "
                        f"LLM workflows, and structured client delivery.\n\n"
                        f"I am self-motivated, comfortable owning end-to-end processes, "
                        f"and ready to contribute immediately in a remote setup with "
                        f"strong Excel/Sheets and sourcing discipline where needed.\n\n"
                        f"Please find my resume attached. Happy to share more detail "
                        f"or join a short call at your convenience.\n\n"
                        f"Best regards,\n{name}"
                    )

                try:
                    obs = execute_tool(
                        "draft_email",
                        {
                            "action": "create",
                            "to": apply["to"],
                            "subject": subject,
                            "body": cover,
                            "attach_resume": "true",
                            "purpose": "job_apply",
                            "reason": "job application — resume attached",
                        },
                        tool_ctx,
                    )
                except Exception as _de:
                    obs = f"Tool error: {_de}"
                obs_s = str(obs)
                logging.getLogger("mojo.agent").info(
                    "FORCE_JOB_APPLY_DRAFT obs=%s", obs_s[:240]
                )
                if "Permission denied" in obs_s:
                    return _reply(
                        "Job apply / email sirf bot owner ke liye available hai."
                    )
                if obs_s.startswith("Draft created") or obs_s.startswith(
                    "Draft updated"
                ):
                    return _reply(
                        obs_s
                        + "\n\nBhejna hai to 'send it' / 'bhej do' likho."
                    )
                return _reply(
                    "Draft banane mein issue aaya. Dobara 'apply to this job' "
                    "try karo ya HR email confirm karo."
                )

        if _parse_email_cmd and _tool_allowed_for_chat(chat_id, "draft_email"):
            parsed = _parse_email_cmd(_email_src)
            if parsed and parsed.get("to") and parsed.get("body"):
                wants_rewrite = (parsed.get("exact") or "true").lower() == "false"
                rewrite_instr = (parsed.get("rewrite") or "").strip()

                if wants_rewrite:
                    # User asked to rewrite (e.g. "make it professional") — do NOT
                    # force the raw body. Steer the model to draft_email with a
                    # rewritten body matching the instruction.
                    print(
                        f"[AGENT] email rewrite requested to={parsed['to']!r} "
                        f"instr={rewrite_instr[:60]!r}"
                    )
                    messages.append({
                        "role": "system",
                        "content": (
                            "OUTBOUND EMAIL — REWRITE MODE (user asked for a rewrite).\n"
                            f"Recipient: {parsed['to']}\n"
                            f"Rewrite instruction: {rewrite_instr or 'professional tone'}\n"
                            f"User's raw notes (use as source facts, NOT as final body):\n"
                            f"{parsed['body']}\n\n"
                            "You MUST call draft_email action=create with:\n"
                            f"- to: {parsed['to']}\n"
                            "- subject: short professional subject (invent if needed)\n"
                            "- body: rewritten email following the instruction; "
                            "do NOT paste the raw notes verbatim; do NOT include the "
                            "rewrite instruction itself in the body\n"
                            "- attach_resume: false\n"
                            "After the tool returns, show the draft card as-is "
                            "(keep newlines). Ask if they want to send."
                        ),
                    })
                else:
                    # DEFAULT: word-for-word exact body — no LLM rewrite
                    print(
                        f"[AGENT] force draft_email EXACT to={parsed['to']!r} "
                        f"subj={parsed.get('subject', '')[:40]!r}"
                    )
                    try:
                        obs = execute_tool(
                            "draft_email",
                            {
                                "action": "create",
                                "to": parsed["to"],
                                "subject": parsed.get("subject") or "Quick note",
                                "body": parsed["body"],
                                "attach_resume": "false",
                                "purpose": "direct_send",
                                "reason": "exact user body (no rewrite asked)",
                            },
                            tool_ctx,
                        )
                    except Exception as _ee:
                        obs = f"Tool error: {_ee}"
                    obs_s = str(obs)
                    logging.getLogger("mojo.agent").info(
                        "FORCE_DRAFT_EMAIL exact=1 obs=%s", obs_s[:200]
                    )
                    if "Permission denied" in obs_s:
                        return _reply(
                            "Email draft/send sirf bot owner ke liye available hai."
                        )
                    if obs_s.startswith("Draft created") or obs_s.startswith(
                        "Draft updated"
                    ):
                        return _reply(
                            obs_s
                            + "\n\nBhejna hai to 'send it' / 'bhej do' likho."
                        )
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_forced_draft_email_0",
                            "type": "function",
                            "function": {
                                "name": "draft_email",
                                "arguments": json.dumps({
                                    "action": "create",
                                    "to": parsed["to"],
                                    "subject": parsed.get("subject")
                                    or "Quick note",
                                    "body": parsed["body"],
                                    "attach_resume": "false",
                                }),
                            },
                        }],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": "call_forced_draft_email_0",
                        "content": obs_s,
                    })
                    messages.append({
                        "role": "system",
                        "content": (
                            "draft_email already ran with the user's EXACT body. "
                            "Show the tool result as-is (preserve newlines). "
                            "Do NOT rewrite the body. Do NOT glue To/Subject/Body "
                            "onto one line."
                        ),
                    })

        if (
            _is_email_confirm
            and _email_get_draft
            and _is_email_confirm(_email_src)
            and _tool_allowed_for_chat(chat_id, "send_email")
        ):
            existing_draft = None
            try:
                existing_draft = _email_get_draft(tool_ctx)
            except Exception:
                existing_draft = None
            if existing_draft:
                print(
                    f"[AGENT] force send_email to={existing_draft.get('to')!r}"
                )
                send_args = {"confirm": True}
                # If draft had attach=ask, default false on bare "send it" unless
                # user said attach/resume/cv in the confirm message.
                att = (existing_draft.get("attach_resume") or "ask").lower()
                low_src = (_email_src or "").lower()
                if att == "ask":
                    if any(
                        w in low_src
                        for w in ("attach", "resume", "cv", "pdf", "with resume")
                    ):
                        send_args["attach_resume_override"] = "true"
                    else:
                        send_args["attach_resume_override"] = "false"
                try:
                    obs = execute_tool("send_email", send_args, tool_ctx)
                except Exception as _se:
                    obs = f"Tool error: {_se}"
                obs_s = str(obs)
                logging.getLogger("mojo.agent").info(
                    "FORCE_SEND_EMAIL obs=%s", obs_s[:240]
                )
                return _reply(obs_s)
            else:
                # Confirm with no draft — tell user clearly (don't invent send)
                messages.append({
                    "role": "system",
                    "content": (
                        "User asked to SEND email but there is NO stored draft for this "
                        "chat. Do NOT claim you sent anything. Ask them to re-state "
                        "recipient + body (or quote the prior draft request) so "
                        "draft_email can run first."
                    ),
                })

        # Force-notice URLs from the CURRENT WhatsApp message (including quoted links)
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
        tw = None
        spoken_mode = None

        def _parse_target_words(text_low: str) -> Optional[int]:
            # 1.2k / 1.5k / 2k words (optional trailing me/mein)
            m = _re.search(
                r"(\d+(?:[.,]\d+)?)\s*[kK]\s*(?:words?|lafz)?",
                text_low,
            )
            if m:
                try:
                    n = int(float(m.group(1).replace(",", ".")) * 1000)
                    return max(80, min(2500, n))
                except ValueError:
                    pass
            m = _re.search(
                r"(\d{1,3}(?:,\d{3})+|\d{2,4})\s*[- ]?\s*(?:words?|lafz)",
                text_low,
            )
            if not m:
                m = _re.search(r"(\d{2,4})\s*word", text_low)
            if not m:
                return None
            try:
                n = int(m.group(1).replace(",", ""))
                return max(80, min(2500, n))
            except ValueError:
                return None

        def _obs_looks_failed_or_thin(obs: str) -> bool:
            o = (obs or "").strip().lower()
            if not o or len(o) < 40:
                return True
            # Error/refusal strings are at the START of observations.
            # Scanning the full 10k+ body for "failed"/"available" false-positives
            # on real lecture transcripts and wrongly triggers link_preview.
            head = o[:300]
            markers = (
                "tool error",
                "disabled for this chat",
                "bahut short",
                "unclear hai",
                "private",
                "restricted",
                "nahi nikal",
                "not available",
                "could not download",
                "transcript bahut",
                "login wall",
                "unavailable",
                "failed",
            )
            return any(m in head for m in markers)

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
            is_video = is_video_url(primary_url)
            blocked_cmd = _is_blocked_force_command(_intent_low)
            spoken_mode = None
            if is_video and not blocked_cmd:
                spoken_mode = video_spoken_intent(_intent_low)
            elif is_video and blocked_cmd:
                print(
                    f"[AGENT] SKIP force video tools — blocked command intent "
                    f"{_strip_intent_noise(_intent_low)[:80]!r}"
                )

            use_video = bool(spoken_mode and is_video and not blocked_cmd)
            can_transcribe = _tool_allowed_for_chat(chat_id, "transcribe_video")
            can_preview = _tool_allowed_for_chat(chat_id, "link_preview")
            can_browse = _tool_allowed_for_chat(chat_id, "browse_url")
            vague_about = use_video and _is_vague_about_ask(_intent_low)
            # Prefer cheap caption preview for vague "ye kya hai" on IG/FB/TikTok
            prefer_preview = vague_about and can_preview and any(
                h in (primary_url or "").lower()
                for h in ("instagram.com", "instagr.am", "facebook.com", "fb.watch", "tiktok.com")
            )

            def _force_tool(name: str, args: dict, call_id: str) -> str:
                try:
                    obs = execute_tool(name, args, tool_ctx)
                except Exception as _fe:
                    obs = f"Tool error: {_fe}"
                obs_s = str(obs)
                cap = (
                    MAX_OBS_CHARS_TRANSCRIPT
                    if name == "transcribe_video"
                    else MAX_OBS_CHARS
                )
                if len(obs_s) > cap:
                    obs_s = obs_s[:cap] + "…"
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args),
                        },
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": obs_s,
                })
                return obs_s

            if prefer_preview:
                print(f"[AGENT] force link_preview (vague about) for: {primary_url}")
                _forced_obs_str = _force_tool(
                    "link_preview", {"url": primary_url}, "call_forced_preview_0"
                )
                # If caption-only is weak and speech tools allowed, try transcribe
                if _obs_looks_failed_or_thin(_forced_obs_str) and can_transcribe:
                    print("[AGENT] link_preview thin/failed → fallback transcribe_video")
                    tw = _parse_target_words(_intent_low)
                    targs: Dict[str, Any] = {
                        "url": primary_url,
                        "mode": spoken_mode or "summary",
                        "language": "auto",
                        "timestamps": False,
                    }
                    if tw:
                        targs["target_words"] = tw
                    _forced_obs_str = _force_tool(
                        "transcribe_video", targs, "call_forced_transcribe_0"
                    )
                direct = _last_tool_obs_for_user(messages)
                if direct and len(direct) >= 20:
                    logging.getLogger("mojo.agent").info(
                        "DIRECT_OBS_REPLY mode=preview/fallback chars=%s", len(direct)
                    )
                    return _reply(direct)
                tools_for_next = None

            elif use_video and can_transcribe:
                tw = _parse_target_words(_intent_low)
                # Caption language stays auto (Hindi auto-captions still fetch).
                # Shared language policy (default Roman Urdu; en only if explicit)
                try:
                    from language_policy import detect_output_lang, to_tool_language_arg
                    _lang = to_tool_language_arg(detect_output_lang(_intent_low or ""))
                except Exception:
                    _lang = "auto"
                    _il = _intent_low or ""
                    if any(
                        p in _il
                        for p in (
                            "in english",
                            "english me",
                            "english mein",
                            "angrezi me",
                            "angrezi mein",
                            "translate to english",
                        )
                    ):
                        _lang = "en"
                tool_args: Dict[str, Any] = {
                    "url": primary_url,
                    "mode": spoken_mode,
                    "language": _lang,
                    "timestamps": False,
                    "user_request": (_intent_text or "")[:500],
                }
                if not tw and spoken_mode == "summary":
                    _il = (_intent_low or "")
                    if any(
                        m in _il
                        for m in (
                            "break down", "breakdown", "har cheez", "in depth",
                            "indepth", "detailed", "sab kuch", "poora", "thorough",
                            "point by point", "step by step", "action by action",
                            "analysis", "analyse", "analyze", "explain",
                        )
                    ):
                        tw = 1000
                if tw and spoken_mode in ("summary", "transcript", "key_points"):
                    tool_args["target_words"] = tw
                print(
                    f"[AGENT] force transcribe_video mode={spoken_mode} "
                    f"lang={_lang} tw={tw} for: {primary_url}"
                )
                _forced_obs_str = _force_tool(
                    "transcribe_video", tool_args, "call_forced_transcribe_0"
                )
                # ASR blocked / thin → caption preview (esp. private IG)
                if _obs_looks_failed_or_thin(_forced_obs_str) and can_preview:
                    print("[AGENT] transcribe thin/failed → fallback link_preview")
                    _forced_obs_str = _force_tool(
                        "link_preview",
                        {"url": primary_url},
                        "call_forced_preview_1",
                    )
                direct = _last_tool_obs_for_user(messages)
                # Never serve YouTube description/promo as a transcript/summary
                if direct:
                    try:
                        from agent_tools import _looks_like_youtube_description
                        if _looks_like_youtube_description(direct):
                            print(
                                "[AGENT] DIRECT_OBS looks like YT description "
                                f"(chars={len(direct)}) — discarding"
                            )
                            logging.getLogger("mojo.agent").info(
                                "DIRECT_OBS_DROP description chars=%s", len(direct)
                            )
                            direct = None
                            # Retry once from raw speech cache (same source as good
                            # short summaries) so we do not fall back to a truncated
                            # free-form synthesis without transcript context.
                            try:
                                from agent_tools import (
                                    _RAW_SPEECH_CACHE,
                                    _extract_youtube_id,
                                    _refine_transcript_compact,
                                    _looks_like_youtube_description as _desc,
                                )
                                import time as _time
                                _vid = _extract_youtube_id(primary_url)
                                _raw = None
                                if _vid and _vid in _RAW_SPEECH_CACHE:
                                    _txt, _exp = _RAW_SPEECH_CACHE[_vid]
                                    if _time.time() <= _exp and _txt and not _desc(_txt):
                                        _raw = _txt
                                if _raw:
                                    try:
                                        from language_policy import detect_output_lang
                                        _retry_lang = detect_output_lang(
                                            _intent_low or ""
                                        )
                                    except Exception:
                                        _retry_lang = "en" if _lang == "en" else "roman_urdu"
                                    _retry = _refine_transcript_compact(
                                        _raw,
                                        title="",
                                        mode=spoken_mode or "summary",
                                        target_words=tw,
                                        output_lang=_retry_lang,
                                        user_request=(_intent_text or "")[:500],
                                    )
                                    if (
                                        _retry
                                        and not _desc(_retry)
                                        and len(_retry) >= 80
                                    ):
                                        direct = _retry
                                        print(
                                            f"[AGENT] summary retry from RAW_CACHE "
                                            f"chars={len(direct)} tw={tw}"
                                        )
                                        logging.getLogger("mojo.agent").info(
                                            "DIRECT_OBS_RETRY raw_cache chars=%s",
                                            len(direct),
                                        )
                                        # Replace last tool OBS so synthesis sees good text
                                        for _mi in range(len(messages) - 1, -1, -1):
                                            if messages[_mi].get("role") == "tool":
                                                messages[_mi]["content"] = (
                                                    f"Source: raw_cache_retry | "
                                                    f"mode={spoken_mode}\n\n{direct}"
                                                )
                                                break
                            except Exception as _re:
                                print(f"[AGENT] raw_cache summary retry failed: {_re}")
                    except Exception as _e:
                        print(f"[AGENT] description gate skip: {_e}")
                # Reject mostly-untranslated Hindi dumps when user wanted
                # English/Roman analysis or summary (not raw speech).
                if direct and spoken_mode in ("summary", "key_points"):
                    try:
                        from language_policy import non_latin_ratio, has_non_latin_script
                        from agent_tools import (
                            _RAW_SPEECH_CACHE,
                            _extract_youtube_id,
                            _refine_transcript_compact,
                            _looks_like_youtube_description as _desc,
                        )
                        import time as _time
                        low_d = (direct or "").lower()
                        bad_dump = (
                            "[untranslated" in low_d
                            or "kuch hissa translate nahi" in low_d
                            or (
                                has_non_latin_script(direct)
                                and non_latin_ratio(direct) > 0.35
                            )
                        )
                        if bad_dump:
                            print(
                                "[AGENT] DIRECT_OBS untranslated/native dump — "
                                "forcing summary refine from RAW_CACHE"
                            )
                            _vid = _extract_youtube_id(primary_url)
                            _raw = None
                            if _vid and _vid in _RAW_SPEECH_CACHE:
                                _txt, _exp = _RAW_SPEECH_CACHE[_vid]
                                if _time.time() <= _exp and _txt and not _desc(_txt):
                                    _raw = _txt
                            if _raw:
                                try:
                                    from language_policy import detect_output_lang
                                    _rl = detect_output_lang(_intent_low or "")
                                except Exception:
                                    _rl = "en" if _lang == "en" else "roman_urdu"
                                _retry = _refine_transcript_compact(
                                    _raw,
                                    title="",
                                    mode="summary",
                                    target_words=tw or 1000,
                                    output_lang=_rl,
                                    user_request=(_intent_text or "")[:500],
                                )
                                if (
                                    _retry
                                    and not _desc(_retry)
                                    and len(_retry) >= 80
                                    and non_latin_ratio(_retry) < 0.25
                                ):
                                    direct = _retry
                                    logging.getLogger("mojo.agent").info(
                                        "DIRECT_OBS_RETRY summary chars=%s",
                                        len(direct),
                                    )
                    except Exception as _ue:
                        print(f"[AGENT] untranslated-dump gate skip: {_ue}")

                if direct and len(direct) >= 20:
                    print(
                        "[AGENT] force transcribe/preview → direct OBS reply "
                        f"(chars={len(direct)}, starts={direct[:40]!r})"
                    )
                    logging.getLogger("mojo.agent").info(
                        "DIRECT_OBS_REPLY mode=%s chars=%s",
                        spoken_mode,
                        len(direct),
                    )
                    return _reply(direct)
                messages.append({
                    "role": "system",
                    "content": (
                        f"You already ran video tools on {primary_url} "
                        f"(mode={spoken_mode}). Answer from that tool result only in "
                        "ONE complete message (do not cut mid-sentence). "
                        "Present the content almost as-is. If the tool "
                        "reported an error (private video, download failed, etc.), "
                        "relay that honestly in 1–2 lines. Do NOT invent a rate-limit / "
                        "request-limit / 'fetch nahi hua' / 'thodi der baad' excuse. "
                        "Do NOT call transcribe_video or browse_url again this turn."
                    ),
                })
                tools_for_next = None  # synthesis only — no tool schema payload
                # Long summary / transcript needs a higher completion budget
                if tw and tw >= 400:
                    # stashed for the model call below
                    messages.append({
                        "role": "system",
                        "content": f"[INTERNAL] target_words={tw} long_reply=1",
                    })
            elif use_video and can_preview:
                # transcribe disabled but preview allowed
                print(f"[AGENT] force link_preview (no transcribe) for: {primary_url}")
                _forced_obs_str = _force_tool(
                    "link_preview", {"url": primary_url}, "call_forced_preview_0"
                )
                direct = _last_tool_obs_for_user(messages)
                if direct and len(direct) >= 20:
                    return _reply(direct)
                tools_for_next = None
            elif use_video and not can_transcribe and not can_preview:
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
            elif can_browse and not blocked_cmd:
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
            elif blocked_cmd and urls_in_last:
                # Remember / reminder / admin intent — do NOT force-browse incidental
                # links embedded in a quoted resume, job post, or standing note.
                print(
                    f"[AGENT] SKIP force URL tools — blocked command intent "
                    f"{_strip_intent_noise(_intent_low)[:80]!r}"
                )
            else:
                # URL present but neither transcribe nor browse is allowed
                print(
                    f"[AGENT] SKIP force URL tools — disabled for {chat_id} url={primary_url}"
                )
                logging.getLogger("mojo.agent").info(
                    "TOOL_BLOCKED browse_url/transcribe chat=%s", chat_id
                )

        # --- Agentic loop ---
        # Long video summaries need far more than the default 1024-token cap
        # (1.5k words ≈ 2k+ tokens; 1024 cut the previous reply mid-sentence).
        _reply_max_tokens = MAX_COMPLETION_TOKENS
        try:
            if tw and int(tw) >= 400:
                _reply_max_tokens = max(
                    MAX_COMPLETION_TOKENS,
                    min(4500, int(int(tw) * 1.8) + 200),
                )
            elif spoken_mode in ("transcript", "summary") and any(
                (m.get("role") == "system" and "long_reply=1" in str(m.get("content") or ""))
                for m in messages
            ):
                _reply_max_tokens = max(MAX_COMPLETION_TOKENS, 3000)
        except Exception:
            _reply_max_tokens = MAX_COMPLETION_TOKENS

        for step in range(MAX_TOOL_STEPS):
            try:
                msg = _chat_completion(
                    messages,
                    tools=tools_for_next,
                    temperature=0.3,
                    max_tokens=_reply_max_tokens,
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
                    return _reply(obs)
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
                            return _reply(ans)
                    except Exception:
                        pass
                return _reply("Thori si technical issue aa gayi — ek second baad dobara try karo.")

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
                    # Structured OBS path — remind model of native markers cheaply
                    messages.append({
                        "role": "system",
                        "content": (
                            "WhatsApp style: use *bold* _italic_ ~strike~ `code` "
                            "(single markers). Never **double** asterisks or ### headers. "
                            "Lists max 5 items."
                        ),
                    })
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
                            return _reply(forced)
                    except Exception as fe:
                        print(f"[AGENT] forced synthesis failed: {fe}")
                    obs = _last_tool_obs_for_user(messages)
                    if obs:
                        return _reply(obs)
                return _reply(answer or "Theek hai 👍")

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
                return _reply(ans)
        except Exception as e:
            print(f"[AGENT] final close failed: {e}")
        obs = _last_tool_obs_for_user(messages)
        if obs:
            return _reply(obs)
        return _reply("Kaam almost complete ho gaya — dobara try kar lo.")

    except Exception as e:
        traceback.print_exc()
        print(f"[AGENT ERROR] {e}")
        logging.getLogger("mojo.agent").exception("run_agent failed: %s", e)
        return _reply("Thori si technical issue aa gayi — ek second baad dobara try karo.")