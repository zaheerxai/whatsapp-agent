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
        # Auto-raise cap when this OBS is a full transcript (not a short summary)
        use_len = max_len
        if max_len <= 3500 and len(obs) > 3500:
            # Prefer full speech when available (transcribe_video mode=transcript)
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
- Bulleted list: start a line with "- " or "* " (hyphen/asterisk + space). Use for 2–5 parallel items.
- Numbered list: start a line with "1. " "2. " etc. Use only for real sequential steps. Hard max 5 items.
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
        lines_out.append(line)
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

    # Numbered list soft-cap (keep more for long summaries — 12 not 5)
    capped: List[str] = []
    num_count = 0
    for line in t.split("\n"):
        m = re.match(r"^(\s*)(\d{1,2})\.\s+(.*)$", line)
        if m:
            num_count += 1
            if num_count > 12:
                capped.append(f"{m.group(1)}- {m.group(3)}")
            else:
                capped.append(f"{m.group(1)}{num_count}. {m.group(3)}")
        else:
            if not line.strip():
                num_count = 0
            elif not re.match(r"^\s*([-*]|\d{1,2}\.)\s+", line):
                num_count = 0
            capped.append(line)
    t = "\n".join(capped)

    # Collapse 3+ blank lines → 2; trim trailing spaces per line
    t = "\n".join(ln.rstrip() for ln in t.split("\n"))
    t = re.sub(r"\n{3,}", "\n\n", t)
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
   - Triggers: apply to this / send resume / cover letter / email this to X / invite to collaborate / offer demo / pitch / bhej do email (any language).
   - Use permanent notes (resume/profile) + quoted JD or recipient. If only a company email is given, draft from notes + user ask; web_search company only when it clearly improves personalization.
   - Flow: (1) draft_email action=create with to/subject/body/attach_resume (2) show draft (3) iterate action=update until user confirms (4) send_email confirm=true ONLY after explicit send/bhej do/confirm.
   - attach_resume: true for job apply / send CV / cover letter; false for pure collab invite without CV; ask when unsure and WAIT for yes/no.
   - Direct form: "email this to a@b.com: subject: …" or subject on first line + body after linebreak → draft then confirm. If no subject, invent a tight one from the body.
   - Never claim sent unless send_email returned success. Body: concise, high-impact, plain text (no markdown fences). Prefer English for intl roles.

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