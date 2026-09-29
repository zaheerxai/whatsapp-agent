"""
agent_tools.py — Tool schemas + executors for Mojo's agentic loop.

All tools return a plain string observation that the LLM sees.
Permission-sensitive tools (send_message_to, python_exec) check OWNER_SENDER_ID.
"""

from __future__ import annotations

import json
import logging
import os
import re
import random
import time
import traceback
import tempfile
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests

# ---------------------------------------------------------------------------
# Single source of truth for script/language matching, shared by agent_loop.py
# (final-answer system prompt) and whatsapp_agent.py's reminder_scheduler()
# (scheduled reminder delivery). Previously each maintained its own copy of
# this rule with different wording — this is the one place to edit it.
# ---------------------------------------------------------------------------
LANGUAGE_POLICY = """- Match the user's QUERY language when clear: English message → English reply; Roman Urdu message → Roman Urdu reply.
- DEFAULT when mixed/unclear: Roman Urdu (Latin letters only). Never output Devanagari or Arabic/Urdu script unless the user explicitly asked for that script.
- Explicit English ask ("in english", "english me", "angrezi") → English.
- Explicit native-script ask only ("urdu script", "hindi letters", "اردو میں لکھو", "देवनागरी") → that script.
- "urdu me" / "hindi me" without script/letters/likhai → Roman Urdu (Latin), NOT native script.
- Voice notes / ASR: match the spoken request language; otherwise Roman Urdu.
- Tools (transcript, summary, notes): same rules — English when the query is English or explicitly asked; otherwise Roman Urdu Latin letters.
"""


# These are injected by whatsapp_agent after import to avoid circular deps
_supabase = None
_client_ai = None
_MODEL_NAME = None
_BUSINESS_KNOWLEDGE = ""
_DEFAULT_TIMEZONE = "Asia/Karachi"
_OWNER_SENDER_ID = ""
_get_contacts_maps = None
_get_user_timezone = None
_set_user_timezone = None
_get_tzinfo = None
_handle_reminder_request = None
_list_reminders = None
_cancel_reminders = None
_get_group_memory = None
_send_proactive_message = None
_file_ops = None
_compute_next_occurrence = None
_extract_reminder_data_via_ai = None
_generate_reminder_confirmation = None
_client_gemini = None
_GEMINI_MODEL = None


def init_tools(
    supabase,
    client_ai,
    model_name,
    business_knowledge,
    default_timezone,
    owner_sender_id,
    get_contacts_maps,
    get_user_timezone,
    set_user_timezone,
    get_tzinfo,
    handle_reminder_request,
    list_reminders,
    cancel_reminders,
    get_group_memory,
    send_proactive_message,
    file_ops_module,
    compute_next_occurrence=None,
    extract_reminder_data_via_ai=None,
    generate_reminder_confirmation=None,
    client_gemini=None,
    gemini_model=None,
):
    global _supabase, _client_ai, _MODEL_NAME, _BUSINESS_KNOWLEDGE
    global _DEFAULT_TIMEZONE, _OWNER_SENDER_ID
    global _get_contacts_maps, _get_user_timezone, _set_user_timezone, _get_tzinfo
    global _handle_reminder_request, _list_reminders, _cancel_reminders
    global _get_group_memory, _send_proactive_message, _file_ops
    global _compute_next_occurrence, _extract_reminder_data_via_ai, _generate_reminder_confirmation
    global _client_gemini, _GEMINI_MODEL

    _supabase = supabase
    _client_ai = client_ai
    _MODEL_NAME = model_name
    _BUSINESS_KNOWLEDGE = business_knowledge or ""
    _DEFAULT_TIMEZONE = default_timezone
    _OWNER_SENDER_ID = (owner_sender_id or "").strip()
    _get_contacts_maps = get_contacts_maps
    _get_user_timezone = get_user_timezone
    _set_user_timezone = set_user_timezone
    _get_tzinfo = get_tzinfo
    _handle_reminder_request = handle_reminder_request
    _list_reminders = list_reminders
    _cancel_reminders = cancel_reminders
    _get_group_memory = get_group_memory
    _send_proactive_message = send_proactive_message
    _file_ops = file_ops_module
    _compute_next_occurrence = compute_next_occurrence
    _extract_reminder_data_via_ai = extract_reminder_data_via_ai
    _generate_reminder_confirmation = generate_reminder_confirmation
    _client_gemini = client_gemini
    _GEMINI_MODEL = gemini_model


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI-compatible)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Search Mojo AI Agency knowledge base (hybrid RAG over agency docs + "
                "OneDrive Documents/aimojo) and permanent chat notes. "
                "Use for questions about services, portfolio, founder, pricing process, "
                "uploaded docs/exports, or standing rules/facts for this chat. "
                "Always call this before answering business or 'what can you do' questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in English or Roman Urdu",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max snippets to return (default 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the live web for current facts, news, prices, or anything "
                "not in the knowledge base. Returns short result snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {
                        "type": "integer",
                        "description": "How many results (1-8)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse_url",
            "description": (
                "Fetch live content of a public URL. "
                "For github.com/username it returns the REAL public repo list via GitHub API "
                "(sorted by last push). For github.com/user/repo it returns repo metadata. "
                "For other sites it returns page title + main text. "
                "ALWAYS call this when the user pastes a URL or asks to fetch/open/latest repos "
                "from a previously shared link. NEVER invent repo names or page content. "
                "If the CURRENT user message contains a URL, that URL has priority over older links in history."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full https URL (GitHub profile, repo, article, etc.)",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Max characters for non-GitHub pages (default 2500)",
                        "default": 2500,
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get live weather for a city (temperature, feels-like, humidity, condition). "
                "Use this for any weather / temperature / mausam question — more accurate than web_search. "
                "Examples: Islamabad, Karachi, Lahore, London, Dubai."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. Islamabad, Karachi",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_memory",
            "description": (
                "Retrieve permanent notes/rules saved for this chat (standing instructions, "
                "preferences, facts about people)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_id": {
                        "type": "string",
                        "description": "Chat ID (usually already known from context)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": (
                "Save a durable fact, rule, preference, or full profile/resume for this chat "
                "so it persists across future conversations. Use when user says "
                "'always remember', 'remember this', 'yaad rakh', 'from now on...', or gives "
                "a standing instruction. When a quoted body / resume / OCR dump is present, "
                "pass the FULL content as note (do not collapse to a 1-line stub)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {
                        "type": "string",
                        "description": (
                            "Content to store. Short fact/rule for simple prefs; "
                            "FULL quoted body / resume / profile when user is storing a document."
                        ),
                    },
                    "sender_id": {
                        "type": "string",
                        "description": "Who stated it (LID)",
                    },
                },
                "required": ["note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": (
                "Create one or more real database reminders (one-off or recurring). "
                "Understands Roman Urdu / mixed language e.g. '1 min me paani peena', "
                "'remind karna 1min me', 'roz 5 baje', voice transcripts about reminders. "
                "ALWAYS use this tool — never tell the user to set a phone timer instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Full natural-language reminder request from the user",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": (
                "List all active reminders for the current user in this chat. "
                "Use for: list reminder, list reminders, my reminders, show reminders, "
                "kya reminders hain, etc."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_reminders",
            "description": (
                "Cancel one or more active reminders. Pass the subject keywords "
                "(e.g. 'paani', 'water', 'debug', 'all') or 'all' to cancel everything."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords matching the reminder subject, or 'all'",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_contacts",
            "description": (
                "Look up known contacts by name fragment or phone. Useful for tagging "
                "or answering 'who is in this chat'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Name or number fragment",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_user",
            "description": "Get details for a specific sender_id / LID (name, number, timezone).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sender_id": {"type": "string"},
                },
                "required": ["sender_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message_to",
            "description": (
                "Send a proactive WhatsApp message to another chat_id. "
                "ONLY the bot owner may use this tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_id": {
                        "type": "string",
                        "description": "Target chat_id e.g. 92300...@s.whatsapp.net or group@g.us",
                    },
                    "text": {"type": "string", "description": "Message body"},
                },
                "required": ["chat_id", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_list_onedrive",
            "description": (
                "List files in the OneDrive knowledge base (Documents/aimojo by default). "
                "Optional folder: agency | docs | chats or a full path under the knowledge root."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string",
                        "description": "Subfolder under Documents/aimojo (agency, docs, chats) or full path",
                    },
                    "limit": {"type": "integer", "default": 50},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": (
                "Run a short pure-Python expression or statement in a sandbox "
                "(no file/network access). Owner only. Useful for quick calculations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code (max ~20 lines)"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "image_describe",
            "description": (
                "Describe a photo/image the user sent or quoted: scene, visible text, objects. "
                "Use for 'ye photo kya hai', 'what is in this image', 'describe this pic'. "
                "Does NOT write to OneDrive. Prefer this over note_down when the user only "
                "wants to understand the image. Vision runs on the media path when an image "
                "is attached or quoted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": (
                            "Optional focus: text_ocr | scene | objects | all. Default all."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "note_down",
            "description": (
                "Save or append a note to the owner's OneDrive journal (Mojo_Notes.txt). "
                "Call whenever the user says note down / note karlo / onedrive me note / save this. "
                "DEFAULT: pass the content WORD-FOR-WORD (full quoted text, full OCR, full "
                "transcript). Do NOT summarize, shorten, or add ellipsis (…) unless the user "
                "explicitly asked for summary / khulasa / key points / bullets. "
                "Only the bot owner can use this."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": (
                            "Full note body to save. Verbatim by default. "
                            "Only condensed if user asked summary/key points."
                        ),
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "link_preview",
            "description": (
                "Fetch public link metadata only: title, caption/description, author. "
                "NO speech-to-text, NO audio download. Use for 'ye kya hai' / 'what is this' "
                "on Instagram/Facebook/TikTok when full transcript is unnecessary, blocked, "
                "or private. Prefer this over transcribe_video for quick about/caption asks. "
                "Do NOT invent captions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full https URL (Instagram reel, Facebook video, TikTok, etc.)",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transcribe_video",
            "description": (
                "Get spoken content from a public video link (YouTube, TikTok, Vimeo, etc.). "
                "Use when user asks for transcript / summary / key points / 'is video me kya bola'. "
                "ALWAYS call again if user changes mode or asks for a longer/shorter version — "
                "do not invent from memory. "
                "mode: transcript | summary | key_points. "
                "If user asks for N words (e.g. 400-word summary), set target_words=N. "
                "Do NOT use for voice notes already transcribed in the chat. "
                "If audio is blocked, prefer link_preview for caption/about."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full public video URL (YouTube, youtu.be, Vimeo, etc.)",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["transcript", "summary", "key_points"],
                        "description": (
                            "transcript = cleaned spoken text. "
                            "summary = overview (use with target_words if user asks for length). "
                            "key_points = bullet list only."
                        ),
                        "default": "transcript",
                    },
                    "target_words": {
                        "type": "integer",
                        "description": (
                            "Optional approximate word count when user asks e.g. "
                            "'400 word summary' or '600 words'. Clamp 80–800."
                        ),
                    },
                    "language": {
                        "type": "string",
                        "description": (
                            "Preferred language code e.g. en, ur, hi, auto. "
                            "Default auto / best available."
                        ),
                        "default": "auto",
                    },
                    "timestamps": {
                        "type": "boolean",
                        "description": "If true, include [MM:SS] timestamps (transcript mode)",
                        "default": False,
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_email",
            "description": (
                "Create, update, show, or cancel an outbound email DRAFT for the owner. "
                "Use for job applications, cover letters, collab invites, demos, or any "
                "email the user wants sent. ALWAYS draft first — never send without "
                "explicit user confirmation (user says 'send it' / 'bhej do' / 'confirm'). "
                "Pull recipient from JD text / quoted email when possible. "
                "Body should be personalized, concise, high-impact. "
                "attach_resume: true | false | ask — reason from intent "
                "(job apply / send CV → true; pure collab invite without CV request → false; "
                "unsure → ask). action=create|update|show|cancel."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "update", "show", "cancel"],
                        "description": "create new draft, update existing, show current, or cancel",
                        "default": "create",
                    },
                    "to": {
                        "type": "string",
                        "description": "Recipient email (required on create unless already drafted)",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Subject line. If empty on create, model should invent one.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Plain-text email body (cover letter / intro / pitch)",
                    },
                    "attach_resume": {
                        "type": "string",
                        "enum": ["true", "false", "ask"],
                        "description": (
                            "true=attach resume PDF, false=no attachment, "
                            "ask=uncertain — draft is saved but user must confirm attachment"
                        ),
                        "default": "ask",
                    },
                    "purpose": {
                        "type": "string",
                        "description": (
                            "Short label e.g. job_apply, collab_invite, demo_offer, "
                            "direct_send — for logging and attachment heuristics"
                        ),
                    },
                    "cc": {"type": "string", "description": "Optional CC"},
                    "reason": {
                        "type": "string",
                        "description": "Why attach_resume was chosen (shown to user when ask)",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": (
                "Send the CURRENT confirmed email draft via Gmail (owner-only). "
                "ONLY call after the user explicitly confirms the draft "
                "('send it', 'bhej do', 'confirm send', 'haan bhej do'). "
                "If no draft exists, refuse. If attach_resume was 'ask' and user "
                "never resolved it, ask again — do not guess."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be true to actually send",
                        "default": False,
                    },
                    "attach_resume_override": {
                        "type": "string",
                        "enum": ["true", "false", ""],
                        "description": (
                            "If draft had attach=ask, user answer: true/false. "
                            "Empty keeps draft setting."
                        ),
                    },
                },
                "required": ["confirm"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Email draft store
# Owner-only; keyed primarily by chat_id so LID vs phone number never misses.
# Also mirrored under chat|sender_id and chat|sender_num for safety.
# Optional /tmp JSON so a quick process bounce still has the last draft.
# ---------------------------------------------------------------------------

_EMAIL_DRAFTS: Dict[str, Dict[str, Any]] = {}
_EMAIL_DRAFT_FILE = os.environ.get(
    "EMAIL_DRAFT_FILE", "/tmp/mojo_email_drafts.json"
)


def _draft_keys(ctx: dict) -> List[str]:
    chat = str((ctx or {}).get("chat_id") or "").strip()
    sid = str((ctx or {}).get("sender_id") or "").strip()
    snum = str((ctx or {}).get("sender_num") or "").strip()
    keys: List[str] = []
    if chat:
        keys.append(f"{chat}|owner")  # primary stable key for owner drafts
        if sid:
            keys.append(f"{chat}|{sid}")
        if snum and snum != sid:
            keys.append(f"{chat}|{snum}")
    return keys


def _draft_key(ctx: dict) -> str:
    keys = _draft_keys(ctx)
    return keys[0] if keys else "unknown"


def _load_drafts_from_disk() -> None:
    if _EMAIL_DRAFTS:
        return
    try:
        if os.path.isfile(_EMAIL_DRAFT_FILE):
            with open(_EMAIL_DRAFT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                _EMAIL_DRAFTS.update(data)
    except Exception as e:
        print(f"[email_draft] load failed: {e}")


def _save_drafts_to_disk() -> None:
    try:
        with open(_EMAIL_DRAFT_FILE, "w", encoding="utf-8") as f:
            json.dump(_EMAIL_DRAFTS, f)
    except Exception as e:
        print(f"[email_draft] save failed: {e}")


def _get_draft(ctx: dict) -> Optional[Dict[str, Any]]:
    _load_drafts_from_disk()
    for k in _draft_keys(ctx):
        d = _EMAIL_DRAFTS.get(k)
        if d:
            return d
    return None


def _put_draft(ctx: dict, draft: Dict[str, Any]) -> None:
    _load_drafts_from_disk()
    for k in _draft_keys(ctx):
        _EMAIL_DRAFTS[k] = draft
    _save_drafts_to_disk()


def _pop_draft(ctx: dict) -> Optional[Dict[str, Any]]:
    _load_drafts_from_disk()
    draft = _get_draft(ctx)
    for k in list(_EMAIL_DRAFTS.keys()):
        # drop every key that points at this same draft object / chat
        chat = str((ctx or {}).get("chat_id") or "")
        if chat and k.startswith(f"{chat}|"):
            _EMAIL_DRAFTS.pop(k, None)
    _save_drafts_to_disk()
    return draft


def _format_draft(d: Dict[str, Any]) -> str:
    """WhatsApp-native draft card — one field per line, no glued *markers*."""
    attach = d.get("attach_resume") or "ask"
    attach_line = {
        "true": "yes (resume PDF)",
        "false": "no",
        "ask": "undecided — reply yes/no",
    }.get(attach, attach)
    if d.get("reason") and attach == "ask":
        attach_line += f" ({d['reason']})"

    body = (d.get("body") or "").strip() or "(empty)"
    lines = [
        "*Email draft*",
        "",
        f"*To:* {d.get('to') or '(missing)'}",
    ]
    if d.get("cc"):
        lines.append(f"*Cc:* {d['cc']}")
    lines.extend(
        [
            f"*Subject:* {d.get('subject') or '(missing)'}",
            f"*Attach:* {attach_line}",
            "",
            "*Body:*",
            body,
            "",
            "_Edit karo, ya 'send it' / 'bhej do' likho._",
        ]
    )
    return "\n".join(lines)


def _owner_ok(ctx: dict) -> bool:
    if not _OWNER_SENDER_ID:
        return False
    owner = str(_OWNER_SENDER_ID).strip()
    sid = str((ctx or {}).get("sender_id") or "").strip()
    snum = str((ctx or {}).get("sender_num") or "").strip()
    # Also accept bare number match when LID/server suffix present
    candidates = {sid, snum}
    for c in list(candidates):
        if c and "@" in c:
            candidates.add(c.split("@")[0])
        if c and c.isdigit():
            candidates.add(c)
    return owner in candidates or any(
        c == owner or c.endswith(owner) or owner.endswith(c)
        for c in candidates
        if c
    )


# Meta instructions that mean "rewrite the body" — NOT part of the email itself.
# Matched anywhere in the user message (before or after the ':').
_EMAIL_REWRITE_PATTERNS = (
    r"make\s+it\s+(?:more\s+)?(?:professional|formal|casual|friendly|polite|short|concise|warm|firm|assertive)",
    r"rewrite(?:\s+it)?(?:\s+in\s+(?:a\s+)?[\w\s\-]+)?",
    r"rephrase(?:\s+it)?",
    r"polish(?:\s+it)?",
    r"improve(?:\s+it)?",
    r"formal\s+(?:bana|kar|tone)",
    r"professional\s+(?:bana|kar|tone|likho|likh)",
    r"(?:bana|kar)\s+(?:do\s+)?(?:professional|formal|casual|polite)",
    r"in\s+a\s+[\w\-]+\s+tone",
    r"tone\s*:\s*\w+",
    r"more\s+professional",
    r"more\s+formal",
    r"keep\s+it\s+(?:short|brief|casual|formal|professional)",
    r"word\s+it\s+(?:professionally|formally|politely|better)",
)


def extract_email_rewrite_instruction(text: str) -> str:
    """Return the rewrite instruction phrase if present, else ''."""
    if not text:
        return ""
    found: List[str] = []
    for pat in _EMAIL_REWRITE_PATTERNS:
        for m in re.finditer(pat, text, flags=re.I):
            found.append(m.group(0).strip())
    # de-dupe preserving order
    seen = set()
    out: List[str] = []
    for f in found:
        k = f.lower()
        if k not in seen:
            seen.add(k)
            out.append(f)
    return "; ".join(out)


def strip_email_rewrite_instruction(text: str) -> str:
    """Remove meta rewrite phrases so they never land in the email body."""
    if not text:
        return ""
    t = text
    for pat in _EMAIL_REWRITE_PATTERNS:
        t = re.sub(pat, " ", t, flags=re.I)
    # leftover glue: " , " / double spaces / trailing "and"
    t = re.sub(r"\s*,\s*,+", ",", t)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\s+,", ",", t)
    t = re.sub(r",\s*$", "", t.strip())
    t = re.sub(r"^(?:and|please|pls)\s+", "", t, flags=re.I)
    t = re.sub(r"\s+(?:and|please|pls)$", "", t, flags=re.I)
    return t.strip(" \t\n\r,;:-")


def parse_direct_email_command(text: str) -> Optional[Dict[str, str]]:
    """
    Parse outbound email commands.

    Default policy: body is WORD-FOR-WORD from the user (after stripping
    recipient / subject / rewrite-meta phrases).

    Returns dict with keys:
      to, subject, body, rewrite (instruction or ''), exact ('true'|'false')
    exact=true → force path should store body as-is.
    exact=false → agent/LLM may rewrite using 'rewrite' instruction.
    """
    if not text:
        return None
    # Strip bot @mention + quoted blocks for parsing (body still from remaining)
    t = re.sub(r"@\d[\d\s]*", " ", text)
    t = re.sub(r"\[Quoted Message\]:.*", " ", t, flags=re.I | re.S)
    t = t.strip()
    if not t:
        return None

    rewrite = extract_email_rewrite_instruction(t)

    # Broad recipient patterns (EN + Roman Urdu)
    #   send email to X: / email this to X / email X ko: / mail X:
    #   send Email lucie@x.com ko:
    m = re.search(
        r"(?:"
        r"email\s+this\s+to|email\s+to|send\s+(?:this\s+)?email\s+to|"
        r"send\s+email|email\s+kar(?:na|o|do)?|"
        r"mail\s+this\s+to|mail\s+to|mail\s+kar(?:na|o|do)?|"
        r"email|mail|send"
        r")"
        r"\s*"
        r"(?:to\s+)?"
        r"([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})"
        r"(?:\s*ko)?"
        r"\s*:?\s*",
        t,
        flags=re.I,
    )
    if not m:
        # Fallback: any email after a send/email/mail keyword
        if not re.search(r"\b(?:email|mail|send)\b", t, flags=re.I):
            return None
        em = re.search(
            r"([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})",
            t,
        )
        if not em:
            return None
        to = em.group(1).strip()
        # body = everything except the email address and command words
        rest = (t[: em.start()] + " " + t[em.end() :]).strip()
        rest = re.sub(
            r"(?i)\b(?:send\s+)?(?:this\s+)?(?:email|mail)(?:\s+this)?(?:\s+to)?\b",
            " ",
            rest,
        )
        rest = re.sub(r"(?i)\bko\b", " ", rest)
        rest = re.sub(r"\s{2,}", " ", rest).strip(" \t\n\r:")
    else:
        to = m.group(1).strip()
        rest = t[m.end() :].strip()
        # Also strip a leading "ko:" leftover
        rest = re.sub(r"(?i)^ko\s*:?\s*", "", rest).strip()

    subject = ""
    body = rest
    sm = re.match(
        r"(?is)^(?:subject|subj)\s*:\s*(.+?)(?:\n+|$)(.*)$",
        rest,
    )
    if sm:
        subject = sm.group(1).strip()
        body = (sm.group(2) or "").strip()

    # Strip rewrite meta from body (never ship "make it professional" to recipient)
    body = strip_email_rewrite_instruction(body)
    if subject:
        subject = strip_email_rewrite_instruction(subject)

    if not body:
        # Maybe the only content was a rewrite instruction — still need something
        body = rest.strip()
        body = strip_email_rewrite_instruction(body)
    if not body:
        return None

    if not subject:
        subject = "Quick note"

    exact = "false" if rewrite else "true"
    return {
        "to": to,
        "subject": subject,
        "body": body,
        "rewrite": rewrite,
        "exact": exact,
    }


def is_job_apply_intent(text: str) -> bool:
    """True when user wants to apply to a quoted/pasted JD."""
    if not text:
        return False
    head = re.split(r"\[quoted message\]:", text, maxsplit=1, flags=re.I)[0]
    head = re.sub(r"@\d[\d\s]*", " ", head)
    head = re.sub(r"https?://\S+", " ", head)
    low = head.lower()
    phrases = (
        "apply to this",
        "apply for this",
        "apply to the job",
        "apply for the job",
        "apply karo",
        "apply kar do",
        "apply kar dena",
        "is job pe apply",
        "is pe apply",
        "job pe apply",
        "send my cv",
        "send my resume",
        "send cv",
        "send resume",
        "cover letter bhej",
        "application bhej",
        "apply now",
    )
    if any(p in low for p in phrases):
        return True
    if re.search(r"(?<!\w)apply(?!\w)", low) and re.search(
        r"(?i)job|hiring|position|recruiter|vacancy|jd\b|role\b",
        text,
    ):
        return True
    return False


def parse_job_apply_command(text: str) -> Optional[Dict[str, str]]:
    """
    Extract apply target from a message that quotes/pastes a JD.

    Returns {to, subject, jd_excerpt, attach_resume} or None if not apply intent.
    Email is taken from the JD (e.g. hr@samagroup.com). Never ask the user for
    an address that is already present in the quoted text.
    """
    if not text or not is_job_apply_intent(text):
        return None

    quoted = ""
    qm = re.search(r"\[Quoted Message\]:\s*(.+)$", text, flags=re.I | re.S)
    if qm:
        quoted = qm.group(1)
    search_blob = quoted or text

    # WhatsApp JDs often use Mathematical Bold/Italic Unicode (𝗛𝗶𝗿𝗶𝗻𝗴 / 𝐑𝐨𝐥𝐞).
    # NFKC maps those to plain ASCII so Role:/Hiring: regexes work.
    try:
        import unicodedata as _ud

        search_blob = _ud.normalize("NFKC", search_blob)
        if quoted:
            quoted = _ud.normalize("NFKC", quoted)
    except Exception:
        pass

    try:
        import email_ops as eo

        emails = eo.extract_emails(search_blob) or eo.extract_emails(text)
    except Exception:
        emails = re.findall(
            r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
            search_blob,
        )

    cleaned: List[str] = []
    seen = set()
    for e in emails or []:
        el = e.lower().strip()
        if el in seen or el.endswith("@lid") or "whatsapp" in el:
            continue
        seen.add(el)
        cleaned.append(e.strip())

    to = cleaned[0] if cleaned else ""

    _TITLE_TOKEN = (
        r"developer|engineer|recruiter|executive|manager|designer|analyst|"
        r"intern|specialist|consultant|lead|sales|marketing|writer|officer|"
        r"associate|architect|scientist|coordinator"
    )

    def _clean_title(raw: str) -> str:
        s = (raw or "").strip()
        s = re.sub(r"[*_`]", "", s)
        if "|" in s:
            # "Hiring | CAD Application Engineer" → keep side that looks like a title
            parts = [p.strip() for p in s.split("|")]
            scored = sorted(
                parts,
                key=lambda p: (
                    0 if re.search(rf"(?i){_TITLE_TOKEN}", p) else 1,
                    len(p),
                ),
            )
            s = scored[0] if scored else s
        s = re.sub(
            r"(?i)^\s*(?:we'?re\s+hiring|hiring\s+now|hiring|position|role|title)\s*[:\-–]?\s*",
            "",
            s,
        ).strip()
        s = re.sub(r"\s{2,}", " ", s).strip(" -–|:")
        return s

    def _is_job_title(s: str) -> bool:
        if not s or len(s) < 3 or len(s) > 70:
            return False
        low = s.lower()
        # Reject skill/responsibility fragments
        if re.search(
            r"(?i)^(improve|develop|perform|provide|work|collaborate|generate|"
            r"maintain|build|create|handle|manage|understand|good|strong|basic|"
            r"experience|knowledge|understanding|ability|exposure)\b",
            low,
        ):
            return False
        if re.search(
            r"(?i)understanding of|knowledge of|experience (?:in|with)|"
            r"workflows through|drawings and|problem.?solving",
            low,
        ):
            return False
        return bool(re.search(rf"(?i){_TITLE_TOKEN}", s))

    subject = ""
    role = ""

    sm = re.search(r"(?im)^(?:email\s+)?subject\s*:\s*(.+)$", search_blob)
    if sm:
        subject = _clean_title(sm.group(1))

    # Explicit Role: / Position: / Title: (after NFKC this catches 𝗥𝗼𝗹𝗲:)
    if not role:
        pm = re.search(
            r"(?i)(?:position|role|title)\s*:\s*\*?([^\n*]+)",
            search_blob,
        )
        if pm:
            cand = _clean_title(pm.group(1))
            if _is_job_title(cand):
                role = cand

    # "We're Hiring: Remote B2B Sales Consultants"
    # "We're Hiring | *CAD Application Engineer*"
    if not role:
        hm = re.search(
            r"(?i)(?:we'?re\s+hiring|hiring\s+now|hiring|looking\s+for)\s*[:\-–|]\s*\*?([^*\n]+)",
            search_blob,
        )
        if hm:
            cand = _clean_title(hm.group(1))
            if _is_job_title(cand):
                role = cand

    # Multi-role hiring lists only (short title-like bullets, not skill lists)
    roles_list: List[str] = []
    for m in re.finditer(
        r"(?im)^\s*[-•*]\s*\*?([^*\n]{3,50}?)\*?\s*$",
        search_blob,
    ):
        cand = _clean_title(m.group(1))
        if _is_job_title(cand) and len(cand.split()) <= 5:
            roles_list.append(cand)
    # Deduplicate preserving order
    seen_r = set()
    roles_list = [
        r
        for r in roles_list
        if not (r.lower() in seen_r or seen_r.add(r.lower()))
    ]

    if not role and roles_list:
        ranked = sorted(
            roles_list,
            key=lambda r: (
                0
                if re.search(
                    r"(?i)wordpress|web|full.?stack|software|developer|engineer",
                    r,
                )
                else 1
            ),
        )
        role = ranked[0]
        if len(ranked) > 1:
            role = " / ".join(ranked[:3])

    # First line often is the title
    if not role:
        first = (search_blob.strip().split("\n", 1)[0] or "").strip()
        first = _clean_title(first)
        if _is_job_title(first):
            role = first

    company = ""
    cm = re.search(
        r"(?i)\b([A-Z][A-Za-z0-9&.\' ]{2,40}?)\s+(?:is\s+hiring|is\s+looking|"
        r"are\s+hiring|we're\s+hiring|we\s+are\s+hiring)",
        search_blob,
    )
    if cm:
        company = cm.group(1).strip().strip("*")

    if not subject:
        if role and " / " not in role:
            subject = f"Application - {role}"
        elif role:
            # Multi-role: subject uses best-fit (first segment)
            subject = f"Application - {role.split(' / ')[0].strip()}"
        elif company:
            subject = f"Application - {company}"
        else:
            subject = "Job Application"

    return {
        "to": to,
        "subject": subject,
        "role": role
        or re.sub(r"(?i)^\s*application\s*[–\-:]\s*", "", subject).strip(),
        "company": company,
        "roles_list": ",".join(roles_list),
        "jd_excerpt": (quoted or search_blob).strip()[:3500],
        "attach_resume": "true",
        "emails_found": ",".join(cleaned),
    }


def _edit_instruction_text(text: str) -> str:
    """
    Prefer quoted instruction when the live message is only a mention/empty.

    WhatsApp UX: user types the edit, then replies to it with @bot — the real
    instruction lives in [Quoted Message], not the mention line.
    """
    if not text:
        return ""
    quoted = ""
    qm = re.search(r"\[Quoted Message\]:\s*(.+)$", text, flags=re.I | re.S)
    if qm:
        quoted = qm.group(1).strip()
    live = re.split(r"\[Quoted Message\]:", text, maxsplit=1, flags=re.I)[0]
    live = re.sub(r"@\d[\d\s]*", " ", live)
    live = re.sub(r"\s+", " ", live).strip()
    if quoted and (not live or len(live) < 8):
        return quoted
    if live and quoted:
        low = live.lower()
        if not any(
            k in low
            for k in (
                "remove",
                "change",
                "subject",
                "rewrite",
                "personalize",
                "edit",
                "update",
                "keep",
            )
        ):
            return f"{live}\n{quoted}"
        return live
    return live or quoted


def parse_draft_field_edits(text: str) -> Optional[Dict[str, str]]:
    """
    Deterministic draft edits from instructions like:
      Change subject to Application - Full Stack Developer
      Remove (0-2 years) wherever it is
      Remove LinkedIn from subject and body, keep everything else

    Replacement values are SAME-LINE only so following paragraphs
    (personalize / format instructions) are never swallowed into body_replace.
    """
    if not text:
        return None
    t = _edit_instruction_text(text)
    t = re.sub(r"@\d[\d\s]*", " ", t)
    t = t.strip()
    if not t:
        return None

    out: Dict[str, str] = {}

    # Prefer quoted subject: Change subject to "Application - CAD …"
    sm = re.search(
        r"(?i)(?:change|set|update)\s+(?:the\s+)?subject\s+to\s+[:\-]?\s*"
        r'[""\']([^""\']+)[""\']',
        t,
    )
    if not sm:
        sm = re.search(
            r"(?i)(?:change|set|update)\s+(?:the\s+)?subject\s+to\s+[:\-]?\s*"
            r"([^\n]+)",
            t,
        )
    if sm:
        subj = sm.group(1).strip().strip("\"'").rstrip(".,;")
        # Stop before "and role…" / "and the body…"
        subj = re.split(
            r"(?i)\s+and\s+(?:(?:the\s+)?role\b|(?:the\s+)?(?:second\s+line|body|line)\b)",
            subj,
            maxsplit=1,
        )[0].strip()
        subj = subj.strip("\"'").rstrip(".,;")
        if subj and len(subj) < 120:
            out["subject"] = subj

    # Change role in email body to "CAD Application Engineer"
    role_m = re.search(
        r"(?i)(?:change\s+)?role\s+(?:in\s+(?:the\s+)?(?:email\s+)?body\s+)?"
        r"to\s+[:\-]?\s*[""\']?([^""\'\n]{3,80})[""\']?",
        t,
    )
    if role_m:
        new_role = role_m.group(1).strip().strip("\"'").rstrip(".,;")
        if new_role and " and " not in new_role.lower():
            out["body_role"] = new_role

    # "old" to "new" — quoted, never cross newlines
    rm = re.search(
        r"(?i)(?:change|replace|update)?\s*[\"']([^\"'\n]{2,80})[\"']\s+"
        r"(?:to|with|->)\s+[\"']?([^\"'\n]{1,80})[\"']?",
        t,
    )
    if rm:
        find_s = rm.group(1).strip()
        repl_s = rm.group(2).strip().rstrip(".,;")
        if find_s and repl_s and "\n" not in repl_s and len(repl_s) <= 80:
            if find_s.lower() not in (out.get("subject") or "").lower():
                out["body_find"] = find_s
                out["body_replace"] = repl_s

    # Unquoted surgical replace (phone / LID fixes):
    #   replace the 73109... number with +923161545875
    #   replace 73109738680505 with +923161545875, everything else as is
    if "body_find" not in out:
        rm2 = re.search(
            r"(?i)replace\s+(?:the\s+)?([+\d][\d.\s…]{3,30}?)\s*"
            r"(?:number\s+)?(?:with|to|->|=)\s*([+\d][\d\s\-()]{6,24})",
            t,
        )
        if not rm2:
            rm2 = re.search(
                r"(?i)replace\s+(?:the\s+)?(.{2,40}?)\s+"
                r"(?:with|to|->)\s+(.{1,40}?)(?:\s*,?\s*(?:everything|keep|no change)|$)",
                t,
            )
        if rm2:
            find_s = rm2.group(1).strip().strip("\"'")
            find_s = re.sub(r"(?i)\s*(?:number|instead)\s*$", "", find_s).strip()
            repl_s = rm2.group(2).strip().strip("\"'").rstrip(".,;")
            repl_s = re.sub(
                r"(?i)\s*(?:instead|everything|as is|no change).*$",
                "",
                repl_s,
            ).strip()
            if find_s and repl_s and len(find_s) <= 40 and len(repl_s) <= 40:
                out["body_find"] = find_s
                out["body_replace"] = repl_s

    # Collect ALL surgical remove phrases in one message
    remove_phrases: List[str] = []

    def _add_remove(p: str) -> None:
        p = (p or "").strip().strip("\"'").rstrip(".,;")
        p = re.sub(r"(?i)\s+from\s+(?:role|subject|body|title).*$", "", p).strip()
        p = re.sub(r"(?i)\s+wherever\s+it\s+is.*$", "", p).strip()
        if p and 1 < len(p) <= 80 and p.lower() not in {
            x.lower() for x in remove_phrases
        }:
            remove_phrases.append(p)

    # remove X from subject/body/role
    for m in re.finditer(
        r"(?i)remove\s+[\"']?([^\"'\n]{1,80}?)[\"']?\s+"
        r"from\s+(?:the\s+)?(?:role\s+in\s+)?(?:subject|body|title|role)",
        t,
    ):
        _add_remove(m.group(1))
    # remove X wherever it is / remove the "Y"
    for m in re.finditer(
        r"(?i)remove\s+(?:the\s+)?[\"']([^\"'\n]{1,80})[\"']",
        t,
    ):
        _add_remove(m.group(1))
    for m in re.finditer(
        r"(?i)remove\s+(\([^)\n]{1,40}\))\s*(?:wherever\s+it\s+is)?",
        t,
    ):
        _add_remove(m.group(1))
    for m in re.finditer(
        r"(?i)remove\s+([A-Za-z0-9][^,\n]{0,60}?)\s+wherever\s+it\s+is",
        t,
    ):
        _add_remove(m.group(1))
    # also remove the "..."
    for m in re.finditer(
        r"(?i)(?:also\s+)?remove\s+(?:the\s+)?[\"']([^\"'\n]{1,80})[\"']",
        t,
    ):
        _add_remove(m.group(1))

    if remove_phrases:
        # Primary single + joined list for multi-remove
        out["remove_phrase"] = remove_phrases[0]
        out["remove_phrases"] = "|||".join(remove_phrases)

    # "The position is only Business Development Executive"
    only_m = re.search(
        r"(?i)(?:position|role|title)\s+is\s+only\s+[:\-]?\s*"
        r"[\"']?([^\"'\n,]{3,80})[\"']?",
        t,
    )
    if only_m:
        only_role = only_m.group(1).strip().strip("\"'").rstrip(".,;")
        only_role = re.split(
            r"(?i)\s*,\s*remove\b|\s+remove\b|\s+and\s+remove\b",
            only_role,
            maxsplit=1,
        )[0].strip()
        if only_role:
            out["position_only"] = only_role
            if "subject" not in out:
                out["subject"] = f"Application - {only_role}"

    low = t.lower()
    # Preserve-as-is cues: do NOT trigger full rewrite
    preserve_cues = (
        "keep everything else",
        "keep the rest",
        "rest as is",
        "as is",
        "same body",
        "only change",
        "just remove",
        "sirf remove",
        "bas remove",
    )
    rewrite_cues = (
        "personalize",
        "personalized",
        "personalise",
        "high impact",
        "high-impact",
        "rewrite",
        "rephrase",
        "polish",
        "paragraph",
        "line break",
        "line breaks",
        "better format",
        "format the body",
        "more relevant",
        "relevantly",
        "improve",
    )
    if any(c in low for c in rewrite_cues) and not any(
        c in low for c in preserve_cues
    ):
        out["wants_rewrite"] = "true"
    if any(c in low for c in preserve_cues):
        out.pop("wants_rewrite", None)

    if not any(
        k in out
        for k in (
            "subject",
            "body_find",
            "body_role",
            "remove_phrase",
            "position_only",
            "wants_rewrite",
        )
    ):
        return None
    return out


def is_email_send_confirm(text: str) -> bool:
    head = re.split(r"\[quoted message\]:", (text or "").lower(), maxsplit=1)[0]
    head = re.sub(r"@\d[\d\s]*", " ", head)
    head = re.sub(r"https?://\S+", " ", head)
    head = re.sub(r"[^\w\s]", " ", head)
    head = re.sub(r"\s+", " ", head).strip()
    # Don't treat "send email to X" as confirm — that's a new draft
    if re.search(r"\b(?:email|mail)\s+to\b", head) or re.search(
        r"\bsend\s+email\b.+\b[a-z0-9._%+\-]+@[a-z0-9.\-]+", head
    ):
        return False
    phrases = (
        "send it",
        "send now",
        "bhej do",
        "bhej dena",
        "bhej de",
        "confirm send",
        "haan bhej",
        "han bhej",
        "yes send",
        "send the email",
        "send the draft",
        "go ahead send",
        "approve send",
    )
    if any(p in head for p in phrases):
        return True
    tokens = head.split()
    return tokens in (["send"], ["bhej"], ["confirm"])


def is_email_draft_edit_intent(text: str) -> bool:
    """User wants the stored draft body/subject rewritten (not a new apply)."""
    if not text:
        return False
    # Prefer quoted instruction when mention-only (user replies to their own edit note)
    instr = _edit_instruction_text(text)
    head = instr or re.split(r"\[quoted message\]:", text, maxsplit=1, flags=re.I)[0]
    head = re.sub(r"@\d[\d\s]*", " ", head)
    low = head.lower()
    # New apply / new email commands are not edits
    if is_job_apply_intent(text) or parse_direct_email_command(text):
        return False
    phrases = (
        "personalize",
        "personalized",
        "personalise",
        "high impact",
        "high-impact",
        "rewrite",
        "rephrase",
        "polish",
        "improve",
        "update draft",
        "edit draft",
        "edit the draft",
        "update the draft",
        "draft update",
        "banao",
        "bana do",
        "better banao",
        "zyada better",
        "more relevant",
        "relevantly",
        "relevant banao",
        "experience ke sath",
        "experience k sath",
        "experience ke saat",
        "experience k saat",
        "mera experience",
        "mere experience",
        "stronger",
        "make it better",
        "make it stronger",
        "subject change",
        "change subject",
        "body change",
        "change body",
        # surgical preserve edits
        "remove ",
        "replace ",
        "replace the",
        "keep everything else",
        "keep the rest",
        "as is",
        "no change",
        "everything else",
        "position is only",
        "role is only",
        "title is only",
        "sirf ",
        "bas remove",
        "from subject",
        "from body",
        "from role",
        "number with",
    )
    return any(p in low for p in phrases)


def is_email_draft_cancel_intent(text: str) -> bool:
    if not text:
        return False
    head = re.split(r"\[quoted message\]:", text, maxsplit=1, flags=re.I)[0]
    head = re.sub(r"@\d[\d\s]*", " ", head)
    low = head.lower()
    return any(
        p in low
        for p in (
            "cancel draft",
            "draft cancel",
            "cancel the draft",
            "discard draft",
            "delete draft",
            "draft hatao",
            "draft cancel karo",
        )
    )


def _tool_draft_email(args: dict, ctx: dict) -> str:
    if not _owner_ok(ctx):
        return "Permission denied: only the bot owner can draft/send emails."

    action = (args.get("action") or "create").strip().lower()
    existing = _get_draft(ctx)

    if action == "show":
        if not existing:
            return "No email draft for this chat yet. Create one with action=create."
        return _format_draft(existing)

    if action == "cancel":
        if existing:
            _pop_draft(ctx)
            return "Draft cancelled."
        return "No draft to cancel."

    if action == "update":
        if not existing:
            return "No draft to update. Use action=create first."
        if args.get("to"):
            existing["to"] = str(args["to"]).strip()
        if args.get("subject") is not None and str(args.get("subject")).strip() != "":
            existing["subject"] = str(args["subject"]).strip()
        if args.get("body") is not None and str(args.get("body")).strip() != "":
            new_body = str(args["body"]).strip()
            old_body = (existing.get("body") or "").strip()
            # Reject truncated/rewritten-short bodies (LLM tool-arg cutoffs)
            truncated = (
                new_body.endswith("…")
                or new_body.endswith("...")
                or new_body.endswith("…")
                or bool(re.search(r"(?i)\bclients?\s*…\s*$", new_body))
            )
            too_short = (
                old_body
                and len(new_body) < max(120, int(len(old_body) * 0.7))
                and not args.get("force_short")
            )
            if truncated or too_short:
                logging.getLogger("mojo.agent").info(
                    "EMAIL_UPDATE_BODY_REJECTED truncated=%s old_len=%s new_len=%s",
                    truncated,
                    len(old_body),
                    len(new_body),
                )
                return (
                    "Update rejected: body looks truncated or incomplete. "
                    "Do not shorten the letter. Re-send the FULL body, or use a "
                    "surgical edit (e.g. remove a phrase / change subject only)."
                )
            existing["body"] = new_body
        if args.get("cc") is not None:
            existing["cc"] = str(args.get("cc") or "").strip() or None
        if args.get("attach_resume") in ("true", "false", "ask"):
            existing["attach_resume"] = args["attach_resume"]
        if args.get("reason"):
            existing["reason"] = str(args["reason"]).strip()
        if args.get("purpose"):
            existing["purpose"] = str(args["purpose"]).strip()
        existing["updated_at"] = time.time()
        _put_draft(ctx, existing)
        return "Draft updated.\n\n" + _format_draft(existing)

    # create
    to = (args.get("to") or "").strip()
    subject = (args.get("subject") or "").strip()
    body = (args.get("body") or "").strip()
    attach = (args.get("attach_resume") or "ask").strip().lower()
    if attach not in ("true", "false", "ask"):
        attach = "ask"
    purpose = (args.get("purpose") or "").strip() or "general"
    reason = (args.get("reason") or "").strip()
    cc = (args.get("cc") or "").strip() or None

    if not to:
        return (
            "Missing recipient (to). Extract email from the JD / quoted message "
            "or ask the user for the address, then call draft_email again."
        )
    if "@" not in to:
        return f"Invalid recipient email: {to!r}"
    if not body:
        return "Missing body. Write a concise high-impact email body, then create the draft."
    if not subject:
        subject = "Introduction / Application"

    draft = {
        "to": to,
        "subject": subject,
        "body": body,
        "cc": cc,
        "attach_resume": attach,
        "purpose": purpose,
        "reason": reason,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    _put_draft(ctx, draft)
    logging.getLogger("mojo.agent").info(
        "EMAIL_DRAFT_SAVED to=%s subject=%r attach=%s", to, subject[:60], attach
    )
    extra = ""
    if attach == "ask":
        extra = (
            "\n\nAttachment undecided. Confirm: attach resume PDF? "
            "(yes / no). Reason: "
            + (reason or "not specified")
        )
    elif attach == "true":
        try:
            import email_ops as eo

            path, src = eo.resolve_resume_pdf()
            if not path:
                extra = f"\n\nattach_resume=true but resume file missing: {src}"
            else:
                extra = f"\n\nResume ready ({src})"
        except Exception as e:
            extra = f"\n\nResume check failed: {e}"

    return "Draft created.\n\n" + _format_draft(draft) + extra


def _tool_send_email(args: dict, ctx: dict) -> str:
    if not _owner_ok(ctx):
        return "Permission denied: only the bot owner can send emails."

    if not args.get("confirm"):
        return "Send aborted: confirm must be true. Show the draft and wait for user OK."

    draft = _get_draft(ctx)
    if not draft:
        return "No draft to send. Create one with draft_email first."

    attach_flag = draft.get("attach_resume") or "ask"
    override = (args.get("attach_resume_override") or "").strip().lower()
    if override in ("true", "false"):
        attach_flag = override

    if attach_flag == "ask":
        return (
            "Attachment still undecided (attach_resume=ask). "
            "Ask the user: attach resume PDF yes or no? Then call send_email "
            "with attach_resume_override=true|false and confirm=true."
        )

    to = (draft.get("to") or "").strip()
    subject = (draft.get("subject") or "").strip()
    body = (draft.get("body") or "").strip()
    if not to or not body:
        return "Draft incomplete (need to + body). Update with draft_email first."

    try:
        import email_ops as eo
    except Exception as e:
        return f"email_ops import failed: {e}"

    if not eo.gmail_configured():
        return (
            "Gmail not configured on this host. Set GMAIL_CLIENT_ID, "
            "GMAIL_CLIENT_SECRET, GMAIL_REFRESH_TOKEN, GMAIL_SENDER."
        )

    attach_path = None
    attach_src = None
    if attach_flag == "true":
        attach_path, attach_src = eo.resolve_resume_pdf()
        if not attach_path:
            return (
                f"Cannot send with attachment — resume not found: {attach_src}. "
                "Set RESUME_PDF_PATH or RESUME_ONEDRIVE_PATH, or resend without attach "
                "(attach_resume_override=false)."
            )

    try:
        result = eo.send_email(
            to=to,
            subject=subject,
            body=body,
            attach_path=attach_path,
            cc=draft.get("cc"),
        )
    except Exception as e:
        return f"Send failed: {e}"

    _pop_draft(ctx)
    att = "yes" if result.get("attached") else "no"
    logging.getLogger("mojo.agent").info(
        "EMAIL_SENT to=%s id=%s attached=%s", result.get("to"), result.get("id"), att
    )
    return (
        f"Sent to {result.get('to')}\n"
        f"Subject: {result.get('subject')}\n"
        f"Gmail id: {result.get('id')}\n"
        f"Attached resume: {att}"
        + (f" ({attach_src})" if attach_src and result.get("attached") else "")
    )


def _tool_search_knowledge(args: dict, ctx: dict) -> str:
    query = (args.get("query") or "").strip()
    top_k = min(int(args.get("top_k") or 5), 10)
    if not query:
        return "Empty query."

    chat_id = (ctx or {}).get("chat_id")
    notes: List[str] = []
    try:
        notes = _get_group_memory(chat_id) or [] if chat_id else []
    except Exception as e:
        notes = []
        print(f"[search_knowledge] group_memory error: {e}")

    # Prefer hybrid RAG when schema + embeddings are available
    try:
        import knowledge_rag as kr

        if kr.schema_ready():
            result = kr.search(
                query,
                top_k=top_k,
                chat_id=chat_id,
                include_chat_notes=True,
                group_notes=notes,
            )
            # If RAG returned real hits (not the init / empty messages), use them
            if result and not result.startswith("RAG knowledge base not initialised"):
                if not result.startswith("No relevant knowledge found"):
                    return result
                # Fall through to keyword FAQ if RAG miss — still useful for
                # un-ingested business_info.txt during first deploy.
    except Exception as e:
        print(f"[search_knowledge] RAG path failed: {e}")

    # Keyword fallback (original behaviour) — agency FAQ + chat notes
    q_low = query.lower()
    chunks: List[str] = []
    paras = [p.strip() for p in re.split(r"\n\s*\n", _BUSINESS_KNOWLEDGE) if p.strip()]
    scored = []
    tokens = set(re.findall(r"\w+", q_low))
    for p in paras:
        p_low = p.lower()
        score = sum(1 for t in tokens if t in p_low)
        if score > 0:
            scored.append((score, p[:800]))
    scored.sort(key=lambda x: -x[0])
    for _, p in scored[:top_k]:
        chunks.append(f"[Agency] {p}")

    for note in notes:
        if any(t in note.lower() for t in tokens) or not tokens:
            chunks.append(f"[Chat note] {note}")
            if len(chunks) >= top_k + 3:
                break

    if not chunks:
        return "No relevant knowledge found. Answer from general knowledge or use web_search."
    return "\n---\n".join(chunks[: top_k + 3])


def _format_search_lines(items: list, num: int) -> str:
    """Normalize heterogeneous search hits into title / snippet / url lines."""
    lines = []
    for item in items[:num]:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or item.get("name") or "").strip()
        snippet = (
            item.get("snippet")
            or item.get("body")
            or item.get("description")
            or ""
        ).strip()
        link = (
            item.get("link")
            or item.get("href")
            or item.get("url")
            or ""
        ).strip()
        if not (title or snippet or link):
            continue
        block = f"• {title}" if title else "•"
        if snippet:
            block += f"\n  {snippet[:280]}"
        if link.startswith("http"):
            block += f"\n  {link}"
        lines.append(block)
    return "\n\n".join(lines)


def _tool_web_search(args: dict, ctx: dict) -> str:
    """Multi-backend web search. Order:
    1) Serper (if SERPER_API_KEY) — Google-quality, paid free-tier
    2) ddgs library — multi-engine (bing/brave/ddg/…), no API key
    3) DuckDuckGo Instant Answer JSON — weak for company/LinkedIn queries
    4) html.duckduckgo.com / lite.duckduckgo.com — last-resort HTML

    Previous failure mode (logs 2026-08-31): no Serper key → Instant Answer
    empty for "iSeeWaves LinkedIn" → lite.duckduckgo.com ConnectTimeout from
    the host. Never fall through was also a bug when Serper raised.
    """
    query = (args.get("query") or "").strip()
    num = min(max(int(args.get("num_results") or 5), 1), 8)
    if not query:
        return "Empty query."

    errors: list = []

    # 1) Serper — only if configured; on failure CONTINUE (do not return)
    serper_key = (os.getenv("SERPER_API_KEY") or "").strip()
    if serper_key:
        try:
            r = requests.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
                json={"q": query, "num": num},
                timeout=12,
            )
            r.raise_for_status()
            data = r.json()
            lines = []
            for item in (data.get("organic") or [])[:num]:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                lines.append(f"• {title}\n  {snippet}\n  {link}".rstrip())
            if data.get("answerBox"):
                ab = data["answerBox"]
                lines.insert(
                    0,
                    f"Answer box: {ab.get('answer') or ab.get('snippet') or ''}",
                )
            if lines:
                return "\n\n".join(lines)
            errors.append("serper: empty organic")
        except Exception as e:
            errors.append(f"serper: {e}")
            print(f"[web_search serper] {e}")

    # 2) ddgs (pip install ddgs) — robust multi-backend, no key
    #    Verified: returns real linkedin.com URLs for company queries.
    try:
        from ddgs import DDGS  # type: ignore

        results = []
        weak_candidate = None
        # Prefer google — auto/bing often return unrelated hits for brand queries.
        # Fall through backends on empty or exception (rate limits / blocks).
        q_tokens = [t.lower() for t in query.split() if len(t) > 2]
        for backend in ("google", "bing,brave", "duckduckgo", "auto"):
            try:
                results = list(
                    DDGS().text(query, max_results=num, backend=backend)
                ) or []
            except TypeError:
                try:
                    with DDGS() as ddgs:
                        results = list(
                            ddgs.text(query, max_results=num, backend=backend)
                        ) or []
                except TypeError:
                    results = list(DDGS().text(query, max_results=num)) or []
            except Exception as be:
                print(f"[web_search ddgs backend={backend}] {be}")
                results = []
            formatted = _format_search_lines(results, num)
            if not formatted:
                continue
            low = formatted.lower()
            if not q_tokens or any(t in low for t in q_tokens):
                return formatted
            # Unrelated SERP noise — keep as last resort, try next backend
            weak_candidate = formatted
            errors.append(f"ddgs:{backend}: weak match")
        if weak_candidate:
            return weak_candidate
        errors.append("ddgs: empty results on all backends")
    except ImportError:
        errors.append("ddgs: not installed (pip install ddgs)")
    except Exception as e:
        errors.append(f"ddgs: {e}")
        print(f"[web_search ddgs] {e}")

    # 3) DuckDuckGo Instant Answer — often empty for company/LinkedIn queries
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=10,
            headers={"User-Agent": "MojoAgent/1.0"},
        )
        data = r.json()
        parts = []
        if data.get("AbstractText"):
            parts.append(
                f"Abstract: {data['AbstractText']}\nSource: {data.get('AbstractURL', '')}"
            )
        for topic in (data.get("RelatedTopics") or [])[:num]:
            if isinstance(topic, dict) and topic.get("Text"):
                url = ""
                if topic.get("FirstURL"):
                    url = f"\n  {topic['FirstURL']}"
                parts.append(f"• {topic['Text']}{url}")
            elif isinstance(topic, dict) and "Topics" in topic:
                for t in topic["Topics"][:2]:
                    if t.get("Text"):
                        parts.append(f"• {t['Text']}")
        if parts:
            return "\n".join(parts)
        errors.append("ddg-ia: empty")
    except Exception as e:
        errors.append(f"ddg-ia: {e}")
        print(f"[web_search DDG-IA] {e}")

    # 4) HTML endpoints last (often blocked / slow from datacenter hosts)
    from urllib.parse import parse_qs, urlparse, unquote

    def _unwrap_ddg_href(href: str) -> str:
        href = (href or "").strip()
        if "uddg=" in href:
            try:
                qs = parse_qs(urlparse(href).query)
                if qs.get("uddg"):
                    return unquote(qs["uddg"][0])
            except Exception:
                pass
        return href

    for endpoint, method in (
        ("https://html.duckduckgo.com/html/", "POST"),
        ("https://lite.duckduckgo.com/lite/", "POST"),
    ):
        try:
            if method == "POST":
                r = requests.post(
                    endpoint,
                    data={"q": query},
                    timeout=12,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/124.0.0.0 Safari/537.36"
                        ),
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                )
            else:
                r = requests.get(
                    endpoint,
                    params={"q": query},
                    timeout=12,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/124.0.0.0 Safari/537.36"
                        ),
                    },
                )
            r.raise_for_status()
            pairs = re.findall(
                r'<a[^>]+href="([^"]+)"[^>]*rel="nofollow"[^>]*>([^<]+)</a>',
                r.text,
                flags=re.I,
            )
            if not pairs:
                pairs = re.findall(
                    r'<a[^>]+rel="nofollow"[^>]+href="([^"]+)"[^>]*>([^<]+)</a>',
                    r.text,
                    flags=re.I,
                )
            if not pairs:
                pairs = re.findall(
                    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                    r.text,
                    flags=re.I | re.S,
                )
            snippets = re.findall(
                r'class="result(?:__|-)?snippet"[^>]*>([^<]+)', r.text, flags=re.I
            )
            lines = []
            for i, (href, title) in enumerate(pairs[:num]):
                title = re.sub(r"<[^>]+>", "", title).strip()
                sn = snippets[i] if i < len(snippets) else ""
                href = _unwrap_ddg_href(href)
                if not title and not href:
                    continue
                block = f"• {title}" if title else "•"
                if sn.strip():
                    block += f"\n  {sn.strip()}"
                if href.startswith("http"):
                    block += f"\n  {href}"
                lines.append(block)
            if lines:
                return "\n\n".join(lines)
            errors.append(f"{endpoint}: no parseable results")
        except Exception as e:
            errors.append(f"{endpoint}: {e}")
            print(f"[web_search html] {endpoint} {e}")

    err_summary = "; ".join(errors[:4]) if errors else "unknown"
    return (
        "Web search unavailable right now "
        f"({err_summary}). Tip: set SERPER_API_KEY or ensure `ddgs` is installed."
    )


def _github_user_from_url(url: str) -> Optional[str]:
    """Extract GitHub username from profile or repo URL."""
    m = re.search(
        r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38})(?:/|$|\?)",
        url,
        re.I,
    )
    if not m:
        return None
    user = m.group(1)
    # skip reserved path segments
    if user.lower() in ("settings", "topics", "explore", "marketplace", "orgs", "login", "features", "pricing", "about"):
        return None
    return user


def _github_repo_from_url(url: str) -> Optional[tuple]:
    m = re.search(
        r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)(?:/|$|\?)",
        url,
        re.I,
    )
    if not m:
        return None
    return m.group(1), m.group(2)


def _fetch_github_user_repos(username: str, limit: int = 12) -> str:
    """Public GitHub API — real repo list, sorted by recently pushed."""
    api = f"https://api.github.com/users/{username}/repos"
    try:
        r = requests.get(
            api,
            params={"sort": "pushed", "direction": "desc", "per_page": min(limit, 30)},
            timeout=15,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "MojoAgent/1.0",
            },
        )
        if r.status_code == 404:
            return f"GitHub user '{username}' not found."
        r.raise_for_status()
        repos = r.json()
        if not isinstance(repos, list) or not repos:
            return f"GitHub user '{username}' has no public repositories."
        lines = [f"GitHub @{username} — {len(repos)} public repos shown (sorted by last push):"]
        for repo in repos[:limit]:
            name = repo.get("full_name") or repo.get("name")
            desc = (repo.get("description") or "").strip() or "(no description)"
            stars = repo.get("stargazers_count", 0)
            lang = repo.get("language") or "?"
            pushed = (repo.get("pushed_at") or "")[:10]
            fork = " [fork]" if repo.get("fork") else ""
            lines.append(
                f"• {name}{fork}\n"
                f"  {desc}\n"
                f"  ★{stars} · {lang} · last push {pushed}\n"
                f"  {repo.get('html_url', '')}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"GitHub API error for {username}: {e}"


def _fetch_github_repo(owner: str, repo: str) -> str:
    api = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        r = requests.get(
            api,
            timeout=15,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "MojoAgent/1.0",
            },
        )
        if r.status_code == 404:
            return f"Repo {owner}/{repo} not found (or private)."
        r.raise_for_status()
        data = r.json()
        return (
            f"Repo: {data.get('full_name')}\n"
            f"Description: {data.get('description') or '(none)'}\n"
            f"Stars: {data.get('stargazers_count', 0)} · Forks: {data.get('forks_count', 0)}\n"
            f"Language: {data.get('language') or '?'}\n"
            f"Default branch: {data.get('default_branch')}\n"
            f"Created: {(data.get('created_at') or '')[:10]} · "
            f"Last push: {(data.get('pushed_at') or '')[:10]}\n"
            f"URL: {data.get('html_url')}\n"
            f"Homepage: {data.get('homepage') or '-'}"
        )
    except Exception as e:
        return f"GitHub repo API error: {e}"


def _tool_browse_url(args: dict, ctx: dict) -> str:
    url = (args.get("url") or "").strip()
    # Keep page extracts tight — long HTML dumps caused 413 payloads and long agent essays
    max_chars = min(int(args.get("max_chars") or 2500), 4000)
    if not url:
        return "Empty URL."
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # --- GitHub special path (reliable JSON API, no JS-render needed) ---
    if "github.com" in url.lower():
        repo_pair = _github_repo_from_url(url)
        if repo_pair and repo_pair[1].lower() not in ("", "repositories", "stars", "followers", "following"):
            # full repo URL
            return _fetch_github_repo(repo_pair[0], repo_pair[1])
        user = _github_user_from_url(url)
        if user:
            return _fetch_github_user_repos(user, limit=12)
        # fall through to HTML if path is unusual

    # --- Generic page fetch ---
    try:
        r = requests.get(
            url,
            timeout=18,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            allow_redirects=True,
        )
        r.raise_for_status()
        ctype = (r.headers.get("content-type") or "").lower()
        if "application/json" in ctype:
            try:
                return json.dumps(r.json(), indent=2)[:max_chars]
            except Exception:
                return r.text[:max_chars]

        text = r.text
        # Prefer meta description + title if present
        title = ""
        tm = re.search(r"(?is)<title[^>]*>(.*?)</title>", text)
        if tm:
            title = re.sub(r"\s+", " ", tm.group(1)).strip()
        desc = ""
        dm = re.search(
            r'(?is)<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
            text,
        ) or re.search(
            r'(?is)<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']description["\']',
            text,
        )
        if dm:
            desc = dm.group(1).strip()

        text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", text)
        text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
        text = re.sub(r"(?is)<noscript[^>]*>.*?</noscript>", " ", text)
        text = re.sub(r"(?is)<nav[^>]*>.*?</nav>", " ", text)
        text = re.sub(r"(?is)<footer[^>]*>.*?</footer>", " ", text)
        text = re.sub(r"(?is)<[^>]+>", " ", text)
        text = re.sub(r"&nbsp;|&amp;|&lt;|&gt;|&quot;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        parts = []
        if title:
            parts.append(f"Title: {title}")
        if desc:
            parts.append(f"Description: {desc}")
        parts.append(text[:max_chars])
        out = "\n".join(parts)
        return out[: max_chars + 200] + ("…" if len(out) > max_chars + 200 else "")
    except Exception as e:
        return f"Failed to fetch URL ({url}): {e}"


def _tool_get_weather(args: dict, ctx: dict) -> str:
    city = (args.get("city") or "").strip()
    if not city:
        return "City name required."

    lat = lon = None
    place = city
    country = ""

    # Geocode with Open-Meteo (free, accurate city coordinates)
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        geo.raise_for_status()
        results = (geo.json() or {}).get("results") or []
        if results:
            lat = results[0]["latitude"]
            lon = results[0]["longitude"]
            place = results[0].get("name", city)
            country = results[0].get("country", "") or ""
    except Exception as e:
        print(f"[weather geo] {e}")

    # Prefer wttr.in at exact coordinates (avoids bad name resolution e.g. Islamabad→wrong village)
    try:
        path = f"{lat},{lon}" if lat is not None else requests.utils.quote(city)
        r = requests.get(
            f"https://wttr.in/{path}?format=j1",
            timeout=12,
            headers={"User-Agent": "MojoAgent/1.0"},
        )
        r.raise_for_status()
        d = r.json()
        cur = d["current_condition"][0]
        if not country:
            area = (d.get("nearest_area") or [{}])[0]
            place = place or area.get("areaName", [{}])[0].get("value") or city
            country = area.get("country", [{}])[0].get("value") or ""
        return (
            f"Weather — {place}, {country}\n"
            f"Temperature: {cur.get('temp_C')}°C (feels like {cur.get('FeelsLikeC')}°C)\n"
            f"Condition: {(cur.get('weatherDesc') or [{}])[0].get('value') or ''}\n"
            f"Humidity: {cur.get('humidity')}% · Wind: {cur.get('windspeedKmph')} km/h\n"
            f"Source: wttr.in (live)"
        )
    except Exception as e1:
        print(f"[weather wttr] {e1}")

    # Fallback: Open-Meteo forecast
    if lat is not None:
        try:
            wx = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
                    "timezone": "auto",
                },
                timeout=12,
            )
            wx.raise_for_status()
            cur = (wx.json() or {}).get("current") or {}
            return (
                f"Weather — {place}, {country}\n"
                f"Temperature: {cur.get('temperature_2m')}°C "
                f"(feels like {cur.get('apparent_temperature')}°C)\n"
                f"Humidity: {cur.get('relative_humidity_2m')}% · "
                f"Wind: {cur.get('wind_speed_10m')} km/h\n"
                f"Source: Open-Meteo (live)"
            )
        except Exception as e2:
            return f"Weather lookup failed for {city}: {e2}"

    return f"Could not resolve weather for '{city}'."


def _tool_get_memory(args: dict, ctx: dict) -> str:
    chat_id = args.get("chat_id") or ctx["chat_id"]
    notes = _get_group_memory(chat_id) or []
    if not notes:
        return "No permanent notes stored for this chat yet."
    return "Permanent notes:\n" + "\n".join(f"- {n}" for n in notes)


def _tool_save_memory(args: dict, ctx: dict) -> str:
    note = (args.get("note") or "").strip()
    if not note:
        return "Empty note — nothing saved."
    # Reject pure command stubs the model used to emit ("Remember @7310…")
    # when the real body lived in the quoted message. Require substance.
    _stub = re.sub(r"@\d[\d\s]*", " ", note, flags=re.I)
    _stub = re.sub(
        r"(?i)\b(always\s+)?remember( this| that)?\b|\byaad\s+rakh\w*\b",
        " ",
        _stub,
    )
    _stub = re.sub(r"\s+", " ", _stub).strip(" \t\n\r-–—:|")
    if len(_stub) < 12 and len(note) < 40:
        return (
            "Note too thin (looks like a command stub). "
            "Pass the FULL quoted body / resume / fact as the note."
        )
    sender_id = args.get("sender_id") or ctx.get("sender_id") or "unknown"
    try:
        _supabase.table("group_memory").insert(
            {
                "chat_id": ctx["chat_id"],
                "sender_id": sender_id,
                "note": note[:12000],
            }
        ).execute()
        preview = note[:120] + ("…" if len(note) > 120 else "")
        return f"Saved permanent note ({len(note)} chars): {preview}"
    except Exception as e:
        return f"Failed to save note: {e}"


def _tool_set_reminder(args: dict, ctx: dict) -> str:
    text = (args.get("text") or "").strip()
    if not text:
        return "No reminder text provided."
    # Re-use the existing robust AI parser + DB insert path
    try:
        return _handle_reminder_request(
            ctx["chat_id"], ctx["sender_id"], text, ctx.get("msg_time") or time.time()
        )
    except Exception as e:
        return f"Could not set reminder: {e}"


def _tool_list_reminders(args: dict, ctx: dict) -> str:
    try:
        return _list_reminders(ctx["chat_id"], ctx["sender_id"])
    except Exception as e:
        return f"Could not list reminders: {e}"


def _tool_cancel_reminders(args: dict, ctx: dict) -> str:
    """Smarter cancel: supports 'all', keyword match, and fuzzy subject match."""
    query = (args.get("query") or "").strip().lower()
    chat_id = ctx["chat_id"]
    sender_id = ctx["sender_id"]
    try:
        response = (
            _supabase.table("reminders")
            .select("*")
            .eq("chat_id", chat_id)
            .eq("sender_id", sender_id)
            .eq("active", True)
            .execute()
        )
        rows = response.data or []
        if not rows:
            return "You don't have any active reminders to cancel."

        if query in ("all", "sab", "saare", "saari", "everything", "pure"):
            for r in rows:
                _supabase.table("reminders").update({"active": False}).eq("id", r["id"]).execute()
            return "Cancelled all your active reminders (" + ", ".join(f'"{r["message"]}"' for r in rows) + ")."

        # Token overlap scoring
        q_tokens = set(re.findall(r"\w+", query))
        matched = []
        for r in rows:
            msg = (r.get("message") or "").lower()
            if query in msg or msg in query:
                matched.append(r)
                continue
            msg_tokens = set(re.findall(r"\w+", msg))
            if q_tokens & msg_tokens:
                matched.append(r)

        if not matched:
            names = ", ".join(f'"{r["message"]}"' for r in rows)
            return f"No reminder matched '{query}'. Active ones: {names}"

        for r in matched:
            _supabase.table("reminders").update({"active": False}).eq("id", r["id"]).execute()
        return "Cancelled: " + ", ".join(f'"{r["message"]}"' for r in matched)
    except Exception as e:
        return f"Could not cancel: {e}"


def _tool_query_contacts(args: dict, ctx: dict) -> str:
    q = (args.get("query") or "").strip().lower()
    if not q:
        return "Empty query."
    try:
        contacts_map, reverse_map = _get_contacts_maps()
        hits = []
        for lid, name in (contacts_map or {}).items():
            if q in (name or "").lower() or q in str(lid).lower():
                hits.append(f"{name or '?'} (id={lid})")
            if len(hits) >= 15:
                break
        if not hits:
            # also try reverse (name -> lid)
            for name, lid in (reverse_map or {}).items():
                if q in name.lower():
                    hits.append(f"{name} (id={lid})")
                if len(hits) >= 15:
                    break
        return "Matches:\n" + "\n".join(hits) if hits else "No contacts matched."
    except Exception as e:
        return f"Contact lookup failed: {e}"


def _tool_lookup_user(args: dict, ctx: dict) -> str:
    sid = (args.get("sender_id") or "").strip()
    if not sid:
        return "Need sender_id."
    try:
        contacts_map, _ = _get_contacts_maps()
        name = contacts_map.get(sid, "?")
        tz = _get_user_timezone(sid)
        # try fetch number from contacts table
        num = None
        try:
            res = (
                _supabase.table("contacts")
                .select("sender_num, push_name, timezone")
                .eq("sender_id", sid)
                .limit(1)
                .execute()
            )
            if res.data:
                num = res.data[0].get("sender_num")
                name = res.data[0].get("push_name") or name
                tz = res.data[0].get("timezone") or tz
        except Exception:
            pass
        return f"id={sid}\nname={name}\nnumber={num or 'unknown'}\ntimezone={tz or 'unknown'}"
    except Exception as e:
        return f"Lookup failed: {e}"


def _tool_send_message_to(args: dict, ctx: dict) -> str:
    # Fail CLOSED: if OWNER_SENDER_ID isn't configured, nobody is the owner —
    # never treat an unset owner as "check disabled". Allow either sender_id
    # (LID) or sender_num (phone) to match, since callers may pass either.
    if not _OWNER_SENDER_ID or (
        str(ctx.get("sender_id")) != str(_OWNER_SENDER_ID)
        and str(ctx.get("sender_num") or "") != str(_OWNER_SENDER_ID)
    ):
        return "Permission denied: only the bot owner can send proactive messages."
    chat_id = (args.get("chat_id") or "").strip()
    text = (args.get("text") or "").strip()
    if not chat_id or not text:
        return "chat_id and text required."
    try:
        _send_proactive_message(chat_id, text)
        return f"Message sent to {chat_id}."
    except Exception as e:
        return f"Send failed: {e}"


def _tool_file_list_onedrive(args: dict, ctx: dict) -> str:
    if not _file_ops or not getattr(_file_ops, "onedrive_configured", lambda: False)():
        return "OneDrive is not configured."
    folder = (args.get("folder") or args.get("path") or "").strip().strip("/")
    try:
        import knowledge_rag as kr

        root = kr.knowledge_root()
        if folder:
            target = folder if folder.startswith(root) else f"{root}/{folder}".rstrip("/")
        else:
            target = root
        files = _file_ops.list_onedrive_recursive(target, max_items=80)
        if not files:
            children = _file_ops.list_onedrive_folder(target)
            if not children:
                return (
                    f"OneDrive folder empty or missing: {target}\n"
                    "Owner can run /ingest ensure then drop files under "
                    f"{root}/agency or {root}/docs."
                )
            lines = [f"Folder: {target} ({len(children)} items)"]
            for it in children[:40]:
                kind = "dir" if "folder" in it else "file"
                lines.append(f"- [{kind}] {it.get('name')}")
            return "\n".join(lines)
        lines = [f"Knowledge files under {target} ({len(files)}):"]
        for f in files[:50]:
            size = f.get("size") or 0
            lines.append(f"- {f.get('path')} ({size} B)")
        if len(files) > 50:
            lines.append(f"… +{len(files) - 50} more")
        return "\n".join(lines)
    except Exception as e:
        return f"OneDrive list failed: {e}"


def _tool_python_exec(args: dict, ctx: dict) -> str:
    if not _OWNER_SENDER_ID or (
        str(ctx.get("sender_id")) != str(_OWNER_SENDER_ID)
        and str(ctx.get("sender_num") or "") != str(_OWNER_SENDER_ID)
    ):
        return "Permission denied: python_exec is owner-only."
    code = (args.get("code") or "").strip()
    if not code or len(code) > 2000:
        return "Code empty or too long."
    # Extremely restricted sandbox
    banned = (
        "import ", "open(", "exec(", "eval(", "__", "os.", "sys.", "subprocess",
        "requests", "socket", "file", "input(", "breakpoint",
    )
    low = code.lower()
    for b in banned:
        if b in low:
            return f"Blocked for safety (contains '{b.strip()}')."
    try:
        # Only allow expression / simple assignment via restricted globals
        safe_builtins = {
            "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
            "range": range, "round": round, "sorted": sorted, "list": list,
            "dict": dict, "str": str, "int": int, "float": float, "bool": bool,
            "print": print,
        }
        local: dict = {}
        exec(code, {"__builtins__": safe_builtins}, local)
        # return last non-private value
        vals = {k: v for k, v in local.items() if not k.startswith("_")}
        if vals:
            return "Result: " + ", ".join(f"{k}={v!r}" for k, v in vals.items())
        return "Executed successfully (no return value)."
    except Exception as e:
        return f"Error: {e}"

def _tool_image_describe(args: dict, ctx: dict) -> str:
    """Describe path is media-native; tool is a schema hook + honest fallback."""
    focus = (args.get("focus") or "all").strip().lower()
    return (
        "image_describe runs when the user sends or quotes an image on the media path "
        f"(focus={focus}). No image bytes in this text-only tool call — ask them to "
        "send/quote the photo, or use note_down only if they want it saved to OneDrive."
    )


def _tool_note_down(args: dict, ctx: dict) -> str:
    # 1. Strict Owner Check (sender_id LID or sender_num phone)
    if not _OWNER_SENDER_ID or (
        str(ctx.get("sender_id") or "") != str(_OWNER_SENDER_ID)
        and str(ctx.get("sender_num") or "") != str(_OWNER_SENDER_ID)
    ):
        return "Permission denied: Only the bot owner can use the note down feature."

    content = (args.get("content") or "").strip()
    if not content:
        return "Empty note. Nothing was saved."

    # Strip model ellipsis truncation markers if the tail is clearly cut
    if content.endswith("…") or content.endswith("..."):
        print("[note_down] warning: content ends with ellipsis — model may have truncated")

    try:
        now_str = datetime.now(timezone.utc).strftime("%A, %Y-%m-%d %I:%M %p UTC")
        formatted_entry = f"{now_str}\n{content}\n\n\n"

        remote_folder = "MojoAgent"
        file_name = "Mojo_Notes.txt"
        remote_path = f"{remote_folder}/{file_name}"
        local_temp = os.path.join(tempfile.gettempdir(), file_name)

        existing_content = ""
        try:
            if _file_ops and _file_ops.onedrive_configured():
                dl_path = _file_ops.download_from_onedrive(remote_path, local_temp)
                with open(dl_path, "r", encoding="utf-8") as f:
                    existing_content = f.read()
        except Exception:
            pass

        new_content = existing_content + formatted_entry
        _file_ops.write_text_file(new_content, file_name, tempfile.gettempdir())
        _file_ops.upload_to_onedrive(
            local_temp, remote_folder=remote_folder, remote_name=file_name
        )

        nchars = len(content)
        nlines = content.count("\n") + 1
        print(f"[note_down] saved chars={nchars} lines={nlines}")
        return (
            f"Successfully saved to OneDrive folder MojoAgent "
            f"({nchars} chars, {nlines} lines)."
        )
    except Exception as e:
        return f"Failed to save note: {e}"


# ---------------------------------------------------------------------------
# Video transcription (YouTube captions → yt-dlp + Groq Whisper fallback)
# ---------------------------------------------------------------------------

_YT_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|embed/|shorts/|live/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)
# Soft in-process cache: url+lang+ts → (text, expiry_ts). Keeps repeated asks cheap.
_TRANSCRIPT_CACHE: Dict[str, tuple] = {}
_TRANSCRIPT_CACHE_TTL = 3600  # 1 hour
_TRANSCRIPT_CACHE_MAX = 24
# Raw speech text by video_id — shared across modes (summary + transcript)
# so a successful caption/ASR fetch is reused instead of falling back to description.
_RAW_SPEECH_CACHE: Dict[str, tuple] = {}  # id -> (text, exp)
_RAW_SPEECH_TTL = 3600
# Safety caps for cloud (Render) — avoid huge downloads / Groq 25 MB limit
_MAX_ASR_SECONDS = 45 * 60  # 45 min hard cap
_TARGET_AUDIO_BITRATE = "48k"  # mono speech is fine at 48 kbps
_GROQ_MAX_UPLOAD_BYTES = 24 * 1024 * 1024


def _extract_youtube_id(url: str) -> Optional[str]:
    m = _YT_ID_RE.search(url or "")
    return m.group(1) if m else None


def _cache_key(url: str, language: str, timestamps: bool) -> str:
    # v3: bust entries that cached refine-hallucinated YouTube description
    return f"v3|{url.strip().lower()}|{language or 'auto'}|{int(bool(timestamps))}"


def _cache_get(key: str) -> Optional[str]:
    entry = _TRANSCRIPT_CACHE.get(key)
    if not entry:
        return None
    text, exp = entry
    if time.time() > exp:
        _TRANSCRIPT_CACHE.pop(key, None)
        return None
    return text


def _cache_set(key: str, text: str) -> None:
    # Never persist description/promo blobs
    if text and _looks_like_youtube_description(text):
        print(
            f"[transcribe_video] skip cache_set — description-like "
            f"(chars={len(text)})"
        )
        return
    if len(_TRANSCRIPT_CACHE) >= _TRANSCRIPT_CACHE_MAX:
        # Drop oldest by expiry
        oldest = min(_TRANSCRIPT_CACHE.items(), key=lambda kv: kv[1][1])
        _TRANSCRIPT_CACHE.pop(oldest[0], None)
    _TRANSCRIPT_CACHE[key] = (text, time.time() + _TRANSCRIPT_CACHE_TTL)


def _format_segments(segments: List[dict], with_timestamps: bool) -> str:
    """segments: list of {text, start?, duration?}"""
    lines = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if with_timestamps and seg.get("start") is not None:
            s = float(seg["start"])
            mm, ss = divmod(int(s), 60)
            hh, mm = divmod(mm, 60)
            if hh:
                ts = f"[{hh:02d}:{mm:02d}:{ss:02d}]"
            else:
                ts = f"[{mm:02d}:{ss:02d}]"
            lines.append(f"{ts} {text}")
        else:
            lines.append(text)
    if with_timestamps:
        return "\n".join(lines)
    # collapse whitespace for plain transcript
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


# Public Piped API instances — free YouTube frontend that often works from cloud IPs
_PIPED_API_HOSTS = [
    "https://api.piped.private.coffee",
    "https://pipedapi.adminforge.de",
    "https://pipedapi.nosebs.ru",
    "https://pipedapi.leptons.xyz",
    "https://api.piped.yt",
]


def _parse_caption_time(s: str) -> float:
    """Parse '12.5', '12.5s', '00:01:02.500', '01:02.5' → seconds."""
    if not s:
        return 0.0
    s = s.strip().rstrip("sS")
    try:
        if ":" not in s:
            return float(s)
        parts = [float(p) for p in s.split(":")]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return float(parts[-1])
    except Exception:
        return 0.0


def _unescape_caption_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    return (
        text.replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&quot;", '"')
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("\xa0", " ")
    )


def _parse_ttml_or_vtt(raw: str) -> List[dict]:
    """Parse TTML/XML or WebVTT into {text, start, duration} segments."""
    segs: List[dict] = []
    if not raw or not raw.strip():
        return segs

    # TTML / XML timed text (Piped default)
    if "<" in raw and (
        "tt " in raw[:300].lower()
        or "transcript" in raw[:300].lower()
        or "<text" in raw[:800].lower()
        or "<p " in raw[:800].lower()
    ):
        for m in re.finditer(
            r'<(?:p|text)[^>]*\bbegin=["\']?([\d:.]+)["\']?[^>]*(?:\bend=["\']?([\d:.]+)["\']?)?[^>]*>(.*?)</(?:p|text)>',
            raw,
            flags=re.I | re.S,
        ):
            begin_s, end_s, body = m.group(1), m.group(2), m.group(3)
            text = re.sub(r"\s+", " ", _unescape_caption_text(body)).strip()
            if not text:
                continue
            start = _parse_caption_time(begin_s)
            end = _parse_caption_time(end_s) if end_s else start + 2.0
            segs.append({"text": text, "start": start, "duration": max(0.0, end - start)})
        if segs:
            return segs
        # srv3 style
        for m in re.finditer(
            r'<text[^>]*\bstart=["\']?([\d.]+)["\']?[^>]*(?:\bdur=["\']?([\d.]+)["\']?)?[^>]*>(.*?)</text>',
            raw,
            flags=re.I | re.S,
        ):
            start = float(m.group(1))
            dur = float(m.group(2)) if m.group(2) else 2.0
            text = re.sub(r"\s+", " ", _unescape_caption_text(m.group(3))).strip()
            if text:
                segs.append({"text": text, "start": start, "duration": dur})
        if segs:
            return segs

    # WebVTT
    if "WEBVTT" in raw[:30] or "-->" in raw:
        blocks = re.split(r"\n\s*\n", raw)
        for block in blocks:
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            if not lines:
                continue
            tline = None
            text_lines: List[str] = []
            for ln in lines:
                if "-->" in ln:
                    tline = ln
                elif tline is not None and not re.match(r"^\d+$", ln):
                    text_lines.append(_unescape_caption_text(ln))
            if not tline or not text_lines:
                continue
            parts = [p.strip() for p in tline.split("-->")]
            if len(parts) < 2:
                continue
            start = _parse_caption_time(parts[0].split()[0])
            end = _parse_caption_time(parts[1].split()[0])
            text = re.sub(r"\s+", " ", " ".join(text_lines)).strip()
            if text:
                segs.append({"text": text, "start": start, "duration": max(0.0, end - start)})
    return segs


# Shared free demo-key cache for youtube2text.org
_Y2T_KEY: Optional[str] = None
_Y2T_KEY_TS: float = 0.0


def _youtube2text_api_key() -> Optional[str]:
    """Prefer env YOUTUBE2TEXT_API_KEY; else shared free demo key."""
    global _Y2T_KEY, _Y2T_KEY_TS
    env_key = (os.getenv("YOUTUBE2TEXT_API_KEY") or "").strip()
    if env_key:
        return env_key
    if _Y2T_KEY and (time.time() - _Y2T_KEY_TS) < 6 * 3600:
        return _Y2T_KEY
    try:
        r = requests.get(
            "https://youtube2text.org/api/demo-key",
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MojoBot/1.0)"},
        )
        if r.status_code == 200:
            data = r.json() or {}
            key = data.get("apiKey") or data.get("api_key")
            if key:
                _Y2T_KEY = str(key)
                _Y2T_KEY_TS = time.time()
                return _Y2T_KEY
    except Exception as e:
        print(f"[transcribe_video] youtube2text demo-key error: {e}")
    return _Y2T_KEY


def _try_youtube2text(
    url: str, language: str, with_timestamps: bool
) -> Optional[str]:
    """
    Fully free, cloud-working YouTube transcripts via youtube2text.org.
    """
    key = _youtube2text_api_key()
    if not key:
        return None

    lang = (language or "auto").strip().lower()
    # Explicitly attempt Hindi/Urdu caption tracks if language is set to auto
    langs_to_try = [lang] if lang not in ("auto", "") else ["hi", "ur", "en", "auto"]

    headers = {
        "x-api-key": key,
        "User-Agent": "Mozilla/5.0 (compatible; MojoBot/1.0)",
        "Accept": "application/json",
    }

    for target_lang in langs_to_try:
        params: Dict[str, Any] = {"url": url, "maxChars": "100000"}
        if target_lang != "auto":
            params["lang"] = target_lang

        try:
            r = requests.get(
                "https://youtube2text.org/api/transcribe",
                params=params,
                headers=headers,
                timeout=90,
            )
            if r.status_code != 200:
                continue

            data = r.json() or {}
            result = data.get("result") if isinstance(data.get("result"), dict) else data
            content = (
                (result or {}).get("content")
                or (result or {}).get("transcript")
                or (result or {}).get("text")
                or data.get("content")
            )

            if not content or not str(content).strip():
                continue

            text = str(content).strip()
            if text.startswith("[] "):
                text = text[3:].strip()

            text = _clean_raw_transcript(text)

            # Reject candidate if it is just video metadata/description
            if _looks_like_youtube_description(text):
                print(f"[transcribe_video] youtube2text returned description for lang={target_lang}, trying next")
                continue

            title = (result or {}).get("title") or ""
            print(f"[transcribe_video] youtube2text OK (lang={target_lang}) chars={len(text)}")
            return f"Title: {title}\n\n{text}" if title else text

        except Exception as e:
            print(f"[transcribe_video] youtube2text error for lang={target_lang}: {e}")
            continue

    return None


def _clean_raw_transcript(text: str) -> str:
    """
    Strip ASR/music noise so the model can refine more easily.
    Removes [संगीत], [music], bracketed sound effects, collapses gaps.
    """
    if not text:
        return text
    # Common auto-caption noise markers (Hindi/English/brackets)
    text = re.sub(
        r"\[\s*(?:संगीत| संगीत |music|Music|MUSIC|applause|laughter|"
        r"hansa|हंसी|नाक से[^\]]*|अचानक[^\]]*|sound[^\]]*)\s*\]",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\[[^\]]{0,40}\]", " ", text)  # other short [tags]
    text = re.sub(r"[♪♫]+", " ", text)
    # Broken currency / number artifacts like ₹000
    text = re.sub(r"₹\s*0{2,}", "₹", text)
    # Collapse whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _looks_like_youtube_description(text: str) -> bool:
    """
    True when text is YouTube description / promo, or a refine hallucination of it.
    """
    if not text or len(text.strip()) < 40:
        return False
    body = text.strip()
    if body.lower().startswith("source:"):
        body = body.split("\n", 1)[-1].strip()
    if body.startswith("Title:"):
        body = body.split("\n", 1)[-1].strip()
    low = body.lower()

    # Any one hard marker = description (Codanics promo specifically)
    hard = (
        "codanics.com",
        "codanics",
        "registration form",
        "whatsapp.com/channel",
        "python's chilla",
        "python ka chilla",
        "looking forward to seeing you in the course",
        "thank you for your interest in the course",
        "assalamu alaikum future founders",
        "build ai agents — from zero to hero",
        "earn millions 💰",
    )
    if any(h in low for h in hard):
        return True

    if low.startswith("about this video") or low.startswith("description:"):
        return True

    promo_hits = 0
    markers = (
        "subscribe", "like and", "click the link", "registration form",
        "whatsapp.com", "follow us", "stay updated",
        "don't forget to", "playlist?list=", "mentorship",
        "best regards", "welcome back to day", "earn millions",
    )
    for m in markers:
        if m in low:
            promo_hits += 1

    link_count = len(re.findall(r"https?://|www\.\w+", body))
    has_speech_cues = bool(
        re.search(
            r"\b(i mean|so basically|let us|let's|today we will|ab hum|"
            r"dekhte hain|samjhte hain|example|for example|"
            r"pehla idea|doosra|number one|number two|aapka|kisi|karein)\b",
            low,
        )
    ) or _has_arabic_or_devanagari(body)

    if promo_hits >= 2:
        return True
    if promo_hits >= 1 and link_count >= 1 and not has_speech_cues:
        return True
    if link_count >= 2 and not has_speech_cues:
        return True
    return False


def _render_speech_transcript(
    raw: str,
    out_lang: str = "auto",
    hard_cap: int = 14000,
) -> str:
    """
    mode=transcript from real speech — NO free-form rewrite.

    Larger chunks (8000) cut Groq RPM pressure; Groq→Gemini fallback on
    each chunk; failed chunks keep raw text instead of silent drop.
    """
    from language_policy import non_latin_ratio, has_non_latin_script

    cleaned = _clean_raw_transcript(raw or "")
    if not cleaned or _looks_like_youtube_description(cleaned):
        return ""

    # 8000 (was 4000) roughly halves Groq request count per video
    step = 8000
    chunks = [cleaned[i : i + step] for i in range(0, len(cleaned), step)]
    if len(chunks) > 8:
        chunks = chunks[:5] + chunks[-3:]
        print(
            f"[transcribe_video] speech_render trimmed to 8 chunks "
            f"(raw was {len(cleaned)} chars)"
        )

    if _client_ai is None and _client_gemini is None:
        return cleaned[:hard_cap]

    lang = (out_lang or "auto").strip().lower()
    if lang in ("en", "english"):
        sys = (
            "You are a literal translator. Translate the SPOKEN transcript into "
            "clear English. Keep meaning and order. Do NOT summarize or skip sections. "
            "Do NOT add course ads, subscribe CTAs, links, or 'Best regards'. "
            "Do NOT output Hindi/Urdu script — English Latin letters only. "
            "Output ONLY the translation of the speech below."
        )
        want_latin = True
    elif lang in ("urdu_script", "ur"):
        sys = (
            "Translate the SPOKEN content into Urdu using Arabic/Nastaliq script. "
            "Keep order. No course ads. Output ONLY the speech."
        )
        want_latin = False
    elif lang in ("hindi_script", "hi"):
        sys = (
            "Translate the SPOKEN content into Hindi Devanagari script. "
            "Keep order. No course ads. Output ONLY the speech."
        )
        want_latin = False
    else:
        sys = (
            "Transliterate the SPOKEN transcript into Roman Urdu (Latin letters only). "
            "Keep full content and order — do NOT summarize or skip sections. "
            "Never use Devanagari or Arabic letters. "
            "Do NOT add course ads, subscribe CTAs, links, or 'Best regards'. "
            "Output ONLY the speech text in Latin letters."
        )
        want_latin = True

    def _call_chunk(ch: str) -> str:
        content, finish, source = _ai_chat_completion_with_fallback(
            [
                {"role": "system", "content": sys},
                {"role": "user", "content": "Speech part:\n\n" + ch},
            ],
            max_tokens=3200,
            temperature=0.1,
            groq_retries=3,
        )
        if content:
            print(
                f"[transcribe_video] chunk render via {source} "
                f"chars={len(content)} finish={finish}"
            )
        return content

    parts: List[str] = []
    ok_count = 0
    for idx, ch in enumerate(chunks):
        if idx > 0:
            time.sleep(0.3)  # spread requests — proactive rate spacing
        piece = _call_chunk(ch)
        if piece and _looks_like_youtube_description(piece):
            piece = ""
        if piece and want_latin and non_latin_ratio(piece) > 0.25:
            print(
                f"[transcribe_video] chunk {idx} still non-Latin "
                f"(ratio={non_latin_ratio(piece):.2f}) — retry"
            )
            piece2 = _call_chunk(ch)
            if (
                piece2
                and non_latin_ratio(piece2) <= 0.25
                and not _looks_like_youtube_description(piece2)
            ):
                piece = piece2
            else:
                piece = ""
        if piece:
            parts.append(piece)
            ok_count += 1
        else:
            # Never silently drop — keeps mid-video content, avoids promo-tail splice
            print(
                f"[transcribe_video] speech_render chunk {idx} translate failed "
                "— keeping raw"
            )
            parts.append(f"[untranslated — source language]\n{ch.strip()}")

    out = "\n\n".join(parts).strip()
    if not out or (ok_count == 0 and _looks_like_youtube_description(out)):
        if want_latin and has_non_latin_script(cleaned):
            return (
                "Transcript mil gaya lekin language convert fail / rate-limit. "
                "Thori der baad dobara try karo (english/roman me)."
            )
        out = cleaned
    elif want_latin and non_latin_ratio(out) > 0.6:
        out = (
            "⚠️ Kuch hissa translate nahi ho saka (rate-limit) — baaki original "
            "language mein hai:\n\n" + out
        )
    if len(out) > hard_cap:
        out = (
            out[:hard_cap].rstrip()
            + "\n\n(transcript long hai — specific hissa chahiye to batao)"
        )
    return out



def _accept_transcript_candidate(text: Optional[str], source: str) -> Optional[str]:
    """Reject description-like blobs so cascade continues to real captions/ASR."""
    if not text or not str(text).strip():
        return None
    if _looks_like_youtube_description(text):
        msg = (
            f"[transcribe_video] REJECT {source}: looks like YouTube description "
            f"(chars={len(text)}) — trying next source"
        )
        print(msg)
        try:
            logging.getLogger("mojo").info(msg)
        except Exception:
            pass
        return None
    return text


def _has_arabic_or_devanagari(text: str) -> bool:
    for ch in text[:4000]:
        o = ord(ch)
        # Arabic / Urdu Nastaliq block or Devanagari
        if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x0900 <= o <= 0x097F:
            return True
    return False



# ---------------------------------------------------------------------------
# Adaptive length + continuation (never ship mid-sentence cuts)

# ---------------------------------------------------------------------------
# Adaptive length, map-reduce summary, safe continuation (no 413 loops)
# ---------------------------------------------------------------------------
_THOROUGH_MARKERS = frozenset({
    "break down", "breakdown", "har cheez", "har baat", "poora", "pura",
    "in depth", "indepth", "in-depth", "detailed", "detail me", "detail mein",
    "sab kuch", "everything", "complete", "full summary", "deep dive",
    "step by step", "step-by-step", "point by point", "har point",
    "thorough", "comprehensive", "exhaustive", "lengthy", "long summary",
})


def _request_is_thorough(user_request: str) -> bool:
    t = (user_request or "").lower()
    return any(m in t for m in _THOROUGH_MARKERS)


def _looks_incomplete(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return True
    if s.endswith(("…", "...", "—")):
        return True
    if s[-1] not in ".!?;:)]}”\"'…":
        last_line = s.split("\n")[-1].strip()
        if last_line.startswith(("*", "-", "•")) and len(last_line) < 100:
            return False
        if last_line.startswith("*") and last_line.endswith("*") and len(last_line) < 80:
            return False
        return True
    return False


def _trim_to_sentence(text: str, limit: int) -> str:
    if not text or len(text) <= limit:
        return text or ""
    chunk = text[:limit]
    for sep in ("\n\n", "\n", ". ", "? ", "! ", "। "):
        i = chunk.rfind(sep)
        if i >= int(limit * 0.55):
            return chunk[: i + len(sep)].rstrip()
    i = chunk.rfind(" ")
    if i >= int(limit * 0.5):
        return chunk[:i].rstrip() + "…"
    return chunk.rstrip() + "…"


def _chunk_text_by_chars(text: str, size: int = 5500, overlap: int = 200) -> list:
    """Split long transcript into overlapping windows for map-reduce."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    parts = []
    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + size)
        # prefer break at paragraph / sentence
        if end < n:
            window = text[i:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("। "))
            if cut >= int(size * 0.4):
                end = i + cut + 1
        parts.append(text[i:end].strip())
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    return [p for p in parts if p]



def _ai_chat_completion_with_fallback(
    messages: list,
    max_tokens: int,
    temperature: float = 0.2,
    groq_retries: int = 3,
) -> tuple:
    """
    Groq primary (exponential backoff + jitter on 429/5xx), Gemini fallback.
    Returns (content, finish_reason, source). Mirrors agent_loop._chat_completion.
    """
    if _client_ai is not None and _MODEL_NAME:
        working = list(messages)
        tokens = max(256, int(max_tokens))
        for attempt in range(groq_retries):
            try:
                resp = _client_ai.chat.completions.create(
                    model=_MODEL_NAME,
                    messages=working,
                    temperature=temperature,
                    max_tokens=tokens,
                )
                choice = resp.choices[0]
                content = (choice.message.content or "").strip()
                finish = (getattr(choice, "finish_reason", None) or "stop").lower()
                if content:
                    return content, finish, "groq"
            except Exception as e:
                err = str(e).lower()
                if "413" in err or "payload" in err or "too large" in err:
                    new_working = []
                    for m in working:
                        role = m.get("role")
                        c = m.get("content") or ""
                        if role == "user" and len(c) > 2000:
                            keep = max(1500, 8000 // (attempt + 2))
                            c = (
                                c[: keep // 2]
                                + "\n\n[...truncated...]\n\n"
                                + c[-(keep // 2):]
                            )
                        elif role == "system" and len(c) > 1500:
                            c = c[:1500]
                        new_working.append({"role": role, "content": c})
                    working = new_working
                    tokens = max(256, int(tokens * 0.75))
                    continue
                is_rate = "429" in err or "rate" in err or "too many" in err
                is_server = "500" in err or "502" in err or "503" in err
                if attempt < groq_retries - 1 and (is_rate or is_server):
                    sleep_s = min(15.0, (2 ** attempt) + random.uniform(0, 1))
                    print(
                        f"[agent_tools] Groq {'429' if is_rate else '5xx'} — "
                        f"retry in {sleep_s:.1f}s "
                        f"(attempt {attempt+1}/{groq_retries})"
                    )
                    time.sleep(sleep_s)
                    continue
                print(f"[agent_tools] Groq call failed, falling to Gemini: {e}")
                break

    if _client_gemini is not None and _GEMINI_MODEL:
        try:
            clean = [
                {
                    "role": (
                        m.get("role")
                        if m.get("role") in ("system", "user", "assistant")
                        else "user"
                    ),
                    "content": (
                        m.get("content")
                        if isinstance(m.get("content"), str)
                        else (m.get("content") or "")
                    ),
                }
                for m in messages
            ]
            resp = _client_gemini.chat.completions.create(
                model=_GEMINI_MODEL,
                messages=clean,
                temperature=temperature,
                max_tokens=max(256, int(max_tokens)),
            )
            choice = resp.choices[0]
            content = (choice.message.content or "").strip()
            finish = (getattr(choice, "finish_reason", None) or "stop").lower()
            if content:
                print(
                    f"[agent_tools] Gemini fallback succeeded ({len(content)} chars)"
                )
                return content, finish, "gemini"
        except Exception as e2:
            print(f"[agent_tools] Gemini fallback also failed: {e2}")

    return "", "error", "none"


def _groq_chat(
    messages: list,
    max_tokens: int,
    temperature: float = 0.2,
    retries: int = 4,
) -> tuple:
    """Returns (content, finish_reason). Groq w/ backoff, then Gemini fallback."""
    if (_client_ai is None or not _MODEL_NAME) and (
        _client_gemini is None or not _GEMINI_MODEL
    ):
        return "", "error"
    content, finish, _source = _ai_chat_completion_with_fallback(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        groq_retries=max(2, retries - 1),
    )
    return content, finish


def _continue_text_only(
    so_far: str,
    *,
    lang_rule: str,
    max_tokens: int = 1200,
    rounds: int = 3,
) -> str:
    """
    Continue a truncated answer WITHOUT resending the raw transcript
    (avoids Groq 413). Minimal context: system + assistant so_far + continue.
    """
    text = (so_far or "").strip()
    if not text:
        return text

    for r in range(rounds):
        if not _looks_incomplete(text) and r > 0:
            break
        # Only continue if clearly incomplete or first round after length stop
        msgs = [
            {
                "role": "system",
                "content": (
                    "Continue the WhatsApp reply exactly from where it stopped. "
                    "Do NOT repeat earlier text. Finish remaining points. "
                    f"{lang_rule} WhatsApp *bold* only (no **). No code fences."
                ),
            },
            {
                "role": "assistant",
                "content": text[-6000:],  # only tail — enough continuity
            },
            {
                "role": "user",
                "content": "Continue from the last sentence. Complete all remaining sections.",
            },
        ]
        piece, finish = _groq_chat(msgs, max_tokens=max_tokens, temperature=0.2)
        if not piece:
            break
        # Avoid pure duplication
        if piece in text:
            break
        # If model repeated a large prefix, strip it
        overlap = 0
        max_o = min(400, len(piece), len(text))
        for o in range(max_o, 40, -20):
            if text.endswith(piece[:o]):
                overlap = o
                break
        addition = piece[overlap:].strip() if overlap else piece
        if not addition:
            break
        text = (text.rstrip() + "\n\n" + addition).strip()
        print(f"[continue] round={r+1} finish={finish} total={len(text)}")
        if finish in ("stop", "end_turn") and not _looks_incomplete(text):
            break
        if finish != "length" and not _looks_incomplete(text):
            break
    return text


def _map_reduce_summary(
    cleaned: str,
    *,
    system_core: str,
    user_request: str,
    title: str,
    max_tok_final: int,
    lang_rule: str,
) -> str:
    """
    For long transcripts: summarize each chunk, then merge.
    Prevents 413 by keeping each call under context limits.
    """
    chunks = _chunk_text_by_chars(cleaned, size=8000, overlap=180)
    print(f"[map_reduce] chunks={len(chunks)} total_chars={len(cleaned)}")

    partials = []
    for idx, ch in enumerate(chunks):
        sys = (
            f"{system_core}\n"
            f"This is part {idx+1}/{len(chunks)} of the transcript. "
            "Extract and summarize ALL important points from THIS part only. "
            "Use short paragraphs or '- ' bullets. Do not invent."
        )
        user = (
            (f"USER ASK: {user_request[:300]}\n\n" if user_request else "")
            + (f"Title: {title}\n\n" if title else "")
            + f"Transcript part {idx+1}/{len(chunks)}:\n{ch}"
        )
        part, finish = _groq_chat(
            [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            max_tokens=min(1200, max_tok_final),
            temperature=0.2,
        )
        if part:
            partials.append(f"[Part {idx+1}]\n{part}")
            print(f"[map_reduce] part {idx+1} chars={len(part)} finish={finish}")
        else:
            print(f"[map_reduce] part {idx+1} EMPTY")

    if not partials:
        return ""

    if len(partials) == 1:
        merged = partials[0]
        # strip [Part 1] header
        merged = re.sub(r"^\[Part \d+\]\s*", "", merged).strip()
    else:
        join_body = "\n\n".join(partials)
        # Cap merge input
        if len(join_body) > 14000:
            join_body = join_body[:14000]
        sys = (
            f"{system_core}\n"
            "Merge the partial summaries into ONE coherent WhatsApp reply. "
            "Remove [Part N] labels and duplication. Cover every major point. "
            "Finish completely — never stop mid-sentence."
        )
        user = (
            (f"USER ASK: {user_request[:400]}\n\n" if user_request else "")
            + (f"Title: {title}\n\n" if title else "")
            + "Partial summaries to merge:\n"
            + join_body
        )
        merged, finish = _groq_chat(
            [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tok_final,
            temperature=0.2,
        )
        print(f"[map_reduce] merge chars={len(merged or '')} finish={finish}")
        if not merged:
            # fallback: concatenate partials
            merged = "\n\n".join(
                re.sub(r"^\[Part \d+\]\s*", "", p).strip() for p in partials
            )
        if finish == "length" or _looks_incomplete(merged or ""):
            merged = _continue_text_only(
                merged or "",
                lang_rule=lang_rule,
                max_tokens=min(1500, max_tok_final),
                rounds=3,
            )

    return (merged or "").strip()


def _refine_transcript_compact(
    raw: str,
    title: str = "",
    mode: str = "transcript",
    target_words: Optional[int] = None,
    output_lang: str = "auto",
    user_request: str = "",
) -> str:
    """
    Robust LLM refine for WhatsApp.

    - Long input → map-reduce (avoids Groq 413)
    - finish_reason=length → minimal-context continuation (no full transcript resend)
    - Thorough asks without word count → ~1000 word budget
    - Never return Devanagari when Latin required
    """
    if not raw or not raw.strip():
        return raw or ""

    mode = (mode or "transcript").strip().lower()
    if mode not in ("transcript", "summary", "key_points"):
        mode = "transcript"
    out_lang = (output_lang or "auto").strip().lower()
    if out_lang in ("auto", "") and user_request:
        try:
            from language_policy import detect_output_lang
            out_lang = detect_output_lang(user_request)
        except Exception:
            out_lang = "roman_urdu"

    tw = None
    if target_words is not None:
        try:
            tw = int(target_words)
        except (TypeError, ValueError):
            tw = None
        if tw is not None:
            tw = max(80, min(2500, tw))
    if tw is None and mode == "summary" and _request_is_thorough(user_request):
        tw = 1000

    cleaned = _clean_raw_transcript(raw)
    if _client_ai is None or not _MODEL_NAME:
        return cleaned[:3000] + ("…" if len(cleaned) > 3000 else "")

    want_latin = out_lang not in ("urdu_script", "hindi_script", "ur", "hi")

    if out_lang in ("en", "english"):
        script_rule = (
            "OUTPUT LANGUAGE: English only. Latin letters only. "
            "Translate any Hindi/Urdu speech into clear English."
        )
    elif out_lang in ("urdu_script", "ur"):
        script_rule = "OUTPUT SCRIPT: Urdu (Arabic/Nastaliq) as requested."
        want_latin = False
    elif out_lang in ("hindi_script", "hi"):
        script_rule = "OUTPUT SCRIPT: Hindi Devanagari as requested."
        want_latin = False
    else:
        script_rule = (
            "OUTPUT SCRIPT: Roman Urdu only (Latin letters). "
            "Transliterate Devanagari/Arabic. Never output native Urdu/Hindi script."
        )

    if mode == "summary":
        if tw:
            hard_cap = min(14000, max(2500, tw * 7))
            max_tok = min(4500, max(1400, int(tw * 2.0)))
            format_rule = (
                f"MODE=summary. Aim ~{tw} words (not under {int(tw * 0.6)}). "
                "Cover every major point. Section titles on own lines, blank lines between "
                "sections, '- ' bullets. WhatsApp *bold* only (never **). "
                "No # headers, no tables, no ---. Finish completely."
            )
        else:
            hard_cap, max_tok = 2500, 1100
            format_rule = (
                "MODE=summary. Title + clear coverage of main points. "
                "WhatsApp *bold* only. Finish completely."
            )
    elif mode == "key_points":
        hard_cap, max_tok = 2200, 1000
        format_rule = (
            "MODE=key_points. Title + 8–15 '- ' bullets covering all main ideas. "
            "WhatsApp *bold* only. Finish the full list."
        )
    else:
        hard_cap, max_tok = 12000, 3500
        format_rule = (
            "MODE=transcript. Clean readable paragraphs, full coverage, finish completely."
        )

    system_core = (
        "You turn a raw video transcript into a WhatsApp reply.\n"
        "CRITICAL: Use ONLY facts from the transcript. Do not invent.\n"
        "Never end mid-sentence.\n"
        f"{script_rule}\n"
        f"{format_rule}"
    )

    try:
        # Long speech → map-reduce; short → single shot
        use_map = mode in ("summary", "key_points") and len(cleaned) > 7000

        if use_map:
            out = _map_reduce_summary(
                cleaned,
                system_core=system_core,
                user_request=user_request or "",
                title=title or "",
                max_tok_final=max_tok,
                lang_rule=script_rule,
            )
        else:
            # Single-shot with safe input size
            body = cleaned
            if len(body) > 12000:
                body = body[:8000] + "\n\n[...middle omitted...]\n\n" + body[-3500:]
            user = (
                (f"USER ASK: {(user_request or '')[:500]}\n\n" if user_request else "")
                + (f"Title: {title}\n\n" if title else "")
                + "Raw transcript:\n"
                + body
            )
            out, finish = _groq_chat(
                [
                    {"role": "system", "content": system_core},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tok,
                temperature=0.2,
            )
            print(f"[refine] single finish={finish} chars={len(out or '')}")
            if out and (finish == "length" or _looks_incomplete(out)):
                out = _continue_text_only(
                    out,
                    lang_rule=script_rule,
                    max_tokens=min(1500, max_tok),
                    rounds=3,
                )

        if not out:
            raise RuntimeError("empty refine")

        if want_latin and _has_arabic_or_devanagari(out):
            print("[transcribe_video] refine non-Latin — transliterate")
            out2, _fr = _groq_chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Transliterate into Roman Urdu (Latin letters only). "
                            "Keep structure and meaning. No Devanagari/Arabic. Finish fully."
                        ),
                    },
                    {"role": "user", "content": out[:10000]},
                ],
                max_tokens=min(max_tok, 3500),
            )
            if out2 and not _has_arabic_or_devanagari(out2):
                out = out2
                if _looks_incomplete(out):
                    out = _continue_text_only(
                        out, lang_rule=script_rule, max_tokens=1200, rounds=2
                    )
            else:
                return (
                    "Summary source Hindi/Urdu script me tha; Roman convert "
                    "complete nahi hua. 30–60 sec baad dobara try karo."
                )

        if _looks_like_youtube_description(out):
            raise RuntimeError("refine_still_description")

        if len(out) > hard_cap:
            out = _trim_to_sentence(out, hard_cap)
        return out
    except Exception as e:
        print(f"[transcribe_video] refine failed: {e}")
        if want_latin and _has_arabic_or_devanagari(cleaned):
            return (
                "Transcript mil gaya lekin summary convert abhi fail. "
                "Thori der baad dobara try karo."
            )
        return _trim_to_sentence(cleaned, 3000)



def _try_piped_captions(
    video_id: str, language: str, with_timestamps: bool
) -> Optional[str]:
    """
    Fully free YouTube captions via public Piped instances.
    Often works from cloud/datacenter IPs where youtube-transcript-api is blocked.
    """
    lang = (language or "auto").strip().lower()
    preferred: List[str] = []
    if lang and lang not in ("auto", ""):
        preferred.append(lang)
    # Hindi/Urdu auto-captions first when auto — English-only prefer often
    # returned empty tracks and upstream APIs fell back to description text.
    preferred.extend(["hi", "ur", "en", "en-US", "en-GB"])

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MojoBot/1.0)",
        "Accept": "application/json",
    }

    for host in _PIPED_API_HOSTS:
        try:
            r = requests.get(
                f"{host}/streams/{video_id}",
                headers=headers,
                timeout=15,
            )
            if r.status_code != 200:
                print(f"[transcribe_video] Piped {host} → HTTP {r.status_code}")
                continue
            data = r.json()
            subs = data.get("subtitles") or []
            if not subs:
                print(f"[transcribe_video] Piped {host}: no subtitles for {video_id}")
                continue

            def score(s: dict) -> tuple:
                code = (s.get("code") or "").lower()
                auto = 1 if s.get("autoGenerated") else 0
                rank = 99
                for i, p in enumerate(preferred):
                    if code == p or code.startswith(p + "-") or p.startswith(code):
                        rank = i
                        break
                return (rank, auto)

            chosen = sorted(subs, key=score)[0]
            sub_url = chosen.get("url") or ""
            if not sub_url:
                continue
            if sub_url.startswith("/"):
                # relative — prefix host
                from urllib.parse import urlparse

                p = urlparse(host)
                sub_url = f"{p.scheme}://{p.netloc}{sub_url}"

            r2 = requests.get(sub_url, headers=headers, timeout=20)
            if r2.status_code != 200 or not (r2.text or "").strip():
                print(f"[transcribe_video] Piped caption body {r2.status_code}")
                continue

            segs = _parse_ttml_or_vtt(r2.text)
            if not segs:
                print("[transcribe_video] Piped: 0 segments after parse")
                continue

            text = _format_segments(segs, with_timestamps)
            if text:
                # Add the rejection check here
                text = _accept_transcript_candidate(text, f"piped_{host}")
                if not text:
                    continue  # It was a description, move to the next host!
                    
                print(
                    f"[transcribe_video] Piped OK {host} "
                    f"lang={chosen.get('code')} segs={len(segs)}"
                )
                return text
        except Exception as e:
            print(f"[transcribe_video] Piped {host} error: {e}")
            continue
    return None


def _try_supadata(url: str, language: str, with_timestamps: bool) -> Optional[str]:
    """
    Managed API — YouTube, TikTok, Instagram, Facebook, X (cloud-friendly).
    Requires SUPADATA_API_KEY (100 free credits/mo).

    Instagram/Facebook rarely expose native captions → try mode=auto then
    mode=generate. Handles async jobId via short-polling.
    """
    api_key = (os.getenv("SUPADATA_API_KEY") or "").strip()
    if not api_key:
        print("[transcribe_video] Supadata skipped — SUPADATA_API_KEY not set")
        return None
    print(
        f"[transcribe_video] Supadata key present (…{api_key[-4:]}) "
        f"url={url[:80]}"
    )

    lang = (language or "auto").strip().lower()

    def _parse_content(data: dict) -> Optional[str]:
        if not isinstance(data, dict):
            return None
        content = data.get("content")
        if isinstance(content, str):
            return content.strip() or None
        if isinstance(content, list):
            segs = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                t = (item.get("text") or "").strip()
                if not t:
                    continue
                start_ms = item.get("offset") or item.get("start") or 0
                try:
                    start_s = float(start_ms) / 1000.0
                except Exception:
                    start_s = 0.0
                dur_ms = item.get("duration") or 0
                try:
                    dur_s = float(dur_ms) / 1000.0
                except Exception:
                    dur_s = 0.0
                segs.append({"text": t, "start": start_s, "duration": dur_s})
            return _format_segments(segs, with_timestamps) if segs else None
        for key in ("transcript", "text", "plainText"):
            v = data.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # Nested result shapes
        for nest in ("data", "result", "transcript"):
            inner = data.get(nest)
            if isinstance(inner, dict):
                got = _parse_content(inner)
                if got:
                    return got
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
        return None

    def _poll_job(job_id: str) -> Optional[str]:
        print(f"[transcribe_video] Supadata job {job_id} — polling")
        job_urls = [
            f"https://api.supadata.ai/v1/transcript/{job_id}",
            f"https://api.supadata.ai/v1/transcript/job/{job_id}",
            f"https://api.supadata.ai/v1/job/{job_id}",
        ]
        for _ in range(18):  # ~45s
            time.sleep(2.5)
            for ju in job_urls:
                try:
                    jr = requests.get(
                        ju,
                        headers={"x-api-key": api_key, "Accept": "application/json"},
                        timeout=30,
                    )
                except Exception as e:
                    print(f"[transcribe_video] Supadata poll error: {e}")
                    continue
                if jr.status_code not in (200, 202):
                    continue
                jdata = jr.json() or {}
                status = str(jdata.get("status") or "").lower()
                parsed = _parse_content(jdata)
                if parsed:
                    # Add the rejection check here
                    parsed = _accept_transcript_candidate(parsed, "supadata-poll")
                    if parsed:
                        print(f"[transcribe_video] Supadata job OK chars={len(parsed)}")
                        return parsed
                    else:
                        print(f"[transcribe_video] Supadata job returned description, rejecting")
                        return None
                if status in ("failed", "error"):
                    print(f"[transcribe_video] Supadata job failed: {str(jdata)[:200]}")
                    return None
                if status in ("pending", "processing", "queued", "running", ""):
                    break  # try next sleep cycle
            else:
                continue
        print("[transcribe_video] Supadata job timed out")
        return None

    def _one_mode(mode: str) -> Optional[str]:
        params: Dict[str, Any] = {
            "url": url,
            "text": "false" if with_timestamps else "true",
            "mode": mode,
        }
        if lang and lang not in ("auto", ""):
            params["lang"] = lang
        try:
            r = requests.get(
                "https://api.supadata.ai/v1/transcript",
                params=params,
                headers={"x-api-key": api_key, "Accept": "application/json"},
                timeout=120,
            )
            if r.status_code == 206:
                print(f"[transcribe_video] Supadata mode={mode} → 206 unavailable")
                return None
            if r.status_code not in (200, 202):
                print(
                    f"[transcribe_video] Supadata mode={mode} HTTP {r.status_code}: "
                    f"{r.text[:200]}"
                )
                return None
            data = r.json() or {}
            job_id = data.get("jobId") or data.get("job_id")
            if job_id:
                return _poll_job(str(job_id))
            parsed = _parse_content(data)
            if parsed:
                # Add the rejection check here
                parsed = _accept_transcript_candidate(parsed, f"supadata-{mode}")
                if parsed:
                    print(
                    f"[transcribe_video] Supadata mode={mode} OK chars={len(parsed)}"
                    )
                    return parsed
                else:
                    return None
            print(
                f"[transcribe_video] Supadata mode={mode} empty body keys="
                f"{list(data.keys())[:12]}"
            )
            return None
        except Exception as e:
            print(f"[transcribe_video] Supadata mode={mode} error: {e}")
            return None

    # IG/FB: native captions rare → try generate first for reliability
    is_meta = _is_instagram_url(url) or _is_facebook_url(url)
    modes = ("generate", "auto") if is_meta else ("auto", "generate")
    for m in modes:
        got = _one_mode(m)
        if got and len(got.strip()) >= 20:
            return got
        if got:
            print(
                f"[transcribe_video] Supadata mode={m} too short "
                f"({len(got.strip())} chars) — trying next"
            )
    return None


def _try_youtube_captions(
    video_id: str, language: str, with_timestamps: bool
) -> Optional[str]:
    """Return formatted transcript or None if captions unavailable / blocked."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        print("[transcribe_video] youtube-transcript-api not installed")
        return None

    # Collect known exception types without hard-failing on missing names
    blocked_types = ()
    try:
        from youtube_transcript_api._errors import (
            TranscriptsDisabled,
            NoTranscriptFound,
            VideoUnavailable,
        )
        blocked_types = (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable)
    except Exception:
        pass
    for name in ("IpBlocked", "RequestBlocked", "TooManyRequests"):
        try:
            mod = __import__("youtube_transcript_api._errors", fromlist=[name])
            blocked_types = blocked_types + (getattr(mod, name),)
        except Exception:
            pass

    lang = (language or "auto").strip().lower()
    if lang in ("", "auto"):
        # Hindi/Urdu first — many Codanics / PK-IN lectures only have hi auto-captions
        preferred = ["hi", "ur", "en", "en-US", "en-GB"]
    else:
        preferred = [lang, "hi", "ur", "en", "en-US"]

    try:
        ytt = YouTubeTranscriptApi()
        try:
            fetched = ytt.fetch(video_id, languages=preferred)
            segs = [
                {"text": sn.text, "start": sn.start, "duration": sn.duration}
                for sn in fetched
            ]
            return _format_segments(segs, with_timestamps)
        except Exception as inner:
            if hasattr(YouTubeTranscriptApi, "get_transcript"):
                try:
                    raw = YouTubeTranscriptApi.get_transcript(
                        video_id, languages=preferred
                    )
                    return _format_segments(raw, with_timestamps)
                except Exception:
                    raise inner
            raise
    except blocked_types as e:
        print(f"[transcribe_video] captions unavailable/blocked for {video_id}: {e}")
        return None
    except Exception as e:
        # Cloud IP blocks, rate limits, etc. — fall through to ASR
        print(f"[transcribe_video] youtube-transcript-api failed for {video_id}: {e}")
        return None


def _resolve_short_url(url: str) -> str:
    """Follow redirects for short links (vt.tiktok.com, vm.tiktok.com, bit.ly, etc.)."""
    try:
        # TikTok short links often need a real browser-like UA or they return a
        # challenge / different landing page.
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
                "Mobile/15E148 Safari/604.1"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = requests.get(
            url,
            headers=headers,
            allow_redirects=True,
            timeout=12,
            stream=True,  # don't download body
        )
        final = (r.url or url).split("?")[0]
        # Prefer the canonical www.tiktok.com/@user/video/ID form when possible
        if "tiktok.com" in final and "/video/" in final:
            return final
        return final or url
    except Exception:
        return url


def _is_tiktok_url(url: str) -> bool:
    u = (url or "").lower()
    return any(
        h in u
        for h in (
            "tiktok.com",
            "vt.tiktok.com",
            "vm.tiktok.com",
            "tiktokv.com",
            "musical.ly",
        )
    )


def _is_instagram_url(url: str) -> bool:
    u = (url or "").lower()
    return "instagram.com" in u or "instagr.am" in u


def _is_facebook_url(url: str) -> bool:
    u = (url or "").lower()
    return any(
        h in u
        for h in (
            "facebook.com/",
            "fb.watch/",
            "fb.gg/",
            "fb.com/",
            "m.facebook.com/",
        )
    )


def _normalize_ig_fb_url(url: str) -> str:
    """Strip tracking query noise; keep reel/p/tv/watch path intact."""
    u = (url or "").strip()
    if not u:
        return u
    # Drop common trackers (igshid, igsi, si, fbclid, …) but keep path
    try:
        from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

        p = urlparse(u)
        qs = parse_qs(p.query)
        keep = {}
        for k, v in qs.items():
            kl = k.lower()
            if kl in ("v", "story_fbid", "id"):
                keep[k] = v
        new_q = urlencode({k: v[0] for k, v in keep.items()}) if keep else ""
        return urlunparse((p.scheme or "https", p.netloc, p.path, "", new_q, ""))
    except Exception:
        return u.split("?")[0] if "?" in u else u


def _try_oembed_about(url: str) -> Optional[str]:
    """
    Public oEmbed / lightweight page metadata for Instagram + Facebook.
    Gives title + author + caption-like description when full ASR is blocked.
    Free, no API key.
    """
    u = _normalize_ig_fb_url(url)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
    }
    endpoints = []
    if _is_instagram_url(u):
        endpoints.append(
            f"https://www.instagram.com/api/v1/oembed/?url={requests.utils.quote(u, safe='')}"
        )
        endpoints.append(
            f"https://api.instagram.com/oembed/?url={requests.utils.quote(u, safe='')}"
        )
    if _is_facebook_url(u):
        endpoints.append(
            "https://www.facebook.com/plugins/video/oembed.json"
            f"?url={requests.utils.quote(u, safe='')}&omitscript=true"
        )
        endpoints.append(
            "https://www.facebook.com/plugins/post/oembed.json"
            f"?url={requests.utils.quote(u, safe='')}&omitscript=true"
        )

    for ep in endpoints:
        try:
            r = requests.get(ep, headers=headers, timeout=12)
            if r.status_code != 200:
                continue
            data = r.json()
            title = (data.get("title") or "").strip()
            author = (
                data.get("author_name")
                or data.get("author_url")
                or data.get("provider_name")
                or ""
            ).strip()
            # Some oEmbed payloads put caption in title; html is noise
            parts = []
            if author:
                parts.append(f"Creator: {author}")
            if title:
                parts.append(title)
            text = "\n".join(parts).strip()
            if text and len(text) >= 8:
                print(f"[transcribe_video] oEmbed OK chars={len(text)}")
                return text
        except Exception as e:
            print(f"[transcribe_video] oEmbed fail {ep[:60]}: {e}")
    return None



def _resolve_ytdlp_cookies(*, context: str = "ytdlp") -> Optional[str]:
    """
    Resolve YTDLP_COOKIES env to a real cookiefile path.
    Logs once per call so Render/deploy misconfig is obvious in mojo logs.
    """
    raw = (os.getenv("YTDLP_COOKIES") or "").strip()
    if not raw:
        msg = f"[cookies] {context}: YTDLP_COOKIES unset — yt-dlp will run without cookies"
        print(msg)
        try:
            logging.getLogger("mojo").info(msg)
        except Exception:
            pass
        return None
    if os.path.isfile(raw):
        try:
            size = os.path.getsize(raw)
        except OSError:
            size = -1
        msg = (
            f"[cookies] {context}: using cookiefile path={raw!r} size={size}b"
        )
        print(msg)
        try:
            logging.getLogger("mojo").info(msg)
        except Exception:
            pass
        return raw
    msg = (
        f"[cookies] {context}: YTDLP_COOKIES set but not a file path={raw!r} "
        f"(cwd={os.getcwd()!r}) — cookies NOT applied"
    )
    print(msg)
    try:
        logging.getLogger("mojo").warning(msg)
    except Exception:
        pass
    return None


def _try_ytdlp_metadata_about(url: str) -> Optional[str]:
    """
    yt-dlp extract_info (no download) → title/description/uploader.
    Works for many public IG/FB/TikTok posts even when audio download is blocked.
    """
    try:
        import yt_dlp
    except ImportError:
        return None

    u = _normalize_ig_fb_url(url)
    opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "skip_download": True,
        "socket_timeout": 25,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
                "Mobile/15E148 Safari/604.1"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    }
    if _is_instagram_url(u):
        opts["http_headers"]["Referer"] = "https://www.instagram.com/"
    elif _is_facebook_url(u):
        opts["http_headers"]["Referer"] = "https://www.facebook.com/"

    cookies_path = _resolve_ytdlp_cookies(context="yt_opts")
    if cookies_path:
        opts["cookiefile"] = cookies_path

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(u, download=False)
        if not info:
            return None
        title = (info.get("title") or info.get("fulltitle") or "").strip()
        desc = (info.get("description") or "").strip()
        uploader = (
            info.get("uploader")
            or info.get("channel")
            or info.get("creator")
            or info.get("uploader_id")
            or ""
        ).strip()
        # Instagram sometimes puts caption only in description
        parts = []
        if uploader:
            parts.append(f"Creator: {uploader}")
        if title and title.lower() not in (desc.lower() if desc else ""):
            parts.append(f"Title: {title}")
        if desc:
            # Cap description — captions can be long
            parts.append(desc[:2500])
        text = "\n".join(parts).strip()
        if text and len(text) >= 12:
            print(
                f"[transcribe_video] yt-dlp metadata OK "
                f"title={title[:40]!r} desc_len={len(desc)}"
            )
            return text
    except Exception as e:
        print(f"[transcribe_video] yt-dlp metadata fail: {e}")
    return None


def _try_supadata_metadata(url: str) -> Optional[str]:
    """Optional Supadata /metadata for IG/FB/TikTok when key is set."""
    api_key = (os.getenv("SUPADATA_API_KEY") or "").strip()
    if not api_key:
        return None
    try:
        r = requests.get(
            "https://api.supadata.ai/v1/metadata",
            params={"url": url},
            headers={"x-api-key": api_key, "Accept": "application/json"},
            timeout=30,
        )
        if r.status_code != 200:
            print(f"[transcribe_video] Supadata metadata HTTP {r.status_code}")
            return None
        data = r.json() or {}
        # Flexible field names across platform payloads
        title = (
            data.get("title")
            or (data.get("post") or {}).get("title")
            or ""
        )
        if isinstance(title, dict):
            title = title.get("text") or ""
        desc = (
            data.get("description")
            or data.get("caption")
            or (data.get("post") or {}).get("description")
            or (data.get("post") or {}).get("caption")
            or ""
        )
        author_raw = data.get("author") or data.get("authorName") or ""
        if isinstance(author_raw, dict):
            author = (
                author_raw.get("name")
                or author_raw.get("username")
                or author_raw.get("handle")
                or ""
            )
        else:
            author = author_raw or ""
        parts = []
        if author:
            parts.append(f"Creator: {str(author).strip()}")
        if title:
            parts.append(f"Title: {str(title).strip()}")
        if desc:
            parts.append(str(desc).strip()[:2500])
        text = "\n".join(p for p in parts if p).strip()
        if text and len(text) >= 8:
            print(f"[transcribe_video] Supadata metadata OK chars={len(text)}")
            return text
    except Exception as e:
        print(f"[transcribe_video] Supadata metadata error: {e}")
    return None


def _about_fallback_pack(url: str, mode: str) -> Optional[str]:
    """
    When speech transcript is unavailable, return best-effort 'about' text
    (caption / description / oEmbed). Useful for IG/FB restricted reels.
    """
    body = None
    # Prefer richer sources first
    for fn, name in (
        (_try_supadata_metadata, "Supadata metadata"),
        (_try_ytdlp_metadata_about, "yt-dlp metadata"),
        (_try_oembed_about, "oEmbed"),
    ):
        try:
            body = fn(url)
        except Exception as e:
            print(f"[transcribe_video] about via {name} error: {e}")
            body = None
        if body:
            header = (
                f"Title: (about / caption — full spoken transcript unavailable)\n\n"
                f"{body}"
            )
            # Light refine still ok for summary/key_points; for transcript return raw
            if mode in ("summary", "key_points"):
                refined = _refine_transcript_compact(
                    body, title="", mode=mode, target_words=None
                )
                return (
                    f"Source: {name} (about/caption) | mode={mode}\n\n{refined}"
                )
            return f"Source: {name} (about/caption) | mode={mode}\n\n{body.strip()}"
    return None


def _try_ytdlp_subs(url: str, with_timestamps: bool) -> Optional[str]:
    """
    Fetch auto/manual subtitles via yt-dlp without downloading audio.
    Prefer hi/ur tracks for South-Asian lectures; works when timedtext APIs
    are blocked but yt-dlp player_client cascade still gets tracks.
    """
    try:
        import yt_dlp
    except ImportError:
        return None

    # Strip tracking
    clean = url
    if "youtu" in url.lower() and "?" in url:
        base, _, qs = url.partition("?")
        if "v=" in qs:
            for part in qs.split("&"):
                if part.startswith("v="):
                    clean = f"{base}?{part}"
                    break
        else:
            clean = base

    clients_try = (
        ["tv", "android", "ios", "web"],
        ["android", "ios"],
        ["tv"],
    )
    cookies = _resolve_ytdlp_cookies(context="caption_or_dl")

    for clients in clients_try:
        opts: Dict[str, Any] = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["hi.*", "ur.*", "en.*", "hi", "ur", "en"],
            "subtitlesformat": "vtt/srv3/best",
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "noplaylist": True,
            "socket_timeout": 25,
            "retries": 2,
            "extractor_args": {"youtube": {"player_client": clients}},
            "http_headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
            },
        }
        if cookies:
            opts["cookiefile"] = cookies
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(clean, download=False)
            if not info:
                continue
            # Prefer requested lang tracks from subtitles / automatic_captions
            subs = info.get("subtitles") or {}
            autos = info.get("automatic_captions") or {}
            # Merge: prefer manual over auto for same lang
            ordered_langs = []
            for key in ("hi", "ur", "en", "en-US", "en-GB"):
                for bag in (subs, autos):
                    for lang in bag:
                        if lang.lower().startswith(key.lower()) and lang not in ordered_langs:
                            ordered_langs.append(lang)
            for lang in list(subs.keys()) + list(autos.keys()):
                if lang not in ordered_langs:
                    ordered_langs.append(lang)

            for lang in ordered_langs:
                entries = (subs.get(lang) or autos.get(lang) or [])
                if not entries:
                    continue
                # Prefer vtt / srv3
                entries = sorted(
                    entries,
                    key=lambda e: (
                        0 if (e.get("ext") or "") in ("vtt", "srv3", "ttml") else 1
                    ),
                )
                sub_url = entries[0].get("url")
                if not sub_url:
                    continue
                r = requests.get(
                    sub_url,
                    headers={"User-Agent": opts["http_headers"]["User-Agent"]},
                    timeout=30,
                )
                if r.status_code != 200 or not (r.text or "").strip():
                    continue
                segs = _parse_ttml_or_vtt(r.text)
                if not segs or len(segs) < 5:
                    # raw text fallback
                    text = re.sub(r"<[^>]+>", " ", r.text)
                    text = re.sub(r"\s+", " ", text).strip()
                    if len(text) < 80:
                        continue
                    if _looks_like_youtube_description(text):
                        continue
                    print(
                        f"[transcribe_video] yt-dlp subs OK lang={lang} "
                        f"raw_chars={len(text)} clients={clients}"
                    )
                    return text
                text = _format_segments(segs, with_timestamps)
                if text and not _looks_like_youtube_description(text):
                    print(
                        f"[transcribe_video] yt-dlp subs OK lang={lang} "
                        f"segs={len(segs)} clients={clients}"
                    )
                    return text
        except Exception as e:
            print(f"[transcribe_video] yt-dlp subs clients={clients} err={e}")
            continue
    return None


def _download_audio_ytdlp(url: str, out_dir: str) -> tuple[str, float]:
    """
    Download best audio, convert to mono 48k mp3.
    Returns (path_to_mp3, duration_seconds).
    Raises on hard failure.
    """
    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError(
            "yt-dlp is not installed. Add 'yt-dlp' to requirements and redeploy."
        ) from e

    # Resolve short links first (especially TikTok vt./vm.)
    original = url
    if any(x in url.lower() for x in ("vt.tiktok.", "vm.tiktok.", "tiktokv.com")):
        url = _resolve_short_url(url)
        if url != original:
            print(f"[transcribe_video] resolved short URL → {url}")

    # Strip tracking params that sometimes confuse extractors
    if "youtu" in url.lower() and "?" in url:
        base, _, _qs = url.partition("?")
        # keep only v= if present; otherwise drop all query
        if "v=" in _qs:
            for part in _qs.split("&"):
                if part.startswith("v="):
                    url = f"{base}?{part}"
                    break
        else:
            url = base

    outtmpl = os.path.join(out_dir, "%(id)s.%(ext)s")

    base_opts: Dict[str, Any] = {
        # Prefer real audio stream; fall back to muxed best (IG often has no separate ba)
        "format": "bestaudio[ext=m4a]/bestaudio/best[height<=720]/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 3,
        "fragment_retries": 3,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "48",
            }
        ],
        "match_filter": lambda info, *, incomplete: (
            "Video too long for transcription "
            f"(>{_MAX_ASR_SECONDS // 60} min)"
            if (info.get("duration") or 0) > _MAX_ASR_SECONDS + 30
            else None
        ),
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/128.0.0.0 Safari/537.36"
            ),
        },
    }

    # Optional cookies file (set YTDLP_COOKIES=/path/to/cookies.txt on the host)
    cookies_path = _resolve_ytdlp_cookies(context="ytdlp_download")
    if cookies_path:
        base_opts["cookiefile"] = cookies_path

    attempt_opts: List[Dict[str, Any]] = []

    if _is_tiktok_url(url):
        # TikTok: Referer is required (Aug 2026+ challenge)
        tiktok_opts = dict(base_opts)
        tiktok_opts["http_headers"] = {
            **base_opts["http_headers"],
            "Referer": "https://www.tiktok.com/",
        }
        tiktok_opts["extractor_args"] = {
            "tiktok": {
                "api_hostname": ["api16-normal-c-useast1a.tiktokv.com"],
            }
        }
        attempt_opts.append(tiktok_opts)
        # Fallback without extractor_args
        fb = dict(tiktok_opts)
        fb.pop("extractor_args", None)
        attempt_opts.append(fb)
    elif _extract_youtube_id(url):
        # YouTube bot-check workaround for datacenter IPs (2026).
        # Clients that often still work without cookies/PO tokens:
        # tv_simply, tv, android_vr, web_embedded, mweb.
        for clients in (
            "tv_simply,tv,android_vr",
            "android_vr,web_embedded,mweb",
            "tv,android,ios",
            "web_embedded",
        ):
            opts = dict(base_opts)
            opts["extractor_args"] = {
                "youtube": {"player_client": [clients]}
            }
            attempt_opts.append(opts)
        attempt_opts.append(dict(base_opts))
    elif _is_instagram_url(url):
        # Instagram: mobile UA + Referer; cookies help a lot for restricted posts
        url = _normalize_ig_fb_url(url)
        ig_opts = dict(base_opts)
        ig_opts["http_headers"] = {
            "User-Agent": (
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
                "Mobile/15E148 Safari/604.1"
            ),
            "Referer": "https://www.instagram.com/",
            "Accept-Language": "en-US,en;q=0.9",
        }
        attempt_opts.append(ig_opts)
        # Desktop UA fallback
        ig2 = dict(base_opts)
        ig2["http_headers"] = {
            **base_opts["http_headers"],
            "Referer": "https://www.instagram.com/",
        }
        attempt_opts.append(ig2)
    elif _is_facebook_url(url):
        url = _normalize_ig_fb_url(url)
        fb_opts = dict(base_opts)
        fb_opts["http_headers"] = {
            **base_opts["http_headers"],
            "Referer": "https://www.facebook.com/",
        }
        attempt_opts.append(fb_opts)
        # Mobile site fallback
        fb_m = dict(base_opts)
        fb_m["http_headers"] = {
            "User-Agent": (
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
                "Mobile/15E148 Safari/604.1"
            ),
            "Referer": "https://m.facebook.com/",
            "Accept-Language": "en-US,en;q=0.9",
        }
        attempt_opts.append(fb_m)
    else:
        attempt_opts.append(base_opts)

    last_err: Optional[Exception] = None
    for opts in attempt_opts:
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if not info:
                    raise RuntimeError("yt-dlp returned no info for this URL.")
                duration = float(info.get("duration") or 0)
                vid = info.get("id") or "audio"
                title = (info.get("title") or info.get("fulltitle") or "")[:80]
                print(
                    f"[transcribe_video] yt-dlp downloaded id={vid} "
                    f"duration={duration:.1f}s title={title!r}"
                )
                mp3_path = os.path.join(out_dir, f"{vid}.mp3")
                if not os.path.isfile(mp3_path):
                    candidates = [
                        os.path.join(out_dir, f)
                        for f in os.listdir(out_dir)
                        if f.startswith(str(vid))
                    ]
                    if not candidates:
                        raise RuntimeError("Audio download finished but no file found.")
                    mp3_path = candidates[0]
                # Reject absurdly short files for non-trivial videos
                try:
                    fsz = os.path.getsize(mp3_path)
                except OSError:
                    fsz = 0
                if duration >= 15 and fsz < 8000:
                    raise RuntimeError(
                        f"Downloaded audio too small ({fsz} bytes for {duration:.0f}s) — "
                        "wrong/partial stream"
                    )
                return mp3_path, duration
        except Exception as e:
            last_err = e
            err_s = str(e).lower()
            # Keep trying alternate clients on bot-check / challenge errors
            retryable = any(
                x in err_s
                for x in (
                    "sign in to confirm",
                    "not a bot",
                    "unexpected response from webpage",
                    "login_required",
                    "confirm you're not a bot",
                )
            )
            print(f"[transcribe_video] yt-dlp attempt failed: {e}")
            if not retryable and len(attempt_opts) > 1:
                # Non-retryable (e.g. private, deleted) — stop early
                if "private" in err_s or "unavailable" in err_s or "too long" in err_s:
                    break
            continue

    raise RuntimeError(str(last_err) if last_err else "yt-dlp download failed")


def _reencode_if_needed(src_path: str, work_dir: str) -> str:
    """Ensure mono low-bitrate mp3 under Groq size limit. Returns path to use."""
    size = os.path.getsize(src_path)
    if size <= _GROQ_MAX_UPLOAD_BYTES and src_path.lower().endswith(".mp3"):
        return src_path

    out = os.path.join(work_dir, "audio_48k_mono.mp3")
    # ffmpeg already confirmed present in environment at install time
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        src_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        _TARGET_AUDIO_BITRATE,
        "-t",
        str(_MAX_ASR_SECONDS),
        out,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0 or not os.path.isfile(out):
        raise RuntimeError(f"ffmpeg re-encode failed: {proc.stderr[-400:]}")
    return out


def _transcribe_with_groq(audio_path: str, language: str) -> str:
    """Use the same Groq Whisper client already used for voice notes."""
    if _client_ai is None:
        raise RuntimeError("AI client not initialised — cannot run Whisper.")

    # Model matches whatsapp_agent.WHISPER_MODEL
    model = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo")
    lang = (language or "").strip().lower()
    kwargs: Dict[str, Any] = {
        "model": model,
        "response_format": "verbose_json",  # segments with timestamps
    }
    if lang and lang not in ("auto", ""):
        kwargs["language"] = lang

    with open(audio_path, "rb") as f:
        result = _client_ai.audio.transcriptions.create(file=f, **kwargs)

    # verbose_json → object with .text and .segments
    if hasattr(result, "segments") and result.segments:
        segs = []
        for s in result.segments:
            segs.append(
                {
                    "text": getattr(s, "text", "") or "",
                    "start": getattr(s, "start", 0.0),
                    "duration": (getattr(s, "end", 0.0) or 0) - (getattr(s, "start", 0.0) or 0),
                }
            )
        return _format_segments(segs, with_timestamps=True)  # always keep ts internally; caller decides
    return (getattr(result, "text", None) or str(result) or "").strip()


def _tool_transcribe_video(args: dict, ctx: dict) -> str:
    url = (args.get("url") or "").strip()
    language = (args.get("language") or "auto").strip().lower() or "auto"
    with_ts = bool(args.get("timestamps"))
    mode = (args.get("mode") or "transcript").strip().lower() or "transcript"
    if mode not in ("transcript", "summary", "key_points"):
        mode = "transcript"
    target_words = args.get("target_words")
    try:
        target_words = int(target_words) if target_words is not None else None
    except (TypeError, ValueError):
        target_words = None
    user_request = (args.get("user_request") or args.get("query") or "").strip()

    if not url:
        return "URL required. Example: https://www.youtube.com/watch?v=..."
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # Include mode + length in cache so summary ≠ 400-word summary ≠ transcript
    cache_k = _cache_key(
        f"{url}|{mode}|tw={target_words or 0}", language, with_ts
    )
    cached = _cache_get(cache_k)
    if cached:
        # Stale process may hold description-as-transcript; never serve it
        if _looks_like_youtube_description(cached):
            print(
                f"[transcribe_video] cache DROP description-like "
                f"(chars={len(cached)}) key={cache_k[:80]!r}"
            )
            _TRANSCRIPT_CACHE.pop(cache_k, None)
        else:
            print(f"[transcribe_video] cache HIT chars={len(cached)}")
            return cached

    yt_id = _extract_youtube_id(url)

    def _raw_cache_get() -> Optional[str]:
        if not yt_id:
            return None
        entry = _RAW_SPEECH_CACHE.get(yt_id)
        if not entry:
            return None
        text, exp = entry
        if time.time() > exp:
            _RAW_SPEECH_CACHE.pop(yt_id, None)
            return None
        if _looks_like_youtube_description(text):
            _RAW_SPEECH_CACHE.pop(yt_id, None)
            return None
        return text

    def _raw_cache_set(text: str, source: str) -> None:
        if not yt_id or not text or _looks_like_youtube_description(text):
            return
        if len(text) < 120:
            return
        _RAW_SPEECH_CACHE[yt_id] = (text, time.time() + _RAW_SPEECH_TTL)
        msg = (
            f"[transcribe_video] RAW_CACHE store id={yt_id} "
            f"source={source} chars={len(text)}"
        )
        print(msg)
        try:
            logging.getLogger("mojo").info(msg)
        except Exception:
            pass

    def _pack(source: str, body: str) -> str:
        """
        Pack speech into user-facing text for the requested mode.
        mode=transcript uses speech-preserving render (same raw as summary).
        mode=summary/key_points still uses compact refine.
        """
        title = ""
        raw_body = body or ""
        if raw_body.startswith("Title:"):
            first, _, rest = raw_body.partition("\n")
            title = first.replace("Title:", "", 1).strip()
            raw_body = rest.lstrip("\n")
        if title and _looks_like_youtube_description(f"Title: {title}\n{title}"):
            title = ""
        if _looks_like_youtube_description(raw_body):
            raise RuntimeError(
                f"raw_is_description source={source} chars={len(raw_body)}"
            )
        # Persist real speech for other modes (transcript ↔ summary)
        _raw_cache_set(raw_body, source)

        out_lang = "roman_urdu"
        if language in ("en", "english"):
            out_lang = "en"
        elif language in ("urdu_script", "ur"):
            out_lang = "urdu_script"
        elif language in ("hindi_script", "hi"):
            out_lang = "hindi_script"

        if mode == "transcript":
            # CRITICAL: do not free-form refine — that was turning Supadata's
            # 20k Hindi speech into the YouTube description (2591 chars).
            refined = _render_speech_transcript(raw_body, out_lang=out_lang)
            if not refined or _looks_like_youtube_description(refined):
                raise RuntimeError(
                    f"speech_render_failed source={source} chars={len(raw_body)}"
                )
            msg = (
                f"[transcribe_video] speech_render source={source} "
                f"raw={len(raw_body)} out={len(refined)}"
            )
            print(msg)
            try:
                logging.getLogger("mojo").info(msg)
            except Exception:
                pass
        else:
            refined = _refine_transcript_compact(
                raw_body,
                title=title,
                mode=mode,
                target_words=target_words,
                output_lang=out_lang,
                user_request=user_request,
            )
            if _looks_like_youtube_description(refined):
                raise RuntimeError(
                    f"refine_still_description source={source} chars={len(refined)}"
                )

        meta = f"Source: {source} | mode={mode}"
        if target_words:
            meta += f" | target_words={target_words}"
        out = f"{meta}\n\n{refined}"
        _cache_set(cache_k, out)
        return out

    is_ig = _is_instagram_url(url)
    is_fb = _is_facebook_url(url)
    if is_ig or is_fb:
        url = _normalize_ig_fb_url(url)

    # Caption FETCH language must stay auto/hi-capable.
    # User "in english" only controls refine output_lang — if we pass language=en
    # into caption APIs, Hindi-auto videos often return DESCRIPTION instead of speech.
    fetch_lang = "auto"
    if language in ("hi", "ur", "en", "en-US", "en-GB") and language not in (
        "en",
        "english",
    ):
        fetch_lang = language
    start_msg = (
        f"[transcribe_video] start url={url[:60]!r} mode={mode} "
        f"fetch_lang={fetch_lang} refine_lang={language}"
    )
    print(start_msg)
    try:
        logging.getLogger("mojo").info(start_msg)
    except Exception:
        pass

    def _take(source: str, candidate: Optional[str]) -> Optional[str]:
        ok = _accept_transcript_candidate(candidate, source)
        if not ok:
            return None
        msg = (
            f"[transcribe_video] ACCEPT {source} chars={len(ok)} "
            f"head={ok[:100]!r}"
        )
        print(msg)
        try:
            logging.getLogger("mojo").info(msg)
        except Exception:
            pass
        try:
            return _pack(source, ok)
        except RuntimeError as e:
            if "description" in str(e):
                print(f"[transcribe_video] REJECT after pack {source}: {e}")
                try:
                    logging.getLogger("mojo").info(
                        "REJECT after pack %s: %s", source, e
                    )
                except Exception:
                    pass
                return None
            raise

    # Reuse raw speech captured by a prior summary/transcript call on this video
    cached_raw = _raw_cache_get()
    if cached_raw:
        msg = (
            f"[transcribe_video] RAW_CACHE hit id={yt_id} "
            f"chars={len(cached_raw)} mode={mode}"
        )
        print(msg)
        try:
            logging.getLogger("mojo").info(msg)
        except Exception:
            pass
        packed = _take(f"raw_cache({yt_id})", cached_raw)
        if packed:
            return packed

    # YouTube: real caption tracks first. youtube2text is last — it often returns
    # the video *description* when speech tracks are missing for a language.
    if yt_id:
        packed = _take(
            f"Piped captions (video {yt_id})",
            _try_piped_captions(yt_id, fetch_lang, with_ts),
        )
        if packed:
            return packed
        packed = _take(
            f"YouTube captions (video {yt_id})",
            _try_youtube_captions(yt_id, fetch_lang, with_ts),
        )
        if packed:
            return packed
        # yt-dlp subtitle URL fetch (no audio) — better on cloud than Innertube
        packed = _take(
            "yt-dlp subtitles",
            _try_ytdlp_subs(url, with_ts),
        )
        if packed:
            return packed

    # Managed API (SUPADATA) — strong for YT + best for IG/FB/TikTok
    packed = _take("Supadata", _try_supadata(url, fetch_lang, with_ts))
    if packed:
        return packed

    # youtube2text: only for summary/key_points, NEVER for full transcript mode
    # (too often returns description which users mistake for speech).
    if mode in ("summary", "key_points") and (yt_id or "youtu" in url.lower()):
        packed = _take(
            "youtube2text.org", _try_youtube2text(url, fetch_lang, with_ts)
        )
        if packed:
            return packed
    elif mode == "transcript" and (yt_id or "youtu" in url.lower()):
        print(
            "[transcribe_video] skip youtube2text for mode=transcript "
            "(description pollution risk) — going to ASR"
        )
        try:
            logging.getLogger("mojo").info(
                "skip youtube2text for mode=transcript — ASR next"
            )
        except Exception:
            pass

    # --- 4. ASR path: yt-dlp → ffmpeg → Groq Whisper ---
    asr_error_msg: Optional[str] = None
    try:
        with tempfile.TemporaryDirectory(prefix="mojo_vid_") as tmp:
            try:
                audio_path, duration = _download_audio_ytdlp(url, tmp)
            except Exception as e:
                msg = str(e)
                low = msg.lower()
                if "too long" in low:
                    return (
                        f"Video is longer than {_MAX_ASR_SECONDS // 60} minutes — "
                        "transcription limit for safety/cost. Try a shorter clip or "
                        "ask for a specific section."
                    )
                if "tiktok" in low or (
                    _is_tiktok_url(url) and "unexpected response from webpage" in low
                ):
                    asr_error_msg = (
                        "TikTok video se audio nahi nikal saka — TikTok abhi yt-dlp "
                        "ko block / challenge kar raha hai (common temporary issue). "
                        "YouTube link try karo, ya video download karke voice note "
                        "bhej do to main transcribe kar sakta hoon."
                    )
                    raise
                if any(
                    x in low
                    for x in (
                        "sign in to confirm",
                        "not a bot",
                        "login_required",
                        "confirm you're not a bot",
                    )
                ) and (yt_id or "youtu" in url.lower()):
                    asr_error_msg = (
                        "YouTube ne is link pe bot-check laga diya hai (cloud server IP "
                        "block). Captions bhi available nahi the. "
                        "Workaround: video download karke voice note bhej do, ya "
                        "host pe YTDLP_COOKIES env set karo (browser cookies.txt)."
                    )
                    raise
                # Instagram / Facebook: keep going to about/caption fallback
                if is_ig or is_fb or "instagram" in low or "facebook" in low:
                    print(f"[transcribe_video] IG/FB audio download failed: {msg[:200]}")
                    asr_error_msg = msg
                    raise
                asr_error_msg = msg
                raise

            if duration and duration > _MAX_ASR_SECONDS + 5:
                return (
                    f"Video is ~{int(duration // 60)} min long. "
                    f"Max supported is {_MAX_ASR_SECONDS // 60} min."
                )

            try:
                audio_path = _reencode_if_needed(audio_path, tmp)
            except Exception as e:
                return f"Audio prepare failed (ffmpeg): {e}"

            size = os.path.getsize(audio_path)
            if size > _GROQ_MAX_UPLOAD_BYTES:
                return (
                    f"Audio still too large after compression ({size // (1024*1024)} MB). "
                    "Try a shorter video."
                )

            try:
                # Whisper: auto-detect speech language (Hindi OK); English is refine-only
                raw = _transcribe_with_groq(audio_path, "auto")
            except Exception as e:
                return f"Speech-to-text failed: {e}"

            if not raw or not raw.strip():
                raise RuntimeError(
                    "Transcription returned empty text (silent video or unsupported language)."
                )

            # If caller did not want timestamps, strip them
            if not with_ts and raw.lstrip().startswith("["):
                cleaned = re.sub(
                    r"(?m)^\[\d{1,2}(?::\d{2}){1,2}\]\s*",
                    "",
                    raw,
                )
                text = re.sub(r"\s+", " ", cleaned).strip()
            else:
                text = raw.strip()

            # Guard: very thin ASR vs longer video → likely wrong/partial download
            # (this caused fake "What's up / Bye" summaries on Instagram reels).
            word_count = len(re.findall(r"\w+", text))
            print(
                f"[transcribe_video] ASR words={word_count} duration={duration:.1f}s "
                f"chars={len(text)}"
            )
            if duration and duration >= 12 and word_count < 12:
                raise RuntimeError(
                    f"ASR too thin ({word_count} words for {duration:.0f}s video) — "
                    "likely wrong/partial audio stream"
                )
            if word_count < 4:
                raise RuntimeError(
                    f"ASR too thin ({word_count} words) — not usable as transcript"
                )

            return _pack("speech-to-text (audio download)", text)
    except Exception as e:
        # Don't dump full traceback for expected platform blocks
        err_s = str(asr_error_msg or e)
        low = err_s.lower()
        print(f"[transcribe_video] ASR path failed: {err_s[:240]}")

        # --- 5. About / caption fallback (esp. Instagram + Facebook) ---
        # Public caption/description is often still readable when audio download
        # is blocked (restricted audience, login wall, cloud IP).
        if is_ig or is_fb or _is_tiktok_url(url):
            about = _about_fallback_pack(url, mode)
            if about:
                _cache_set(cache_k, about)
                return about

        if is_ig:
            if any(
                x in low
                for x in (
                    "isn't available to everyone",
                    "not available to everyone",
                    "can't be seen by certain audiences",
                    "cannot be seen by certain audiences",
                    "login_required",
                    "private",
                    "only available to",
                    "restricted",
                )
            ):
                return (
                    "Ye Instagram reel/post public nahi hai (private ya restricted "
                    "audience). Main iska audio/transcript nahi nikaal sakta. "
                    "Public reel share karo, ya video download karke voice note bhej do."
                )
            return (
                "Instagram se spoken transcript nahi nikal saka (login wall / block "
                "common hai cloud servers pe). Caption/about bhi available nahi tha. "
                "Public reel try karo, SUPADATA_API_KEY set karo, ya video download "
                "karke voice note bhej do."
            )
        if is_fb:
            return (
                "Facebook video/reel se transcript nahi nikal saka (public + "
                "login-free posts best work karte hain). Caption/about bhi empty. "
                "Public link try karo, ya video download karke voice note bhej do."
            )
        if asr_error_msg and (
            "TikTok" in asr_error_msg or "YouTube" in asr_error_msg
        ):
            return asr_error_msg
        traceback.print_exc()
        return (
            f"Could not download audio from this link ({err_s[:180]}). "
            "Is the video public and supported?"
        )


def _tool_link_preview(args: dict, ctx: dict) -> str:
    """Title + caption + author only — no ASR. Cheap path for IG/FB about asks."""
    url = (args.get("url") or "").strip()
    if not url:
        return "URL required for link_preview."
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    if _is_instagram_url(url) or _is_facebook_url(url):
        url = _normalize_ig_fb_url(url)

    # Reuse the about cascade (Supadata metadata → yt-dlp info → oEmbed)
    body = None
    source = None
    for fn, name in (
        (_try_supadata_metadata, "Supadata metadata"),
        (_try_ytdlp_metadata_about, "yt-dlp metadata"),
        (_try_oembed_about, "oEmbed"),
    ):
        try:
            body = fn(url)
        except Exception as e:
            print(f"[link_preview] {name} error: {e}")
            body = None
        if body and len(body.strip()) >= 8:
            source = name
            break

    if not body:
        if _is_instagram_url(url):
            return (
                "Is Instagram link ka public caption/title nahi mila "
                "(private, restricted, ya login wall). "
                "Public reel share karo ya video download karke voice note bhej do."
            )
        if _is_facebook_url(url):
            return (
                "Is Facebook link ka public title/caption nahi mila. "
                "Public post try karo."
            )
        return (
            "Link preview nahi nikal saka (title/caption unavailable). "
            "Public URL confirm karo."
        )

    lines = [f"Source: link_preview ({source})", "", body.strip()]
    out = "\n".join(lines)
    print(f"[link_preview] OK source={source} chars={len(out)}")
    return out


TOOL_EXECUTORS: Dict[str, Callable[[dict, dict], str]] = {
    "search_knowledge": _tool_search_knowledge,
    "web_search": _tool_web_search,
    "browse_url": _tool_browse_url,
    "get_weather": _tool_get_weather,
    "get_memory": _tool_get_memory,
    "save_memory": _tool_save_memory,
    "set_reminder": _tool_set_reminder,
    "list_reminders": _tool_list_reminders,
    "cancel_reminders": _tool_cancel_reminders,
    "query_contacts": _tool_query_contacts,
    "lookup_user": _tool_lookup_user,
    "send_message_to": _tool_send_message_to,
    "file_list_onedrive": _tool_file_list_onedrive,
    "python_exec": _tool_python_exec,
    "note_down": _tool_note_down,
    "image_describe": _tool_image_describe,
    "transcribe_video": _tool_transcribe_video,
    "link_preview": _tool_link_preview,
    "draft_email": _tool_draft_email,
    "send_email": _tool_send_email,
}


# Tools that already enforce OWNER_SENDER_ID inside the executor.
# Owner may use these from any chat even if the group tool flag is OFF.
_OWNER_GATED_TOOLS = frozenset({
    "note_down",
    "python_exec",
    "send_message_to",
    "draft_email",
    "send_email",
})


def execute_tool(name: str, arguments: dict, ctx: dict) -> str:
    fn = TOOL_EXECUTORS.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    # Admin tool flags (default OFF). Fail closed if control plane unavailable.
    # Owner-gated tools: if the caller is the owner, allow even when the chat
    # flag is OFF (owner journal / exec / proactive send are personal).
    try:
        import admin_commands as _ac
        chat_id = (ctx or {}).get("chat_id") or ""
        if name in getattr(_ac, "KNOWN_TOOLS", ()) and chat_id:
            if not _ac.is_tool_enabled(chat_id, name):
                is_owner = False
                if name in _OWNER_GATED_TOOLS and _OWNER_SENDER_ID:
                    sid = str((ctx or {}).get("sender_id") or "")
                    snum = str((ctx or {}).get("sender_num") or "")
                    is_owner = sid == str(_OWNER_SENDER_ID) or snum == str(
                        _OWNER_SENDER_ID
                    )
                if not is_owner:
                    return (
                        f"Tool '{name}' is disabled for this chat. "
                        "Ask the bot owner to enable it if needed."
                    )
                print(
                    f"[execute_tool] owner bypass chat-flag OFF for {name} "
                    f"chat={chat_id}"
                )
    except Exception as e:
        print(f"[execute_tool] permission check failed for {name}: {e}")
        return f"Tool '{name}' unavailable (permission check failed)."
    try:
        return fn(arguments or {}, ctx)
    except Exception as e:
        traceback.print_exc()
        return f"Tool {name} raised: {e}"


# time is imported at module top (used by set_reminder + transcript cache)