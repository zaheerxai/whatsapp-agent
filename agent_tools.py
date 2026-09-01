"""
agent_tools.py — Tool schemas + executors for Mojo's agentic loop.

All tools return a plain string observation that the LLM sees.
Permission-sensitive tools (send_message_to, python_exec) check OWNER_SENDER_ID.
"""

from __future__ import annotations

import json
import os
import re
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
LANGUAGE_POLICY = """- Mirror the SCRIPT the person actually used in THIS message — not the topic, not the vibe, just what script they typed.
  - They wrote in English -> reply in English.
  - They wrote in Roman Urdu/Hindi (Latin letters — e.g. "kya haal hai", "paani peene ka reminder set kar do") -> reply in Roman Urdu/Hindi, same script.
  - They wrote in native Urdu script (نستعلیق / Arabic letters) -> reply in Urdu script.
  - They explicitly asked for Urdu script (e.g. "Urdu mein likho", "اردو میں جواب دو") -> reply in Urdu script even if their own message was typed in Roman.
- Default for Urdu/Hindi speakers who haven't specified a script = Roman Urdu (Latin letters). Never switch to native Urdu script just because a topic or reminder subject feels "more Urdu" — only the person's own explicit script or request decides this, never your own judgment call.
- Voice notes have no script the person "chose" — a voice transcript is treated as Roman Urdu by default unless the person has explicitly asked for Urdu script elsewhere in the conversation.
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
):
    global _supabase, _client_ai, _MODEL_NAME, _BUSINESS_KNOWLEDGE
    global _DEFAULT_TIMEZONE, _OWNER_SENDER_ID
    global _get_contacts_maps, _get_user_timezone, _set_user_timezone, _get_tzinfo
    global _handle_reminder_request, _list_reminders, _cancel_reminders
    global _get_group_memory, _send_proactive_message, _file_ops
    global _compute_next_occurrence, _extract_reminder_data_via_ai, _generate_reminder_confirmation

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


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI-compatible)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Search Mojo AI Agency knowledge base + permanent chat notes. "
                "Use for questions about services, portfolio, founder, pricing process, "
                "or any standing rules/facts saved for this chat. Always call this before "
                "answering business or 'what can you do' questions."
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
                "Save a durable fact, rule, or preference for this chat so it persists "
                "across future conversations. Use when user says 'remember that...', "
                "'from now on...', or gives a standing instruction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {
                        "type": "string",
                        "description": "Short clean fact/rule to store",
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
            "description": "List recent files uploaded by the agent to OneDrive (if configured).",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 10},
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
            "name": "note_down",
            "description": (
                "Save or append a note to the owner's OneDrive journal. "
                "Use whenever the user says 'note down'. If they ask for specific formatting "
                "(e.g., 'concise', 'bullets', 'Urdu'), format the text exactly as requested "
                "BEFORE calling this tool. Only the owner can use this."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The finalized, perfectly formatted text to save."
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transcribe_video",
            "description": (
                "Get the spoken transcript of a public video from a link (YouTube, Vimeo, "
                "TikTok, direct video URL, etc.). First tries existing captions (YouTube) for "
                "speed; if none exist, downloads audio and runs speech-to-text. "
                "Use when the user pastes a video link and asks what was said, transcript, "
                "summary of spoken content, 'is video me kya bola', 'transcript nikaalo', etc. "
                "Do NOT use for voice notes already transcribed in the chat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full public video URL (YouTube, youtu.be, Vimeo, etc.)",
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
                        "description": "If true, include [MM:SS] timestamps per segment",
                        "default": False,
                    },
                },
                "required": ["url"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------

def _tool_search_knowledge(args: dict, ctx: dict) -> str:
    query = (args.get("query") or "").strip().lower()
    top_k = min(int(args.get("top_k") or 5), 10)
    if not query:
        return "Empty query."

    chunks: List[str] = []

    # 1. Business knowledge – simple paragraph split + keyword score
    paras = [p.strip() for p in re.split(r"\n\s*\n", _BUSINESS_KNOWLEDGE) if p.strip()]
    scored = []
    tokens = set(re.findall(r"\w+", query))
    for p in paras:
        p_low = p.lower()
        score = sum(1 for t in tokens if t in p_low)
        if score > 0:
            scored.append((score, p[:800]))
    scored.sort(key=lambda x: -x[0])
    for _, p in scored[:top_k]:
        chunks.append(f"[Agency] {p}")

    # 2. Group memory notes
    try:
        notes = _get_group_memory(ctx["chat_id"]) or []
        for note in notes:
            if any(t in note.lower() for t in tokens) or not tokens:
                chunks.append(f"[Chat note] {note}")
                if len(chunks) >= top_k + 3:
                    break
    except Exception as e:
        chunks.append(f"(memory lookup error: {e})")

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
    sender_id = args.get("sender_id") or ctx.get("sender_id") or "unknown"
    try:
        _supabase.table("group_memory").insert(
            {
                "chat_id": ctx["chat_id"],
                "sender_id": sender_id,
                "note": note,
            }
        ).execute()
        return f"Saved permanent note: {note}"
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
    # Minimal stub — full listing would need Graph list API; keep honest
    return (
        "OneDrive is configured. Use admin /uploadimg or the export log commands "
        "for file operations. Full directory listing is not exposed via tools yet."
    )


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

def _tool_note_down(args: dict, ctx: dict) -> str:
    # 1. Strict Owner Check
    if not _OWNER_SENDER_ID or (
        str(ctx.get("sender_id")) != str(_OWNER_SENDER_ID)
        and str(ctx.get("sender_num") or "") != str(_OWNER_SENDER_ID)
    ):
        return "Permission denied: Only the bot owner can use the note down feature."

    content = (args.get("content") or "").strip()
    if not content:
        return "Empty note. Nothing was saved."

    try:
        # 2. Format with Date, Time, and 3-line gap
        # Assuming we use the UTC time context or we can adapt to PKT
        now_str = datetime.now(timezone.utc).strftime("%A, %Y-%m-%d %I:%M %p UTC")
        formatted_entry = f"{now_str}\n{content}\n\n\n"

        remote_folder = "MojoAgent"
        file_name = "Mojo_Notes.txt"
        remote_path = f"{remote_folder}/{file_name}"
        local_temp = os.path.join(tempfile.gettempdir(), file_name)

        # 3. Try downloading existing file to append (file_ops.py)
        existing_content = ""
        try:
            if _file_ops.onedrive_configured():
                dl_path = _file_ops.download_from_onedrive(remote_path, local_temp)
                with open(dl_path, "r", encoding="utf-8") as f:
                    existing_content = f.read()
        except Exception:
            pass # File likely doesn't exist yet, we will create a new one

        # 4. Append and write back locally
        new_content = existing_content + formatted_entry
        _file_ops.write_text_file(new_content, file_name, tempfile.gettempdir())

        # 5. Upload back to OneDrive
        _file_ops.upload_to_onedrive(local_temp, remote_folder=remote_folder, remote_name=file_name)
        
        return "Successfully saved to OneDrive folder MojoAgent."
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
# Safety caps for cloud (Render) — avoid huge downloads / Groq 25 MB limit
_MAX_ASR_SECONDS = 45 * 60  # 45 min hard cap
_TARGET_AUDIO_BITRATE = "48k"  # mono speech is fine at 48 kbps
_GROQ_MAX_UPLOAD_BYTES = 24 * 1024 * 1024


def _extract_youtube_id(url: str) -> Optional[str]:
    m = _YT_ID_RE.search(url or "")
    return m.group(1) if m else None


def _cache_key(url: str, language: str, timestamps: bool) -> str:
    return f"{url.strip().lower()}|{language or 'auto'}|{int(bool(timestamps))}"


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


def _try_youtube_captions(
    video_id: str, language: str, with_timestamps: bool
) -> Optional[str]:
    """Return formatted transcript or None if captions unavailable / blocked."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import (
            TranscriptsDisabled,
            NoTranscriptFound,
            VideoUnavailable,
        )
    except ImportError:
        return None

    # Language priority: requested → Urdu/Hindi/English → any
    lang = (language or "auto").strip().lower()
    if lang in ("", "auto"):
        preferred = ["en", "ur", "hi", "en-US", "en-GB"]
    else:
        preferred = [lang, "en", "ur", "hi"]

    try:
        # New-style API (v1.x+)
        ytt = YouTubeTranscriptApi()
        try:
            fetched = ytt.fetch(video_id, languages=preferred)
            segs = [
                {"text": sn.text, "start": sn.start, "duration": sn.duration}
                for sn in fetched
            ]
            return _format_segments(segs, with_timestamps)
        except Exception:
            # Older static API fallback
            if hasattr(YouTubeTranscriptApi, "get_transcript"):
                raw = YouTubeTranscriptApi.get_transcript(video_id, languages=preferred)
                return _format_segments(raw, with_timestamps)
            raise
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return None
    except Exception as e:
        # IP blocks / transient — fall through to ASR
        print(f"[transcribe_video] youtube-transcript-api failed: {e}")
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

    outtmpl = os.path.join(out_dir, "%(id)s.%(ext)s")

    # Common options
    base_opts: Dict[str, Any] = {
        "format": "bestaudio/best",
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
        # Critical for TikTok (Aug 2026+): missing Referer triggers
        # "Unexpected response from webpage request"
        "http_headers": {
            "Referer": "https://www.tiktok.com/",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/128.0.0.0 Safari/537.36"
            ),
        },
    }

    # TikTok-specific: try mobile API path first when possible; also keep
    # webpage path with proper headers. Latest yt-dlp (2026.08.19+) fixed
    # the challenge + referer issues.
    if _is_tiktok_url(url):
        base_opts["extractor_args"] = {
            "tiktok": {
                # Prefer mobile app-style extraction when web is blocked
                "api_hostname": ["api16-normal-c-useast1a.tiktokv.com"],
            }
        }

    last_err: Optional[Exception] = None
    # Two attempts: (1) with TikTok-friendly headers, (2) plain fallback
    attempt_opts = [base_opts]
    if _is_tiktok_url(url):
        # Second attempt: strip extractor_args, keep only referer (web path)
        fallback = dict(base_opts)
        fallback.pop("extractor_args", None)
        attempt_opts.append(fallback)

    for opts in attempt_opts:
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if not info:
                    raise RuntimeError("yt-dlp returned no info for this URL.")
                duration = float(info.get("duration") or 0)
                vid = info.get("id") or "audio"
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
                return mp3_path, duration
        except Exception as e:
            last_err = e
            err_s = str(e).lower()
            # Retry only on the known TikTok webpage challenge
            if "unexpected response from webpage" not in err_s and "tiktok" not in err_s:
                break
            print(f"[transcribe_video] yt-dlp attempt failed, retrying: {e}")

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

    if not url:
        return "URL required. Example: https://www.youtube.com/watch?v=..."
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    cache_k = _cache_key(url, language, with_ts)
    cached = _cache_get(cache_k)
    if cached:
        return cached

    # --- 1. YouTube captions (fast path) ---
    yt_id = _extract_youtube_id(url)
    if yt_id:
        text = _try_youtube_captions(yt_id, language, with_ts)
        if text:
            # Soft length guard for WhatsApp / LLM context
            if len(text) > 12000:
                text = text[:12000] + "\n… [transcript truncated]"
            header = f"Source: YouTube captions (video {yt_id})\n\n"
            out = header + text
            _cache_set(cache_k, out)
            return out

    # --- 2. ASR path: yt-dlp → ffmpeg → Groq Whisper ---
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
                if "tiktok" in low or "unexpected response from webpage" in low:
                    return (
                        "TikTok video se audio nahi nikal saka — TikTok abhi yt-dlp "
                        "ko block / challenge kar raha hai (common temporary issue). "
                        "YouTube link try karo, ya video download karke voice note "
                        "bhej do to main transcribe kar sakta hoon."
                    )
                return (
                    f"Could not download audio from this link ({msg[:180]}). "
                    "Is the video public and supported?"
                )

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
                raw = _transcribe_with_groq(audio_path, language)
            except Exception as e:
                return f"Speech-to-text failed: {e}"

            if not raw or not raw.strip():
                return "Transcription returned empty text (silent video or unsupported language)."

            # If caller did not want timestamps, strip them
            if not with_ts and raw.lstrip().startswith("["):
                # remove [MM:SS] / [HH:MM:SS] prefixes
                cleaned = re.sub(
                    r"(?m)^\[\d{1,2}(?::\d{2}){1,2}\]\s*",
                    "",
                    raw,
                )
                text = re.sub(r"\s+", " ", cleaned).strip()
            else:
                text = raw.strip()

            if len(text) > 12000:
                text = text[:12000] + "\n… [transcript truncated]"

            header = "Source: speech-to-text (audio download)\n\n"
            out = header + text
            _cache_set(cache_k, out)
            return out
    except Exception as e:
        traceback.print_exc()
        return f"transcribe_video failed: {e}"


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
    "transcribe_video": _tool_transcribe_video,
}


def execute_tool(name: str, arguments: dict, ctx: dict) -> str:
    fn = TOOL_EXECUTORS.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    try:
        return fn(arguments or {}, ctx)
    except Exception as e:
        traceback.print_exc()
        return f"Tool {name} raised: {e}"


# time is imported at module top (used by set_reminder + transcript cache)