"""
agent_tools.py — Tool schemas + executors for Mojo's agentic loop.

All tools return a plain string observation that the LLM sees.
Permission-sensitive tools (send_message_to, python_exec) check OWNER_SENDER_ID.
"""

from __future__ import annotations

import json
import os
import re
import traceback
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests

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
                "from a previously shared link. NEVER invent repo names or page content."
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
                        "description": "Max characters for non-GitHub pages (default 6000)",
                        "default": 6000,
                    },
                },
                "required": ["url"],
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


def _tool_web_search(args: dict, ctx: dict) -> str:
    query = (args.get("query") or "").strip()
    num = min(max(int(args.get("num_results") or 5), 1), 8)
    if not query:
        return "Empty query."

    # Prefer Serper if key present, else DuckDuckGo Instant Answer + HTML fallback
    serper_key = os.getenv("SERPER_API_KEY")
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
                lines.append(f"• {title}\n  {snippet}\n  {link}")
            if data.get("answerBox"):
                ab = data["answerBox"]
                lines.insert(0, f"Answer box: {ab.get('answer') or ab.get('snippet') or ''}")
            return "\n\n".join(lines) if lines else "No results."
        except Exception as e:
            return f"Serper search failed: {e}"

    # DuckDuckGo Instant Answer API (no key)
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
            parts.append(f"Abstract: {data['AbstractText']}\nSource: {data.get('AbstractURL', '')}")
        for topic in (data.get("RelatedTopics") or [])[:num]:
            if isinstance(topic, dict) and topic.get("Text"):
                parts.append(f"• {topic['Text']}")
            elif isinstance(topic, dict) and "Topics" in topic:
                for t in topic["Topics"][:2]:
                    if t.get("Text"):
                        parts.append(f"• {t['Text']}")
        if parts:
            return "\n".join(parts)
    except Exception as e:
        print(f"[web_search DDG] {e}")

    # Last resort: very light HTML scrape of DDG lite
    try:
        r = requests.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
            timeout=12,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MojoBot/1.0)"},
        )
        # crude extract of result snippets
        texts = re.findall(r"<a rel=\"nofollow\"[^>]*>([^<]+)</a>", r.text)
        snippets = re.findall(r'class="result-snippet"[^>]*>([^<]+)', r.text)
        lines = []
        for i, t in enumerate(texts[:num]):
            sn = snippets[i] if i < len(snippets) else ""
            lines.append(f"• {t.strip()}\n  {sn.strip()}")
        return "\n\n".join(lines) if lines else "No results found."
    except Exception as e:
        return f"Web search unavailable right now ({e})."


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
    max_chars = min(int(args.get("max_chars") or 6000), 15000)
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
    if not _OWNER_SENDER_ID or str(ctx.get("sender_id")) != str(_OWNER_SENDER_ID):
        # also allow if sender_num matches owner
        owner = _OWNER_SENDER_ID
        if owner and str(ctx.get("sender_num") or "") != str(owner):
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


TOOL_EXECUTORS: Dict[str, Callable[[dict, dict], str]] = {
    "search_knowledge": _tool_search_knowledge,
    "web_search": _tool_web_search,
    "browse_url": _tool_browse_url,
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


# late import for time used in set_reminder
import time  # noqa: E402
