"""
admin_commands.py — control plane for the WhatsApp agent.

Kept separate from whatsapp_agent.py on purpose: this is a distinct concern
(who's allowed to do what) from message/media/AI handling, and keeping it
apart means future edits to one rarely collide with edits to the other.
"""

import os
import re
import unicodedata

OWNER_SENDER_ID = os.getenv("OWNER_SENDER_ID")  # bare number/LID, no @server suffix

KNOWN_FEATURES = [
    "ai_chat",      # General conversational LLM text responses
    "reminders",    # Setting, listing, cancelling, and receiving reminders
    "chat_memory",  # Permanent chat instructions/memory ("always remember")
    "documents",    # Reading uploaded documents/PDFs
    "images",       # Reading images and stickers
    "videos",       # Processing videos and GIFs
    "audio"         # Voice note transcription and responses
]

# Agent tools (ReAct loop). Stored in feature_flags as "tool:<name>".
# DEFAULT OFF for every chat (including owner and global "*") until explicitly enabled.
# Owner-only tools still require OWNER_SENDER_ID even when the flag is ON.
KNOWN_TOOLS = [
    "search_knowledge",
    "web_search",
    "browse_url",
    "get_weather",
    "get_memory",
    "save_memory",
    "set_reminder",
    "list_reminders",
    "cancel_reminders",
    "query_contacts",
    "lookup_user",
    "send_message_to",   # owner-only at executor
    "file_list_onedrive",
    "python_exec",       # owner-only at executor
    "note_down",         # owner-only at executor
    "transcribe_video",
    "link_preview",      # title/caption/author without ASR
    "image_describe",    # vision caption; no OneDrive write
]

OWNER_ONLY_TOOLS = frozenset({
    "send_message_to",
    "python_exec",
    "note_down",
})

def _tool_flag_key(tool_name: str) -> str:
    return f"tool:{tool_name}"

COMMANDS = {}

_supabase = None
_client = None
_get_contacts_map = None
_ai_client = None
_model_name = None

def init(supabase_client, whatsapp_client, contacts_map_fn, ai_client=None, model_name=None):
    """Call once, right after the WhatsApp client is created."""
    global _supabase, _client, _get_contacts_map, _ai_client, _model_name
    _supabase = supabase_client
    _client = whatsapp_client
    _get_contacts_map = contacts_map_fn
    _ai_client = ai_client
    _model_name = model_name


def command(name, help_text=""):
    """Decorator that registers a new admin command."""
    def decorator(func):
        COMMANDS[name] = {"handler": func, "help": help_text}
        return func
    return decorator


def is_admin_message(sender_id, is_group):
    """Sender is configured owner AND message is in private chat with the bot."""
    return bool(OWNER_SENDER_ID) and (not is_group) and sender_id == OWNER_SENDER_ID


def _group_display_name(g) -> str:
    """Extract group name safely handling string, byte types, and custom wrappers."""
    try:
        gn = getattr(g, "GroupName", None) or getattr(g, "group_name", None)
        if gn is not None:
            val = getattr(gn, "Name", None) or getattr(gn, "name", None) or gn
            if isinstance(val, bytes):
                return val.decode("utf-8", errors="ignore").strip()
            v_str = str(val).strip()
            if v_str:
                return v_str
                
        for attr in ("Name", "name", "Subject", "subject"):
            val = getattr(g, attr, None)
            if val and not callable(val):
                if isinstance(val, bytes):
                    return val.decode("utf-8", errors="ignore").strip()
                v_str = str(val).strip()
                if v_str:
                    return v_str
    except Exception:
        pass
    return ""


def resolve_chat_id(target):
    target_str = (target or "").strip()
    if not target_str:
        return target_str
    if target_str.lower() in ("all", "*"):
        return "*"
        
    # Accept full JIDs as-is
    if "@g.us" in target_str or "@s.whatsapp.net" in target_str:
        return target_str

    debug_groups = []
    available_chats = []

    # 1. Collect Active Groups
    if _client:
        try:
            groups = list(_client.get_joined_groups() or [])
            for g in groups:
                jid = getattr(g, "JID", None)
                if not jid or not getattr(jid, "User", None):
                    continue
                
                user_id = str(jid.User)
                server = getattr(jid, "Server", None) or "g.us"
                chat_id = f"{user_id}@{server}"
                
                # Direct match for bare group numeric ID
                if target_str.isdigit() and target_str == user_id:
                    return chat_id

                name = _group_display_name(g)
                if not name:
                    continue
                    
                debug_groups.append(name)
                available_chats.append(f"Group: '{name}' -> {chat_id}")

        except Exception as e:
            print(f"Error checking groups: {e}")

    # 2. Collect Active Contacts
    if _get_contacts_map:
        try:
            res = _get_contacts_map()
            contacts_map = res[0] if isinstance(res, tuple) else res
            for s_id, name in (contacts_map or {}).items():
                if name:
                    cid = s_id if "@" in str(s_id) else f"{s_id}@s.whatsapp.net"
                    available_chats.append(f"Contact: '{name}' -> {cid}")
        except Exception as e:
            print(f"Error fetching contacts maps: {e}")

    # 3. Use LLM to intelligently match the target string
    if _ai_client and _model_name and available_chats:
        try:
            chats_list_str = "\n".join(available_chats)
            system_prompt = (
                "You are an intelligent routing assistant. "
                "I will give you a list of available WhatsApp groups and contacts with their IDs, "
                "and a user's search query.\n"
                "Your task is to find the best match for the user's query from the list.\n"
                "Reply with ONLY the exact ID (e.g., 123456@g.us or 98765@s.whatsapp.net) of the best match. "
                "Do NOT include any extra text, markdown, or punctuation.\n"
                "If the query absolutely does not match anything in the list, reply EXACTLY with: NONE"
            )
            
            user_prompt = f"AVAILABLE CHATS:\n{chats_list_str}\n\nUSER QUERY: '{target_str}'"

            response = _ai_client.chat.completions.create(
                model=_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )
            
            llm_match = response.choices[0].message.content.strip()
            
            if llm_match != "NONE" and "@" in llm_match:
                return llm_match
        except Exception as e:
            print(f"LLM match failed, falling back: {e}")

    # 4. Fallback: Clean phone numbers
    clean_num = re.sub(r"\D", "", target_str)
    if len(clean_num) >= 8:
        return f"{clean_num}@s.whatsapp.net"

    # 5. Nothing matched - Highly descriptive error
    err = f"Could not find '{target_str}'."
    if debug_groups:
        sample = ", ".join(f"'{x}'" for x in debug_groups[:6])
        err += f" Groups I checked: {sample}."
    else:
        err += " (No joined groups found in memory right now)."
    err += " Paste the full ID from /chats if the name isn't working."
    
    raise ValueError(err)


def handle_admin_command(text_content):
    """Parses '/command args' and dispatches to registered handler."""
    text_content = text_content.strip()
    if not text_content.startswith("/"):
        return None
    parts = text_content[1:].split(maxsplit=1)
    if not parts:
        return None
    cmd_name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    if cmd_name not in COMMANDS:
        return f"Unknown command '/{cmd_name}'. Try /help."
    try:
        return COMMANDS[cmd_name]["handler"](args)
    except Exception as e:
        print(f"Error running admin command '{cmd_name}': {e}")
        return f"Something went wrong running that: {e}"


# --- Global settings ---

def get_global_setting(key, default=True):
    try:
        response = _supabase.table("bot_settings").select("value").eq("key", key).execute()
        if response.data:
            return bool(response.data[0]["value"])
    except Exception as e:
        print(f"Error reading setting {key}: {e}")
    return default


def set_global_setting(key, value):
    try:
        _supabase.table("bot_settings").upsert({"key": key, "value": value}).execute()
    except Exception as e:
        print(f"Error saving setting {key}: {e}")


# --- Per-chat feature flags ---

def is_feature_enabled(chat_id, feature):
    """Checks chat_id+feature row first. If owner private chat, defaults to ON.
    Everywhere else: defaults to OFF unless '*' (all chats) row exists."""
    try:
        response = _supabase.table("feature_flags").select("enabled") \
            .eq("chat_id", chat_id).eq("feature", feature).execute()
        if response.data:
            return bool(response.data[0]["enabled"])

        if OWNER_SENDER_ID and chat_id.split("@")[0] == OWNER_SENDER_ID:
            return True

        response = _supabase.table("feature_flags").select("enabled") \
            .eq("chat_id", "*").eq("feature", feature).execute()
        if response.data:
            return bool(response.data[0]["enabled"])
    except Exception as e:
        print(f"Error checking feature flag {feature} for {chat_id}: {e}")
    return False


def has_any_feature_enabled(chat_id):
    """Returns True if at least one feature in KNOWN_FEATURES is enabled for this chat."""
    return any(is_feature_enabled(chat_id, feat) for feat in KNOWN_FEATURES)


def set_feature_enabled(chat_id, feature, enabled):
    try:
        _supabase.table("feature_flags").upsert({
            "chat_id": chat_id,
            "feature": feature,
            "enabled": enabled
        }).execute()
    except Exception as e:
        print(f"Error saving feature flag {feature} for {chat_id}: {e}")


# --- Agent tool flags ---
# Non-owner chats: default OFF (explicit enable required).
# Owner private chat: default ON for every known tool (and any future entry in
# KNOWN_TOOLS) unless an explicit row disables it — same policy as features.

def is_tool_enabled(chat_id, tool_name: str) -> bool:
    """Per-chat tool flag.

    Resolution order:
      1. Exact (chat_id, tool:<name>) row
      2. Owner private chat → True (admin gets all tools + future additions ON)
      3. Global (*, tool:<name>) row
      4. False

    Owner-only tools still require OWNER_SENDER_ID at executor time; this flag
    only controls whether the tool appears in the agent schema / force-path.
    """
    if tool_name not in KNOWN_TOOLS:
        return False
    key = _tool_flag_key(tool_name)
    try:
        response = _supabase.table("feature_flags").select("enabled") \
            .eq("chat_id", chat_id).eq("feature", key).execute()
        if response.data:
            return bool(response.data[0]["enabled"])

        # Admin / owner private chat: all tools and future KNOWN_TOOLS entries ON
        if OWNER_SENDER_ID and chat_id and chat_id.split("@")[0] == OWNER_SENDER_ID:
            return True

        response = _supabase.table("feature_flags").select("enabled") \
            .eq("chat_id", "*").eq("feature", key).execute()
        if response.data:
            return bool(response.data[0]["enabled"])
    except Exception as e:
        print(f"Error checking tool flag {tool_name} for {chat_id}: {e}")
    return False


def set_tool_enabled(chat_id, tool_name: str, enabled: bool) -> None:
    if tool_name not in KNOWN_TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}")
    set_feature_enabled(chat_id, _tool_flag_key(tool_name), enabled)


def get_enabled_tools(chat_id) -> list:
    """Return list of tool names enabled for this chat (order = KNOWN_TOOLS)."""
    return [t for t in KNOWN_TOOLS if is_tool_enabled(chat_id, t)]


# --- Commands ---

def _handle_tool_enable_disable(rest: str, enabled: bool) -> str:
    """Handle: tool <name|all> <phone|group|all>"""
    parts = rest.split(maxsplit=1)
    if not parts:
        return (
            f"Usage: /{'enable' if enabled else 'disable'} tool <tool_name|all> "
            f"<phone|group_name|all>\nKnown tools: {', '.join(KNOWN_TOOLS)}"
        )
    target_tool = parts[0].lower().strip()
    if target_tool not in ("all",) and target_tool not in KNOWN_TOOLS:
        return (
            f"Unknown tool '{target_tool}'. Known tools: {', '.join(KNOWN_TOOLS)} or 'all'"
        )
    if len(parts) < 2:
        return (
            f"Usage: /{'enable' if enabled else 'disable'} tool {target_tool} "
            f"<phone|group_name|all>. Try /chats to see chats."
        )
    raw_target = parts[1].strip()
    try:
        chat_target = resolve_chat_id(raw_target)
    except ValueError as e:
        return f"❌ {e}"
    if chat_target is None:
        return f"❌ Could not find a group or contact named '{raw_target}'."

    tools_to_toggle = KNOWN_TOOLS if target_tool == "all" else [target_tool]
    for t in tools_to_toggle:
        set_tool_enabled(chat_target, t, enabled)

    where = "ALL chats" if chat_target == "*" else f"'{raw_target}' ({chat_target})"
    tool_str = "ALL tools" if target_tool == "all" else f"tool '{target_tool}'"
    return f"{'✅' if enabled else '🛑'} {tool_str} {'enabled' if enabled else 'disabled'} for {where}."


def _handle_enable_disable(args, enabled):
    parts = args.split(maxsplit=1)
    if not parts:
        return (
            "Usage:\n"
            "  /enable agent\n"
            "  /enable <feature|all> <phone|group_name|all>\n"
            "  /enable tool <tool_name|all> <phone|group_name|all>\n"
            f"Features: {', '.join(KNOWN_FEATURES)}\n"
            f"Tools: {', '.join(KNOWN_TOOLS)}"
        )
    target_feature = parts[0].lower()

    if target_feature == "agent":
        set_global_setting("agent_enabled", enabled)
        return f"{'✅' if enabled else '🛑'} Agent {'enabled' if enabled else 'disabled'} everywhere."

    # Tool path: /enable tool <name|all> <chat|all>
    if target_feature in ("tool", "tools"):
        rest = parts[1] if len(parts) > 1 else ""
        return _handle_tool_enable_disable(rest, enabled)

    if target_feature != "all" and target_feature not in KNOWN_FEATURES:
        return (
            f"Unknown feature '{target_feature}'. "
            f"Known features: {', '.join(KNOWN_FEATURES)} or 'all'. "
            f"For agent tools use: /{'enable' if enabled else 'disable'} tool <name|all> <chat|all>"
        )

    if len(parts) < 2:
        return f"Usage: /{'enable' if enabled else 'disable'} {target_feature} <phone|group_name|all>. Try /chats to see chats."

    raw_target = parts[1].strip()
    try:
        chat_target = resolve_chat_id(raw_target)
    except ValueError as e:
        return f"❌ {e}"

    if chat_target is None:
        return f"❌ Could not find a group or contact named '{raw_target}'."

    features_to_toggle = KNOWN_FEATURES if target_feature == "all" else [target_feature]

    for feat in features_to_toggle:
        set_feature_enabled(chat_target, feat, enabled)

    where = "ALL chats" if chat_target == "*" else f"'{raw_target}' ({chat_target})"
    feat_str = "ALL features" if target_feature == "all" else f"'{target_feature}'"
    return f"{'✅' if enabled else '🛑'} {feat_str} {'enabled' if enabled else 'disabled'} for {where}."


@command(
    "enable",
    "Usage: /enable agent | /enable <feature|all> <chat|all> | /enable tool <tool|all> <chat|all>",
)
def cmd_enable(args):
    return _handle_enable_disable(args, True)


@command(
    "disable",
    "Usage: /disable agent | /disable <feature|all> <chat|all> | /disable tool <tool|all> <chat|all>",
)
def cmd_disable(args):
    return _handle_enable_disable(args, False)


@command(
    "status",
    "Usage: /status [phone|group_name|chat_id|all] — agent, features, and tools.",
)
def cmd_status(args):
    agent_on = get_global_setting("agent_enabled", default=True)
    lines = [f"Agent Global Switch: {'🟢 ON' if agent_on else '🔴 OFF'}"]
    target = args.strip()
    if target:
        try:
            resolved_id = resolve_chat_id(target)
        except ValueError as e:
            return f"❌ {e}"

        lines.append(f"\nFeature flags for {target} ({resolved_id}):")
        for feature in KNOWN_FEATURES:
            state = "🟢 ON" if is_feature_enabled(resolved_id, feature) else "🔴 OFF"
            lines.append(f"- {feature}: {state}")

        lines.append(
            f"\nTool flags for {target} ({resolved_id}) "
            "(owner private: default ON; others: default OFF):"
        )
        for tool in KNOWN_TOOLS:
            state = "🟢 ON" if is_tool_enabled(resolved_id, tool) else "🔴 OFF"
            owner_tag = " (owner-only)" if tool in OWNER_ONLY_TOOLS else ""
            lines.append(f"- {tool}: {state}{owner_tag}")
    else:
        lines.append(
            "\nPass a phone number, group name, or 'all' to view feature + tool flags.\n"
            "Policy: owner private chat → features+tools default ON; "
            "all other chats → default OFF until /enable.\n"
            f"Tools: {', '.join(KNOWN_TOOLS)}"
        )
    return "\n".join(lines)


@command("chats", "Lists known groups and private chats with chat_id.")
def cmd_chats(args):
    lines = ["GROUPS:"]
    try:
        groups = list(_client.get_joined_groups() or [])
        if not groups:
            lines.append("(none)")
        for g in groups:
            name = _group_display_name(g) or "?"
            jid = getattr(g, "JID", None)
            if jid and getattr(jid, "User", None):
                chat_id = f"{jid.User}@{getattr(jid, 'Server', None) or 'g.us'}"
                lines.append(f"- {name} → {chat_id}")
            else:
                lines.append(f"- {name} → (no jid)")
    except Exception as e:
        lines.append(f"(couldn't fetch groups: {e})")

    lines.append("\nPRIVATE CHATS / CONTACTS:")
    # phone_digits -> display name  (one row per real number)
    by_phone = {}

    def _looks_like_phone(s: str) -> bool:
        d = re.sub(r"\D", "", s or "")
        # real WA numbers are usually 8–15 digits; LIDs are often longer
        return 8 <= len(d) <= 15

    def _remember_phone(phone: str, label: str):
        digits = re.sub(r"\D", "", phone or "")
        if not _looks_like_phone(digits):
            return
        label = (label or "").strip() or digits
        prev = by_phone.get(digits)
        # Prefer a real name over a bare number
        if not prev or (prev == digits and label != digits):
            by_phone[digits] = label

    # --- A) Contacts table (best names + sender_num) ---
    try:
        rows = (
            _supabase.table("contacts")
            .select("sender_id, display_name, nickname, sender_num")
            .execute()
            .data
            or []
        )
        for row in rows:
            name = (row.get("nickname") or row.get("display_name") or "").strip()
            sid = str(row.get("sender_id") or "")
            snum = str(row.get("sender_num") or "").strip()

            # Prefer explicit sender_num
            if snum and _looks_like_phone(snum):
                _remember_phone(snum, name or snum)
                continue

            # sender_id is already a phone (no @lid)
            bare = sid.split("@")[0] if sid else ""
            if "@lid" not in sid.lower() and _looks_like_phone(bare):
                _remember_phone(bare, name or bare)
    except Exception as e:
        lines.append(f"(contacts fetch failed: {e})")

    # --- B) chat_history: only phone-shaped chat_ids ---
    try:
        page_size = 1000
        start = 0
        while True:
            response = (
                _supabase.table("chat_history")
                .select("chat_id, sender_num")
                .range(start, start + page_size - 1)
                .execute()
            )
            rows = response.data or []
            if not rows:
                break
            for r in rows:
                cid = (r.get("chat_id") or "").strip()
                snum = (r.get("sender_num") or "").strip()

                if snum and _looks_like_phone(snum):
                    # Try to name from contacts map
                    label = None
                    try:
                        res = _get_contacts_map() if _get_contacts_map else ({}, {})
                        cmap = res[0] if isinstance(res, tuple) else res
                        bare = re.sub(r"\D", "", snum)
                        label = (cmap or {}).get(bare) or (cmap or {}).get(snum)
                    except Exception:
                        pass
                    _remember_phone(snum, label or snum)
                    continue

                if not cid or "@g.us" in cid or cid == "status@broadcast":
                    continue
                # Only accept …@s.whatsapp.net (skip pure @lid)
                if cid.endswith("@lid"):
                    continue
                bare = cid.split("@")[0]
                if _looks_like_phone(bare):
                    label = None
                    try:
                        res = _get_contacts_map() if _get_contacts_map else ({}, {})
                        cmap = res[0] if isinstance(res, tuple) else res
                        label = (cmap or {}).get(bare) or (cmap or {}).get(cid)
                    except Exception:
                        pass
                    _remember_phone(bare, label or bare)

            if len(rows) < page_size:
                break
            start += page_size
            if start > 20000:
                break
    except Exception as e:
        lines.append(f"(chat_history fetch failed: {e})")

    if not by_phone:
        lines.append("(none)")
    else:
        for phone, name in sorted(by_phone.items(), key=lambda x: (x[1] or "").casefold()):
            lines.append(f"- {name} → {phone}")

    return "\n".join(lines)

@command("exportlog", "Usage: /exportlog [n] — last n msgs (all chats) → TXT → OneDrive (default n=200, max 5000)")
def cmd_exportlog(args):
    n = 200
    if args:
        try:
            n = int(args.strip().split()[0])  # full number, not first char
        except (ValueError, IndexError):
            return "Usage: /exportlog [n]  e.g. /exportlog 500"
    n = max(1, min(n, 5000))
    import whatsapp_agent as wa
    return wa.export_chat_log_to_onedrive(n)


@command(
    "uploadsession",
    "Usage: /uploadsession — force-upload WhatsApp session DB to Supabase Storage now "
    "(bypasses change-detection). Use before a manual deploy if you want the latest keys saved.",
)
def cmd_uploadsession(args):
    try:
        import whatsapp_agent as wa
        result = wa._upload_session_to_bucket(force=True)
        if result.get("skipped") and result.get("reason") == "local_mode":
            return "Local mode — session uploads are disabled."
        if result.get("skipped") and result.get("reason") == "no_local_file":
            return "No local session file found."
        if not result.get("ok"):
            return f"Upload failed: {result.get('error') or result}"
        kb = (result.get("bytes") or 0) / 1024
        return f"✅ Session uploaded to Supabase ({kb:.1f} KB, forced)."
    except Exception as e:
        return f"Upload failed: {e}"


@command(
    "sessionstatus",
    "Usage: /sessionstatus — local session file size, last upload fingerprint, interval.",
)
def cmd_sessionstatus(args):
    try:
        import whatsapp_agent as wa
        st = wa.get_session_upload_status()
        kb = (st.get("bytes") or 0) / 1024
        last_b = st.get("last_upload_bytes")
        last_kb = f"{last_b / 1024:.1f} KB" if last_b is not None else "never"
        lines = [
            f"Local mode: {st.get('local_mode')}",
            f"Path: {st.get('path')} ({'exists' if st.get('exists') else 'missing'})",
            f"Size: {kb:.1f} KB",
            f"Last uploaded size: {last_kb}",
            f"Interval: {st.get('interval_sec')}s (change-detected)",
            f"Bucket: {st.get('bucket')}",
        ]
        # Linked-device 30-day inactivity keepalive (Presence.AVAILABLE)
        try:
            pr = wa.get_presence_keepalive_status()
            ago = pr.get("last_sent_ago_sec")
            ago_s = f"{ago:.0f}s ago" if ago is not None else "never"
            lines.append(
                f"Presence keepalive: "
                f"{'ON' if pr.get('enabled') else 'OFF'} "
                f"every {pr.get('interval_hours')}h | "
                f"started={pr.get('started')} | last={ago_s}"
            )
        except Exception as _pe:
            lines.append(f"Presence keepalive: (status error: {_pe})")
        return "\n".join(lines)
    except Exception as e:
        return f"sessionstatus failed: {e}"


@command(
    "uploadlog",
    "Usage: /uploadlog [yesterday] — upload current local agent log to OneDrive "
    "(mojo_agent_live.log). Optional 'yesterday' also uploads prior day's rotated file. "
    "No auto-sync; run this when you want the file on OneDrive.",
)
def cmd_uploadlog(args):
    """Owner-only (admin command path). Pushes local log on demand — stops bandwidth burn."""
    include_yesterday = bool(args and "yesterday" in args.strip().lower())
    try:
        import mojo_logging
        import file_ops

        # Ensure file handler is live even if module-level setup was skipped
        mojo_logging.setup_logging()

        if not file_ops.onedrive_configured():
            return "OneDrive not configured (missing ONEDRIVE_CLIENT_ID / ONEDRIVE_REFRESH_TOKEN)."

        result = mojo_logging.upload_current_log(
            file_ops,
            include_yesterday=include_yesterday,
        )
        kb = (result.get("bytes") or 0) / 1024
        lines = [
            f"✅ Log uploaded → OneDrive/{result['remote_folder']}/{result['remote_name']}",
            f"Size: {kb:.1f} KB | day: {result.get('day')}",
        ]
        if result.get("empty"):
            lines.append(
                "Note: local log was empty (process just started or free-tier disk was wiped)."
            )
        lines.append(
            "Note: Render Free has no persistent disk — local logs only cover this process uptime."
        )
        if result.get("webUrl"):
            lines.append(f"Link: {result['webUrl']}")
        if result.get("yesterday"):
            y = result["yesterday"]
            lines.append(
                f"Also uploaded yesterday: {y.get('remote_name')} "
                f"({(y.get('bytes') or 0) / 1024:.1f} KB)"
            )
        if result.get("yesterday_error"):
            lines.append(f"(yesterday skip: {result['yesterday_error']})")
        return "\n".join(lines)
    except FileNotFoundError as e:
        return (
            f"No local log file yet.\n{e}\n"
            "Tip: free-tier /tmp is wiped on restart. Wait for the bot to log something, "
            "then try /uploadlog again. Use /logstatus to check."
        )
    except Exception as e:
        return f"Upload failed: {e}"



@command(
    "logstatus",
    "Usage: /logstatus — show local agent log size, current day, and any rotated files.",
)
def cmd_logstatus(args):
    try:
        import mojo_logging

        mojo_logging.setup_logging()
        st = mojo_logging.get_log_status()
        kb = (st.get("bytes") or 0) / 1024
        lines = [
            f"Local log: {st.get('path')}",
            f"Size: {kb:.1f} KB | day (UTC): {st.get('day')}",
            f"Remote name on /uploadlog: {st.get('remote_name')}",
            "Auto OneDrive sync: OFF (on-demand only)",
            "Storage: ephemeral on Render Free (/tmp wiped on restart/spin-down/deploy)",
        ]
        dated = st.get("dated") or []
        if dated:
            lines.append("Rotated (local, this process only):")
            for d in dated:
                lines.append(f"  - {d['name']}: {d['bytes'] / 1024:.1f} KB")
        else:
            lines.append("No rotated day files yet (normal after restart on free tier).")
        return "\n".join(lines)
    except Exception as e:
        return f"logstatus failed: {e}"



@command("uploadimg", "Usage: send or reply to an image with /uploadimg [optional_name] — uploads to OneDrive")
def cmd_uploadimg(args):
    # Real work is done in process_message (needs media). This is only for /help + text-only misuse.
    return (
        "Send an image with caption /uploadimg, or reply to an image with /uploadimg.\n"
        "Optional: /uploadimg my_screenshot"
    )

@command(
    "ingest",
    "Usage: /ingest agency | onedrive [subfolder] | file <local-or-onedrive-path> | status\n"
    "Seeds RAG from business_info.txt, walks Documents/aimojo, or ingests one path.",
)
def cmd_ingest(args):
    """Owner-only RAG ingestion control plane."""
    try:
        import knowledge_rag as kr
    except Exception as e:
        return f"knowledge_rag import failed: {e}"

    raw = (args or "").strip()
    parts = raw.split(None, 1)
    mode = (parts[0] if parts else "status").lower()
    rest = parts[1].strip() if len(parts) > 1 else ""

    if mode in ("status", "kbstatus", ""):
        return cmd_kbstatus("")

    if mode == "agency":
        result = kr.ingest_agency_business_info()
        return _format_ingest_result("agency", result)

    if mode in ("onedrive", "od", "aimojo", "tree"):
        sub = rest.strip().strip("/")
        # Convenience: /ingest onedrive docs → Documents/aimojo/docs
        result = kr.ingest_onedrive_knowledge_tree(subfolder=sub)
        lines = [
            f"OneDrive root: {result.get('root')}/{result.get('subfolder')}",
            f"Scanned: {result.get('scanned')} | ingested: {result.get('ingested')} | "
            f"unchanged: {result.get('unchanged')}",
        ]
        errs = result.get("errors") or []
        if errs:
            lines.append(f"Errors ({len(errs)} shown):")
            for e in errs[:5]:
                lines.append(f"  - {e.get('remote') or e.get('path')}: {e.get('error')}")
        return "\n".join(lines)

    if mode in ("file", "path"):
        if not rest:
            return "Usage: /ingest file <Documents/aimojo/...> or local path"
        path = rest.strip().strip('"').strip("'")
        # Prefer OneDrive if path looks remote
        if path.startswith("Documents/") or path.startswith("onedrive:"):
            path = path.replace("onedrive:", "", 1)
            result = kr.ingest_onedrive_path(path)
        elif os.path.isfile(path):
            result = kr.ingest_local_file(path)
        else:
            # Try as OneDrive path under knowledge root
            result = kr.ingest_onedrive_path(path)
        return _format_ingest_result(path, result)

    if mode == "ensure":
        result = kr.ensure_onedrive_knowledge_tree()
        if result.get("ok"):
            return (
                "OneDrive knowledge tree ready:\n"
                + "\n".join(f"  - {f}" for f in (result.get("folders") or []))
            )
        return f"Failed: {result.get('error')}"

    return (
        "Usage:\n"
        "  /ingest agency          — seed from business_info.txt (+ mirror to OneDrive)\n"
        "  /ingest onedrive [sub]  — walk Documents/aimojo[/sub]\n"
        "  /ingest file <path>     — one local or OneDrive path\n"
        "  /ingest ensure          — create Documents/aimojo/{agency,docs,chats}\n"
        "  /ingest status          — same as /kbstatus"
    )


def _format_ingest_result(label: str, result: dict) -> str:
    status = result.get("status")
    if status == "ok":
        lines = [
            f"✅ Ingested {label}",
            f"document_id: {result.get('document_id')}",
            f"chunks: {result.get('chunks')} | version: {result.get('source_version')}",
        ]
        if result.get("onedrive_mirror"):
            lines.append(f"OneDrive mirror: {result['onedrive_mirror']}")
        if result.get("onedrive_mirror_error"):
            lines.append(f"(mirror warn: {result['onedrive_mirror_error']})")
        return "\n".join(lines)
    if status == "unchanged":
        return f"Unchanged (same content hash): {label} → {result.get('document_id')}"
    if status == "skipped":
        return f"Skipped {label}: {result.get('reason')}"
    return f"Failed {label}: {result.get('error') or result}"


@command(
    "kbstatus",
    "Usage: /kbstatus — RAG schema readiness, document/chunk counts, embedding model, OneDrive root.",
)
def cmd_kbstatus(args):
    try:
        import knowledge_rag as kr

        st = kr.kb_status()
        lines = [
            f"Schema ready: {st.get('schema_ready')}",
            f"Embedding: {st.get('embedding_provider')} / {st.get('embedding_model')} "
            f"(dim={st.get('embedding_dim')})",
            f"OneDrive configured: {st.get('onedrive_configured')}",
            f"Knowledge root: {st.get('onedrive_root')}",
        ]
        if st.get("schema_ready"):
            lines.append(f"Documents: {st.get('documents')} | Chunks: {st.get('chunks')}")
            by = st.get("by_source_type") or {}
            if by:
                lines.append(
                    "By source_type: "
                    + ", ".join(f"{k}={v}" for k, v in sorted(by.items()))
                )
        else:
            lines.append(
                "→ Run sql/rag_schema.sql in Supabase SQL editor, then /ingest agency"
            )
        if st.get("count_error"):
            lines.append(f"Count error: {st['count_error']}")
        return "\n".join(lines)
    except Exception as e:
        return f"kbstatus failed: {e}"


@command("help", "Lists all available commands.")
def cmd_help(args):
    lines = ["Available admin commands:"]
    for name, info in COMMANDS.items():
        lines.append(f"/{name} — {info['help']}")
    return "\n".join(lines)