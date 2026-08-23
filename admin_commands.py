"""
admin_commands.py — control plane for the WhatsApp agent.

Kept separate from whatsapp_agent.py on purpose: this is a distinct concern
(who's allowed to do what) from message/media/AI handling, and keeping it
apart means future edits to one rarely collide with edits to the other.

Design choices, and why:
- Commands are plain deterministic "/command args" syntax, NOT LLM-parsed.
  A kill switch has to work even if Groq and Gemini are both down — it can't
  depend on the thing it might need to turn off.
- Two toggle layers: a single global "agent" switch (defaults ON), and
  per-chat, per-feature flags (default OFF, as requested) that don't need a
  schema change to add a new feature — just a name.
- Authorization = sender matches OWNER_SENDER_ID AND the message is in the
  private chat with the bot, never a group. Admin commands are checked BEFORE
  the global kill switch and bypass it, or disabling the agent would lock out
  the one command that re-enables it.

Setup:
1. Run the SQL in admin_schema.sql once in Supabase.
2. Add OWNER_SENDER_ID to your .env (see below for how to find your own ID).
3. In whatsapp_agent.py: `import admin_commands` near the top, then
   `admin_commands.init(supabase, client, get_contacts_map)` right after
   `client = NewClient(...)` is created.
"""

import os
import re

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

COMMANDS = {}

_supabase = None
_client = None
_get_contacts_map = None


def init(supabase_client, whatsapp_client, contacts_map_fn):
    """Call once, right after the WhatsApp client is created."""
    global _supabase, _client, _get_contacts_map
    _supabase = supabase_client
    _client = whatsapp_client
    _get_contacts_map = contacts_map_fn


def command(name, help_text=""):
    """Decorator that registers a new admin command."""
    def decorator(func):
        COMMANDS[name] = {"handler": func, "help": help_text}
        return func
    return decorator


def is_admin_message(sender_id, is_group):
    """Sender is configured owner AND message is in private chat with the bot."""
    return bool(OWNER_SENDER_ID) and (not is_group) and sender_id == OWNER_SENDER_ID


def resolve_chat_id(target):
    """Resolves phone numbers, group names, or 'all'/'*' into proper WhatsApp chat_ids."""
    target_str = str(target).strip()
    
    # 1. Global target
    if target_str.lower() in ("all", "*"):
        return "*"
    
    # 2. Already a valid ID
    if "@s.whatsapp.net" in target_str or "@g.us" in target_str:
        return target_str

    # 3. Search Groups by exact/case-insensitive name
    if _client:
        try:
            for g in _client.get_joined_groups():
                # Handle potential protobuf byte/string issues and invisible spaces
                g_name = getattr(g.GroupName, "Name", "")
                if isinstance(g_name, bytes):
                    g_name = g_name.decode("utf-8", errors="ignore")
                
                clean_name = str(g_name).strip().replace('\xa0', ' ').lower()
                if clean_name == target_str.lower():
                    return f"{g.JID.User}@{g.JID.Server}"
        except Exception as e:
            print(f"Error resolving group name: {e}")

    # 4. Search Contacts by display name or nickname
    if _get_contacts_map:
        try:
            res = _get_contacts_map()
            contacts_map = res[0] if isinstance(res, tuple) else res
            for s_id, name in contacts_map.items():
                if name and str(name).strip().lower() == target_str.lower():
                    return s_id if "@" in s_id else f"{s_id}@s.whatsapp.net"
        except Exception as e:
            print(f"Error resolving contact name: {e}")

    # 5. Clean phone numbers (Only if it's an actual number format)
    clean_num = re.sub(r'\D', '', target_str)
    if clean_num and len(clean_num) > 7:
        return f"{clean_num}@s.whatsapp.net"

    # 6. Failed to resolve
    return None

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


# --- Commands ---

def _handle_enable_disable(args, enabled):
    parts = args.split(maxsplit=1)
    if not parts:
        return "Usage: /enable agent   OR   /enable <feature|all> <phone|group_name|all>"
    target_feature = parts[0].lower()

    if target_feature == "agent":
        set_global_setting("agent_enabled", enabled)
        return f"{'✅' if enabled else '🛑'} Agent {'enabled' if enabled else 'disabled'} everywhere."

    if target_feature != "all" and target_feature not in KNOWN_FEATURES:
        return f"Unknown feature '{target_feature}'. Known features: {', '.join(KNOWN_FEATURES)} or 'all'"

    if len(parts) < 2:
        return f"Usage: /{'enable' if enabled else 'disable'} {target_feature} <phone|group_name|all>. Try /chats to see chats."

    raw_target = parts[1].strip()
    chat_target = resolve_chat_id(raw_target)

    # Halt execution and warn if the name didn't map to a real group/contact
    if chat_target is None:
        return f"❌ Could not find a group or contact named '{raw_target}'. Make sure the bot is added to the group and the name is spelled exactly right."

    features_to_toggle = KNOWN_FEATURES if target_feature == "all" else [target_feature]

    for feat in features_to_toggle:
        set_feature_enabled(chat_target, feat, enabled)

    where = "ALL chats" if chat_target == "*" else f"'{raw_target}' ({chat_target})"
    feat_str = "ALL features" if target_feature == "all" else f"'{target_feature}'"
    return f"{'✅' if enabled else '🛑'} {feat_str} {'enabled' if enabled else 'disabled'} for {where}."


@command("enable", "Usage: /enable agent   OR   /enable <feature|all> <phone|group_name|all>")
def cmd_enable(args):
    return _handle_enable_disable(args, True)


@command("disable", "Usage: /disable agent   OR   /disable <feature|all> <phone|group_name|all>")
def cmd_disable(args):
    return _handle_enable_disable(args, False)


@command("status", "Usage: /status [phone|group_name|chat_id|all] — shows agent state and feature flags.")
def cmd_status(args):
    agent_on = get_global_setting("agent_enabled", default=True)
    lines = [f"Agent Global Switch: {'🟢 ON' if agent_on else '🔴 OFF'}"]
    target = args.strip()
    if target:
        resolved_id = resolve_chat_id(target)
        
        # Explicit error mapping if it couldn't be resolved
        if resolved_id is None:
            return f"❌ Could not find a group or contact named '{target}'. Make sure the bot is added to the group and the name is spelled exactly right."
            
        lines.append(f"\nFeature flags for {target} ({resolved_id}):")
        for feature in KNOWN_FEATURES:
            state = "🟢 ON" if is_feature_enabled(resolved_id, feature) else "🔴 OFF"
            lines.append(f"- {feature}: {state}")
    else:
        lines.append("\nPass a phone number, group name, or 'all' to view feature flags (e.g. /status 923001234567, /status My Group, or /status all).")
    return "\n".join(lines)


@command("chats", "Lists known groups and private chats with chat_id.")
def cmd_chats(args):
    lines = ["GROUPS:"]
    try:
        for g in _client.get_joined_groups():
            chat_id = f"{g.JID.User}@{g.JID.Server}"
            lines.append(f"- {g.GroupName.Name} → {chat_id}")
        if len(lines) == 1:
            lines.append("(none)")
    except Exception as e:
        lines.append(f"(couldn't fetch groups: {e})")

    lines.append("\nPRIVATE CHATS SEEN RECENTLY:")
    try:
        response = _supabase.table("chat_history").select("chat_id").execute()
        seen = sorted(set(r["chat_id"] for r in response.data if "@g.us" not in r["chat_id"]))
        res = _get_contacts_map() if _get_contacts_map else {}
        contacts = res[0] if isinstance(res, tuple) else res
        if not seen:
            lines.append("(none)")
        for chat_id in seen:
            bare_id = chat_id.split("@")[0]
            name = contacts.get(bare_id, bare_id)
            lines.append(f"- {name} → {chat_id}")
    except Exception as e:
        lines.append(f"(couldn't fetch private chats: {e})")

    return "\n".join(lines)


@command("help", "Lists all available commands.")
def cmd_help(args):
    lines = ["Available admin commands:"]
    for name, info in COMMANDS.items():
        lines.append(f"/{name} — {info['help']}")
    return "\n".join(lines)