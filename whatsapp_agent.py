import os
import time
import base64
import tempfile
import re
import json
import threading
import logging
import traceback
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# 1. LOAD CONFIGURATION FIRST
load_dotenv()


class _SuppressReminderPolling(logging.Filter):
    """Filters out the httpx 'HTTP Request' lines from the reminder scheduler's
    own polling specifically — those fire every 30s forever and drown out
    everything else. Every OTHER httpx log (Supabase writes, Groq/Gemini calls,
    contacts lookups) stays visible, since those are what actually help when
    debugging a real conversation."""
    def filter(self, record):
        return "/reminders?" not in record.getMessage()


logging.getLogger("httpx").addFilter(_SuppressReminderPolling())

from openai import OpenAI, RateLimitError
from neonize.client import NewClient
from neonize.events import MessageEv
from neonize.utils import build_jid
from supabase import create_client, Client

# Import this AFTER load_dotenv() so it can see the variables
import admin_commands
import file_ops

import time as _time

def _timed(label):
    class Timer:
        def __enter__(self):
            self.start = _time.perf_counter()
            return self
        def __exit__(self, *args):
            ms = (_time.perf_counter() - self.start) * 1000
            print(f"[TIMING] {label}: {ms:.0f} ms")
    return Timer()


DEFAULT_UTC_OFFSET = 5.0  # Default to Pakistan Standard Time (+5)
DEFAULT_TIMEZONE = "Asia/Karachi"
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

RELATIVE_TIME_RE = re.compile(
    r'\b(\d+)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d)\b', re.IGNORECASE
)
CLOCK_TIME_RE = re.compile(r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b', re.IGNORECASE)
UNIT_SECONDS = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
}
TZ_OFFSET_RE = re.compile(
    r'(?:timezone|time\s*zone|utc|gmt)\s*(?:is|=|:)?\s*([+-]?\d{1,2}(?:[:.]\d{2})?)', re.IGNORECASE
)


# --- CHOOSE YOUR FREE AI PROVIDER ---

# OPTION 1: Google Gemini (Generous free tier: 1500 req/day)
# client_ai = OpenAI(
#     api_key=os.getenv("GEMINI_API_KEY"),
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )
# MODEL_NAME = "gemini-1.5-flash" 

# OPTION 2: Groq (Blazing fast Llama 3 models)
# To use Groq instead, comment out Option 1 above, and uncomment this:

client_ai = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    max_retries=0
)
MODEL_NAME = "openai/gpt-oss-120b"
WHISPER_MODEL = "whisper-large-v3-turbo"  # Groq's fast/cheap dedicated transcription model

# Gemini comes in specifically for what Groq/Llama isn't built for: reading images,
# documents, and PDFs. Groq stays the default for fast text replies below — this is
# additive, not a replacement.
client_gemini = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    max_retries=1,  # avoid nested retry storms with our own backoff
)
GEMINI_MODEL = "gemini-3.6-flash"  # current GA Flash model as of mid-2026

# Load Business Information
with open("business_info.txt", "r", encoding="utf-8") as f:
    BUSINESS_KNOWLEDGE = f.read()

# --- SUPABASE SETUP ---
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)


# --- SESSION FILE + RENDER HEALTH (minimal) ---
LOCAL_SESSION_FILE = "whatsapp_session.db"
SESSION_BUCKET = "wa-sessions"
SESSION_OBJECT = "whatsapp_session.db"

# Set once at startup. True = local testing (different number) → never upload.
IS_LOCAL_MODE = False

def _download_session_from_bucket():
    """Download session DB from Supabase Storage if it exists."""
    try:
        data = supabase.storage.from_(SESSION_BUCKET).download(SESSION_OBJECT)
        with open(LOCAL_SESSION_FILE, "wb") as f:
            f.write(data)
        print(f"[SESSION] Downloaded session from bucket '{SESSION_BUCKET}/{SESSION_OBJECT}'")
        return True
    except Exception as e:
        print(f"[SESSION] No existing session in bucket (or download failed): {e}")
        return False

def _upload_session_to_bucket():
    """Upload current local session file to Supabase Storage (upsert).
    NEVER runs in local testing mode."""
    if IS_LOCAL_MODE:
        return
    if not os.path.exists(LOCAL_SESSION_FILE):
        return
    try:
        with open(LOCAL_SESSION_FILE, "rb") as f:
            supabase.storage.from_(SESSION_BUCKET).upload(
                path=SESSION_OBJECT,
                file=f,
                file_options={"content-type": "application/octet-stream", "upsert": "true"}
            )
        print(f"[SESSION] Uploaded session to bucket '{SESSION_BUCKET}/{SESSION_OBJECT}'")
    except Exception as e:
        print(f"[SESSION] Upload failed: {e}")

def get_session_path():
    """
    Priority:
    1. Local file exists → local testing mode (different number). Never touch the bucket.
    2. No local file → Render/ephemeral mode → download from bucket if present.
    """
    global IS_LOCAL_MODE

    if os.path.exists(LOCAL_SESSION_FILE):
        IS_LOCAL_MODE = True
        print("[SESSION] Local session file found → local testing mode (uploads DISABLED)")
        return LOCAL_SESSION_FILE

    IS_LOCAL_MODE = False
    print("[SESSION] No local session file → ephemeral/Render mode, checking bucket...")
    _download_session_from_bucket()
    return LOCAL_SESSION_FILE

def start_health_server():
    """Minimal HTTP health endpoint for Render free tier + UptimeRobot."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import os

    port = int(os.environ.get("PORT", 10000))

    class HealthHandler(BaseHTTPRequestHandler):
        def _send_ok(self):
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def do_GET(self):
            if self.path in ("/", "/health", "/healthz"):
                self._send_ok()
            else:
                self.send_response(404)
                self.end_headers()

        def do_HEAD(self):
            # UptimeRobot (and many monitors) use HEAD
            if self.path in ("/", "/health", "/healthz"):
                self.send_response(200)
                self.send_header("Content-type", "text/plain")
                self.send_header("Content-Length", "2")
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            return  # silence access logs

    def run():
        server = HTTPServer(("0.0.0.0", port), HealthHandler)
        print(f"[HEALTH] Listening on 0.0.0.0:{port}  (GET/HEAD / /health /healthz)")
        server.serve_forever()

    t = threading.Thread(target=run, daemon=True)
    t.start()

def start_session_uploader(interval_seconds=300):
    """Periodically upload the session file. Skips entirely in local mode."""
    if IS_LOCAL_MODE:
        print("[SESSION] Local mode — session uploader not started")
        return

    def loop():
        while True:
            time.sleep(interval_seconds)
            _upload_session_to_bucket()
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    print(f"[SESSION] Background uploader started (every {interval_seconds}s)")

def start_self_ping(interval_seconds=600):
    """Optional self-ping (backup). Only useful on Render when RENDER_EXTERNAL_URL is set."""
    import urllib.request
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        print("[SELF-PING] RENDER_EXTERNAL_URL not set — skipping self-ping")
        return

    def loop():
        while True:
            time.sleep(interval_seconds)
            try:
                urllib.request.urlopen(url, timeout=10)
                print("[SELF-PING] ok")
            except Exception as e:
                print(f"[SELF-PING] failed: {e}")
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    print(f"[SELF-PING] Started (every {interval_seconds}s) → {url}")

# Trigger words that tell the bot to save something permanently.
# This is the simple/cheap version — see maybe_save_memory_smart() below for a
# version that catches rules that don't use any of these exact words.
MEMORY_TRIGGERS = ["always remember"]




def build_chat_debug_log_text(n: int = 200) -> str:
    n = max(1, min(int(n), 5000))
    rows = (
        supabase.table("chat_history")
        .select("created_at, chat_id, role, sender_id, sender_num, content")
        .order("created_at", desc=True)
        .limit(n)
        .execute()
        .data
        or []
    )
    rows = list(reversed(rows))
    contacts_map, _ = get_contacts_maps()

    lines = [
        f"# last {n} messages (all chats)",
        f"# generated_utc={datetime.now(timezone.utc).isoformat()}",
        "# timestamp | chat_id | role | lid | number | name | message",
    ]
    for row in rows:
        lid = row.get("sender_id") or ""
        number = row.get("sender_num") or ""
        role = row.get("role") or ""
        name = "mojo_agent" if role == "assistant" else (contacts_map.get(lid) or "-")
        msg = (row.get("content") or "").replace("\n", "\\n")
        lines.append(
            f"{row.get('created_at','')} | {row.get('chat_id','')} | {role} | "
            f"{lid} | {number} | {name} | {msg}"
        )
    return "\n".join(lines) + "\n"


def export_chat_log_to_onedrive(n: int = 200) -> str:
    text = build_chat_debug_log_text(n)
    if not file_ops.onedrive_configured():
        path = file_ops.write_text_file(
            text, file_ops.make_timestamped_name(f"wa_chat_log_last{n}")
        )
        return f"⚠️ OneDrive not configured. Log saved locally: {path}"
    try:
        result = file_ops.write_and_upload_text(
            text, filename_prefix=f"wa_chat_log_last{n}"
        )
        return (
            f"✅ Exported last {n} messages → OneDrive/"
            f"{result['folder']}/{result['remote_name']}\n"
            f"{result.get('webUrl') or ''}"
        )
    except Exception as e:
        return f"❌ Export failed: {e}"


BOT_PN = None
BOT_LID = None

# --- GROUP METADATA CACHE ---
GROUP_METADATA_CACHE = {}
GROUP_CACHE_TTL = 300  # Cache expires after 5 minutes (300 seconds)

# --- CONTACTS MAP CACHE ---
CONTACTS_CACHE = {
    "contacts_map": {},
    "reverse_map": {},
    "timestamp": 0
}
CONTACTS_CACHE_TTL = 300  # Cache expires after 5 minutes (300 seconds)

# --- CHAT HISTORY CACHE ---
CHAT_HISTORY_CACHE = {}
LAST_URL_BY_CHAT = {}  # chat_id -> most recent URL seen in that chat
MAX_HISTORY_CACHE = 150  # Matches the max limit in detect_summary_history_limit


def async_update_group_cache(chat_id, group_jid, contacts_map):
    """Fetches group metadata in the background so it doesn't block AI replies."""
    try:
        group_info = client.get_group_info(group_jid)
        active_names = set()
        
        if hasattr(group_info, "Participants"):
            for participant in group_info.Participants:
                p_jid = getattr(participant, "JID", None)
                p_user = _user_of(p_jid)
                
                if p_user:
                    p_name = contacts_map.get(p_user, p_user)
                    active_names.add(p_name)
                    
        # Save to global cache with timestamp
        GROUP_METADATA_CACHE[chat_id] = {
            "names": active_names,
            "timestamp": time.time()
        }
        print(f"[CACHE] Warmed group metadata for {chat_id} ({len(active_names)} members)")
    except Exception as e:
        print(f"[CACHE] Error fetching group metadata for {chat_id}: {e}")

def _user_of(jid_or_str) -> str:
    if not jid_or_str:
        return ""
    if isinstance(jid_or_str, str):
        u = jid_or_str.split("@")[0].strip()
    else:
        u = str(getattr(jid_or_str, "User", getattr(jid_or_str, "user", "")) or "").strip()
    return u.split(":")[0].strip()  # Strips multi-device IDs like :4 or :12

def _server_of(jid_or_str) -> str:
    if not jid_or_str:
        return ""
    if isinstance(jid_or_str, str):
        parts = jid_or_str.split("@", 1)
        return parts[1].strip().lower() if len(parts) > 1 else ""
    return str(getattr(jid_or_str, "Server", getattr(jid_or_str, "server", "")) or "").strip().lower()

def refresh_bot_identities(client):
    """Load bot phone + LID from session (whatsmeow Device)."""
    global BOT_PN, BOT_LID
    try:
        me = client.get_me()
        if not me:
            return
            
        # Extract Phone
        jid = getattr(me, "JID", None)
        if jid:
            BOT_PN = _user_of(jid) or BOT_PN
            
        # Extract LID: Check common attribute variations
        for attr in ("LID", "Lid", "lid"):
            lid_jid = getattr(me, attr, None)
            if lid_jid:
                u = _user_of(lid_jid)
                if u:
                    BOT_LID = u
                    break
                    
        # Extract LID: Check nested paths if not found
        if not BOT_LID:
            for path in (("ID",), ("Device", "LID"), ("device", "LID")):
                obj = me
                try:
                    for p in path:
                        obj = getattr(obj, p, None)
                    u = _user_of(obj)
                    if u and (_server_of(obj) == "lid" or (obj and not _server_of(obj))):
                        if u != BOT_PN:
                            BOT_LID = u
                            break
                except Exception:
                    pass
        print(f"[IDENTITY] BOT_PN={BOT_PN} BOT_LID={BOT_LID}")
    except Exception as e:
        print(f"[IDENTITY] refresh failed: {e}")

def learn_bot_lid_from_message(message, bot_pn):
    """Fallback: Learn LID from outbound traffic."""
    global BOT_LID
    if BOT_LID or not bot_pn:
        return
    try:
        src = message.Info.MessageSource
        is_from_me = getattr(message.Info, "IsFromMe", getattr(message.Info, "fromMe", False))
        
        if is_from_me:
            sender = getattr(src, "Sender", None)
            if _server_of(sender) == "lid":
                BOT_LID = _user_of(sender)
                print(f"[IDENTITY] Learned BOT_LID from own message: {BOT_LID}")
                return
                
            alt = getattr(src, "SenderAlt", None)
            if _server_of(alt) == "lid":
                BOT_LID = _user_of(alt)
                print(f"[IDENTITY] Learned BOT_LID from SenderAlt: {BOT_LID}")
    except Exception:
        pass

def is_bot_natively_mentioned(ctx, bot_pn, bot_lid) -> bool:
    """ONLY native WhatsApp @mention via contextInfo.mentionedJID."""
    if not ctx:
        return False
        
    mentioned = getattr(ctx, "mentionedJID", None) or getattr(ctx, "MentionedJID", None)
    if not mentioned:
        return False
        
    bot_ids = {x for x in (bot_pn, bot_lid) if x}
    for raw in mentioned:
        u = _user_of(raw)
        if not u:
            continue
        if u in bot_ids:
            return True
            
        s = str(raw)
        if bot_lid and bot_lid in s:
            return True
        if bot_pn and bot_pn in s and "@lid" not in s:
            if s.startswith(bot_pn + "@") or f"/{bot_pn}" in s:
                return True
                
    return False

def is_quote_of_bot(ctx, bot_pn, bot_lid, bot_jid_user) -> bool:
    if not ctx:
        return False
    bot_ids = {x for x in (bot_pn, bot_lid, bot_jid_user) if x}
    participant = getattr(ctx, "participant", None) or getattr(ctx, "Participant", None)
    if participant:
        u = _user_of(participant)
        if u and u in bot_ids:
            return True
        s = str(participant)
        if any(b and b in s for b in bot_ids):
            return True
    return False

def quoted_matches_recent_bot_reply(chat_id, quoted_text) -> bool:
    q = (quoted_text or "").strip()
    if not q:
        return False
    try:
        rows = (
            supabase.table("chat_history")
            .select("content")
            .eq("chat_id", chat_id)
            .eq("role", "assistant")
            .order("created_at", desc=True)
            .limit(20)
            .execute()
            .data
            or []
        )
        q_norm = " ".join(q.lower().split())
        for row in rows:
            c = " ".join((row.get("content") or "").lower().split())
            if not c:
                continue
            if q_norm in c or c in q_norm:
                return True
    except Exception as e:
        print(f"[REACTION] history match failed: {e}")
    return False

def get_tzinfo(tz_val):
    """Robustly parse IANA strings, numeric float offsets, or fallback to default."""
    if not tz_val:
        return ZoneInfo(DEFAULT_TIMEZONE)
    try:
        # Handle float/int offsets (e.g. 5.0)
        if isinstance(tz_val, (int, float)):
            return timezone(timedelta(hours=tz_val))
        if isinstance(tz_val, str):
            try:
                # Handle string offsets (e.g. "+5", "-8.5")
                offset = float(tz_val)
                return timezone(timedelta(hours=offset))
            except ValueError:
                # Handle IANA strings (e.g. "Asia/Karachi")
                return ZoneInfo(tz_val)
    except Exception:
        pass
    return ZoneInfo(DEFAULT_TIMEZONE)


def get_world_clocks():
    """Calculates exact current times with Daylight Saving Time (DST) 
    for major global regions to prevent LLM time calculation errors."""
    now = datetime.now(timezone.utc)
    zones = {
        "UTC": "UTC",
        "Pakistan (PKT)": "Asia/Karachi",
        "US/Canada Eastern (Toronto/NY - EDT/EST)": "America/Toronto",
        "US/Canada Central (Chicago/Winnipeg)": "America/Chicago",
        "US/Canada Mountain (Denver/Calgary)": "America/Edmonton",
        "US/Canada Pacific (LA/Vancouver)": "America/Vancouver",
        "UK (London - BST/GMT)": "Europe/London",
        "UAE (Dubai)": "Asia/Dubai",
        "Saudi Arabia (Riyadh)": "Asia/Riyadh",
        "India (IST)": "Asia/Kolkata",
    }
    clocks = []
    for name, tz_str in zones.items():
        try:
            tz_time = now.astimezone(ZoneInfo(tz_str))
            clocks.append(f"- {name}: {tz_time.strftime('%A, %I:%M %p')}")
        except Exception:
            pass
    return "\n".join(clocks)


def upsert_contact(sender_id, push_name, sender_num=None):
    if not push_name:
        return
    try:
        data = {
            "sender_id": sender_id,
            "display_name": push_name,
        }
        if sender_num:
            data["sender_num"] = sender_num
            
        supabase.table("contacts").upsert(data).execute()
        # Proactively inject into the cache so the bot knows their name instantly
        if CONTACTS_CACHE["timestamp"] > 0: 
            CONTACTS_CACHE["contacts_map"][sender_id] = push_name
            CONTACTS_CACHE["reverse_map"][push_name.lower()] = sender_id
    except Exception as e:
        print(f"Error upserting contact: {e}")


def get_contacts_maps():
    """Fetches contacts with a TTL cache to heavily cut down Supabase reads."""
    global CONTACTS_CACHE
    
    # 1. Return from cache if it's still fresh
    if time.time() - CONTACTS_CACHE["timestamp"] < CONTACTS_CACHE_TTL:
        return CONTACTS_CACHE["contacts_map"], CONTACTS_CACHE["reverse_map"]

    # 2. Otherwise, fetch from Supabase
    try:
        response = supabase.table("contacts").select("sender_id, display_name, nickname, sender_num").execute()
        contacts_map = {}
        reverse_map = {}
        
        for row in response.data:
            s_id = row["sender_id"]
            name = row.get("nickname") or row.get("display_name") or s_id
            contacts_map[s_id] = name
            
            for display in (row.get("display_name"), row.get("nickname")):
                if display:
                    reverse_map[display.lower()] = s_id
        
        # 3. Update the global cache
        CONTACTS_CACHE["contacts_map"] = contacts_map
        CONTACTS_CACHE["reverse_map"] = reverse_map
        CONTACTS_CACHE["timestamp"] = time.time()
        
        print(f"[CACHE] Warmed contacts map cache ({len(contacts_map)} contacts)")
        return contacts_map, reverse_map

    except Exception as e:
        print(f"Error fetching contacts maps: {e}")
        # Graceful fallback: Use stale cache if the DB read fails
        if CONTACTS_CACHE["contacts_map"]:
            print("[CACHE] Returning stale contacts due to fetch error.")
            return CONTACTS_CACHE["contacts_map"], CONTACTS_CACHE["reverse_map"]
        return {}, {}


def resolve_mentions(text, reverse_map_or_tuple):
    """Swap @Name for @<number> so WhatsApp's mention parser picks it up.
    Robustly accepts either a reverse_map dict or the full (contacts_map, reverse_map) tuple."""
    if not text:
        return text

    # Handle case where get_contacts_maps() tuple was passed directly
    if isinstance(reverse_map_or_tuple, tuple):
        reverse_map = reverse_map_or_tuple[1] if len(reverse_map_or_tuple) > 1 else {}
    elif isinstance(reverse_map_or_tuple, dict):
        reverse_map = reverse_map_or_tuple
    else:
        return text

    try:
        for name in sorted(reverse_map.keys(), key=len, reverse=True):
            pattern = re.compile(re.escape("@" + name), re.IGNORECASE)
            text = pattern.sub(f"@{reverse_map[name]}", text)
    except Exception as e:
        print(f"Error resolving mentions: {e}")

    return text


def maybe_save_memory(chat_id, sender_id, text_content):
    """Keyword version: if the message contains a 'remember this' trigger, save it
    permanently for this chat. Fetched separately from chat_history with NO .limit(),
    so — unlike the rolling 50-message window — it never ages out.
    Known gap: this is a plain substring match, so it will miss instructions that
    don't use one of the MEMORY_TRIGGERS words (e.g. "no more jokes, izzat se baat
    karna" has no trigger word in it and would slip through). See
    maybe_save_memory_smart() for the fix."""
    lowered = text_content.lower()
    if not any(trigger in lowered for trigger in MEMORY_TRIGGERS):
        return
    try:
        supabase.table("group_memory").insert({
            "chat_id": chat_id,
            "sender_id": sender_id,
            "note": text_content,
        }).execute()
        print(f"[MEMORY SAVED] {text_content}")
    except Exception as e:
        print(f"Error saving memory: {e}")


def maybe_save_memory_smart(chat_id, sender_id, text_content):
    """Optional upgrade: asks the LLM itself whether this message just set a durable
    rule, correction, or fact about a person — catches things like "no more jokes,
    izzat se baat karo" which the keyword version above would miss entirely, since it
    never says "remember". Costs one extra Groq call per bot-directed message.
    To use: swap the call to maybe_save_memory(...) for this one in Logic 2 below."""
    try:
        classification = client_ai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": (
                    "Reply with ONLY the fact/rule in one short sentence if this message "
                    "sets a standing instruction, correction, or a fact about a specific "
                    "person (name, relationship, preference). Otherwise reply with exactly: NONE"
                )},
                {"role": "user", "content": text_content}
            ],
            temperature=0
        ).choices[0].message.content.strip()

        if classification and classification.upper() != "NONE":
            supabase.table("group_memory").insert({
                "chat_id": chat_id,
                "sender_id": sender_id,
                "note": classification
            }).execute()
            print(f"[MEMORY SAVED - SMART] {classification}")
    except Exception as e:
        print(f"Error in smart memory save: {e}")


def get_group_memory(chat_id):
    """All permanent notes/rules for this chat — fetched in full every time, never
    truncated the way chat_history is."""
    try:
        response = supabase.table("group_memory").select("note").eq("chat_id", chat_id).execute()
        return [row["note"] for row in response.data]
    except Exception as e:
        print(f"Error fetching group memory: {e}")
        return []


def get_media_kind(message):
    try:
        msg_obj = message.Message
        if msg_obj.imageMessage and (msg_obj.imageMessage.mimetype or msg_obj.imageMessage.URL):
            return "image"
        if msg_obj.videoMessage and (msg_obj.videoMessage.mimetype or msg_obj.videoMessage.URL):
            if getattr(msg_obj.videoMessage, "gifPlayback", False):
                return "gif"
            return "video"
        if msg_obj.audioMessage and (msg_obj.audioMessage.mimetype or msg_obj.audioMessage.URL):
            return "audio"
        # Documents: mimetype OR fileName/title is enough
        doc = getattr(msg_obj, "documentMessage", None)
        if doc and (
            getattr(doc, "mimetype", None)
            or getattr(doc, "fileName", None)
            or getattr(doc, "title", None)
            or getattr(doc, "URL", None)
        ):
            return "document"
        sticker = getattr(msg_obj, "stickerMessage", None)
        if sticker and (
            getattr(sticker, "mimetype", None)
            or getattr(sticker, "URL", None)
            or getattr(sticker, "directPath", None)
            or getattr(sticker, "mediaKey", None)
        ):
            return "sticker"
    except AttributeError:
        pass
    return None

def get_context_info(msg_obj):
    if not msg_obj:
        return None
    for msg_type in ("stickerMessage", "extendedTextMessage", "imageMessage", "videoMessage", "documentMessage", "audioMessage"):
        sub_msg = getattr(msg_obj, msg_type, None)
        if sub_msg:
            for attr in ("contextInfo", "ContextInfo", "context_info"):
                ctx = getattr(sub_msg, attr, None)
                if ctx:
                    return ctx
    return None

def detect_summary_history_limit(text):
    """If text is asking for a summary, figure out how many messages to pull in.
    Tightened from a bare \\d+ match — which grabbed ANY number in the message,
    e.g. an order number or a year — to only numbers actually tied to a message
    count: "last 80 messages", "80 msgs", "past 30", etc.
    Returns None if this isn't a summary request at all, so callers can tell
    'not a summary' apart from 'summary with no number given'."""
    lowered = text.lower()
    if not any(word in lowered for word in ("summarize", "summary", "recap")):
        return None

    match = (re.search(r'(\d+)\s*(?:msgs?|messages?)\b', lowered)
             or re.search(r'\b(?:last|past|previous)\s+(\d+)\b', lowered))

    if match:
        requested = int(match.group(1))
        # Cap at 150 so "summarize the last 1000 messages" can't blow up the prompt.
        return min(requested + 5, 150)  # +5 so the request message itself is included
    return 50  # asked to summarize, no number given — a reasonable default


def fetch_chat_history(chat_id, history_limit):
    """Fetches history from RAM if available, otherwise warms the cache from Supabase."""
    global CHAT_HISTORY_CACHE
    
    # 1. Warm up the cache with the maximum possible context window (150 msgs)
    if chat_id not in CHAT_HISTORY_CACHE:
        try:
            response = supabase.table("chat_history") \
                .select("*") \
                .eq("chat_id", chat_id) \
                .order("created_at", desc=True) \
                .limit(MAX_HISTORY_CACHE) \
                .execute()
            # Store chronologically in RAM
            CHAT_HISTORY_CACHE[chat_id] = response.data[::-1] 
            print(f"[CACHE] Warmed chat history for {chat_id} ({len(response.data)} msgs)")
        except Exception as e:
            print(f"Error fetching history for cache: {e}")
            return []
            
    # 2. Return exactly what the AI asked for from the end of the list
    return CHAT_HISTORY_CACHE[chat_id][-history_limit:]


def insert_chat_message(chat_id, sender_id, role, content, sender_num=None):
    """Saves a message to Supabase and instantly injects it into the RAM cache."""
    global CHAT_HISTORY_CACHE
    
    data = {
        "chat_id": chat_id,
        "sender_id": sender_id,
        "role": role,
        "content": content
    }
    if sender_num:
        data["sender_num"] = sender_num
        
    try:
        # Write to DB and grab the exact timestamp it generated
        response = supabase.table("chat_history").insert(data).execute()
        inserted_row = response.data[0] if response.data else data
        
        # Fallback timestamp if the DB return payload is empty
        if "created_at" not in inserted_row:
            inserted_row["created_at"] = datetime.now(timezone.utc).isoformat()
        
        # Proactively inject into the cache so the next read is instant
        if chat_id in CHAT_HISTORY_CACHE:
            CHAT_HISTORY_CACHE[chat_id].append(inserted_row)
            
            # Prune memory to prevent leaks in long-running chats
            if len(CHAT_HISTORY_CACHE[chat_id]) > MAX_HISTORY_CACHE:
                CHAT_HISTORY_CACHE[chat_id].pop(0)
                
    except Exception as e:
        print(f"Error saving message: {e}")


def _guess_doc_meta(doc_msg, tmp_path):
    mime = (getattr(doc_msg, "mimetype", None) or "").strip().lower()
    name = (
        getattr(doc_msg, "fileName", None)
        or getattr(doc_msg, "title", None)
        or os.path.basename(tmp_path)
        or "file"
    )
    ext = os.path.splitext(name)[1].lower()

    if not mime or mime in ("application/octet-stream", "application/zip"):
        mime = {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".csv": "text/csv",
            ".json": "application/json",
            ".pdf": "application/pdf",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xls": "application/vnd.ms-excel",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".ppt": "application/vnd.ms-powerpoint",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        }.get(ext, mime or "application/octet-stream")
    return mime, name, ext

def _file_magic_kind(path: str) -> str:
    """Return 'pdf' | 'zip_ooxml' | 'ole' | 'text' | 'unknown' from bytes."""
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except Exception:
        return "unknown"
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK"):  # zip → xlsx/docx/pptx
        return "zip_ooxml"
    if head.startswith(b"\xd0\xcf\x11\xe0"):  # old .doc/.xls/.ppt
        return "ole"
    # rough text
    try:
        head.decode("utf-8")
        return "text"
    except Exception:
        return "unknown"


def extract_document_text(tmp_path, mime, ext, max_chars=30000) -> str:
    """Return plain text from common document types. Raises on hard failure."""
    path = tmp_path
    text = ""

    # --- plain text family ---
    if (
        mime.startswith("text/")
        or mime in ("application/json", "application/xml")
        or ext in (".txt", ".md", ".csv", ".json", ".log")
    ):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

    # --- Word (.docx) ---
    elif ext == ".docx" or "wordprocessingml" in mime:
        from docx import Document
        doc = Document(path)
        parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        # tables
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        text = "\n".join(parts)

    # --- Excel (.xlsx) ---
    elif ext == ".xlsx" or "spreadsheetml" in mime:
        if _file_magic_kind(path) != "zip_ooxml":
            raise ValueError(
                "File is not a real .xlsx (content is not an Excel package). "
                "Open it in Excel and Save As .xlsx, or send PDF/CSV."
            )
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        chunks = []
        for sheet in wb.worksheets:
            chunks.append(f"## Sheet: {sheet.title}")
            for i, row in enumerate(sheet.iter_rows(values_only=True)):
                if i > 200:  # cap rows per sheet
                    chunks.append("…[more rows truncated]…")
                    break
                vals = ["" if v is None else str(v) for v in row]
                if any(v.strip() for v in vals):
                    chunks.append("\t".join(vals))
        wb.close()
        text = "\n".join(chunks)

    # --- PowerPoint (.pptx) ---
    elif ext == ".pptx" or "presentationml" in mime:
        from pptx import Presentation
        prs = Presentation(path)
        parts = []
        for i, slide in enumerate(prs.slides, 1):
            parts.append(f"## Slide {i}")
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text and shape.text.strip():
                    parts.append(shape.text.strip())
        text = "\n".join(parts)

    # --- PDF: leave to Gemini (return empty → caller uses vision path) ---
    elif ext == ".pdf" or mime == "application/pdf":
        return ""

    # --- legacy .doc / .xls / .ppt ---
    elif ext in (".doc", ".xls", ".ppt"):
        raise ValueError(
            f"Legacy format {ext} is not supported. "
            "Please re-save as .docx / .xlsx / .pptx or PDF."
        )

    else:
        raise ValueError(f"Unsupported document type: {name_safe(ext, mime)}")

    text = (text or "").strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n…[truncated]…"
    return text


def name_safe(ext, mime):
    return f"{ext or '?'} ({mime or 'unknown'})"


def is_quote_of_bot(ctx, bot_pn, bot_lid, bot_jid_user) -> bool:
    if not ctx:
        return False
    bot_ids = {x for x in (bot_pn, bot_lid, bot_jid_user) if x}
    participant = getattr(ctx, "participant", None) or getattr(ctx, "Participant", None)
    if participant:
        u = _user_of(participant)
        if u and u in bot_ids:
            return True
        s = str(participant)
        if any(b and b in s for b in bot_ids):
            return True
    return False


def handle_media_message(message, media_kind, chat_id, sender_id, text_content="", target_media_msg=None, msg_time=None, history_limit=20, is_reaction_to_bot=False,):
    
    # --- ADMIN FEATURE FLAG CHECKS (FRIENDLY REJECTION) ---
    chat_has_any_feature = admin_commands.has_any_feature_enabled(chat_id)

    if media_kind == "document" and not admin_commands.is_feature_enabled(chat_id, "documents"):
        print(f"[FEATURE OFF] 'documents' disabled for {chat_id}.")
        return "📄 Document reading is currently turned off for this chat." if chat_has_any_feature else None
            
    if media_kind in ("video", "gif") and not admin_commands.is_feature_enabled(chat_id, "videos"):
        print(f"[FEATURE OFF] 'videos' disabled for {chat_id}.")
        return "🎥 Video processing is currently turned off for this chat." if chat_has_any_feature else None

    if media_kind in ("image", "sticker", "user_created_sticker") and not admin_commands.is_feature_enabled(chat_id, "images"):
        print(f"[FEATURE OFF] 'images' disabled for {chat_id}.")
        return "🖼️ Image processing is currently turned off for this chat." if chat_has_any_feature else None

    if media_kind in ("audio", "ptt") and not admin_commands.is_feature_enabled(chat_id, "audio"):
        print(f"[FEATURE OFF] 'audio' disabled for {chat_id}.")
        return "🎤 Voice notes are currently turned off for this chat." if chat_has_any_feature else None


    # --- MEDIA DOWNLOAD & PROCESSING ---
    if media_kind == "document":
        # Get the real file extension before creating the temp file
        doc_src = (
            target_media_msg.documentMessage
            if target_media_msg and getattr(target_media_msg, "documentMessage", None)
            else getattr(message.Message, "documentMessage", None)
        )
        orig_name = getattr(doc_src, "fileName", None) or getattr(doc_src, "title", None) or "file.pdf"
        _, ext = os.path.splitext(orig_name)
        suffix = ext.lower() if ext else ".pdf"
    else:
        # Standard media mappings
        suffix = {
            "image": ".jpg", 
            "video": ".mp4", 
            "gif": ".mp4",
            "audio": ".ogg", 
            "ptt": ".ogg", 
            "sticker": ".webp",
            "user_created_sticker": ".webp"
        }.get(media_kind, ".bin")
        
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
        
        msg_to_download = target_media_msg if target_media_msg else message.Message
        client.download_any(msg_to_download, path=tmp_path)


        if media_kind == "document":
            if not admin_commands.is_feature_enabled(chat_id, "documents"):
                print(f"[FEATURE OFF] 'documents' disabled for {chat_id}.")
                return (
                    "📄 Document reading is currently turned off for this chat."
                    if admin_commands.has_any_feature_enabled(chat_id)
                    else None
                )

            doc_src = (
                target_media_msg.documentMessage
                if target_media_msg and getattr(target_media_msg, "documentMessage", None)
                else message.Message.documentMessage
            )

            mime, filename, ext = _guess_doc_meta(doc_src, tmp_path)
            magic = _file_magic_kind(tmp_path)
            print(f"[DOC] name={filename!r} ext={ext!r} mime={mime!r} magic={magic}")

            # Filename/extension lied — content is actually PDF
            if magic == "pdf":
                ext = ".pdf"
                mime = "application/pdf"

            # --- PDF → Gemini (do NOT openpyxl) ---
            if magic == "pdf" or ext == ".pdf" or mime == "application/pdf":
                with open(tmp_path, "rb") as f:
                    media_b64 = base64.b64encode(f.read()).decode("utf-8")
                user_prompt = (
                    text_content.strip()
                    if text_content and text_content.strip()
                    else f"Summarize or explain this PDF ({filename})."
                )
                try:
                    gemini_response = client_gemini.chat.completions.create(
                        model=GEMINI_MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are Mojo. The user sent a PDF. "
                                    "Answer from its content. Keep it WhatsApp-short."
                                ),
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:application/pdf;base64,{media_b64}"
                                        },
                                    },
                                ],
                            },
                        ],
                    )
                    return gemini_response.choices[0].message.content
                except Exception as e:
                    print(f"[PDF GEMINI ERROR] {e}")
                    return "📄 Couldn't read that PDF right now. Try again or paste the text."

            # Named like Office but bytes are not OOXML/text
            if magic not in ("zip_ooxml", "text") and ext in (".xlsx", ".docx", ".pptx"):
                return (
                    f"📄 `{filename}` is named like Office, but the file content is not "
                    f"(detected: {magic}). Re-export from Excel/Word as real .xlsx/.docx, "
                    "or send a real PDF."
                )

            # --- Office + text: extract → normal chat LLM ---
            try:
                body = extract_document_text(tmp_path, mime, ext)
            except Exception as e:
                print(f"[DOC EXTRACT ERROR] {filename}: {e}")
                return (
                    f"📄 Got `{filename}`, but couldn't read it ({e}). "
                    "Try PDF, DOCX, XLSX, PPTX, or TXT."
                )

            if body:
                context_str = (
                    f"[Document: {filename}]\n{body}"
                    if not (text_content and text_content.strip())
                    else f"User said: {text_content}\n\n[Document: {filename}]\n{body}"
                )
                insert_chat_message(chat_id, "document_text", "user", context_str)
                if admin_commands.is_feature_enabled(chat_id, "ai_chat") or admin_commands.is_feature_enabled(chat_id, "reminders"):
                    from agent_loop import run_agent
                    is_group = "g.us" in (chat_id or "")
                    return run_agent(
                        chat_id=chat_id,
                        sender_id=sender_id,
                        history_limit=history_limit,
                        is_group=is_group,
                        msg_time=msg_time,
                        extra_user_note=context_str,
                    )
                return (
                    f"📄 Read `{filename}` ({len(body)} chars). "
                    "AI chat is off for this chat."
                )

            return (
                f"📄 Received `{filename}` ({mime}). "
                "Supported: TXT, CSV, JSON, MD, PDF, DOCX, XLSX, PPTX."
            )


        # A. Voice Notes (Whisper)
        if media_kind in ("audio", "ptt"):
            with open(tmp_path, "rb") as f:
                transcript = client_ai.audio.transcriptions.create(
                    model=WHISPER_MODEL,
                    file=f,
                    language="ur"
                ).text
            print(f"[VOICE NOTE TRANSCRIBED]: {transcript}")
            
            context_str = f"[Voice note transcript]: {transcript}"
            if text_content and text_content.strip():
                context_str = f"User said: {text_content}\n\n[Quoted Audio Transcript]: {transcript}"

            transcript_limit = detect_summary_history_limit(transcript)
            if transcript_limit is not None:
                history_limit = transcript_limit

            insert_chat_message(chat_id, "voice_transcript", "user", context_str)

            # Route voice through the SAME agent loop as text so tools
            # (set_reminder, cancel, knowledge, …) actually run.
            if admin_commands.is_feature_enabled(chat_id, "ai_chat") or admin_commands.is_feature_enabled(chat_id, "reminders"):
                from agent_loop import run_agent
                is_group = "g.us" in (chat_id or "")
                return run_agent(
                    chat_id=chat_id,
                    sender_id=sender_id,
                    sender_num=None,
                    history_limit=history_limit,
                    is_group=is_group,
                    msg_time=msg_time,
                    extra_user_note=context_str,
                )
            return None

        # B. Visual & Document Media (Gemini)
        elif media_kind in ("image", "sticker", "user_created_sticker", "gif", "video", "document"):
            
            if "sticker" in media_kind:
                mime = "image/webp"
            elif media_kind in ("gif", "video"):
                mime = "video/mp4"
            elif media_kind == "document":
                mime = "application/pdf" 
            else:
                mime = "image/jpeg"

            with open(tmp_path, "rb") as f:
                media_b64 = base64.b64encode(f.read()).decode("utf-8")
            
            user_prompt = text_content if (text_content and text_content.strip()) else "What is in this media?"
            
            gemini_response = client_gemini.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are Mojo, the AI assistant for Mojo AI Agency. Someone sent "
                        "or replied to an image, sticker, GIF, video, or document. Extract details, "
                        "describe it, or react naturally based on the user's prompt. Keep it WhatsApp-short."
                    )},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{media_b64}"}}
                    ]}
                ]
            )
            return gemini_response.choices[0].message.content

        else:
            return None
            
    except Exception as e:
        print(f"Error downloading/processing media: {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def send_proactive_message(chat_id, text):
    """Send a message that ISN'T a reply to anything. Needed because reminders
    fire on a timer, not in response to an incoming WhatsApp message — this is
    the one place the bot speaks first."""
    try:
        user, server = chat_id.split("@", 1)
        jid = build_jid(user, server=server)
        client.send_message(jid, text, mentions_are_lids=True)
    except Exception as e:
        print(f"Error sending proactive message to {chat_id}: {e}")


def extract_reminder_data_via_ai(text_content, user_tz_str):
    """Uses Groq/Llama-3 in JSON mode to perfectly parse multiple complex times and recurrences."""
    now_utc = datetime.now(timezone.utc)
    
    try:
        tz = ZoneInfo(user_tz_str)
    except Exception:
        tz = ZoneInfo(DEFAULT_TIMEZONE)
        user_tz_str = DEFAULT_TIMEZONE
        
    now_local = now_utc.astimezone(tz)
    
    system_prompt = f"""You are an advanced time-parsing engine for WhatsApp reminders.
Users often write in Roman Urdu / mixed English-Urdu (e.g. "1 min me paani peena hai", "remind mér ek minute me", "roz subah 7 baje").
Extract the reminder details and return ONLY a valid JSON object.

Current UTC Time: {now_utc.isoformat()}
User's Local Time: {now_local.isoformat()}
User's Timezone: {user_tz_str}

Roman Urdu time hints:
- "ek min / 1 min / ek minute me" = 1 minute from now
- "do min / 2 mins" = 2 minutes
- "ek ghanta" = 1 hour
- "kal" = tomorrow
- "roz / daily / everyday" = recurring every day
- "subah / morning", "sham / evening" — map to reasonable clock times if no exact hour given

JSON Schema:
{{
  "message": "Cleaned subject only (e.g. 'paani peena hai' or 'debug'). Strip trigger words like remind/reminder/riemind/ریمائنڈ. Keep original language.",
  "is_recurring": boolean (true if daily/roz/everyday or specific weekdays),
  "recurring_days": array of lowercase English weekday names or null if everyday,
  "recurring_local_times": array of "HH:MM" 24h local times (null if not recurring),
  "one_off_utc_times": array of ISO-8601 UTC datetimes (null if recurring)
}}

If you cannot parse any concrete time, still return valid JSON with empty arrays so the caller can ask for clarification."""

    try:
        response = client_ai.chat.completions.create(
            model=MODEL_NAME, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_content}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in AI JSON reminder parsing: {e}")
        return None

def get_user_timezone(sender_id):
    try:
        response = supabase.table("contacts").select("timezone").eq("sender_id", sender_id).execute()
        if response.data and response.data[0].get("timezone"):
            return response.data[0]["timezone"]
    except Exception as e:
        print(f"Error fetching timezone for {sender_id}: {e}")
    return None

def set_user_timezone(sender_id, tz_string):
    try:
        supabase.table("contacts").update({"timezone": tz_string}).eq("sender_id", sender_id).execute()
    except Exception as e:
        print(f"Error saving timezone for {sender_id}: {e}")

def maybe_set_timezone(sender_id, text_content):
    match = TZ_OFFSET_RE.search(text_content)
    if not match:
        return None
    raw = match.group(1)
    try:
        if ":" in raw or "." in raw:
            sign = -1 if raw.strip().startswith("-") else 1
            hh, mm = re.split(r"[:.]", raw.lstrip("+-"))
            offset = sign * (int(hh) + int(mm) / 60)
        else:
            offset = float(raw)
    except ValueError:
        return None
    if not (-12 <= offset <= 14):
        return "That doesn't look like a real UTC offset — try something like +5 or -8."
    set_user_timezone(sender_id, offset)
    sign_str = f"+{offset:g}" if offset >= 0 else f"{offset:g}"
    return f"Got it — using UTC{sign_str} for your reminders from now on. 🕒"

def generate_reminder_confirmation(chat_id, sender_id, message_text, schedule_desc, history_limit=20):
    """Fast template confirmation — avoids an extra LLM call (was a major rate-limit / latency source).
    Agent loop will polish the final WhatsApp reply if needed."""
    subj = (message_text or "your reminder").strip().strip('"').strip()
    # Keep it short and bilingual-friendly (Roman Urdu / English)
    if schedule_desc:
        return f'Reminder set — "{subj}" {schedule_desc} 🚀'
    return f'Reminder set — "{subj}" 🚀'



def handle_reminder_request(chat_id, sender_id, text_content, msg_time):
    user_tz = get_user_timezone(sender_id)

    # --- AI Timezone Detection for First-Time Users ---
    if user_tz is None:
        print(f"[TIMEZONE] First-time reminder for {sender_id}, checking for location cues...")
        current_date = datetime.now(timezone.utc).strftime("%B %d, %Y")
        
        ai_tz_prompt = (
            f"Today is {current_date}. The user wants to set a reminder: '{text_content}'. "
            "If they mentioned a specific country, city, or timezone (like 'Canada', 'London', 'EST'), "
            "reply ONLY with the exact IANA timezone database name (e.g., 'America/Toronto', 'Europe/London', 'Asia/Dubai'). "
            "If no location is mentioned at all, reply EXACTLY with: NONE"
        )
        try:
            ai_tz = client_ai.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": ai_tz_prompt}],
                temperature=0
            ).choices[0].message.content.strip()

            if ai_tz != "NONE" and "/" in ai_tz:
                user_tz = ai_tz
                set_user_timezone(sender_id, user_tz)
                print(f"[TIMEZONE SET] Automatically set {sender_id} to {user_tz}")
            else:
                return "I can definitely set that up! Since this is your first time, please let me know your city or country so I get the timing exactly right."
        except Exception as e:
            print(f"Error checking AI timezone: {e}")
            user_tz = DEFAULT_TIMEZONE
    # --- END TIMEZONE ---

    # 1. Ask the AI to parse the times into structured JSON arrays
    parsed_data = extract_reminder_data_via_ai(text_content, user_tz)
    
    if not parsed_data or (not parsed_data.get("one_off_utc_times") and not parsed_data.get("recurring_local_times")):
        return ('Sure — what time? Try something like "remind me in 1 min and 2 mins", '
                '"remind me at 4pm and 7pm", or "remind me daily to pray 5 times".')

    message_text = parsed_data.get("message", "your reminder")
    is_recurring = parsed_data.get("is_recurring", False)
    days = parsed_data.get("recurring_days")

    try:
        tz = get_tzinfo(user_tz)
    except Exception:
        tz = get_tzinfo(DEFAULT_TIMEZONE)

    now_utc = datetime.now(timezone.utc)
    
    try:
        if is_recurring:
            # Handle multiple recurring times (e.g., Pray 5 times a day)
            daily_times = parsed_data.get("recurring_local_times", [])
            
            # Get the very next valid occurrence for the base schedule
            remind_at_utc = compute_next_occurrence(daily_times, days, now_utc, user_tz)
            
            supabase.table("reminders").insert({
                "chat_id": chat_id,
                "sender_id": sender_id,
                "message": message_text,
                "remind_at": remind_at_utc.isoformat(),
                "recurring": True,
                "daily_times": daily_times,
                "days": days,
                "timezone": user_tz,
                "active": True
            }).execute()
            
            # Format nicely for the confirmation message
            am_pm_times = [datetime.strptime(t, "%H:%M").strftime("%I:%M %p") for t in daily_times]
            schedule_desc = (f"every {', '.join(d.capitalize() for d in days)}" if days else "every day") \
                + f" at {', '.join(am_pm_times)}"

        else:
            # Handle multiple separate one-off times (e.g., in 1min and 2min)
            one_off_times = parsed_data.get("one_off_utc_times", [])
            local_formatted_times = []
            
            for time_str in one_off_times:
                clean_time_str = time_str.replace("Z", "+00:00")
                remind_at_utc = datetime.fromisoformat(clean_time_str)
                
                # Guarantee UTC awareness before converting and saving
                if remind_at_utc.tzinfo is None:
                    remind_at_utc = remind_at_utc.replace(tzinfo=timezone.utc)
                
                supabase.table("reminders").insert({
                    "chat_id": chat_id,
                    "sender_id": sender_id,
                    "message": message_text,
                    "remind_at": remind_at_utc.isoformat(),
                    "recurring": False,
                    "daily_times": None,
                    "days": None,
                    "timezone": user_tz,
                    "active": True
                }).execute()
                
                local_formatted_times.append(remind_at_utc.astimezone(tz).strftime('%I:%M %p'))
            
            schedule_desc = f"at {', '.join(local_formatted_times)}"

    except Exception as e:
        print(f"Error saving reminder to database: {e}")
        return "Something went wrong saving those times — mind trying again?"

    return generate_reminder_confirmation(chat_id, sender_id, message_text, schedule_desc)


def compute_next_occurrence(daily_times, days, after_utc, user_tz_str):
    try:
        tz = ZoneInfo(user_tz_str)
    except Exception:
        tz = ZoneInfo(DEFAULT_TIMEZONE)

    days_set = {d.lower() for d in days} if days else None
    after_local = after_utc.astimezone(tz)
    
    for day_offset in range(8):
        candidate_day = after_local + timedelta(days=day_offset)
        if days_set and candidate_day.strftime("%A").lower() not in days_set:
            continue
        for t in daily_times:
            hour, minute = map(int, t.split(":"))
            candidate_local = candidate_day.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if candidate_local > after_local:
                return candidate_local.astimezone(timezone.utc)
                
    hour, minute = map(int, daily_times[0].split(":"))
    fallback_local = (after_local + timedelta(days=1)).replace(hour=hour, minute=minute, second=0, microsecond=0)
    return fallback_local.astimezone(timezone.utc)

def list_reminders(chat_id, sender_id):
    try:
        response = supabase.table("reminders").select("*") \
            .eq("chat_id", chat_id).eq("sender_id", str(sender_id)).eq("active", True).execute()
        rows = response.data or []

        # Fallback: same chat, any sender (handles rare LID/number mismatch)
        if not rows:
            response = supabase.table("reminders").select("*") \
                .eq("chat_id", chat_id).eq("active", True).execute()
            rows = response.data or []
            if sender_id and rows:
                rows = [r for r in rows if str(r.get("sender_id")) == str(sender_id)] or rows

        if not rows:
            return "Abhi koi active reminder nahi hai."

        lines = []
        for r in rows:
            tz_val = r.get("timezone", DEFAULT_TIMEZONE)
            try:
                tz = get_tzinfo(tz_val)
            except Exception:
                tz = ZoneInfo(DEFAULT_TIMEZONE)

            dt = datetime.fromisoformat(str(r["remind_at"]).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            local_time = dt.astimezone(tz)

            if r.get("recurring"):
                schedule = f"every {', '.join(r['days'])}" if r.get("days") else "every day"
                when = f"{schedule} at {local_time.strftime('%I:%M %p')}"
            else:
                when = local_time.strftime("%b %d at %I:%M %p")
            lines.append(f'- "{r["message"]}" — {when}')
        return "Active reminders:\n" + "\n".join(lines)
    except Exception as e:
        print(f"Error listing reminders: {e}")
        traceback.print_exc()
        return f"Reminders load nahi ho sake: {e}"

def reminder_scheduler():
    while True:
        try:
            now = datetime.now(timezone.utc)
            response = supabase.table("reminders").select("*") \
                .eq("active", True).lte("remind_at", now.isoformat()).execute()
            for reminder in response.data:
                try:
                    if not admin_commands.is_feature_enabled(reminder["chat_id"], "reminders"):
                        supabase.table("reminders").update({"active": False}).eq("id", reminder["id"]).execute()
                        continue
                    
                    # --- UPGRADED REMINDER DISPATCH ---
                    contacts_map, _ = get_contacts_maps()
                    user_name = contacts_map.get(reminder["sender_id"], "")
                    tag_name = f"@{user_name}" if user_name else f"@{reminder['sender_id']}"

                    system_instruction = (
                        "You are Mojo, an intelligent, warm personal WhatsApp assistant. "
                        "Your job is to deliver a quick scheduled reminder to the user.\n\n"
                        "STRICT RULES:\n"
                        "1. Keep it to ONE short, warm, and completely natural sentence.\n"
                        "2. MATCH THE LANGUAGE: If the reminder subject is in Roman Urdu/Hindi (e.g., 'kapre dhone hain', 'dawai leni hai'), you MUST reply in natural Roman Urdu/Hindi.\n"
                        "3. NEVER use forced slang (e.g., avoid awkward 'heads up', 'after yaar', or rigid templates).\n"
                        "4. Naturally tag the user at the start."
                    )

                    messages = [
                        {"role": "system", "content": system_instruction},
                        # Few-shot example 1
                        {"role": "user", "content": f"Remind @{reminder['sender_id']}: dawai leni hai"},
                        {"role": "assistant", "content": f"@{reminder['sender_id']} dawai lene ka time ho gaya hai, bilkul mat bhooliye ga! 💊"},
                        # Few-shot example 2
                        {"role": "user", "content": f"Remind @{reminder['sender_id']}: meeting in 5 mins"},
                        {"role": "assistant", "content": f"@{reminder['sender_id']} aap ki meeting 5 minute mein start ho rahi hai!"},
                        # Real Request
                        {"role": "user", "content": f"Remind @{reminder['sender_id']}: {reminder['message']}"}
                    ]

                    try:
                        ai_msg = client_ai.chat.completions.create(
                            model=MODEL_NAME,
                            messages=messages,
                            temperature=0.5
                        ).choices[0].message.content.strip()
                        final_msg = f"⏰ {ai_msg}"
                    except Exception as e:
                        print(f"Dynamic reminder generation failed: {e}")
                        final_msg = f"⏰ {tag_name} Reminder: {reminder['message']}"

                    send_proactive_message(reminder["chat_id"], final_msg)

                    if reminder.get("recurring") and reminder.get("daily_times"):
                        # Fetch the timezone string, default to Karachi if missing
                        user_tz_str = reminder.get("timezone", DEFAULT_TIMEZONE)
                        next_time = compute_next_occurrence(reminder["daily_times"], reminder.get("days"), now, user_tz_str)
                        supabase.table("reminders").update({
                            "remind_at": next_time.isoformat(),
                            "last_sent_at": now.isoformat()
                        }).eq("id", reminder["id"]).execute()
                    else:
                        supabase.table("reminders").update({"active": False}).eq("id", reminder["id"]).execute()
                except Exception as e:
                    print(f"Error sending reminder {reminder.get('id')}: {e}")
        except Exception as e:
            print(f"Error checking reminders: {e}")
        time.sleep(30)



def cancel_reminders(chat_id, sender_id, text_content):
    """Deactivates matching reminders. Supports 'all'/sab and token overlap
    so Roman Urdu like 'paani wale reminders cancel' works."""
    try:
        response = supabase.table("reminders").select("*") \
            .eq("chat_id", chat_id).eq("sender_id", sender_id).eq("active", True).execute()
        if not response.data:
            return "You don't have any active reminders to cancel."

        lowered = (text_content or "").lower()
        if any(w in lowered for w in (" all", "all ", "sab ", " saare", "saari", "everything", "pure ")):
            for r in response.data:
                supabase.table("reminders").update({"active": False}).eq("id", r["id"]).execute()
            return "Cancelled all your active reminders: " + ", ".join(
                f'"{r["message"]}"' for r in response.data
            )

        q_tokens = set(re.findall(r"\w+", lowered))
        # drop common cancel verbs so they don't dilute match
        q_tokens -= {"cancel", "karo", "delete", "remove", "reminder", "reminders", "the", "to", "from", "now", "on", "wale", "wali", "wala"}
        matched = []
        for r in response.data:
            msg = (r.get("message") or "").lower()
            if msg and (msg in lowered or lowered in msg):
                matched.append(r)
                continue
            msg_tokens = set(re.findall(r"\w+", msg))
            if q_tokens & msg_tokens:
                matched.append(r)

        if not matched:
            names = ", ".join(f'"{r["message"]}"' for r in response.data)
            return f"Which one do you mean? You have: {names}"

        for r in matched:
            supabase.table("reminders").update({"active": False}).eq("id", r["id"]).execute()
        return "Cancelled: " + ", ".join(f'"{r["message"]}"' for r in matched)
    except Exception as e:
        print(f"Error cancelling reminder: {e}")
        return "Couldn't cancel that just now — try again?"


def compute_next_daily_time(daily_times, after):
    """Kept for anything still calling the old name — every-day case only."""
    return compute_next_occurrence(daily_times, None, after)



# 2. DEFINE AI LOGIC
def get_ai_response(chat_id, sender_id, msg_time=None, history_limit=20):
    try:
        user_tz = get_user_timezone(sender_id)
        
        now_utc = datetime.now(timezone.utc)
        utc_time_str = now_utc.strftime("%A, %Y-%m-%d %I:%M %p UTC")
        world_clocks_str = get_world_clocks()

        if user_tz is not None:
            try:
                tz = ZoneInfo(user_tz)
                time_obj = now_utc.astimezone(tz)
                formatted_time = time_obj.strftime("%A, %I:%M %p")
                
                time_context = (
                    f"CURRENT UTC TIME: {utc_time_str}\n"
                    f"THIS SENDER'S LOCAL TIME: {formatted_time} ({user_tz}).\n\n"
                    f"REAL-TIME WORLD CLOCKS (ACCURATE DST CALCULATED):\n{world_clocks_str}\n\n"
                    "CRITICAL INSTRUCTIONS FOR TIME QUERIES:\n"
                    "1. Refer to the 'REAL-TIME WORLD CLOCKS' above for exact current times in major cities.\n"
                    "2. Do NOT confuse the sender's local timezone with the target city being asked about.\n"
                    "3. Do NOT invent or estimate offsets manually."
                )
            except Exception:
                pass # Fallback to default block below if tz fails

        else:
            time_context = (
                f"CURRENT UTC TIME: {utc_time_str}\n\n"
                f"REAL-TIME WORLD CLOCKS (ACCURATE DST CALCULATED):\n{world_clocks_str}\n\n"
                "(Note: The sender's specific local timezone offset is currently unknown)."
            )
        
        # Fetch chat history from cache
        raw_history = fetch_chat_history(chat_id, history_limit)

        # Fetch contacts and memory
        contacts_map, reverse_map = get_contacts_maps()
        memory_notes = get_group_memory(chat_id)
        
        # NEW: Fetch all timezones in one go to provide group context
        tz_map = {}
        try:
            tz_res = supabase.table("contacts").select("sender_id, timezone").execute()
            for row in tz_res.data:
                if row.get("timezone"):
                    tz_map[row["sender_id"]] = row["timezone"]
        except Exception as e:
            print(f"Error fetching timezone map: {e}")
        
        memory_block = ""
        if memory_notes:
            memory_block = "PERMANENT NOTES FOR THIS CHAT:\n" + "\n".join(f"- {note}" for note in memory_notes)

        # --- DYNAMIC GROUP METADATA FETCH (WITH ASYNC CACHE) ---
        tag_block = ""
        try:
            active_names = set()
            
            # 1. If it's a group, use the asynchronous cache system
            if "g.us" in chat_id:
                c_user, c_server = chat_id.split("@", 1)
                group_jid = build_jid(c_user, server=c_server)
                
                now = time.time()
                cache_entry = GROUP_METADATA_CACHE.get(chat_id)
                
                # Trigger background fetch if cache is empty or older than TTL
                if not cache_entry or (now - cache_entry["timestamp"] > GROUP_CACHE_TTL):
                    threading.Thread(
                        target=async_update_group_cache, 
                        args=(chat_id, group_jid, contacts_map),
                        daemon=True
                    ).start()
                
                # If we have valid cached data, use it instantly
                if cache_entry:
                    active_names = cache_entry["names"]
                    
            else:
                # 2. If it's a private chat, just list the user
                sender_name = contacts_map.get(sender_id, sender_id)
                active_names.add(sender_name)

            # 3. Format the names into the prompt
            if active_names:
                tag_block = (
                    "PEOPLE CURRENTLY IN THIS CHAT: " + ", ".join(sorted(active_names)) + ".\n"
                    "CRITICAL TAGGING RULE: If asked to tag everyone or specific people, you MUST generate a list using the EXACT items from the list above, preceded by '@' (e.g., @Zaheer, @73109738680505). Do NOT ask the user for proper names; if you see numbers, those numbers ARE their official handles."
                )
            elif "g.us" in chat_id:
                # FALLBACK: If cache is empty (e.g., first message ever), fallback to recent history instantly 
                # so the LLM doesn't wait for the background thread to finish.
                active_senders = {msg["sender_id"] for msg in raw_history if msg["sender_id"] != "mojo_agent"}
                active_names_fallback = {contacts_map.get(sid, sid) for sid in active_senders}
                if active_names_fallback:
                    tag_block = (
                        "PEOPLE RECENTLY ACTIVE IN THIS CHAT: " + ", ".join(sorted(active_names_fallback)) + ".\n"
                        "CRITICAL TAGGING RULE: If asked to tag everyone or specific people, you MUST generate a list using the EXACT items from the list above, preceded by '@' (e.g., @Zaheer, @73109738680505). Do NOT ask the user for proper names; if you see numbers, those numbers ARE their official handles."
                    )
                    
        except Exception as e:
            print(f"Error checking group metadata cache: {e}")
            # FALLBACK (Safety net)
            active_senders = {msg["sender_id"] for msg in raw_history if msg["sender_id"] != "mojo_agent"}
            active_names = {contacts_map.get(sid, sid) for sid in active_senders}
            if active_names:
                tag_block = (
                    "PEOPLE RECENTLY ACTIVE IN THIS CHAT: " + ", ".join(sorted(active_names)) + ".\n"
                    "CRITICAL TAGGING RULE: If asked to tag everyone or specific people, you MUST generate a list using the EXACT items from the list above, preceded by '@' (e.g., @Zaheer, @73109738680505). Do NOT ask the user for proper names; if you see numbers, those numbers ARE their official handles."
                )
        # -------------------------------------

        if "g.us" in chat_id:
            chat_context = (
                "ENVIRONMENT: GROUP CHAT.\n"
                "BEHAVIOR: Be informal, intelligent, friendly, and natural. Do NOT act like a rigid sales bot."
            )
        else:
            chat_context = (
                "ENVIRONMENT: PRIVATE CHAT (DIRECT MESSAGE).\n"
                "BEHAVIOR: Be professional, welcoming, and helpful."
            )

        messages = [
            {"role": "system", "content": f"You are Mojo, official AI Assistant for Mojo AI Agency (Founder: Muhammad Zaheer). Knowledge: {BUSINESS_KNOWLEDGE}\n\n{chat_context}\n\n{time_context}\n\n{tag_block}\n\n{memory_block}\n\nIMPORTANT RULES:\n1. History messages contain timestamps and sender names. Do NOT include timestamps or brackets around names in your replies.\n2. LANGUAGE MATCHING: If the user speaks Roman Urdu or Hindi (e.g. 'kya haal hai', 'batao'), you MUST reply in natural, casual Roman Urdu. Do not use overly formal words.\n3. CONTEXT: If you see [Quoted Message] in the prompt, the user is directly referring to that text or media. Address it specifically."}
        ]

        for msg in raw_history:
            content = msg["content"]
            
            # --- NEW FIX: Replace raw bot IDs with its name so the LLM understands ---
            if BOT_PN:
                content = content.replace(f"@{BOT_PN}", "@mojo")
            if BOT_LID:
                content = content.replace(f"@{BOT_LID}", "@mojo")
            # -------------------------------------------------------------------------
            
            time_str = ""
            if "created_at" in msg:
                try:
                    dt_str = msg["created_at"].replace('Z', '+00:00')
                    msg_dt = datetime.fromisoformat(dt_str)
                    time_str = f"[{msg_dt.strftime('%Y-%m-%d %I:%M %p')}] "
                except Exception:
                    pass

            if msg["role"] == "user":
                sender_name = contacts_map.get(msg["sender_id"], msg["sender_id"])
                
                # NEW: Inject this specific user's current local time natively into the history
                local_time_tag = ""
                s_tz = tz_map.get(msg["sender_id"])
                if s_tz:
                    try:
                        user_tz_obj = get_tzinfo(s_tz)
                        current_local = now_utc.astimezone(user_tz_obj).strftime('%I:%M %p')
                        local_time_tag = f" (Local Time: {current_local})"
                    except Exception:
                        pass
                
                content = f"{time_str}{sender_name}{local_time_tag}: {content}"
                
            messages.append({"role": msg["role"], "content": content})

        # --- CATCH-ALL FAILOVER TO GEMINI ---
        try:
            ai_response = client_ai.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.7
            )
            answer = ai_response.choices[0].message.content
        except Exception as primary_err:
            print(f"[PRIMARY AI FAILED] ({primary_err}). Falling back to Gemini ({GEMINI_MODEL})...")
            try:
                ai_response = client_gemini.chat.completions.create(
                    model=GEMINI_MODEL,
                    messages=messages,
                    temperature=0.7
                )
                answer = ai_response.choices[0].message.content
            except Exception as gemini_err:
                print(f"[GEMINI FALLBACK FAILED]: {gemini_err}")
                return "I'm having a little trouble connecting right now. Please try again in a moment."

        return answer or "Got that — thinking it over 🤔"

    except Exception as e:
        print(f"Error in get_ai_response: {e}")
        return "I'm having a little trouble connecting right now. Please try again in a moment."

# 3. INITIALIZE WHATSAPP CLIENT
print("--- AIMOJO WHATSAPP AI AGENT ---")
print("Connecting... PLEASE WAIT FOR THE QR CODE.")

# Create Client — session storage.
# Local default (SESSION_DB_URL unset): fine on your own machine, wiped on any host
# with an ephemeral filesystem. Setting SESSION_DB_URL to a Postgres connection
# string is genuinely supported (confirmed straight from the compiled whatsmeow
# extension — it embeds separate SQL for both sqlite3 and postgres dialects).
#
# BUT: this isn't just app data — it's whatsmeow's own signal-protocol session
# store (sessions, prekeys, sender keys), which gets read/written MULTIPLE TIMES
# PER MESSAGE on the actual crypto path. Local sqlite = a sub-millisecond disk
# read each time. A cross-region Postgres pooler = a real network round trip each
# time, and those add up fast — this is almost certainly why replies got slow and
# the connection got flaky right after switching this on.
# Recommended: leave this UNSET while developing locally (your PC already has a
# real disk, so there's no restart-survival problem to solve here) and only set
# it when actually deploying somewhere with an ephemeral filesystem, like Render.

# Resolve session path (local file wins for testing; otherwise pull from Supabase Storage)
session_db = get_session_path()
client = NewClient(session_db)
# Inject the LLM client and model name into the admin commands module
admin_commands.init(supabase, client, get_contacts_maps, client_ai, MODEL_NAME)

# --- Agentic tools + loop ---
import agent_tools
import agent_loop

agent_tools.init_tools(
    supabase=supabase,
    client_ai=client_ai,
    model_name=MODEL_NAME,
    business_knowledge=BUSINESS_KNOWLEDGE,
    default_timezone=DEFAULT_TIMEZONE,
    owner_sender_id=os.getenv("OWNER_SENDER_ID") or "",
    get_contacts_maps=get_contacts_maps,
    get_user_timezone=get_user_timezone,
    set_user_timezone=set_user_timezone,
    get_tzinfo=get_tzinfo,
    handle_reminder_request=handle_reminder_request,
    list_reminders=list_reminders,
    cancel_reminders=cancel_reminders,
    get_group_memory=get_group_memory,
    send_proactive_message=send_proactive_message,
    file_ops_module=file_ops,
    compute_next_occurrence=compute_next_occurrence,
    extract_reminder_data_via_ai=extract_reminder_data_via_ai,
    generate_reminder_confirmation=generate_reminder_confirmation,
)

agent_loop.init_agent(
    client_ai=client_ai,
    client_gemini=client_gemini,
    model_name=MODEL_NAME,
    gemini_model=GEMINI_MODEL,
    business_knowledge=BUSINESS_KNOWLEDGE,
    default_timezone=DEFAULT_TIMEZONE,
    get_user_timezone=get_user_timezone,
    get_world_clocks=get_world_clocks,
    fetch_chat_history=fetch_chat_history,
    get_contacts_maps=get_contacts_maps,
    get_group_memory=get_group_memory,
    get_tzinfo=get_tzinfo,
)

# Start health endpoint + session uploader + optional self-ping
start_health_server()
start_session_uploader(interval_seconds=300)   # every 5 min is safe
start_self_ping(interval_seconds=600)         # optional, every 10 min

# Full agent log → local file + periodic overwrite to OneDrive/MojoAgent/mojo_agent_live.log
import mojo_logging
mojo_logging.setup_logging()
mojo_logging.start_onedrive_log_sync(file_ops, interval_sec=90)


def send_reaction(message, chat_id, emoji):
    """React to a message (⏳ while working on a reply, ✅ once it's sent) instead
    of leaving someone staring at a chat with no signal anything's happening.
    Confirmed via your own diagnostics: ReactionMessage.key is a WACommon.MessageKey
    with remoteJID, fromMe, ID, and participant fields.
    The import is done INSIDE this function, not at the top of the file, on
    purpose — if the exact module path below turns out to be off for your
    installed neonize version, this fails quietly and only disables reactions,
    instead of crashing the entire bot on startup the way a bad top-level import
    would."""
    try:
        # type(message.Message) grabs the exact Message class your neonize version is using
        MessageClass = type(message.Message)
        
        # Create a fresh, empty message wrapper
        reaction = MessageClass()
        
        # Drill down into the structure your diagnostic confirmed and set the fields
        reaction.reactionMessage.key.remoteJID = chat_id
        reaction.reactionMessage.key.fromMe = False
        reaction.reactionMessage.key.ID = message.Info.ID
        reaction.reactionMessage.text = emoji
        
        # Send the reaction
        client.send_message(message.Info.MessageSource.Chat, reaction)
    except Exception as e:
        print(f"Reaction not sent (safe to ignore): {e}")


# --- URL extraction helpers (must stay above the event handler) ---
def extract_urls_from_text(*parts) -> list:
    """Pull http(s) URLs from plain text fragments. Fast, no object walking."""
    import re as _re
    found, seen = [], set()
    pat = _re.compile(r"https?://[^\s<>\"'\]\)]+")
    for part in parts:
        if not part or not isinstance(part, str):
            continue
        for u in pat.findall(part):
            u = u.rstrip(".,;:!?")
            if u not in seen:
                seen.add(u)
                found.append(u)
    return found


def _safe_str_attr(obj, *names):
    if obj is None:
        return ""
    for n in names:
        try:
            v = getattr(obj, n, None)
            if isinstance(v, str) and v.strip():
                return v.strip()
        except Exception:
            pass
    return ""


def extract_quoted_text_and_urls(ctx) -> tuple:
    """
    Return (quoted_text, urls) from ContextInfo without recursive dir()-walking
    (that was freezing process_message for 60s+ on neonize protobufs).
    Hardened: also pulls URLs from quoted conversation / extendedText / nested
    link previews so "Iski details" while quoting a link actually gets the URL.
    """
    if ctx is None:
        return "", []

    urls = []
    quoted_text = ""

    # 1) Direct string fields on contextInfo (link previews / matched text)
    for name in (
        "matchedText", "MatchedText",
        "conversionSource", "ConversionSource",
        "entryPointConversionSource", "EntryPointConversionSource",
    ):
        s = _safe_str_attr(ctx, name)
        if s:
            urls.extend(extract_urls_from_text(s))
            # matchedText is often the URL itself when user pastes a link
            if not quoted_text and s.startswith("http"):
                quoted_text = s

    # 2) externalAdReply / link preview card (current message)
    for reply_name in ("externalAdReply", "ExternalAdReply", "extAdReply", "ExtAdReply"):
        reply = getattr(ctx, reply_name, None)
        if reply is None:
            continue
        for sub in (
            "originalURL", "OriginalURL", "originalUrl",
            "sourceURL", "SourceURL", "sourceUrl",
            "mediaURL", "MediaURL", "mediaUrl",
            "thumbnailURL", "ThumbnailURL",
            "title", "Title", "body", "Body", "containsAutoReply",
        ):
            s = _safe_str_attr(reply, sub)
            if s:
                urls.extend(extract_urls_from_text(s))
                if not quoted_text and not s.startswith("http"):
                    quoted_text = s

    # 3) quotedMessage payload — dig hard for the URL the user is referring to
    quoted = getattr(ctx, "quotedMessage", None) or getattr(ctx, "QuotedMessage", None)
    if quoted is not None:
        # plain conversation
        q_conv = _safe_str_attr(quoted, "conversation", "Conversation")
        # extended text
        q_ext = getattr(quoted, "extendedTextMessage", None) or getattr(quoted, "ExtendedTextMessage", None)
        q_ext_text = _safe_str_attr(q_ext, "text", "Text") if q_ext else ""
        # also dig preview on the *quoted* extended message
        if q_ext is not None:
            q_ctx = getattr(q_ext, "contextInfo", None) or getattr(q_ext, "ContextInfo", None)
            if q_ctx is not None:
                for name in ("matchedText", "MatchedText"):
                    mt = _safe_str_attr(q_ctx, name)
                    if mt:
                        urls.extend(extract_urls_from_text(mt))
                for reply_name in ("externalAdReply", "ExternalAdReply", "extAdReply", "ExtAdReply"):
                    reply = getattr(q_ctx, reply_name, None)
                    if reply is None:
                        continue
                    for sub in (
                        "originalURL", "OriginalURL", "originalUrl",
                        "sourceURL", "SourceURL", "sourceUrl",
                        "mediaURL", "MediaURL", "mediaUrl",
                    ):
                        s = _safe_str_attr(reply, sub)
                        if s:
                            urls.extend(extract_urls_from_text(s))
        # captions on media quotes
        q_img = getattr(quoted, "imageMessage", None) or getattr(quoted, "ImageMessage", None)
        q_vid = getattr(quoted, "videoMessage", None) or getattr(quoted, "VideoMessage", None)
        q_doc = getattr(quoted, "documentMessage", None) or getattr(quoted, "DocumentMessage", None)
        q_cap = (
            _safe_str_attr(q_img, "caption", "Caption")
            or _safe_str_attr(q_vid, "caption", "Caption")
            or _safe_str_attr(q_doc, "caption", "Caption", "title", "Title", "fileName", "FileName")
        )
        # Some clients put the URL only in the quoted conversation / extended text
        piece = q_conv or q_ext_text or q_cap or ""
        if piece:
            quoted_text = piece if not quoted_text else quoted_text
            urls.extend(extract_urls_from_text(piece))
        # Final fallback: if quoted object has a string-ish canonicalUrl-style attr
        for attr in ("canonicalUrl", "CanonicalUrl", "url", "URL", "href"):
            s = _safe_str_attr(quoted, attr)
            if s:
                urls.extend(extract_urls_from_text(s))

    # dedupe urls
    out, seen = [], set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return quoted_text, out


def urls_from_recent_history(chat_id, limit=12) -> list:
    """Most recent URLs the user (or anyone) posted in this chat."""
    try:
        hist = fetch_chat_history(chat_id, limit) or []
        found = []
        for msg in reversed(hist):  # newest first after reverse of chrono list... 
            # hist is chronological; walk newest-first
            pass
        for msg in reversed(hist):
            found.extend(extract_urls_from_text(msg.get("content") or ""))
        out, seen = [], set()
        for u in found:
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out
    except Exception as e:
        print(f"[URL history fallback] {e}")
        return []


# 4. WHATSAPP MESSAGE HANDLER (Using Decorators)
@client.event(MessageEv)
def on_message(client: NewClient, message: MessageEv):
    # Instantly hand off the heavy lifting to a background thread — daemon=True so
    # a thread that's still mid-reply when the process is stopped doesn't hang it.
    try:
        import logging as _lg
        _lg.getLogger("mojo").info("incoming MessageEv — dispatching process_message")
    except Exception:
        pass
    threading.Thread(target=process_message, args=(client, message), daemon=True).start()

def _jid_user(jid):
    """Safely extract the User (number/LID) from a JID."""
    if not jid: 
        return ""
    return str(getattr(jid, "User", getattr(jid, "user", "")) or "").strip()

def _jid_server(jid):
    """Safely extract the Server (s.whatsapp.net, g.us, lid) from a JID."""
    if not jid: 
        return ""
    return str(getattr(jid, "Server", getattr(jid, "server", "")) or "").strip().lower()

def process_message(client, message):
    try:
        _process_message_inner(client, message)
    except Exception as e:
        import logging as _lg
        _lg.getLogger("mojo").exception("process_message CRASHED: %s", e)
        print(f"[PROCESS_MESSAGE CRASH] {e}")
        traceback.print_exc()

def _process_message_inner(client, message):
    is_from_me = False
    try:
        if hasattr(message.Info, "IsFromMe"): is_from_me = message.Info.IsFromMe
        elif hasattr(message.Info, "fromMe"): is_from_me = message.Info.fromMe
        elif hasattr(message, "from_me"): is_from_me = message.from_me
    except: pass
    
    # Learn bot LID from our own outbound traffic, then ignore the message
    if is_from_me:
        if not BOT_PN:
            refresh_bot_identities(client)
        learn_bot_lid_from_message(message, BOT_PN)
        return

    # Broadcast/bulk sends
    try:
        if message.Info.Multicast:
            return
    except AttributeError:
        pass

    # Pull WhatsApp's own display name for the sender
    push_name = None
    try:
        if hasattr(message.Info, "Pushname"): 
            push_name = message.Info.Pushname
        elif hasattr(message.Info, "PushName"): 
            push_name = message.Info.PushName
        elif hasattr(message.Info, "VerifiedName"):
            push_name = message.Info.VerifiedName
    except: pass

    # 1. EXTRACT ORIGINAL TEXT FIRST (Isolate from quotes)
    original_text = ""
    if message.Message.conversation:
        original_text = message.Message.conversation
    elif message.Message.extendedTextMessage and message.Message.extendedTextMessage.text:
        original_text = message.Message.extendedTextMessage.text
    elif message.Message.imageMessage and message.Message.imageMessage.caption:
        original_text = message.Message.imageMessage.caption
    elif message.Message.videoMessage and message.Message.videoMessage.caption:
        original_text = message.Message.videoMessage.caption
    elif message.Message.documentMessage and message.Message.documentMessage.caption:
        original_text = message.Message.documentMessage.caption

    text_content = original_text
    message_urls = []

    media_kind = get_media_kind(message)
    target_media_msg = None

    # --- CHECK FOR QUOTED MESSAGES AND TAGS ---
    quoted_text = ""
    is_reply_to_bot = False
    is_bot_mentioned = False
    # Dynamically fetch the bot's current ID from the active session
    try:
        bot_jid = client.get_me().JID.User
    except Exception:
        bot_jid = ""

    ctx = None
    try:
        # WhatsApp buries contextInfo inside whatever message type is currently being sent
        if message.Message.extendedTextMessage and message.Message.extendedTextMessage.contextInfo:
            ctx = message.Message.extendedTextMessage.contextInfo
        elif message.Message.imageMessage and message.Message.imageMessage.contextInfo:
            ctx = message.Message.imageMessage.contextInfo
        elif message.Message.videoMessage and message.Message.videoMessage.contextInfo:
            ctx = message.Message.videoMessage.contextInfo
        elif message.Message.documentMessage and message.Message.documentMessage.contextInfo:
            ctx = message.Message.documentMessage.contextInfo
        elif message.Message.audioMessage and message.Message.audioMessage.contextInfo:
            ctx = message.Message.audioMessage.contextInfo
        elif getattr(message.Message, "stickerMessage", None) and message.Message.stickerMessage.contextInfo:
            ctx = message.Message.stickerMessage.contextInfo
    except AttributeError:
        pass

    # Use robust context extraction helper
    ctx = get_context_info(message.Message)

    if ctx:
        if ctx.participant and bot_jid and bot_jid in ctx.participant:
            is_reply_to_bot = True
            
        if hasattr(ctx, "mentionedJID") and ctx.mentionedJID:
            for jid in ctx.mentionedJID:
                if bot_jid and bot_jid in jid:
                    is_bot_mentioned = True
                    
        # Quoted message + URL extraction (FAST — no recursive dir() walks)
        quoted = getattr(ctx, "quotedMessage", None) or getattr(ctx, "QuotedMessage", None)
        q_text, q_urls = extract_quoted_text_and_urls(ctx)
        if q_text:
            quoted_text = q_text
        if quoted is not None and not media_kind:
            # media-kind detection on quoted payload (same as before, light)
            q_audio = getattr(quoted, "audioMessage", None) or getattr(quoted, "AudioMessage", None)
            q_img = getattr(quoted, "imageMessage", None) or getattr(quoted, "ImageMessage", None)
            q_sticker = getattr(quoted, "stickerMessage", None) or getattr(quoted, "StickerMessage", None)
            q_vid = getattr(quoted, "videoMessage", None) or getattr(quoted, "VideoMessage", None)
            q_doc = getattr(quoted, "documentMessage", None) or getattr(quoted, "DocumentMessage", None)
            if q_audio and getattr(q_audio, "mimetype", None):
                media_kind = "audio"
                target_media_msg = quoted
            elif q_img and getattr(q_img, "mimetype", None):
                media_kind = "image"
                target_media_msg = quoted
            elif q_sticker and getattr(q_sticker, "mimetype", None):
                media_kind = "sticker"
                target_media_msg = quoted
            elif q_vid and getattr(q_vid, "mimetype", None):
                media_kind = "gif" if getattr(q_vid, "gifPlayback", False) else "video"
                target_media_msg = quoted
            elif q_doc and (getattr(q_doc, "mimetype", None) or getattr(q_doc, "fileName", None)):
                media_kind = "document"
                target_media_msg = quoted

    # Append quoted text for AI context
    if quoted_text:
        if not text_content:
            text_content = f"[Quoted Message]: {quoted_text}"
        else:
            text_content = f"{text_content}\n\n[Quoted Message]: {quoted_text}"

    # URLs from current body + quote/preview
    message_urls = extract_urls_from_text(original_text, quoted_text)
    if ctx is not None:
        _, ctx_urls = extract_quoted_text_and_urls(ctx)
        for u in ctx_urls:
            if u not in message_urls:
                message_urls.append(u)

    # History fallback: ONLY when user is clearly asking about a previously shared link.
    # Do NOT inject on pure @mentions, short replies, or unrelated questions (weather, etc.).
    # Broad "batao"/"ye" alone is NOT enough — requires link-referential intent.
    _ref = (original_text or "").strip().lower()
    # Strip @mentions so pure tags don't look "short + contextual"
    _ref_clean = re.sub(r"@\S+", " ", _ref).strip()
    _link_intent = any(w in _ref_clean for w in (
        "detail", "details", "iska", "is ka", "is ki", "uska", "us ki",
        "about this", "about it", "about the", "this link", "this site",
        "this url", "ye link", "ye site", "ye url", "is link", "is site",
        "open this", "fetch this", "fetch latest", "latest repo",
    ))
    # Also allow when they quote a message that itself contained a URL (quoted_text has http)
    _quoted_has_url = bool(quoted_text and re.search(r"https?://", quoted_text))
    _referential = _link_intent or _quoted_has_url
    if not message_urls and _referential:
        try:
            _ms = getattr(message.Info, "MessageSource", None)
            _cj = getattr(_ms, "Chat", None)
            _sj = getattr(_ms, "Sender", None)
            _sa = getattr(_ms, "SenderAlt", None)
            _cu, _cs = _jid_user(_cj), _jid_server(_cj)
            if _cs == "lid":
                _alt = _jid_user(_sa) or _jid_user(_sj)
                _early_chat = f"{_alt}@s.whatsapp.net" if _alt and str(_alt).isdigit() else (f"{_cu}@{_cs}" if _cu and _cs else None)
            else:
                _early_chat = f"{_cu}@{_cs}" if _cu and _cs else None
        except Exception:
            _early_chat = None
        if _early_chat:
            hist_urls = urls_from_recent_history(_early_chat, limit=15)
            if not hist_urls and _early_chat in LAST_URL_BY_CHAT:
                hist_urls = [LAST_URL_BY_CHAT[_early_chat]]
            if not hist_urls:
                try:
                    _raw = f"{_cu}@{_cs}" if _cu and _cs else None
                    if _raw and _raw != _early_chat and _raw in LAST_URL_BY_CHAT:
                        hist_urls = [LAST_URL_BY_CHAT[_raw]]
                except Exception:
                    pass
            if hist_urls:
                message_urls = [hist_urls[0]]
                print(f"[URL HISTORY FALLBACK] chat={_early_chat} -> {message_urls}")
                try:
                    import logging as _lg
                    _lg.getLogger("mojo").info("URL HISTORY FALLBACK %s -> %s", _early_chat, message_urls)
                except Exception:
                    pass

    if message_urls:
        missing = [u for u in message_urls if u not in (text_content or "")]
        if missing:
            text_content = (text_content or "") + "\n\n[Linked URL]: " + " ".join(missing)
        print(f"[URL DETECT] {message_urls}")
        # Remember for later "Details"/"iska" replies in this chat
        try:
            _ms2 = getattr(message.Info, "MessageSource", None)
            _cj2 = getattr(_ms2, "Chat", None)
            _cu2, _cs2 = _jid_user(_cj2), _jid_server(_cj2)
            _cid = f"{_cu2}@{_cs2}" if _cu2 and _cs2 else None
            if _cs2 == "lid":
                _sa2 = getattr(_ms2, "SenderAlt", None)
                _sj2 = getattr(_ms2, "Sender", None)
                _alt2 = _jid_user(_sa2) or _jid_user(_sj2)
                if _alt2 and str(_alt2).isdigit():
                    _cid = f"{_alt2}@s.whatsapp.net"
            if _cid and message_urls:
                LAST_URL_BY_CHAT[_cid] = message_urls[0]
        except Exception:
            pass
        try:
            import logging as _lg
            _lg.getLogger("mojo").info("URL DETECT %s quoted=%r", message_urls, (quoted_text or "")[:120])
        except Exception:
            pass
    else:
        try:
            import logging as _lg
            _qm = getattr(ctx, "quotedMessage", None) or getattr(ctx, "QuotedMessage", None) if ctx else None
            _sid = getattr(ctx, "stanzaId", None) or getattr(ctx, "StanzaID", None) or getattr(ctx, "stanzaID", None) if ctx else None
            _lg.getLogger("mojo").info(
                "URL MISS original=%r quoted=%r has_ctx=%s quotedMessage=%s stanzaId=%s",
                (original_text or "")[:80],
                (quoted_text or "")[:120],
                bool(ctx),
                _qm is not None,
                _sid,
            )
        except Exception:
            pass

    if not text_content and not media_kind:
        return

    # Determine Group vs Private
    is_group = False
    try:
        if message.Info.MessageSource.Chat.Server == "g.us":
            is_group = True
    except AttributeError:
        if "g.us" in str(message.Info.MessageSource.Chat):
            is_group = True

    # Get Identifiers (With LID to MSISDN Resolution)
    msg_source = getattr(message.Info, "MessageSource", None)
    
    chat_jid = getattr(msg_source, "Chat", None)
    sender_jid = getattr(msg_source, "Sender", None)
    sender_alt_jid = getattr(msg_source, "SenderAlt", None)

    # 1. Resolve Sender Number and LID dynamically
    sender_user = _jid_user(sender_jid)
    sender_server = _jid_server(sender_jid)
    alt_user = _jid_user(sender_alt_jid)
    alt_server = _jid_server(sender_alt_jid)
    
    sender_lid = None
    sender_number = "Unknown"

    # Handle incoming route (Standard vs LID)
    if sender_server == "lid":
        sender_lid = sender_user
        sender_number = alt_user if alt_user else "Unknown"
    else:
        sender_number = sender_user or "Unknown"
        sender_lid = alt_user if alt_server == "lid" else sender_number # Fallback
        
    # Set the variables for Supabase
    db_sender_id = sender_lid
    db_sender_num = sender_number
        
    # Determine the active addressing mode for this specific message
    is_lid_mode = (sender_server == "lid")

    # 2. Resolve Chat ID (Group vs Private)
    chat_user = _jid_user(chat_jid)
    chat_server = _jid_server(chat_jid)
    
    # If it's a private chat routed via LID, normalize the chat_id back to standard MSISDN
    if chat_server == "lid" and sender_number != "Unknown":
        chat_id = f"{sender_number}@s.whatsapp.net"
    elif chat_user and chat_server:
        chat_id = f"{chat_user}@{chat_server}"
    else:
        chat_id = sender_number

    if "status@broadcast" in chat_id:
        return

    if admin_commands.is_admin_message(sender_number, is_group):
        # --- /uploadimg needs media (caption or reply-to-image) ---
        low = (original_text or "").strip().lower()
        is_upload = low.startswith("/uploadimg") or low.startswith("/upload ")

        if is_upload and media_kind in ("image", "sticker", "document", "video", "gif"):
            # optional custom name from args
            name_hint = ""
            if " " in (original_text or "").strip():
                name_hint = (original_text or "").strip().split(maxsplit=1)[1].strip()
            name_hint = re.sub(r"[^\w\-]+", "_", name_hint)[:40] if name_hint else ""

            suffix = {
                "image": ".jpg",
                "sticker": ".webp",
                "video": ".mp4",
                "gif": ".mp4",
                "document": os.path.splitext(
                    getattr(
                        getattr(message.Message, "documentMessage", None),
                        "fileName",
                        None,
                    )
                    or "file.bin"
                )[1]
                or ".bin",
            }.get(media_kind, ".bin")

            prefix = name_hint or f"wa_{media_kind}"
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp_path = tmp.name
                msg_to_dl = target_media_msg if target_media_msg else message.Message
                client.download_any(msg_to_dl, path=tmp_path)

                result = file_ops.write_and_upload_file(
                    tmp_path,
                    filename_prefix=prefix,
                    ext=suffix.lstrip("."),
                    delete_local=True,
                )
                tmp_path = None  # already removed by helper
                admin_reply = (
                    f"✅ Uploaded → OneDrive/{result['folder']}/{result['remote_name']}\n"
                    f"{result.get('webUrl') or ''}"
                )
            except Exception as e:
                print(f"[UPLOADIMG] {e}")
                admin_reply = f"❌ Upload failed: {e}"
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

            client.reply_message(admin_reply, message)
            return

        # normal text admin commands
        admin_reply = admin_commands.handle_admin_command(text_content)
        if admin_reply is not None:
            print(f"[ADMIN COMMAND] {text_content}")
            client.reply_message(admin_reply, message)
            return

    if not admin_commands.get_global_setting("agent_enabled", default=True):
        return

    insert_chat_message(chat_id, db_sender_id, "user", text_content, sender_num=db_sender_num)

    upsert_contact(db_sender_id, push_name, db_sender_num)

    # ==========================================
    # LOGIC 2: CHECK IF WE SHOULD REPLY
    # ==========================================
    
    # C. Fallback textual tag check: Catch standard IDs or the bot's name
    if bot_jid and f"@{bot_jid}" in original_text.lower():
        is_bot_mentioned = True
        
    # Catch textual tags in LID groups where the bot's LID isn't known
    if "@mojo" in original_text.lower():
        is_bot_mentioned = True

    if not BOT_PN:
        refresh_bot_identities(client)
    
    learn_bot_lid_from_message(message, BOT_PN)

    is_reply_to_bot = is_quote_of_bot(ctx, BOT_PN, BOT_LID, bot_jid)

    REACTION_MEDIA = {"sticker", "image", "gif"}  # optional: add "video" if you want

    has_quote = bool(ctx and getattr(ctx, "quotedMessage", None))

    # Group: participant is often bot LID — if LID unknown, match quoted text to our last replies
    if (
        not is_reply_to_bot
        and is_group
        and has_quote
        and quoted_text
        and media_kind in ("sticker", "image", "gif")
    ):
        if quoted_matches_recent_bot_reply(chat_id, quoted_text):
            is_reply_to_bot = True
            print("[REACTION] Quote matched recent bot reply via chat_history")

    native = is_bot_natively_mentioned(ctx, BOT_PN, BOT_LID)
    text_hit = False
    if original_text:
        low = original_text.lower()
        if BOT_PN and f"@{BOT_PN}" in low:
            text_hit = True
        if BOT_LID and f"@{BOT_LID}" in low:
            text_hit = True
        if "@mojo" in low or "@aimojo" in low:
            text_hit = True

    is_bot_mentioned = native or text_hit

    REACTION_MEDIA = {"sticker", "image", "gif"}
    has_quote = bool(ctx and getattr(ctx, "quotedMessage", None))

    # Groups: must be a reply to the bot.
    # Private: any sticker/image/GIF that quotes *something* counts as a reaction
    # (DMs often have no participant on the quote).
    is_media_reaction_to_bot = (
        media_kind in REACTION_MEDIA
        and (
            is_reply_to_bot
            or (not is_group and has_quote)
        )
    )

    # --- temporary debug (remove after it works) ---
    print(
        f"[DEBUG REACTION] media_kind={media_kind!r} "
        f"ctx={bool(ctx)} is_reply_to_bot={is_reply_to_bot} "
        f"is_media_reaction_to_bot={is_media_reaction_to_bot} "
        f"is_group={is_group} has_quote={has_quote} "
        f"BOT_PN={BOT_PN} BOT_LID={BOT_LID} "
        f"participant={getattr(ctx, 'participant', None) if ctx else None}"
    )
    # -----------------------------------------------

    if is_group:
        if not (is_bot_mentioned or is_media_reaction_to_bot):
            return
        if is_media_reaction_to_bot and not is_bot_mentioned:
            print(f"\n[GROUP MEDIA REACTION TO BOT] {media_kind} from {sender_number}")
        else:
            print(f"\n[GROUP WAKE WORD DETECTED] from {sender_number}")
    else:
        print(f"\n[PRIVATE MESSAGE] from {sender_number}")

    if is_media_reaction_to_bot:
        q = (quoted_text or "").strip()
        kind_label = {
            "sticker": "sticker",
            "image": "image",
            "gif": "GIF",
        }.get(media_kind, media_kind or "media")
        reaction_note = (
            f"[User is reacting to your previous message with a {kind_label}. "
            f"Treat this as their reaction/feedback to what you said"
            + (f': "{q[:300]}"' if q else "")
            + ". Respond naturally to the reaction — short, in-character, same language vibe.]"
        )
        if text_content.strip():
            text_content = f"{reaction_note}\n\nUser caption: {text_content}"
        else:
            text_content = reaction_note

    # --- TIMESTAMP EXTRACTION & OFFLINE SPAM GUARD ---
    msg_time = time.time()
    try:
        ts = message.Info.Timestamp
        extracted_time = ts.timestamp() if hasattr(ts, "timestamp") else float(ts)
        
        # If the timestamp is too large (milliseconds), convert it to seconds
        if extracted_time > 10000000000:
            extracted_time /= 1000
            
        msg_time = extracted_time
    except (AttributeError, TypeError, ValueError):
        pass

    # If the message is older than 5 minutes (300 seconds), skip replying.
    if time.time() - msg_time > 300:
        print(f"[STALE MESSAGE SKIPPED] from {sender_number} (Age: {int(time.time() - msg_time)}s)")
        return

    with _timed("1. feature flags + memory + timezone"):
        if admin_commands.is_feature_enabled(chat_id, "chat_memory"):
            # FIX: Mapped to db_sender_id (LID)
            maybe_save_memory(chat_id, db_sender_id, text_content)

        # ==========================================
        # LOGIC 3: GENERATE AND SAVE AI REPLY (AGENTIC)
        # ==========================================

        lowered_text = text_content.lower()

        history_limit = detect_summary_history_limit(text_content)
        if history_limit is not None:
            print(f"[DYNAMIC CONTEXT EXTENSION] Fetching {history_limit} messages for summary request.")
        else:
            history_limit = 20

        ai_answer = None

        # Fast path: explicit timezone offset setting (still keyword for reliability)
        tz_reply = maybe_set_timezone(db_sender_id, text_content)
        if tz_reply:
            ai_answer = tz_reply

    if not ai_answer:
        if media_kind:
            with _timed("2. media handling"):
                ai_answer = handle_media_message(
                    message, media_kind, chat_id, db_sender_id, text_content,
                    target_media_msg, msg_time, history_limit,
                    is_reaction_to_bot=is_media_reaction_to_bot,
                )
        else:
            # Agentic path — tools handle reminders, knowledge, web, memory, etc.
            if admin_commands.is_feature_enabled(chat_id, "ai_chat") or admin_commands.is_feature_enabled(chat_id, "reminders"):
                with _timed("3. agent loop (tools + LLM)"):
                    from agent_loop import run_agent
                    import logging as _logging
                    _logging.getLogger("mojo").info(
                        "run_agent chat=%s sender=%s urls=%s text=%r",
                        chat_id, db_sender_id, message_urls, (text_content or "")[:200],
                    )
                    ai_answer = run_agent(
                        chat_id=chat_id,
                        sender_id=db_sender_id,
                        sender_num=db_sender_num,
                        history_limit=history_limit,
                        is_group=is_group,
                        msg_time=msg_time,
                        force_urls=message_urls or None,
                        latest_user_text=text_content,
                    )
            else:
                print(f"[FEATURE OFF] 'ai_chat'/'reminders' disabled for {chat_id}.")
                if admin_commands.has_any_feature_enabled(chat_id):
                    ai_answer = "💬 AI text chat is currently turned off for this chat."
                else:
                    return

    if not ai_answer:
        return

    print(f"[{(push_name or sender_number).upper()}]: {lowered_text}")
    print(f"[AI RESPONSE]: {ai_answer}")

    _, reverse_map = get_contacts_maps()
    ai_answer = resolve_mentions(ai_answer, reverse_map)

    insert_chat_message(chat_id, "mojo_agent", "assistant", ai_answer)

    ai_answer = re.sub(r'^\[\d{4}-\d{2}-\d{2}\s\d{1,2}:\d{2}\s[AP]M\]\s*', '', ai_answer).strip()

    client.reply_message(ai_answer, message, mentions_are_lids=is_lid_mode)


# 5. START THE BOT
if __name__ == "__main__":
    # Start the reminder loop in the background
    scheduler_thread = threading.Thread(target=reminder_scheduler, daemon=True)
    scheduler_thread.start()
    print("[SYSTEM] Background reminder scheduler started.")

    try:
        client.connect()
        refresh_bot_identities(client)
        # Persist the session as soon as we are connected (covers first QR scan too)
        _upload_session_to_bucket()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")