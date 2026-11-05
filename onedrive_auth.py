"""
onedrive_auth.py — OneDrive (Microsoft Graph) OAuth for the whole bot: cached, single-flight, rotation-safe.

Why (prod log 2026-11-03): `[file_ops] NOTE: Microsoft rotated refresh_token — update ONEDRIVE_REFRESH_TOKEN`
appeared 35 times in one day. Two root causes in the old _onedrive_access_token():
  1. Every OneDrive call POSTed to the token endpoint — no access-token caching (an access token lives
     ~60-90 min) — so each file operation, each RAG sync file and each upload rotated the refresh token.
  2. The rotated refresh token was thrown away and the ORIGINAL one from the env var was reused forever.
     Per Microsoft's docs the old token is not revoked on use, so this "works" — but it works only until
     90 days after that original token was minted. Then every OneDrive feature (RAG sync, resume, logs,
     file tools) dies at once and needs a manual re-auth + redeploy. A time bomb, not a warning.

Behaviour now:
  * Access token cached until ~2 min before expiry; refreshes are single-flight (one thread refreshes,
    the others reuse the result), so a RAG sync burst costs one token call, not hundreds.
  * Every rotated refresh token is persisted encrypted (secret_store -> Supabase `oauth_tokens`, in-process
    memory as the floor) and preferred on the next refresh. Each redemption yields a token with a fresh
    90 days.
  * The env var is the seed. If it changes (you re-mint a token, even for another account) the stored
    token is ignored automatically (seed hash mismatch), so a manual re-auth always wins.
  * Candidates are tried newest-first; if one is rejected (invalid_grant) the next is tried before giving
    up. Transient failures (429/5xx/timeouts) retry with backoff and never mark a token bad.
  * Permanent failure raises OneDriveAuthError with the exact fix, is logged once per cool-off window, and
    is not hammered (5 min) — instead of one failing token POST per file operation.
  * Owner alert: the first permanent failure messages the owner (via an injected handler, see
    set_alert_handler) with the exact fix, repeats every ONEDRIVE_ALERT_REPEAT_H hours (default 12) while
    it stays down — also across restarts — and sends one "reconnected" message when it recovers.
    Transient outages, startup grace and "not configured" never alert. ONEDRIVE_ALERTS=0 disables.
Security: the refresh token is as sensitive as the env var it replaces, so it is stored ENCRYPTED (Fernet,
key in env TOKEN_ENCRYPTION_KEY) in its own RLS-locked table `oauth_tokens` (sql/oauth_tokens.sql) via
secret_store.py. Fail closed: with no key nothing is written to the database (memory only). The plaintext
copy that earlier versions kept in `kv_cache` is migrated and then blanked. Tokens are never logged (only a
short hash).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import requests

import secret_store

log = logging.getLogger("mojo.onedrive")

TOKEN_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
SCOPE = "https://graph.microsoft.com/Files.ReadWrite offline_access"
_PROVIDER = "onedrive"
_LEGACY_KV_KEY = "oauth:onedrive:rt"   # plaintext copy kept by the previous version (migrated, then blanked)
_SKEW_S = 120                    # refresh this long before the access token expires
_DEAD_COOLOFF_S = 300.0
_STARTUP_GRACE_S = 120.0         # Supabase is wired in a moment after boot; see get_access_token()
_T0 = time.time()
_TRANSIENT_RETRIES = 3
_PERMANENT_ERRORS = frozenset({"invalid_grant", "interaction_required", "invalid_client", "unauthorized_client",
                               "invalid_request", "consent_required"})

_ALERTS_ON = (os.getenv("ONEDRIVE_ALERTS", "1").strip() != "0")
_ALERT_REPEAT_S = float(os.getenv("ONEDRIVE_ALERT_REPEAT_H", "12")) * 3600
_ALERT_KV_KEY = "oauth:onedrive:alert"     # {"first": ts, "last": ts} while an outage is open (survives restarts)
_ALERT_KV_TTL_S = 30 * 86400

_lock = threading.RLock()
_alert_lock = threading.Lock()
_alert_handler: Optional[Callable[[str, str], None]] = None   # fn(kind: "dead"|"recovered", text)
_alert_state: Optional[Dict[str, float]] = None
_alert_loaded = False
_cache: Dict[str, Any] = {"token": None, "exp": 0.0}
_mem_rt: Optional[Dict[str, Any]] = None   # {"rt":..., "seed":...} newest token seen by this process
_dead_until = 0.0
_dead_msg = ""
_last_rotation_ts = 0.0
_persist_mode = "unknown"                  # "db" (encrypted, durable) | "memory"
_warned_memory = False


class OneDriveAuthError(RuntimeError):
    """OneDrive authentication failed. `permanent` means re-authorisation is needed (not a blip)."""

    def __init__(self, message: str, permanent: bool = False):
        super().__init__(message)
        self.permanent = permanent


class _InvalidGrant(Exception):
    pass


class _Transient(Exception):
    pass


# ------------------------------------------------------------------------------------------------ helpers
def configured() -> bool:
    return bool(os.getenv("ONEDRIVE_CLIENT_ID") and os.getenv("ONEDRIVE_REFRESH_TOKEN"))


def _h(token: Optional[str]) -> str:
    """Short, non-reversible fingerprint for logs and the seed check (never the token itself)."""
    return hashlib.sha256((token or "").encode("utf-8")).hexdigest()[:12]


def _ub():
    try:
        import usage_budget
        return usage_budget
    except Exception:  # noqa: BLE001
        return None


def _read_legacy(env_rt: str) -> Optional[str]:
    """Plaintext token written by the previous version into kv_cache (seed-checked), or None."""
    kv = _ub()
    if not kv:
        return None
    try:
        raw = kv.kv_get(_LEGACY_KV_KEY)
        if not raw:
            return None
        d = json.loads(raw)
        if d.get("seed") == _h(env_rt) and d.get("rt"):
            return d["rt"]
    except Exception:  # noqa: BLE001
        pass
    return None


def _blank_legacy() -> None:
    kv = _ub()
    if kv:
        try:
            kv.kv_set(_LEGACY_KV_KEY, "", 1)   # no delete API: overwrite with an empty value that expires at once
        except Exception:  # noqa: BLE001
            pass


def _load_stored(env_rt: str) -> Optional[str]:
    """Newest persisted refresh token that was minted from the CURRENT env seed (else None)."""
    rec = _mem_rt
    if rec and rec.get("seed") == _h(env_rt):
        return rec.get("rt")
    d = secret_store.get(_PROVIDER)
    if d and d.get("rt"):
        if d.get("seed") == _h(env_rt):
            return d["rt"]
        log.info("ONEDRIVE_TOKEN stored token ignored (env token changed since it was saved)")
        return None
    legacy = _read_legacy(env_rt)
    if legacy:
        # migrate: encrypt it into the dedicated table, then blank the plaintext copy
        if secret_store.put(_PROVIDER, {"rt": legacy, "seed": _h(env_rt)}) == "db":
            _blank_legacy()
            log.info("ONEDRIVE_TOKEN migrated plaintext kv_cache token into the encrypted store")
        else:
            log.warning("ONEDRIVE_TOKEN a plaintext token is still in kv_cache — set %s (and run "
                        "sql/oauth_tokens.sql) so it can be migrated and removed", secret_store.ENV_KEY)
        return legacy
    return None


def _persist(new_rt: str, env_rt: str) -> str:
    """Save the rotated token encrypted. Returns 'db' (durable) or 'memory' (this process only)."""
    global _mem_rt, _persist_mode, _warned_memory
    _mem_rt = {"rt": new_rt, "seed": _h(env_rt)}
    res = secret_store.put(_PROVIDER, {"rt": new_rt, "seed": _h(env_rt)})
    mode = "db" if res == "db" else "memory"
    if mode == "db":
        _blank_legacy()
    _persist_mode = mode
    if mode == "memory" and not _warned_memory:
        _warned_memory = True
        log.warning("ONEDRIVE_TOKEN rotated token is held in MEMORY only (%s). After a restart the original env "
                    "token is used again; it stops working 90 days after it was created. Set %s and run "
                    "sql/oauth_tokens.sql to make rotation durable.",
                    "no encryption key" if res == "no_key" else "token store unavailable", secret_store.ENV_KEY)
    return mode


def _store_ready() -> bool:
    return secret_store.store_ready()


def _candidates(env_rt: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    stored = _load_stored(env_rt)
    if stored and stored != env_rt:
        out.append({"rt": stored, "src": "stored"})
    out.append({"rt": env_rt, "src": "env"})
    return out


def _redeem(rt: str, client_id: str, client_secret: Optional[str]) -> Dict[str, Any]:
    data = {"client_id": client_id, "grant_type": "refresh_token", "refresh_token": rt, "scope": SCOPE}
    if client_secret:
        data["client_secret"] = client_secret
    try:
        r = requests.post(TOKEN_URL, data=data, timeout=30)
    except (requests.Timeout, requests.ConnectionError) as e:
        raise _Transient(f"{type(e).__name__}: {e}") from e
    if r.status_code == 200:
        try:
            body = r.json()
        except ValueError as e:
            raise _Transient("token endpoint returned non-JSON") from e
        if not body.get("access_token"):
            raise _Transient("token endpoint returned no access_token")
        return body
    try:
        err = r.json()
    except ValueError:
        err = {}
    code = str(err.get("error") or "")
    desc = str(err.get("error_description") or r.text or "")[:240].replace("\n", " ")
    if r.status_code in (429, 500, 502, 503, 504) or (r.status_code == 400 and code in ("temporarily_unavailable",)):
        raise _Transient(f"HTTP {r.status_code} {code} {desc}")
    if code in _PERMANENT_ERRORS or r.status_code in (400, 401):
        raise _InvalidGrant(f"HTTP {r.status_code} {code}: {desc}")
    raise _Transient(f"HTTP {r.status_code} {code} {desc}")


# ------------------------------------------------------------------------------------------ owner alerts
def set_alert_handler(fn: Optional[Callable[[str, str], None]]) -> None:
    """Register fn(kind, text) for owner alerts. kind is 'dead' or 'recovered'. The handler is called on a
    background thread and must do its own delivery; exceptions are swallowed. Keeps this module free of any
    WhatsApp/transport dependency."""
    global _alert_handler
    _alert_handler = fn


def _alert_load() -> None:
    """Read the persisted outage marker once per process (so a restart neither re-alerts inside the repeat
    window nor forgets to announce recovery). Caller holds _alert_lock."""
    global _alert_state, _alert_loaded
    if _alert_loaded:
        return
    _alert_loaded = True
    ub = _ub()
    try:
        raw = ub.kv_get(_ALERT_KV_KEY) if ub else None
        d = json.loads(raw) if raw else None
        if isinstance(d, dict) and d.get("first") and d.get("last"):
            _alert_state = {"first": float(d["first"]), "last": float(d["last"])}
    except Exception:  # noqa: BLE001
        _alert_state = None


def _alert_persist(state: Optional[Dict[str, float]]) -> None:
    ub = _ub()
    if not ub:
        return
    try:
        ub.kv_set(_ALERT_KV_KEY, json.dumps(state) if state else "", _ALERT_KV_TTL_S if state else 1)
    except Exception:  # noqa: BLE001
        pass


def _alert_dispatch(kind: str, text: str, state: Optional[Dict[str, float]]) -> None:
    """Persist the marker and call the handler OFF the caller's thread: the caller may hold the token lock and
    must never wait on a Supabase write or a WhatsApp send."""
    handler = _alert_handler

    def run() -> None:
        _alert_persist(state)
        try:
            handler(kind, text)  # type: ignore[misc]
        except Exception as e:  # noqa: BLE001
            log.warning("ONEDRIVE_ALERT handler failed: %s", e)

    threading.Thread(target=run, daemon=True, name="onedrive-alert").start()


def _fmt_age(seconds: float) -> str:
    m = int(seconds // 60)
    return f"{m // 60}h{m % 60:02d}m" if m >= 60 else f"{m}m"


def _alert_dead(reason: str) -> None:
    """First permanent failure -> tell the owner how to fix it; repeat every ONEDRIVE_ALERT_REPEAT_H while down."""
    global _alert_state
    if not _ALERTS_ON or _alert_handler is None:
        return                      # no handler yet: don't mark as alerted, the next failure will try again
    now = time.time()
    with _alert_lock:
        _alert_load()
        st = _alert_state
        if st and now - st["last"] < _ALERT_REPEAT_S:
            return
        new = {"first": st["first"] if st else now, "last": now}
        _alert_state = new
        repeat = bool(st)
    hours = int(_ALERT_REPEAT_S // 3600)
    text = (
        ("🔑 *OneDrive still disconnected*" if repeat else "🔑 *OneDrive disconnected*") + "\n"
        f"Microsoft rejected the saved login ({reason[:200]}).\n"
        "Until you re-authorise, these stop working: knowledge-base sync (RAG), resume/file downloads, "
        "log uploads and the OneDrive file tools.\n\n"
        "*Fix (about 2 min):*\n"
        "1. On your PC run: python scripts/get_onedrive_token.py and sign in\n"
        "2. Put the printed value into Render → ONEDRIVE_REFRESH_TOKEN. Render redeploys by itself; the "
        "stored encrypted copy is replaced automatically.\n\n"
        f"I'll remind you every {hours}h while it is down"
        + (f" (down for {_fmt_age(now - new['first'])})" if repeat else "")
        + " and tell you when it is back."
    )
    log.error("ONEDRIVE_ALERT dead sent=%s repeat=%s", True, repeat)
    _alert_dispatch("dead", text, new)


def _alert_recovered() -> None:
    """A refresh succeeded: if an outage was announced, announce the recovery once."""
    global _alert_state
    if not _ALERTS_ON or _alert_handler is None:
        return
    with _alert_lock:
        _alert_load()
        st = _alert_state
        if not st:
            return
        _alert_state = None
    text = (f"✅ *OneDrive reconnected* — token refresh works again "
            f"(was down {_fmt_age(time.time() - st['first'])}). Knowledge-base sync, file downloads and log "
            "uploads are back.")
    log.info("ONEDRIVE_ALERT recovered after %s", _fmt_age(time.time() - st["first"]))
    _alert_dispatch("recovered", text, None)


# --------------------------------------------------------------------------------------------- public API
def get_access_token(force: bool = False) -> str:
    """A valid Graph access token (cached; single-flight refresh)."""
    global _dead_until, _dead_msg, _last_rotation_ts
    client_id = os.getenv("ONEDRIVE_CLIENT_ID")
    client_secret = os.getenv("ONEDRIVE_CLIENT_SECRET") or None
    env_rt = (os.getenv("ONEDRIVE_REFRESH_TOKEN") or "").strip()
    if not client_id or not env_rt:
        raise OneDriveAuthError("OneDrive not configured. Set ONEDRIVE_CLIENT_ID and ONEDRIVE_REFRESH_TOKEN.",
                                permanent=True)
    now = time.time()
    if not force and _cache["token"] and _cache["exp"] - _SKEW_S > now:
        return _cache["token"]
    with _lock:
        now = time.time()
        if not force and _cache["token"] and _cache["exp"] - _SKEW_S > now:   # another thread just refreshed
            return _cache["token"]
        if _dead_until > now:
            raise OneDriveAuthError(_dead_msg, permanent=True)

        rejected: List[str] = []
        transient_err: Optional[str] = None
        for cand in _candidates(env_rt):
            body = None
            for attempt in range(_TRANSIENT_RETRIES):
                try:
                    body = _redeem(cand["rt"], client_id, client_secret)
                    break
                except _InvalidGrant as e:
                    rejected.append(f"{cand['src']}[{_h(cand['rt'])}]: {e}")
                    transient_err = None   # a definitive answer supersedes an earlier blip on this token
                    break
                except _Transient as e:
                    transient_err = str(e)
                    log.warning("ONEDRIVE_TOKEN transient failure (attempt %d/%d): %s", attempt + 1,
                                _TRANSIENT_RETRIES, e)
                    if attempt + 1 < _TRANSIENT_RETRIES:
                        time.sleep(min(1.5 * (2 ** attempt), 6.0))
            if body is None:
                if transient_err and not rejected:
                    break          # the service is struggling: don't burn through candidates, just report
                continue
            # ---- success ----
            _cache["token"] = body["access_token"]
            _cache["exp"] = time.time() + int(body.get("expires_in") or 3600)
            _dead_until, _dead_msg = 0.0, ""
            _alert_recovered()
            new_rt = body.get("refresh_token")
            if new_rt and new_rt != cand["rt"]:
                mode = _persist(new_rt, env_rt)
                _last_rotation_ts = time.time()
                log.info("ONEDRIVE_TOKEN rotated source=%s old=%s new=%s persisted=%s expires_in=%ss",
                         cand["src"], _h(cand["rt"]), _h(new_rt), mode, body.get("expires_in"))
            elif cand["src"] == "env" and not new_rt:
                log.info("ONEDRIVE_TOKEN refreshed (no rotation returned) source=env")
            return _cache["token"]

        if rejected and not transient_err and not _store_ready() and not _mem_rt \
                and time.time() - _T0 < _STARTUP_GRACE_S:
            # Right after boot the persisted token store is not attached yet, so a newer stored token may
            # exist that we could not consult. Don't declare the account dead (and don't cool off) on that.
            raise OneDriveAuthError("OneDrive auth is starting up (token store not attached yet) — retry shortly.",
                                    permanent=False)
        if rejected and not transient_err:
            _dead_msg = ("OneDrive refresh token was rejected by Microsoft (" + "; ".join(rejected)[:400] + "). "
                         "Re-authorise: run scripts/get_onedrive_token.py and set ONEDRIVE_REFRESH_TOKEN.")
            _dead_until = time.time() + _DEAD_COOLOFF_S
            log.error("ONEDRIVE_AUTH_DEAD %s", _dead_msg)
            _alert_dead(rejected[0].split("]: ", 1)[-1] if rejected else "token rejected")
            raise OneDriveAuthError(_dead_msg, permanent=True)
        raise OneDriveAuthError(f"OneDrive token service temporarily unavailable: {transient_err or 'unknown'}",
                                permanent=False)


def invalidate() -> None:
    """Drop the cached access token (Graph answered 401 — e.g. revoked early)."""
    with _lock:
        _cache["token"], _cache["exp"] = None, 0.0


def status() -> Dict[str, Any]:
    """Cheap, secret-free snapshot for diagnostics / admin commands."""
    now = time.time()
    return {
        "configured": configured(),
        "access_token_valid_for_s": max(0, int(_cache["exp"] - now)) if _cache["token"] else 0,
        "persist_mode": _persist_mode,
        "last_rotation_age_s": int(now - _last_rotation_ts) if _last_rotation_ts else None,
        "dead_for_s": max(0, int(_dead_until - now)),
        "alert_open": bool(_alert_state),
    }


def reset_state() -> None:  # tests
    global _mem_rt, _dead_until, _dead_msg, _last_rotation_ts, _persist_mode, _warned_memory
    global _alert_handler, _alert_state, _alert_loaded
    _alert_handler, _alert_state, _alert_loaded = None, None, False
    with _lock:
        _cache["token"], _cache["exp"] = None, 0.0
        _mem_rt = None
        _dead_until, _dead_msg, _last_rotation_ts = 0.0, "", 0.0
        _persist_mode, _warned_memory = "unknown", False
