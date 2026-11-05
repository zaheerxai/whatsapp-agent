"""
secret_store.py — encrypted-at-rest storage for long-lived credentials (OAuth refresh tokens, API keys).

Design
  * Dedicated Supabase table `public.oauth_tokens` (sql/oauth_tokens.sql): RLS on, no policies, anon/
    authenticated privileges revoked — only the service-role key can touch it. It is NOT the shared
    `kv_cache` cache table.
  * Values are encrypted with Fernet (AES-128-CBC + HMAC-SHA256, authenticated: tampering is detected)
    using the key(s) in env TOKEN_ENCRYPTION_KEY. The key lives in the host's environment, the ciphertext in
    Supabase, so a database leak alone reveals nothing.
  * Fail closed: no valid key -> nothing is written to the database (callers keep the value in memory).
    A secret is never stored in plaintext.
  * Key rotation: TOKEN_ENCRYPTION_KEY may hold several comma-separated keys. The FIRST encrypts, all
    decrypt (MultiFernet). A record that decrypts only with an older key is re-encrypted with the first key
    on read, so rotating a key is: put the new key first, keep the old one until a record has been read.
  * Each ciphertext binds its provider name inside the authenticated payload, so a row copied under another
    provider name is rejected.
  * Backend failures (missing table, Supabase outage) never raise into callers: a short breaker avoids
    hammering, and one warning per condition is logged.

Generate a key:  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
Keep it ONLY in the environment (Render dashboard). If it is lost the stored tokens are unreadable and the
bot falls back to the env seed (re-authorise if that is expired too).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("mojo.secrets")

ENV_KEY = "TOKEN_ENCRYPTION_KEY"
TABLE = "oauth_tokens"
_BACKEND_DOWN_S = 300.0

_warned: set = set()
_wlock = threading.Lock()


def _warn_once(tag: str, msg: str, *args: Any) -> None:
    with _wlock:
        if tag in _warned:
            return
        _warned.add(tag)
    log.warning(msg, *args)


# ------------------------------------------------------------------------------------------------- keys
_fernet_cache: Tuple[str, Optional[List[Any]]] = ("", None)


def _fernets() -> List[Any]:
    """Valid Fernet instances from TOKEN_ENCRYPTION_KEY, first = primary. [] when unusable."""
    global _fernet_cache
    raw = (os.getenv(ENV_KEY) or "").strip()
    if _fernet_cache[0] == raw and _fernet_cache[1] is not None:
        return _fernet_cache[1]
    out: List[Any] = []
    if raw:
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            _warn_once("nolib", "secret_store: the 'cryptography' package is not installed — secrets stay in memory only")
            _fernet_cache = (raw, [])
            return []
        for i, part in enumerate(p.strip() for p in raw.split(",") if p.strip()):
            try:
                out.append(Fernet(part.encode("utf-8")))
            except Exception:  # noqa: BLE001 - ValueError for malformed keys
                _warn_once(f"badkey{i}", "secret_store: key #%d in %s is not a valid Fernet key (ignored). "
                           "Generate one with Fernet.generate_key().", i + 1, ENV_KEY)
    _fernet_cache = (raw, out)
    return out


def key_configured() -> bool:
    return bool(_fernets())


def encrypt(provider: str, payload: Dict[str, Any]) -> Optional[str]:
    fs = _fernets()
    if not fs:
        return None
    body = json.dumps({"p": provider, "v": payload, "t": int(time.time())}, separators=(",", ":"))
    return fs[0].encrypt(body.encode("utf-8")).decode("ascii")


def decrypt(provider: str, token: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Returns (payload or None, decrypted_with_non_primary_key). Wrong key / tampering / wrong provider -> None."""
    fs = _fernets()
    if not fs or not token:
        return None, False
    try:
        from cryptography.fernet import InvalidToken
    except ImportError:
        return None, False
    for i, f in enumerate(fs):
        try:
            data = json.loads(f.decrypt(token.encode("ascii")).decode("utf-8"))
        except InvalidToken:
            continue
        except Exception:  # noqa: BLE001 - corrupt JSON etc.
            return None, False
        if data.get("p") != provider:
            _warn_once(f"prov:{provider}", "secret_store: stored record for %r is bound to another provider — ignored", provider)
            return None, False
        v = data.get("v")
        return (v if isinstance(v, dict) else None), i > 0
    _warn_once(f"undecryptable:{provider}", "secret_store: stored %r record cannot be decrypted with the configured "
               "%s (wrong/missing key, or tampered). Ignoring it; the env seed will be used.", provider, ENV_KEY)
    return None, False


# ----------------------------------------------------------------------------------------------- backend
class BackendUnavailable(Exception):
    pass


class SupabaseBackend:
    """oauth_tokens(provider pk, ciphertext, updated_at) through the shared Supabase client."""

    def __init__(self) -> None:
        self._down_until = 0.0

    @staticmethod
    def _client() -> Any:
        try:
            import usage_budget
            return usage_budget.db_client()
        except Exception:  # noqa: BLE001
            return None

    def available(self) -> bool:
        return self._client() is not None and time.time() >= self._down_until

    def _fail(self, e: Exception) -> None:
        self._down_until = time.time() + _BACKEND_DOWN_S
        msg = str(e)
        if "PGRST205" in msg or "does not exist" in msg or "schema cache" in msg:
            _warn_once("notable", "secret_store: table public.%s is missing — run sql/oauth_tokens.sql in the Supabase "
                       "SQL editor (secrets stay in memory until then)", TABLE)
        else:
            _warn_once("backend:" + msg[:40], "secret_store: Supabase unavailable (%s) — retrying in %.0fs", msg[:160],
                       _BACKEND_DOWN_S)
        raise BackendUnavailable(msg) from e

    def get(self, provider: str) -> Optional[str]:
        c = self._client()
        if c is None or time.time() < self._down_until:
            raise BackendUnavailable("no client")
        try:
            rows = c.table(TABLE).select("ciphertext").eq("provider", provider).limit(1).execute().data or []
        except Exception as e:  # noqa: BLE001
            self._fail(e)
        return rows[0]["ciphertext"] if rows else None

    def put(self, provider: str, ciphertext: str) -> None:
        c = self._client()
        if c is None or time.time() < self._down_until:
            raise BackendUnavailable("no client")
        try:
            c.table(TABLE).upsert({"provider": provider, "ciphertext": ciphertext,
                                   "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                                  on_conflict="provider").execute()
        except Exception as e:  # noqa: BLE001
            self._fail(e)

    def delete(self, provider: str) -> None:
        c = self._client()
        if c is None or time.time() < self._down_until:
            raise BackendUnavailable("no client")
        try:
            c.table(TABLE).delete().eq("provider", provider).execute()
        except Exception as e:  # noqa: BLE001
            self._fail(e)


_backend: Any = SupabaseBackend()


def set_backend(b: Any) -> None:  # tests
    global _backend
    _backend = b


# ------------------------------------------------------------------------------------------------ API
def store_ready() -> bool:
    """True when a database is attached (used by callers for their startup-grace logic)."""
    try:
        return bool(_backend.available())
    except Exception:  # noqa: BLE001
        return False


def put(provider: str, payload: Dict[str, Any]) -> str:
    """Persist. Returns 'db' (stored encrypted), 'no_key' (fail-closed: nothing written) or 'unavailable'."""
    token = encrypt(provider, payload)
    if token is None:
        _warn_once("nokey", "secret_store: %s is not set (or invalid) — %r is NOT persisted (never stored in plaintext). "
                   "Generate a key: python -c \"from cryptography.fernet import Fernet; "
                   "print(Fernet.generate_key().decode())\" and set %s.", ENV_KEY, provider, ENV_KEY)
        return "no_key"
    try:
        _backend.put(provider, token)
        return "db"
    except BackendUnavailable:
        return "unavailable"
    except Exception as e:  # noqa: BLE001
        _warn_once("putfail", "secret_store: put failed: %s", str(e)[:160])
        return "unavailable"


def get(provider: str) -> Optional[Dict[str, Any]]:
    """Decrypted payload or None. Transparently re-encrypts under the primary key after a key rotation."""
    try:
        token = _backend.get(provider)
    except BackendUnavailable:
        return None
    except Exception as e:  # noqa: BLE001
        _warn_once("getfail", "secret_store: get failed: %s", str(e)[:160])
        return None
    if not token:
        return None
    payload, via_old_key = decrypt(provider, token)
    if payload is not None and via_old_key:
        if put(provider, payload) == "db":
            log.info("secret_store: %r re-encrypted with the primary key", provider)
    return payload


def delete(provider: str) -> bool:
    try:
        _backend.delete(provider)
        return True
    except Exception:  # noqa: BLE001
        return False


def reset_state() -> None:  # tests
    global _fernet_cache
    _fernet_cache = ("", None)
    with _wlock:
        _warned.clear()
