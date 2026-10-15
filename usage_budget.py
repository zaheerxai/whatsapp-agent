"""
usage_budget.py — (1) a persistent key/value cache, (2) a credit budget for Supadata.

Why this exists (Supadata FREE plan = 100 credits/month, no rollover, 1 req/s):
  * native transcript          = 1 credit
  * AI-generated transcript    = 2 credits PER VIDEO MINUTE   <- the budget killer
    (most Instagram/TikTok reels have no native captions, so a 90 s reel = 4 credits)
  * metadata                   = assumed 1 credit (verified against /me deltas in logs)
  * failed / rate-limited      = 0 credits (per Supadata docs)
  * GET /v1/me                 = free; returns maxCredits / usedCredits for the period

Policy implemented here:
  * The REAL balance comes from GET /v1/me (cached 2 min). If /me is unreachable we fall
    back to a persistent local counter (Supabase kv_cache), then to an in-memory one.
  * A reserve (SUPADATA_OWNER_RESERVE, default 20) is kept for the owner: normal chats are
    refused once remaining - cost < reserve; the owner may spend down to 0. (Owner/admin
    features stay ON by default.)
  * Nothing paid is ever called twice for the same video: callers read kv_get() first
    (transcripts are cached 30 days, metadata 7 days, across restarts — Render Free wipes
    memory on every spin-down, which used to re-bill the same reel).

KV storage needs the table in schema_kv_cache.sql; without it we silently stay in-memory
(logged once) so nothing breaks.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import requests

log = logging.getLogger("mojo.budget")

_supabase = None
_lock = threading.Lock()
_mem: Dict[str, Tuple[str, float]] = {}   # key -> (value, expires_epoch)
_kv_down_until = 0.0
_kv_warned = False


def init(supabase_client) -> None:
    global _supabase
    _supabase = supabase_client


# ----------------------------------------------------------------------------- KV cache
def _kv_ok() -> bool:
    return _supabase is not None and time.time() >= _kv_down_until


def _kv_fail(e: Exception) -> None:
    global _kv_down_until, _kv_warned
    _kv_down_until = time.time() + 600
    if not _kv_warned:
        _kv_warned = True
        log.warning(
            "kv_cache table unavailable (%s) — running memory-only. Create it with "
            "schema_kv_cache.sql to keep paid results across restarts.", str(e)[:140],
        )


def kv_get(key: str) -> Optional[str]:
    now = time.time()
    with _lock:
        ent = _mem.get(key)
        if ent and ent[1] > now:
            return ent[0]
    if not _kv_ok():
        return None
    try:
        rows = (
            _supabase.table("kv_cache").select("value,expires_at").eq("key", key).limit(1).execute().data
        ) or []
        if not rows:
            return None
        exp = datetime.fromisoformat(str(rows[0]["expires_at"]).replace("Z", "+00:00"))
        if exp.timestamp() <= now:
            return None
        val = rows[0]["value"]
        with _lock:
            _mem[key] = (val, exp.timestamp())
        return val
    except Exception as e:
        _kv_fail(e)
        return None


def kv_set(key: str, value: str, ttl_sec: int) -> None:
    exp = time.time() + ttl_sec
    with _lock:
        _mem[key] = (value, exp)
    if not _kv_ok():
        return
    try:
        _supabase.table("kv_cache").upsert(
            {
                "key": key,
                "value": value,
                "expires_at": datetime.fromtimestamp(exp, tz=timezone.utc).isoformat(),
            },
            on_conflict="key",
        ).execute()
    except Exception as e:
        _kv_fail(e)


# ----------------------------------------------------------------------------- budget
def _month_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _next_month_str() -> str:
    d = datetime.now(timezone.utc).replace(day=1) + timedelta(days=32)
    return d.replace(day=1).strftime("%d %b")


def _flatten(d: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{str(k).lower().replace('_', '')}."))
    else:
        out[prefix.rstrip(".")] = d
    return out


def parse_me(data: dict) -> Optional[Dict[str, Any]]:
    """Tolerant parse of GET /v1/me -> {max, used, plan}. Field names vary by version."""
    flat = _flatten(data or {})
    mx = used = None
    plan = ""
    for k, v in flat.items():
        kk = k.replace(".", "")  # nested paths ("credits.max") must match like flat camelCase keys
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            if k.endswith("plan") and isinstance(v, str):
                plan = v
            continue
        if mx is None and ("maxcredit" in kk or "creditlimit" in kk or "creditsmax" in kk):
            mx = int(v)
        elif used is None and ("usedcredit" in kk or "creditsused" in kk or "creditused" in kk):
            used = int(v)
    if mx is None or used is None:
        return None
    return {"max": mx, "used": used, "plan": plan}


class SupadataBudget:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._me: Optional[dict] = None
        self._me_exp = 0.0
        self._warned: set = set()

    # -- config (read live so env changes need no code change)
    @staticmethod
    def monthly_limit() -> int:
        return int(os.getenv("SUPADATA_MONTHLY_CREDITS") or "100")

    @staticmethod
    def reserve() -> int:
        return int(os.getenv("SUPADATA_OWNER_RESERVE") or "20")

    @staticmethod
    def max_gen_minutes() -> int:
        return int(os.getenv("SUPADATA_MAX_GEN_MINUTES") or "4")

    def _fetch_me(self) -> Optional[dict]:
        key = (os.getenv("SUPADATA_API_KEY") or "").strip()
        if not key:
            return None
        try:
            r = requests.get(
                "https://api.supadata.ai/v1/me",
                headers={"x-api-key": key, "Accept": "application/json"},
                timeout=10,
            )
            if r.status_code != 200:
                log.warning("supadata /me HTTP %s", r.status_code)
                return None
            return parse_me(r.json() or {})
        except Exception as e:
            log.warning("supadata /me failed: %s", str(e)[:120])
            return None

    def snapshot(self, force: bool = False) -> Dict[str, Any]:
        """{max, used, remaining, source: 'me'|'local', plan}. Never raises."""
        now = time.time()
        with self._lock:
            if not force and self._me and self._me_exp > now:
                return dict(self._me)
        me = self._fetch_me()
        if me:
            snap = {**me, "remaining": max(0, me["max"] - me["used"]), "source": "me"}
        else:
            used = 0
            try:
                used = int(kv_get(f"supadata_used:{_month_key()}") or 0)
            except ValueError:
                pass
            mx = self.monthly_limit()
            snap = {"max": mx, "used": used, "remaining": max(0, mx - used), "source": "local", "plan": ""}
        with self._lock:
            self._me, self._me_exp = snap, now + (120 if snap["source"] == "me" else 30)
        return dict(snap)

    def can_spend(self, cost: int, privileged: bool = False) -> Tuple[bool, str]:
        snap = self.snapshot()
        rem = snap["remaining"]
        if cost > rem:
            return False, (
                f"Supadata monthly credits khatam ({rem}/{snap['max']} left, need {cost}); "
                f"resets ~{_next_month_str()}"
            )
        if not privileged and rem - cost < self.reserve():
            return False, (
                f"Supadata credits low ({rem}/{snap['max']} left) — kept for the owner; "
                f"resets ~{_next_month_str()}"
            )
        return True, ""

    def record(self, cost: int, what: str) -> None:
        """Log the real /me delta; keep the local ledger (the fallback when /me is down) in sync
        with the TRUE balance instead of piling up pessimistic estimates (prod: est=8, real=3)."""
        with self._lock:
            before = (self._me or {}).get("used")
        k = f"supadata_used:{_month_key()}"
        after = self.snapshot(force=True)
        if after.get("source") == "me":
            kv_set(k, str(after["used"]), 40 * 86400)
            real = (after["used"] - before) if isinstance(before, int) else None
            log.info("SUPADATA_CHARGE what=%s estimated=%s actual=%s used=%s/%s",
                     what, cost, real if real is not None else "?", after["used"], after["max"])
        else:
            try:
                kv_set(k, str(int(kv_get(k) or 0) + int(cost)), 40 * 86400)
            except ValueError:
                kv_set(k, str(int(cost)), 40 * 86400)
            after = self.snapshot(force=True)
            log.info("SUPADATA_CHARGE what=%s est=%s used~%s/%s (local ledger)", what, cost, after.get("used"), after.get("max"))
        pct = after["used"] / after["max"] if after.get("max") else 0
        for th in (0.8, 0.95):
            tag = (_month_key(), th)
            if pct >= th and tag not in self._warned:
                self._warned.add(tag)
                log.warning("SUPADATA_QUOTA %.0f%% used (%s/%s)", pct * 100, after["used"], after["max"])

    # -- cost model
    @staticmethod
    def native_cost() -> int:
        return 1

    @staticmethod
    def generate_cost(minutes: float) -> int:
        return 2 * max(1, math.ceil(minutes))

    @staticmethod
    def metadata_cost() -> int:
        return 1

    def status_text(self) -> str:
        s = self.snapshot(force=True)
        bar_n = int(10 * s["used"] / s["max"]) if s["max"] else 0
        return (
            f"Supadata credits: {s['used']}/{s['max']} used — {s['remaining']} left "
            f"[{'█' * bar_n}{'░' * (10 - bar_n)}]\n"
            f"Source: {'live /v1/me' if s['source'] == 'me' else 'local ledger (/me unreachable)'}"
            f"{' · plan ' + s['plan'] if s.get('plan') else ''}\n"
            f"Owner reserve: {self.reserve()} (normal chats stop when ≤ {self.reserve()} left)\n"
            f"AI-transcribe cap: {self.max_gen_minutes()} min/video · resets ~{_next_month_str()}\n"
            "Costs: native transcript 1 · AI transcript 2/min · metadata 1."
        )


supadata = SupadataBudget()
