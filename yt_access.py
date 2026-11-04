"""
yt_access.py — everything yt-dlp / YouTube needs to work from a datacenter host, in one place.

Why (prod log 2026-11-03 09:48): the only free route that can work from a cloud IP is yt-dlp with valid
cookies, and it never even reached YouTube — all three attempts died in ~7 s with
`[Errno 30] Read-only file system: '/etc/secrets/cookies.txt'` (yt-dlp re-saves its cookie jar on close and
Render mounts secret files read-only). Supadata (100 credits/month) was then spent on a video yt-dlp may
well have handled for free.

What this module provides (no network, no LLM calls — pure, unit-tested helpers):
  * prepare_cookiefile()   writable per-call copy of the cookie file (+ header/CRLF repair)
  * cookie_report()        is the cookie file actually usable? (auth cookies present / expired)
  * js_runtime_opts()      yt-dlp needs a JS runtime for full YouTube support; only deno is on by default,
                           so a found node/bun/quickjs is enabled explicitly (Python API: {'node': {}})
  * subtitle_attempts()    cookie-correct client plan. Facts (yt-dlp PO-Token guide): android / ios /
                           tv_simply IGNORE account cookies; web needs a PO token for subtitles; tv needs
                           neither a PO token nor is blocked from cookies; android_vr needs no PO token.
  * classify()             stable yt-dlp / transcript-api error strings -> control-flow reasons. This is a
                           handful of exact library messages on the hot path, so a deterministic table is
                           the right tool (an LLM call here would add latency/cost for no gain).
  * Health                 per-source circuit breakers (dead Piped hosts, IP-blocked transcript API) so a
                           broken free source stops costing seconds on every request, and stops adding
                           more traffic from an already-flagged IP.
  * proxy_url()            optional YT_PROXY (residential/ISP proxy) honoured by every YouTube client.
  * log_environment_once() one INFO line with versions / runtime / cookie state, so the next log says
                           exactly why yt-dlp works or not.
"""
from __future__ import annotations

import atexit
import importlib.util
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("mojo.yt")

# ------------------------------------------------------------------------------------------------ cookies
_COOKIE_DIR = os.path.join(tempfile.gettempdir(), "mojo_ytcookies")
_COPY_MAX_AGE_S = 3600.0
_copies: Dict[str, float] = {}
_copies_lock = threading.Lock()
_NETSCAPE_HEADERS = ("# netscape http cookie file", "# http cookie file")
_AUTH_COOKIE_NAMES = frozenset({"SID", "__Secure-1PSID", "__Secure-3PSID", "LOGIN_INFO", "SAPISID"})


def _cleanup_copies(max_age: float = _COPY_MAX_AGE_S) -> None:
    now = time.time()
    with _copies_lock:
        for path, born in list(_copies.items()):
            if max_age <= 0 or now - born > max_age:
                try:
                    os.remove(path)
                except OSError:
                    pass
                _copies.pop(path, None)


atexit.register(lambda: _cleanup_copies(0))


def prepare_cookiefile(src: str, context: str = "ytdlp") -> Optional[str]:
    """Return a WRITABLE per-call copy of the cookie file at `src` (or None if it cannot be read).

    * yt-dlp rewrites the jar when it closes -> the original /etc/secrets mount must never be handed over.
    * A fresh copy per call: two concurrent yt-dlp runs never write the same file.
    * Self-heals the two classic paste mistakes: missing '# Netscape HTTP Cookie File' header (yt-dlp then
      refuses the file) and Windows CRLF line endings.
    * Copies are swept after an hour and at exit."""
    try:
        with open(src, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as e:
        log.warning("[cookies] %s: cannot read %s (%s) — running without cookies", context, src, e)
        return None
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    first = next((ln.strip().lower() for ln in text.split("\n") if ln.strip()), "")
    if not first.startswith(_NETSCAPE_HEADERS):
        text = "# Netscape HTTP Cookie File\n" + text
        log.info("[cookies] %s: header added to cookie copy (file had none)", context)
    if not text.endswith("\n"):
        text += "\n"
    try:
        os.makedirs(_COOKIE_DIR, mode=0o700, exist_ok=True)
        fd, dst = tempfile.mkstemp(prefix="c_", suffix=".txt", dir=_COOKIE_DIR)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as out:
            out.write(text)
        os.chmod(dst, 0o600)
    except OSError as e:
        log.warning("[cookies] %s: cannot create writable copy (%s) — running without cookies", context, e)
        return None
    with _copies_lock:
        _copies[dst] = time.time()
    _cleanup_copies()
    return dst


def cookie_report(path: str, now: Optional[float] = None) -> Dict[str, Any]:
    """Inspect a Netscape cookie file: are there YouTube/Google cookies, are the login ones still valid?"""
    now = time.time() if now is None else now
    rep: Dict[str, Any] = {"ok": False, "total": 0, "google": 0, "expired": 0, "auth": False,
                           "auth_expired": False, "earliest_auth_expiry": None}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return rep
    for ln in lines:
        s = ln.strip()
        if s.startswith("#HttpOnly_"):
            s = s[len("#HttpOnly_"):]
        elif not s or s.startswith("#"):
            continue
        parts = s.split("\t")
        if len(parts) < 7:
            continue
        domain, _flag, _path, _secure, expiry, name = parts[:6]
        rep["total"] += 1
        if "youtube.com" in domain or "google.com" in domain:
            rep["google"] += 1
        try:
            exp = int(float(expiry))
        except ValueError:
            exp = 0
        expired = 0 < exp < now
        if expired:
            rep["expired"] += 1
        if name in _AUTH_COOKIE_NAMES and ("google.com" in domain or "youtube.com" in domain):
            if expired:
                rep["auth_expired"] = True
            else:
                rep["auth"] = True
                if exp > 0 and (rep["earliest_auth_expiry"] is None or exp < rep["earliest_auth_expiry"]):
                    rep["earliest_auth_expiry"] = exp
    rep["ok"] = rep["google"] > 0 and rep["auth"]
    return rep


# -------------------------------------------------------------------------------------------- JS runtime
_RUNTIME_BINARIES = (("deno", "deno"), ("node", "node"), ("bun", "bun"), ("quickjs", "qjs"))


def detect_js_runtime() -> Tuple[Optional[str], Optional[str]]:
    """(yt-dlp runtime name, binary path) of the best available runtime, or (None, None)."""
    forced = (os.getenv("YTDLP_JS_RUNTIME") or "").strip().lower()
    for name, binary in _RUNTIME_BINARIES:
        if forced and forced != name:
            continue
        path = shutil.which(binary)
        if path:
            return name, path
    return None, None


def js_runtime_opts() -> Dict[str, Any]:
    """YoutubeDL params enabling a JS runtime. yt-dlp enables only deno by default, so node/bun/quickjs must
    be switched on explicitly. Python-API format is a dict with an (empty) config: {'node': {}}."""
    name, _path = detect_js_runtime()
    if name and name != "deno":
        return {"js_runtimes": {name: {}}}
    return {}


# --------------------------------------------------------------------------------------------------- proxy
def proxy_url() -> Optional[str]:
    """Optional YT_PROXY (e.g. a residential/ISP proxy). Off by default."""
    v = (os.getenv("YT_PROXY") or "").strip()
    return v or None


def requests_proxies() -> Optional[Dict[str, str]]:
    p = proxy_url()
    return {"http": p, "https": p} if p else None


def apply_common(opts: Dict[str, Any]) -> Dict[str, Any]:
    """Options every YouTube yt-dlp call shares (proxy + JS runtime). Mutates and returns `opts`."""
    p = proxy_url()
    if p:
        opts["proxy"] = p
    for k, v in js_runtime_opts().items():
        opts.setdefault(k, v)
    return opts


# ---------------------------------------------------------------------------------------------- client plan
@dataclass(frozen=True)
class Attempt:
    name: str
    clients: Tuple[str, ...]
    use_cookies: bool


def has_pot_provider() -> bool:
    """True if a yt-dlp PO-token provider plugin (e.g. bgutil) is installed."""
    try:
        return importlib.util.find_spec("yt_dlp_plugins") is not None
    except Exception:  # noqa: BLE001
        return False


def subtitle_attempts(has_cookies: bool, pot: Optional[bool] = None) -> List[Attempt]:
    """Ordered yt-dlp attempts for SUBTITLES. Cookie-free first: it needs no PO token and never exposes the
    account to a datacenter IP; then tv (supports cookies, no PO token); then yt-dlp's own defaults."""
    plan = [Attempt("android_vr", ("android_vr",), False),
            Attempt("tv+cookies" if has_cookies else "tv", ("tv",), has_cookies)]
    if has_cookies or (has_pot_provider() if pot is None else pot):
        plan.append(Attempt("default+cookies" if has_cookies else "default", ("default",), has_cookies))
    return plan


def audio_attempts(has_cookies: bool) -> List[Attempt]:
    """Ordered yt-dlp attempts for AUDIO (ASR). Clients are passed as separate list items — the Python API
    does not split 'a,b,c' the way the CLI does."""
    plan = [Attempt("android_vr", ("android_vr",), False),
            Attempt("tv+cookies" if has_cookies else "tv", ("tv",), has_cookies),
            Attempt("web_embedded", ("web_embedded",), False)]
    if has_cookies:
        plan.append(Attempt("default+cookies", ("default",), True))
    else:
        plan.append(Attempt("default", ("default",), False))
    return plan


# ----------------------------------------------------------------------------------- error classification
BLOCKED = frozenset({"signin", "blocked"})          # YouTube is refusing this IP/account: more requests won't help
TERMINAL = frozenset({"private", "unavailable", "members", "age"})  # the video itself can't be fetched


def classify(err: Any) -> str:
    """Map a yt-dlp / youtube-transcript-api failure to a reason code."""
    s = str(err).lower()
    cls = type(err).__name__.lower() if isinstance(err, BaseException) else ""
    if "read-only file system" in s or "errno 30" in s:
        return "cookies_fs"
    if "netscape" in s and "cookie" in s:
        return "cookies_invalid"
    if ("sign in to confirm" in s or "not a bot" in s or "login_required" in s
            or "confirm you’re not a bot" in s or "use --cookies" in s):
        return "signin"
    if "ipblocked" in cls or "requestblocked" in cls or "toomanyrequests" in cls or "blocking requests from your ip" in s or \
            "http error 429" in s or "too many requests" in s or "http error 403" in s or "forbidden" in s:
        return "blocked"
    if "private video" in s or "this video is private" in s:
        return "private"
    if "members-only" in s or "join this channel" in s:
        return "members"
    if "confirm your age" in s or "age-restricted" in s or "inappropriate for some users" in s:
        return "age"
    if ("video unavailable" in s or "this video is not available" in s or "has been removed" in s
            or "account associated with this video has been terminated" in s or "no video with id" in s):
        return "unavailable"
    if "transcriptsdisabled" in cls or "subtitles are disabled" in s:
        return "no_captions"            # authoritative: the uploader switched captions off
    if "notranscriptfound" in cls:
        return "subs_other_lang"        # NOT proof of "no captions": none in the requested languages
    if "timed out" in s or "timeout" in s or "timeouterror" in cls:
        return "timeout"
    return "unknown"


# --------------------------------------------------------------------------------------- per-video outcome
_OUTCOME_TTL_S = 600.0
_outcomes: Dict[str, Tuple[str, float]] = {}
_out_lock = threading.Lock()


_AUTHORITATIVE = frozenset({"no_captions"})   # direct evidence; weaker later outcomes must not overwrite it


def set_outcome(video_id: str, reason: str) -> None:
    with _out_lock:
        ent = _outcomes.get(video_id)
        if ent and ent[0] in _AUTHORITATIVE and ent[1] >= time.time() and reason not in _AUTHORITATIVE:
            return
        _outcomes[video_id] = (reason, time.time() + _OUTCOME_TTL_S)


def get_outcome(video_id: str) -> Optional[str]:
    with _out_lock:
        ent = _outcomes.get(video_id)
        if not ent:
            return None
        if ent[1] < time.time():
            _outcomes.pop(video_id, None)
            return None
        return ent[0]


# --------------------------------------------------------------------------------------- circuit breakers
class Health:
    """Per-source circuit breaker. `failure(hard=True)` trips immediately; otherwise after `threshold`
    consecutive failures. `success()` closes it. Cooldowns are per source."""

    POLICY: Dict[str, Tuple[int, float]] = {      # name -> (threshold, cooldown seconds)
        "piped": (3, 6 * 3600.0),                 # public Piped hosts are mostly dead/blocked
        "transcript_api": (3, 30 * 60.0),         # IpBlocked is tripped hard (see agent_tools)
        "ytdlp_subs": (2, 10 * 60.0),             # two consecutive IP/bot blocks
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._st: Dict[str, Dict[str, float]] = {}

    def _get(self, name: str) -> Dict[str, float]:
        return self._st.setdefault(name, {"fails": 0, "until": 0.0})

    def allow(self, name: str) -> bool:
        with self._lock:
            return self._get(name)["until"] <= time.time()

    def remaining(self, name: str) -> float:
        with self._lock:
            return max(0.0, self._get(name)["until"] - time.time())

    def success(self, name: str) -> None:
        with self._lock:
            st = self._get(name)
            st["fails"], st["until"] = 0, 0.0

    def failure(self, name: str, *, hard: bool = False, cooldown: Optional[float] = None) -> bool:
        """Record a failure; returns True if the breaker is now open."""
        thr, cd = self.POLICY.get(name, (3, 600.0))
        with self._lock:
            st = self._get(name)
            st["fails"] += 1
            if hard or st["fails"] >= thr:
                st["until"] = time.time() + (cooldown if cooldown is not None else cd)
                st["fails"] = 0
                log.warning("YT_HEALTH source=%s OPEN for %.0fs (%s)", name,
                            cooldown if cooldown is not None else cd, "hard" if hard else "threshold")
                return True
            return False

    def reset(self) -> None:
        with self._lock:
            self._st.clear()

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            now = time.time()
            return {n: max(0.0, s["until"] - now) for n, s in self._st.items()}


health = Health()


# ----------------------------------------------------------------------------------------- diagnostics
_env_logged = False


def environment_report(cookie_path: Optional[str] = None) -> str:
    def has(mod: str) -> bool:
        try:
            return importlib.util.find_spec(mod) is not None
        except Exception:  # noqa: BLE001
            return False

    try:
        import yt_dlp  # type: ignore
        ver = yt_dlp.version.__version__
    except Exception:  # noqa: BLE001
        ver = "missing"
    name, path = detect_js_runtime()
    parts = [f"yt_dlp={ver}", f"ejs={'yes' if has('yt_dlp_ejs') else 'NO'}",
             f"js_runtime={name or 'NONE'}", f"curl_cffi={'yes' if has('curl_cffi') else 'no'}",
             f"pot_provider={'yes' if has_pot_provider() else 'no'}", f"proxy={'on' if proxy_url() else 'off'}"]
    if cookie_path:
        c = cookie_report(cookie_path)
        exp = c["earliest_auth_expiry"]
        parts.append(
            "cookies=%s(total=%d google=%d expired=%d auth=%s%s%s)" % (
                "ok" if c["ok"] else "UNUSABLE", c["total"], c["google"], c["expired"], c["auth"],
                " AUTH_EXPIRED" if c["auth_expired"] else "",
                f" auth_expires_in={int((exp - time.time()) / 86400)}d" if exp else ""))
    else:
        parts.append("cookies=none")
    return " ".join(parts)


def log_environment_once(cookie_src: Optional[str] = None) -> None:
    """One INFO line per process. `cookie_src` is the ORIGINAL cookie path (inspected read-only)."""
    global _env_logged
    if _env_logged:
        return
    _env_logged = True
    try:
        log.info("YTDLP_ENV %s", environment_report(cookie_src))
        name, _ = detect_js_runtime()
        if name is None:
            log.warning("YTDLP_ENV no JS runtime found (deno/node/bun/qjs): yt-dlp may miss formats. "
                        "Install node/deno on the host or `pip install deno`.")
    except Exception as e:  # noqa: BLE001
        log.warning("YTDLP_ENV report failed: %s", e)


def reset_state() -> None:  # tests
    global _env_logged
    _env_logged = False
    health.reset()
    with _out_lock:
        _outcomes.clear()
    _cleanup_copies(0)
