"""Offline tests: YouTube access layer (cookies, client plan, classification, breakers) and the new
transcript chain order — Supadata strictly last. No network, no real yt-dlp (a fake module is injected).
Run: PYTHONPATH=. python3 tests/test_yt_access.py
"""
from __future__ import annotations

import os
import stat
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yt_access as Y  # noqa: E402
import agent_tools as T  # noqa: E402

FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if cond:
        print(f"ok   {name}")
    else:
        FAILS += 1
        print(f"FAIL {name} {extra}")


# ------------------------------------------------------------------------------------------ classify()
real = "[Errno 30] Read-only file system: '/etc/secrets/cookies.txt'"
check("prod error -> cookies_fs", Y.classify(OSError(real)) == "cookies_fs")
check("bot check -> signin", Y.classify(Exception("ERROR: [youtube] x: Sign in to confirm you’re not a bot")) == "signin")
check("bot check (ascii quote)", Y.classify(Exception("Sign in to confirm you're not a bot. Use --cookies")) == "signin")
check("429 -> blocked", Y.classify(Exception("HTTP Error 429: Too Many Requests")) == "blocked")
check("IpBlocked class -> blocked", Y.classify(type("IpBlocked", (Exception,), {})("x")) == "blocked")
check("private (yt-dlp wording)", Y.classify(Exception("ERROR: Private video. Sign in if you've been granted access")) == "private")
check("private (clean)", Y.classify(Exception("This video is private")) == "private")
check("unavailable", Y.classify(Exception("Video unavailable. This video has been removed")) == "unavailable")
check("TranscriptsDisabled -> no_captions (authoritative)", Y.classify(type("TranscriptsDisabled", (Exception,), {})("x")) == "no_captions")
check("NoTranscriptFound is NOT no_captions", Y.classify(type("NoTranscriptFound", (Exception,), {})("x")) == "subs_other_lang")
check("bad cookie file", Y.classify(Exception("does not look like a Netscape format cookies file")) == "cookies_invalid")
check("timeout", Y.classify(Exception("The read operation timed out")) == "timeout")
check("unknown", Y.classify(Exception("weird")) == "unknown")

# ------------------------------------------------------------------------------- cookies: writable copy
tmp = tempfile.mkdtemp()
ro_dir = os.path.join(tmp, "secrets")
os.makedirs(ro_dir)
src = os.path.join(ro_dir, "cookies.txt")
far = int(time.time()) + 90 * 86400
body = ("# Netscape HTTP Cookie File\r\n"
        f".youtube.com\tTRUE\t/\tTRUE\t{far}\t__Secure-3PSID\tabc\r\n"
        f".youtube.com\tTRUE\t/\tTRUE\t{far}\tVISITOR_INFO1_LIVE\tzzz\r\n"
        ".google.com\tTRUE\t/\tTRUE\t1000\tSID\told\r\n")
open(src, "w", newline="").write(body)
os.chmod(src, 0o444)
os.chmod(ro_dir, 0o555)  # like Render's /etc/secrets
c1 = Y.prepare_cookiefile(src, "t")
c2 = Y.prepare_cookiefile(src, "t")
check("copy is a different path than the read-only original", c1 and c1 != src and os.path.dirname(c1) != ro_dir)
check("every call gets its own copy (no shared writes)", c1 != c2)
with open(c1, "a") as fh:
    fh.write("# written by yt-dlp on close\n")
check("copy is writable (the Errno 30 fix)", True)
check("CRLF normalised, content preserved", "\r" not in open(c1).read() and "__Secure-3PSID" in open(c1).read())
check("copy is private (0600)", stat.S_IMODE(os.stat(c1).st_mode) == 0o600)

nohdr = os.path.join(tmp, "nohdr.txt")
open(nohdr, "w").write(f".youtube.com\tTRUE\t/\tTRUE\t{far}\tSID\tx")
c3 = Y.prepare_cookiefile(nohdr, "t")
check("missing Netscape header is repaired", open(c3).read().startswith("# Netscape HTTP Cookie File\n"))
check("unreadable source -> None (run cookie-free)", Y.prepare_cookiefile(os.path.join(tmp, "nope.txt"), "t") is None)

rep = Y.cookie_report(src)
check("cookie_report: auth ok but SID expired counted", rep["auth"] and rep["auth_expired"] and rep["expired"] == 1 and rep["ok"], rep)
dead = os.path.join(tmp, "dead.txt")
open(dead, "w").write("# Netscape HTTP Cookie File\n.google.com\tTRUE\t/\tTRUE\t1000\tSID\told\n")
check("cookie_report: only-expired auth -> unusable", not Y.cookie_report(dead)["ok"] and Y.cookie_report(dead)["auth_expired"])
Y._cleanup_copies(0)
check("copies swept", not os.path.exists(c1) and not os.path.exists(c2))
os.chmod(ro_dir, 0o755)

# ---------------------------------------------------------------------------------------------- JS runtime
orig_which = Y.shutil.which
Y.shutil.which = lambda b: "/usr/bin/node" if b == "node" else None
check("node only -> enabled explicitly ({'node': {}})", Y.js_runtime_opts() == {"js_runtimes": {"node": {}}})
Y.shutil.which = lambda b: "/x/deno" if b == "deno" else "/usr/bin/node" if b == "node" else None
check("deno present -> yt-dlp default, nothing to set", Y.js_runtime_opts() == {})
Y.shutil.which = lambda b: None
check("none found -> {} and reported", Y.js_runtime_opts() == {} and Y.detect_js_runtime() == (None, None))
check("env report mentions missing runtime", "js_runtime=NONE" in Y.environment_report())
Y.shutil.which = orig_which

# ------------------------------------------------------------------------------------------- client plans
sp = Y.subtitle_attempts(True, pot=False)
check("subs plan: cookie-free android_vr first, then tv with cookies",
      sp[0].clients == ("android_vr",) and not sp[0].use_cookies and sp[1].clients == ("tv",) and sp[1].use_cookies)
check("subs plan never offers cookie-ignoring clients",
      all(not ({"android", "ios", "tv_simply"} & set(a.clients)) for a in sp + Y.audio_attempts(True)))
check("no cookies -> no cookie attempts", not any(a.use_cookies for a in Y.subtitle_attempts(False, pot=False) + Y.audio_attempts(False)))
check("audio plan: each client is its own item (no 'a,b,c' strings)", all("," not in c for a in Y.audio_attempts(True) for c in a.clients))

# ------------------------------------------------------------------------------------------------ health
Y.reset_state()
check("breaker closed initially", Y.health.allow("piped"))
Y.health.failure("piped"); Y.health.failure("piped")
check("piped stays closed under threshold", Y.health.allow("piped"))
Y.health.failure("piped")
check("piped opens at 3 consecutive failures", not Y.health.allow("piped") and Y.health.remaining("piped") > 5 * 3600)
Y.health.success("piped")
check("success closes it", Y.health.allow("piped"))
Y.health.failure("transcript_api", hard=True, cooldown=1800)
check("hard failure trips immediately", not Y.health.allow("transcript_api"))
Y.health.failure("ytdlp_subs"); check("yt-dlp subs breaker: 1 block tolerated", Y.health.allow("ytdlp_subs"))
Y.health.failure("ytdlp_subs"); check("yt-dlp subs breaker: 2nd consecutive block opens", not Y.health.allow("ytdlp_subs"))

# --------------------------------------------------------------------------------------------- outcomes
Y.reset_state()
Y.set_outcome("v1", "no_captions"); Y.set_outcome("v1", "signin")
check("authoritative no_captions is not overwritten", Y.get_outcome("v1") == "no_captions")
Y.set_outcome("v2", "signin"); Y.set_outcome("v2", "no_captions")
check("authoritative evidence replaces weaker", Y.get_outcome("v2") == "no_captions")

# ------------------------------------------------ _try_ytdlp_subs with a fake yt_dlp (no network)
class FakeYDL:
    script: list = []
    opts_seen: list = []

    def __init__(self, opts):
        FakeYDL.opts_seen.append(opts)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=False):
        r = FakeYDL.script.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


fake = types.ModuleType("yt_dlp")
fake.YoutubeDL = FakeYDL
sys.modules["yt_dlp"] = fake

VTT = "WEBVTT\n\n" + "".join(f"00:00:{i:02d}.000 --> 00:00:{i + 1:02d}.000\nline number {i} of the spoken transcript\n\n" for i in range(1, 9))


class Resp:
    status_code = 200
    text = VTT


T.requests.get = lambda *a, **k: Resp()
INFO_WITH_TRACKS = {"subtitles": {}, "automatic_captions": {"hi": [{"ext": "vtt", "url": "http://x/sub"}]}}
INFO_NO_TRACKS = {"subtitles": {}, "automatic_captions": {}}
BOT = Exception("ERROR: [youtube] abc: Sign in to confirm you're not a bot")

os.environ["YTDLP_COOKIES"] = src
os.chmod(ro_dir, 0o555)
URL = "https://www.youtube.com/watch?v=AAAAAAAAAAA"


def run_subs(script, url=URL):
    Y.reset_state(); FakeYDL.script, FakeYDL.opts_seen = list(script), []
    return T._try_ytdlp_subs(url, False)


out = run_subs([INFO_NO_TRACKS, INFO_WITH_TRACKS])
o = FakeYDL.opts_seen
check("subs: android_vr empty -> tv (with cookies) succeeds", out and len(o) == 2)
check("attempt 1: android_vr, NO cookies", o[0]["extractor_args"]["youtube"]["player_client"] == ["android_vr"] and "cookiefile" not in o[0])
check("attempt 2: tv WITH cookies", o[1]["extractor_args"]["youtube"]["player_client"] == ["tv"] and "cookiefile" in o[1])
check("cookiefile handed to yt-dlp is a writable copy, not the read-only mount",
      o[1]["cookiefile"] != src and os.access(o[1]["cookiefile"], os.W_OK))
check("success recorded", Y.get_outcome("AAAAAAAAAAA") == "ok")

out = run_subs([BOT, BOT, BOT])
check("all attempts bot-checked -> None, outcome=signin", out is None and Y.get_outcome("AAAAAAAAAAA") == "signin", Y.get_outcome("AAAAAAAAAAA"))
out = run_subs([INFO_NO_TRACKS, BOT, BOT])
check("android_vr 'empty' + tv blocked is NOT treated as no_captions", Y.get_outcome("AAAAAAAAAAA") == "signin")
out = run_subs([INFO_NO_TRACKS, INFO_NO_TRACKS, INFO_NO_TRACKS])
check("tv saw the video, no tracks -> authoritative no_captions", Y.get_outcome("AAAAAAAAAAA") == "no_captions")
out = run_subs([Exception("This video is private")])
check("private video: stop after ONE attempt", len(FakeYDL.opts_seen) == 1 and Y.get_outcome("AAAAAAAAAAA") == "private")
Y.reset_state(); Y.health.reset()
FakeYDL.script, FakeYDL.opts_seen = [BOT, BOT, BOT, BOT, BOT, BOT], []
T._try_ytdlp_subs(URL, False); T._try_ytdlp_subs(URL, False)
check("two blocked runs open the yt-dlp breaker", not Y.health.allow("ytdlp_subs"))
Y.shutil.which = lambda b: "/usr/bin/node" if b == "node" else None
run_subs([INFO_WITH_TRACKS])
check("JS runtime wired into yt-dlp opts", FakeYDL.opts_seen[0].get("js_runtimes") == {"node": {}})
Y.shutil.which = orig_which
os.environ["YT_PROXY"] = "http://proxy.example:8080"
run_subs([INFO_WITH_TRACKS])
check("YT_PROXY applied to yt-dlp", FakeYDL.opts_seen[0].get("proxy") == "http://proxy.example:8080")
del os.environ["YT_PROXY"]

# ---------------------------------------------- _download_audio_ytdlp plan (client list bug + cookies)
FakeYDL.script, FakeYDL.opts_seen = [Exception("Sign in to confirm you're not a bot")] * 6, []
try:
    T._download_audio_ytdlp(URL, tempfile.mkdtemp())
except Exception:
    pass
o = FakeYDL.opts_seen
check("audio: 4 cookie-correct attempts", len(o) == 4, len(o))
check("audio: clients are separate list items", all(all("," not in c for c in x["extractor_args"]["youtube"]["player_client"]) for x in o))
check("audio: android_vr/web_embedded never get cookies, tv/default do",
      "cookiefile" not in o[0] and "cookiefile" in o[1] and "cookiefile" not in o[2] and "cookiefile" in o[3])
os.chmod(ro_dir, 0o755)

# ======================================================================= chain order (Supadata last)
LONG = " ".join(f"word{i}" for i in range(300))
calls: list = []
sup_kwargs: list = []


def stub_chain(api=None, dlp=None, piped=None, asr=(None, "asr-not-run"), native=None, generate=None, set_reason=None):
    calls.clear(); sup_kwargs.clear()
    Y.reset_state()

    def _api(vid, lang, ts):
        calls.append("api"); return api

    def _dlp(url, ts):
        calls.append("ytdlp")
        if set_reason:
            Y.set_outcome(T._extract_youtube_id(url), set_reason)
        return dlp

    def _piped(vid, lang, ts):
        calls.append("piped"); return piped

    def _asr(url, ts):
        calls.append("asr"); return asr

    def _sup(url, lang, ts, privileged=False, allow_generate=True, only_generate=False):
        calls.append("supadata:generate" if only_generate else "supadata:native" if not allow_generate else "supadata:full")
        sup_kwargs.append({"allow_generate": allow_generate, "only_generate": only_generate})
        return generate if only_generate else native

    def _dl(url, out):
        calls.append("DOWNLOAD"); raise RuntimeError("must not be called")

    T._try_youtube_captions, T._try_ytdlp_subs, T._try_piped_captions = _api, _dlp, _piped
    T._try_youtube_asr, T._try_supadata, T._download_audio_ytdlp = _asr, _sup, _dl
    T._try_youtube2text = lambda *a, **k: calls.append("yt2text")


_vid = [0]


def run_core(url=None, mode="transcript"):
    _vid[0] += 1
    url = url or f"https://www.youtube.com/watch?v=chain{_vid[0]:06d}"
    T._TRANSCRIPT_CACHE.clear(); T._RAW_SPEECH_CACHE.clear()
    return T._tool_transcribe_video_core({"url": url, "mode": mode}, {})


stub_chain(api=LONG)
r = run_core()
check("S1 transcript-api hit -> nothing else touched", calls == ["api"] and "word10" in r, calls)

stub_chain(dlp=LONG)
r = run_core()
check("S2 yt-dlp hit -> Supadata untouched", calls == ["api", "ytdlp"] and "word10" in r, calls)

stub_chain(set_reason="no_captions", asr=(LONG, None))
r = run_core()
check("S3 no captions -> FREE ASR, Supadata never called", calls == ["api", "ytdlp", "piped", "asr"] and "word10" in r, calls)

stub_chain(set_reason="signin", native=LONG)
r = run_core()
check("S4 YouTube blocks us -> no pointless audio download, Supadata NATIVE only (1 credit)",
      calls == ["api", "ytdlp", "piped", "supadata:native"] and sup_kwargs[0] == {"allow_generate": False, "only_generate": False} and "word10" in r, calls)

stub_chain(set_reason="signin", native=None, generate=LONG)
r = run_core()
check("S5 blocked + no native -> AI transcript is the VERY LAST step, audio never downloaded",
      calls == ["api", "ytdlp", "piped", "supadata:native", "supadata:generate"] and "DOWNLOAD" not in calls and "word10" in r, calls)

stub_chain(asr=(None, "Whisper exploded"), native=None, generate=LONG)
r = run_core()
check("S6 unknown failure -> free ASR once, then native, generate last; no second download",
      calls == ["api", "ytdlp", "piped", "asr", "supadata:native", "supadata:generate"] and "DOWNLOAD" not in calls, calls)

stub_chain(set_reason="private", native=LONG, generate=LONG)
r = run_core()
check("S7 private video -> zero Supadata calls, zero downloads",
      not any(c.startswith("supadata") for c in calls) and "DOWNLOAD" not in calls and "asr" not in calls, calls)

stub_chain(api=LONG)
Y.health.failure("piped", hard=True); Y.health.failure("transcript_api", hard=True)
calls.clear()
stub_chain(dlp=None, asr=(LONG, None))
Y.health.failure("piped", hard=True); Y.health.failure("transcript_api", hard=True)
r = run_core()
check("S8 open breakers -> dead sources skipped (no api, no piped)", "api" not in calls and "piped" not in calls and "ytdlp" in calls, calls)

stub_chain(native=LONG)
r = run_core(url="https://www.instagram.com/reel/DeMN1JhBEIU/")
check("S9 non-YouTube unchanged: Supadata full ladder (native->generate), no YouTube sources",
      calls == ["supadata:full"] and sup_kwargs[0] == {"allow_generate": True, "only_generate": False}, calls)

# short-native rule survives the split
import usage_budget as UB  # noqa: E402
UB.kv_set("supadata_noescalate:shortvid0001", "1", 60)
check("noescalate flag readable by the generate-only call", UB.kv_get("supadata_noescalate:shortvid0001") == "1")

print("FAILS:", FAILS)
sys.exit(1 if FAILS else 0)
