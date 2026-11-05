"""Offline tests: OneDrive token cache / rotation persistence (onedrive_auth) and file_ops._graph.
Run: PYTHONPATH=. python3 tests/test_onedrive_auth.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import onedrive_auth as A  # noqa: E402
import file_ops as F  # noqa: E402
import usage_budget as UB  # noqa: E402
import secret_store as SS  # noqa: E402
from cryptography.fernet import Fernet  # noqa: E402

FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if cond:
        print(f"ok   {name}")
    else:
        FAILS += 1
        print(f"FAIL {name} {extra}")


class Resp:
    def __init__(self, status=200, body=None, headers=None, text="", content=b""):
        self.status_code, self._body, self.headers, self.text, self.content = status, body, headers or {}, text, content

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def tok(at="AT", rt=None, exp=3600):
    b = {"access_token": at, "expires_in": exp}
    if rt:
        b["refresh_token"] = rt
    return Resp(200, b)


def bad(code="invalid_grant", status=400):
    return Resp(status, {"error": code, "error_description": "AADSTS70000 token expired"})


posts: list = []
script: list = []


def fake_post(url, data=None, timeout=None):
    posts.append(dict(data))
    r = script.pop(0)
    if callable(r):
        return r()
    return r


A.requests.post = fake_post
A.time.sleep = lambda s: None
os.environ["ONEDRIVE_CLIENT_ID"] = "cid"
A._T0 = time.time() - 3600               # process "booted an hour ago" unless a test says otherwise
ENV_RT = "ENV-REFRESH-TOKEN-ORIGINAL"


class MemBackend:
    def __init__(self):
        self.rows, self.up = {}, True

    def available(self):
        return self.up

    def get(self, p):
        if not self.up:
            raise SS.BackendUnavailable("down")
        return self.rows.get(p)

    def put(self, p, ct):
        if not self.up:
            raise SS.BackendUnavailable("down")
        self.rows[p] = ct

    def delete(self, p):
        self.rows.pop(p, None)


TEST_KEY = Fernet.generate_key().decode()
BACKEND = MemBackend()


def fresh(*resps, env=ENV_RT, key=TEST_KEY, backend=None):
    global BACKEND
    A.reset_state()
    UB._mem.clear()
    BACKEND = backend or MemBackend()
    SS.reset_state(); SS.set_backend(BACKEND)
    if key is None:
        os.environ.pop(SS.ENV_KEY, None)
    else:
        os.environ[SS.ENV_KEY] = key
    posts.clear(); script.clear(); script.extend(resps)
    os.environ["ONEDRIVE_REFRESH_TOKEN"] = env


# ---------------------------------------------------------------- caching: the 35-rotations-a-day bug
fresh(tok("AT1", "RT2"), tok("AT-never", "RT3"))
a = A.get_access_token(); b = A.get_access_token(); c = A.get_access_token()
check("access token cached: 3 calls -> 1 token POST", a == b == c == "AT1" and len(posts) == 1, len(posts))
check("first redemption used the env seed", posts[0]["refresh_token"] == ENV_RT and posts[0]["client_id"] == "cid")

# ---------------------------------------------------------------- rotation persisted + preferred next time
rec = SS.get("onedrive")
check("rotated refresh token persisted (encrypted store)", rec["rt"] == "RT2" and rec["seed"] == A._h(ENV_RT))
check("DB row is ciphertext: token value appears nowhere in it", "RT2" not in BACKEND.rows["onedrive"] and ENV_RT not in BACKEND.rows["onedrive"]
      and BACKEND.rows["onedrive"].startswith("gAAAA"))
check("plaintext kv_cache is NOT used any more", UB.kv_get("oauth:onedrive:rt") in (None, ""))
A.invalidate()
A.get_access_token()
check("next refresh uses the NEWEST token, not the env original", posts[1]["refresh_token"] == "RT2", posts[1])

# --------------------------------------------- survives a 'restart' (module state lost, DB remains)
db_snapshot = dict(BACKEND.rows)
fresh(tok("AT9", "RT4"))
BACKEND.rows.update(db_snapshot)
A.get_access_token()
check("after restart the persisted token is used (env original not reused)", posts[0]["refresh_token"] == "RT3", posts[0])

# -------------------------------------- seed check: re-minting the env token beats a stale stored one
fresh(tok("ATn", "RTn"))
SS.put("onedrive", {"rt": "OLD-ACCOUNT-RT", "seed": A._h("SOMETHING-ELSE")})
A.get_access_token()
check("env token changed -> stale stored token ignored", posts[0]["refresh_token"] == ENV_RT, posts[0])

# ---------------------------------------- stored rejected -> falls back to env instead of dying
fresh(bad(), tok("ATf", "RTf"))
A._mem_rt = {"rt": "STALE-STORED", "seed": A._h(ENV_RT)}
out = A.get_access_token()
check("stored token invalid_grant -> env candidate tried and works",
      out == "ATf" and [p["refresh_token"] for p in posts] == ["STALE-STORED", ENV_RT], posts)

# ---------------------------------------- everything rejected: clear error, no hammering
fresh(bad(), bad())
A._mem_rt = {"rt": "STALE-STORED", "seed": A._h(ENV_RT)}
try:
    A.get_access_token(); check("dead tokens raise", False)
except A.OneDriveAuthError as e:
    check("dead tokens -> permanent OneDriveAuthError with the fix",
          e.permanent and "get_onedrive_token.py" in str(e) and "ONEDRIVE_REFRESH_TOKEN" in str(e), str(e))
n = len(posts)
for _ in range(5):
    try:
        A.get_access_token()
    except A.OneDriveAuthError:
        pass
check("cool-off: 5 more calls -> ZERO token POSTs", len(posts) == n, (n, len(posts)))
A._dead_until = 0.0
script.append(tok("ATback", "RTback"))
check("after cool-off it retries and recovers", A.get_access_token() == "ATback")

# ---------------------------------------- startup grace: store not attached yet -> don't declare dead
fresh(bad())
BACKEND.up = False                       # token store not attached yet
A._T0 = time.time()                      # "just booted"
try:
    A.get_access_token(); check("startup rejection raises", False)
except A.OneDriveAuthError as e:
    check("rejected during startup grace -> NON-permanent, no cool-off",
          (not e.permanent) and A.status()["dead_for_s"] == 0 and "starting up" in str(e), str(e))
A._T0 = time.time() - 3600
fresh(bad())
try:
    A.get_access_token()
except A.OneDriveAuthError as e:
    check("same rejection after the grace window -> permanent + cool-off", e.permanent and A.status()["dead_for_s"] > 0)

# ---------------------------------------- transient errors: retry, never mark token bad
fresh(Resp(503, {"error": "temporarily_unavailable"}), Resp(429, {}), tok("ATt", "RTt"))
check("503 then 429 then 200 -> success", A.get_access_token() == "ATt" and len(posts) == 3)
check("transient retries all used the SAME token", len({p["refresh_token"] for p in posts}) == 1)

fresh(*[Resp(503, {})] * 3)
A._mem_rt = {"rt": "STORED", "seed": A._h(ENV_RT)}
try:
    A.get_access_token(); check("outage raises", False)
except A.OneDriveAuthError as e:
    check("outage -> NON-permanent error", not e.permanent, str(e))
check("outage does not burn through candidates (only first tried)", {p["refresh_token"] for p in posts} == {"STORED"})
check("outage does not trigger the dead cool-off", A.status()["dead_for_s"] == 0)

fresh(Resp(503, {}), bad(), tok("ATq", "RTq"))   # blip then definitive reject on stored, env works
A._mem_rt = {"rt": "STORED", "seed": A._h(ENV_RT)}
check("blip + invalid_grant on stored -> falls to env", A.get_access_token() == "ATq")

# ---------------------------------------- expiry skew
fresh(tok("AT-short", "RTs", exp=100), tok("AT-next", "RTs2"))
A.get_access_token(); A.get_access_token()
check("token inside the 120s skew is refreshed", len(posts) == 2)

# ---------------------------------------- no rotation returned
fresh(tok("ATx"))
A.get_access_token()
check("no new refresh token -> nothing persisted", BACKEND.rows == {})

# ---------------------------------------- single-flight under concurrency
fresh(lambda: (time.sleep(0.3), tok("ATsf", "RTsf"))[1], tok("AT-extra"), tok("AT-extra2"))
res = []
ts = [threading.Thread(target=lambda: res.append(A.get_access_token())) for _ in range(10)]
[t.start() for t in ts]; [t.join() for t in ts]
check("10 concurrent callers -> ONE token POST", len(posts) == 1 and set(res) == {"ATsf"}, (len(posts), set(res)))

# ---------------------------------------- memory-only persistence is flagged
fresh(tok("ATm", "RTm"))
A.get_access_token()
check("persist mode reported as durable encrypted store", A.status()["persist_mode"] == "db")
A.invalidate(); script.append(tok("ATm2", "RTm2")); A.get_access_token()
check("memory floor still prefers newest token within the process", posts[-1]["refresh_token"] == "RTm", posts[-1])

# ---------------------------------------- secrets never logged
class Cap(logging.Handler):
    def __init__(self):
        super().__init__(); self.lines = []

    def emit(self, rec):
        self.lines.append(rec.getMessage())


cap = Cap(); logging.getLogger("mojo.onedrive").addHandler(cap); logging.getLogger("mojo.onedrive").setLevel(logging.DEBUG)
fresh(tok("SECRET-ACCESS", "SECRET-NEW-RT"), bad(), bad())
A.get_access_token()
A.invalidate(); A._mem_rt = {"rt": "SECRET-NEW-RT", "seed": A._h(ENV_RT)}
try:
    A.get_access_token()
except A.OneDriveAuthError:
    pass
blob = "\n".join(cap.lines)
check("no token value ever appears in logs", all(s not in blob for s in ("SECRET-ACCESS", "SECRET-NEW-RT", ENV_RT)), blob[:300])
check("rotation is logged once per actual rotation with fingerprints", "ONEDRIVE_TOKEN rotated" in blob)

# ---------------------------------------- fail closed: no encryption key -> memory only, nothing in DB
fresh(tok("ATk", "RTk"), tok("ATk2", "RTk2"), key=None)
A.get_access_token()
check("no TOKEN_ENCRYPTION_KEY -> NOTHING written to the database", BACKEND.rows == {})
check("no key -> persist mode is memory (and process still uses newest token)", A.status()["persist_mode"] == "memory")
A.invalidate(); A.get_access_token()
check("in-process rotation still works without a key", posts[-1]["refresh_token"] == "RTk", posts[-1])

# ---------------------------------------- migration of the plaintext kv_cache token from the previous version
legacy = json.dumps({"rt": "LEGACY-PLAIN-RT", "seed": A._h(ENV_RT), "ts": 1})
fresh(tok("ATl", "RTl"))
UB.kv_set("oauth:onedrive:rt", legacy, 3600)
A.get_access_token()
check("legacy plaintext token is used (not the env original)", posts[0]["refresh_token"] == "LEGACY-PLAIN-RT", posts[0])
check("legacy token migrated into the encrypted store (then superseded by the rotated one)", SS.get("onedrive")["rt"] == "RTl")
check("plaintext copy blanked in kv_cache", UB.kv_get("oauth:onedrive:rt") in (None, ""))
check("no plaintext token in the DB row", "LEGACY-PLAIN-RT" not in BACKEND.rows["onedrive"] and "RTl" not in BACKEND.rows["onedrive"])

fresh(tok("ATl2", "RTl2"), key=None)
UB.kv_set("oauth:onedrive:rt", legacy, 3600)
A.get_access_token()
check("legacy token still USED when no key (nothing breaks)", posts[0]["refresh_token"] == "LEGACY-PLAIN-RT")
check("no key -> legacy plaintext is not copied anywhere new", BACKEND.rows == {})

fresh(tok("ATl3", "RTl3"))
UB.kv_set("oauth:onedrive:rt", json.dumps({"rt": "OTHER-ACCOUNT", "seed": A._h("different-seed"), "ts": 1}), 3600)
A.get_access_token()
check("legacy token from a different env seed is ignored", posts[0]["refresh_token"] == ENV_RT)

# ---------------------------------------- key rotation / loss never breaks auth
k_old, k_new = TEST_KEY, Fernet.generate_key().decode()
fresh(tok("A1", "R1"), key=k_old)
A.get_access_token()
os.environ[SS.ENV_KEY] = f"{k_new},{k_old}"; SS.reset_state(); A._mem_rt = None
A.invalidate(); script.append(tok("A2", "R2")); A.get_access_token()
check("after key rotation the stored token is still found (old key in list)", posts[-1]["refresh_token"] == "R1", posts[-1])
os.environ[SS.ENV_KEY] = Fernet.generate_key().decode(); SS.reset_state(); A._mem_rt = None
A.invalidate(); script.append(tok("A3", "R3")); A.get_access_token()
check("lost key -> unreadable record ignored, env seed used, auth keeps working", posts[-1]["refresh_token"] == ENV_RT, posts[-1])

# ---------------------------------------- not configured
A.reset_state(); del os.environ["ONEDRIVE_REFRESH_TOKEN"]
try:
    A.get_access_token(); check("unconfigured raises", False)
except A.OneDriveAuthError as e:
    check("unconfigured -> clear permanent error", e.permanent and "not configured" in str(e))
check("onedrive_configured() unchanged semantics", F.onedrive_configured() is False)
os.environ["ONEDRIVE_REFRESH_TOKEN"] = ENV_RT
check("onedrive_configured() true when set", F.onedrive_configured() is True)

# =================================================================================== file_ops._graph
reqs: list = []
rscript: list = []


def fake_request(method, url, headers=None, **kw):
    reqs.append({"method": method, "url": url, "auth": (headers or {}).get("Authorization"), "headers": headers, "kw": kw})
    r = rscript.pop(0)
    return r() if callable(r) else r


F.requests.request = fake_request
sleeps: list = []
F.time.sleep = lambda s: sleeps.append(s)


def gfresh(*graph, tokens=()):
    fresh(*tokens)
    reqs.clear(); rscript.clear(); rscript.extend(graph); sleeps.clear()


gfresh(Resp(200, {"ok": 1}), tokens=[tok("G1", "RT-g1")])
r = F._graph("GET", "https://graph.microsoft.com/v1.0/me/drive/root")
check("_graph sends Bearer from the shared cache", reqs[0]["auth"] == "Bearer G1" and r.status_code == 200)
rscript.append(Resp(200, {}))
F._graph("GET", "https://graph.microsoft.com/v1.0/me/drive/root")
check("second Graph call reuses the access token (no new token POST)", len(posts) == 1)

gfresh(Resp(401), Resp(200, {"ok": 1}), tokens=[tok("OLD", "R1"), tok("NEW", "R2")])
r = F._graph("GET", "https://graph.microsoft.com/v1.0/x")
check("401 -> token dropped, ONE retry with a fresh token",
      r.status_code == 200 and [q["auth"] for q in reqs] == ["Bearer OLD", "Bearer NEW"], [q["auth"] for q in reqs])

gfresh(Resp(401), Resp(401), tokens=[tok("A1", "R1"), tok("A2", "R2")])
r = F._graph("GET", "https://graph.microsoft.com/v1.0/x")
check("persistent 401 is returned after ONE retry (no loop)", r.status_code == 401 and len(reqs) == 2)

gfresh(Resp(429, headers={"Retry-After": "3"}), Resp(503), Resp(200, {}), tokens=[tok("T", "R")])
r = F._graph("GET", "https://graph.microsoft.com/v1.0/x")
check("429/503 retried with Retry-After, then success", r.status_code == 200 and len(reqs) == 3 and sleeps[0] == 3.0, sleeps)

gfresh(*[Resp(503)] * 5, tokens=[tok("T", "R")])
r = F._graph("GET", "https://graph.microsoft.com/v1.0/x")
check("retries are bounded (1 + 2) and the last response is returned", r.status_code == 503 and len(reqs) == 3)

gfresh(Resp(404), tokens=[tok("T", "R")])
check("404 is not retried", F._graph("GET", "https://graph.microsoft.com/v1.0/x").status_code == 404 and len(reqs) == 1)

# end-to-end through the public functions
gfresh(Resp(200, content=b"hello", headers={}), tokens=[tok("D", "R")])
import tempfile
p = F.download_from_onedrive("AgentLogs/a.txt", os.path.join(tempfile.mkdtemp(), "a.txt"))
check("download_from_onedrive works through _graph", open(p, "rb").read() == b"hello" and reqs[0]["auth"] == "Bearer D")

src = os.path.join(tempfile.mkdtemp(), "up.txt"); open(src, "w").write("data")
gfresh(Resp(200, {"size": 4, "name": "up.txt"}), tokens=[tok("U", "R")])
F.upload_to_onedrive(src, "AgentLogs")
check("upload sends content-type + bearer, PUT", reqs[0]["method"] == "PUT" and reqs[0]["headers"]["Content-Type"] == "application/octet-stream"
      and reqs[0]["auth"] == "Bearer U" and reqs[0]["kw"]["data"] == b"data")

gfresh(Resp(200, {"value": [{"name": "f"}]}), tokens=[tok("L", "R")])
check("list_onedrive_folder works through _graph", F.list_onedrive_folder("AgentLogs") == [{"name": "f"}])

gfresh(Resp(200, {"id": "root-a"}), Resp(404), Resp(201, {"id": "b"}), tokens=[tok("E", "R")])
meta = F.ensure_onedrive_folder("a/b")
check("ensure_onedrive_folder: probe/create via _graph with JSON content-type",
      meta == {"id": "b"} and reqs[2]["method"] == "POST" and reqs[2]["headers"]["Content-Type"] == "application/json")
check("one token for the whole multi-request operation", len(posts) == 1)

check("legacy _onedrive_access_token shim still works", (gfresh(tokens=[tok("LEG", "R")]) or True) and F._onedrive_access_token() == "LEG")

print("FAILS:", FAILS)
sys.exit(1 if FAILS else 0)
