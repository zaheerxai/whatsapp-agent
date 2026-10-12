"""Offline tests for the Supadata credit policy (free plan = 100 credits/month).
Run: python test_supadata_credits.py

A fake Supadata bills like the real one: native = 1 credit (0 if unavailable),
generate = 2 credits per started video minute, /v1/me reports max/used.
"""
from __future__ import annotations

import json
import math
import os
import sys
from types import SimpleNamespace as NS

os.environ["SUPADATA_API_KEY"] = "test-key"

import agent_tools as at
import usage_budget as ub

FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if cond:
        print(f"ok   {name}")
    else:
        FAILS += 1
        print(f"FAIL {name} {extra}")


IG = "https://www.instagram.com/reel/DeMN1JhBEIU/?stkn=abc"
SPEECH = "Anthropic is giving everyone one free year of Claude Code. Comment CLAUDE for the link."


class FakeSupadata:
    def __init__(self, native=None, generate=None, minutes=1.5, used=0, maxc=100):
        self.native, self.generate, self.minutes = native, generate, minutes
        self.used, self.max = used, maxc
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        if url.endswith("/v1/me"):
            return NS(status_code=200, json=lambda: {"maxCredits": self.max, "usedCredits": self.used, "plan": "Free"})
        mode = (params or {}).get("mode")
        self.calls.append(mode)
        if mode == "native":
            if self.native is None:
                return NS(status_code=206, text="unavailable", json=lambda: {})
            self.used += 1
            return NS(status_code=200, text="", json=lambda: {"content": self.native, "lang": "en"})
        if mode == "generate":
            if self.generate is None:
                return NS(status_code=500, text="boom", json=lambda: {})
            self.used += 2 * math.ceil(self.minutes)
            return NS(status_code=200, text="", json=lambda: {"content": self.generate, "lang": "en",
                                                              "duration": self.minutes * 60})
        raise AssertionError(f"unexpected mode {mode}")


def setup(fake):
    at.requests.get = fake.get
    ub.requests.get = fake.get
    ub._mem.clear()
    ub._supabase = None
    ub.supadata = ub.SupadataBudget()  # fresh /me cache
    at.usage_budget.supadata = ub.supadata
    at._META_CACHE.clear()


# 1. native available -> exactly ONE call, 1 credit (old code: generate first = 2+/min)
f = FakeSupadata(native=SPEECH, generate="SHOULD NOT BE USED")
setup(f)
out = at._try_supadata(IG, "auto", False)
check("IG: native tried FIRST and used", out and "free year" in out, out)
check("IG: no paid AI call when native exists", f.calls == ["native"], f.calls)
check("IG: cost 1 credit", f.used == 1, f.used)

# 2. no native -> exactly one generate (never 'auto' on top = double billing)
f = FakeSupadata(native=None, generate=SPEECH, minutes=1.5)
setup(f)
out = at._try_supadata(IG, "auto", False)
check("native unavailable -> falls to generate", out and "free year" in out)
check("exactly native then generate (no 'auto')", f.calls == ["native", "generate"], f.calls)
check("90 s reel billed 4 credits (2 min x 2) and tracked", f.used == 4, f.used)

# 3. short-but-real native transcript must NOT escalate to paid AI
f = FakeSupadata(native="ok bye", generate=SPEECH)
setup(f)
out = at._try_supadata(IG, "auto", False)
check("tiny native transcript: no escalation to paid generate", out is None and f.calls == ["native"], (out, f.calls))

# 4. budget: normal chat blocked near the limit, owner allowed
f = FakeSupadata(native=SPEECH, used=95)  # 5 left, reserve 20
setup(f)
out = at._try_supadata(IG, "auto", False, privileged=False)
check("normal chat: ZERO transcript calls when under owner reserve", out is None and f.calls == [], f.calls)
out = at._try_supadata(IG, "auto", False, privileged=True)
check("owner: still served from the reserve", out and f.calls == ["native"], f.calls)

# 5. nobody is served past the hard limit
f = FakeSupadata(native=SPEECH, used=100)
setup(f)
out = at._try_supadata(IG, "auto", False, privileged=True)
check("100/100 used: not even the owner makes a call", out is None and f.calls == [], f.calls)

# 6. real generate failure remembered; budget block is NOT remembered as a failure
f = FakeSupadata(native=None, generate=None)
setup(f)
at._try_supadata(IG, "auto", False)
n1 = len(f.calls)
out = at._try_supadata(IG, "auto", False)
check("failed reel: second attempt makes ZERO calls (6h negative cache)", out is None and len(f.calls) == n1, f.calls)

f = FakeSupadata(native=SPEECH, used=95)
setup(f)
at._try_supadata(IG, "auto", False, privileged=False)   # blocked by budget, no HTTP
out = at._try_supadata(IG, "auto", False, privileged=True)
check("budget block does NOT poison the owner's retry", out and "free year" in out)

# 7. long videos: AI transcription skipped for normal chats above the minute cap
f = FakeSupadata(native=None, generate=SPEECH, minutes=12)
setup(f)
ub.kv_set(f"meta:{at._media_key(IG)}", json.dumps({"duration": 12 * 60}), 3600)
out = at._try_supadata(IG, "auto", False, privileged=False)
check("12-min video: AI-generate skipped for normal chat (would cost 24)", out is None and f.calls == ["native"], f.calls)
f2 = FakeSupadata(native=None, generate=SPEECH, minutes=12)
setup(f2)
ub.kv_set(f"meta:{at._media_key(IG)}", json.dumps({"duration": 12 * 60}), 3600)
out = at._try_supadata(IG, "auto", False, privileged=True)
check("12-min video: owner may still do it", out and f2.calls == ["native", "generate"], f2.calls)

# 8. persisted metadata is read back without paying again
f = FakeSupadata(native=SPEECH)
setup(f)
ub.kv_set(f"meta:{at._media_key(IG)}", json.dumps({"title": "T", "caption": "cached caption", "author_name": "A",
                                                   "username": "", "verified": False, "tags": [], "created": ""}), 3600)
m = at._supadata_metadata(IG)
check("metadata served from persistent cache (0 credits)", m and m["caption"] == "cached caption" and f.used == 0)

# 9. /credits status text
setup(FakeSupadata(used=37))
t = ub.supadata.status_text()
check("/credits shows used/left and the cost model", "37/100" in t and "63 left" in t and "2/min" in t, t)

print("\nall passed" if FAILS == 0 else f"\n{FAILS} failure(s)")
sys.exit(1 if FAILS else 0)
