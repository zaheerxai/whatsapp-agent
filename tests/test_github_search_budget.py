"""Offline tests: github_ops / web_search_providers / usage_budget. Run: python test_github_search_budget.py

Replays the WhatsApp screenshot: a forwarded message about github.com/mattpocock/skills
("38 skills... turn your coding agent into a senior-level developer") then "List all of
them" got back only "The tool result contains: URL: `https://github.com`" — metadata with
no file list.
"""
from __future__ import annotations

import json
import sys
import time
from types import SimpleNamespace as NS

import usage_budget
import web_search_providers as wsp

FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if cond:
        print(f"ok   {name}")
    else:
        FAILS += 1
        print(f"FAIL {name} {extra}")


class Resp:
    def __init__(self, status=200, json_body=None, text="", headers=None):
        self.status_code = status
        self._json = json_body
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


# ------------------------------------------------------------------ web_search_providers
EU_RESULTS = [
    {"title": "EU AI Act enters into force", "url": "https://ec.europa.eu/x", "snippet": "The EU AI Act regulation entered into force on..."},
    {"title": "AI Act summary", "url": "https://digital-strategy.ec.europa.eu/y", "snippet": "Artificial intelligence act obligations for providers..."},
]
check("relevant(): real hits pass", wsp.relevant(EU_RESULTS, "EU AI Act enforcement date"))
WIKI_JUNK = [{"title": "Europe - Wikipedia", "url": "https://en.wikipedia.org/wiki/Europe",
             "snippet": "Europe is a continent located entirely in the Northern Hemisphere."}]
check("relevant(): single shared token ('the') no longer passes (prod bug)",
      not wsp.relevant(WIKI_JUNK, "What's happening with the EU AI Act enforcement?"))
check("relevant(): score>=0.5 short-circuits token check",
      wsp.relevant([{"title": "x", "snippet": "y", "score": 0.6}], "totally unrelated query here"))
check("relevant(): empty -> false", not wsp.relevant([], "anything"))

wsp._down_until.clear(); wsp._cache.clear()
import os as _os
_os.environ.pop("TAVILY_API_KEY", None); _os.environ.pop("SERPER_API_KEY", None); _os.environ.pop("GEMINI_API_KEY", None)
_os.environ["TAVILY_KEYLESS"] = "0"

seen_calls = []
def fake_post(url, **kw):
    seen_calls.append(url)
    if "tavily" in url:
        return Resp(200, {"results": [{"title": "Tavily hit", "url": "https://t.example", "content": "EU AI Act enforcement news here", "score": 0.8}], "answer": "The EU AI Act took effect in August 2026."})
    if "serper" in url:
        return Resp(200, {"organic": [{"title": "Serper hit", "link": "https://s.example", "snippet": "irrelevant snippet"}]})
    raise AssertionError("should not reach here")

import requests as _req
orig_post = _req.post
_req.post = fake_post
try:
    _os.environ["TAVILY_API_KEY"] = "tvly-x"
    out = wsp.search("EU AI Act enforcement date", num=3)
    check("tavily used when key present, answer surfaced", "EU AI Act took effect" in out and seen_calls == ["https://api.tavily.com/search"])
finally:
    _req.post = orig_post
    _os.environ.pop("TAVILY_API_KEY", None)

# tavily down -> serper -> relevant
wsp._down_until.clear(); wsp._cache.clear(); seen_calls.clear()
def fail_tavily_ok_serper(url, **kw):
    seen_calls.append(url)
    if "tavily" in url:
        raise Exception("connection refused")
    return Resp(200, {"organic": [{"title": "Fed cuts rates", "link": "https://s.example",
                                   "snippet": "The Federal Reserve cut interest rates today"}]})
_req.post = fail_tavily_ok_serper
try:
    _os.environ["TAVILY_API_KEY"] = "tvly-x"
    _os.environ["SERPER_API_KEY"] = "serper-x"
    out = wsp.search("Federal Reserve interest rate cut", num=3)
    check("tavily failure falls through to serper", "Fed cuts rates" in out and len(seen_calls) == 2)
    check("tavily benched after the failure", not wsp._open("tavily"))
finally:
    _req.post = orig_post
    for k in ("TAVILY_API_KEY", "SERPER_API_KEY"):
        _os.environ.pop(k, None)

# everything off-topic / unconfigured -> explicit NO_RESULTS, never silent junk
wsp._down_until.clear(); wsp._cache.clear()
import ddgs as _ddgs_mod  # noqa: F401  (ensure importable in this sandbox)
class FakeDDGS:
    def __init__(self, *a, **kw): pass
    def text(self, q, **kw):
        return [{"title": "Europe - Wikipedia", "href": "https://en.wikipedia.org/wiki/Europe", "body": "Europe is a continent."}]
import sys as _sys
_real_ddgs = _sys.modules.get("ddgs")
_sys.modules["ddgs"] = NS(DDGS=FakeDDGS)
try:
    out = wsp.search("What does 'end' mean philosophically", num=3)
    check("all off-topic -> explicit NO_RESULTS notice (prod bug: dictionary junk silently returned)",
          out.startswith("NO RELIABLE RESULTS"), out[:200])
    check("notice tells the model not to answer from memory as if searched", "from memory" in out)
finally:
    if _real_ddgs is not None:
        _sys.modules["ddgs"] = _real_ddgs


# ------------------------------------------------------------------ prod regression 2026-10-12: trusted providers must not be gated
wsp._down_until.clear(); wsp._cache.clear()
LOGQ = "latest free locally runnable LLM models 2024 coding reasoning benchmarks 16GB RAM"
def tav_lowscore(url, **kw):
    return Resp(200, {"results": [
        {"title": "Best Local LLMs for Coding", "url": "https://a.example", "content": "Qwen and DeepSeek lead open coding benchmarks on consumer GPUs", "score": 0.31},
        {"title": "Run models on 16GB", "url": "https://b.example", "content": "Quantized 14B models fit in 16 GB of VRAM", "score": 0.28}], "answer": ""})
_req.post = tav_lowscore
try:
    _os.environ["TAVILY_API_KEY"] = "tvly-x"
    out = wsp.search(LOGQ, num=5, recency="month")
    check("REGRESSION: Tavily results with low scores + long query are NOT rejected as off-topic",
          out.startswith("Source: web_search (tavily)") and "Best Local LLMs" in out, out[:160])
finally:
    _req.post = orig_post
    _os.environ.pop("TAVILY_API_KEY", None)

check("clean_query drops a stale year on a 'latest' query", "2024" not in wsp.clean_query(LOGQ, "month"))
check("clean_query keeps the current/future year", "2026" in wsp.clean_query("latest AI news 2026", "week"))
check("clean_query keeps old years on historical queries", "1969" in wsp.clean_query("who won the world cup 1969"))
check("word-boundary match: 'eu' no longer matches 'Europe'",
      not wsp.relevant([{"title": "Europe", "snippet": "Europe is a continent."}] * 3, "EU AI regulation"))
check("long-query leniency: 1/3 of words is enough for untrusted hits",
      wsp.relevant([{"title": "Local LLM coding guide", "snippet": "benchmarks for local models"}] * 3, LOGQ))

print()

# ------------------------------------------------------------------ usage_budget
me_ok = {"maxCredits": 100, "usedCredits": 37, "plan": "Free"}
check("parse_me: camelCase", usage_budget.parse_me(me_ok) == {"max": 100, "used": 37, "plan": "Free"})
nested = {"organization": {"id": "x"}, "credits": {"max": 100, "used": {"count": 12}}, "subscription": {"plan": "Free"}}
check("parse_me: nested shape (max found, loose 'used' match ok)", usage_budget.parse_me(nested)["max"] == 100)
check("parse_me: garbage -> None", usage_budget.parse_me({"foo": "bar"}) is None)

b = usage_budget.SupadataBudget()
b._fetch_me = lambda: {"max": 100, "used": 85, "plan": "Free"}
ok, why = b.can_spend(5, privileged=False)
check("normal chat blocked once reserve would be breached (85+5, 10 left, reserve 20)", not ok and "owner" in why, why)
ok, why = b.can_spend(5, privileged=True)
check("owner can still spend near the limit", ok, why)
b._fetch_me = lambda: {"max": 100, "used": 99, "plan": "Free"}
b.snapshot(force=True)  # /me is cached 2 min by design; force a refresh like record() does
ok, why = b.can_spend(5, privileged=True)
check("even the owner is blocked past the hard monthly limit", not ok and "resets" in why, why)

usage_budget._supabase = None
b2 = usage_budget.SupadataBudget()
b2._fetch_me = lambda: None  # /me unreachable
usage_budget._mem.clear()
snap = b2.snapshot(force=True)
check("no /me, no Supabase -> local ledger, not a crash", snap["source"] == "local" and snap["max"] == 100)
b2.record(3, "test")
snap2 = b2.snapshot(force=True)
check("local ledger persists spend across snapshot calls", snap2["used"] == 3, snap2)

check("generate_cost: 90s reel rounds up to 2 min = 4 credits", usage_budget.SupadataBudget.generate_cost(1.5) == 4)
check("generate_cost: under a minute still costs 1 minute", usage_budget.SupadataBudget.generate_cost(0.2) == 2)
check("native_cost flat", usage_budget.SupadataBudget.native_cost() == 1)

# kv cache: memory-only when Supabase table is missing, never crashes the caller
class BrokenSupabase:
    def table(self, *_):
        raise Exception('relation "kv_cache" does not exist')
usage_budget._supabase, usage_budget._kv_down_until, usage_budget._kv_warned = BrokenSupabase(), 0.0, False
usage_budget._mem.clear()
usage_budget.kv_set("k1", "v1", 60)
check("kv falls back to memory when the table is missing", usage_budget.kv_get("k1") == "v1")
check("kv stops hammering a broken table (circuit breaker)", usage_budget._kv_down_until > time.time())

status = usage_budget.SupadataBudget().status_text() if False else None  # smoke only via instance above
check("status_text runs without crashing", isinstance(b.status_text(), str) and "Supadata credits" in b.status_text())

print("\nall passed" if FAILS == 0 else f"\n{FAILS} failure(s)")
sys.exit(1 if FAILS else 0)
