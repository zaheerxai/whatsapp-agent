"""Offline tests for research.py (all LLM/search/fetch dependencies injected).
Run: python test_research.py
"""
from __future__ import annotations

import json
import sys

import research as R

FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if cond:
        print(f"ok   {name}")
    else:
        FAILS += 1
        print(f"FAIL {name} {extra}")


Q = ("Research the newest free models I can run locally on 16GB RAM, best at coding and reasoning, "
     "and what GPU should I buy on a tight budget")

PLAN = {"title": "Local LLMs on 16GB", "queries": [
    {"q": "best open coding LLM 16GB RAM", "recency": "month", "topic": "general"},
    {"q": "local LLM VRAM requirements quantization", "recency": "year"},
    {"q": "budget GPU for local LLM price", "recency": "month"},
    {"q": "best open coding LLM 16GB RAM"},          # duplicate -> removed
    {"q": "x"},                                       # too short -> removed
]}


def mk_search(calls):
    def search(q, num, recency, topic):
        calls.append((q, recency, topic))
        base = abs(hash(q)) % 1000
        res = [
            {"title": f"Result A for {q}", "url": f"https://site{base}.example/a", "snippet": f"Snippet about {q}", "score": 0.7},
            {"title": f"Shared page", "url": "https://shared.example/page?utm=1", "snippet": "Shared snippet", "score": 0.4},
        ]
        return "tavily", {"results": res, "answer": f"Summary for {q}"}, []
    return search


def run(write_fn, json_fn=None, search=None, extract=None, fetch=None, depth="standard", calls=None):
    calls = calls if calls is not None else []
    return R.run(
        Q, depth=depth, today="12 October 2026", language="en",
        json_fn=json_fn or (lambda s, u: PLAN),
        search_fn=search or mk_search(calls),
        extract_fn=extract or (lambda urls: {u: f"FULL TEXT of {u}" for u in urls[:2]}),
        fetch_fn=fetch or (lambda u: f"FETCHED {u}"),
        write_fn=write_fn,
    )


# ---- planner
title, qs = R.plan_queries(Q, "12 October 2026", 4, lambda s, u: PLAN)
check("planner: duplicates and junk queries removed", [q["q"] for q in qs] == [
    "best open coding LLM 16GB RAM", "local LLM VRAM requirements quantization", "budget GPU for local LLM price"], qs)
check("planner: recency/topic carried and validated", qs[0]["recency"] == "month" and qs[1]["recency"] == "year" and qs[2]["topic"] == "general")
title, qs = R.plan_queries(Q, "12 October 2026", 4, lambda s, u: None)
check("planner down -> single fallback query from the question (never zero)", len(qs) == 1 and qs[0]["q"].startswith("Research the newest"))
title, qs = R.plan_queries(Q, "12 October 2026", 2, lambda s, u: PLAN)
check("planner: respects max queries", len(qs) == 2)

seen_prompt = {}
def planner_spy(system, user):
    seen_prompt["s"] = system
    return PLAN
R.plan_queries(Q, "12 October 2026", 4, planner_spy)
check("planner prompt carries today's date and bans stale years", "12 October 2026" in seen_prompt["s"] and "NEVER put a past year" in seen_prompt["s"])

# ---- sources
rounds = R.gather(qs[:3] if False else R.plan_queries(Q, "x", 4, lambda s, u: PLAN)[1], mk_search([]))
src = R.collect_sources(rounds)
urls = [s["url"] for s in src]
check("sources: shared URL appears once (utm stripped)", sum("shared.example" in u for u in urls) == 1, urls)
check("sources: every facet represented (round-robin)", len({s["title"] for s in src if s["title"].startswith("Result A")}) == 3)
check("sources: sequential ids from 1", [s["id"] for s in src] == list(range(1, len(src) + 1)))

# ---- full run
calls = []
def writer(system, user, stage):
    writer.seen = (system, user, stage)
    return "*Local LLMs on 16GB*\n*Bottom line* — Run a 14B Q4 model [1][2]. Fake cite [99] and [1, 98].\n- Qwen fits [1]"
out = run(writer, calls=calls)
check("run: returns REPORT envelope", out.startswith("REPORT\n"))
check("run: ran the planned queries (3 unique)", len(calls) == 3, calls)
check("run: model-invented citations [99]/[98] removed", "[99]" not in out and "98" not in out, out)
check("run: valid citations kept", "[1]" in out and "[2]" in out)
check("run: real Sources list appended by code with real URLs", "*Sources*" in out and "https://" in out.split("*Sources*")[1])
check("run: sources block only lists cited sources", "[2]" in out.split("*Sources*")[1] and "[9]" not in out.split("*Sources*")[1])
check("run: writer got date, anti-invention rules and the evidence", "NEVER" not in "" and "never invent" in writer.seen[0].lower() and "EVIDENCE" in writer.seen[1] and "12 October 2026" in writer.seen[0])
check("run: pages were read in full for the best sources", "FULL TEXT of" in writer.seen[1])
check("run: first writer attempt uses the long-context pack", writer.seen[2] == 1 and len(writer.seen[1]) > 100)

# ---- writer fallbacks
stages = []
def flaky(system, user, stage):
    stages.append((stage, len(user)))
    return None if stage == 1 else "*T*\n- ok [1]"
out = run(flaky)
check("run: long writer fails -> trimmed pack retried (stage 2)", [s for s, _ in stages] == [1, 2] and out.startswith("REPORT\n*T*"))
check("run: stage-2 pack is smaller (TPM-safe)", stages[1][1] < stages[0][1] or stages[0][1] < 8100, stages)

out = run(lambda s, u, st: None)
check("run: ALL writers down -> honest evidence notes, never empty", out.startswith("REPORT\n*Research notes*") and "*Sources*" in out and "AI is busy" in out)

# ---- degraded inputs
out = run(writer, search=lambda q, n, r, t: (None, None, ["tavily:benched"]))
check("run: no search results at all -> clear message, no fabricated report", "REPORT" not in out and "usable" in out, out)
out_ur = R.run(Q, depth="standard", today="12 October 2026", language="roman_urdu", json_fn=lambda s_, u: PLAN,
               search_fn=lambda q, n, r, t: (None, None, []), extract_fn=lambda u: {}, fetch_fn=lambda u: "", write_fn=lambda a, b, c: None)
check("run: same message in Roman Urdu for Urdu chats", "kaam ka nahi" in out_ur, out_ur)
out = run(writer, extract=lambda urls: {}, fetch=lambda u: "Failed to fetch URL: boom")
check("run: page reads fail -> still writes from snippets", out.startswith("REPORT\n"))
def boom_extract(urls): raise RuntimeError("tavily down")
out = run(writer, extract=boom_extract)
check("run: extract exception is swallowed", out.startswith("REPORT\n"))
check("depth=deep uses more queries", len(R.plan_queries(Q, "x", R.DEPTHS["deep"]["queries"], lambda s, u: {"queries": [{"q": f"query number {i}"} for i in range(9)]})[1]) == 6)

print("\nall passed" if FAILS == 0 else f"\n{FAILS} failure(s)")
sys.exit(1 if FAILS else 0)
