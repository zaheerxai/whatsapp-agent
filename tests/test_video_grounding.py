"""Offline tests: video summaries stay grounded in the transcript; research intent routes to web_search.
Replays prod 2026-11-03 09:36 (a ~280-word promo reel turned into a reply with invented steps, quotas and a
'$9.99/month Pro tier'): (1) a 1000-word auto-target applied to a tiny source, (2) an ungrounded 'continue'
after finish=stop. Run: PYTHONPATH=. python3 tests/test_video_grounding.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent_tools as T  # noqa: E402
import agent_loop as A  # noqa: E402
import video_intent as VI  # noqa: E402
import llm_json  # noqa: E402

FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if cond:
        print(f"ok   {name}")
    else:
        FAILS += 1
        print(f"FAIL {name} {extra}")


T._client_ai = object()
T._MODEL_NAME = "test-model"

REEL = (
    "If you're paying monthly subscription for AI tools then stop because Google just launched 10 AI tools "
    "completely for free. Number one is Omni. You can create fully edited videos from just text, raw footage "
    "or an image. Number two is Stitch. It generates a complete landing page from a simple prompt with no design "
    "or coding needed. Number three is Whisk, a high quality image generator. Upload a picture and it writes a "
    "detailed prompt describing the visuals which you can reuse. Number four is Pomelli, which acts like a brand "
    "marketing manager and builds social media posts and ad creatives by analysing your website. Number five is "
    "Notebook LM. Dump your notes documents or YouTube videos and get summaries mind maps or a podcast style audio. "
    "These were five of the ten. If you want the other five tools just comment Google on this post and I will send "
    "you the full list. Follow for more AI tools every day."
)
WORDS = len(REEL.split())
calls: list = []


def stub(script):
    """script: list of (text, finish) returned in order by _groq_chat; records every call."""
    calls.clear()
    seq = list(script)

    def fake(msgs, max_tokens=1000, temperature=0.2, **kw):
        calls.append({"msgs": msgs, "max_tokens": max_tokens})
        return seq.pop(0) if seq else ("", "stop")

    T._groq_chat = fake


def sysmsg(i=0):
    return calls[i]["msgs"][0]["content"]


# ----------------------------------------------------------- mechanism 1: target length vs source size
stub([("*Google's 5 AI tools*\n\n- Omni: videos from text\n- Stitch: landing pages", "stop")])
out = T._refine_transcript_compact(REEL, mode="summary", target_words=1000, output_lang="en",
                                   user_request="tell me in detail and research about the other 5")
check("source is small", 150 < WORDS < 300, WORDS)
check("1000-word target is NOT passed to the model for a tiny source", "~1000" not in sysmsg() and "Aim ~" in sysmsg())
import re  # noqa: E402
aim = int(re.search(r"Aim ~(\d+) words", sysmsg()).group(1))
check("target capped to ~1.15x the source", aim <= int(WORDS * 1.15) + 2 and aim >= 80, (aim, WORDS))
check("grounding rules are in the system prompt", "GROUNDING" in sysmsg() and "does not say it" in sysmsg()
      and "comment / link / DM" in sysmsg() and "short transcript deserves a short reply" in sysmsg())
check("user is told the video simply has no more content (English)", "only has about" in out and f"{WORDS}" in out or "words of spoken content" in out, out[-160:])

stub([("*Summary*\n\n- Omni: videos", "stop")])
out = T._refine_transcript_compact(REEL, mode="summary", target_words=1000, output_lang="roman_urdu",
                                   user_request="detail me batao")
check("Roman Urdu footnote variant", "alfaaz" in out and "poora content" in out, out[-120:])

stub([("*Summary*\n\n- Omni: videos from text", "stop")])
out = T._refine_transcript_compact(REEL, mode="summary", target_words=100, output_lang="en", user_request="100 word summary")
check("a target BELOW the source size is honoured (no cap, no footnote)", "Aim ~100 words" in sysmsg() and "spoken content" not in out)

# ------------------------------------------ mechanism 2: no ungrounded continuation after finish=stop
stub([("*Google's 5 free AI tools*\n\n• *Omni* – videos from text\n• *Notebook LM* – summaries and audio", "stop")])
out = T._refine_transcript_compact(REEL, mode="summary", target_words=None, output_lang="en", user_request="summarize")
check("finish=stop -> exactly ONE model call (no 'continue' padding)", len(calls) == 1, len(calls))

# finish=length: continuation allowed but GROUNDED
stub([("*Summary*\n\n- Omni: videos from text\n- Stitch: landing", "length"),
      ("pages from a prompt.\n- Whisk: image generator.", "stop")])
out = T._refine_transcript_compact(REEL, mode="summary", target_words=None, output_lang="en", user_request="summarize")
check("finish=length -> continuation runs", len(calls) >= 2)
cs = calls[1]["msgs"][0]["content"]
check("continuation carries the SOURCE transcript", "SOURCE (the ONLY facts you may use)" in cs and "Omni" in cs and "comment Google" in cs)
check("continuation forbids steps/pricing/quotas unless in the source",
      "Do NOT add how-to steps" in cs and "pricing" in cs and "quotas" in cs)
check("continuation can say DONE", "exactly: DONE" in cs)

stub([("*Summary*\n\n- Omni: videos from text", "length"), ("DONE", "stop")])
out = T._refine_transcript_compact(REEL, mode="summary", target_words=None, output_lang="en", user_request="summarize")
check("DONE adds nothing", "DONE" not in out and out.count("Omni") == 1, out)

stub([("*Summary*\n\n- Omni: videos from text", "length"), ("- Pomelli: brand posts from your website.", "stop")])
out = T._refine_transcript_compact(REEL, mode="summary", target_words=None, output_lang="en", user_request="summarize")
check("legit, source-backed continuation is still appended", "Pomelli" in out)

# unknown finish reason: the incompleteness heuristic still applies (provider fallbacks)
stub([("*Summary*\n\n- Omni: videos from text and then…", None), ("raw footage.", "stop")])
out = T._refine_transcript_compact(REEL, mode="summary", target_words=None, output_lang="en", user_request="summarize")
check("unknown finish + cut-off text -> continues (grounded)", len(calls) >= 2 and "SOURCE" in calls[1]["msgs"][0]["content"])

# ---------------------------------------------------------- _continue_text_only direct behaviour
stub([("addition", "stop")])
T._continue_text_only("so far", lang_rule="OUTPUT: English", rounds=1)
check("no source -> legacy prompt unchanged (back-compat)", "SOURCE" not in sysmsg())
stub([("x", "stop")])
T._continue_text_only("so far", lang_rule="OUTPUT: English", rounds=1, source="A" * 30000)
check("huge source is bounded (no 413)", len(sysmsg()) < T._CONTINUE_SOURCE_CHARS + 1500, len(sysmsg()))

# ---------------------------------------------------- map-reduce merge continuation is grounded too
LONG = " ".join(f"fact{i} is stated here." for i in range(380))            # ~9k chars: > 7000 (map-reduce), 2 chunks
stub([("part one summary", "stop"), ("part two summary", "stop"), ("merged summary cut off", "length"), ("DONE", "stop")])
out = T._refine_transcript_compact(LONG, mode="summary", target_words=None, output_lang="en", user_request="summarize")
cont = [c for c in calls if "Continue the WhatsApp reply" in c["msgs"][0]["content"]]
check("map-reduce merge continuation carries its source", cont and "SOURCE (the ONLY facts you may use)" in cont[0]["msgs"][0]["content"]
      and "part one summary" in cont[0]["msgs"][0]["content"], len(cont))

# ============================================================================== router: research intent
payload_research = {"task": "summary", "wants_caption": False, "wants_stats": False, "verify": False,
                    "research": True, "research_query": "Google free AI tools Omni Stitch Whisk Pomelli NotebookLM full list",
                    "question": "research the other 5 tools", "target_words": None}
llm_json.json_chat = lambda *a, **k: payload_research
p = VI.route("tell me in detail and research about the other 5 https://instagram.com/reel/x", "https://instagram.com/reel/x")
check("router: research flag + query surfaced", p.research is True and "Google free AI tools" in p.research_query and p.task == "summary")
check("router prompt documents the research field", '"research": boolean' in VI._SYSTEM and "research_query" in VI._SYSTEM
      and "the other N" in VI._SYSTEM)

llm_json.json_chat = lambda *a, **k: {**payload_research, "research": False}
p = VI.route("summarize this https://instagram.com/reel/x", "https://instagram.com/reel/x")
check("no research -> flag False and a stray query is dropped", p.research is False and p.research_query == "")

llm_json.json_chat = lambda *a, **k: None
p = VI.route("what is this", "https://x.com/y", legacy_mode=lambda t: None, legacy_blocked=lambda t: False)
check("offline fallback never claims research", p.research is False and p.source == "legacy")
check("VideoPlan default has research off", VI.VideoPlan().research is False and VI.VideoPlan().research_query == "")

# ======================================================================= agent_loop follow-up decision
plan = lambda **kw: VI.VideoPlan(**kw)
check("research + web_search allowed -> follow-up 'research'", A._video_followup_kind(plan(task="summary", research=True), "x" * 50, True) == "research")
check("verify still works", A._video_followup_kind(plan(task="answer", verify=True), "x" * 50, True) == "verify")
check("verify takes priority over research", A._video_followup_kind(plan(verify=True, research=True), "x" * 50, True) == "verify")
check("web_search disabled -> direct reply path", A._video_followup_kind(plan(research=True), "x" * 50, False) is None)
check("no video content -> no follow-up", A._video_followup_kind(plan(research=True), "", True) is None and A._video_followup_kind(plan(research=True), None, True) is None)
check("plain summary -> no follow-up", A._video_followup_kind(plan(task="summary"), "x" * 50, True) is None)
check("no plan -> no follow-up", A._video_followup_kind(None, "x" * 50, True) is None)

note = A._RESEARCH_FOLLOWUP_NOTE
check("research note: two labelled parts", "*From the video*" in note and "*From the web*" in note)
check("research note: bounded search + no invention", "at most 2" in note and "never fill gaps from memory" in note
      and "Do NOT state prices, quotas, limits" in note)
check("research note: video is the only source for video claims", "ONLY source for what the video says" in note)
check("research note: says what the video does NOT say", "does NOT say or name" in note)
check("hint placeholder is formattable", "{hint}" in note and "{hint}" not in note.replace("{hint}", " (x)"))
check("web-search-off note is honest and short", "web search is turned off" in A._RESEARCH_OFF_NOTE and len(A._RESEARCH_OFF_NOTE) < 160)

print("FAILS:", FAILS)
sys.exit(1 if FAILS else 0)
