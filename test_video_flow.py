"""Offline tests: video routing / Supadata metadata / answer mode / history replay.
Run: python test_video_flow.py

Replays the 2026-10-10 prod failures:
  * "Is this true? Give me the steps @bot" + quoted reel link -> forced browse -> title "Instagram"
  * quoted 130 s voice note + "ye written text me likh kar..." -> answered with an OLDER message
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace as NS

import llm_json
import video_intent as vi

FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if cond:
        print(f"ok   {name}")
    else:
        FAILS += 1
        print(f"FAIL {name} {extra}")


class Chat:
    """completions.create -> scripted str | Exception | callable(kwargs)->str"""
    def __init__(self, *behaviors):
        self.b, self.calls, self.completions = list(behaviors), [], self

    def create(self, **kw):
        self.calls.append(kw)
        b = self.b.pop(0) if len(self.b) > 1 else self.b[0]
        if callable(b) and not isinstance(b, Exception):
            b = b(kw)
        if isinstance(b, Exception):
            raise b
        return NS(choices=[NS(message=NS(content=b))])


def client(*b):
    return NS(chat=Chat(*b))


IG = "https://www.instagram.com/reel/DeMN1JhBEIU/?stkn=OG52NmVyM24xenVi"
LOG_MSG = f"Is this true? Give me the steps @73109738680505\n\n[Quoted Message]: {IG}"

# ------------------------------------------------------------------ router
PLAN = json.dumps({"task": "answer", "wants_caption": False, "wants_stats": False, "verify": True,
                   "question": "Is this offer true? Give me the steps.", "target_words": None})
g = client(PLAN)
llm_json._dead.clear()
p = vi.route(LOG_MSG, IG, groq_client=g)
check("log message -> task=answer (was: no force, title-only browse)", p.task == "answer" and p.source == "llm")
check("verify flag set for 'is this true'", p.verify is True)
check("caption/stats NOT requested by default", not p.wants_caption and not p.wants_stats)
sent = json.loads(g.chat.calls[0]["messages"][1]["content"])
check("router sees user text without @mention/URL", sent["user_message"] == "Is this true? Give me the steps", sent["user_message"])
check("router sees quote separately", sent["quoted_message"].startswith("https://www.instagram.com"))
check("tracking query stripped from platform_url", "stkn" not in sent["platform_url"])

for task in ("summary", "key_points", "transcript", "caption", "about", "none"):
    pl = vi.route("x", IG, groq_client=client(json.dumps({"task": task})))
    check(f"task '{task}' accepted", pl.task == task)

pl = vi.route("caption batao", IG, groq_client=client(json.dumps({"task": "caption", "wants_caption": True, "wants_stats": True})))
check("explicit caption+stats ask honoured", pl.wants_caption and pl.wants_stats)

pl = vi.route("400 word summary", IG, groq_client=client(json.dumps({"task": "summary", "target_words": "400"})))
check("target_words coerced from string", pl.target_words == 400)

# LLM chain exhausted -> safety net must still avoid the title-only browse
import agent_loop as al
dead = client(Exception("429"))
llm_json._dead.clear()
pl = vi.route(LOG_MSG, IG, groq_client=dead, legacy_mode=al.video_spoken_intent, legacy_blocked=al._is_blocked_force_command)
check("LLM down: real question still -> answer (not None)", pl.task == "answer" and pl.source == "legacy", pl)
pl = vi.route(f"summarize {IG}", IG, groq_client=dead, legacy_mode=al.video_spoken_intent, legacy_blocked=al._is_blocked_force_command)
check("LLM down: 'summarize' -> summary", pl.task == "summary")
pl = vi.route(f"remind me tomorrow to watch {IG}", IG, groq_client=dead, legacy_mode=al.video_spoken_intent, legacy_blocked=al._is_blocked_force_command)
check("LLM down: reminder with incidental link -> none", pl.task == "none")
pl = vi.route(IG, IG, groq_client=dead, legacy_mode=al.video_spoken_intent, legacy_blocked=al._is_blocked_force_command)
check("bare link -> summary/about, never none", pl.task in ("summary", "about"))

bad = vi.route("hi", IG, groq_client=client(json.dumps({"task": "weird"})), legacy_mode=al.video_spoken_intent)
check("invalid task from LLM rejected -> safety net", bad.source == "legacy")

# ------------------------------------------------------------------ history replay (the quoted-voice bug)
import agent_loop

hist = [{"role": "user", "sender_id": "s", "content": "x" * 5000} for _ in range(3)]
hist += [{"role": "assistant", "sender_id": "mojo_agent", "content": "y" * 5000}]
quoted_turn = "@bot ye written text me likh do\n\n[Quoted Message]: " + ("السلام علیکم " * 200)
hist += [{"role": "user", "sender_id": "s", "content": quoted_turn}]
out = agent_loop._replay_history_messages(hist, {"s": "Zaheer"})
check("latest turn keeps the full quoted voice transcript", out[-1]["content"].endswith(quoted_turn.strip()[-40:]) and "…" not in out[-1]["content"])
check("older turns still capped (TPM budget)", all(len(m["content"]) <= agent_loop.MAX_MSG_CHARS + 20 for m in out[:-1]))
check("latest turn is attributed to its sender", out[-1]["content"].startswith("Zaheer: "))
check("quote-operand rule present in system prompt", "QUOTE = OPERAND" in open("agent_loop.py", encoding="utf-8").read())

# ------------------------------------------------------------------ tools
import agent_tools as at

DOC = {
    "platform": "instagram", "type": "video", "id": "x", "url": IG,
    "title": None, "description": "Claude Code free for 1 year! #ai #claude",
    "author": {"displayName": "AI Hub", "username": "aihub", "verified": True},
    "stats": {"likes": 5000, "comments": 200, "shares": 100, "views": 1250000},
    "media": {"type": "video", "duration": 38},
    "tags": ["ai", "claude"], "createdAt": "2026-10-09T10:30:00Z",
}
flat = at._flatten_supadata_meta(DOC)
check("metadata flattened (displayName, stats, tags, date)", flat["author_name"] == "AI Hub" and flat["views"] == 1250000 and flat["created"] == "2026-10-09")

default = at._format_media_info(flat, ["title", "caption", "author"])
check("default fields: caption + creator, NO stats/tags/date", "Caption" in default and "Creator" in default and "Stats" not in default and "Posted" not in default)
check("caption only when asked (not in stats-only render)", "Caption" not in at._format_media_info(flat, ["stats"]))
check("stats render humanized (1.2M views)", "1250000" not in at._format_media_info(flat, ["stats"]) and "👁" in at._format_media_info(flat, ["stats"]))
check("tags + date + duration formats", all(x in at._format_media_info(flat, ["tags", "date", "duration"]) for x in ("#ai", "2026-10-09", "0:38")))

k1 = at._media_key(IG)
k2 = at._media_key("https://instagram.com/reel/DeMN1JhBEIU/?igsh=zzz&utm_source=qr")
check("same reel, different tracking params -> same cache key", k1 == k2, (k1, k2))
check("different reel -> different key", k1 != at._media_key("https://www.instagram.com/reel/OTHER12345/"))
check("YouTube keeps bare video id (agent_loop reads cache by it)", at._media_key("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ")

# link_preview honours `fields`
at._supadata_metadata = lambda url, privileged=False: flat
out = at._tool_link_preview({"url": IG}, {})
check("link_preview default: caption yes, stats no", "Claude Code free" in out and "Stats" not in out)
out = at._tool_link_preview({"url": IG, "fields": ["stats", "date"]}, {})
check("link_preview stats-only on request", "Stats" in out and "Posted" in out and "Caption" not in out)

# transcribe wrapper: caption only when asked
calls = {"core": 0, "block": 0}
def fake_core(args, ctx):
    calls["core"] += 1
    return "Source: Supadata | mode=summary\n\nSUMMARY TEXT"
def fake_block(url, caption=False, stats=False, privileged=False):
    calls["block"] += 1
    return "*Caption / description:*\nCAP" if caption else ""
at._tool_transcribe_video_core, at._media_info_block = fake_core, fake_block
o = at._tool_transcribe_video({"url": IG, "mode": "summary"}, {})
check("no flags -> no caption appended, no metadata call", "CAP" not in o and calls["block"] == 0)
o = at._tool_transcribe_video({"url": IG, "mode": "summary", "include_caption": True}, {})
check("include_caption -> appended", o.endswith("CAP") and calls["block"] == 1)

# answer mode
SPEECH = "Anthropic tumhe Claude Code poore ek saal free de raha hai. Comment karo link ke liye."
seen = {}
def llm_answer(kw):
    seen["user"] = kw["messages"][1]["content"]
    seen["system"] = kw["messages"][0]["content"]
    return "*Claims:* 1 saal free. *Red flag:* 'comment karo' scam pattern."
at._client_ai, at._client_gemini, at._GEMINI_MODEL, at._MODEL_NAME = client(llm_answer), None, None, "openai/gpt-oss-120b"
llm_json._dead.clear()
at._tool_transcribe_video_core = lambda a, c: f"Source: Supadata | mode=raw\n\n{SPEECH}"
o = at._video_answer({"url": IG, "question": "Is this offer real? Give steps.", "language": "auto"}, {})
check("answer mode: question + speech reach the LLM", "Is this offer real?" in seen["user"] and "Comment karo" in seen["user"])
check("answer mode: caption NOT sent unless asked", "(not provided)" in seen["user"])
check("answer mode: ASR brand repair instructed", "Anthropic" in seen["system"] and "Claude Code" in seen["system"])
check("answer mode: Roman Urdu rule by default", "Roman Urdu" in seen["system"])
check("answer mode returns the LLM answer", o.endswith("scam pattern.") and o.startswith("Source: transcript | mode=answer"))

at._media_info_block = lambda url, c=False, s=False, privileged=False: "*Caption / description:*\nREAL CAPTION"
o = at._video_answer({"url": IG, "question": "caption me kya likha hai aur kya bola?", "include_caption": True}, {})
check("answer mode: caption included when asked", "REAL CAPTION" in seen["user"])

at._tool_transcribe_video_core = lambda a, c: "Source: Supadata metadata (about/caption) | mode=transcript\n\nOnly a caption exists"
o = at._video_answer({"url": IG, "question": "q"}, {})
check("no speech -> caption-only answer, labelled", "Only a caption exists" in seen["user"] and o.startswith("Source: caption"))

at._tool_transcribe_video_core = lambda a, c: "Is Instagram link ka public caption/title nahi mila (private)."
o = at._video_answer({"url": IG, "question": "q"}, {})
check("tool failure message passed through untouched", o.startswith("Is Instagram link"))

# long speech -> Gemini first (Groq free tier TPM)
order = []
gem = client(lambda kw: (order.append("gemini"), "long answer")[1])
grq = client(lambda kw: (order.append("groq"), "short answer")[1])
at._client_ai, at._client_gemini, at._GEMINI_MODEL = grq, gem, "gemini-x"
at._tool_transcribe_video_core = lambda a, c: "Source: Supadata | mode=raw\n\n" + ("kuch bola. " * 1200)
o = at._video_answer({"url": IG, "question": "q"}, {})
check("long transcript prefers Gemini (8K TPM Groq)", order and order[0] == "gemini" and o.endswith("long answer"))

# all LLMs down -> never an empty reply
at._client_ai, at._client_gemini = client(Exception("429")), client(Exception("503"))
llm_json._dead.clear()
at._tool_transcribe_video_core = lambda a, c: f"Source: Supadata | mode=raw\n\n{SPEECH}"
o = at._video_answer({"url": IG, "question": "q"}, {})
check("LLM chain down -> falls back to showing the speech", "Anthropic tumhe" in o)

# schemas expose the new knobs
tv = next(t for t in at.TOOL_SCHEMAS if t["function"]["name"] == "transcribe_video") if hasattr(at, "TOOL_SCHEMAS") else None
if tv is None:
    for nm in dir(at):
        v = getattr(at, nm)
        if isinstance(v, list) and v and isinstance(v[0], dict) and v[0].get("type") == "function":
            tv = next((t for t in v if t["function"]["name"] == "transcribe_video"), None)
            lp = next((t for t in v if t["function"]["name"] == "link_preview"), None)
            if tv:
                break
else:
    lp = next(t for t in at.TOOL_SCHEMAS if t["function"]["name"] == "link_preview")
check("transcribe_video schema: answer mode + question + include_* flags",
      tv and "answer" in tv["function"]["parameters"]["properties"]["mode"]["enum"]
      and all(k in tv["function"]["parameters"]["properties"] for k in ("question", "include_caption", "include_stats")))
check("link_preview schema: fields", lp and "fields" in lp["function"]["parameters"]["properties"])

print("\nall passed" if FAILS == 0 else f"\n{FAILS} failure(s)")
sys.exit(1 if FAILS else 0)
