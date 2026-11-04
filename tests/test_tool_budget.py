"""Offline tests for tool_budget.py, router budget-fitting and agent_loop._fit_tools_for_budget (no network).
Run: PYTHONPATH=. python3 tests/test_tool_budget.py
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm_router as R  # noqa: E402
import tool_budget as TB  # noqa: E402

FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if cond:
        print(f"ok   {name}")
    else:
        FAILS += 1
        print(f"FAIL {name} {extra}")


def schema(name, desc="Does a thing. " * 40):
    return {"type": "function", "function": {"name": name, "description": desc,
                                              "parameters": {"type": "object", "properties": {}}}}


SCHEMAS = [schema(n) for n in ("get_memory", "forget_memory", "web_search", "browse_url", "set_reminder")]

# --- catalog / subset -------------------------------------------------------------------------------
cat = TB.catalog(SCHEMAS)
check("catalog has every tool", [c["name"] for c in cat] == [tool["function"]["name"] for tool in SCHEMAS])
check("catalog summaries bounded", all(len(c["use"]) <= TB._SUMMARY_CHARS + 2 for c in cat))
check("subset keeps ORIGINAL order", [TB.tool_name(s) for s in TB.subset(SCHEMAS, ["web_search", "get_memory"])]
      == ["get_memory", "web_search"])

# --- plan_tools (fake json_chat) --------------------------------------------------------------------
calls = []


def fake_json(answer):
    def _f(system, user, **kw):
        calls.append((kw.get("gemini_model"), user))
        return answer
    return _f


names = TB.plan_tools(SCHEMAS, "forget my notes about X", ["user: hi"], gemini_client=object(), models=["m1"],
                      json_chat=fake_json({"tools": ["get_memory", "forget_memory", "not_a_tool", "get_memory"]}))
check("unknown + duplicate names dropped", names == ["get_memory", "forget_memory"], names)
check("catalog + message sent to planner", '"latest_message": "forget my notes about X"' in calls[-1][1]
      and "forget_memory" in calls[-1][1])
check("empty selection is valid (no tools)", TB.plan_tools(SCHEMAS, "hi", [], gemini_client=object(), models=["m1"],
                                                            json_chat=fake_json({"tools": []})) == [])
check("max_tools enforced", len(TB.plan_tools(SCHEMAS, "x", [], gemini_client=object(), models=["m1"], max_tools=2,
                                              json_chat=fake_json({"tools": [t["function"]["name"] for t in SCHEMAS]}))) == 2)

seq = {"m1": None, "m2": {"tools": ["web_search"]}}
failed = []
n = TB.plan_tools(SCHEMAS, "x", [], gemini_client=object(), models=["m1", "m2"],
                  json_chat=lambda s, u, **kw: seq[kw["gemini_model"]], mark_failed=lambda m, s: failed.append(m))
check("falls to next planner model + marks failed one", n == ["web_search"] and failed == ["m1"], (n, failed))
check("all planners fail -> None (caller fails open)",
      TB.plan_tools(SCHEMAS, "x", [], gemini_client=object(), models=["m1"], json_chat=fake_json(None)) is None)
cooled_calls = []
check("cooled planner models skipped",
      TB.plan_tools(SCHEMAS, "x", [], gemini_client=object(), models=["m1"], is_cooled=lambda m: True,
                    json_chat=lambda *a, **k: cooled_calls.append(1)) is None and not cooled_calls)
check("no schemas / no client -> None", TB.plan_tools([], "x", gemini_client=object(), models=["m"]) is None
      and TB.plan_tools(SCHEMAS, "x", gemini_client=None, models=["m"]) is None)

# --- router: trim_history / fit_messages -----------------------------------------------------------
def u(t): return {"role": "user", "content": t}
def a(t): return {"role": "assistant", "content": t}


hist = [{"role": "system", "content": "SYS1"}] + [x for i in range(10) for x in (u("q%d " % i + "x" * 1500), a("a%d " % i + "x" * 1500))]
hist += [{"role": "system", "content": "NOTE long_reply=1"}, u("LATEST")]
t = R.trim_history(hist, 4)
check("trim keeps ALL system messages", [m["content"] for m in t if m["role"] == "system"] == ["SYS1", "NOTE long_reply=1"])
check("trim keeps latest user turn", t[-1]["content"] == "LATEST")
check("trim keeps only last 4 non-system", sum(1 for m in t if m["role"] != "system") == 4)

orphan = [{"role": "system", "content": "S"}, u("old"), {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
          {"role": "tool", "tool_call_id": "1", "content": "r"}, a("done"), u("new")]
t = R.trim_history(orphan, 3)
check("orphaned leading tool message dropped", t[1]["role"] != "tool", [m["role"] for m in t])

budget = R.groq_budget()
fit = R.fit_messages(hist, None, 1024, budget)
check("fit_messages trims to budget", fit is not None and fit[1] <= budget and len(fit[0]) < len(hist), fit and fit[1])
check("fit_messages None when sys alone too big",
      R.fit_messages([{"role": "system", "content": "x" * 40000}, u("hi")], None, 1024, budget) is None)
check("fit_messages untouched when already fits", R.fit_messages([u("hi")], None, 100, budget)[0] == [u("hi")])

# --- router end-to-end: oversize history is trimmed, Groq still serves --------------------------------
class FakeClient:
    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        r = self.script.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def resp(text, tokens=900, cached=None):
    usage = NS(total_tokens=tokens, prompt_tokens_details=NS(cached_tokens=cached))
    return NS(choices=[NS(message=NS(content=text, tool_calls=None), finish_reason="stop")], usage=usage)


R.reset_state()
g = FakeClient([resp("from-groq", cached=512)])
msg, fin, prov = R.complete(hist, None, 0.3, 1024, groq_client=g, groq_model="openai/gpt-oss-120b", groq_sibling=None,
                            gemini_client=None, gemini_model=None, shrink=lambda m, k: m, flatten=lambda m: m)
sent = g.calls[0]["messages"]
check("oversize turn now served by Groq (trimmed)", prov == "groq" and msg.content == "from-groq")
check("sent request is smaller than original and keeps system + latest",
      len(sent) < len(hist) and sent[-1]["content"] == "LATEST" and sent[0]["content"] == "SYS1")

# --- agent_loop hook ---------------------------------------------------------------------------------
import agent_loop as A  # noqa: E402

BIG = [schema("t%d" % i, "word " * 300) for i in range(12)]          # ~ big toolset
msgs = [{"role": "system", "content": "S" * 3000}, u("do the thing")]
A._client_ai, A._client_gemini, A._GEMINI_MODEL = object(), object(), "gemini-3.6-flash"
R.reset_state()
import llm_json  # noqa: E402
orig_json_chat = llm_json.json_chat
llm_json.json_chat = lambda system, user, **kw: {"tools": ["t3", "t7"]}
sub = A._fit_tools_for_budget(msgs, BIG, 1024, "do the thing")
check("hook narrows oversize toolset via planner", [TB.tool_name(s) for s in sub] == ["t3", "t7"], sub and len(sub))
check("narrowed request now fits Groq budget", R.est_tokens(msgs, sub, 1024) <= R.groq_budget())

llm_json.json_chat = lambda system, user, **kw: {"tools": []}
check("planner [] -> None (chat-only turn)", A._fit_tools_for_budget(msgs, BIG, 1024, "hi") is None)

llm_json.json_chat = lambda system, user, **kw: None
R.reset_state()
check("planner failure -> full set kept (fail-open)", A._fit_tools_for_budget(msgs, BIG, 1024, "x") == BIG)

llm_json.json_chat = lambda system, user, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
R.reset_state()
check("planner exception -> full set kept", A._fit_tools_for_budget(msgs, BIG, 1024, "x") == BIG)

llm_json.json_chat = lambda system, user, **kw: (_ for _ in ()).throw(AssertionError("must not be called"))
SMALL = BIG[:1]
check("fits already -> planner NOT called", A._fit_tools_for_budget(msgs, SMALL, 1024, "x") == SMALL)
os.environ["TOOL_PLANNER"] = "0"
check("TOOL_PLANNER=0 disables", A._fit_tools_for_budget(msgs, BIG, 1024, "x") == BIG)
del os.environ["TOOL_PLANNER"]
A._client_ai = None
check("no Groq client -> nothing to protect", A._fit_tools_for_budget(msgs, BIG, 1024, "x") == BIG)
llm_json.json_chat = orig_json_chat

# --- the real owner payload: measure before/after ---------------------------------------------------
from agent_tools import TOOL_SCHEMAS  # noqa: E402
sysp = A._build_system_prompt("c", "s", False, "TIME", "TAGS", "MEMORY")
m_real = [{"role": "system", "content": sysp}] + [x for i in range(6) for x in (u("q" + "x" * 500), a("a" + "x" * 500))] + [u("latest")]
full = R.est_tokens(m_real, TOOL_SCHEMAS, 1024)
for pick in (["browse_url", "web_search"], ["get_memory", "forget_memory"], ["set_reminder", "list_reminders", "cancel_reminders"],
             ["web_search", "deep_research", "browse_url", "search_knowledge", "get_weather", "send_message_to"]):
    e = R.est_tokens(m_real, TB.subset(TOOL_SCHEMAS, pick), 1024)
    print(f"     full={full} -> {len(pick)} tools est={e} fits={e <= R.groq_budget()} ({', '.join(pick)})")
check("real owner payload: full set does NOT fit (the bug)", full > R.groq_budget(), full)
check("real owner payload: typical 2-3 tool subset fits", R.est_tokens(m_real, TB.subset(TOOL_SCHEMAS, ["browse_url", "web_search"]), 1024) <= R.groq_budget())

print("FAILS:", FAILS)
sys.exit(1 if FAILS else 0)
