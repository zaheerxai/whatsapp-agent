"""Offline tests: Gemini tool calling + thought signatures through llm_router / agent_loop (no network).
Run: PYTHONPATH=. python3 tests/test_gemini_tools.py
"""
from __future__ import annotations

import copy
import json
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm_router as R  # noqa: E402
import agent_loop as A  # noqa: E402

FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if cond:
        print(f"ok   {name}")
    else:
        FAILS += 1
        print(f"FAIL {name} {extra}")


class HTTPErr(Exception):
    def __init__(self, status):
        super().__init__(f"Error code: {status}")
        self.status_code = status
        self.response = NS(headers={})
        self.body = {}


class FakeClient:
    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(copy.deepcopy(kw))
        r = self.script.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def resp(text="ok", tool_calls=None):
    return NS(choices=[NS(message=NS(content=text, tool_calls=tool_calls), finish_reason="stop")],
              usage=NS(total_tokens=100, prompt_tokens_details=NS(cached_tokens=None)))


TOOLS = [{"type": "function", "function": {"name": "web_search", "description": "d", "parameters": {
    "type": "object",
    "properties": {"query": {"type": "string"},
                   "num_results": {"type": "integer", "default": 5},
                   "default": {"type": "string", "description": "a property literally named default"},
                   "recency": {"type": "string", "enum": ["", "day"], "default": ""}},
    "required": ["query"]}}}]
SCHEMA_BEFORE = copy.deepcopy(TOOLS)

SIG = "CvcQAdHtim/REALSIGNATURE=="
HIST = [
    {"role": "system", "content": "SYS-A"}, {"role": "system", "content": "SYS-B"},
    {"role": "user", "content": "search two things"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "web_search", "arguments": '{"query":"a"}'},
         "extra_content": {"google": {"thought_signature": SIG}}},
        {"id": "c2", "type": "function", "function": {"name": "web_search", "arguments": '{"query":"b"}'}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "res a"},
    {"role": "tool", "tool_call_id": "c2", "content": {"obj": 1}},
    {"role": "system", "content": "NOTE mid"},
]


def gem_run(gem, messages=HIST, tools=TOOLS, **kw):
    return R.complete(messages, tools, 0.3, 1024, groq_client=None, groq_model=None, groq_sibling=None,
                      gemini_client=gem, gemini_model="g-a", gemini_fallbacks=[],
                      shrink=lambda m, k: m, flatten=lambda m: [{"role": "user", "content": "FLAT"}], **kw)


R.time.sleep = lambda s: None

# --- schema sanitizer ----------------------------------------------------------------------------------
cl = R.gemini_tool_schemas(TOOLS)
props = cl[0]["function"]["parameters"]["properties"]
check("'default' keyword stripped from nested schema", "default" not in props["num_results"] and "default" not in props["recency"])
check("a PROPERTY named default is kept", "default" in props and props["default"]["type"] == "string")
check("enum/required kept", props["recency"]["enum"] == ["", "day"] and cl[0]["function"]["parameters"]["required"] == ["query"])
check("original schema object untouched", TOOLS == SCHEMA_BEFORE)

# --- tool turn on Gemini -------------------------------------------------------------------------------
R.reset_state()
gem = FakeClient([resp("done")])
msg, fin, prov = gem_run(gem)
kw = gem.calls[0]
check("tools offered to Gemini + tool_choice auto", kw.get("tool_choice") == "auto" and kw["tools"][0]["function"]["name"] == "web_search")
check("offered schemas are sanitized", "default" not in kw["tools"][0]["function"]["parameters"]["properties"]["num_results"])
sent = kw["messages"]
check("history NOT flattened (tool_calls kept)", any(m.get("tool_calls") for m in sent) and all(m["content"] != "FLAT" for m in sent if "content" in m))
check("leading system messages merged", sent[0]["role"] == "system" and "SYS-A" in sent[0]["content"] and "SYS-B" in sent[0]["content"]
      and sum(1 for m in sent if m["role"] == "system") == 1)
check("later system message became a user note", any(m["role"] == "user" and m["content"].startswith("[System note]") for m in sent))
check("tool result content is always a string", all(isinstance(m["content"], str) for m in sent if m["role"] == "tool"))
tcs = next(m for m in sent if m.get("tool_calls"))["tool_calls"]
check("real thought_signature replayed verbatim", tcs[0]["extra_content"]["google"]["thought_signature"] == SIG)
check("parallel 2nd call gets no signature", "extra_content" not in tcs[1])
check("input history not mutated", "extra_content" not in HIST[3]["tool_calls"][1] and HIST[5]["content"] == {"obj": 1})

# --- foreign tool call (from Groq) gets the documented dummy signature ----------------------------------
foreign = [{"role": "user", "content": "q"},
           {"role": "assistant", "content": None, "tool_calls": [
               {"id": "g1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}]},
           {"role": "tool", "tool_call_id": "g1", "content": "r"}]
R.reset_state(); gem = FakeClient([resp()])
gem_run(gem, foreign)
tc = next(m for m in gem.calls[0]["messages"] if m.get("tool_calls"))["tool_calls"][0]
check("Groq-made call gets dummy signature", tc["extra_content"]["google"]["thought_signature"] == "skip_thought_signature_validator")

# --- Groq never receives extra_content -----------------------------------------------------------------
R.reset_state(); R._TPM_SAFETY = 0.92
g = FakeClient([resp("from-groq")])
R.complete(HIST[2:6], None, 0.3, 512, groq_client=g, groq_model="openai/gpt-oss-120b", groq_sibling=None,
           gemini_client=None, gemini_model=None, shrink=lambda m, k: m, flatten=lambda m: m)
check("extra_content stripped for Groq", not any("extra_content" in t for m in g.calls[0]["messages"] for t in (m.get("tool_calls") or [])))
check("caller history keeps the signature", HIST[3]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == SIG)

# --- fail-safe: Gemini rejects tools -> flattened retry, remembered -----------------------------------
R.reset_state()
gem = FakeClient([HTTPErr(400), resp("flat-ok"), resp("flat-again")])
msg, _, _ = gem_run(gem)
check("400 on tools -> same model, no tools, flattened", msg.content == "flat-ok" and len(gem.calls) == 2
      and "tools" in gem.calls[0] and "tools" not in gem.calls[1] and gem.calls[1]["messages"][0]["content"] == "FLAT")
msg, _, _ = gem_run(gem)
check("rejection remembered (no 400 round trip next time)", len(gem.calls) == 3 and "tools" not in gem.calls[2])
R._gem_tools_bad["g-a"] = 0.0
gem.script.append(resp("tools-again"))
gem_run(gem)
check("tools retried after TTL expiry", "tools" in gem.calls[3])

# --- opt-out + synthesis turns unchanged --------------------------------------------------------------
R.reset_state(); old = R._GEMINI_TOOLS; R._GEMINI_TOOLS = False
gem = FakeClient([resp()]); gem_run(gem)
check("GEMINI_TOOLS=0 -> previous flat/no-tools behaviour", "tools" not in gem.calls[0] and gem.calls[0]["messages"][0]["content"] == "FLAT")
R._GEMINI_TOOLS = old
R.reset_state(); gem = FakeClient([resp()]); gem_run(gem, tools=None)
check("synthesis turn (no tools) still flat", "tools" not in gem.calls[0] and gem.calls[0]["messages"][0]["content"] == "FLAT")

# --- 400 on reasoning after tools rejected still recovers (3 attempts) -------------------------------
R.reset_state()
gem = FakeClient([HTTPErr(400), HTTPErr(400), resp("recovered")])
msg, _, _ = gem_run(gem)
check("tools-400 then reasoning-400 still recovers", msg.content == "recovered" and len(gem.calls) == 3
      and "extra_body" not in gem.calls[2] and "tools" not in gem.calls[2])

# --- agent_loop keeps the signature in history --------------------------------------------------------
fn = NS(name="web_search", arguments='{"query":"x"}')
obj_tc = NS(id="t1", function=fn, extra_content={"google": {"thought_signature": SIG}})
e = A._tool_call_entry(obj_tc, "call_0")
check("object-style call: signature carried", e["extra_content"]["google"]["thought_signature"] == SIG and e["function"]["name"] == "web_search")
check("no signature -> no extra_content key", "extra_content" not in A._tool_call_entry(NS(id="t2", function=fn), "call_0"))
check("pydantic model_extra style", A._tool_call_entry(NS(id="t3", function=fn, model_extra={"extra_content": {"google": {"thought_signature": SIG}}}), "c")
      ["extra_content"]["google"]["thought_signature"] == SIG)
dict_msg = NS(tool_calls=[{"id": "d1", "function": {"name": "web_search", "arguments": "{}"}, "extra_content": {"google": {"thought_signature": SIG}}}])
norm = A._normalize_tool_calls(dict_msg)
check("dict-style call round-trips signature", A._tool_call_entry(norm[0], "c")["extra_content"]["google"]["thought_signature"] == SIG)

try:  # the real SDK object: do pydantic extras expose extra_content as an attribute?
    from openai.types.chat import ChatCompletionMessageToolCall as TC
    real = TC.model_validate({"id": "r1", "type": "function", "function": {"name": "web_search", "arguments": "{}"},
                              "extra_content": {"google": {"thought_signature": SIG}}})
    check("REAL openai SDK tool_call carries signature through _tool_call_entry",
          A._tool_call_entry(real, "c").get("extra_content", {}).get("google", {}).get("thought_signature") == SIG)
except ImportError:
    print("skip real SDK object test (openai not installed)")

# --- end-to-end: Gemini answers with a tool call -> history entry -> next Gemini step accepted -------
R.reset_state()
sdk_like = NS(id="e1", function=NS(name="web_search", arguments='{"query":"q"}'), extra_content={"google": {"thought_signature": SIG}})
gem = FakeClient([resp(None, tool_calls=[sdk_like]), resp("final")])
m1, _, _ = gem_run(gem, [{"role": "user", "content": "find q"}])
tcs = A._normalize_tool_calls(m1)
conv = [{"role": "user", "content": "find q"}, {"role": "assistant", "content": None, "tool_calls": [A._tool_call_entry(t, "c") for t in tcs]},
        {"role": "tool", "tool_call_id": "e1", "content": "result"}]
m2, _, _ = gem_run(gem, conv)
replayed = next(m for m in gem.calls[1]["messages"] if m.get("tool_calls"))["tool_calls"][0]
check("multi-step: signature from step 1 replayed on step 2", replayed["extra_content"]["google"]["thought_signature"] == SIG and m2.content == "final")

print("FAILS:", FAILS)
sys.exit(1 if FAILS else 0)
