"""Offline tests for llm_router.py (fake clients, no network).
Run: python tests/test_llm_router.py   (from repo root: python -m tests.test_llm_router also works)
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm_router as R  # noqa: E402

FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if cond:
        print(f"ok   {name}")
    else:
        FAILS += 1
        print(f"FAIL {name} {extra}")


class HTTPErr(Exception):
    def __init__(self, status, retry_after=None, body=""):
        super().__init__(f"Error code: {status}")
        self.status_code = status
        self.body = body
        self.response = NS(headers={"retry-after": str(retry_after)} if retry_after is not None else {}, text=body)


def resp(text, finish="stop", tokens=500, tool_calls=None):
    return NS(choices=[NS(message=NS(content=text, tool_calls=tool_calls), finish_reason=finish)],
              usage=NS(total_tokens=tokens))


class FakeClient:
    """script: list of results per call; Exception instances are raised."""
    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        r = self.script.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def run(groq, gem, messages=None, tools=None, max_tokens=1024, full=False):
    fn = R.complete_full if full else R.complete
    return fn(
        messages or [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}],
        tools, 0.3, max_tokens,
        groq_client=groq, groq_model="openai/gpt-oss-120b", groq_sibling="openai/gpt-oss-20b",
        gemini_client=gem, gemini_model="gemini-3.6-flash",
        shrink=lambda m, k: m[-k:], flatten=lambda m: [{"role": "user", "content": "FLAT"}],
    )


# 1) 429 + retry-after -> sibling answers, primary cooled for next call (no wasted 429 round-trip)
R.reset_state()
g = FakeClient([HTTPErr(429, retry_after=30), resp("from-20b"), resp("again-20b")])
msg, fin, prov = run(g, FakeClient([]))
check("429 -> sibling serves", msg.content == "from-20b" and prov == "groq" and len(g.calls) == 2)
check("sibling got its own model id", g.calls[1]["model"] == "openai/gpt-oss-20b")
msg, _, _ = run(g, FakeClient([]))
check("primary skipped while cooled", g.calls[2]["model"] == "openai/gpt-oss-20b" and len(g.calls) == 3)

# 2) TPM governor: oversized request goes straight to Gemini, with low reasoning + roomy cap
R.reset_state()
big = [{"role": "system", "content": "x" * 30000}, {"role": "user", "content": "list all"}]
g, gem = FakeClient([]), FakeClient([resp("gemini-ok")])
msg, fin, prov = run(g, gem, messages=big)
check("oversize skips Groq entirely", prov == "gemini" and not g.calls)
check("gemini reasoning_effort=low", gem.calls[0]["extra_body"] == {"reasoning_effort": "low"})
check("gemini max_tokens >= 2048", gem.calls[0]["max_tokens"] >= 2048, gem.calls[0]["max_tokens"])

# 3) finish_reason=length -> continuation stitched
R.reset_state()
g = FakeClient([resp("1. a\n2. b\n", finish="length"), resp("3. c\n4. d", finish="stop")])
msg = run(g, FakeClient([]), full=True)
check("length -> continued", msg.content == "1. a\n2. b\n3. c\n4. d", repr(msg.content))
check("continuation asks to continue", "Continue" in g.calls[1]["messages"][-1]["content"])

# 3b) continuation is bounded
R.reset_state()
g = FakeClient([resp("A", finish="length")] + [resp("B", finish="length")] * 5)
msg = run(g, FakeClient([]), full=True)
check("continuation bounded (1 + 2 calls)", len(g.calls) == 3 and msg.content == "ABB", (len(g.calls), msg.content))

# 3c) tool calls are never 'continued'
R.reset_state()
g = FakeClient([resp("", finish="length", tool_calls=[NS(id="1")])])
msg = run(g, FakeClient([]), full=True)
check("tool_calls untouched", getattr(msg, "tool_calls", None) and len(g.calls) == 1)

# 4) 400 on a synthesis turn -> retried flattened without tools; error body is logged by err_detail
R.reset_state()
e400 = HTTPErr(400, body='{"error":{"code":"tool_use_failed"}}')
g = FakeClient([e400, resp("recovered")])
msg, _, _ = run(g, FakeClient([]))
check("400 -> flatten retry", msg.content == "recovered" and g.calls[1]["messages"][0]["content"] == "FLAT")
check("err_detail includes body", "tool_use_failed" in R.err_detail(e400))

# 5) gpt-oss synthesis turn asks for low reasoning; tool-selection turn keeps tools and default effort
R.reset_state()
g = FakeClient([resp("x"), resp("y")])
run(g, FakeClient([]))
check("synthesis: reasoning_effort low", g.calls[0].get("extra_body") == {"reasoning_effort": "low"})
run(g, FakeClient([]), tools=[{"type": "function", "function": {"name": "t"}}])
check("tool turn: tools kept, no forced effort", "tools" in g.calls[1] and "extra_body" not in g.calls[1])

# 6) Gemini 503 retried once, then served
R.reset_state()
R._TPM_SAFETY = 0.0  # force Gemini path
gem = FakeClient([HTTPErr(503), resp("gem-after-503")])
orig_sleep = R.time.sleep
R.time.sleep = lambda s: None
msg, _, prov = run(FakeClient([]), gem)
R.time.sleep = orig_sleep
R._TPM_SAFETY = 0.92
check("gemini 503 -> served (fails over to next model)", msg.content == "gem-after-503" and len(gem.calls) == 2)

# 7) everything fails -> raises last error (callers fall back to tool OBS)
R.reset_state()
try:
    run(FakeClient([HTTPErr(429, 5)]), FakeClient([HTTPErr(500), HTTPErr(500)]))
    check("raises when all fail", False)
except Exception:
    check("raises when all fail", True)

# ---- Gemini ladder (Task 1: 2026-11-03 12:00 two 503s ended the turn) ----------------------------------
SLEEPS = []
R.time.sleep = lambda s: SLEEPS.append(s)
R._TPM_SAFETY = 0.0  # force the Gemini path for everything below


def grun(gem, fallbacks=("g-b", "g-c"), primary="g-a", **kw):
    return R.complete(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}], None, 0.3, 1024,
        groq_client=None, groq_model=None, groq_sibling=None,
        gemini_client=gem, gemini_model=primary, gemini_fallbacks=list(fallbacks),
        shrink=lambda m, k: m, flatten=lambda m: m, **kw)


class NetErr(Exception):
    pass


NetErr.__name__ = "APIConnectionError"

# 8) 503 on primary -> immediate fail-over to next model (no sleep), primary cooled for the next request
R.reset_state(); SLEEPS.clear()
gem = FakeClient([HTTPErr(503), resp("from-b"), resp("again-b")])
msg, _, prov = grun(gem)
check("503 -> next Gemini model", msg.content == "from-b" and [c["model"] for c in gem.calls] == ["g-a", "g-b"])
check("no sleep on fail-over", SLEEPS == [])
msg, _, _ = grun(gem)
check("failed model skipped on next request", [c["model"] for c in gem.calls][2] == "g-b" and len(gem.calls) == 3)

# 9) every model 503 -> raises; non-last models tried once, last model gets one backoff retry
R.reset_state(); SLEEPS.clear()
gem = FakeClient([HTTPErr(503)] * 4)
try:
    grun(gem); check("all 503 raises", False)
except Exception as e:
    check("all 503 raises", getattr(e, "status_code", None) == 503)
check("1 try per non-last model + 2 on last", [c["model"] for c in gem.calls] == ["g-a", "g-b", "g-c", "g-c"], [c["model"] for c in gem.calls])
check("one jittered backoff <= cap", len(SLEEPS) == 1 and 0 < SLEEPS[0] <= R._BACKOFF_CAP_S, SLEEPS)

# 10) retry-after honoured (capped) on the last model
R.reset_state(); SLEEPS.clear()
gem = FakeClient([resp("ok-after-ra")] if False else [HTTPErr(429, retry_after=3), resp("ok-after-ra")])
msg, _, _ = grun(gem, fallbacks=())
check("single-model ladder retries after 429", msg.content == "ok-after-ra" and len(gem.calls) == 2)
check("retry-after respected", SLEEPS and SLEEPS[0] >= 3, SLEEPS)

# 11) 404 (retired model id) -> next model, and cooled for an hour
R.reset_state()
gem = FakeClient([HTTPErr(404), resp("from-b")])
msg, _, _ = grun(gem)
check("404 -> next model", msg.content == "from-b")
check("404 cools model ~1h", R._cooled("g-a") > 3000, R._cooled("g-a"))

# 12) 400 on reasoning_effort -> same model retried without extra_body
R.reset_state()
gem = FakeClient([HTTPErr(400, body="reasoning_effort unsupported"), resp("no-reasoning")])
msg, _, _ = grun(gem)
check("400 -> same model without extra_body",
      msg.content == "no-reasoning" and gem.calls[1]["model"] == "g-a" and "extra_body" not in gem.calls[1]
      and "extra_body" in gem.calls[0])

# 13) connection/timeout errors (no HTTP status) are treated as transient
R.reset_state()
gem = FakeClient([NetErr("Connection error."), resp("from-b")])
msg, _, _ = grun(gem)
check("connection error -> next model", msg.content == "from-b")

# 14) every model cooling down -> still probes one instead of failing untried
R.reset_state()
for m in ("g-a", "g-b", "g-c"):
    R._cool(m, 30)
gem = FakeClient([resp("probed")])
msg, _, _ = grun(gem)
check("all cooled -> probe one", msg.content == "probed" and len(gem.calls) == 1)

# 15) deadline bounds the walk
R.reset_state()
old = R._GEMINI_DEADLINE_S
R._GEMINI_DEADLINE_S = 0.0
gem = FakeClient([HTTPErr(503), resp("never")])
try:
    grun(gem); check("deadline stops ladder", False)
except Exception:
    check("deadline stops ladder", len(gem.calls) == 1, len(gem.calls))
R._GEMINI_DEADLINE_S = old

# 16) SDK retries are disabled for Gemini calls (no nested retry storm)
R.reset_state()
class OptClient(FakeClient):
    def __init__(self, script):
        super().__init__(script); self.opts = None
    def with_options(self, **o):
        self.opts = o; return self
gem = OptClient([resp("ok")])
grun(gem)
check("with_options(max_retries=0)", gem.opts == {"max_retries": 0}, gem.opts)

# 17) default ladder comes from env / built-in list, primary first, de-duplicated
R.reset_state()
os.environ["GEMINI_FALLBACK_MODELS"] = "g-b, g-a ,g-c"
check("env ladder dedup/order", R._gemini_models("g-a", None) == ["g-a", "g-b", "g-c"], R._gemini_models("g-a", None))
del os.environ["GEMINI_FALLBACK_MODELS"]
check("built-in ladder", R._gemini_models("gemini-3.6-flash", None)[0] == "gemini-3.6-flash"
      and "gemini-3.8-flash" in R._gemini_models("gemini-3.6-flash", None))
R._TPM_SAFETY = 0.92
R.time.sleep = orig_sleep

print("FAILS:", FAILS)
sys.exit(1 if FAILS else 0)
