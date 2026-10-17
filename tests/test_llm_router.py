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
check("gemini 503 retry", msg.content == "gem-after-503" and len(gem.calls) == 2)

# 7) everything fails -> raises last error (callers fall back to tool OBS)
R.reset_state()
try:
    run(FakeClient([HTTPErr(429, 5)]), FakeClient([HTTPErr(500), HTTPErr(500)]))
    check("raises when all fail", False)
except Exception:
    check("raises when all fail", True)

print("FAILS:", FAILS)
sys.exit(1 if FAILS else 0)
