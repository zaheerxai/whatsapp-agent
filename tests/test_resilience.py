"""Offline tests for resilience.py. Run: PYTHONPATH=. python tests/test_resilience.py"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MOJO_LOG_DIR"] = tempfile.mkdtemp()
os.environ["GUARD_SELF_HEAL"] = "0"
import mojo_logging  # noqa: E402

mojo_logging.setup_logging()
import resilience as R  # noqa: E402

FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    print(("ok   " if cond else "FAIL ") + name, extra)
    FAILS += 0 if cond else 1


def wait(pred, t=3.0):
    end = time.time() + t
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


sent, replies = [], []
R.init(lambda text: sent.append(text))

# 1) success path
t = R.run_admin("/ok x", lambda c: "RESULT", replies.append)
t.join(2)
check("admin command result delivered", replies == ["RESULT"], replies)

# 2) crash -> full traceback to owner + incident report
replies.clear(); sent.clear()
def boom(c): raise ValueError("kaboom")
R.run_admin("/boom", boom, replies.append).join(2)
check("crash reply has type + traceback", replies and "ValueError: kaboom" in replies[0] and "Traceback" in replies[0], replies[:1])
check("crash raises incident to owner", wait(lambda: any("admin command crashed" in s for s in sent)), sent[:1])

# 3) slow then hung: notices + stack of the stuck thread, and the late result still arrives
replies.clear(); sent.clear()
gate = threading.Event()
def slow(c): gate.wait(3); return "LATE RESULT"
R.run_admin("/slow", slow, replies.append, slow_after=0.2, hang_after=0.5)
check("slow notice sent", wait(lambda: any("taking longer" in r for r in replies)), replies)
check("hang report names where it is stuck", wait(lambda: any("looks HUNG" in r and "gate.wait" in r or "looks HUNG" in r and "slow" in r for r in replies)), [r[:80] for r in replies])
gate.set()
check("late result still delivered", wait(lambda: "LATE RESULT" in replies), replies)

# 4) reply failure -> fallback channel
fb = []
def bad(_): raise RuntimeError("send failed")
R.run_admin("/ok", lambda c: "VIA FALLBACK", bad, fallback_reply=fb.append).join(2)
check("fallback reply used when primary send fails", wait(lambda: fb == ["VIA FALLBACK"]), fb)

# 5) incident rate-limit + force
sent.clear(); R._last_sent.clear()
for _ in range(3):
    R.report_incident("dup kind", RuntimeError("same"), stage="s")
wait(lambda: len(sent) >= 1); time.sleep(0.2)
check("identical incidents rate-limited to 1", len(sent) == 1, len(sent))
R.report_incident("dup kind", RuntimeError("same"), stage="s", force=True)
check("force bypasses cooldown", wait(lambda: len(sent) == 2))
check("report carries log tail + hint", "Last log:" in sent[0] and "/tail" in sent[0])

# 6) watchdog flags a stuck handler with its stack
sent.clear(); R._last_sent.clear()
release = threading.Event()
def stuck_worker():
    with R.track("process_message:test", limit_s=0.1):
        release.wait(5)
th = threading.Thread(target=stuck_worker, daemon=True); th.start()
time.sleep(0.3)
R.watchdog_tick({})
check("stuck handler reported to owner", wait(lambda: any("handler stuck" in s and "process_message:test" in s for s in sent)), [s[:60] for s in sent])
check("stack points into the stuck function", wait(lambda: any("release.wait" in s or "stuck_worker" in s for s in sent)))
R.watchdog_tick({})
n = len([s for s in sent if "handler stuck" in s]); release.set(); th.join(1)
check("stuck handler reported only once", n == 1, n)

# 7) undelivered reports are retried
R._undelivered.clear(); R._last_sent.clear()
state = {"up": False}; got = []
def flaky(text):
    if not state["up"]: raise RuntimeError("link down")
    got.append(text)
R.init(flaky)
R.report_incident("net down", RuntimeError("x"), stage="a", force=True)
check("undelivered report queued", wait(lambda: len(R._undelivered) == 1), len(R._undelivered))
state["up"] = True
R.watchdog_tick({})
check("queued report delivered after link returns", any(g.startswith("(delayed)") for g in got), got[:1])

# 8) logging probe + diag + auth
check("logging pipeline probe ok", R.logging_pipeline_ok())
d = R.diag_payload("abc")
check("diag payload has log tail + threads", d["logging_ok"] and d["log_tail"] and "MainThread" in d["threads"])
os.environ["DIAG_TOKEN"] = "s3cret"
check("diag auth accepts/denies", R.diag_authorized("s3cret") and not R.diag_authorized("nope") and not R.diag_authorized(""))
del os.environ["DIAG_TOKEN"]
check("diag disabled without token", not R.diag_authorized("anything"))

print("FAILS:", FAILS)
os._exit(1 if FAILS else 0)
