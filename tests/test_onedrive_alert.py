"""Offline tests: owner alert when the OneDrive token dies (and recovers), plus resilience.notify_owner.
Run: PYTHONPATH=. python3 tests/test_onedrive_alert.py
"""
from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import onedrive_auth as A  # noqa: E402
import secret_store as SS  # noqa: E402
import usage_budget as UB  # noqa: E402
import resilience as RS  # noqa: E402
from cryptography.fernet import Fernet  # noqa: E402

FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if cond:
        print(f"ok   {name}")
    else:
        FAILS += 1
        print(f"FAIL {name} {extra}")


def wait_for(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


class Resp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body, self.headers, self.text = status, body, {}, ""

    def json(self):
        return self._body


def tok(at="AT", rt=None):
    b = {"access_token": at, "expires_in": 3600}
    if rt:
        b["refresh_token"] = rt
    return Resp(200, b)


def bad():
    return Resp(400, {"error": "invalid_grant", "error_description": "AADSTS700082: The refresh token has expired due to inactivity."})


class MemBackend:
    def __init__(self):
        self.rows = {}

    def available(self): return True
    def get(self, p): return self.rows.get(p)
    def put(self, p, ct): self.rows[p] = ct
    def delete(self, p): self.rows.pop(p, None)


script: list = []
A.requests.post = lambda url, data=None, timeout=None: script.pop(0)
A.time.sleep = lambda s: None
ENV_RT = "ENV-RT-SECRET-VALUE"
calls: list = []


def handler(kind, text):
    calls.append((kind, text))


def fresh(*resps, with_handler=True, keep_kv=False):
    A.reset_state()
    if not keep_kv:
        UB._mem.clear()
    SS.reset_state(); SS.set_backend(MemBackend())
    os.environ["TOKEN_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
    os.environ["ONEDRIVE_CLIENT_ID"] = "cid"
    os.environ["ONEDRIVE_REFRESH_TOKEN"] = ENV_RT
    A._T0 = time.time() - 3600
    A._ALERTS_ON = True
    script.clear(); script.extend(resps)
    calls.clear()
    if with_handler:
        A.set_alert_handler(handler)


def fail_once():
    script.append(bad())
    try:
        A.get_access_token(force=True)
    except A.OneDriveAuthError:
        pass


# ------------------------------------------------------------------------------------- first alert
fresh(bad())
t0 = time.time()
try:
    A.get_access_token()
except A.OneDriveAuthError as e:
    check("permanent failure still raises as before", e.permanent)
check("alert delivered", wait_for(lambda: len(calls) == 1), calls)
kind, text = calls[0]
check("kind is 'dead'", kind == "dead")
check("tells the owner exactly how to fix it", "get_onedrive_token.py" in text and "ONEDRIVE_REFRESH_TOKEN" in text and "Render" in text)
check("says what stops working", "knowledge-base sync" in text and "log uploads" in text)
check("includes Microsoft's reason", "invalid_grant" in text)
check("promises reminders + recovery notice", "remind you every 12h" in text and "tell you when it is back" in text)
check("no token value in the alert", ENV_RT not in text and "RT" not in text.replace("ONEDRIVE_REFRESH_TOKEN", ""))
check("status shows an open alert", A.status()["alert_open"] is True)

# ------------------------------------------------------------------------- dedupe inside the window
for _ in range(5):
    try:
        A.get_access_token()
    except A.OneDriveAuthError:
        pass
check("calls inside the cool-off -> no extra alerts", len(calls) == 1)
A._dead_until = 0.0
fail_once()
time.sleep(0.15)
check("cool-off over but still inside the repeat window -> still ONE alert", len(calls) == 1, len(calls))

# ----------------------------------------------------------------------------------- repeat reminder
A._alert_state["last"] = time.time() - 13 * 3600
A._alert_state["first"] = time.time() - 26 * 3600
A._dead_until = 0.0
fail_once()
check("after 12h still down -> reminder", wait_for(lambda: len(calls) == 2), len(calls))
check("reminder wording + downtime", "still disconnected" in calls[1][1] and "down for 26h" in calls[1][1], calls[1][1][:120])

# --------------------------------------------------------------- restart must not re-alert in window
fresh()
fail_once()
check("(setup) alert sent", wait_for(lambda: len(calls) == 1))
persisted = UB.kv_get("oauth:onedrive:alert")
check("outage marker persisted", bool(persisted))
A.reset_state(); A.set_alert_handler(handler)          # simulated restart: memory gone, kv remains
SS.reset_state(); SS.set_backend(MemBackend()); A._T0 = time.time() - 3600
script.append(bad())
try:
    A.get_access_token()
except A.OneDriveAuthError:
    pass
time.sleep(0.15)
check("restart inside the window -> NO duplicate alert", len(calls) == 1, len(calls))

# ------------------------------------------------------------------------ never blocks the caller
slow_done = threading.Event()


def slow_handler(kind, text):
    time.sleep(1.5); slow_done.set()


fresh(bad(), with_handler=False)
A.set_alert_handler(slow_handler)
t0 = time.time()
try:
    A.get_access_token()
except A.OneDriveAuthError:
    pass
check("a slow WhatsApp send does not delay the failing call", time.time() - t0 < 0.5, time.time() - t0)
check("...and it still completes in the background", slow_done.wait(3))

# ------------------------------------------------------------------------------ handler problems
def boom(kind, text):
    raise RuntimeError("whatsapp down")


fresh(bad(), with_handler=False)
A.set_alert_handler(boom)
try:
    A.get_access_token(); check("raises", False)
except A.OneDriveAuthError as e:
    check("failing handler never changes auth behaviour", e.permanent)
time.sleep(0.15)

fresh(with_handler=False)
fail_once()
check("no handler yet -> not marked as alerted", A.status()["alert_open"] is False and UB.kv_get("oauth:onedrive:alert") in (None, ""))
A.set_alert_handler(handler); A._dead_until = 0.0
fail_once()
check("handler registered later -> the next failure alerts", wait_for(lambda: len(calls) == 1), calls)

fresh(bad()); A._ALERTS_ON = False
fail_once(); time.sleep(0.15)
check("ONEDRIVE_ALERTS=0 disables alerts", calls == [])

# --------------------------------------------------------------------------------------- recovery
fresh()
fail_once()
wait_for(lambda: len(calls) == 1)
A._dead_until = 0.0
script.append(tok("ATok", "RTnew"))
A.get_access_token(force=True)
check("recovery announced", wait_for(lambda: len(calls) == 2 and calls[1][0] == "recovered"), calls)
check("recovery text", "OneDrive reconnected" in calls[1][1] and "was down" in calls[1][1])
check("marker cleared", A.status()["alert_open"] is False and UB.kv_get("oauth:onedrive:alert") in (None, ""))
script.append(tok("AT2", "RT2")); A.get_access_token(force=True); time.sleep(0.15)
check("later successes stay silent", len(calls) == 2)

fresh()
fail_once()
wait_for(lambda: len(calls) == 1)
A.reset_state(); A.set_alert_handler(handler); A._T0 = time.time() - 3600       # restart while down
SS.reset_state(); SS.set_backend(MemBackend())
script.append(tok("ATrestart", "RTr"))
A.get_access_token()
check("recovery after a RESTART is still announced", wait_for(lambda: len(calls) == 2 and calls[1][0] == "recovered"), calls)

# ------------------------------------------------------------------------- situations that never alert
fresh(Resp(503, {}), Resp(503, {}), Resp(503, {}))
try:
    A.get_access_token()
except A.OneDriveAuthError as e:
    check("outage raises non-permanent", not e.permanent)
time.sleep(0.15)
check("transient outage -> no alert", calls == [])

fresh(bad())
class DownBackend(MemBackend):
    def available(self): return False
SS.set_backend(DownBackend()); A._T0 = time.time()
try:
    A.get_access_token()
except A.OneDriveAuthError:
    pass
time.sleep(0.15)
check("startup grace (token store not attached) -> no alert", calls == [])

fresh()
del os.environ["ONEDRIVE_REFRESH_TOKEN"]
try:
    A.get_access_token()
except A.OneDriveAuthError:
    pass
time.sleep(0.15)
check("OneDrive not configured at all -> no alert", calls == [])
os.environ["ONEDRIVE_REFRESH_TOKEN"] = ENV_RT

# =================================================================== resilience.notify_owner
delivered: list = []
gate = threading.Event()


def slow_send(text):
    gate.wait(2); delivered.append(text)


RS._last_sent.clear(); RS._undelivered.clear()
RS.init(slow_send)
t0 = time.time()
ok = RS.notify_owner("hello owner")
check("notify_owner returns immediately (off-thread)", ok and time.time() - t0 < 0.3 and delivered == [])
gate.set()
check("message delivered VERBATIM (no incident header)", wait_for(lambda: delivered == ["hello owner"]), delivered)

delivered.clear(); RS._last_sent.clear()
RS.init(lambda t: delivered.append(t))
check("per-key cooldown: first send", RS.notify_owner("a", key="k", cooldown_s=60) is True)
check("per-key cooldown: second suppressed", RS.notify_owner("b", key="k", cooldown_s=60) is False)
check("different key unaffected", RS.notify_owner("c", key="other", cooldown_s=60) is True)
check("no cooldown requested -> never suppressed", RS.notify_owner("d", key="k") and RS.notify_owner("e", key="k"))
wait_for(lambda: len(delivered) >= 4)
check("only the allowed messages were sent", sorted(delivered)[:3] == ["a", "c", "d"] or {"a", "c", "d", "e"} <= set(delivered), delivered)


def failing_send(text):
    raise RuntimeError("WhatsApp disconnected")


RS._undelivered.clear(); RS._last_sent.clear()
RS.init(failing_send)
check("send failure never raises into the caller", RS.notify_owner("important") is True)
check("undelivered message parked for watchdog retry", wait_for(lambda: "important" in list(RS._undelivered)), list(RS._undelivered))
RS._notify = None
check("no notifier configured -> still never raises", RS.notify_owner("x") in (True, False))

print("FAILS:", FAILS)
sys.exit(1 if FAILS else 0)
