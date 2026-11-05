"""
resilience.py — the agent may break; the ADMIN must never be left blind.

Incident 2026-10-17 19:51 UTC: a self-deadlock in the log layer froze the WhatsApp callback thread.
The bot went silent, /uploadlog never answered, and the only way to learn anything was the Render
console. This module makes sure that class of failure is (a) detected, (b) reported to the owner with
detail, (c) never able to stop admin commands, and (d) recoverable without touching Render.

Pieces
  report_incident()   detailed, rate-limited owner report (WhatsApp) + retry queue + local ring buffer
  track()             register a running handler so the watchdog can see it is stuck and where
  run_admin()         runs an admin command on its OWN thread: slow notice, hang report with stacks,
                      full traceback on crash, reply fallbacks. Admin commands never touch the LLM/RAG path
  start_watchdog()    logging-pipeline probe (self-heal: dump stacks + exit so Render restarts us),
                      stuck-handler detection, delayed re-delivery of undelivered incident reports
  diag_payload()      JSON for GET /diag (token protected) — works when WhatsApp is completely wedged
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import sys
import threading
import time
import traceback
from collections import deque
from typing import Any, Callable, Dict, List, Optional

import mojo_logging

log = logging.getLogger("mojo.guard")

START_TS = time.time()
STUCK_AFTER_S = int(os.getenv("GUARD_STUCK_AFTER_S") or "120")
INCIDENT_COOLDOWN_S = int(os.getenv("GUARD_INCIDENT_COOLDOWN_S") or "300")
SELF_HEAL_EXIT = (os.getenv("GUARD_SELF_HEAL") or "1").strip() not in ("0", "false", "no")

_notify: Optional[Callable[[str], None]] = None
_incidents: "deque[Dict[str, Any]]" = deque(maxlen=60)
_last_sent: Dict[str, float] = {}
_undelivered: "deque[str]" = deque(maxlen=8)
_tasks: Dict[int, Dict[str, Any]] = {}
_tasks_lock = threading.Lock()
_task_seq = 0
_watchdog_started = False


def init(notify_fn: Callable[[str], None]) -> None:
    """notify_fn(text): deliver a message to the owner. May block/raise — we always call it off-thread."""
    global _notify
    _notify = notify_fn


# ------------------------------------------------------------------ incident reporting
def _sig(kind: str, exc: Optional[BaseException], stage: str) -> str:
    return hashlib.sha1(f"{kind}|{type(exc).__name__ if exc else ''}|{stage}".encode()).hexdigest()[:12]


def _fmt_incident(kind: str, exc: Optional[BaseException], stage: str, extra: str, stacks: str) -> str:
    up = int(time.time() - START_TS)
    lines = [
        f"🚨 *Mojo incident* — {kind}",
        f"When: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC | uptime {up // 3600}h{(up % 3600) // 60}m",
    ]
    if stage:
        lines.append(f"Where: {stage}")
    if exc is not None:
        lines.append(f"Error: {type(exc).__name__}: {str(exc)[:300]}")
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-4:])
        lines.append("Trace:\n" + tb[-1200:])
    if extra:
        lines.append(extra[:600])
    if stacks:
        lines.append("Stuck at:\n" + stacks[-1500:])
    lines.append("Last log:\n" + mojo_logging.tail_log(8, 900))
    lines.append("Live view (if DIAG_TOKEN set): /diag • WhatsApp: /tail 80, /threads, /incidents")
    return "\n".join(lines)[:3400]


def _deliver(text: str) -> bool:
    if _notify is None:
        return False
    box: Dict[str, Any] = {}

    def run():
        try:
            _notify(text)
            box["ok"] = True
        except Exception as e:  # noqa: BLE001
            box["err"] = e

    t = threading.Thread(target=run, daemon=True, name="guard-notify")
    t.start()
    t.join(25)  # never let a wedged send block the caller
    if not box.get("ok"):
        log.warning("incident delivery failed: %s", box.get("err") or "timeout")
        return False
    return True


def report_incident(kind: str, exc: Optional[BaseException] = None, *, stage: str = "",
                    extra: str = "", stacks: str = "", force: bool = False) -> None:
    """Record + (rate-limited) send a detailed report to the owner. Never raises, never blocks the caller."""
    try:
        text = _fmt_incident(kind, exc, stage, extra, stacks)
        sig = _sig(kind, exc, stage)
        _incidents.append({"ts": time.time(), "kind": kind, "stage": stage, "sig": sig,
                           "error": f"{type(exc).__name__}: {exc}" if exc else "", "text": text})
        log.error("INCIDENT %s stage=%s err=%s", kind, stage, exc)
        now = time.time()
        if not force and now - _last_sent.get(sig, 0) < INCIDENT_COOLDOWN_S:
            return
        _last_sent[sig] = now

        def send():
            if not _deliver(text):
                _undelivered.append(text)  # watchdog retries once the link is back

        threading.Thread(target=send, daemon=True, name="guard-report").start()
    except Exception:  # noqa: BLE001 — the reporter must never become the problem
        pass


def notify_owner(text: str, *, key: str = "", cooldown_s: float = 0.0) -> bool:
    """Send a plain, already-formatted message to the owner (no incident header / log tail).
    Same delivery guarantees as report_incident: sent off-thread with a timeout, and parked in the
    undelivered queue (retried by the watchdog) if WhatsApp is down. Optional per-`key` cooldown.
    Never raises, never blocks the caller. Returns True when a send was dispatched."""
    try:
        now = time.time()
        if key and cooldown_s > 0:
            if now - _last_sent.get("notify:" + key, 0) < cooldown_s:
                return False
            _last_sent["notify:" + key] = now

        def send():
            if not _deliver(text):
                _undelivered.append(text)

        threading.Thread(target=send, daemon=True, name="guard-notify-owner").start()
        return True
    except Exception:  # noqa: BLE001 — a notifier must never become the problem
        return False


# ------------------------------------------------------------------ handler tracking
@contextlib.contextmanager
def track(name: str, limit_s: Optional[int] = None):
    global _task_seq
    with _tasks_lock:
        _task_seq += 1
        tid = _task_seq
        _tasks[tid] = {"name": name, "ident": threading.get_ident(), "start": time.time(),
                       "limit": limit_s or STUCK_AFTER_S, "reported": False}
    try:
        yield tid
    finally:
        with _tasks_lock:
            _tasks.pop(tid, None)


def running_tasks() -> List[Dict[str, Any]]:
    now = time.time()
    with _tasks_lock:
        return [{"name": v["name"], "age_s": int(now - v["start"]), "ident": v["ident"]} for v in _tasks.values()]


# ------------------------------------------------------------------ admin runner
def _chunks(text: str, n: int = 3300) -> List[str]:
    text = text or ""
    return [text[i:i + n] for i in range(0, len(text), n)] or [""]


def run_admin(cmd_text: str, handler: Callable[[str], Optional[str]], reply: Callable[[str], None],
              fallback_reply: Optional[Callable[[str], None]] = None,
              slow_after: float = 30.0, hang_after: float = 120.0) -> threading.Thread:
    """Run an admin command on a dedicated thread and ALWAYS tell the owner what happened:
    result, slow notice, hang report with stacks, or full traceback."""
    done = threading.Event()
    box: Dict[str, Any] = {}
    cmd = (cmd_text or "").strip().split()[0] if (cmd_text or "").strip() else "?"

    def send(text: str) -> None:
        for part in _chunks(text):
            try:
                reply(part)
            except Exception as e1:  # noqa: BLE001
                log.warning("admin reply failed (%s) — trying fallback", e1)
                try:
                    if fallback_reply is None:
                        raise e1
                    fallback_reply(part)
                except Exception as e2:  # noqa: BLE001
                    log.error("admin reply fallback failed: %s", e2)
                    _undelivered.append(part)

    def work():
        with track(f"admin:{cmd}", limit_s=int(hang_after)):
            box["ident"] = threading.get_ident()
            try:
                out = handler(cmd_text)
                log.info("ADMIN_COMMAND %r ok", cmd_text[:80])
                if out is not None:
                    send(out)
            except BaseException as e:  # noqa: BLE001
                tb = traceback.format_exc()
                log.error("ADMIN_COMMAND %r crashed: %s\n%s", cmd_text[:80], e, tb)
                send(f"❌ {cmd} crashed: {type(e).__name__}: {str(e)[:300]}\n{tb[-1400:]}")
                report_incident("admin command crashed", e, stage=cmd, force=True)
            finally:
                done.set()

    def supervise():
        if not done.wait(slow_after):
            send(f"⏳ {cmd} is taking longer than usual ({int(slow_after)}s+). Still working — I'll reply when it finishes.")
            if not done.wait(max(1.0, hang_after - slow_after)):
                stacks = mojo_logging.thread_stacks({box["ident"]}) if box.get("ident") else ""
                send(f"⚠️ {cmd} looks HUNG after {int(hang_after)}s. Where it is stuck:\n{stacks[-1800:]}")
                report_incident("admin command hung", stage=cmd, stacks=stacks, force=True)

    t = threading.Thread(target=work, daemon=True, name=f"admin-{cmd}")
    t.start()
    threading.Thread(target=supervise, daemon=True, name=f"admin-sup-{cmd}").start()
    return t


# ------------------------------------------------------------------ watchdog
def _raw_stderr(msg: str) -> None:
    try:
        os.write(2, (msg + "\n").encode("utf-8", "replace"))  # bypass sys.stderr tee and every lock
    except Exception:  # noqa: BLE001
        pass


def logging_pipeline_ok(timeout: float = 5.0) -> bool:
    """Probe: can a *different* thread log, and is the listener draining? (Both failed in the incident.)"""
    ok = threading.Event()

    def p():
        log.info("guard probe")
        ok.set()

    threading.Thread(target=p, daemon=True, name="guard-probe").start()
    if not ok.wait(timeout):
        return False
    lst = getattr(mojo_logging, "_listener", None)
    th = getattr(lst, "_thread", None)
    return bool(th is None or th.is_alive())


def watchdog_tick(strikes: Dict[str, int]) -> None:
    # 1) logging pipeline
    if logging_pipeline_ok():
        strikes["log"] = 0
    else:
        strikes["log"] = strikes.get("log", 0) + 1
        _raw_stderr(f"[GUARD] logging pipeline wedged (strike {strikes['log']})\n" + mojo_logging.thread_stacks())
        if strikes["log"] >= 2 and SELF_HEAL_EXIT:
            _raw_stderr("[GUARD] self-heal: exiting so the platform restarts the service")
            os._exit(70)
    # 2) stuck handlers
    now = time.time()
    with _tasks_lock:
        stuck = [(tid, dict(v)) for tid, v in _tasks.items() if now - v["start"] > v["limit"] and not v["reported"]]
        for tid, _ in stuck:
            _tasks[tid]["reported"] = True
    for _tid, v in stuck:
        stacks = mojo_logging.thread_stacks({v["ident"]})
        log.error("STUCK handler %s age=%ds\n%s", v["name"], int(now - v["start"]), stacks)
        report_incident("handler stuck", stage=v["name"], stacks=stacks,
                        extra=f"Running for {int(now - v['start'])}s (limit {v['limit']}s).")
    # 3) re-deliver reports that could not be sent earlier
    while _undelivered and _notify is not None:
        text = _undelivered[0]
        if _deliver("(delayed) " + text):
            _undelivered.popleft()
        else:
            break


def start_watchdog(interval: float = 10.0) -> None:
    global _watchdog_started
    if _watchdog_started:
        return
    _watchdog_started = True
    strikes: Dict[str, int] = {}

    def loop():
        time.sleep(30)  # let startup settle
        while True:
            try:
                watchdog_tick(strikes)
            except Exception as e:  # noqa: BLE001
                _raw_stderr(f"[GUARD] watchdog error: {e}")
            time.sleep(interval)

    threading.Thread(target=loop, daemon=True, name="guard-watchdog").start()


# ------------------------------------------------------------------ diagnostics (HTTP /diag)
def diag_payload(sha: str = "") -> Dict[str, Any]:
    return {
        "sha": sha,
        "uptime_s": int(time.time() - START_TS),
        "running_handlers": running_tasks(),
        "incidents": [{"t": time.strftime("%H:%M:%S", time.gmtime(i["ts"])), "kind": i["kind"],
                       "stage": i["stage"], "error": i["error"]} for i in list(_incidents)[-15:]],
        "undelivered_reports": len(_undelivered),
        "logging_ok": logging_pipeline_ok(2.0),
        "log_tail": mojo_logging.tail_log(200, 20000).splitlines(),
        "threads": mojo_logging.thread_stacks(max_chars=20000),
    }


def diag_authorized(provided: str) -> bool:
    import hmac
    token = os.getenv("DIAG_TOKEN") or ""
    return bool(token) and hmac.compare_digest(token.encode(), (provided or "").encode())
