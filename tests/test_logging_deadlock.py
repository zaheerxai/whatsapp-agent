"""Regression: /uploadlog must never wedge the agent (prod 2026-10-17 19:51 UTC).

Original cause: upload_current_log() held mojo_logging._lock (non-reentrant) across the OneDrive HTTP
call; file_ops.upload_to_onedrive() print()s -> stdout tee -> log handler -> _maybe_rotate() -> same
lock -> self-deadlock, and every other thread that logged then blocked behind the handler lock.

Run: PYTHONPATH=. python tests/test_logging_deadlock.py [path/to/mojo_logging.py]
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
import tempfile
import threading
import time

target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mojo_logging.py")
d = tempfile.mkdtemp()
os.environ["MOJO_LOG_DIR"] = d
spec = importlib.util.spec_from_file_location("mojo_logging_under_test", target)
ml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)
ml.setup_logging()

FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    print(("ok   " if cond else "FAIL ") + name, extra)
    FAILS += 0 if cond else 1


class FakeFileOps:
    @staticmethod
    def onedrive_configured():
        return True

    @staticmethod
    def upload_to_onedrive(path, remote_folder=None, remote_name=None):
        time.sleep(0.3)                          # network
        print("[file_ops] uploaded -> x (123 bytes)")  # real file_ops prints here
        logging.getLogger("mojo").info("inside upload")
        return {"size": os.path.getsize(path), "webUrl": "http://x"}


res = {}


def uploader():
    res["r"] = ml.upload_current_log(FakeFileOps)


def chatter():
    for i in range(20):
        logging.getLogger("mojo").info("other thread log %d", i)
        time.sleep(0.05)
    res["chat"] = True


t1, t2 = threading.Thread(target=uploader, daemon=True), threading.Thread(target=chatter, daemon=True)
t1.start(); time.sleep(0.05); t2.start()
t1.join(8); t2.join(8)
check("upload_current_log returns (no self-deadlock)", not t1.is_alive() and "r" in res)
check("other threads keep logging during upload", not t2.is_alive() and res.get("chat"))

probe = {}
def p():
    logging.getLogger("mojo").info("probe"); probe["ok"] = True
tp = threading.Thread(target=p, daemon=True); tp.start(); tp.join(3)
check("logging still works afterwards", probe.get("ok"))
print("FAILS:", FAILS)
os._exit(1 if FAILS else 0)  # daemon threads may be wedged in the failing case
