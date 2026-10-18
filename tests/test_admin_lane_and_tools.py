"""Offline tests: owner fast lane, forget_memory, GitHub API-URL normalisation.
Run: PYTHONPATH=. python tests/test_admin_lane_and_tools.py"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
import time
import types
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MOJO_LOG_DIR"] = tempfile.mkdtemp()
import mojo_logging  # noqa: E402

mojo_logging.setup_logging()
import agent_tools as t  # noqa: E402
import resilience  # noqa: E402

FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    print(("ok   " if cond else "FAIL ") + name, extra)
    FAILS += 0 if cond else 1


# ---------------------------------------------------------------- GitHub URL normalisation
check("api users/repos url -> real user", t._github_user_from_url(
    "https://api.github.com/users/zaheerxai/repos?sort=pushed&direction=desc&per_page=12") == "zaheerxai")
check("api repos url -> owner/repo", t._github_repo_from_url("https://api.github.com/repos/mattpocock/skills") == ("mattpocock", "skills"))
check("plain profile still works", t._github_user_from_url("https://github.com/zaheerxai") == "zaheerxai")
check("plain repo still works", t._github_repo_from_url("https://github.com/mattpocock/skills") == ("mattpocock", "skills"))
check("gist host never parsed as github.com", t._github_user_from_url("https://gist.github.com/abc/123") is None)


# ---------------------------------------------------------------- forget_memory
class Q:
    def __init__(self, db): self.db, self.op, self.f = db, "select", []
    def select(self, *_a, **_k): self.op = "select"; return self
    def delete(self): self.op = "delete"; return self
    def eq(self, k, v): self.f.append(lambda r: r.get(k) == v); return self
    def in_(self, k, vs): self.f.append(lambda r: r.get(k) in vs); return self
    def order(self, *_a, **_k): return self
    def execute(self):
        rows = self.db.rows
        sel = [r for r in rows if all(f(r) for f in self.f)]
        if self.op == "delete":
            for r in sel: rows.remove(r)
        return NS(data=[dict(r) for r in sel])


class DB:
    def __init__(self): self.rows = [
        {"id": 1, "chat_id": "A", "note": "Work Seeker X is a European job portal"},
        {"id": 2, "chat_id": "A", "note": "Work Seeker X details: 30+ countries"},
        {"id": 3, "chat_id": "A", "note": "Reply in Roman Urdu"},
        {"id": 9, "chat_id": "B", "note": "Other chat's note about Work Seeker X"},
    ]
    def table(self, _n): return Q(self)


db = DB(); t._supabase = db
out = t._tool_get_memory({}, {"chat_id": "A"})
check("get_memory exposes ids", "[#1]" in out and "[#3]" in out and "[#9]" not in out, out[:80])
res = t._tool_forget_memory({"ids": [1, 2, "#9"]}, {"chat_id": "A"})
left = [r["id"] for r in db.rows]
check("forget deletes only this chat's matching ids", left == [3, 9], left)
check("forget result names what was removed", "Forgot 2" in res, res)
check("other chat's note untouched", 9 in left)
check("empty ids -> guidance", "get_memory" in t._tool_forget_memory({"ids": []}, {"chat_id": "A"}))
check("forget_memory registered + schema present",
      "forget_memory" in t.TOOL_EXECUTORS and any(s["function"]["name"] == "forget_memory" for s in t.TOOL_SCHEMAS))

# ---------------------------------------------------------------- owner fast lane (extracted from whatsapp_agent.py)
src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "whatsapp_agent.py"), encoding="utf-8").read()
tree = ast.parse(src)
want = {"_jid_user", "_jid_server", "_fast_admin_lane"}
code = "\n".join(ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want)
calls = []
fake_admin = types.SimpleNamespace(
    is_admin_message=lambda num, grp: num == "923161545875" and not grp,
    handle_admin_command=lambda text: f"handled {text}",
)
fake_res = types.SimpleNamespace(run_admin=lambda text, h, reply, fallback_reply=None: (calls.append((text, h(text))), reply("x")))
ns = {"admin_commands": fake_admin, "resilience": fake_res, "_should_skip_duplicate": lambda m: False}
exec(code, ns)
lane = ns["_fast_admin_lane"]


def jid(user, server): return NS(User=user, Server=server)
def msg(text, sender=("923161545875", "s.whatsapp.net"), alt=None, chat_server="s.whatsapp.net", from_me=False):
    return NS(
        Message=NS(conversation=text, extendedTextMessage=None),
        Info=NS(IsFromMe=from_me, MessageSource=NS(
            Chat=jid("923161545875", chat_server), Sender=jid(*sender), SenderAlt=alt)),
    )


client = NS(reply_message=lambda *a, **k: None, send_message=lambda *a, **k: None)
check("owner /command consumed by fast lane", lane(client, msg("/uploadlog")) is True and calls[-1][0] == "/uploadlog")
calls.clear()
check("LID sender resolved via SenderAlt", lane(client, msg("/help", sender=("27118708768825", "lid"), alt=jid("923161545875", "s.whatsapp.net"))) is True and calls)
calls.clear()
check("non-owner slash text falls through", lane(client, msg("/uploadlog", sender=("923000000000", "s.whatsapp.net"))) is False and not calls)
check("plain text falls through", lane(client, msg("hello")) is False)
check("/uploadimg left to media pipeline", lane(client, msg("/uploadimg logo")) is False)
check("group chat never admin", lane(client, msg("/help", chat_server="g.us")) is False)
check("own outbound message ignored", lane(client, msg("/help", from_me=True)) is False)

print("FAILS:", FAILS)
os._exit(1 if FAILS else 0)
