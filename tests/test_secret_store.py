"""Offline tests: secret_store (Fernet at rest, fail-closed, rotation, tamper/binding) + SQL/accessor wiring.
Run: PYTHONPATH=. python3 tests/test_secret_store.py
"""
from __future__ import annotations

import logging
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import secret_store as S  # noqa: E402
import usage_budget as UB  # noqa: E402
from cryptography.fernet import Fernet  # noqa: E402

FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if cond:
        print(f"ok   {name}")
    else:
        FAILS += 1
        print(f"FAIL {name} {extra}")


class MemBackend:
    def __init__(self):
        self.rows, self.up, self.puts = {}, True, 0

    def available(self):
        return self.up

    def get(self, p):
        if not self.up:
            raise S.BackendUnavailable("down")
        return self.rows.get(p)

    def put(self, p, ct):
        if not self.up:
            raise S.BackendUnavailable("down")
        self.rows[p] = ct
        self.puts += 1

    def delete(self, p):
        self.rows.pop(p, None)


K1, K2, K3 = (Fernet.generate_key().decode() for _ in range(3))
SECRET = {"rt": "SUPER-SECRET-REFRESH-TOKEN", "seed": "abc"}


def setup(key=K1, backend=None):
    S.reset_state()
    b = backend or MemBackend()
    S.set_backend(b)
    if key is None:
        os.environ.pop(S.ENV_KEY, None)
    else:
        os.environ[S.ENV_KEY] = key
    return b


class Cap(logging.Handler):
    def __init__(self):
        super().__init__(); self.lines = []

    def emit(self, r):
        self.lines.append(r.getMessage())


cap = Cap(); logging.getLogger("mojo.secrets").addHandler(cap)

# ------------------------------------------------------------------------------------------ fail closed
b = setup(key=None)
check("no key -> put is fail-closed ('no_key')", S.put("onedrive", SECRET) == "no_key")
check("no key -> NOTHING written to the database", b.rows == {} and b.puts == 0)
check("no key -> warns once with how to fix it", sum("TOKEN_ENCRYPTION_KEY" in l for l in cap.lines) == 1)
S.put("onedrive", SECRET)
check("warning is not repeated", sum("is NOT persisted" in l for l in cap.lines) == 1)
b = setup(key="not-a-valid-fernet-key")
check("invalid key -> treated as no key (fail-closed)", not S.key_configured() and S.put("onedrive", SECRET) == "no_key" and b.rows == {})

# ------------------------------------------------------------------------------------------- encryption
b = setup()
check("put -> 'db'", S.put("onedrive", SECRET) == "db")
ct = b.rows["onedrive"]
check("stored value is NOT plaintext (no token, no JSON keys)", "SUPER-SECRET" not in ct and "refresh" not in ct and "rt" not in ct.split("gAAAA")[0])
check("ciphertext is a Fernet token", ct.startswith("gAAAA"))
check("round trip", S.get("onedrive") == SECRET)
check("two encryptions differ (random IV)", S.encrypt("onedrive", SECRET) != S.encrypt("onedrive", SECRET))

# ------------------------------------------------------------------------- tamper / wrong key / binding
bad = ct[:-6] + ("AAAAAA" if not ct.endswith("AAAAAA") else "BBBBBB")
b.rows["onedrive"] = bad
check("tampered ciphertext -> None (authenticated encryption)", S.get("onedrive") is None)
b.rows["onedrive"] = ct
os.environ[S.ENV_KEY] = K2; S.reset_state()
_before = sum("cannot be decrypted" in l for l in cap.lines)
check("wrong key -> None, no exception", S.get("onedrive") is None)
S.get("onedrive")
check("wrong key warns exactly once", sum("cannot be decrypted" in l for l in cap.lines) - _before == 1)
b = setup()
S.put("onedrive", SECRET)
b.rows["other"] = b.rows["onedrive"]
check("row copied under another provider name is rejected", S.get("other") is None)

# -------------------------------------------------------------------------------------- key rotation
b = setup(key=K1)
S.put("onedrive", SECRET)
old_ct = b.rows["onedrive"]
os.environ[S.ENV_KEY] = f"{K2},{K1}"; S.reset_state()
check("new key first, old second: still readable", S.get("onedrive") == SECRET)
check("record re-encrypted with the primary key on read", b.rows["onedrive"] != old_ct)
os.environ[S.ENV_KEY] = K2; S.reset_state()
check("old key can now be retired", S.get("onedrive") == SECRET)
os.environ[S.ENV_KEY] = f"garbage,{K3}"; S.reset_state()
check("an invalid key in the list is skipped, valid one used", S.key_configured() and S.put("onedrive", SECRET) == "db")

# -------------------------------------------------------------------------------- backend unavailable
b = setup(); b.up = False
check("backend down -> put 'unavailable', no raise", S.put("onedrive", SECRET) == "unavailable")
check("backend down -> get None, no raise", S.get("onedrive") is None)
check("store_ready reflects availability", S.store_ready() is False)
b.up = True
check("store_ready true again", S.store_ready() is True)

# -------------------------------------------------------------- SupabaseBackend against a fake client
class Q:
    def __init__(self, c, name):
        self.c, self.name, self.op = c, name, None

    def select(self, cols): self.op = ("select", cols); return self
    def eq(self, k, v): self.op = (*self.op, k, v); return self
    def limit(self, n): return self
    def upsert(self, row, on_conflict=None): self.op = ("upsert", row, on_conflict); return self
    def delete(self): self.op = ("delete",); return self

    def execute(self):
        self.c.calls.append((self.name, self.op))
        if self.c.err:
            raise self.c.err
        if self.op[0] == "select":
            return types.SimpleNamespace(data=[{"ciphertext": self.c.store[self.op[3]]}] if self.op[3] in self.c.store else [])
        if self.op[0] == "upsert":
            self.c.store[self.op[1]["provider"]] = self.op[1]["ciphertext"]
        if self.op[0] == "delete":
            self.c.store.pop(self.op[2], None)
        return types.SimpleNamespace(data=[])


class FakeSB:
    def __init__(self):
        self.calls, self.store, self.err = [], {}, None

    def table(self, name): return Q(self, name)


sb = FakeSB(); UB.init(sb)
check("usage_budget.db_client() exposes the attached client", UB.db_client() is sb)
S.reset_state(); S.set_backend(S.SupabaseBackend()); os.environ[S.ENV_KEY] = K1
check("Supabase backend put -> 'db'", S.put("onedrive", SECRET) == "db")
tbl, op = sb.calls[-1]
check("writes to the DEDICATED table with upsert on provider", tbl == "oauth_tokens" and op[0] == "upsert" and op[2] == "provider")
check("only ciphertext leaves the process", "SUPER-SECRET" not in str(sb.calls) and sb.store["onedrive"].startswith("gAAAA"))
check("Supabase backend get round-trips", S.get("onedrive") == SECRET)
S.delete("onedrive")
check("delete removes the row", "onedrive" not in sb.store)

sb2 = FakeSB(); sb2.err = RuntimeError("{'code': 'PGRST205', 'message': \"Could not find the table 'public.oauth_tokens' in the schema cache\"}")
UB.init(sb2); S.reset_state(); S.set_backend(S.SupabaseBackend())
check("missing table -> put 'unavailable' (no crash)", S.put("onedrive", SECRET) == "unavailable")
check("missing table -> tells you to run sql/oauth_tokens.sql", any("sql/oauth_tokens.sql" in l for l in cap.lines))
n = len(sb2.calls); S.put("onedrive", SECRET); S.get("onedrive")
check("breaker: no hammering while the backend is down", len(sb2.calls) == n)
UB.init(None)
check("no client attached -> store_ready False", S.SupabaseBackend().available() is False)

# ------------------------------------------------------------------------------------------- SQL file
sql = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sql", "oauth_tokens.sql")).read().lower()
check("SQL: dedicated table with provider pk + ciphertext", "create table if not exists public.oauth_tokens" in sql and "provider   text primary key" in sql and "ciphertext text" in sql)
check("SQL: RLS enabled and anon/authenticated revoked", "enable row level security" in sql and "revoke all on public.oauth_tokens from anon, authenticated" in sql)
check("SQL: no policies granting access", "create policy" not in sql)

print("FAILS:", FAILS)
sys.exit(1 if FAILS else 0)
