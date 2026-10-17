"""Offline tests for knowledge_rag.py: fake Supabase + deterministic fake embeddings (no network).
Run: PYTHONPATH=. python tests/test_knowledge_rag.py
"""
from __future__ import annotations

import importlib
import math
import os
import re
import sys
import time
import uuid
import zlib
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["EMBEDDING_MODEL"] = "text-embedding-004"  # retired name must be auto-mapped
import knowledge_rag as kr  # noqa: E402

FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if cond:
        print(f"ok   {name}")
    else:
        FAILS += 1
        print(f"FAIL {name} {extra}")


# ------------------------------------------------------------------ fake embeddings: hashed bag-of-words
def _vec(text):
    v = [0.0] * kr.EMBEDDING_DIM
    for w in re.findall(r"\w+", text.lower()):
        v[zlib.crc32(w.encode()) % kr.EMBEDDING_DIM] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


EMBED_CALLS = []


def fake_embed_texts(texts, *, is_query=False, provider=None, model=None):
    EMBED_CALLS.append(len(texts))
    return [_vec(t) for t in texts]


kr.embed_texts = fake_embed_texts
kr.embed_query = lambda q: _vec(q)


# ------------------------------------------------------------------ fake supabase
class Q:
    def __init__(self, db, name):
        self.db, self.name = db, name
        self.op, self.payload, self.filters, self._limit, self._count = "select", None, [], None, False

    def select(self, cols="*", count=None, **_):
        self.op, self._count = "select", bool(count)
        return self

    def insert(self, payload):
        self.op, self.payload = "insert", payload
        return self

    def update(self, payload):
        self.op, self.payload = "update", payload
        return self

    def delete(self):
        self.op = "delete"
        return self

    def _f(self, fn):
        self.filters.append(fn)
        return self

    def eq(self, k, v):
        return self._f(lambda r: r.get(k) == v)

    def neq(self, k, v):
        return self._f(lambda r: r.get(k) != v)

    def like(self, k, pat):
        rx = re.compile("^" + re.escape(pat).replace("%", ".*") + "$")
        return self._f(lambda r: bool(rx.match(str(r.get(k) or ""))))

    def ilike(self, k, pat):
        rx = re.compile("^" + re.escape(pat).replace("%", ".*") + "$", re.I)
        return self._f(lambda r: bool(rx.match(str(r.get(k) or ""))))

    def limit(self, n):
        self._limit = n
        return self

    def execute(self):
        if self.db.fail_table == self.name and self.op in self.db.fail_ops:
            raise RuntimeError("boom")
        rows = self.db.t[self.name]
        if self.op == "insert":
            items = self.payload if isinstance(self.payload, list) else [self.payload]
            out = []
            for it in items:
                it = dict(it)
                it.setdefault("id", str(uuid.uuid4()))
                rows.append(it)
                out.append(it)
            return NS(data=out, count=None)
        sel = [r for r in rows if all(f(r) for f in self.filters)]
        if self.op == "update":
            for r in sel:
                r.update(self.payload)
            return NS(data=sel, count=None)
        if self.op == "delete":
            for r in sel:
                rows.remove(r)
                if self.name == "documents":  # cascade
                    self.db.t["chunks"][:] = [c for c in self.db.t["chunks"] if c["document_id"] != r["id"]]
            return NS(data=sel, count=None)
        data = sel[: self._limit] if self._limit else sel
        return NS(data=[dict(r) for r in data], count=len(sel) if self._count else None)


class FakeDB:
    def __init__(self):
        self.t = {"documents": [], "chunks": []}
        self.fail_table, self.fail_ops = None, ()
        self.rpc_missing = set()
        self.rpc_calls = []

    def table(self, name):
        return Q(self, name)

    def rpc(self, name, params):
        self.rpc_calls.append(name)
        db = self

        class R:
            def execute(_self):
                if name in db.rpc_missing:
                    raise RuntimeError(f"function {name} does not exist")
                return NS(data=db._rpc(name, params))
        return R()

    def _rpc(self, name, p):
        if name == "replace_document_chunks":
            did = p["p_document_id"]
            self.t["chunks"][:] = [c for c in self.t["chunks"] if c["document_id"] != did]
            for c in p["p_chunks"]:
                self.t["chunks"].append(dict(c, document_id=did, id=str(uuid.uuid4())))
            return len(p["p_chunks"])
        cid = p.get("filter_chat_id")
        scoped = [c for c in self.t["chunks"] if c.get("chat_id") in (None, "") or c.get("chat_id") == cid]
        qv = p["query_embedding"]
        out = []
        for c in scoped:
            sim = sum(a * b for a, b in zip(qv, c["embedding"]))
            qw = set(re.findall(r"\w+", (p.get("query_text") or "").lower()))
            lex = bool(qw & set(re.findall(r"\w+", c["content"].lower()))) if p.get("query_text") else False
            out.append(dict(c, similarity=sim, lex_hit=lex, score=sim))
        out.sort(key=lambda r: -r["score"])
        return out[: p["match_count"]]


def fresh():
    db = FakeDB()
    kr.init(db)
    kr.reset_schema_probe()
    EMBED_CALLS.clear()
    return db


# ================================================================== tests
# 0) retired model env var is remapped; vectors normalised + truncated; short vectors rejected
check("retired embedding model remapped", kr.DEFAULT_EMBEDDING_MODEL == "gemini-embedding-001", kr.DEFAULT_EMBEDDING_MODEL)
v = kr._fit_dim([3.0, 4.0] + [0.0] * 2000)
check("fit_dim truncates + L2-normalises", len(v) == kr.EMBEDDING_DIM and abs(math.sqrt(sum(x * x for x in v)) - 1) < 1e-9)
try:
    kr._fit_dim([0.1] * 10)
    check("short vector rejected (no zero-padding)", False)
except RuntimeError:
    check("short vector rejected (no zero-padding)", True)

# 0b) Gemini request shape + retry on 429
posts = []


def fake_post(url, headers=None, json=None, timeout=None):
    posts.append((url, headers, json))
    if len(posts) == 1:
        return NS(status_code=429, headers={"retry-after": "0"}, text="slow down", json=lambda: {})
    n = len(json["requests"])
    return NS(status_code=200, headers={}, text="", json=lambda: {"embeddings": [{"values": [0.5] * 3072} for _ in range(n)]})


os.environ["GEMINI_API_KEY"] = "k"
orig_post, orig_embed = kr.requests.post, kr.embed_texts
kr.requests.post = fake_post
real_sleep, kr.time.sleep = kr.time.sleep, lambda s: None
out = kr._gemini_embed(["a"] * 40, "gemini-embedding-001")
kr.time.sleep = real_sleep
kr.requests.post = orig_post
check("batch embed: 40 texts -> 2 requests (+1 retry)", len(posts) == 3 and len(out) == 40, len(posts))
check("key in header, not URL", "key=" not in posts[0][0] and posts[0][1].get("x-goog-api-key") == "k")
check("outputDimensionality + taskType sent", posts[1][2]["requests"][0]["outputDimensionality"] == kr.EMBEDDING_DIM
      and posts[1][2]["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT")
check("returned vectors normalised", abs(sum(x * x for x in out[0]) - 1) < 1e-9)

# 1) upsert: new -> unchanged -> changed -> model change re-embeds
db = fresh()
doc = "Mojo AI Agency builds WhatsApp automation agents for clinics and shops. " * 3
r = kr.upsert_document(source_type="agency", source_id="a1", text=doc, title="About")
check("upsert new ok", r["status"] == "ok" and r["chunks"] >= 1 and len(db.t["chunks"]) == r["chunks"])
n_embeds = len(EMBED_CALLS)
r = kr.upsert_document(source_type="agency", source_id="a1", text=doc, title="About")
check("same content -> unchanged, no embed", r["status"] == "unchanged" and len(EMBED_CALLS) == n_embeds)
r = kr.upsert_document(source_type="agency", source_id="a1", text=doc + " New pricing section starts at 50 dollars.", title="About")
check("changed content -> re-embed, v2", r["status"] == "ok" and r["source_version"] == 2)
r = kr.upsert_document(source_type="agency", source_id="a1", text=doc + " New pricing section starts at 50 dollars.", title="About", embedding_model="other-model")
check("embedding model change -> re-embed", r["status"] == "ok")
check("atomic RPC used", "replace_document_chunks" in db.rpc_calls)

# 1b) old schema (RPC missing) -> fallback still works
db = fresh(); db.rpc_missing.add("replace_document_chunks")
r = kr.upsert_document(source_type="agency", source_id="a2", text=doc, title="About")
check("fallback insert when RPC missing", r["status"] == "ok" and len(db.t["chunks"]) == r["chunks"])

# 1c) chunk write failure never leaves a 'indexed' doc without chunks
db = fresh(); db.rpc_missing.add("replace_document_chunks"); db.fail_table, db.fail_ops = "chunks", ("insert",)
r = kr.upsert_document(source_type="agency", source_id="a3", text=doc, title="About")
check("failed chunk write -> error + doc rolled back", r["status"] == "error" and not db.t["documents"])
db.fail_table = None
r = kr.upsert_document(source_type="agency", source_id="a3", text=doc, title="About")
check("retry after failure ingests (not 'unchanged')", r["status"] == "ok")

# 2) retrieval: relevance gate, scoping, merge
db = fresh()
kr.upsert_document(source_type="agency", source_id="g", text="Mojo AI Agency pricing: WhatsApp agent setup costs 300 dollars. " * 4, title="Pricing")
kr.upsert_document(source_type="chat_note", source_id="chat_note:A", chat_id="A@g.us", text="Zaheer resume: skills Python n8n Supabase, internship at Abdanix.", title="Chat notes")
kr.upsert_document(source_type="chat_note", source_id="chat_note:B", chat_id="B@g.us", text="SECRET other chat plan: launch tomorrow with the blue banner.", title="Chat notes")
res = kr.search("whatsapp agent pricing", chat_id="A@g.us")
check("relevant hit returned", "Pricing" in res and "300 dollars" in res, res[:200])
res = kr.search("zaheer skills python", chat_id="A@g.us")
check("own chat note retrievable", "Abdanix" in res)
res = kr.search("blue banner launch plan", chat_id="A@g.us")
check("another chat's note NEVER returned", "SECRET" not in res and "blue banner" not in res, res[:200])
res = kr.search("blue banner launch plan", chat_id=None)
check("no chat scope -> global only", "SECRET" not in res)
res = kr.search("quantum chromodynamics lagrangian", chat_id="A@g.us", top_k=3)
check("irrelevant query -> clean miss (no filler)", res.startswith("No relevant knowledge found"), res[:120])

# 2b) adjacent chunks are stitched without duplicating the overlap
rows = [
    {"document_id": "d", "ordinal": 0, "content": "AAAA " * 60 + "OVERLAPOVERLAPOVERLAP", "_score": 0.5, "source_type": "document", "title": "T"},
    {"document_id": "d", "ordinal": 1, "content": "OVERLAPOVERLAPOVERLAP" + " BBBB" * 60, "_score": 0.4, "source_type": "document", "title": "T"},
    {"document_id": "d", "ordinal": 5, "content": "FAR", "_score": 0.3, "source_type": "document", "title": "T"},
]
m = kr._merge_adjacent(rows)
check("adjacent merged, far chunk kept separate", len(m) == 2 and m[0]["content"].count("OVERLAPOVERLAPOVERLAP") == 1, [len(x["content"]) for x in m])

# 3) chat notes sync + removal + debounce
db = fresh()
NOTES = {"C@g.us": ["Always reply in Roman Urdu to Zaheer", "Zaheer prefers short answers"]}
kr.init(db, get_notes_fn=lambda cid: NOTES.get(cid, []), all_notes_fn=lambda: dict(NOTES))
kr.reset_schema_probe()
r = kr.sync_chat_notes("C@g.us")
check("notes bundled into one chat-scoped doc", r["status"] == "ok" and db.t["documents"][0]["chat_id"] == "C@g.us")
NOTES["C@g.us"] = []
kr.sync_chat_notes("C@g.us")
check("notes removed -> doc removed", not db.t["documents"])
NOTES["C@g.us"] = ["Standing rule: invoices go out on the 1st of every month."]
before = len(EMBED_CALLS)
for _ in range(5):
    kr.sync_chat_notes_async("C@g.us", delay=0.15)
time.sleep(0.6)
check("burst of note writes -> one debounced sync", len(EMBED_CALLS) - before == 1 and len(db.t["documents"]) == 1, len(EMBED_CALLS) - before)

# 4) chat documents: save last doc, scopes, owner rule, vision fallback, forget
db = fresh()
kr.init(db, vision_extract_fn=lambda b64, mime, name: "SCANNED TEXT " + ("invoice total 4500 rupees " * 5))
kr.reset_schema_probe()
kr._doc_cache.clear()
kr._doc_cache["X|a.pdf"] = {"chat_id": "X", "name": "a.pdf", "text": "", "b64": "AAAA", "mime": "application/pdf", "ts": time.time()}
r = kr.save_to_knowledge(chat_id="X", sender_id="u", scope="chat")
check("scanned doc read via vision LLM then saved", r["status"] == "ok" and db.t["documents"][0]["chat_id"] == "X")
r = kr.save_to_knowledge(chat_id="X", scope="global", is_owner=False)
check("global save refused for non-owner", r["status"] == "error")
r = kr.save_to_knowledge(chat_id="X", scope="global", is_owner=True)
check("global save allowed for owner", r["status"] == "ok" and any(d["chat_id"] is None for d in db.t["documents"]))
r = kr.save_to_knowledge(chat_id="NOPE")
check("no recent doc -> clear error", r["status"] == "error" and "send the file" in r["error"])
r = kr.save_to_knowledge(chat_id="X", title="Plan", text="Quarterly plan: ship RAG v2, then LangSmith traces." * 2)
check("explicit text saved", r["status"] == "ok")
d = kr.delete_documents("a.pdf", chat_id="Y")
check("forget cannot touch another chat's docs", d["deleted"] == [])
d = kr.delete_documents("a.pdf", chat_id="X", is_owner=False)
check("forget own-chat doc (global survives for non-owner)", d["deleted"] == ["a.pdf"] and any(x["chat_id"] is None for x in db.t["documents"]))

# 5) OneDrive: ingest + prune deleted files, never prune on empty listing
db = fresh()
FILES = [{"path": "Documents/aimojo/docs/a.txt"}, {"path": "Documents/aimojo/docs/b.txt"}]
TEXT = {"Documents/aimojo/docs/a.txt": "Alpha document about onboarding clinics. " * 3,
        "Documents/aimojo/docs/b.txt": "Beta document about pricing tiers. " * 3}


class FO:
    onedrive_configured = staticmethod(lambda: True)
    ensure_onedrive_folder = staticmethod(lambda f: None)
    list_onedrive_recursive = staticmethod(lambda folder, extensions=None, **k: list(FILES))

    @staticmethod
    def download_from_onedrive(remote, local):
        open(local, "w").write(TEXT[remote])


kr.init(db, file_ops_module=FO)
kr.reset_schema_probe()
r = kr.ingest_onedrive_knowledge_tree()
check("onedrive ingest 2 files", r["ingested"] == 2 and r["pruned"] == 0)
FILES[:] = [FILES[0]]
r = kr.ingest_onedrive_knowledge_tree()
check("deleted OneDrive file pruned from KB", r["pruned"] == 1 and len(db.t["documents"]) == 1)
FILES[:] = []
r = kr.ingest_onedrive_knowledge_tree()
check("empty listing never wipes KB (API-failure guard)", r["pruned"] == 0 and len(db.t["documents"]) == 1)

# 6) schema probe retries after TTL (no restart needed after running the SQL)
db = FakeDB(); kr.init(db); kr.reset_schema_probe()
db.fail_table, db.fail_ops = "documents", ("select",)
check("probe fails while tables missing", kr.schema_ready() is False)
db.fail_table = None
check("negative cache honoured inside TTL", kr.schema_ready() is False)
kr._schema_checked_at -= 120
check("re-probe after TTL succeeds", kr.schema_ready() is True)

# 7) status
db = fresh()
kr.upsert_document(source_type="agency", source_id="s", text=doc, title="S")
st = kr.kb_status()
check("kb_status counts", st["documents"] == 1 and st["by_source_type"].get("agency") == 1 and st["stale_embedding_chunks"] == 0, st)

print("FAILS:", FAILS)
sys.exit(1 if FAILS else 0)
