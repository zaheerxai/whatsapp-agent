"""
knowledge_rag.py — RAG foundation for Mojo WhatsApp agent.

Responsibilities:
  - Recursive / semantic-friendly chunking
  - Multilingual embeddings (Gemini text-embedding-004 default, OpenAI optional)
  - Idempotent document + chunk upsert into Supabase pgvector tables
  - Hybrid retrieval (RPC hybrid_search → match_chunks → local keyword fallback)
  - OneDrive Documents/aimojo ingestion (agency + docs trees)
  - Bootstrap from local business_info.txt

Env:
  GEMINI_API_KEY / OPENAI_API_KEY
  EMBEDDING_PROVIDER = gemini | openai   (default gemini)
  EMBEDDING_MODEL    = text-embedding-004 | text-embedding-3-small | ...
  KNOWLEDGE_ONEDRIVE_ROOT = Documents/aimojo   (default)
"""

from __future__ import annotations

import base64
import hashlib
import logging
import math
import os
import re
import tempfile
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

log = logging.getLogger("mojo.rag")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_PROVIDER = (os.getenv("EMBEDDING_PROVIDER") or "gemini").strip().lower()
# Retired by Google (404 since early 2026) -> transparently mapped so a stale env var can't kill RAG.
_RETIRED_GEMINI_MODELS = {
    "text-embedding-004", "embedding-001", "embedding-gecko-001",
    "gemini-embedding-exp-03-07", "text-embedding-005",
}
_EMBED_MODEL_RAW = (
    os.getenv("EMBEDDING_MODEL")
    or ("gemini-embedding-001" if DEFAULT_EMBEDDING_PROVIDER == "gemini" else "text-embedding-3-small")
).strip()
if DEFAULT_EMBEDDING_PROVIDER == "gemini" and _EMBED_MODEL_RAW.lower() in _RETIRED_GEMINI_MODELS:
    log.warning("EMBEDDING_MODEL=%s is retired; using gemini-embedding-001", _EMBED_MODEL_RAW)
    _EMBED_MODEL_RAW = "gemini-embedding-001"
DEFAULT_EMBEDDING_MODEL = _EMBED_MODEL_RAW
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM") or "768")
KNOWLEDGE_ONEDRIVE_ROOT = (
    os.getenv("KNOWLEDGE_ONEDRIVE_ROOT") or "Documents/aimojo"
).strip().strip("/")

# Chunk targets (characters ≈ tokens for mixed English / Roman Urdu)
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE") or "1200")
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP") or "150")
MAX_CHUNKS_PER_DOC = int(os.getenv("RAG_MAX_CHUNKS_PER_DOC") or "400")

# Retrieval
# A chunk is kept when it matched full-text (lexical hit) OR its cosine similarity >= MIN_SIM.
# Everything else is top-k filler and is dropped so the LLM says "not found" instead of improvising.
# Tune from the RAG_HIT log lines (they print similarity for every candidate).
MIN_SIM = float(os.getenv("RAG_MIN_SIM") or "0.50")
RERANK_ENABLED = (os.getenv("RAG_RERANK") or "0").strip() in ("1", "true", "yes")
AUTO_SYNC = (os.getenv("RAG_AUTO_SYNC") or "1").strip() not in ("0", "false", "no")
SYNC_INTERVAL_MIN = int(os.getenv("RAG_SYNC_MIN") or "180")  # 0 = boot sync only
_GEMINI_BATCH = 32  # batchEmbedContents allows up to 100 per request

# Supported ingest extensions
TEXT_EXTS = {".txt", ".md", ".csv", ".json", ".log", ".html", ".htm"}
DOC_EXTS = {".docx", ".xlsx", ".pptx", ".pdf"}
SUPPORTED_EXTS = TEXT_EXTS | DOC_EXTS

_supabase = None
_file_ops = None
_extract_document_text = None  # injected from whatsapp_agent to reuse XLSX smart extract
_get_notes = None        # chat_id -> List[str]   (permanent notes for one chat)
_all_notes = None        # () -> Dict[chat_id, List[str]]
_vision_extract = None   # (b64, mime, name) -> str   LLM text extraction for scanned PDFs / images
_rerank = None           # (query, passages) -> List[int]   optional LLM reranker (best first)

# Soft in-process caches (query embedding + schema readiness)
_query_embed_cache: Dict[str, Tuple[float, List[float]]] = {}
_QUERY_EMBED_TTL = 45.0
_schema_ok: Optional[bool] = None
_schema_checked_at: float = 0.0
_SCHEMA_RETRY_S = 60.0  # a failed probe is re-tried after this (no restart needed after running the SQL)


def init(
    supabase_client,
    file_ops_module=None,
    extract_document_text_fn=None,
    *,
    get_notes_fn=None,
    all_notes_fn=None,
    vision_extract_fn=None,
    rerank_fn=None,
) -> None:
    global _supabase, _file_ops, _extract_document_text, _schema_ok
    global _get_notes, _all_notes, _vision_extract, _rerank
    _supabase = supabase_client
    _file_ops = file_ops_module
    _extract_document_text = extract_document_text_fn
    _get_notes, _all_notes = get_notes_fn, all_notes_fn
    _vision_extract, _rerank = vision_extract_fn, rerank_fn
    _schema_ok = None


# ---------------------------------------------------------------------------
# Hashing / normalise
# ---------------------------------------------------------------------------

def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def normalise_text(text: str) -> str:
    if not text:
        return ""
    # UTF-8 already; collapse exotic whitespace, keep newlines for structure
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


# ---------------------------------------------------------------------------
# Chunking (recursive separators — low latency, no LLM)
# ---------------------------------------------------------------------------

_SEPARATORS = ("\n## ", "\n# ", "\n### ", "\n\n", "\n", ". ", " ", "")


def recursive_chunk(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split text into overlapping chunks preferring structure boundaries."""
    text = normalise_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    def _split(src: str, seps: Sequence[str]) -> List[str]:
        if len(src) <= chunk_size:
            return [src] if src.strip() else []
        if not seps:
            # hard cut with overlap
            out = []
            step = max(1, chunk_size - overlap)
            for i in range(0, len(src), step):
                piece = src[i : i + chunk_size].strip()
                if piece:
                    out.append(piece)
                if i + chunk_size >= len(src):
                    break
            return out

        sep = seps[0]
        parts = src.split(sep) if sep else [src]
        chunks: List[str] = []
        buf = ""
        for i, part in enumerate(parts):
            piece = part if i == 0 or not sep else sep + part
            if not piece:
                continue
            if len(buf) + len(piece) <= chunk_size:
                buf += piece
            else:
                if buf.strip():
                    chunks.extend(_split(buf, seps[1:]))
                if len(piece) > chunk_size:
                    chunks.extend(_split(piece, seps[1:]))
                    buf = ""
                else:
                    buf = piece
        if buf.strip():
            chunks.extend(_split(buf, seps[1:]))
        return chunks

    raw = _split(text, _SEPARATORS)
    # Apply overlap between adjacent chunks when they share no natural overlap
    if overlap <= 0 or len(raw) <= 1:
        return raw[:MAX_CHUNKS_PER_DOC]

    out: List[str] = []
    for i, c in enumerate(raw):
        if i == 0:
            out.append(c)
            continue
        prev_tail = out[-1][-overlap:] if len(out[-1]) > overlap else out[-1]
        if not c.startswith(prev_tail[: min(40, len(prev_tail))]):
            merged = (prev_tail + c).strip()
            if len(merged) <= chunk_size + overlap:
                out.append(merged)
            else:
                out.append(c)
        else:
            out.append(c)
    return out[:MAX_CHUNKS_PER_DOC]


def chunk_with_meta(
    text: str,
    title: str = "",
) -> List[Dict[str, Any]]:
    """Return list of {ordinal, content, section, token_count}."""
    chunks = recursive_chunk(text)
    results = []
    for i, c in enumerate(chunks):
        section = ""
        m = re.search(r"^(?:#{1,3}\s+|.+\n[=-]{3,})", c)
        if m:
            section = m.group(0).strip().split("\n")[0][:120]
        results.append(
            {
                "ordinal": i,
                "content": c,
                "section": section or (title[:80] if i == 0 else ""),
                "token_count": max(1, len(c) // 4),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def _l2(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _fit_dim(values: List[float]) -> List[float]:
    """Matryoshka models: a truncated prefix is valid but MUST be re-normalised (Google docs:
    only the native 3072-dim vector is pre-normalised). Never zero-pad: a short vector means the
    wrong model/dimension was configured."""
    if len(values) < EMBEDDING_DIM:
        raise RuntimeError(f"embedding has {len(values)} dims < EMBEDDING_DIM={EMBEDDING_DIM}")
    return _l2(list(values[:EMBEDDING_DIM]))


def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], what: str) -> Dict[str, Any]:
    """POST with bounded retry on 429/5xx (honours Retry-After). Raises RuntimeError on failure."""
    last = ""
    for attempt in range(4):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
        except requests.RequestException as e:
            last = f"{what} network: {e}"
            time.sleep(min(2 ** attempt, 8))
            continue
        if r.status_code < 400:
            return r.json()
        last = f"{what} {r.status_code}: {r.text[:300]}"
        if r.status_code in (429, 500, 502, 503, 504):
            try:
                ra = float(r.headers.get("retry-after") or 0)
            except ValueError:
                ra = 0.0
            time.sleep(min(ra if ra > 0 else 2 ** attempt, 20))
            continue
        break  # 4xx other than 429: retrying won't help (bad key, retired model, bad request)
    raise RuntimeError(last)


def _gemini_embed(texts: List[str], model: str, is_query: bool = False) -> List[List[float]]:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set (required for default embeddings)")
    base = "https://generativelanguage.googleapis.com/v1beta"
    task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}  # key out of URLs/logs
    out: List[List[float]] = []
    for i in range(0, len(texts), _GEMINI_BATCH):
        batch = texts[i : i + _GEMINI_BATCH]
        reqs = [
            {
                "model": f"models/{model}",
                "content": {"parts": [{"text": t[:8000]}]},
                "taskType": task,
                "outputDimensionality": EMBEDDING_DIM,
            }
            for t in batch
        ]
        body = _post_json(f"{base}/models/{model}:batchEmbedContents", headers, {"requests": reqs}, "Gemini embed")
        embs = body.get("embeddings") or []
        if len(embs) != len(batch):
            raise RuntimeError(f"Gemini embed returned {len(embs)} vectors for {len(batch)} inputs")
        for e in embs:
            vals = e.get("values")
            if not vals:
                raise RuntimeError(f"Gemini embed empty vector: {str(e)[:200]}")
            out.append(_fit_dim(vals))
    return out


def _openai_embed(texts: List[str], model: str) -> List[List[float]]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    payload: Dict[str, Any] = {"model": model, "input": [t[:8000] for t in texts]}
    if model.startswith("text-embedding-3"):
        payload["dimensions"] = EMBEDDING_DIM
    body = _post_json(
        "https://api.openai.com/v1/embeddings",
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        payload,
        "OpenAI embed",
    )
    data = sorted(body.get("data") or [], key=lambda x: x.get("index", 0))
    return [_fit_dim(item.get("embedding") or []) for item in data]


def embed_texts(
    texts: List[str],
    *,
    is_query: bool = False,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> List[List[float]]:
    if not texts:
        return []
    provider = (provider or DEFAULT_EMBEDDING_PROVIDER).lower()
    model = model or DEFAULT_EMBEDDING_MODEL
    if provider == "openai":
        prepared = texts
        if "e5" in model.lower():
            prefix = "query: " if is_query else "passage: "
            prepared = [prefix + t for t in texts]
        return _openai_embed(prepared, model)
    return _gemini_embed(texts, model, is_query=is_query)


def embed_query(query: str) -> List[float]:
    q = (query or "").strip()
    if not q:
        return [0.0] * EMBEDDING_DIM
    now = time.time()
    cached = _query_embed_cache.get(q)
    if cached and now - cached[0] < _QUERY_EMBED_TTL:
        return cached[1]
    vec = embed_texts([q], is_query=True)[0]
    _query_embed_cache[q] = (now, vec)
    # bound cache size
    if len(_query_embed_cache) > 256:
        oldest = sorted(_query_embed_cache.items(), key=lambda x: x[1][0])[:64]
        for k, _ in oldest:
            _query_embed_cache.pop(k, None)
    return vec


# ---------------------------------------------------------------------------
# Schema probe
# ---------------------------------------------------------------------------

def schema_ready() -> bool:
    """True if documents + chunks tables are reachable. A failed probe is retried after
    _SCHEMA_RETRY_S so running the SQL doesn't require a restart."""
    global _schema_ok, _schema_checked_at
    now = time.time()
    if _schema_ok is True:
        return True
    if _schema_ok is False and now - _schema_checked_at < _SCHEMA_RETRY_S:
        return False
    if _supabase is None:
        _schema_ok, _schema_checked_at = False, now
        return False
    try:
        _supabase.table("documents").select("id").limit(1).execute()
        _supabase.table("chunks").select("id").limit(1).execute()
        _schema_ok = True
    except Exception as e:
        log.warning("schema not ready: %s", e)
        _schema_ok = False
    _schema_checked_at = now
    return _schema_ok


def reset_schema_probe() -> None:
    global _schema_ok, _schema_checked_at
    _schema_ok, _schema_checked_at = None, 0.0


# ---------------------------------------------------------------------------
# Upsert document + chunks
# ---------------------------------------------------------------------------

def upsert_document(
    *,
    source_type: str,
    source_id: str,
    text: str,
    title: str = "",
    chat_id: Optional[str] = None,
    sender_id: Optional[str] = None,
    page_or_sheet: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    embedding_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Idempotent ingest. Skips re-embedding only when content, scope AND embedding model are all
    unchanged (a model switch puts vectors in a different space, so it must re-embed).
    Chunk swap is atomic via replace_document_chunks(); falls back to delete+insert on old schemas.
    Returns {status, document_id, chunks, ...}.
    """
    if _supabase is None:
        return {"status": "error", "error": "supabase not initialised"}
    if not schema_ready():
        return {"status": "error", "error": "RAG tables missing — run sql/rag_schema.sql (+ rag_schema_v2.sql) in Supabase"}

    text = normalise_text(text)
    if not text or len(text) < 20:
        return {"status": "skipped", "reason": "text too short"}

    ch = content_hash(text)
    model = embedding_model or DEFAULT_EMBEDDING_MODEL
    meta = dict(metadata or {})
    meta["embedding_model"] = model

    existing = (
        _supabase.table("documents")
        .select("id, content_hash, source_version, chat_id, sender_id, metadata")
        .eq("source_type", source_type)
        .eq("source_id", source_id)
        .limit(1)
        .execute()
    )
    rows = existing.data or []
    if rows:
        r0 = rows[0]
        same = (
            r0.get("content_hash") == ch
            and (r0.get("chat_id") or None) == (chat_id or None)
            and ((r0.get("metadata") or {}).get("embedding_model") == model)
        )
        if same:
            return {"status": "unchanged", "document_id": r0["id"], "chunks": 0, "source_id": source_id}

    chunk_rows = chunk_with_meta(text, title=title or source_id)
    if not chunk_rows:
        return {"status": "skipped", "reason": "no chunks"}

    # Embed first: if this raises, nothing in the DB has changed.
    contents = [c["content"] for c in chunk_rows]
    vectors = embed_texts(contents, is_query=False, model=model)
    if len(vectors) != len(contents):
        return {"status": "error", "error": "embedding count mismatch"}

    version = 1
    new_doc = not rows
    if rows:
        doc_id = rows[0]["id"]
        version = int(rows[0].get("source_version") or 1) + 1
        _supabase.table("documents").update(
            {
                "content_hash": ch, "source_version": version, "title": title or source_id,
                "chat_id": chat_id, "sender_id": sender_id, "metadata": meta,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("id", doc_id).execute()
    else:
        ins = (
            _supabase.table("documents")
            .insert(
                {
                    "source_type": source_type, "source_id": source_id, "chat_id": chat_id,
                    "sender_id": sender_id, "title": title or source_id, "content_hash": ch,
                    "source_version": 1, "metadata": meta,
                }
            )
            .execute()
        )
        doc_id = (ins.data or [{}])[0].get("id")
        if not doc_id:
            return {"status": "error", "error": "document insert returned no id"}

    payload = [
        {
            "ordinal": c["ordinal"],
            "content": c["content"],
            "token_count": c["token_count"],
            "embedding": vec,
            "embedding_model": model,
            "source_type": source_type,
            "chat_id": chat_id,
            "sender_id": sender_id,
            "title": title or source_id,
            "section": c.get("section") or "",
            "page_or_sheet": page_or_sheet or "",
            "content_hash": content_hash(c["content"]),
            "metadata": meta,
        }
        for c, vec in zip(chunk_rows, vectors)
    ]
    try:
        try:
            _supabase.rpc("replace_document_chunks", {"p_document_id": doc_id, "p_chunks": payload}).execute()
        except Exception as rpc_err:
            log.warning("replace_document_chunks RPC unavailable (%s) — non-atomic fallback", str(rpc_err)[:160])
            _supabase.table("chunks").delete().eq("document_id", doc_id).execute()
            for i in range(0, len(payload), 50):
                batch = [dict(p, document_id=doc_id) for p in payload[i : i + 50]]
                _supabase.table("chunks").insert(batch).execute()
    except Exception as e:
        # Never leave a doc whose hash says "indexed" but has no chunks (would read as unchanged forever).
        try:
            if new_doc:
                _supabase.table("documents").delete().eq("id", doc_id).execute()
            else:
                _supabase.table("documents").update({"content_hash": "failed-reindex"}).eq("id", doc_id).execute()
        except Exception:
            pass
        return {"status": "error", "error": f"chunk write failed: {e}"}

    log.info("RAG_UPSERT type=%s id=%s chunks=%d v=%d model=%s", source_type, source_id[:80], len(payload), version, model)
    return {
        "status": "ok", "document_id": doc_id, "chunks": len(payload), "source_id": source_id,
        "source_version": version, "content_hash": ch,
    }


# ---------------------------------------------------------------------------
# File → text helpers
# ---------------------------------------------------------------------------

def _ext(path: str) -> str:
    return Path(path).suffix.lower()


def extract_text_from_local(path: str, max_chars: int = 200_000) -> str:
    """Reuse whatsapp_agent extractors when available; plain fallback otherwise."""
    ext = _ext(path)
    mime = {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".csv": "text/csv",
        ".json": "application/json",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".pdf": "application/pdf",
    }.get(ext, "application/octet-stream")

    if _extract_document_text is not None:
        try:
            text = _extract_document_text(path, mime, ext, max_chars=max_chars)
            if text:
                return text
            # PDF returns "" to signal vision path — try basic pdfminer-less read skip
            if ext == ".pdf":
                return _extract_pdf_basic(path, max_chars)
        except Exception as e:
            log.warning("extract via wa failed %s: %s", path, e)

    if ext in TEXT_EXTS:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()[:max_chars]

    if ext == ".docx":
        from docx import Document

        doc = Document(path)
        parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n".join(parts)[:max_chars]

    if ext == ".xlsx":
        # Prefer smart extract if available through injected fn; else minimal
        try:
            import whatsapp_agent as wa

            return wa._extract_xlsx_smart(path, max_chars=max_chars)
        except Exception:
            from openpyxl import load_workbook

            wb = load_workbook(path, read_only=True, data_only=True)
            parts = []
            for sheet in wb.worksheets:
                parts.append(f"## Sheet: {sheet.title}")
                for i, row in enumerate(sheet.iter_rows(values_only=True)):
                    if i > 200:
                        break
                    vals = [str(v) for v in row if v is not None]
                    if vals:
                        parts.append("\t".join(vals))
            wb.close()
            return "\n".join(parts)[:max_chars]

    if ext == ".pptx":
        from pptx import Presentation

        prs = Presentation(path)
        parts = []
        for i, slide in enumerate(prs.slides, 1):
            parts.append(f"## Slide {i}")
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text and shape.text.strip():
                    parts.append(shape.text.strip())
        return "\n".join(parts)[:max_chars]

    if ext == ".pdf":
        return _extract_pdf_basic(path, max_chars)

    raise ValueError(f"Unsupported file type: {ext}")


def _extract_pdf_basic(path: str, max_chars: int) -> str:
    """Best-effort PDF text without heavy deps. Returns empty if binary-only."""
    try:
        # Prefer pypdf if installed; otherwise skip
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(path)
        parts = []
        for page in reader.pages[:100]:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
        return "\n\n".join(parts)[:max_chars]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# OneDrive knowledge root helpers
# ---------------------------------------------------------------------------

def knowledge_root() -> str:
    return KNOWLEDGE_ONEDRIVE_ROOT


def ensure_onedrive_knowledge_tree() -> Dict[str, Any]:
    """Create Documents/aimojo/{agency,docs,chats} if missing."""
    if not _file_ops or not _file_ops.onedrive_configured():
        return {"ok": False, "error": "OneDrive not configured"}
    root = knowledge_root()
    created = []
    for sub in ("", "agency", "docs", "chats"):
        folder = f"{root}/{sub}".rstrip("/") if sub else root
        try:
            _file_ops.ensure_onedrive_folder(folder)
            created.append(folder)
        except Exception as e:
            return {"ok": False, "error": str(e), "created": created}
    return {"ok": True, "folders": created, "root": root}


def list_onedrive_knowledge_files(subfolder: str = "") -> List[Dict[str, Any]]:
    """Recursive file list under Documents/aimojo[/subfolder]."""
    if not _file_ops or not _file_ops.onedrive_configured():
        return []
    root = knowledge_root()
    folder = f"{root}/{subfolder}".rstrip("/") if subfolder else root
    return _file_ops.list_onedrive_recursive(folder, extensions=SUPPORTED_EXTS)


def ingest_local_file(
    path: str,
    *,
    source_type: str = "document",
    source_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    sender_id: Optional[str] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return {"status": "error", "error": f"file not found: {path}"}
    text = extract_text_from_local(path)
    if not text.strip():
        return {"status": "skipped", "reason": "empty extract", "path": path}
    sid = source_id or path
    return upsert_document(
        source_type=source_type,
        source_id=sid,
        text=text,
        title=title or Path(path).name,
        chat_id=chat_id,
        sender_id=sender_id,
        metadata={"local_path": path, "ext": _ext(path)},
    )


def ingest_onedrive_path(
    remote_path: str,
    *,
    source_type: str = "onedrive",
    chat_id: Optional[str] = None,
    sender_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not _file_ops or not _file_ops.onedrive_configured():
        return {"status": "error", "error": "OneDrive not configured"}
    remote_path = remote_path.lstrip("/")
    local = os.path.join(tempfile.gettempdir(), "kb_" + Path(remote_path).name)
    try:
        _file_ops.download_from_onedrive(remote_path, local)
        text = extract_text_from_local(local)
        if not text.strip():
            return {"status": "skipped", "reason": "empty extract", "remote": remote_path}
        # Map agency subtree → source_type agency
        root = knowledge_root()
        st = source_type
        if f"{root}/agency/" in remote_path.replace("\\", "/") or remote_path.startswith(
            f"{root}/agency"
        ):
            st = "agency"
        elif f"{root}/docs/" in remote_path.replace("\\", "/"):
            st = "document"
        return upsert_document(
            source_type=st,
            source_id=f"onedrive:{remote_path}",
            text=text,
            title=Path(remote_path).name,
            chat_id=chat_id,
            sender_id=sender_id,
            metadata={"onedrive_path": remote_path},
        )
    except Exception as e:
        return {"status": "error", "error": str(e), "remote": remote_path}
    finally:
        try:
            if os.path.isfile(local):
                os.remove(local)
        except OSError:
            pass


def ingest_agency_business_info(
    local_path: Optional[str] = None, *, mirror_on_unchanged: bool = True
) -> Dict[str, Any]:
    """Seed / refresh agency knowledge from business_info.txt."""
    if local_path is None:
        local_path = os.path.join(os.path.dirname(__file__), "business_info.txt")
    if not os.path.isfile(local_path):
        return {"status": "error", "error": f"business_info.txt not found at {local_path}"}
    with open(local_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    result = upsert_document(
        source_type="agency",
        source_id="agency:business_info.txt",
        text=text,
        title="Mojo AI Agency — business_info",
        metadata={"origin": "business_info.txt"},
    )
    # Mirror to OneDrive agency folder when configured
    _mirror_states = ("ok", "unchanged") if mirror_on_unchanged else ("ok",)
    if _file_ops and _file_ops.onedrive_configured() and result.get("status") in _mirror_states:
        try:
            ensure_onedrive_knowledge_tree()
            remote_folder = f"{knowledge_root()}/agency"
            _file_ops.upload_to_onedrive(
                local_path,
                remote_folder=remote_folder,
                remote_name="business_info.txt",
            )
            result["onedrive_mirror"] = f"{remote_folder}/business_info.txt"
        except Exception as e:
            result["onedrive_mirror_error"] = str(e)
    return result


def ingest_onedrive_knowledge_tree(
    subfolder: str = "",
    limit: int = 100,
    prune: Optional[bool] = None,
) -> Dict[str, Any]:
    """Walk Documents/aimojo and ingest supported files. On a FULL walk (no subfolder, nothing cut
    by `limit`, listing non-empty) documents whose OneDrive file was deleted are pruned, so the KB
    never answers from files you removed."""
    ensure = ensure_onedrive_knowledge_tree()
    if not ensure.get("ok") and ensure.get("error"):
        log.info("ensure tree: %s", ensure)
    files = list_onedrive_knowledge_files(subfolder)
    results = []
    seen: List[str] = []
    for item in files[:limit]:
        path = item.get("path") or item.get("remote_path")
        if not path:
            continue
        seen.append(f"onedrive:{path.lstrip('/')}")
        results.append(ingest_onedrive_path(path))
    ok = sum(1 for r in results if r.get("status") == "ok")
    unchanged = sum(1 for r in results if r.get("status") == "unchanged")
    errors = [r for r in results if r.get("status") == "error"]
    pruned = 0
    do_prune = (not subfolder and len(files) <= limit and bool(files)) if prune is None else prune
    if do_prune and not errors and schema_ready() and _supabase is not None:
        try:
            have = (
                _supabase.table("documents").select("id, source_id").like("source_id", "onedrive:%").execute().data
                or []
            )
            keep = set(seen)
            for d in have:
                if d.get("source_id") not in keep:
                    _supabase.table("documents").delete().eq("id", d["id"]).execute()
                    pruned += 1
        except Exception as e:
            log.warning("prune failed: %s", e)
    return {
        "status": "ok", "scanned": len(files), "ingested": ok, "unchanged": unchanged,
        "pruned": pruned, "errors": errors[:10], "root": knowledge_root(),
        "subfolder": subfolder or "(all)",
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

_SRC_LABEL = {
    "agency": "Agency", "document": "Document", "chat_note": "Chat note",
    "export": "Export", "onedrive": "OneDrive",
}


def _merge_adjacent(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stitch consecutive chunks of the same document back into one passage (removing the
    chunk-overlap duplication), so an answer split across a boundary reaches the LLM whole."""
    by_doc: "OrderedDict[Any, List[Dict[str, Any]]]" = OrderedDict()
    for r in rows:
        by_doc.setdefault(r.get("document_id"), []).append(r)
    out: List[Dict[str, Any]] = []
    for _doc, group in by_doc.items():
        group.sort(key=lambda r: r.get("ordinal") or 0)
        cur: Optional[Dict[str, Any]] = None
        last_ord = None
        for r in group:
            o = r.get("ordinal") or 0
            if cur is not None and last_ord is not None and o == last_ord + 1:
                prev, nxt = cur["content"], r.get("content") or ""
                k = min(len(prev), len(nxt), CHUNK_OVERLAP + 40)
                while k > 10 and not prev.endswith(nxt[:k]):
                    k -= 1
                cur["content"] = prev + (nxt[k:] if k > 10 else "\n" + nxt)
                cur["_score"] = max(cur["_score"], r["_score"])
            else:
                if cur is not None:
                    out.append(cur)
                cur = dict(r)
                cur["_score"] = r["_score"]
            last_ord = o
        if cur is not None:
            out.append(cur)
    out.sort(key=lambda r: -r["_score"])
    return out


def _format_hit(row: Dict[str, Any], score: Optional[float] = None) -> str:
    label = _SRC_LABEL.get(row.get("source_type") or "doc", row.get("source_type") or "doc")
    bits = [label]
    for key, n in (("title", 60), ("section", 40), ("page_or_sheet", 30)):
        v = row.get(key)
        if v:
            bits.append(str(v)[:n])
    body = (row.get("content") or "").strip()
    if len(body) > 1600:
        body = body[:1600] + "…"
    return f"[{' | '.join(bits)}] {body}"


def _hit_score(row: Dict[str, Any]) -> float:
    for k in ("score", "similarity"):
        try:
            if row.get(k) is not None:
                return float(row[k])
        except (TypeError, ValueError):
            pass
    return 0.0


def hybrid_retrieve(
    query: str,
    *,
    top_k: int = 6,
    chat_id: Optional[str] = None,
    source_types: Optional[List[str]] = None,
    min_sim: Optional[float] = None,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Scoped, relevance-gated hits. Scope = global rows + this chat's rows; never another chat's.
    Empty list if schema missing, embeddings fail, or nothing clears the relevance gate."""
    if not query or not schema_ready() or _supabase is None:
        return []
    top_k = max(1, min(int(top_k or 6), 12))
    floor = MIN_SIM if min_sim is None else float(min_sim)
    try:
        qvec = embed_query(query)
    except Exception as e:
        log.error("embed_query failed: %s", e)
        return []

    rows: List[Dict[str, Any]] = []
    params: Dict[str, Any] = {"query_text": query, "query_embedding": qvec, "match_count": top_k}
    if chat_id:
        params["filter_chat_id"] = chat_id
    if source_types:
        params["filter_source_types"] = source_types
    try:
        rows = _supabase.rpc("hybrid_search", params).execute().data or []
    except Exception as e:
        log.warning("hybrid_search RPC failed: %s", str(e)[:200])
    if not rows:
        try:
            vparams = {k: v for k, v in params.items() if k not in ("query_text",)}
            rows = _supabase.rpc("match_chunks", vparams).execute().data or []
        except Exception as e:
            log.warning("match_chunks RPC failed: %s", str(e)[:200])
            return []

    # Defence in depth: whatever the SQL version, never surface another chat's private rows.
    rows = [r for r in rows if r.get("chat_id") in (None, "", chat_id)]

    kept: List[Dict[str, Any]] = []
    for r in rows:
        sim = r.get("similarity")
        lex = r.get("lex_hit")
        ok = bool(lex) or sim is None or float(sim) >= floor
        log.info("RAG_HIT ok=%s sim=%s lex=%s src=%s title=%r", ok,
                 None if sim is None else round(float(sim), 3), lex, r.get("source_type"), (r.get("title") or "")[:40])
        if ok or debug:
            r["_kept"] = ok
            kept.append(r)
    return kept


def _maybe_rerank(query: str, rows: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    if not (RERANK_ENABLED and _rerank and len(rows) > top_k):
        return rows
    try:
        order = _rerank(query, [(r.get("content") or "")[:700] for r in rows])
        seen, ranked = set(), []
        for i in order or []:
            if isinstance(i, int) and 0 <= i < len(rows) and i not in seen:
                seen.add(i)
                ranked.append(rows[i])
        ranked += [r for i, r in enumerate(rows) if i not in seen]
        return ranked
    except Exception as e:
        log.warning("rerank failed (keeping fused order): %s", e)
        return rows


def search(
    query: str,
    *,
    top_k: int = 6,
    chat_id: Optional[str] = None,
    include_chat_notes: bool = True,
    group_notes: Optional[List[str]] = None,
) -> str:
    """Agent-facing search: gated hybrid hits, adjacent chunks merged, labelled with source.
    Raw chat notes are added only when RAG has no chat_note hit (e.g. not synced yet)."""
    query = (query or "").strip()
    if not query:
        return "Empty query."

    pool = top_k * 2 if (RERANK_ENABLED and _rerank) else top_k
    hits = hybrid_retrieve(query, top_k=pool, chat_id=chat_id)
    for r in hits:
        r["_score"] = _hit_score(r)
    hits = _maybe_rerank(query, hits, top_k)[:top_k]
    merged = _merge_adjacent(hits) if hits else []
    chunks = [_format_hit(h) for h in merged]

    if include_chat_notes and group_notes and not any(h.get("source_type") == "chat_note" for h in merged):
        tokens = set(re.findall(r"\w+", query.lower()))
        for note in group_notes:
            nlow = (note or "").lower()
            if note and (not tokens or any(t in nlow for t in tokens)):
                chunks.append(f"[Chat note] {note[:800]}")
            if len(chunks) >= top_k + 4:
                break

    if not chunks:
        if not schema_ready():
            return (
                "RAG knowledge base not initialised "
                "(run sql/rag_schema.sql + sql/rag_schema_v2.sql, then /ingest agency). "
                "Falling back unavailable — use web_search or answer from general knowledge."
            )
        return "No relevant knowledge found. Answer from general knowledge or use web_search."
    return "\n---\n".join(chunks[: top_k + 3])


def kb_status() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "schema_ready": schema_ready(),
        "embedding_provider": DEFAULT_EMBEDDING_PROVIDER,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "min_sim": MIN_SIM,
        "rerank": bool(RERANK_ENABLED and _rerank),
        "onedrive_root": knowledge_root(),
        "onedrive_configured": bool(_file_ops and getattr(_file_ops, "onedrive_configured", lambda: False)()),
        "last_sync": _last_sync,
    }
    if _supabase is not None and info["schema_ready"]:
        try:
            info["documents"] = _supabase.table("documents").select("id", count="exact").limit(1).execute().count
            info["chunks"] = _supabase.table("chunks").select("id", count="exact").limit(1).execute().count
            counts: Dict[str, int] = {}
            for st in _SRC_LABEL:
                n = _supabase.table("documents").select("id", count="exact").eq("source_type", st).limit(1).execute().count
                if n:
                    counts[st] = n
            info["by_source_type"] = counts
            stale = (
                _supabase.table("chunks").select("id", count="exact")
                .neq("embedding_model", DEFAULT_EMBEDDING_MODEL).limit(1).execute().count
            )
            info["stale_embedding_chunks"] = stale
        except Exception as e:
            info["count_error"] = str(e)
    return info


# ---------------------------------------------------------------------------
# Chat documents (user sends a file → "save this to knowledge") + forget
# ---------------------------------------------------------------------------

_DOC_TTL_S = 2 * 3600
_DOC_MAX = 24
_MAX_RAW_B64 = 14_000_000  # ~10 MB of file bytes
_doc_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_doc_lock = threading.Lock()


def remember_document(
    chat_id: str,
    name: str,
    *,
    text: Optional[str] = None,
    local_path: Optional[str] = None,
    mime: Optional[str] = None,
) -> None:
    """Keep the most recent documents of a chat in a small TTL cache so the agent can later save one
    to the knowledge base on request. Text is extracted now when cheap; scanned PDFs / images keep
    their bytes and are read by the vision LLM only if the user actually asks to save them."""
    try:
        body = (text or "").strip()
        raw_b64 = None
        if not body and local_path and os.path.isfile(local_path):
            try:
                body = extract_text_from_local(local_path).strip()
            except Exception:
                body = ""
            if len(body) < 40 and os.path.getsize(local_path) * 4 // 3 <= _MAX_RAW_B64:
                with open(local_path, "rb") as f:
                    raw_b64 = base64.b64encode(f.read()).decode()
        if not body and not raw_b64:
            return
        with _doc_lock:
            now = time.time()
            for k in [k for k, v in _doc_cache.items() if now - v["ts"] > _DOC_TTL_S]:
                _doc_cache.pop(k, None)
            _doc_cache[f"{chat_id}|{name}"] = {
                "chat_id": chat_id, "name": name, "text": body[:300_000],
                "b64": raw_b64, "mime": mime or "application/pdf", "ts": now,
            }
            while len(_doc_cache) > _DOC_MAX:
                _doc_cache.popitem(last=False)
    except Exception as e:
        log.warning("remember_document failed: %s", e)


def _pick_cached(chat_id: str, title: Optional[str]) -> Optional[Dict[str, Any]]:
    with _doc_lock:
        now = time.time()
        cands = [v for v in _doc_cache.values() if v["chat_id"] == chat_id and now - v["ts"] <= _DOC_TTL_S]
    if not cands:
        return None
    if title:
        t = title.lower()
        for v in sorted(cands, key=lambda v: -v["ts"]):
            if t in v["name"].lower():
                return v
    return max(cands, key=lambda v: v["ts"])


def save_to_knowledge(
    *,
    chat_id: str,
    sender_id: Optional[str] = None,
    title: Optional[str] = None,
    text: Optional[str] = None,
    scope: str = "chat",
    is_owner: bool = False,
) -> Dict[str, Any]:
    """Index explicit text, or the most recent document sent in this chat. scope 'global' is
    owner-only; 'chat' rows are visible only in this chat."""
    scope = (scope or "chat").lower()
    if scope == "global" and not is_owner:
        return {"status": "error", "error": "global knowledge can only be changed by the owner"}
    body = (text or "").strip()
    name = (title or "").strip()
    if not body:
        doc = _pick_cached(chat_id, name or None)
        if not doc:
            return {"status": "error", "error": "no recent document in this chat — send the file first (documents are kept for 2 hours)"}
        name = name or doc["name"]
        body = doc["text"]
        if len(body) < 40 and doc.get("b64") and _vision_extract:
            try:
                body = (_vision_extract(doc["b64"], doc["mime"], doc["name"]) or "").strip()
            except Exception as e:
                return {"status": "error", "error": f"could not read scanned document: {e}"}
        if not body:
            return {"status": "error", "error": "document has no extractable text"}
    name = name or "Untitled note"
    scope_chat = None if scope == "global" else chat_id
    return upsert_document(
        source_type="document",
        source_id=f"chatdoc:{scope_chat or 'global'}:{name.lower()}",
        text=body,
        title=name,
        chat_id=scope_chat,
        sender_id=sender_id,
        metadata={"origin": "chat", "scope": scope},
    )


def delete_documents(match: str, *, chat_id: Optional[str], is_owner: bool = False) -> Dict[str, Any]:
    """Delete documents whose title contains `match`, restricted to this chat (+ global when owner).
    Chunks cascade. Agency/OneDrive-synced docs are re-created by sync unless the source file is removed."""
    match = (match or "").strip()
    if len(match) < 3:
        return {"status": "error", "error": "give at least 3 characters of the title"}
    if _supabase is None or not schema_ready():
        return {"status": "error", "error": "RAG not ready"}
    try:
        q = _supabase.table("documents").select("id, title, chat_id, source_type").ilike("title", f"%{match}%").limit(25)
        rows = q.execute().data or []
        victims = [
            r for r in rows
            if (r.get("chat_id") == chat_id and chat_id) or (r.get("chat_id") in (None, "") and is_owner)
        ]
        for r in victims:
            _supabase.table("documents").delete().eq("id", r["id"]).execute()
        return {"status": "ok", "deleted": [r.get("title") for r in victims]}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Chat notes → RAG (permanent notes become searchable by meaning, not just keywords)
# ---------------------------------------------------------------------------

def sync_chat_notes(chat_id: str, notes: Optional[List[str]] = None) -> Dict[str, Any]:
    if not chat_id:
        return {"status": "skipped", "reason": "no chat_id"}
    if notes is None:
        notes = (_get_notes(chat_id) if _get_notes else []) or []
    notes = [str(n).strip() for n in notes if n and str(n).strip()]
    sid = f"chat_note:{chat_id}"
    if not notes:
        try:
            if _supabase is not None and schema_ready():
                _supabase.table("documents").delete().eq("source_type", "chat_note").eq("source_id", sid).execute()
        except Exception:
            pass
        return {"status": "skipped", "reason": "no notes"}
    return upsert_document(
        source_type="chat_note",
        source_id=sid,
        text="\n\n".join(f"- {n}" for n in notes),
        title="Chat notes",
        chat_id=chat_id,
        metadata={"origin": "group_memory", "notes": len(notes)},
    )


_note_timers: Dict[str, threading.Timer] = {}
_note_lock = threading.Lock()


def sync_chat_notes_async(chat_id: str, delay: float = 4.0) -> None:
    """Debounced background re-index after a note write (bursts of saves collapse into one embed)."""
    if not chat_id or _supabase is None:
        return

    def _run():
        with _note_lock:
            _note_timers.pop(chat_id, None)
        try:
            r = sync_chat_notes(chat_id)
            log.info("RAG_NOTES_SYNC chat=%s %s", chat_id, r.get("status"))
        except Exception as e:
            log.warning("notes sync failed for %s: %s", chat_id, e)

    with _note_lock:
        old = _note_timers.pop(chat_id, None)
        if old:
            old.cancel()
        t = threading.Timer(delay, _run)
        t.daemon = True
        _note_timers[chat_id] = t
        t.start()


def sync_all_chat_notes() -> Dict[str, Any]:
    if not _all_notes:
        return {"status": "skipped", "reason": "all_notes_fn not configured"}
    done = changed = 0
    for cid, notes in (_all_notes() or {}).items():
        r = sync_chat_notes(cid, notes)
        done += 1
        changed += 1 if r.get("status") == "ok" else 0
    return {"status": "ok", "chats": done, "reindexed": changed}


# ---------------------------------------------------------------------------
# Background sync (boot + periodic): agency file, OneDrive tree (+prune), chat notes
# ---------------------------------------------------------------------------

_sync_lock = threading.Lock()
_sync_started = False
_last_sync: Dict[str, Any] = {}


def run_full_sync() -> Dict[str, Any]:
    """Single-flight. Every step is hash-gated, so a no-change sync costs a few SELECTs, no embeds."""
    if not _sync_lock.acquire(blocking=False):
        return {"status": "busy"}
    try:
        res: Dict[str, Any] = {"at": datetime.now(timezone.utc).isoformat()}
        if not schema_ready():
            res["status"] = "schema_missing"
        else:
            res["agency"] = ingest_agency_business_info(mirror_on_unchanged=False).get("status")
            if _file_ops and getattr(_file_ops, "onedrive_configured", lambda: False)():
                od = ingest_onedrive_knowledge_tree()
                res["onedrive"] = {k: od.get(k) for k in ("scanned", "ingested", "unchanged", "pruned")}
            res["notes"] = sync_all_chat_notes()
            res["status"] = "ok"
        global _last_sync
        _last_sync = res
        log.info("RAG_SYNC %s", res)
        return res
    except Exception as e:
        log.error("RAG_SYNC failed: %s", e)
        return {"status": "error", "error": str(e)}
    finally:
        _sync_lock.release()


def start_background_sync() -> None:
    """Boot sync after a short delay (lets the WhatsApp session connect first), then every
    RAG_SYNC_MIN minutes. Render free tier restarts often, so boot sync is the one that matters."""
    global _sync_started
    if _sync_started or not AUTO_SYNC or _supabase is None:
        return
    if not os.getenv("GEMINI_API_KEY") and DEFAULT_EMBEDDING_PROVIDER == "gemini":
        log.warning("RAG auto-sync disabled: GEMINI_API_KEY not set")
        return
    _sync_started = True

    def _loop():
        time.sleep(25)
        while True:
            run_full_sync()
            if SYNC_INTERVAL_MIN <= 0:
                return
            time.sleep(SYNC_INTERVAL_MIN * 60)

    threading.Thread(target=_loop, name="rag-sync", daemon=True).start()
