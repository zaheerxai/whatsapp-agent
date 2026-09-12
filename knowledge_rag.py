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

import hashlib
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_PROVIDER = (os.getenv("EMBEDDING_PROVIDER") or "gemini").strip().lower()
DEFAULT_EMBEDDING_MODEL = (
    os.getenv("EMBEDDING_MODEL")
    or ("text-embedding-004" if DEFAULT_EMBEDDING_PROVIDER == "gemini" else "text-embedding-3-small")
).strip()
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM") or "768")
KNOWLEDGE_ONEDRIVE_ROOT = (
    os.getenv("KNOWLEDGE_ONEDRIVE_ROOT") or "Documents/aimojo"
).strip().strip("/")

# Chunk targets (characters ≈ tokens for mixed English / Roman Urdu)
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE") or "1200")
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP") or "150")
MAX_CHUNKS_PER_DOC = int(os.getenv("RAG_MAX_CHUNKS_PER_DOC") or "400")

# Supported ingest extensions
TEXT_EXTS = {".txt", ".md", ".csv", ".json", ".log", ".html", ".htm"}
DOC_EXTS = {".docx", ".xlsx", ".pptx", ".pdf"}
SUPPORTED_EXTS = TEXT_EXTS | DOC_EXTS

_supabase = None
_file_ops = None
_extract_document_text = None  # injected from whatsapp_agent to reuse XLSX smart extract

# Soft in-process caches (query embedding + schema readiness)
_query_embed_cache: Dict[str, Tuple[float, List[float]]] = {}
_QUERY_EMBED_TTL = 45.0
_schema_ok: Optional[bool] = None


def init(supabase_client, file_ops_module=None, extract_document_text_fn=None) -> None:
    global _supabase, _file_ops, _extract_document_text, _schema_ok
    _supabase = supabase_client
    _file_ops = file_ops_module
    _extract_document_text = extract_document_text_fn
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

def _gemini_embed(texts: List[str], model: str, is_query: bool = False) -> List[List[float]]:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set (required for default embeddings)")
    out: List[List[float]] = []
    # Batch one-by-one for compatibility; Gemini batch embed exists but varies by region
    base = "https://generativelanguage.googleapis.com/v1beta"
    task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
    for t in texts:
        url = f"{base}/models/{model}:embedContent?key={key}"
        payload = {
            "model": f"models/{model}",
            "content": {"parts": [{"text": t[:8000]}]},
            "taskType": task,
        }
        r = requests.post(url, json=payload, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"Gemini embed {r.status_code}: {r.text[:300]}")
        body = r.json()
        values = (
            body.get("embedding", {}).get("values")
            or body.get("embeddings", [{}])[0].get("values")
        )
        if not values:
            raise RuntimeError(f"Gemini embed empty response: {str(body)[:200]}")
        # Truncate / pad to expected dim if model returns different size
        if len(values) > EMBEDDING_DIM:
            values = values[:EMBEDDING_DIM]
        elif len(values) < EMBEDDING_DIM:
            values = values + [0.0] * (EMBEDDING_DIM - len(values))
        out.append(values)
        time.sleep(0.05)  # polite rate limit
    return out


def _openai_embed(texts: List[str], model: str) -> List[List[float]]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    r = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "input": [t[:8000] for t in texts]},
        timeout=90,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI embed {r.status_code}: {r.text[:300]}")
    data = r.json().get("data") or []
    data = sorted(data, key=lambda x: x.get("index", 0))
    out = []
    for item in data:
        values = item.get("embedding") or []
        if len(values) > EMBEDDING_DIM:
            values = values[:EMBEDDING_DIM]
        elif len(values) < EMBEDDING_DIM:
            values = values + [0.0] * (EMBEDDING_DIM - len(values))
        out.append(values)
    return out


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
    # Prefix for e5-style models when using OpenAI-hosted e5 (rare); Gemini uses taskType
    prepared = texts
    if provider == "openai" and "e5" in model.lower():
        prefix = "query: " if is_query else "passage: "
        prepared = [prefix + t for t in texts]

    if provider == "openai":
        return _openai_embed(prepared, model)
    return _gemini_embed(prepared, model, is_query=is_query)


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
    """True if documents + chunks tables are reachable."""
    global _schema_ok
    if _schema_ok is not None:
        return _schema_ok
    if _supabase is None:
        _schema_ok = False
        return False
    try:
        _supabase.table("documents").select("id").limit(1).execute()
        _supabase.table("chunks").select("id").limit(1).execute()
        _schema_ok = True
    except Exception as e:
        print(f"[knowledge_rag] schema not ready: {e}")
        _schema_ok = False
    return _schema_ok


def reset_schema_probe() -> None:
    global _schema_ok
    _schema_ok = None


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
    Idempotent ingest. Skips re-embed when content_hash matches existing row.
    Returns {status, document_id, chunks, skipped_reason?}.
    """
    if _supabase is None:
        return {"status": "error", "error": "supabase not initialised"}
    if not schema_ready():
        return {
            "status": "error",
            "error": "RAG tables missing — run sql/rag_schema.sql in Supabase",
        }

    text = normalise_text(text)
    if not text or len(text) < 20:
        return {"status": "skipped", "reason": "text too short"}

    ch = content_hash(text)
    model = embedding_model or DEFAULT_EMBEDDING_MODEL
    meta = metadata or {}

    # Existing?
    existing = (
        _supabase.table("documents")
        .select("id, content_hash, source_version")
        .eq("source_type", source_type)
        .eq("source_id", source_id)
        .limit(1)
        .execute()
    )
    rows = existing.data or []
    if rows and rows[0].get("content_hash") == ch:
        return {
            "status": "unchanged",
            "document_id": rows[0]["id"],
            "chunks": 0,
            "source_id": source_id,
        }

    chunk_rows = chunk_with_meta(text, title=title or source_id)
    if not chunk_rows:
        return {"status": "skipped", "reason": "no chunks"}

    # Embed in small batches
    vectors: List[List[float]] = []
    batch_size = 8
    contents = [c["content"] for c in chunk_rows]
    for i in range(0, len(contents), batch_size):
        vectors.extend(embed_texts(contents[i : i + batch_size], is_query=False, model=model))

    version = 1
    doc_id = None
    if rows:
        doc_id = rows[0]["id"]
        version = int(rows[0].get("source_version") or 1) + 1
        _supabase.table("documents").update(
            {
                "content_hash": ch,
                "source_version": version,
                "title": title or source_id,
                "chat_id": chat_id,
                "sender_id": sender_id,
                "metadata": meta,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("id", doc_id).execute()
        # Replace chunks
        _supabase.table("chunks").delete().eq("document_id", doc_id).execute()
    else:
        ins = (
            _supabase.table("documents")
            .insert(
                {
                    "source_type": source_type,
                    "source_id": source_id,
                    "chat_id": chat_id,
                    "sender_id": sender_id,
                    "title": title or source_id,
                    "content_hash": ch,
                    "source_version": 1,
                    "metadata": meta,
                }
            )
            .execute()
        )
        doc_id = (ins.data or [{}])[0].get("id")
        if not doc_id:
            return {"status": "error", "error": "document insert returned no id"}

    # Insert chunks (batch)
    payload = []
    for c, vec in zip(chunk_rows, vectors):
        payload.append(
            {
                "document_id": doc_id,
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
        )
    # supabase-py insert in chunks of 50
    for i in range(0, len(payload), 50):
        _supabase.table("chunks").insert(payload[i : i + 50]).execute()

    return {
        "status": "ok",
        "document_id": doc_id,
        "chunks": len(payload),
        "source_id": source_id,
        "source_version": version if rows else 1,
        "content_hash": ch,
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
            print(f"[knowledge_rag] extract via wa failed {path}: {e}")

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


def ingest_agency_business_info(local_path: Optional[str] = None) -> Dict[str, Any]:
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
    if _file_ops and _file_ops.onedrive_configured() and result.get("status") in (
        "ok",
        "unchanged",
    ):
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
) -> Dict[str, Any]:
    """Walk Documents/aimojo and ingest supported files."""
    ensure = ensure_onedrive_knowledge_tree()
    if not ensure.get("ok") and ensure.get("error"):
        # still try list if folders already exist
        print(f"[knowledge_rag] ensure tree: {ensure}")
    files = list_onedrive_knowledge_files(subfolder)
    results = []
    for item in files[:limit]:
        path = item.get("path") or item.get("remote_path")
        if not path:
            continue
        results.append(ingest_onedrive_path(path))
    ok = sum(1 for r in results if r.get("status") == "ok")
    unchanged = sum(1 for r in results if r.get("status") == "unchanged")
    errors = [r for r in results if r.get("status") == "error"]
    return {
        "status": "ok",
        "scanned": len(files),
        "ingested": ok,
        "unchanged": unchanged,
        "errors": errors[:10],
        "root": knowledge_root(),
        "subfolder": subfolder or "(all)",
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _format_hit(row: Dict[str, Any], score: Optional[float] = None) -> str:
    st = row.get("source_type") or "doc"
    label = {
        "agency": "Agency",
        "document": "Document",
        "chat_note": "Chat note",
        "export": "Export",
        "onedrive": "OneDrive",
    }.get(st, st)
    title = row.get("title") or ""
    section = row.get("section") or ""
    sheet = row.get("page_or_sheet") or ""
    sc = score if score is not None else row.get("score") or row.get("similarity")
    head_bits = [label]
    if title:
        head_bits.append(str(title)[:60])
    if section:
        head_bits.append(str(section)[:40])
    if sheet:
        head_bits.append(str(sheet)[:30])
    if sc is not None:
        try:
            head_bits.append(f"score={float(sc):.3f}")
        except (TypeError, ValueError):
            pass
    header = " | ".join(head_bits)
    body = (row.get("content") or "").strip()
    if len(body) > 900:
        body = body[:900] + "…"
    return f"[{header}] {body}"


def hybrid_retrieve(
    query: str,
    *,
    top_k: int = 6,
    chat_id: Optional[str] = None,
    source_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Return raw hit dicts. Empty list if schema missing or no hits."""
    if not query or not schema_ready() or _supabase is None:
        return []
    top_k = max(1, min(int(top_k or 6), 12))
    try:
        qvec = embed_query(query)
    except Exception as e:
        print(f"[knowledge_rag] embed_query failed: {e}")
        return []

    # Prefer hybrid_search RPC
    try:
        params: Dict[str, Any] = {
            "query_text": query,
            "query_embedding": qvec,
            "match_count": top_k,
        }
        if chat_id:
            params["filter_chat_id"] = chat_id
        if source_types:
            params["filter_source_types"] = source_types
        resp = _supabase.rpc("hybrid_search", params).execute()
        rows = resp.data or []
        if rows:
            return rows
    except Exception as e:
        print(f"[knowledge_rag] hybrid_search RPC failed: {e}")

    # Fallback pure vector
    try:
        params = {
            "query_embedding": qvec,
            "match_count": top_k,
        }
        if chat_id:
            params["filter_chat_id"] = chat_id
        if source_types:
            params["filter_source_types"] = source_types
        resp = _supabase.rpc("match_chunks", params).execute()
        return resp.data or []
    except Exception as e:
        print(f"[knowledge_rag] match_chunks RPC failed: {e}")
        return []


def search(
    query: str,
    *,
    top_k: int = 6,
    chat_id: Optional[str] = None,
    include_chat_notes: bool = True,
    group_notes: Optional[List[str]] = None,
) -> str:
    """
    Agent-facing search. Tries RAG hybrid; merges group_memory keyword notes;
    returns labelled snippets or a clear miss message.
    """
    query = (query or "").strip()
    if not query:
        return "Empty query."

    chunks: List[str] = []
    hits = hybrid_retrieve(query, top_k=top_k, chat_id=chat_id)
    for h in hits:
        chunks.append(_format_hit(h))

    # Chat notes: prefer DB RAG chat_note type; also accept caller-supplied notes
    if include_chat_notes and group_notes:
        tokens = set(re.findall(r"\w+", query.lower()))
        for note in group_notes:
            if not note:
                continue
            nlow = note.lower()
            if not tokens or any(t in nlow for t in tokens):
                chunks.append(f"[Chat note] {note[:800]}")
            if len(chunks) >= top_k + 4:
                break

    if not chunks:
        if not schema_ready():
            return (
                "RAG knowledge base not initialised "
                "(run sql/rag_schema.sql + /ingest agency). "
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
        "onedrive_root": knowledge_root(),
        "onedrive_configured": bool(
            _file_ops and getattr(_file_ops, "onedrive_configured", lambda: False)()
        ),
    }
    if _supabase is not None and info["schema_ready"]:
        try:
            docs = _supabase.table("documents").select("id", count="exact").execute()
            ch = _supabase.table("chunks").select("id", count="exact").execute()
            info["documents"] = docs.count if docs.count is not None else len(docs.data or [])
            info["chunks"] = ch.count if ch.count is not None else len(ch.data or [])
            by_type = (
                _supabase.table("documents")
                .select("source_type")
                .execute()
            )
            counts: Dict[str, int] = {}
            for r in by_type.data or []:
                st = r.get("source_type") or "?"
                counts[st] = counts.get(st, 0) + 1
            info["by_source_type"] = counts
        except Exception as e:
            info["count_error"] = str(e)
    return info
