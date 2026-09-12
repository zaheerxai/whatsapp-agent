-- RAG knowledge schema for Mojo WhatsApp agent.
-- Run once in Supabase SQL editor (service role / dashboard).
-- Requires: pgvector extension.

create extension if not exists vector;
create extension if not exists pg_trgm;

-- Canonical documents (one row per source file / note bundle).
create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  source_type text not null check (source_type in (
    'agency', 'document', 'chat_note', 'export', 'onedrive'
  )),
  source_id text not null,                -- stable path or logical id
  chat_id text,                           -- null = global
  sender_id text,                         -- null = shared
  title text,
  content_hash text not null,
  source_version int not null default 1,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (source_type, source_id)
);

create index if not exists documents_chat_idx on documents (chat_id);
create index if not exists documents_source_type_idx on documents (source_type);
create index if not exists documents_hash_idx on documents (content_hash);

-- Chunks with dense embeddings + full-text for hybrid retrieval.
-- Default dimension 768 matches gemini text-embedding-004 / e5-base-class models.
-- If you switch embedding model, re-embed and alter the column type.
create table if not exists chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null references documents(id) on delete cascade,
  ordinal int not null,
  content text not null,
  token_count int,
  embedding vector(768) not null,
  embedding_model text not null,
  -- denormalised for fast filter without join
  source_type text not null,
  chat_id text,
  sender_id text,
  title text,
  section text,
  page_or_sheet text,
  content_hash text not null,
  fts tsvector generated always as (to_tsvector('simple', content)) stored,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists chunks_document_idx on chunks (document_id);
create index if not exists chunks_source_type_idx on chunks (source_type);
create index if not exists chunks_chat_idx on chunks (chat_id);
create index if not exists chunks_fts_idx on chunks using gin (fts);
create index if not exists chunks_content_trgm on chunks using gin (content gin_trgm_ops);

-- HNSW for low-latency ANN. Build may take a moment on first large ingest.
create index if not exists chunks_embedding_hnsw
  on chunks using hnsw (embedding vector_cosine_ops)
  with (m = 16, ef_construction = 64);

-- Hybrid search: vector + full-text fused with Reciprocal Rank Fusion.
-- match_count = final rows; internal candidates = match_count * 4 (capped).
create or replace function hybrid_search(
  query_text text,
  query_embedding vector(768),
  match_count int default 8,
  filter_chat_id text default null,
  filter_source_types text[] default null,
  full_text_weight float default 1.0,
  semantic_weight float default 1.0,
  rrf_k int default 60
)
returns table (
  id uuid,
  document_id uuid,
  content text,
  source_type text,
  chat_id text,
  sender_id text,
  title text,
  section text,
  page_or_sheet text,
  content_hash text,
  embedding_model text,
  ordinal int,
  score float,
  metadata jsonb
)
language sql
stable
as $$
  with cand as (
    select greatest(20, least(match_count * 4, 80)) as n
  ),
  vector_hits as (
    select
      c.id,
      row_number() over (order by c.embedding <=> query_embedding) as rk
    from chunks c, cand
    where
      (filter_chat_id is null or c.chat_id is null or c.chat_id = filter_chat_id)
      and (filter_source_types is null or c.source_type = any(filter_source_types))
    order by c.embedding <=> query_embedding
    limit (select n from cand)
  ),
  lex_hits as (
    select
      c.id,
      row_number() over (
        order by ts_rank_cd(c.fts, websearch_to_tsquery('simple', query_text)) desc
      ) as rk
    from chunks c, cand
    where
      c.fts @@ websearch_to_tsquery('simple', query_text)
      and (filter_chat_id is null or c.chat_id is null or c.chat_id = filter_chat_id)
      and (filter_source_types is null or c.source_type = any(filter_source_types))
    order by ts_rank_cd(c.fts, websearch_to_tsquery('simple', query_text)) desc
    limit (select n from cand)
  ),
  fused as (
    select
      coalesce(vh.id, lh.id) as id,
      (
        coalesce(semantic_weight, 1.0) * coalesce(1.0 / (rrf_k + vh.rk), 0.0)
        + coalesce(full_text_weight, 1.0) * coalesce(1.0 / (rrf_k + lh.rk), 0.0)
      ) as score
    from vector_hits vh
    full outer join lex_hits lh on vh.id = lh.id
  )
  select
    c.id,
    c.document_id,
    c.content,
    c.source_type,
    c.chat_id,
    c.sender_id,
    c.title,
    c.section,
    c.page_or_sheet,
    c.content_hash,
    c.embedding_model,
    c.ordinal,
    f.score::float,
    c.metadata
  from fused f
  join chunks c on c.id = f.id
  order by f.score desc
  limit match_count;
$$;

-- Optional: pure vector fallback when FTS query is empty / too short.
create or replace function match_chunks(
  query_embedding vector(768),
  match_count int default 8,
  filter_chat_id text default null,
  filter_source_types text[] default null
)
returns table (
  id uuid,
  document_id uuid,
  content text,
  source_type text,
  chat_id text,
  sender_id text,
  title text,
  section text,
  page_or_sheet text,
  content_hash text,
  embedding_model text,
  ordinal int,
  similarity float,
  metadata jsonb
)
language sql
stable
as $$
  select
    c.id,
    c.document_id,
    c.content,
    c.source_type,
    c.chat_id,
    c.sender_id,
    c.title,
    c.section,
    c.page_or_sheet,
    c.content_hash,
    c.embedding_model,
    c.ordinal,
    (1 - (c.embedding <=> query_embedding))::float as similarity,
    c.metadata
  from chunks c
  where
    (filter_chat_id is null or c.chat_id is null or c.chat_id = filter_chat_id)
    and (filter_source_types is null or c.source_type = any(filter_source_types))
  order by c.embedding <=> query_embedding
  limit match_count;
$$;
