-- RAG v2 migration (idempotent). Run AFTER sql/rag_schema.sql in the Supabase SQL editor.
--
-- What it fixes
--   1. SCOPE LEAK: v1 hybrid_search/match_chunks returned EVERY chat's chunks when
--      filter_chat_id was NULL. v2: NULL => global (chat_id IS NULL) rows only.
--   2. Relevance gating: returns the real cosine `similarity` and `lex_hit` (matched full-text)
--      so the app can drop irrelevant top-k filler instead of feeding it to the LLM.
--   3. Atomic re-index: replace_document_chunks() swaps a document's chunks in ONE transaction
--      (v1 deleted then inserted client-side; a failed insert left a doc with a hash and no chunks,
--      which then reported "unchanged" forever).
--
-- Embedding model change: text-embedding-004 was retired (404). The app now uses
-- gemini-embedding-001 @ 768 dims (L2-normalised). Old vectors live in a different space, so the
-- app re-embeds every document once automatically (metadata.embedding_model mismatch).
-- To force it now:  update documents set content_hash = 'reindex';

drop function if exists hybrid_search(text, vector, int, text, text[], float, float, int);
drop function if exists match_chunks(vector, int, text, text[]);

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
  id uuid, document_id uuid, content text, source_type text, chat_id text, sender_id text,
  title text, section text, page_or_sheet text, content_hash text, embedding_model text,
  ordinal int, score float, similarity float, lex_hit boolean, metadata jsonb
)
language sql stable
as $$
  with cand as (select greatest(20, least(match_count * 4, 80)) as n),
  scoped as (
    select c.* from chunks c
    where (c.chat_id is null or c.chat_id = filter_chat_id)
      and (filter_source_types is null or c.source_type = any(filter_source_types))
  ),
  vector_hits as (
    select s.id, row_number() over (order by s.embedding <=> query_embedding) as rk
    from scoped s
    order by s.embedding <=> query_embedding
    limit (select n from cand)
  ),
  lex_hits as (
    select s.id,
           row_number() over (order by ts_rank_cd(s.fts, websearch_to_tsquery('simple', query_text)) desc) as rk
    from scoped s
    where s.fts @@ websearch_to_tsquery('simple', query_text)
    order by ts_rank_cd(s.fts, websearch_to_tsquery('simple', query_text)) desc
    limit (select n from cand)
  ),
  fused as (
    select coalesce(vh.id, lh.id) as id,
           (coalesce(semantic_weight, 1.0)  * coalesce(1.0 / (rrf_k + vh.rk), 0.0)
          + coalesce(full_text_weight, 1.0) * coalesce(1.0 / (rrf_k + lh.rk), 0.0)) as score,
           (lh.id is not null) as lex_hit
    from vector_hits vh full outer join lex_hits lh on vh.id = lh.id
  )
  select c.id, c.document_id, c.content, c.source_type, c.chat_id, c.sender_id, c.title, c.section,
         c.page_or_sheet, c.content_hash, c.embedding_model, c.ordinal,
         f.score::float,
         (1 - (c.embedding <=> query_embedding))::float as similarity,
         f.lex_hit,
         c.metadata
  from fused f join chunks c on c.id = f.id
  order by f.score desc
  limit match_count;
$$;

create or replace function match_chunks(
  query_embedding vector(768),
  match_count int default 8,
  filter_chat_id text default null,
  filter_source_types text[] default null
)
returns table (
  id uuid, document_id uuid, content text, source_type text, chat_id text, sender_id text,
  title text, section text, page_or_sheet text, content_hash text, embedding_model text,
  ordinal int, similarity float, metadata jsonb
)
language sql stable
as $$
  select c.id, c.document_id, c.content, c.source_type, c.chat_id, c.sender_id, c.title, c.section,
         c.page_or_sheet, c.content_hash, c.embedding_model, c.ordinal,
         (1 - (c.embedding <=> query_embedding))::float as similarity, c.metadata
  from chunks c
  where (c.chat_id is null or c.chat_id = filter_chat_id)
    and (filter_source_types is null or c.source_type = any(filter_source_types))
  order by c.embedding <=> query_embedding
  limit match_count;
$$;

-- Atomic chunk swap. p_chunks: jsonb array of
--   {ordinal, content, token_count, embedding:[...], embedding_model, source_type, chat_id,
--    sender_id, title, section, page_or_sheet, content_hash, metadata}
create or replace function replace_document_chunks(p_document_id uuid, p_chunks jsonb)
returns int
language plpgsql
as $$
declare n int;
begin
  delete from chunks where document_id = p_document_id;
  insert into chunks (document_id, ordinal, content, token_count, embedding, embedding_model,
                      source_type, chat_id, sender_id, title, section, page_or_sheet,
                      content_hash, metadata)
  select p_document_id,
         (x->>'ordinal')::int,
         x->>'content',
         nullif(x->>'token_count','')::int,
         (x->>'embedding')::vector(768),
         x->>'embedding_model',
         x->>'source_type',
         nullif(x->>'chat_id',''),
         nullif(x->>'sender_id',''),
         x->>'title',
         x->>'section',
         x->>'page_or_sheet',
         x->>'content_hash',
         coalesce(x->'metadata', '{}'::jsonb)
  from jsonb_array_elements(p_chunks) as x;
  get diagnostics n = row_count;
  return n;
end;
$$;

-- Fast prune/lookup helpers
create index if not exists documents_source_id_idx on documents (source_id);
create index if not exists documents_title_trgm on documents using gin (title gin_trgm_ops);
