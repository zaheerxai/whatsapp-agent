-- Persistent cache for paid/slow results (Supadata transcripts + metadata, credit ledger).
-- Run once in the Supabase SQL editor. Without it the agent still works (memory-only cache),
-- but Render Free wipes memory on every spin-down, so the same reel would be paid for again.
create table if not exists public.kv_cache (
  key        text primary key,
  value      text        not null,
  expires_at timestamptz not null
);

create index if not exists kv_cache_expires_idx on public.kv_cache (expires_at);

-- The bot uses the service-role key, which bypasses RLS. Keep RLS ON so the anon key
-- can never read cached transcripts.
alter table public.kv_cache enable row level security;

-- Optional housekeeping (run manually or via pg_cron): delete from public.kv_cache where expires_at < now();
