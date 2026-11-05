-- Encrypted storage for long-lived credentials (OneDrive refresh token, ...). Run once in the Supabase SQL editor.
-- Values are Fernet ciphertext written by secret_store.py using TOKEN_ENCRYPTION_KEY (kept in the host
-- environment, never in the database). Without this table the bot still works: tokens stay in memory and
-- the env seed is used again after a restart.
create table if not exists public.oauth_tokens (
  provider   text primary key,
  ciphertext text        not null,
  updated_at timestamptz not null default now()
);

-- Only the service-role key (which bypasses RLS) may read/write. RLS on + no policies denies everyone else;
-- the explicit REVOKE is defence in depth in case a policy is added by mistake later.
alter table public.oauth_tokens enable row level security;
revoke all on public.oauth_tokens from anon, authenticated;
