-- Run this in the Supabase SQL editor.

-- One row per global setting. Currently just "agent_enabled", but built to
-- hold any future global on/off switch without a new table.
create table if not exists bot_settings (
  key text primary key,
  value boolean not null,
  updated_at timestamp default now()
);

-- Per-chat, per-feature toggle. chat_id = '*' means "all chats" for that
-- feature. No row at all = OFF (features are opt-in by default).
create table if not exists feature_flags (
  chat_id text not null,
  feature text not null,
  enabled boolean not null,
  updated_at timestamp default now(),
  primary key (chat_id, feature)
);

-- Nothing to seed — agent defaults ON with no row (see get_global_setting's
-- default), and every feature defaults OFF with no row, exactly as requested.
-- Everything from here on is done through commands from your private admin
-- chat with the bot, e.g.:
--   /enable reminders 120363428074513004@g.us
--   /disable agent
--   /status 120363428074513004@g.us