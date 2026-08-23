-- Run this in the Supabase SQL editor.

create table if not exists contacts (
  sender_id text primary key,
  display_name text,
  nickname text,
  updated_at timestamp with time zone default now()
);

create table if not exists group_memory (
  id bigint generated always as identity primary key,
  chat_id text not null,
  sender_id text,
  note text not null,
  created_at timestamp with time zone default now()
);

-- If you already have a "reminders" table from testing, this adds the one new
-- column it needs for day-of-week recurrence without touching existing rows.
create table if not exists reminders (
  id bigint generated always as identity primary key,
  chat_id text not null,
  sender_id text not null,
  message text not null,
  remind_at timestamp not null,
  recurring boolean default false,
  daily_times text[],           -- e.g. ['08:00', '20:00'], null if one-time
  days text[],                  -- e.g. ['monday', 'thursday'], null = every day
  active boolean default true,
  created_at timestamp default now(),
  last_sent_at timestamp
);

alter table reminders add column if not exists days text[];

-- Per-chat on/off switch for the reminder feature. No row for a chat = falls
-- back to REMINDERS_DEFAULT_ENABLED in .env.
create table if not exists chat_settings (
  chat_id text primary key,
  reminders_enabled boolean default true,
  updated_at timestamp default now()
);

-- Examples:
-- Turn reminders OFF for one specific group, leaving everywhere else on:
--   insert into chat_settings (chat_id, reminders_enabled) values ('120363428074513004@g.us', false);
-- Turn reminders ON for just one private chat, with REMINDERS_DEFAULT_ENABLED=false in .env:
--   insert into chat_settings (chat_id, reminders_enabled) values ('142245760102636@lid', true);