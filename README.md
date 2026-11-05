# Mojo — WhatsApp AI Agent

Mojo is a Python-based AI assistant that runs through a **WhatsApp linked device**. It combines an LLM-driven agent loop with chat history, scoped memory, reminders, document and media processing, web research, GitHub and OneDrive integrations, Gmail workflows, and operational diagnostics.

The WhatsApp connection is handled by [Neonize](https://github.com/krypton-byte/neonize). Supabase provides the application's persistent data layer and, on hosts with ephemeral filesystems, storage for the linked-device session backup.

> This README describes the code currently present in the repository's `main` branch. Some capabilities require third-party credentials, database migrations, provider quotas, or external account setup. A feature being implemented does not mean its external service is configured for every deployment.

## Contents

- [Capabilities](#capabilities)
- [Architecture and message flow](#architecture-and-message-flow)
- [Agent tools](#agent-tools)
- [Access control](#access-control)
- [Requirements](#requirements)
- [Local installation](#local-installation)
- [Supabase configuration](#supabase-configuration)
- [Environment variables](#environment-variables)
- [Pairing WhatsApp](#pairing-whatsapp)
- [Render deployment](#render-deployment)
- [Owner commands](#owner-commands)
- [Tests](#tests)
- [Operations and troubleshooting](#operations-and-troubleshooting)
- [Repository structure](#repository-structure)
- [Security and privacy](#security-and-privacy)
- [License](#license)

## Capabilities

| Area | What is implemented |
|---|---|
| WhatsApp conversations | Private and group chat handling, mentions, quoted messages, contact resolution, duplicate-message protection and long-response chunking. |
| Agent orchestration | LLM-driven tool calling with per-chat tool allowlists, recent chat context and memory injection. |
| LLM routing | Groq primary generation with fallback routing to Gemini, token-budget estimates, history trimming, model cooldowns, bounded retries and response-continuation handling. |
| Gemini function calling | Native tool calling on tool-selection turns, provider-specific thought-signature preservation and a flattened no-tools fallback for incompatible requests. |
| Dynamic tool selection | An LLM planner can reduce the offered tool list when all tool schemas do not fit the configured Groq request budget. |
| Memory | Save, retrieve and forget chat-scoped notes. The RAG subsystem can index these notes for retrieval. |
| Knowledge base | Document chunking, embeddings, Supabase pgvector storage, hybrid lexical/vector retrieval, relevance gating, optional reranking and chat scoping. |
| Reminders | One-off and recurring reminders, selected weekdays, cancellation and sender timezone handling. |
| Web research | Provider-fallback web search plus a multi-query research pipeline that reads selected sources and assembles a cited report. |
| GitHub | Reads public repository overviews, README content and files; an optional GitHub token improves API rate limits. |
| Email | Owner-only draft creation, editing, preview, cancellation and Gmail sending after explicit confirmation. Job-application drafts are grounded in the supplied job description and resume text. |
| OneDrive and files | Lists and transfers supported OneDrive files, ingests knowledge documents, exports logs and stores owner notes. |
| Documents and images | Extracts supported PDF and Office document content; vision-capable model calls support image description and scanned-document extraction. |
| Voice notes | Groq Whisper transcription with optional LLM normalization for English, Roman Urdu and mixed-language speech. |
| Video/social links | Link metadata and transcript/summary workflows for supported public URLs. YouTube uses a free-first transcript/ASR fallback chain before metered providers. |
| Reliability | Queue-based logging, log rotation, a watchdog, incident reporting, stuck-handler diagnostics, health checks and optional token-protected out-of-band diagnostics. |

Provider access, model availability, rate limits and third-party terms can change. The application includes fallbacks, but no fallback chain can guarantee availability when every configured provider is unavailable or quota-limited.

## Architecture and message flow

```text
WhatsApp linked device (Neonize)
            |
            v
Message event / duplicate guard
            |
            +---- Owner slash-command fast lane
            |
            v
Message processing
  |         |          |
  |         |          +--> Images / PDFs / Office documents
  |         +-------------> Voice transcription and normalization
  +-----------------------> Chat history, contact context, memory
            |
            v
Agent loop
  |  - build system and conversation context
  |  - expose tools enabled for this chat
  |  - fit history and tool schemas to model budgets
  |  - execute model/tool-call turns
  |
  +--> Groq primary and fallback models
  +--> Gemini fallback, vision and native tool calling
  +--> Tool executors
          |-- Supabase memory, reminders and RAG
          |-- Web search and multi-step research
          |-- GitHub and OneDrive
          |-- Gmail draft/send workflow
          |-- Video metadata and transcription
          |-- Contacts, proactive WhatsApp messages and utilities
            |
            v
WhatsApp formatting, chunking and reply
```

### Main modules

| Module | Responsibility |
|---|---|
| `whatsapp_agent.py` | Neonize client, WhatsApp events, message/media handling, chat history, reminders, session persistence, health server and startup wiring. |
| `agent_loop.py` | Conversation context, memory injection, per-chat tool selection, provider-compatible tool calls and the agent loop. |
| `agent_tools.py` | Tool schemas, executor registration and integration implementations. |
| `llm_router.py` | Provider/model routing, TPM estimates, history trimming, cooldowns, retries and Gemini tool-call compatibility. |
| `llm_json.py` | Structured JSON LLM helper used by planning, writing, routing and extraction tasks. |
| `tool_budget.py` | Dynamic selection of a smaller tool set when the full catalog exceeds the request budget. |
| `knowledge_rag.py` | Chunking, embeddings, indexing, hybrid retrieval, chat-note synchronization and OneDrive knowledge ingestion. |
| `email_agent.py` / `email_ops.py` | Owner email-draft workflow, Gmail OAuth, resume resolution and email sending. |
| `file_ops.py` | Microsoft Graph/OneDrive file operations. |
| `github_ops.py` | GitHub repository/file reading and caching. |
| `research.py` / `web_search_providers.py` | Multi-step research and provider-fallback web search. |
| `voice_pipeline.py` | Audio transcription and transcript normalization. |
| `video_intent.py` / `yt_access.py` | Video request classification and YouTube/yt-dlp access helpers. |
| `usage_budget.py` | Persistent key/value cache and Supadata credit tracking. |
| `admin_commands.py` | Owner commands and Supabase-backed chat/feature/tool flags. |
| `resilience.py` / `mojo_logging.py` | Watchdog, incidents, diagnostics and logging pipeline. |
| `wa_format.py` / `bot_voice.py` / `language_policy.py` | WhatsApp output formatting, bilingual response strings and language policy. |

### LLM routing and tool budgets

The application configures the OpenAI-compatible SDK to use Groq for primary text generation and Gemini for fallback and multimodal tasks. The router tracks estimated and actual token usage, observes model cooldowns, trims older conversation history when that can make a request fit, and bounds retries and Gemini fallback time.

Gemini function calling is enabled for tool-selection turns. The agent loop preserves Gemini's opaque thought-signature metadata when replaying calls, strips provider-specific metadata before sending requests to Groq, and sanitizes tool schemas for Gemini's accepted subset. If Gemini rejects a tools request with a compatible 400 response, the router can retry that turn without tools and temporarily remember the incompatibility.

When the full tool catalog is too large for the configured Groq budget, `tool_budget.py` asks an LLM to choose plausible tools for the current turn. This planner narrows the offered schemas for that turn; it does not change the underlying chat permissions. A failed/invalid planner response falls back to the original tool list. Set `TOOL_PLANNER=0` to disable the planner and `GEMINI_TOOLS=0` to disable Gemini-native tool calling.

Model identifiers and free-tier limits are provider-controlled and may change. Check the model defaults in `whatsapp_agent.py` and `llm_router.py` before deployment.

## Agent tools

The current executor registers these tools:

| Tool | Purpose |
|---|---|
| `search_knowledge` | Search agency knowledge and knowledge available in the current chat's scope. |
| `save_to_knowledge` | Save a document or text into the knowledge index. |
| `forget_knowledge` | Remove a matching knowledge entry within the caller's permitted scope. |
| `get_memory` | Retrieve saved notes for the current chat. |
| `save_memory` | Save a durable note, preference or profile for the current chat. |
| `forget_memory` | Remove saved notes from the current chat. |
| `web_search` | Search for current facts and sources. |
| `deep_research` | Plan multiple web queries, gather evidence and produce a cited report. |
| `browse_url` | Fetch and read public URL content. |
| `get_weather` | Retrieve weather information for a location. |
| `set_reminder` | Create one-off or recurring reminders. |
| `list_reminders` | List active reminders in the current chat. |
| `cancel_reminders` | Cancel matching reminders. |
| `query_contacts` | Search known contacts by name fragment or phone. |
| `lookup_user` | Resolve details for a WhatsApp sender ID/LID. |
| `send_message_to` | Send a proactive WhatsApp message to a contact or group. |
| `file_list_onedrive` | List files under the OneDrive knowledge root. |
| `note_down` | Save or append a note to the owner's OneDrive journal. |
| `python_exec` | Run a limited pure-Python calculation/statement in the owner-only tool. |
| `image_describe` | Describe a supplied image. |
| `link_preview` | Retrieve supported public-link metadata. |
| `transcribe_video` | Retrieve spoken content from supported public video links. |
| `draft_email` | Create, update, show or cancel the owner's current email draft. |
| `send_email` | Send the current draft through Gmail after explicit owner confirmation. |

Tools are not automatically exposed to every chat. The owner enables the required tools through chat-level admin controls.

## Access control

Mojo has two layers of control:

1. **Owner administration:** `OWNER_SENDER_ID` identifies the owner. Admin slash commands are accepted only from that sender in a private one-to-one chat with the bot. They are not admin commands in a group.
2. **Per-chat feature and tool flags:** features and tools are controlled through Supabase-backed flags. Tool access defaults to off when no per-chat tool flag is present; the agent exposes only enabled tools.

Main chat features include:

- `ai_chat` — conversational agent and tool use.
- `reminders` — reminder handling.
- `chat_memory` — saving chat memory.
- `audio` — audio/voice-note handling.

Owner-only tools include the owner's OneDrive journal, restricted Python execution, proactive WhatsApp messaging and outbound email drafting/sending. Tool executors perform owner checks for these actions.

Use `/help` in the owner's private chat for the exact live command syntax.

## Requirements

- A WhatsApp account and phone with access to **Linked Devices**.
- Python and packages listed in `requirements.txt`. The Render Blueprint currently specifies Python `3.14.6`; using the same runtime locally is recommended.
- A Supabase project with the required tables, functions, extensions and Storage bucket.
- A Groq API key for configured primary text and speech-transcription paths.
- A Gemini API key for fallback generation, vision and the default embedding/search paths.
- A persistent local filesystem for local sessions, or configured Supabase Storage session backup for an ephemeral host.
- Optional credentials for OneDrive, Gmail, GitHub, search and transcript integrations you intend to use.

## Local installation

### 1. Clone the repository

```bash
git clone https://github.com/zaheerxai/whatsapp-agent.git
cd whatsapp-agent
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Create the environment file

Copy `.env.example` to `.env`.

macOS/Linux:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

At minimum, configure:

```dotenv
GROQ_API_KEY=...
GEMINI_API_KEY=...
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_KEY=...
OWNER_SENDER_ID=
```

`OWNER_SENDER_ID` can be discovered during the first local connection; see [Pairing WhatsApp](#pairing-whatsapp).

Use a server-side Supabase key with the database and Storage permissions the application needs. Never commit real credentials.

### 5. Review business context

`business_info.txt` is read at startup and included as business/agency context. Replace its example contents with information appropriate for your deployment. Do not put API keys, passwords, private customer messages or other secrets in this file.

### 6. Configure Supabase

Run the SQL scripts listed in [Supabase configuration](#supabase-configuration), provision the `chat_history` table described there, and create the session Storage bucket for hosted deployments.

### 7. Start Mojo

```bash
python whatsapp_agent.py
```

On Windows, `RUN_LOCALLY.bat` installs `requirements.txt` and starts the same entry point.

## Supabase configuration

The repository includes these SQL scripts. Run them in the Supabase SQL Editor using a database role permitted to create the required extensions, tables and functions.

| Order | Script | Purpose |
|---:|---|---|
| 1 | `sql/admin_schema.sql` | Global bot settings and per-chat feature flags. |
| 2 | `sql/reminders_schema.sql` | Contacts, group memory, reminders and chat reminder settings. |
| 3 | `sql/rag_schema.sql` | pgvector/pg_trgm extensions, RAG documents/chunks and initial retrieval functions. |
| 4 | `sql/rag_schema_v2.sql` | RAG v2 migration: corrected chat scoping, retrieval signals and atomic chunk replacement. Run after `rag_schema.sql`. |
| 5 | `sql/schema_kv_cache.sql` | Optional persistent key/value cache for transcript/metadata caches and usage-budget state. Recommended for ephemeral hosting. |

### `chat_history` table

The runtime reads and writes a `chat_history` table, but the repository does not currently include a SQL script that creates it. Provision a table with at least the columns used by the application:

```sql
create table if not exists public.chat_history (
    id bigint generated by default as identity primary key,
    chat_id text not null,
    sender_id text not null,
    sender_num text,
    role text not null,
    content text not null,
    created_at timestamptz not null default now()
);

create index if not exists chat_history_chat_created_idx
    on public.chat_history (chat_id, created_at desc);
```

This is a starter schema inferred from the current read/write code, not a migration shipped with the project. Review constraints, access policies and retention to suit your deployment. The server-side key must be able to read and insert the required rows.

### Supabase Storage

For a hosted deployment with an ephemeral filesystem, create a **private** Storage bucket named:

```text
wa-sessions
```

The application restores and backs up the linked-device session at `whatsapp_session.db` inside this bucket. The server-side Supabase key needs the required Storage permissions.

`sql/schema_kv_cache.sql` enables RLS on `kv_cache`. Its comments specify that the application uses a service-role key to bypass RLS. Keep that key private and server-side.

## Environment variables

The full example is in [`.env.example`](.env.example). The most relevant settings are summarized below.

### Core settings

| Variable | Required? | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Yes for the configured primary provider | Groq API key. |
| `GEMINI_API_KEY` | Strongly recommended | Gemini fallback, vision, embeddings and grounding/search paths. |
| `SUPABASE_URL` | Yes | Supabase project URL. |
| `SUPABASE_KEY` | Yes | Server-side key with the database/Storage permissions required by the app. |
| `OWNER_SENDER_ID` | Required for owner administration | Bare WhatsApp sender number/LID; do not append a server suffix. |
| `PORT` | Optional | Health server port; defaults to `10000`. |
| `RENDER_EXTERNAL_URL` | Hosted deployments | Render URL and one of the cloud-host detection signals. |
| `FORCE_CLOUD_SESSION` | Optional | Set to `1`/`true` to force cloud session mode. |
| `LOCAL_MODE` | Optional | Set to `1`/`true` to force local mode and prevent production session upload/download. |
| `SESSION_UPLOAD_SEC` | Optional | Cloud session backup interval in seconds; default `1800` (30 minutes). |
| `PRESENCE_KEEPALIVE_HOURS` | Optional | WhatsApp presence keepalive interval; default `6`. Set `0` to disable. |

### Web search, GitHub and video

| Variable | Required? | Purpose |
|---|---|---|
| `GITHUB_TOKEN` | Optional | Authenticated GitHub API requests and higher rate limits. |
| `TAVILY_API_KEY` | Optional | Tavily search provider. |
| `TAVILY_KEYLESS` | Optional | Enables/disables Tavily keyless mode; defaults to enabled in code. |
| `SERPER_API_KEY` | Optional | Serper search provider. |
| `DDGS_BACKENDS` | Optional | Backends used by the `ddgs` fallback. |
| `GEMINI_SEARCH_MODEL` | Optional | Overrides model used by Gemini web grounding. |
| `SUPADATA_API_KEY` | Optional, metered | Supadata metadata/transcription fallback. Check its current quota and pricing. |
| `YOUTUBE2TEXT_API_KEY` | Optional | Additional YouTube transcript fallback; check current quota and pricing. |
| `YTDLP_COOKIES` | Optional | Path to a Netscape-format cookies file for yt-dlp. Treat it as a secret. |
| `YTDLP_JS_RUNTIME` | Optional | Restricts yt-dlp JavaScript runtime selection to a named installed runtime. |
| `YT_PROXY` | Optional | Proxy used by supported YouTube clients. |
| `YTDLP_SUBS_DEADLINE_S` | Optional | yt-dlp subtitle-attempt time budget; default `45` seconds. |

### OneDrive and Gmail

| Variable | Required? | Purpose |
|---|---|---|
| `ONEDRIVE_CLIENT_ID` | Optional | Microsoft app/client ID. |
| `ONEDRIVE_CLIENT_SECRET` | As required by the OAuth app type | Microsoft confidential-client secret. |
| `ONEDRIVE_REFRESH_TOKEN` | Optional | Microsoft OAuth refresh token. |
| `ONEDRIVE_FOLDER` | Optional | OneDrive folder for logs; example `AgentLogs`. |
| `KNOWLEDGE_ONEDRIVE_ROOT` | Optional | Root for knowledge ingestion; code default `Documents/aimojo`. |
| `GMAIL_CLIENT_ID` | Optional | Google OAuth client ID for outbound Gmail. |
| `GMAIL_CLIENT_SECRET` | Optional | Google OAuth client secret. |
| `GMAIL_REFRESH_TOKEN` | Optional | Gmail OAuth refresh token with required send scope. |
| `GMAIL_SENDER` | Optional | Sender email address. |
| `GMAIL_FROM_NAME` / `EMAIL_SIGN_NAME` | Optional | Display/signature name. |
| `RESUME_PDF_PATH` | Optional | Local resume PDF path. |
| `RESUME_ONEDRIVE_PATH` | Optional | Alternative OneDrive resume path. |
| `RESUME_FILENAME` | Optional | Attachment filename. |
| `EMAIL_WRITER_MODELS` | Optional | Comma-separated preferred models for job-application writing and fact checks. |

OAuth helper scripts are in `scripts/get_onedrive_token.py` and `scripts/get_gmail_refresh_token.py`.

### RAG, model routing and advanced settings

| Variable | Default / behavior | Purpose |
|---|---|---|
| `EMBEDDING_PROVIDER` | `gemini` | Selects Gemini or OpenAI embeddings. |
| `EMBEDDING_MODEL` | `gemini-embedding-001` for Gemini; `text-embedding-3-small` for OpenAI | Embedding model. |
| `EMBEDDING_DIM` | `768` | Must match the Supabase vector schema. |
| `OPENAI_API_KEY` | Optional | Required if using OpenAI embeddings. |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `1200` / `150` | Document chunk target and overlap in characters. |
| `RAG_MAX_CHUNKS_PER_DOC` | `400` | Maximum indexed chunks per document. |
| `RAG_MIN_SIM` | `0.50` | Semantic relevance threshold. |
| `RAG_RERANK` | Off by default | Enables optional LLM reranking. |
| `RAG_AUTO_SYNC` | On by default | Enables background knowledge synchronization. |
| `RAG_SYNC_MIN` | `180` | Sync interval in minutes; `0` means boot sync only. |
| `TOOL_PLANNER` | Enabled unless `0` | Disables dynamic tool selection when set to `0`. |
| `TOOL_PLANNER_MODELS` | Code-defined fallback list | Comma-separated planner model preference. |
| `GEMINI_TOOLS` | Enabled unless `0` | Disables Gemini-native function calling when set to `0`. |
| `GEMINI_FALLBACK_MODELS` | Code-defined model ladder | Ordered Gemini fallback model list. |
| `LLM_GEMINI_DEADLINE_S` | `25` | Total time budget for Gemini fallback routing. |
| `GROQ_TPM_LIMIT` | `8000` | Local per-model TPM budget estimate; actual provider limits vary. |
| `VOICE_GROUP_WAKEWORD` | Off by default | Allows group voice-note transcription/intent checks without a direct mention/quote when enabled. |
| `VOICE_MAX_SECONDS` | `600` | Maximum accepted voice-note duration. |
| `VOICE_WHISPER_MODELS` | `whisper-large-v3,whisper-large-v3-turbo` | Ordered Whisper model list. |
| `VOICE_LLM_MODELS` | Code-defined side-model list | Models used for voice transcript normalization. |
| `VOICE_ALWAYS_NORMALIZE` | Off by default | Always run transcript normalization. |
| `SUPADATA_OWNER_RESERVE` | `20` | Credit reserve held for owner requests. |
| `GITHUB_MAX_ZIP_MB` | `25` | Maximum repository archive size accepted by the GitHub overview path. |
| `DIAG_TOKEN` | Unset = diagnostics disabled | Enables out-of-band `/diag` and `/diag/restart`; use a long random secret. |
| `GUARD_SELF_HEAL` | Off unless enabled | Lets the watchdog exit after repeated logging-pipeline failures so the host can restart the process. |
| `GUARD_STUCK_AFTER_S` | Code default | Threshold for reporting a stuck handler. |
| `GUARD_INCIDENT_COOLDOWN_S` | Code default | Rate limit for repeated incident notifications. |

**Embedding migration note:** `.env.example` still contains the older `text-embedding-004` value. The current `knowledge_rag.py` recognizes this as retired and remaps it to `gemini-embedding-001`. Leave `EMBEDDING_MODEL` unset or set it to `gemini-embedding-001`; keep `EMBEDDING_DIM=768` to match the SQL vector schema.

**Session database note:** `.env.example` contains older `SESSION_DB_URL` guidance, but current startup creates Neonize's client with the resolved local `whatsapp_session.db` path and does not pass `SESSION_DB_URL` to it. Do not rely on that variable for session persistence. Cloud session restore/backup is currently implemented through Supabase Storage.

## Pairing WhatsApp

1. Run `python whatsapp_agent.py`.
2. Wait for the QR code in the terminal.
3. On your phone, open **WhatsApp → Linked Devices → Link a Device** and scan the QR code.
4. Send the bot a private message from the account that should administer it.
5. If `OWNER_SENDER_ID` is not configured, inspect the log for the private-message sender identifier (`[PRIVATE MESSAGE] from ...`).
6. Set that value as `OWNER_SENDER_ID` without a WhatsApp server suffix, then restart the process.
7. In the owner's private chat, use `/help` and enable only the features/tools required for each chat.

Local and cloud sessions are deliberately separated. Local mode uses a local `whatsapp_session.db` and avoids uploading or replacing the production session in Supabase Storage. Cloud mode attempts to restore the session from Storage and backs up changes. Do not run a local development instance in cloud mode against the same live session unless that is intentional.

## Render deployment

A Render Blueprint is included in [`render.yaml`](render.yaml).

1. Push the repository to GitHub and create a Render Blueprint/Web Service from it.
2. Review the service name, region and plan in `render.yaml`.
3. Configure the required environment variables in Render: `GROQ_API_KEY`, `GEMINI_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY` and `OWNER_SENDER_ID`. Set `DIAG_TOKEN` if out-of-band diagnostics are needed.
4. Create the private `wa-sessions` Supabase Storage bucket and grant the server-side Supabase key access.
5. Apply the Supabase SQL setup, including the separately provisioned `chat_history` table.
6. Deploy and inspect the service logs. Pair WhatsApp if no valid session was restored.
7. Set `RENDER_EXTERNAL_URL` if required by the deployment; Render also provides environment markers used by cloud-session detection.

The current Blueprint uses the Render **free** service plan and runs `python whatsapp_agent.py` with `/health` as the health-check path. Free hosted instances may sleep/restart and have ephemeral local filesystems. Supabase Storage backup protects the linked-device session across filesystem loss, but it does not keep the service continuously running. A sleeping service can delay incoming-message handling until it wakes. Use an always-on host with appropriate persistent storage for latency-sensitive production workloads.

### Health and diagnostics endpoints

- `GET /`, `GET /health` and `GET /healthz` return a plain-text `ok` response with a best-effort running Git SHA.
- `HEAD /health` is supported for uptime monitors.
- When `DIAG_TOKEN` is configured, `GET /diag` returns a diagnostic snapshot containing log/thread/incident details.
- When `DIAG_TOKEN` is configured, `GET /diag/restart` requests a process restart.

Diagnostics are disabled when `DIAG_TOKEN` is unset. Protect this token like a password. Prefer the `X-Diag-Token` request header instead of a URL query parameter because URLs may be captured in logs and monitoring systems.

## Owner commands

The current registered commands include:

| Command | Purpose |
|---|---|
| `/help` | Show available admin commands and usage. |
| `/enable` | Enable the agent or a feature/tool for a scope. |
| `/disable` | Disable the agent or a feature/tool for a scope. |
| `/status` | Show bot and feature status. |
| `/chats` | List known chats and their identifiers. |
| `/exportlog` | Export recent chat log content. |
| `/uploadsession` | Force a WhatsApp session backup. |
| `/sessionstatus` | Show local/cloud session and backup status. |
| `/uploadlog` | Manually upload the current log. |
| `/logstatus` | Show current log status. |
| `/uploadimg` | Upload an image to OneDrive. |
| `/ingest` | Ingest supported knowledge sources. |
| `/kbstatus` | Show knowledge-base indexing status. |
| `/kbsearch` | Search the knowledge base for diagnostics. |
| `/credits` | Show tracked external transcript-credit usage. |
| `/tail` | Show the end of the live log. |
| `/threads` | Inspect running handlers and thread stacks. |
| `/incidents` | Show recorded crashes/hangs and guard status. |

Use `/help` in the owner's private chat for the exact live syntax and argument formats.

## Tests

The repository contains offline, script-style tests under `tests/`. They use fakes and mocks for external services. Coverage includes:

- Admin lane, feature flags and tool access.
- LLM routing, Gemini function calls and dynamic tool budgeting.
- Email drafting, validation, sending and failure handling.
- RAG ingestion, retrieval, scoping and caches.
- GitHub operations, web research and usage budgets.
- Voice pipeline and WhatsApp formatting.
- Video intent, transcript fallback and yt-dlp access helpers.
- Logging deadlock regression and process resilience.

Run all test scripts from the repository root.

macOS/Linux:

```bash
for test_file in tests/test_*.py; do
  echo "== $test_file"
  python "$test_file" || exit 1
done
```

Windows PowerShell:

```powershell
Get-ChildItem tests\test_*.py | ForEach-Object {
    Write-Host "== $($_.Name)"
    python $_.FullName
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
```

These scripts are offline tests, not a substitute for live WhatsApp pairing or integration checks against your own Supabase, Microsoft, Google and model-provider accounts. The repository does not currently include a GitHub Actions workflow.

## Operations and troubleshooting

- **`/uploadlog` is manual.** The timestamp of the most recent log upload records when the owner last invoked the upload command. It is not an uptime signal and does not mean Mojo has been offline since that time. Use `/health`, host service status and diagnostic commands to assess runtime health.
- **WhatsApp session persistence:** local development uses a local session file. Cloud mode restores/backups it through Supabase Storage. Avoid running local development in cloud mode against the production session unless intended.
- **OneDrive:** log upload and knowledge ingestion are separate operations. OneDrive operations require valid Microsoft OAuth configuration and appropriate permissions.
- **RAG:** run `rag_schema_v2.sql` after `rag_schema.sql`. The v2 migration contains the current chat-scoping and atomic chunk replacement behavior.
- **Transcript services:** YouTube tries available direct/free sources before metered providers. Optional services may enforce quotas, credits or charges. Check provider terms and monitor `/credits`.
- **Provider failures:** fallback and circuit-breaker logic reduce repeated failures but cannot guarantee service when all configured providers are unavailable or rate-limited.
- **Persistence:** Render's local filesystem is ephemeral. Supabase is required for chat/application data and cloud session restoration. The optional `kv_cache` table keeps cached results across process restarts.
- **Runtime:** `render.yaml` is the deployment source of truth for the configured Python version, build command, start command and health-check path. Keep it synchronized with this README when changing deployment settings.

## Repository structure

```text
.
├── whatsapp_agent.py       # WhatsApp client, events, media, scheduler, health server
├── agent_loop.py           # Conversation context and LLM-driven agent loop
├── agent_tools.py          # Tool schemas, executors and integrations
├── llm_router.py            # Provider routing, budgets, retries and failover
├── llm_json.py              # Structured JSON LLM helper
├── tool_budget.py           # Dynamic tool selection for oversized tool catalogs
├── knowledge_rag.py         # Chunking, embeddings, hybrid retrieval and ingestion
├── email_agent.py           # Owner email-draft workflow
├── email_ops.py             # Gmail OAuth, resume resolution and sending
├── file_ops.py              # OneDrive file operations
├── github_ops.py             # GitHub repository/file reading and caching
├── research.py               # Multi-query research and cited report assembly
├── web_search_providers.py  # Tavily, Serper, Gemini grounding and DDGS
├── voice_pipeline.py        # Whisper transcription and transcript normalization
├── video_intent.py          # Video-link intent planning
├── yt_access.py             # yt-dlp cookies, runtimes, proxy and source health
├── usage_budget.py          # Persistent cache and transcript usage budget
├── admin_commands.py        # Owner commands and feature/tool flags
├── resilience.py            # Watchdog, incidents and diagnostic helpers
├── mojo_logging.py          # File logging, rotation and manual upload
├── bot_voice.py             # English/Roman Urdu response catalogue
├── language_policy.py       # Language detection/output policy
├── wa_format.py              # WhatsApp-compatible formatting
├── business_info.txt        # Business context loaded at startup
├── requirements.txt        # Python dependencies
├── .env.example             # Environment-variable reference
├── render.yaml              # Render Blueprint
├── scripts/                 # Microsoft/Google OAuth token helpers
├── sql/                     # Supabase schemas and migrations
└── tests/                   # Offline test scripts
```

## Security and privacy

Mojo may process private chat messages, sender identifiers, contacts, uploaded files, voice recordings/transcripts, email content and business documents. Before using it with other people or real customer data:

- Keep `.env`, API keys, OAuth refresh tokens, cookies files and Supabase service-role keys out of Git.
- Use a private Supabase Storage bucket for linked-device session backups.
- Restrict `OWNER_SENDER_ID` to the intended administrator.
- Enable only required features and tools in each chat.
- Be deliberate about which external AI/search/transcription providers receive text, documents, images or audio. Provider retention and training policies vary by service and plan.
- Restrict diagnostic endpoints and rotate `DIAG_TOKEN` if it is exposed.
- Configure database access policies and retention/deletion rules for chat history, memories, transcripts and uploaded files.
- Treat transcripts, documents and retrieved content as untrusted input rather than trusted instructions.
- Review the applicable WhatsApp, Google, Microsoft and provider terms before operating this as a public or commercial service.

## License

There is currently no `LICENSE` file at the repository root. Until a license is added, do not assume that the public repository grants permission to redistribute or reuse its code.
