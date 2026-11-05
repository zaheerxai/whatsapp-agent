# Mojo — WhatsApp AI Agent

Mojo is a Python-based AI assistant that runs through a WhatsApp linked device. It combines an LLM-driven agent loop with chat history, scoped memory, reminders, document and media processing, web research, GitHub and OneDrive integrations, Gmail workflows, and operational diagnostics.

The WhatsApp connection is handled by [Neonize](https://github.com/krypton-byte/neonize). Supabase provides persistent application data and, on hosts with ephemeral filesystems, storage for linked-device session backups.

> This README describes the implementation in the repository. Features may require third-party credentials, database migrations, provider quotas, or external account setup. Implementation does not imply that every integration is configured in every deployment.

## Contents

- [Capabilities](#capabilities)
- [Architecture](#architecture)
- [Agent tools](#agent-tools)
- [Access control](#access-control)
- [Requirements and setup](#requirements-and-setup)
- [Environment variables](#environment-variables)
- [Supabase setup](#supabase-setup)
- [WhatsApp pairing](#whatsapp-pairing)
- [Deployment](#deployment)
- [Owner commands](#owner-commands)
- [Tests](#tests)
- [Operations and troubleshooting](#operations-and-troubleshooting)
- [Project structure](#project-structure)
- [Security](#security)
- [License](#license)

## Capabilities

| Area | Implementation |
|---|---|
| WhatsApp | Private and group chat handling, mentions, quoted messages, contact resolution, duplicate-message protection and long-response chunking. |
| Agent orchestration | LLM-driven tool calling, per-chat tool allowlists, recent conversation context and memory injection. |
| LLM routing | Groq primary generation with Gemini fallback, token-budget estimates, history trimming, model cooldowns, bounded retries and response continuation. |
| Tool planning | Dynamic selection of a smaller tool set when the complete tool catalog exceeds the configured request budget. |
| Memory | Save, retrieve and forget chat-scoped notes; notes can also be indexed for retrieval. |
| Knowledge base | Document chunking, embeddings, Supabase pgvector storage, hybrid lexical/vector retrieval, relevance gating, optional reranking and chat scoping. |
| Reminders | One-off and recurring reminders, selected weekdays, cancellation and sender timezone handling. |
| Web research | Provider-fallback search and multi-query research that reads selected sources and assembles cited reports. |
| GitHub | Read public repository overviews, README content and files; an optional token improves API rate limits. |
| Gmail | Owner-only email drafting, editing, preview, cancellation and sending after explicit confirmation. |
| OneDrive | File listing and transfer, knowledge-document ingestion, log export and owner notes. |
| Documents and images | PDF/Office document extraction and vision-model workflows for image description and scanned-document extraction. |
| Audio and video | Whisper transcription, transcript normalization and supported public video/social-link workflows. |
| Operations | Queue-based logging, log rotation, watchdog, incident reporting, stuck-handler diagnostics, health checks and optional token-protected diagnostics. |

Provider availability, model identifiers, rate limits and third-party terms can change. Fallbacks reduce single-provider dependency but cannot guarantee availability when all configured services are unavailable or quota-limited.

## Architecture

```text
WhatsApp linked device (Neonize)
            |
            v
Message event / duplicate guard
            |
            +---- Owner command fast lane
            |
            v
Message processing
  |         |          |
  |         |          +--> Images / PDFs / Office documents
  |         +-------------> Voice transcription and normalization
  +-----------------------> Chat history, contacts and scoped memory
            |
            v
Agent loop
  |-- Build system and conversation context
  |-- Expose tools enabled for this chat
  |-- Fit history and tool schemas to model budgets
  |-- Execute model/tool-call turns
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

### Core modules

| Module | Responsibility |
|---|---|
| `whatsapp_agent.py` | Neonize client, WhatsApp events, message/media handling, chat history, reminders, session persistence, health server and startup wiring. |
| `agent_loop.py` | Conversation context, memory injection, per-chat tool selection, provider-compatible tool calls and agent loop. |
| `agent_tools.py` | Tool schemas, executor registration and integration implementations. |
| `llm_router.py` | Provider/model routing, token estimates, history trimming, cooldowns, retries and Gemini compatibility. |
| `llm_json.py` | Structured JSON LLM helper used by planning, writing, routing and extraction. |
| `tool_budget.py` | Dynamic tool selection when the full catalog exceeds the request budget. |
| `knowledge_rag.py` | Chunking, embeddings, indexing, hybrid retrieval and knowledge ingestion. |
| `email_agent.py`, `email_ops.py` | Owner email-draft workflow, Gmail OAuth, resume resolution and sending. |
| `file_ops.py`, `github_ops.py` | OneDrive file operations and GitHub repository/file reading. |
| `research.py`, `web_search_providers.py` | Multi-step research and provider-fallback web search. |
| `voice_pipeline.py` | Audio transcription and transcript normalization. |
| `video_intent.py`, `yt_access.py` | Video request classification and YouTube/yt-dlp access helpers. |
| `admin_commands.py` | Owner commands and Supabase-backed chat/feature/tool flags. |
| `resilience.py`, `mojo_logging.py` | Watchdog, incidents, diagnostics and logging pipeline. |

The LLM router uses Groq for primary text generation and Gemini for fallback and multimodal tasks. Gemini function calling is supported on tool-selection turns, including preservation of provider-specific thought-signature metadata. The tool planner narrows the offered schemas for a turn; it does not change chat permissions. Set `TOOL_PLANNER=0` to disable planning and `GEMINI_TOOLS=0` to disable Gemini-native tool calling.

## Agent tools

The executor currently registers tools for:

- **Knowledge and memory:** `search_knowledge`, `save_to_knowledge`, `forget_knowledge`, `get_memory`, `save_memory`, `forget_memory`.
- **Research:** `web_search`, `deep_research`, `browse_url`.
- **Reminders:** `set_reminder`, `list_reminders`, `cancel_reminders`.
- **Contacts and messaging:** `query_contacts`, `lookup_user`, `send_message_to`.
- **Microsoft files:** `file_list_onedrive` and related file/knowledge operations.
- **GitHub:** repository and file lookup operations.
- **Email:** owner-scoped Gmail draft, preview, edit, cancel and send operations.
- **Utilities and media:** weather, document/image/audio/video processing and related helpers.

The live tool schemas and chat allowlists in the code are the source of truth; available tools can differ by chat and configuration.

## Access control

Mojo has an owner/admin fast lane and chat-scoped feature and tool controls. The owner identity is configured with `OWNER_SENDER_ID`. Admin commands manage whether the agent, features and individual tools are enabled for a chat. Owner-only operations include sensitive email and operational workflows.

Use `/help` in the owner's private chat for the exact command syntax. Do not assume a tool is available in every chat simply because it is registered in the application.

## Requirements and setup

- Python version: follow the version configured in `render.yaml`.
- A WhatsApp account for linked-device pairing.
- Supabase project and server-side key.
- Groq API key; Gemini API key for fallback/multimodal features.
- Optional Microsoft, Google/Gmail, GitHub and web-search provider credentials depending on the integrations you enable.

### Local installation

1. Clone the repository and enter its root directory.
2. Create and activate a virtual environment.
3. Install dependencies.
4. Copy `.env.example` to `.env` and fill in the required values.
5. Configure the Supabase database and Storage bucket as described below.
6. Start the application and complete WhatsApp pairing.

macOS/Linux:

```bash
git clone https://github.com/zaheerxai/whatsapp-agent.git
cd whatsapp-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python whatsapp_agent.py
```

Windows PowerShell:

```powershell
git clone https://github.com/zaheerxai/whatsapp-agent.git
cd whatsapp-agent
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python whatsapp_agent.py
```

Review `.env.example` and the startup configuration before running. Do not commit the populated `.env` file.

## Environment variables

The exact variables and defaults are documented in [`.env.example`](.env.example). The main configuration groups include:

| Group | Examples / purpose |
|---|---|
| Required core | `GROQ_API_KEY`, `GEMINI_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`, `OWNER_SENDER_ID`. |
| Model routing | Provider/model selection, fallback settings, token budgets and tool-planner toggles. |
| Knowledge/RAG | Embedding configuration, database tables, retrieval and reranking settings. |
| Microsoft / Gmail | OAuth client configuration, refresh tokens and Graph/Gmail access. |
| Search / transcription | Optional search-provider keys and transcript-service configuration. |
| Operations | `DIAG_TOKEN`, session mode, logging and health/diagnostic configuration. |

Important configuration notes from the current code:

- The embedding code maps the configured embedding model to `gemini-embedding-001` and expects 768-dimensional embeddings. Verify the model and dimensions together before changing either.
- `SESSION_DB_URL` appears in the example configuration but is not the active Neonize session persistence mechanism. The application uses its configured local session path and cloud Supabase Storage backup logic.
- Never expose service-role keys, OAuth refresh tokens, cookies or diagnostic tokens in logs or source control.

## Supabase setup

The repository includes SQL schemas and migrations under `sql/`. Apply the base RAG schema and then its versioned migration in order:

1. Apply `sql/rag_schema.sql`.
2. Apply `sql/rag_schema_v2.sql`.
3. Apply any other schema files required by the enabled features.
4. Create a **private** Supabase Storage bucket named `wa-sessions` for cloud WhatsApp session backup.
5. Ensure the server-side Supabase key has only the permissions required by the application.

**Chat history setup:** the repository code expects a `chat_history` table, but the repository does not ship a complete standalone schema for that table. Create and validate the table against the columns queried/inserted by the current application code before deploying. Do not assume that applying only the RAG migrations creates every application table.

The optional `kv_cache` table is used for persistent cached values where configured. Check the SQL files and actual code paths before adding or changing database constraints.

## WhatsApp pairing

On first startup, Neonize needs to link the application as a WhatsApp companion device. Follow the pairing instructions emitted by the running process and approve the linked device from WhatsApp on the primary phone. Keep the session database private.

Local and cloud sessions are deliberately separated:

- **Local mode** uses the local Neonize session database and avoids replacing the production session in Supabase Storage.
- **Cloud mode** attempts to restore the session from Storage and backs up changes.

Do not run a local development instance in cloud mode against the same live session unless that is intentional. Avoid sharing session database files or pairing credentials.

## Deployment

A Render Blueprint is included in [`render.yaml`](render.yaml).

1. Push the repository to GitHub and create a Render Blueprint/Web Service from it.
2. Review the service name, region and plan in `render.yaml`.
3. Configure the required environment variables in Render.
4. Create the private `wa-sessions` Storage bucket and apply the database setup.
5. Deploy and inspect logs. Pair WhatsApp if no valid session was restored.
6. Set optional diagnostic configuration only if you need out-of-band diagnostics.

The current Blueprint uses Render's **free** service plan and runs `python whatsapp_agent.py`, with `/health` as its health-check path. Free hosted instances can sleep/restart and have ephemeral local filesystems. Supabase Storage backup helps restore the linked-device session after filesystem loss, but it does not keep the service continuously running. A sleeping service can delay incoming-message handling until it wakes. Use an always-on host and suitable persistent storage for latency-sensitive production workloads.

### Health and diagnostics endpoints

- `GET /`, `GET /health` and `GET /healthz` return a plain-text `ok` response with a best-effort running Git SHA.
- `HEAD /health` is supported for uptime monitors.
- When `DIAG_TOKEN` is configured, `GET /diag` returns a diagnostic snapshot.
- When `DIAG_TOKEN` is configured, `GET /diag/restart` requests a process restart.

Diagnostics are disabled when `DIAG_TOKEN` is unset. Protect this token like a password. Prefer the `X-Diag-Token` request header over a URL query parameter because URLs may be captured in logs.

## Owner commands

The registered command set includes:

| Command | Purpose |
|---|---|
| `/help` | Show available admin commands and usage. |
| `/enable`, `/disable` | Enable or disable the agent, a feature or a tool for a scope. |
| `/status` | Show bot and feature status. |
| `/chats` | List known chats and identifiers. |
| `/exportlog` | Export recent chat log content. |
| `/uploadsession`, `/sessionstatus` | Back up or inspect the WhatsApp session. |
| `/uploadlog`, `/logstatus` | Manually upload the current log or inspect log status. |
| `/uploadimg` | Upload an image to OneDrive. |
| `/ingest`, `/kbstatus`, `/kbsearch` | Ingest and inspect/search knowledge sources. |
| `/credits` | Show tracked external transcript-credit usage. |
| `/tail`, `/threads`, `/incidents` | Inspect logs, running handlers and recorded incidents. |

Use `/help` for exact live syntax and argument formats.

## Tests

The repository contains offline, script-style tests under `tests/`. They use fakes and mocks for external services. Coverage includes admin commands and access flags, LLM routing and tool budgets, email workflows, RAG, GitHub operations, web research, usage budgets, voice processing, WhatsApp formatting, video intent and resilience/logging regressions.

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

These scripts are offline tests, not a substitute for live integration checks against your own WhatsApp, Supabase, Microsoft, Google and model-provider accounts. The repository does not currently include a GitHub Actions workflow.

## Operations and troubleshooting

- **Log uploads are manual.** Mojo's log-upload process runs only when the owner invokes `/uploadlog`. The timestamp of the most recent upload records when the owner last uploaded the logs; it is **not** an uptime/health signal and does not mean Mojo has been offline since that time. Use `/health`, host service status and diagnostic commands to assess runtime health.
- **WhatsApp session persistence:** local development uses a local session file. Cloud mode restores/backups it through Supabase Storage. Avoid running local development in cloud mode against the production session unless intended.
- **OneDrive:** log upload and knowledge ingestion are separate operations. OneDrive operations require valid Microsoft OAuth configuration and appropriate permissions.
- **RAG:** apply `rag_schema_v2.sql` after `rag_schema.sql`. The v2 migration contains current chat-scoping and atomic chunk-replacement behavior.
- **Transcript services:** YouTube attempts available direct/free sources before metered providers. Optional services may enforce quotas, credits or charges. Check provider terms and monitor `/credits`.
- **Provider failures:** fallback and circuit-breaker logic reduce repeated failures but cannot guarantee service when all configured providers are unavailable or rate-limited.
- **Persistence:** Render's local filesystem is ephemeral. Supabase is required for chat/application data and cloud session restoration.
- **Runtime:** `render.yaml` is the deployment source of truth for the configured Python version, build command, start command and health-check path. Keep it synchronized with this README.

## Project structure

```text
.
├── whatsapp_agent.py       # WhatsApp client, events, media, scheduler and health server
├── agent_loop.py            # Conversation context and LLM-driven agent loop
├── agent_tools.py            # Tool schemas, executors and integrations
├── llm_router.py             # Provider routing, budgets, retries and failover
├── llm_json.py               # Structured JSON LLM helper
├── tool_budget.py            # Dynamic tool selection
├── knowledge_rag.py          # Chunking, embeddings, hybrid retrieval and ingestion
├── email_agent.py            # Owner email-draft workflow
├── email_ops.py              # Gmail OAuth and sending
├── file_ops.py               # OneDrive file operations
├── github_ops.py             # GitHub repository/file reading
├── research.py               # Multi-query research and cited reports
├── web_search_providers.py   # Search-provider fallbacks
├── voice_pipeline.py         # Whisper transcription and normalization
├── video_intent.py           # Video-link intent planning
├── yt_access.py              # YouTube/yt-dlp access helpers
├── usage_budget.py           # Persistent cache and usage budget
├── admin_commands.py         # Owner commands and feature/tool flags
├── resilience.py             # Watchdog, incidents and diagnostics
├── mojo_logging.py            # File logging, rotation and manual upload
├── bot_voice.py               # English/Roman Urdu response catalogue
├── language_policy.py         # Language detection/output policy
├── wa_format.py               # WhatsApp-compatible formatting
├── business_info.txt          # Business context loaded at startup
├── requirements.txt          # Python dependencies
├── .env.example               # Environment-variable reference
├── render.yaml                # Render Blueprint
├── scripts/                   # OAuth/token helper scripts
├── sql/                       # Supabase schemas and migrations
└── tests/                     # Offline test scripts
```

## Security

Mojo may process private chat messages, sender identifiers, contacts, uploaded files, voice recordings/transcripts, email content and business documents. Before using it with other people or customer data:

- Keep `.env`, API keys, OAuth refresh tokens, cookies and Supabase service-role keys out of Git.
- Use a private Storage bucket for linked-device session backups.
- Restrict `OWNER_SENDER_ID` to the intended administrator.
- Enable only the features and tools required in each chat.
- Review which external AI/search/transcription providers receive messages, documents, images or audio; retention policies vary by provider and plan.
- Restrict diagnostic endpoints and rotate `DIAG_TOKEN` if exposed.
- Configure database access policies and retention/deletion rules for chat history, memories, transcripts and uploaded files.
- Treat transcripts, documents and retrieved content as untrusted input rather than trusted instructions.
- Review applicable WhatsApp, Google, Microsoft and provider terms before operating this as a public or commercial service.

## License

There is currently no `LICENSE` file at the repository root. Until a license is added, do not assume that the public repository grants permission to redistribute or reuse its code.
