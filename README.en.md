# Enterprise Knowledge Operations Agent

[简体中文](README.md) | [English](README.en.md)

A local demo project for AI Agent, RAG, and LLM application development. The default entry is an enterprise knowledge operations Agent: the model first decides whether a tool is needed, answers ordinary conversation directly, and invokes the RAG tool for enterprise-material questions. It cites retrieved sources, pauses write operations for human approval, and resumes from a PostgreSQL LangGraph checkpoint after a service restart. The system also provides document ingestion, Chroma retrieval, hybrid retrieval, Dynamic RAG, and RAGAS evaluation.

This is a runnable demo prototype, not a production system. It does not claim real company deployment, user scale, revenue, or online traffic.

## Tech Stack

- Python, FastAPI, Pydantic, Uvicorn
- LangChain, LangGraph, Dynamic RAG, Tool
- DeepSeek Chat and Qwen `text-embedding-v4`
- Alibaba Cloud Model Studio OpenAI-compatible Embedding API
- MinerU official API token workflow
- Chroma vector database and PostgreSQL metadata storage
- RAGAS with DeepSeek as judge model
- Native HTML / CSS / JavaScript frontend
- uv for dependency and environment management

## Architecture

```mermaid
flowchart LR
    USER[User / Web console] --> API[FastAPI]
    API --> AGENT[LangGraph Agent]
    AGENT --> MODEL{DeepSeek task decision}
    MODEL -->|No enterprise evidence needed| DIRECT[Direct answer]
    MODEL -->|Tool required| TOOLS[ToolNode]
    TOOLS --> MODEL
    TOOLS --> APPROVAL{Write approval}
    APPROVAL --> EFFECT[Retry / reindex / delete]
    TOOLS -->|search_knowledge_base| RETRIEVE[Dense / BM25 / RRF]
    RETRIEVE --> MODEL
    RETRIEVE --> CONF{Evidence confidence}
    CONF -->|Low and web enabled| WEB[Tavily]
    WEB --> MODEL
    CONF -->|Low and web disabled| REFUSE[Safe refusal]
    AGENT --> CHECKPOINT[(PostgreSQL checkpoint)]
    API --> INGEST[Ingestion pipeline]
    INGEST --> PARSE[MinerU / pypdf]
    PARSE --> SPLIT[Markdown + recursive split]
    SPLIT --> EMB[Qwen embedding]
    EMB --> CHROMA[(Chroma)]
    CHROMA --> RETRIEVE
```

## Latest Agent Capabilities

- **On-demand RAG routing:** `search_knowledge_base` is an Agent tool. Greetings, thanks, and requests that do not depend on enterprise material can finish without embedding or vector retrieval.
- **Evidence guard:** when a user explicitly requests knowledge-base, uploaded-document, or internal evidence, LangGraph supplies one controlled retrieval if the model omitted it and refuses unsupported answers when evidence is unavailable.
- **Recoverable execution trace:** SSE and PostgreSQL checkpoints retain safe LLM-stage summaries, tool activity, sources, and approval state across page refreshes and FastAPI restarts.
- **Controlled web fallback:** Tavily is available only when the workbench toggle is enabled and knowledge-base confidence is low.
- **Conversation lifecycle:** Agent threads can be created, switched, restored, and deleted; running or approval-pending threads are protected from deletion.

The runtime uses PostgreSQL for document metadata, upload batches, fingerprints,
legacy chat history, and LangGraph checkpoints. Chroma and uploaded/parsed files
remain on the local `data/` volume. Qwen, DeepSeek, MinerU, and Tavily are cloud
APIs; no local model download is required.

## Features

- Upload `txt`, `md`, `pdf`, Office documents, images, and HTML files.
- Parse complex documents through MinerU and convert them to Markdown.
- Split Markdown by headers first, then recursively split long sections.
- Generate embeddings with Qwen and persist vectors in Chroma.
- Retrieve top-k chunks through a Chroma retriever with metadata and score.
- Fuse dense and BM25 rankings with RRF, and filter retrieval by document ID or filename.
- Support standard 2-step RAG and SSE streaming responses.
- Keep simple conversation history with `thread_id`.
- Support an explicit Dynamic RAG workflow with query rewrite, HyDE, context sufficiency checks, and context injection.
- Evaluate answers with RAGAS and cache generated responses/contexts to reduce repeated token cost.
- Provide a local web console for upload, chat, retrieval debugging, and evaluation.
- Support single-document deletion, reindexing, request IDs, stage timings, and Docker Compose deployment.
- Use a `MessagesState` + `ToolNode` loop with a six-call safety limit and PostgreSQL checkpoint persistence.
- Provide a Web Search toggle: the model still selects tools on demand, while Tavily is allowed only after low-confidence knowledge retrieval.
- Require approval for retry, reindex, and delete tools; expose SSE events and a recoverable Agent workbench.
- Show safe LLM stage summaries alongside tool calls, tool results, and approval status.
- Record upload batch status, SHA-256 content deduplication, failed stages, and retry attempts.
- Keep a one-time SQLite-to-PostgreSQL migration script for metadata from older versions.

## Project Structure

```text
app/
  api/routes.py                  FastAPI routes
  agent/graph.py                 LangGraph tool-routing agent
  agent/checkpoint.py            PostgreSQL AsyncPostgresSaver lifecycle
  agent/dynamic_prompt_agent.py  Dynamic RAG compatibility entrypoint
  agent/tools.py                 Agent tools
  core/config.py                 Environment variables and paths
  rag/embeddings.py              Qwen / OpenAI-compatible embeddings
  rag/llm.py                     Prompt building and model calls
  rag/text_splitter.py           Markdown and plain text splitters
  rag/vector_store.py            Chroma, BM25, and RRF retrieval wrapper
  services/document_service.py   Document upload, parsing, chunking, ingestion
  services/evaluation_service.py RAGAS and retrieval evaluation service
  services/document_management_service.py Single-document delete/reindex
  services/mineru_client.py      MinerU API client
  services/rag_service.py        Standard RAG flow
  services/agent_service.py      Tool loop, SSE, approvals, and checkpoint recovery
  services/vector_store_service.py Vector store status, debug, and cleanup
  static/                        Native web console
  storage/database.py            PostgreSQL metadata and chat history
eval_data/
  ai_agents_in_depth_eval_dataset.json  60-question PDF evaluation set
  ragas_ab_dataset.json          Legacy RAGAS A/B dataset
  agent_tasks.json               Agent tool-behavior evaluation set
sample_docs/
  company_handbook.md            Sample knowledge-base document
scripts/
  smoke_test.py                  Smoke test
  evaluate_ragas.py              CLI RAGAS evaluation
  compare_evaluations.py         Baseline/HyDE summary comparison
  evaluate_agent_tasks.py        Agent tool behavior evaluation
  migrate_sqlite_to_postgres.py  One-time legacy metadata migration
docs/
  agent-workbench.md             Agent loop, approvals, SSE, and checkpoint recovery
  batch-upload-engineering.md    Batch upload reliability design
  docker-deployment.md           Docker deployment and troubleshooting
  postgresql-migration.md        SQLite-to-PostgreSQL migration guide
```

## Quick Start

### Option 1: Docker Compose

Docker Compose starts both FastAPI and PostgreSQL. This is the simplest way to
run the complete demo:

```powershell
git clone https://github.com/LutxOVO/enterprise-rag-agent.git
cd enterprise-rag-agent
copy .env.example .env
# Edit .env and provide QWEN_API_KEY and DEEPSEEK_API_KEY.
# Add MINERU_API_TOKEN when uploading PDF/Office/image files.
docker compose config --quiet
docker compose up -d --build
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

Open `http://127.0.0.1:8000/` for the console and
`http://127.0.0.1:8000/docs` for Swagger.

### Option 2: uv + Compose PostgreSQL

Use this mode when you want to edit and run Python locally while PostgreSQL is
provided by Compose:

```powershell
git clone https://github.com/LutxOVO/enterprise-rag-agent.git
cd enterprise-rag-agent
uv venv .venv --python 3.12
uv sync --frozen
copy .env.example .env
docker compose up -d postgres
.\scripts\run_local.ps1 -Reload
```

For local Python execution, set `DATABASE_URL` to use `localhost`. Inside the
`rag` container, Compose uses the service hostname `postgres` automatically.

## Configuration

Configure API keys in `.env`:

```env
EMBEDDING_PROVIDER="qwen"
CHAT_PROVIDER="deepseek"

DATABASE_URL="postgresql+psycopg://rag:local-only-change-me@localhost:5432/rag"
POSTGRES_DB="rag"
POSTGRES_USER="rag"
POSTGRES_PASSWORD="local-only-change-me"
POSTGRES_PORT=5432

QWEN_API_KEY="your Alibaba Cloud Model Studio API key"
MINERU_API_TOKEN="your MinerU API token"
DEEPSEEK_API_KEY="your DeepSeek API key"

# The workbench uses Tavily only when the knowledge-base evidence is insufficient.
TAVILY_API_KEY="your Tavily API key"
TAVILY_MAX_RESULTS=5
TAVILY_SEARCH_DEPTH="basic"
```

`QWEN_API_KEY` is required to index and retrieve documents. `DEEPSEEK_API_KEY`
is required for RAG answers, Agent routing, and RAGAS judging. `MINERU_API_TOKEN`
is required only for document formats sent to MinerU. `TAVILY_API_KEY` is
optional and is used only when the workbench switch is enabled and knowledge-base
evidence is below the configured confidence threshold. Never commit `.env`.

Start the server:

```powershell
./scripts/run_local.ps1 -Reload
```

Open:

```text
Web console: http://127.0.0.1:8000/
Swagger docs: http://127.0.0.1:8000/docs
```

## Updating a Docker Deployment

`docker compose restart` only restarts the existing container; it does not copy
new source files into the image. After changing Python, HTML, CSS, JavaScript,
the Dockerfile, or dependencies, rebuild and recreate the application:

```powershell
docker compose up -d --build --force-recreate rag
```

After changing only `.env`, recreation is enough:

```powershell
docker compose up -d --force-recreate rag
```

`docker compose down` keeps the `postgres_data` named volume and the host
`data/` directory. Do not use `docker compose down -v` unless you intend to
delete PostgreSQL data.

## Main APIs

The Agent workbench includes a `web_search_enabled` toggle. When it is `false`, the
system prompt restricts the model to knowledge-base context and the server blocks
direct `search_web` calls. When it is `true`, the graph still searches the knowledge
base first and calls Tavily only when the retrieval result is marked
`fallback_to_web_search=true`. Web results are returned with titles and URLs and are
treated as untrusted reference material. Failed or empty web searches never become
an answer source.

- `POST /api/documents/upload` uploads and indexes a document.
- `DELETE /api/documents/{document_id}` removes one document and its local artifacts.
- `POST /api/documents/{document_id}/reindex` reparses the original file.
- `GET /api/documents` lists uploaded document metadata.
- `GET /api/status` returns system, database, and vector-store status.
- `POST /api/rag/ask` runs standard RAG.
- `POST /api/rag/stream` streams a RAG answer with SSE.
- `POST /api/agent/invoke` runs the LangGraph tool-routing agent (`web_search_enabled` is optional).
- `POST /api/agent/runs/stream` starts a checkpointed Agent run and streams SSE events.
- `POST /api/agent/threads/{thread_id}/resume/stream` resumes an approved write operation.
- `GET /api/agent/threads/{thread_id}/state` returns the persisted thread state.
- `DELETE /api/agent/threads/{thread_id}` deletes an idle Agent thread and its browser session data.
- `POST /api/agent/dynamic-rag` runs the Dynamic RAG workflow with query rewrite and HyDE.
- `GET /api/vector-store/chunks` inspects stored chunks.
- `DELETE /api/vector-store/clear` clears vector data and metadata.
- `POST /api/evaluation/ragas/build-samples` builds cached RAGAS samples.
- `POST /api/evaluation/ragas/run` runs RAGAS evaluation.
- `GET /api/documents/upload-batches` and the batch detail/retry routes expose upload history.

The Agent emits `run_started`, `llm_stage`, `tool_call`, `tool_result`, `approval_required`,
`answer`, `error`, and `done` events. Retry, reindex, and delete are paused with
`interrupt()` before side effects. Resume uses the same `thread_id` and the
`approval_id` shown in the approval event. A running or approval-pending thread
cannot be deleted and returns `409`; deleting a missing thread is idempotent.

The Agent now decides whether a tool is needed before executing it. Greetings,
thanks, and requests that do not depend on enterprise material can be answered
directly. When the user explicitly asks for the knowledge base, uploaded
documents, or internal material, `search_knowledge_base` is required; a graph
guard prevents an unsupported model-only answer when evidence is missing.
Web search remains disabled or enabled per run, and is only used as a fallback
after low-confidence knowledge-base evidence.

The workbench timeline combines model analysis, tool execution, and approval
stages in `trace_order`. `llm_stage` contains a safe summary, duration, and a
`reasoning_available` flag; it never exposes DeepSeek's raw
`reasoning_content`. The default `.env.example` enables DeepSeek thinking mode,
which can increase latency and token usage. Set `DEEPSEEK_THINKING_TYPE=disabled`
to keep lifecycle stages without the provider reasoning signal.

## RAGAS Evaluation

The evaluation flow uses a fixed question set and runs the same questions through
two retrieval modes: `baseline` and `hyde_rewrite`. It records RAGAS metrics,
Hit@1, Hit@3, MRR, empty-value counts, question categories, and stage timings.
The judge model is DeepSeek, so evaluation consumes API quota.

```powershell
uv run python scripts\evaluate_ragas.py --retrieval-mode baseline --max-samples 10 --force-rebuild
uv run python scripts\evaluate_ragas.py --retrieval-mode hyde_rewrite --max-samples 10 --force-rebuild
uv run python scripts\compare_evaluations.py
```

The generated `eval_outputs/` files are local artifacts and are ignored by Git.
Do not present an improvement percentage in a resume until both runs have been
performed on the same corpus, question set, `top_k`, and model configuration.

## Legacy SQLite Migration

Runtime storage is PostgreSQL. Older projects may migrate metadata once without
regenerating Chroma vectors:

```powershell
copy data\app.db data\app.db.bak
docker compose run --rm rag python scripts/migrate_sqlite_to_postgres.py `
  --sqlite-path /app/data/app.db --dry-run
docker compose run --rm rag python scripts/migrate_sqlite_to_postgres.py `
  --sqlite-path /app/data/app.db
```

The script migrates documents, messages, upload batches, batch items, and file
fingerprints with idempotent upserts. It does not delete the old SQLite file or
regenerate vectors. See `docs/postgresql-migration.md` for path mapping and
verification details.

## Resume Highlights

- Built a local knowledge-base RAG demo with document upload, parsing, chunking, embedding, retrieval, prompt construction, and answer generation.
- Wrapped Chroma as a retriever and returned top-k chunks with metadata and scores for retrieval debugging.
- Integrated MinerU API to parse PDF and other unstructured documents into Markdown before chunking.
- Implemented Agentic RAG with LangGraph, query rewrite, HyDE, context checks, and observable graph paths.
- Added RAGAS evaluation using DeepSeek as judge model and cached generated responses/contexts to avoid repeated token usage.
- Implemented PostgreSQL-backed checkpoint recovery, human approval for write tools,
  bounded tool execution, Tavily fallback, and batch-upload idempotency/retry.

## Notes

- Do not commit `.env`, `data/`, `.venv/`, `logs/`, or `eval_outputs/`.
- Use `uv add package-name` to add new dependencies.
- Python 3.12 is recommended for better compatibility with LangChain, Chroma, and RAGAS.
- This project is intended for single-machine learning and demonstration. It does
  not include authentication, RBAC, Redis, Celery, or multi-process coordination.
