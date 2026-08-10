# Enterprise Knowledge Base RAG Agent

[简体中文](README.md) | [English](README.en.md)

A local demo project for AI Agent, RAG, and LLM application development. The system provides a FastAPI backend and a lightweight web console for document ingestion, chunking, embedding, Chroma retrieval, hybrid retrieval, RAG question answering, Dynamic RAG, and RAGAS evaluation.

This is a runnable demo prototype, not a production system. It does not claim real company deployment, user scale, revenue, or online traffic.

## Tech Stack

- Python, FastAPI, Pydantic, Uvicorn
- LangChain, LangGraph, Dynamic RAG, Tool
- DeepSeek Chat and Qwen `text-embedding-v4`
- Alibaba Cloud Model Studio OpenAI-compatible Embedding API
- MinerU official API token workflow
- Chroma vector database and SQLite
- RAGAS with DeepSeek as judge model
- Native HTML / CSS / JavaScript frontend
- uv for dependency and environment management

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

## Project Structure

```text
app/
  api/routes.py                  FastAPI routes
  agent/graph.py                 LangGraph tool-routing agent
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
  services/vector_store_service.py Vector store status, debug, and cleanup
  static/                        Native web console
  storage/database.py            SQLite metadata and chat history
eval_data/
  ragas_eval_dataset.json        RAGAS evaluation dataset
sample_docs/
  company_handbook.md            Sample knowledge-base document
scripts/
  smoke_test.py                  Smoke test
  evaluate_ragas.py              CLI RAGAS evaluation
  compare_evaluations.py         Baseline/HyDE summary comparison
```

## Setup

```powershell
cd D:\pycharm项目\RAG
uv venv .venv --python 3.12
uv sync
copy .env.example .env
```

Configure API keys in `.env`:

```env
EMBEDDING_PROVIDER="qwen"
CHAT_PROVIDER="deepseek"

QWEN_API_KEY="your Alibaba Cloud Model Studio API key"
MINERU_API_TOKEN="your MinerU API token"
DEEPSEEK_API_KEY="your DeepSeek API key"
```

Start the server:

```powershell
uv run uvicorn app.main:app --reload
```

Open:

```text
Web console: http://127.0.0.1:8000/
Swagger docs: http://127.0.0.1:8000/docs
```

## Main APIs

- `POST /api/documents/upload` uploads and indexes a document.
- `DELETE /api/documents/{document_id}` removes one document and its local artifacts.
- `POST /api/documents/{document_id}/reindex` reparses the original file.
- `GET /api/documents` lists uploaded document metadata.
- `GET /api/status` returns system, database, and vector-store status.
- `POST /api/rag/ask` runs standard RAG.
- `POST /api/rag/stream` streams a RAG answer with SSE.
- `POST /api/agent/invoke` runs the LangGraph tool-routing agent.
- `POST /api/agent/dynamic-rag` runs the Dynamic RAG workflow with query rewrite and HyDE.
- `GET /api/vector-store/chunks` inspects stored chunks.
- `DELETE /api/vector-store/clear` clears vector data and metadata.
- `POST /api/evaluation/ragas/build-samples` builds cached RAGAS samples.
- `POST /api/evaluation/ragas/run` runs RAGAS evaluation.
- `GET /api/documents/upload-batches` and the batch detail/retry routes expose upload history.

## Resume Highlights

- Built a local knowledge-base RAG demo with document upload, parsing, chunking, embedding, retrieval, prompt construction, and answer generation.
- Wrapped Chroma as a retriever and returned top-k chunks with metadata and scores for retrieval debugging.
- Integrated MinerU API to parse PDF and other unstructured documents into Markdown before chunking.
- Implemented Agentic RAG with LangGraph, query rewrite, HyDE, context checks, and observable graph paths.
- Added RAGAS evaluation using DeepSeek as judge model and cached generated responses/contexts to avoid repeated token usage.

## Notes

- Do not commit `.env`, `data/`, `.venv/`, `logs/`, or `eval_outputs/`.
- Use `uv add package-name` to add new dependencies.
- Python 3.12 is recommended for better compatibility with LangChain, Chroma, and RAGAS.
