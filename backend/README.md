# Backend

DocuMind's backend is a FastAPI application that handles auth, document upload/indexing, session chat, SSE streaming, research-style structured answers, and the agent tool-routing layer.

This file intentionally mirrors `../BACKEND.md` so the backend overview is available from both the repository root and the backend folder.

## Current Stack

- FastAPI for HTTP APIs and Swagger docs.
- PostgreSQL with `pgvector` for documents, chunks, embeddings, users, sessions, messages, and feedback.
- SQLAlchemy async sessions from `backend/db/base.py`.
- GitHub Models/OpenAI-compatible APIs for embeddings and cloud LLM calls.
- Ollama as a supported local LLM provider.
- Tavily web search through `httpx`.
- Server-Sent Events for chat streaming through `sse-starlette`.
- LangChain tool schemas for LLM-visible tool contracts.

## Folder Map

```text
backend/
├── main.py                    FastAPI app, lifespan, CORS, router mounting
├── api/
│   ├── deps.py                Auth/current-user dependencies
│   └── routes/
│       ├── auth.py            Login/signup/password auth flows
│       ├── chat.py            SSE chat endpoint
│       ├── documents.py       Upload/list/delete/status/test retrieval
│       ├── feedback.py        Answer feedback routes
│       ├── research.py        Structured agent-routed research endpoint
│       ├── sessions.py        Session CRUD/history/document links
│       └── user.py            User profile routes
├── agent/
│   ├── tools.py               Tool schemas plus backend implementation functions
│   ├── router.py              Native tool-call routing, escalation, synthesis
│   └── README.md              Agent-specific architecture notes
├── core/
│   ├── config.py              Pydantic settings from `.env`
│   ├── indexer.py             PDF loading, chunking, embeddings, BM25 persistence
│   ├── retriever.py           Hybrid pgvector + BM25 retrieval and reranking
│   └── utils.py               Tokenization and shared reranker
├── db/
│   └── base.py                Async engine, session factory, declarative Base
├── models/
│   ├── models.py              SQLAlchemy ORM models
│   ├── schemas.py             Pydantic API schemas
│   └── auth_schemas.py        Auth request/response schemas
└── services/
    ├── chat_service.py        Chat orchestration and SSE metadata
    ├── llm_provider.py        Provider abstraction: github/groq/ollama
    ├── session_service.py     Session/message/document-link persistence
    ├── auth_service.py        User auth persistence
    ├── security.py            JWT/password utilities
    └── background_tasks.py    Background task helpers
```

## Runtime Flow

### Upload and Index

```text
POST /api/v1/documents/upload
    -> validate bearer token
    -> create Document row
    -> optionally link document to a session
    -> schedule async indexing job
    -> Indexer loads PDF pages
    -> split text into 600/150 chunks
    -> create deterministic hash chunk ids
    -> persist BM25 index to backend/data/bm25
    -> embed chunks with text-embedding-3-small
    -> insert Chunk rows with pgvector embeddings
```

### Chat

```text
POST /api/v1/chat/{session_id}
    -> validate session ownership
    -> save user message
    -> load last 6 history messages as role/content only
    -> load document ids linked to the session
    -> answer_query(...)
    -> emit SSE token events
    -> save assistant message
    -> emit metadata event with sources/chunks/tool_trace
```

History passed to the LLM contains only:

```python
{"role": msg.role, "content": msg.content}
```

It does not include citations, chunks, scores, debug metadata, or source payloads.

### Research

```text
POST /api/v1/research
    -> validate bearer token
    -> optional collection lookup by document public_id or numeric document id
    -> answer_query(...)
    -> return one structured JSON response
```

`/research` shares the same agent brain as chat but is non-streaming and stateless. It is intended for API consumers that want structured JSON:

```json
{
  "summary": "...",
  "key_findings": ["..."],
  "sources": [],
  "confidence": 0.82,
  "tool_trace": ["retrieve_from_document"],
  "follow_up_questions": []
}
```

## Agent Routing

`backend/agent/router.py` is the main orchestration entry point:

```python
await answer_query(
    query=question,
    document_ids=document_ids,
    db=db,
    history=history,
    include_web=True,
)
```

The router asks the configured LLM for native tool calls when supported by the provider:

- `retrieve_from_document(query)`
- `web_search(query)`
- `summarize_document(query)`
- `generate_quiz(query, num_questions=5)`

There is no LLM-visible `direct_answer` tool. If no external tool is needed, the model should return zero tool calls; the router records this as:

```json
"tool_trace": ["none"]
```

The backend then runs the normal direct-answer path with recent role/content history.

Private context such as `db`, `document_ids`, `user_id`, JWTs, and API keys is never exposed to the model as tool arguments. The model chooses only public args like `query`; Python injects trusted backend context during execution.

## Key Endpoints

- `POST /api/v1/auth/...` - authentication.
- `GET /api/v1/user/me` - current user.
- `POST /api/v1/documents/upload` - upload and index a PDF.
- `GET /api/v1/documents/all` - list user documents.
- `DELETE /api/v1/document/{public_id}` - delete document and indexed chunks.
- `GET /api/v1/documents/status/{job_id}` - in-memory indexing job status.
- `POST /api/v1/chat/{session_id}` - SSE chat.
- `POST /api/v1/research` - structured research response.
- `GET /api/v1/sessions/...` - session/history routes.
- `POST /api/v1/feedback` - feedback.
- `GET /health` and `GET /api/v1/health/db` - health checks.
- `GET /docs` - Swagger UI.

## Configuration

Important settings are in `backend/core/config.py` and loaded from `.env`:

```bash
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db
GITHUB_TOKEN=...
GROQ_API_KEY=...
LLM_PROVIDER=github        # github | groq | ollama
LLM_MODEL=gpt-4o-mini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b
TAVILY_API_KEY=...
ENABLE_WEB_SEARCH=true
SECRET_KEY=...
```

The app verifies database connectivity and creates the `vector` extension during startup.

## Notes and Limitations

- Chat streams word-like chunks from a completed answer, not true provider token streaming.
- `/research` is implemented, but follow-up questions are still returned as an empty list.
- Research `key_findings` are currently extracted with simple text splitting.
- Indexing job status is stored in memory, so it resets when the server restarts.
- `tool_used` is stored as a compact string; a JSONB `tool_trace` column would be cleaner for production.
