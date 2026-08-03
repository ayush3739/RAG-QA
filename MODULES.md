# Modules

This document maps the current DocuMind codebase by responsibility. For detailed flow diagrams, see `ARCHITECTURE.md`.

## Root-Level Areas

```text
backend/        FastAPI backend and RAG/agent logic
frontend/       Lightweight HTML/JS test clients for chat and history
docs/           PRD and implementation plans
alembic/        Database migrations
```

## Backend Entry Point

### `backend/main.py`

Owns:

- FastAPI app creation.
- Lifespan startup/shutdown.
- Database connectivity check.
- `CREATE EXTENSION IF NOT EXISTS vector`.
- CORS setup.
- Static test frontend mounting at `/frontend`.
- Test pages:
  - `/test-chat`
  - `/test-history`
- Router registration under `/api/v1`.

## API Layer

### `backend/api/deps.py`

FastAPI dependencies:

- OAuth2 bearer token extraction.
- JWT decode through `backend/services/security.py`.
- Current user lookup through `AuthService`.
- Re-export of `get_db`.

### `backend/api/routes/auth.py`

Authentication routes for user signup/login/password flows.

### `backend/api/routes/user.py`

Current-user profile routes.

### `backend/api/routes/documents.py`

Document API:

- Upload files.
- Create `Document` rows.
- Link uploaded documents to sessions.
- Start async indexing jobs.
- List user documents.
- Delete documents and local files.
- Poll in-memory indexing job status.
- Manual retrieval/answer test endpoints.

Key note: upload currently handles PDF indexing through `Indexer`.

### `backend/api/routes/chat.py`

SSE chat API:

```text
POST /api/v1/chat/{session_id}
```

Delegates all chat behavior to `ChatService.stream_chat(...)`.

### `backend/api/routes/research.py`

Structured, non-streaming research API:

```text
POST /api/v1/research
```

Uses the same agent router as chat but returns a `ResearchResponse` JSON object with:

- `summary`
- `key_findings`
- `sources`
- `confidence`
- `tool_trace`
- `follow_up_questions`

`collection` is resolved as either a document `public_id` or a numeric document id owned by the authenticated user.

### `backend/api/routes/sessions.py`

Session creation, listing, deletion, history, and document linking routes.

### `backend/api/routes/feedback.py`

Feedback API for answer ratings/comments.

## Agent Layer

### `backend/agent/router.py`

Main agent orchestrator.

Responsibilities:

- Ask the LLM for native tool calls when the provider supports it.
- Fall back to strict JSON routing for unsupported providers.
- Route to document retrieval, web search, summarization, quiz generation, or zero-tool direct answer.
- Validate tool choices against backend constraints.
- Inject private backend context into tool implementation functions.
- Escalate low-confidence document retrieval to web search once.
- Synthesize final answers.
- Return answer metadata for chat/research APIs.

The public entry point is:

```python
await answer_query(
    query=query,
    document_ids=document_ids,
    db=db,
    history=history,
    include_web=True,
)
```

### `backend/agent/tools.py`

Contains two kinds of functions:

LLM-visible schemas:

- `retrieve_from_document(query)`
- `web_search(query)`
- `summarize_document(query)`
- `generate_quiz(query, num_questions=5)`

Backend implementation functions:

- `retrieve_from_document_impl(query, document_ids, db)`
- `web_search_impl(query)`
- `direct_answer_impl(query, history=None)`
- `summarize_document_impl(query, document_ids, db)`
- `generate_quiz_impl(query, document_ids, db, num_questions=5)`

The schema functions deliberately raise if called directly. The router should call the implementation functions.

There is no LLM-visible `direct_answer` tool. Direct/general responses are represented by zero tool calls and logged as `tool_trace: ["none"]`.

## Core RAG Layer

### `backend/core/config.py`

Pydantic settings loaded from `.env`.

Important settings:

- `DATABASE_URL`
- `github_token`
- `groq_api_key`
- `llm_provider`
- `llm_model`
- `ollama_base_url`
- `ollama_model`
- `tavily_api_key`
- `enable_web_search`
- chunking and retrieval parameters
- JWT and mail settings

### `backend/core/indexer.py`

Indexes uploaded PDFs.

Flow:

- Load PDF with `PyPDFLoader`.
- Split text into 600/150 chunks.
- Create stable hash-based `chunk_id`.
- Persist BM25 index to `backend/data/bm25`.
- Generate embeddings with `text-embedding-3-small`.
- Insert chunks into PostgreSQL with pgvector embeddings.

### `backend/core/retriever.py`

Retrieves document context.

Flow:

- Sanitize query.
- Embed query.
- Run pgvector cosine-distance search.
- Run BM25 search when the BM25 index is available.
- Merge with reciprocal rank fusion.
- Rerank with CrossEncoder.
- Return structured chunks plus raw confidence.

It also still has `generate_response()` and `answer()` methods for direct RAG answering, but the main chat/research path now goes through the agent router.

### `backend/core/utils.py`

Shared helpers:

- `simple_tokenize(...)` for BM25.
- Shared `RERANKER` CrossEncoder instance.

## Database and Models

### `backend/db/base.py`

Owns:

- Async SQLAlchemy engine.
- `AsyncSessionLocal`.
- Declarative `Base`.
- `get_db()` dependency.

### `backend/models/models.py`

ORM tables:

- `User`
- `PasswordResetToken`
- `Session`
- `Message`
- `Document`
- `SessionDocument`
- `Chunk`
- `Feedback`

### `backend/models/schemas.py`

Pydantic request/response schemas for chat, research, documents, sessions, messages, and feedback.

### `backend/models/auth_schemas.py`

Auth-specific Pydantic schemas.

## Services

### `backend/services/chat_service.py`

Coordinates one chat turn:

- Validate session ownership.
- Save the user message.
- Load recent role/content history.
- Load linked document ids.
- Call `answer_query(...)`.
- Emit SSE token events.
- Save assistant response.
- Emit metadata with sources, chunks, confidence, and tool trace.

### `backend/services/session_service.py`

Session persistence:

- Create sessions.
- Fetch sessions.
- Add messages.
- Read recent history.
- List sessions.
- Link documents to sessions.
- Delete sessions.

### `backend/services/llm_provider.py`

Single abstraction for model calls:

- `invoke(messages)` for full responses.
- `stream(messages)` for provider streaming.
- `tool_call(messages, tools)` for native tool-call providers.
- `supports_native_tool_calls()`.

### `backend/services/auth_service.py`

User creation and lookup helpers.

### `backend/services/security.py`

Password hashing and JWT encode/decode helpers.

### `backend/services/email_service.py`

Email support for auth/password flows.

## Frontend Test Clients

### `frontend/index.html`, `frontend/app.js`, `frontend/styles.css`

Simple SSE chat test client served by:

```text
GET /test-chat
```

### `frontend/history.html`, `frontend/history.js`

Simple session-history test client served by:

```text
GET /test-history
```

### `frontend/dev-config.json`

Local testing config, such as saved API base URL/session/token values.

## Read First

For backend work:

1. `BACKEND.md`
2. `ARCHITECTURE.md`
3. `backend/agent/README.md`
4. `backend/core/README.md`
5. The specific module you plan to edit
