# 🔧 Backend — FastAPI Production Layer

> Phase 2: Decouple RAG logic from Streamlit UI. This folder contains all production backend code.

---

## 📦 Folder Structure

```
backend/
├── __init__.py
├── main.py                           ← FastAPI app entry point (WIP)
├── db/                                ← DB infra: engine, Base, sessions, migrations
│   ├── base.py                        ← engine, AsyncSessionLocal, Base, get_db
│   └── migrations/                    ← Alembic migrations (optional)
├── core/
│   ├── __init__.py
│   ├── README.md                     ← Core modules breakdown
│   ├── config.py                     ← Pydantic settings from .env (WIP)
│   ├── indexer.py                    ← PDF indexing (from Phase 1)
│   ├── retriever.py                  ← RAG retrieval & generation (from Phase 1)
│   └── utils.py                      ← Tokenizers, helpers
├── api/
│   ├── __init__.py
│   ├── deps.py                       ← FastAPI dependencies (WIP)
│   ├── models.py                     ← Pydantic request/response schemas (WIP)
│   └── routes/
│       ├── __init__.py
│       ├── collections.py            ← POST/GET/DELETE collections (WIP)
│       ├── chat.py                   ← SSE streaming chat endpoint (WIP)
│       ├── research.py               ← Phase 3: /research agent endpoint (WIP)
│       ├── feedback.py               ← Phase 4: thumbs up/down (WIP)
│       └── health.py                 ← GET /health (WIP)
├── agent/                            ← Phase 3: LangGraph agent
│   ├── __init__.py
│   ├── graph.py                      ← State machine + node definitions (TODO)
│   ├── intent.py                     ← Intent classification (TODO)
│   ├── tools.py                      ← Tool definitions (TODO)
│   └── trust.py                      ← Confidence + citation logic (TODO)
├── services/                         ← Business logic
│   ├── __init__.py
│   ├── session_service.py            ← SQLite chat history (WIP)
│   ├── background_tasks.py           ← Async indexing (WIP)
│   └── cache_service.py              ← Optional: Redis caching (TODO)
└── migrations/                       ← DB schema migrations (TODO)
  └── __init__.py
```

---

## 🚀 Quick Start (Phase 2)

### 1. Install Dependencies

```bash
pip install fastapi uvicorn "pydantic-settings>=2.0" sse-starlette loguru aiosqlite httpx
```

### 2. Start the Server

```bash
# From project root
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Swagger UI

Visit: `http://localhost:8000/docs`

### 4. Database

- This backend uses Postgres with `pgvector` for storing embeddings and `SQLite` for lightweight local fallback in development (if configured).
- Ensure `DATABASE_URL` in your `.env` points to an asyncpg URL, e.g. `postgresql+asyncpg://user:pass@db-host:5432/documind` and that the Postgres server has the `vector` extension enabled.

---

## 📋 Module Breakdown

### `main.py` — FastAPI App

```python
# Example structure (to be implemented)
from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: validate env, connect Qdrant, init models
    yield
    # Shutdown: close connections

app = FastAPI(
    title="DocuMind API",
    version="2.0.0",
    lifespan=lifespan
)

# Include route modules
app.include_router(health.router, prefix="/api/v1")
app.include_router(collections.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(research.router, prefix="/api/v1")
app.include_router(feedback.router, prefix="/api/v1")
```

### `core/config.py` — Pydantic Settings

```python
# Example (to be implemented)
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    github_token: str  # required
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    top_k: int = 15
    rerank_top_n: int = 5
    confidence_threshold: float = 0.3
    chunk_size: int = 600
    chunk_overlap: int = 150
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### `api/routes/health.py`

```python
# GET /api/v1/health
# Response: {"status": "ok", "version": "2.0.0"}
```

### `api/routes/collections.py`

- `POST /api/v1/collections/upload` → Index PDF, return job_id
- `GET /api/v1/collections` → List all collections
- `DELETE /api/v1/collections/{name}` → Remove collection
- `GET /api/v1/collections/status/{job_id}` → Progress %

### `api/routes/chat.py`

- `POST /api/v1/chat/{collection}` → SSE streaming chat
- Request: `{"query": str, "history": list}`
- Response: Server-Sent Events with tokens + metadata

### `services/session_service.py`

SQLite-based conversation history:
- `create_session(collection_name)` → session_id
- `add_message(session_id, role, content, sources, tool_used)`
- `get_history(session_id)` → last N messages
- `delete_session(session_id)`

---

## 🔌 Environment Variables

```bash
# Required
GITHUB_TOKEN=ghp_xxx                    # For GitHub model API

# Optional
QDRANT_URL=http://localhost:6333        # Qdrant connection
OLLAMA_BASE_URL=http://localhost:11434  # Ollama LLM server
OLLAMA_MODEL=qwen3:4b                   # Which Ollama model to use
TAVILY_API_KEY=xxx                      # For web search (Phase 3)
```

---

## 🔄 API Response Format (Standard)

All endpoints return JSON with this structure:

```json
{
  "status": "ok" | "error",
  "data": {...},                // endpoint-specific
  "error": "error message if status=error",
  "timestamp": "2026-05-19T10:30:00Z"
}
```

### Chat Response (SSE Streaming)

```
event: token
data: "The"

event: token
data: " answer"

event: token
data: " is"

event: metadata
data: {
  "sources": [...],
  "confidence": 0.82,
  "tool_used": "retrieve_from_document"
}

event: done
data: "[DONE]"
```

---

## 🧪 Testing

### Unit Tests

```bash
pytest tests/test_unit_rag.py -v
```

### Integration Tests

```bash
# Start server first
uvicorn backend.main:app

# In another terminal
python tests/integration_test.py
```

### Manual API Testing

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Upload PDF
curl -X POST -F "file=@docs/sample.pdf" \
  http://localhost:8000/api/v1/collections/upload

# Chat
curl -X POST http://localhost:8000/api/v1/chat/sample.pdf \
  -H "Content-Type: application/json" \
  -d '{"query":"What is this about?","history":[]}'
```

---

## 📊 Database Schema (Phase 2+)

### `sessions` Table

```sql
CREATE TABLE sessions (
  id TEXT PRIMARY KEY,
  collection_name TEXT NOT NULL,
  created_at TIMESTAMP,
  updated_at TIMESTAMP,
  metadata JSON
);

CREATE TABLE messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT FOREIGN KEY,
  role TEXT,  -- "user" | "assistant"
  content TEXT,
  sources JSON,
  tool_used TEXT,
  confidence FLOAT,
  timestamp TIMESTAMP
);
```

---

## 🔐 Security

- **Query sanitization** — inherited from Phase 1 Retriever
- **API key validation** — GitHub token required for embeddings
- **Rate limiting** — TODO: implement in Phase 2
- **CORS** — configure in main.py
- **Input validation** — Pydantic models auto-validate

---

## 📈 Monitoring & Logging

```python
from loguru import logger

logger.info("Chat request", extra={"user_id": "123", "collection": "report.pdf"})
logger.error("Retrieval failed", extra={"error": str(e)})
```

Logs written to: `logs/backend.log`

---

## 🚦 Phase 2 Checklist

- [ ] `backend/main.py` — FastAPI app with lifespan
- [ ] `backend/core/config.py` — Pydantic settings
- [ ] `backend/api/deps.py` — dependency injection
- [ ] `backend/api/models.py` — request/response schemas
- [ ] `backend/api/routes/health.py` — health check
- [ ] `backend/api/routes/collections.py` — collection CRUD
- [ ] `backend/api/routes/chat.py` — SSE streaming
- [ ] `backend/services/session_service.py` — chat history DB
- [ ] `/docs` Swagger UI working
- [ ] Integration tests passing

---

**Last Updated:** May 19, 2026
