# Backend — FastAPI Production Layer

> Phase 2 documentation for the backend boundary that will replace direct UI-to-RAG calls.

---

## What Lives Here

The backend layer will host the FastAPI app, request/response schemas, route modules, and service code that wraps the Phase 1 RAG pipeline.

### Planned Structure

```text
backend/
├── main.py              ← FastAPI app + lifespan
├── core/
│   ├── config.py        ← Pydantic settings
│   ├── indexer.py       ← PDF indexing
│   ├── retriever.py     ← Retrieval + generation
│   └── utils.py         ← Helpers
├── api/
│   ├── routes/
│   │   ├── collections.py
│   │   ├── chat.py
│   │   ├── research.py
│   │   ├── feedback.py
│   │   └── health.py
│   ├── deps.py
│   └── models.py
├── agent/
└── services/
```

---

## Primary Responsibilities

- Expose `/api/v1/health` for readiness checks.
- Expose collection management endpoints for upload/list/delete/status.
- Expose `/api/v1/chat/{collection}` for streaming chat.
- Expose `/api/v1/research` for the Phase 3 agent layer.
- Persist sessions and chat history in SQLite.

---

## Configuration

Expected env vars:

```bash
GITHUB_TOKEN=ghp_xxx
QDRANT_URL=http://localhost:6333
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b
TAVILY_API_KEY=xxx
```

---

## Key Notes

- Keep the Streamlit app thin once Phase 2 starts.
- Reuse `backend/core/retriever.py` and `backend/core/indexer.py` rather than duplicating logic.
- Add Pydantic schemas before implementing routes.
- Keep API responses structured and stable for the UI.

---

## Phase 2 Goal

Make the backend the single source of truth for indexing, retrieval, chat, and later agent orchestration.
