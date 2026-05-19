# DocuMind — Production RAG-QA System

> **Status:** Phase 1 ✅ Complete | Phase 2 🚀 In Progress
> **Current:** Streamlit UI + Hybrid RAG pipeline (BM25 + Vector) with CrossEncoder reranking
> **Goal:** Build an autonomous research agent (Phase 3) on top of production-grade RAG

---

## 🎯 What This Project Does

DocuMind is a **Retrieval-Augmented Generation (RAG) system** that answers questions by:
1. Indexing user-uploaded PDFs into a vector database (Qdrant)
2. Performing **hybrid search** (BM25 keyword + vector semantic)
3. **Reranking** results using a CrossEncoder model
4. Grounding LLM answers strictly to retrieved document chunks
5. Returning structured JSON with citations, sources, and confidence scores

### Key Features
- ✅ **Hybrid retrieval** — BM25 + vector search → RRF fusion
- ✅ **Deterministic chunk IDs** — stable cross-store identification
- ✅ **CrossEncoder reranking** — improved relevance ranking
- ✅ **Query sanitization** — prompt injection prevention
- ✅ **Qdrant fallback** — graceful degradation when vector DB unavailable
- ✅ **Structured output** — machine-readable JSON with metadata
- ✅ **Citation tracking** — every answer cites source page + excerpt
- ✅ **Confidence scores** — helps UI decide answer reliability

---

## 📦 Project Structure

```
rag-qa/
├── README.md                 ← High-level overview (this file)
├── ARCHITECTURE.md           ← System design & data flow
├── .agent.md                 ← Agent quick reference & guidelines
├── requirements.txt          ← Python dependencies
├── .env                      ← Secrets (GITHUB_TOKEN, QDRANT_URL)
│
├── app.py                    ← Streamlit UI entry point
├── indexing.py               ← PDF → chunks → Qdrant indexer
├── retrieving.py             ← Legacy retriever (DEPRECATED)
├── test.py                   ← Legacy test runner
│
├── backend/                  ← Production code (Phase 2+)
│   ├── BACKEND.md            ← Backend structure & setup
│   ├── main.py               ← FastAPI app (WIP)
│   ├── core/
│   │   ├── MODULES.md        ← Core modules breakdown
│   │   ├── config.py         ← Settings & env vars
│   │   ├── indexer.py        ← PDF indexing logic
│   │   ├── retriever.py      ← RAG retrieval & generation
│   │   └── utils.py          ← Tokenizers, helpers
│   ├── api/
│   │   ├── routes/
│   │   │   ├── collections.py
│   │   │   ├── chat.py
│   │   │   └── health.py
│   │   ├── deps.py
│   │   └── models.py         ← Pydantic schemas
│   ├── agent/                ← Phase 3: LangGraph
│   └── services/
│
├── data/
│   ├── bm25/                 ← Pickled BM25 models
│   └── *.pdf                 ← Uploaded documents
│
├── tests/
│   ├── test_unit_rag.py      ← Unit tests (mocked)
│   └── rag_output.py         ← Query runner with pretty printer
│
└── docs/
    └── (user-uploaded PDFs)
```

---

## 🔄 Phase Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| **1** | RAG core quality fixes (chunking, hybrid search, reranking, citations) | ✅ DONE |
| **2** | FastAPI backend migration (decouple logic from Streamlit UI) | 🚀 WIP |
| **3** | LangGraph agent layer (intelligent tool routing: doc/web/direct) | 📋 PLANNED |
| **4** | UX & conversation features (sessions, follow-ups, export) | 📋 PLANNED |
| **5** | Intelligence add-ons (quiz generation, feedback loop) | 📋 PLANNED |

---

## 🚀 Quick Start

### Install & Run

```bash
# Setup Python environment
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env: add GITHUB_TOKEN, QDRANT_URL

# Start Qdrant vector DB
docker run -p 6333:6333 qdrant/qdrant:latest

# Run Streamlit UI
streamlit run app.py

# Or test RAG directly
python tests/rag_output.py
```

### Test the RAG Pipeline

```bash
python tests/rag_output.py
```

Output shows:
- **[Model-produced answer]** — what the LLM generated
- **[Sources & metadata added by retriever]** — citations, pages, scores

---

## 🔑 Key Components

### Phase 1 — Production RAG (✅ Complete)

| Component | File | Purpose |
|-----------|------|---------|
| **Indexer** | `backend/core/indexer.py` | Parse PDF → chunks with deterministic IDs → Qdrant + BM25 |
| **Retriever** | `backend/core/retriever.py` | Hybrid search, RRF fusion, reranking, generation, sanitization |
| **Utils** | `backend/core/utils.py` | Tokenizer, helpers |

### Phase 2 — FastAPI Backend (🚀 In Progress)

| Component | File | Purpose |
|-----------|------|---------|
| **Config** | `backend/core/config.py` | Pydantic settings from `.env` |
| **API App** | `backend/main.py` | FastAPI with lifespan, route registration |
| **Routes** | `backend/api/routes/*.py` | `/collections`, `/chat`, `/health`, `/research` |
| **Models** | `backend/api/models.py` | Request/response schemas |
| **Services** | `backend/services/` | Session DB, background tasks |

### Phase 3 — LangGraph Agent (📋 Planned)

| Component | File | Purpose |
|-----------|------|---------|
| **State** | `backend/agent/graph.py` | AgentState TypedDict + graph definition |
| **Intent** | `backend/agent/intent.py` | Classify: factual / summarize / web / direct / quiz |
| **Tools** | `backend/agent/tools.py` | Registered tools: retrieve_doc, web_search, summarize, direct |

---

## 📊 Data Flow

```
User Query
    ↓
[Query Sanitization]
    ↓
[Hybrid Search]
  ├─ Vector search (Qdrant) → 15 results
  └─ BM25 search (pickle) → 15 results
    ↓
[Reciprocal Rank Fusion] — merge by chunk_id
    ↓
[CrossEncoder Reranking] — top 10
    ↓
[Context Building] — format for LLM
    ↓
[LLM Generation] — gpt-4o-mini or Ollama
    ↓
[JSON Assembly] — Python builds citations + metadata
    ↓
Structured Response:
{
  "answer": "...",
  "citations": [...],
  "chunks": [...],
  "confidence": 0.82,
  "debug": {...}
}
```

---

## 🔐 Security & Configuration

### Environment Variables
```
GITHUB_TOKEN=ghp_xxx           # For OpenAI embeddings via GitHub
QDRANT_URL=http://localhost:6333
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b
```

### Query Sanitization
- Max 1000 chars
- Blocks injection patterns: "ignore all instructions", "ignore previous", etc.
- Raises `ValueError` if malicious detected

---

## 📚 For Agents: Quick Reference

Read these in order to understand the project:
1. **README.md** (this file) — overview
2. **ARCHITECTURE.md** — data flow & design decisions
3. **BACKEND.md** — backend structure
4. **MODULES.md** — module breakdown
5. **.agent.md** — agent-specific guidelines

---

## ✅ Phase 1 Completion Checklist

- [x] Chunk size fixed (600/150)
- [x] Duplicate removal on re-index
- [x] Hybrid BM25 + vector retrieval
- [x] RRF fusion keyed by `chunk_id`
- [x] CrossEncoder reranker integrated
- [x] Query sanitization (prompt injection defense)
- [x] Qdrant fallback to BM25-only
- [x] Structured JSON output with citations
- [x] Confidence scoring
- [x] Pretty-print client parser (shows LLM vs app-added metadata)

---

## 🎯 Next Steps (Phase 2)

1. Install FastAPI deps: `pip install fastapi uvicorn pydantic-settings sse-starlette`
2. Scaffold `backend/main.py` with lifespan & routers
3. Add `backend/core/config.py` with Pydantic settings
4. Create `/api/v1/health`, `/api/v1/chat/{collection}`, `/api/v1/collections` endpoints
5. Test with Swagger UI at `http://localhost:8000/docs`

---

**Last Updated:** May 19, 2026  
**Author:** DocuMind Team  
**License:** MIT

app.py  (Streamlit UI)
   ├── Tab 1 — Upload new PDF → index → chat
   └── Tab 2 — Pick existing collection → chat instantly
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Vector DB | Qdrant (Docker) |
| Embeddings | `text-embedding-3-large` via GitHub Models |
| LLM | `gpt-4o-mini` via GitHub Models |
| PDF Loader | LangChain `PyPDFLoader` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |

---

## 🚀 Setup

### 1. Start Qdrant
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create `.env`
```env
GITHUB_TOKEN=your_github_models_token
```

### 4. Run
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
rag-qa/
├── app.py          # Streamlit UI — tabs, caching, streaming
├── indexing.py     # Indexer class — load → chunk → embed → store
├── retrieving.py   # Retriver class — search → stream answer
├── docs/           # Uploaded PDFs saved here automatically
├── src/            # Screenshots
└── .env            # API keys (not committed)
```
