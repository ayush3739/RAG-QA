# KnowVex ⚡
### Production-Grade Agentic RAG & Knowledge Intelligence Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19.0-61DAFB.svg?style=flat&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg?style=flat&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1.svg?style=flat&logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v4-38B2AC.svg?style=flat&logo=tailwind-css&logoColor=white)](https://tailwindcss.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**KnowVex** is an open-source, production-ready **Agentic Retrieval-Augmented Generation (RAG)** system designed to move beyond naive vector search. Instead of blindly embedding queries and hoping for the best, KnowVex employs an **intelligent tool-routing agent** that classifies intent, selects the optimal tool (Document RAG, Live Web Search, or Direct Reasoning), executes hybrid retrieval with CrossEncoder reranking, and streams grounded answers with verifiable citations.

---

## 🌟 Why KnowVex?

Most RAG tutorials build a simple script that dumps PDFs into a vector database, performs cosine similarity, and stuffs the prompt into an LLM. In production, this fails because:
- Greetings and general reasoning waste vector queries.
- Comparative or exact-keyword queries fail dense vector search alone.
- Answers lack provenance, leaving users unable to verify truth from hallucination.

**KnowVex solves this by combining:**
1. **Agentic Tool Routing:** Uses an intent router to intelligently choose between Document Retrieval, Real-Time Web Search, or Direct LLM Reasoning.
2. **Hybrid Search + RRF:** Merges BM25 keyword matching with dense vector embeddings using Reciprocal Rank Fusion (RRF).
3. **CrossEncoder Reranking:** Computes cross-attention relevance scores on candidate chunks to isolate the highest-signal context.
4. **Verifiable Citations:** Returns exact document names, page numbers, excerpt snippets, and confidence scores with every response.
5. **Full Production App Shell:** Features a React 19 frontend with JWT auth, OAuth (GitHub & Google), document workspace, session export, and real-time SSE streaming.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    User([User Query]) --> Auth[FastAPI Auth & Rate Limiting]
    Auth --> Router{Agent Intent Router}
    
    Router -->|Document Question| RAG[Hybrid RAG Pipeline]
    Router -->|Real-time / Current Event| Web[Web Search Tool / Tavily]
    Router -->|Greeting / General Reasoning| Direct[Direct LLM Synthesis]
    
    subgraph Hybrid RAG Engine
        RAG --> Dense[Dense Vector Search \n pgvector]
        RAG --> Sparse[Sparse Keyword Search \n BM25]
        Dense & Sparse --> RRF[Reciprocal Rank Fusion]
        RRF --> Rerank[CrossEncoder Reranker]
        Rerank --> Context[Top Context Chunks]
    end
    
    Context --> Synthesizer[Response Synthesizer & Citation Builder]
    Web --> Synthesizer
    Direct --> Synthesizer
    
    Synthesizer --> Stream[SSE Token Streaming]
    Stream --> Client([React Frontend Workspace])
```

---

## ✨ Core Features

### 🧠 Agentic Retrieval Layer
- **Intent Router:** Classifies user prompts into actionable tool paths (`retrieve_doc`, `web_search`, `direct_answer`).
- **LangChain / LLM Provider Abstraction:** Seamlessly switch between Groq, Gemini, OpenRouter, NVIDIA NIM, AWS Bedrock, or local Ollama models.
- **Query Sanitization:** Defends against prompt injection and jailbreak payloads before reaching the model.

### 🔍 Advanced Hybrid Search
- **pgvector & Qdrant Support:** Scalable vector similarity search over chunk embeddings.
- **BM25 Keyword Matching:** Catches acronyms, exact IDs, and domain-specific terminology that vector embeddings miss.
- **CrossEncoder Reranking:** Re-scores top candidates with fine-grained cross-attention to filter noise.
- **Deterministic Chunk IDs:** Consistent hashing across document updates and sessions.

### 🛡️ Enterprise Security & Multi-Tenancy
- **JWT Authentication:** Dual-token strategy with short-lived access tokens and secure refresh token rotation.
- **Social OAuth:** One-click authentication with GitHub and Google OAuth 2.0.
- **Email Verification & Password Reset:** Built-in email token flows with secure expiration checks.
- **Isolated Document Namespaces:** Multi-tenant document storage and collection isolation.

### 💻 Modern Frontend Experience
- **React 19 + TypeScript + Vite:** High-performance single-page app with instant HMR.
- **Cyberpunk Dark Aesthetic:** Custom Tailwind styles, living animated ribbon veils, and interactive ASCII / pixel typography.
- **Server-Sent Events (SSE):** Token-by-token streaming responses with live status feedback.
- **Workspace Views:**
  - 💬 **Chat Workspace:** Interactive chat with citation inspection, token efficiency metrics, and tool badges.
  - 📁 **Document Manager:** Drag-and-drop PDF ingestion, chunk inspector, and indexing status.
  - 🕒 **Session History:** Searchable conversation log with one-click Markdown session export.
  - ⚙️ **Settings:** Configurable temperature, token efficiency, and custom system instructions.

---

## 📁 Repository Structure

```
rag-qa/
├── backend/                  # Production FastAPI Application
│   ├── main.py               # Application entry point & lifespan
│   ├── agent/                # Agent layer: router, tools, graph
│   │   ├── router.py         # Intent classification & routing
│   │   └── tools.py          # RAG, web search, and direct tools
│   ├── api/                  # API routes & schemas
│   │   ├── routes/
│   │   │   ├── auth.py       # Login, register, OAuth, refresh
│   │   │   ├── chat.py       # SSE streaming chat endpoints
│   │   │   ├── documents.py  # PDF upload, ingestion, deletion
│   │   │   ├── sessions.py   # Conversation history & export
│   │   │   ├── user.py       # Profile & account settings
│   │   │   └── analytics.py  # Usage and token tracking
│   │   └── models.py         # Pydantic request/response schemas
│   ├── core/                 # Core utilities & configuration
│   │   ├── config.py         # Pydantic settings & environment vars
│   │   ├── indexer.py        # PDF chunking & vector indexing
│   │   └── retriever.py      # Hybrid search & CrossEncoder pipeline
│   └── services/             # LLM provider & database clients
│
├── frontend/                 # React 19 + TypeScript + Vite SPA
│   ├── src/
│   │   ├── components/       # Workspace UI, Auth, Modals, Chat
│   │   ├── landing/          # Modern landing page & architecture deep dive
│   │   ├── store/            # Zustand global state management
│   │   ├── lib/              # API client & error handling
│   │   └── App.tsx           # Main application routing
│   ├── index.html            # App shell
│   └── vite.config.ts        # Vite configuration
│
├── tests/                    # Unit tests & verification scripts
├── requirements.txt          # Python backend dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.10+**
- **Node.js 18+ & npm**
- **PostgreSQL with pgvector** (or Docker)

---

### 1. Database Setup (Docker)

Run PostgreSQL with the `pgvector` extension:

```bash
docker run -d \
  --name knowvex-db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=knowvex \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

---

### 2. Backend Setup

```bash
# Navigate to the project root
cd rag-qa

# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
```

Edit `.env` with your API keys and configuration:

```env
# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/knowvex

# LLM Providers (Pick at least one)
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIza...
TAVILY_API_KEY=tvly-...

# Security & JWT
SECRET_KEY=generate-a-strong-random-key-for-jwt
FRONTEND_URL=http://localhost:3000

# OAuth (Optional)
GITHUB_CLIENT_ID=...
GITHUB_CLIENT_SECRET=...
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
```

Start the FastAPI development server:

```bash
uvicorn backend.main:app --reload --port 8000
```

Interactive API documentation will be available at:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

### 3. Frontend Setup

In a new terminal:

```bash
cd rag-qa/frontend

# Install dependencies
npm install

# Start the Vite development server
npm run dev
```

Open your browser at **`http://localhost:3000`**.

---

## ⚙️ Configuration Reference

| Variable | Description | Default |
|:---|:---|:---|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://...` |
| `LLM_PROVIDER` | Active LLM provider (`groq`, `gemini`, `ollama`) | `groq` |
| `LLM_MODEL` | Active model identifier | `openai/gpt-oss-120b` |
| `TAVILY_API_KEY` | API key for Tavily live web search | `""` |
| `ENABLE_WEB_SEARCH`| Enable/disable web search fallback tool | `true` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT Access Token lifetime | `60` |
| `REFRESH_TOKEN_EXPIRE_DAYS` | Refresh Token lifetime | `7` |
| `CHUNK_SIZE` | Chunk character length for document splitting | `600` |
| `CHUNK_OVERLAP` | Overlapping characters between chunks | `150` |
| `RERANKER_INPUT_CHUNKS` | Number of candidate chunks passed to reranker | `25` |
| `LLM_CONTEXT_CHUNKS` | Final top chunks injected into LLM prompt | `10` |

---

## 🧪 Testing

Run unit tests and verification scripts:

```bash
# Run backend test suite
pytest tests/

# Test RAG retrieval directly in terminal
python tests/rag_output.py
```

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author & Acknowledgments

Built by **[Ayush Maurya](https://github.com/ayush3739)**.

If you find KnowVex useful, please consider giving it a ⭐ on [GitHub](https://github.com/ayush3739/RAG-QA)!
