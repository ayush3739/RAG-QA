# DocuMind Architecture

DocuMind is a FastAPI-based, tool-routed RAG backend. It combines document retrieval, web search, direct LLM answers, session memory, source metadata, and SSE streaming behind API endpoints.

## System Shape

```text
Client / test frontend / API consumer
    |
    | HTTP + SSE
    v
FastAPI app: backend/main.py
    |
    +-- Auth and user dependencies
    +-- Documents API
    +-- Chat SSE API
    +-- Research JSON API
    +-- Sessions and feedback APIs
    |
    v
Agent router: backend/agent/router.py
    |
    +-- Native tool-call decision when provider supports it
    +-- JSON routing fallback for unsupported providers
    +-- Deterministic low-confidence web escalation
    +-- Final answer synthesis
    |
    +-- retrieve_from_document_impl -> Retriever
    +-- web_search_impl             -> Tavily
    +-- summarize_document_impl     -> indexed chunks + LLM
    +-- generate_quiz_impl          -> indexed chunks + LLM
    +-- zero tool calls             -> direct answer path
```

## Main Data Stores

- PostgreSQL stores users, sessions, messages, documents, chunks, feedback, and pgvector embeddings.
- `pgvector` enables vector similarity search directly from the `chunks.embedding` column.
- BM25 indexes are persisted as pickle files under `backend/data/bm25`.
- Uploaded source files are saved under `backend/data/uploads`.

## LLM Providers

`backend/services/llm_provider.py` supports:

- `github` through an OpenAI-compatible client.
- `groq` through an OpenAI-compatible client.
- `ollama` through `langchain_ollama`.

GitHub/Groq use native tool-call responses through `LLMProvider.tool_call(...)`. Ollama currently uses the router's strict JSON fallback unless a compatible local tool-call adapter is added.

## Indexing Flow

```text
POST /api/v1/documents/upload
    |
    +-- authenticate user
    +-- save uploaded file
    +-- create Document row with status="queued"
    +-- optionally link document to session_id
    +-- create in-memory indexing job
    +-- schedule _run_index_job_async(...)

Indexer.index()
    |
    +-- load PDF with PyPDFLoader
    +-- split pages with RecursiveCharacterTextSplitter
    |      chunk_size=600, chunk_overlap=150
    +-- create deterministic chunk_id from source/page/text hash
    +-- build BM25 from chunk tokens
    +-- save BM25 pickle to backend/data/bm25/{document_public_id}_bm25.pkl
    +-- embed chunk texts with text-embedding-3-small
    +-- insert Chunk rows with page text, metadata, and pgvector embedding
    +-- mark Document status="indexed"
```

Important detail: `Retriever` currently looks for BM25 files by numeric document id, while `Indexer` writes them by document public id. If BM25 is not loading during retrieval, this naming mismatch is the first thing to check.

## Retrieval Flow

```text
retrieve_from_document_impl(query, document_ids, db)
    |
    v
Retriever(document_ids, db).similarity_search(query)
    |
    +-- sanitize query
    +-- embed query
    +-- pgvector cosine-distance search over selected document ids
    +-- BM25 keyword search when a BM25 index is available
    +-- reciprocal-rank fusion merge
    +-- CrossEncoder rerank top merged chunks
    +-- return structured chunks, raw confidence, used_vector_db
```

Returned chunks include:

```python
{
    "chunk_id": "...",
    "page_label": 4,
    "source": "...",
    "text": "...",
    "bm25_score": 1.2,
    "reranker_score": 3.4,
    "vector_score": 0.82,
}
```

The agent router normalizes raw reranker confidence with a sigmoid before exposing it to SSE metadata or API responses.

## Chat Flow

```text
POST /api/v1/chat/{session_id}
    |
    +-- validate session exists and belongs to current user
    +-- save user message
    +-- load recent history with limit=6
    +-- pass only role/content history into answer_query(...)
    +-- load document ids linked to the session
    +-- run agent router
    +-- stream answer as SSE token events
    +-- save assistant message
    +-- emit metadata event
```

History passed to the LLM is intentionally compact:

```python
{"role": "user" | "assistant", "content": "..."}
```

It does not include chunks, citations, confidence, vector scores, debug objects, or source metadata.

SSE event shape:

```text
event: token
data: partial text

event: metadata
data: {"confidence": ..., "sources": [...], "chunks": [...], "tool_trace": [...]}

event: done
data: [DONE]
```

## Research Flow

`POST /api/v1/research` is non-streaming and report/API oriented.

```text
ResearchRequest(topic, collection?, include_web, output_format)
    |
    +-- authenticate user
    +-- if collection is provided, resolve it as document public_id or numeric id
    +-- run answer_query(...) with those document ids
    +-- return ResearchResponse
```

`/research` uses the same agent brain as chat but does not require a session and does not write chat history.

## Tool Routing

The LLM-visible tools are:

- `retrieve_from_document(query)`
- `web_search(query)`
- `summarize_document(query)`
- `generate_quiz(query, num_questions=5)`

There is no LLM-visible `direct_answer` tool. For simple greetings, math, or general questions, the native tool-call response should contain zero tool calls. The router records this as:

```json
"tool_trace": ["none"]
```

For providers that fall back to JSON routing, `"tools": ["none"]` means the same thing.

Backend-only context injection:

```python
await retrieve_from_document_impl(
    query=tool_call["args"]["query"],
    document_ids=session_document_ids,
    db=db,
)
```

The model never receives `db`, `document_ids`, `user_id`, session ids, JWTs, or API keys.

## Design Decisions

### Postgres + pgvector instead of Qdrant

The current code stores embeddings in PostgreSQL with the `pgvector` extension. This keeps document metadata, user ownership, chunks, sessions, and vectors in one database boundary.

### Hybrid retrieval

Vector search catches semantic matches. BM25 catches exact terms, names, and keywords. Reciprocal rank fusion merges both result sets while deduplicating by `chunk_id`.

### Reranking

A CrossEncoder reranker scores query/chunk pairs after hybrid retrieval. The top reranker score is used as raw retrieval confidence, then the agent normalizes it for the API/UI.

### Tool schema vs implementation

`@tool` functions in `backend/agent/tools.py` define safe public schemas for the model. They deliberately raise if called directly. Real execution happens in `*_impl()` functions, where Python injects trusted backend state.

### Direct answers as zero tool calls

Direct answers are not modeled as a tool because no external capability is needed. This matches the PRD: the model either calls a tool or answers directly.

### Chat vs research

`/chat` is conversational, session-based, and streamed.

`/research` is stateless, non-streaming, and structured for API consumers.

## Known Gaps

- Chat streams a completed answer split by words, not true provider token streaming.
- `/research` follow-up questions are currently empty.
- `/research` key findings use simple text splitting.
- BM25 index file naming should be verified because indexing and retrieval currently appear to use different identifiers.
- In-memory indexing jobs are lost on server restart.
- Guardrails, RAGAS evals, and groundedness checks are not implemented yet.
