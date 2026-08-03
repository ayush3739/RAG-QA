# Core

The `backend/core/` folder contains the lower-level RAG pieces used by the API and agent layers: configuration, PDF indexing, document retrieval, and shared utilities.

## Files

```text
backend/core/
├── config.py      Pydantic settings loaded from `.env`
├── indexer.py     PDF -> chunks -> BM25 -> embeddings -> Chunk rows
├── retriever.py   pgvector + BM25 retrieval, RRF merge, CrossEncoder rerank
└── utils.py       BM25 tokenizer and shared reranker instance
```

## `config.py`

Defines the global `settings` object with `pydantic-settings`.

Important fields:

- `DATABASE_URL`
- `github_token`
- `groq_api_key`
- `llm_provider`
- `llm_model`
- `secret_key`
- `algorithm`
- `access_token_expire_minutes`
- `ollama_base_url`
- `ollama_model`
- `tavily_api_key`
- `enable_web_search`
- `top_k`
- `rerank_top_n`
- `confidence_threshold`
- `chunk_size`
- `chunk_overlap`
- mail settings

`.env` is loaded from the repository root.

## `indexer.py`

Main class:

```python
Indexer(
    file_path: str,
    db_session: AsyncSession,
    document_id: int,
    document_public_id: int,
)
```

### What It Does

`Indexer.index()`:

1. Loads the PDF with `PyPDFLoader`.
2. Splits pages with `RecursiveCharacterTextSplitter`.
3. Uses chunk size `600` and overlap `150`.
4. Creates a deterministic `chunk_id` from `source | page_label | normalized_text`.
5. Persists a BM25 index to `data/bm25/{document_public_id}_bm25.pkl`.
6. Embeds all chunks with `text-embedding-3-small`.
7. Inserts `Chunk` rows into PostgreSQL.
8. Rolls back the DB transaction and raises `RuntimeError` if indexing fails.

### Stored Chunk Fields

Each indexed chunk becomes a `Chunk` ORM row with:

- `document_id`
- `chunk_id`
- `page_content`
- `page_number`
- `chunk_index`
- `source`
- `embedding`

### BM25 File

The BM25 pickle stores:

```python
{
    "bm25": BM25Okapi(...),
    "meta": [
        {
            "chunk_id": "...",
            "page_content": "...",
            "page_label": "...",
            "source": "...",
        }
    ]
}
```

Important: `Indexer` writes the BM25 file by `document_public_id`. `Retriever` currently looks for files by numeric document id. If BM25 search is not loading, align these identifiers.

## `retriever.py`

Main class:

```python
Retriever(document_ids: list[int], db: AsyncSession)
```

The retriever is scoped to one or more document ids and an async database session.

### Initialization

On init it:

- Reads GitHub token and settings.
- Creates an OpenAI-compatible client for GitHub Models.
- Creates an Ollama LLM instance.
- Creates an OpenAI embedding model.
- Loads BM25 metadata for selected documents if files exist.
- Builds a combined BM25 corpus.
- Reuses `RERANKER` from `utils.py`.

### `sanitize_query(query)`

Basic query guard:

- Truncates to 1000 characters.
- Blocks simple prompt-injection phrases:
  - `ignore all instructions`
  - `ignore previous`
  - `you are now`

### `similarity_search(query, k=10)`

Main retrieval method used by the agent tool implementation.

Flow:

1. Sanitize query.
2. Embed query.
3. Run pgvector cosine-distance search:
   ```python
   Chunk.embedding.cosine_distance(query_embedding)
   ```
4. Build vector result objects with:
   - `chunk_id`
   - `page_label`
   - `source`
   - `vector_score`
5. Run BM25 search if a BM25 corpus exists.
6. Merge vector and BM25 results with reciprocal rank fusion.
7. Rerank top merged results with CrossEncoder.
8. Return structured chunks and raw confidence.

Return shape:

```python
{
    "chunks": [
        {
            "chunk_id": "...",
            "page_label": 4,
            "source": "...",
            "text": "...",
            "bm25_score": 1.23,
            "reranker_score": 4.56,
            "vector_score": 0.82,
            "bm25_len": 15,
        }
    ],
    "used_vector_db": True,
    "debug": {},
    "confidence": 4.56,
}
```

### `reciprocal_rank_fusion(vector_results, bm25_results, k=60)`

Merges ranked vector and BM25 lists.

Deduplication key:

- Prefer `metadata["chunk_id"]`.
- Fall back to `page_content`.

Score:

```text
1 / (k + rank)
```

### `rerank_(query, chunks, top_n=5)`

Uses the shared CrossEncoder to score `(query, chunk_text)` pairs.

Returns:

```python
(ranked_chunks[:top_n], max_score)
```

If no chunks are available, returns:

```python
([], 0.0)
```

### `generate_response(...)` and `answer(...)`

These methods still support direct RAG answering from the retriever itself:

- `answer(query)` calls retrieval and then `generate_response(...)`.
- `generate_response(...)` builds a strict document-context prompt and calls GitHub Models.

The main chat and research APIs now usually go through `backend/agent/router.py`, which uses `similarity_search(...)` through `retrieve_from_document_impl(...)`.

## `utils.py`

### `simple_tokenize(text)`

Lowercases text and extracts word tokens for BM25:

```python
re.findall(r"\w+", text.lower())
```

### `RERANKER`

Loads:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

with `local_files_only=True`.

If the model is not available locally, `RERANKER` becomes `None` and retrieval reranking can fail unless handled by the caller. In the current retriever, rerank failures fall back to the pre-reranked merged results.

## Core Data Flow

```text
Upload PDF
    -> documents.py creates Document
    -> Indexer.index()
    -> Chunk rows + BM25 pickle

Chat/research question
    -> agent router chooses retrieve_from_document
    -> retrieve_from_document_impl(...)
    -> Retriever.similarity_search(...)
    -> chunks returned to router
    -> router synthesizes final answer
```

## Operational Notes

- The codebase currently uses PostgreSQL + pgvector for vector search, not Qdrant.
- `settings.qdrant_url` remains in config but is not the active vector store path.
- Query history is handled above core, in `ChatService` and `agent/router.py`.
- Source/citation shaping for chat metadata is handled in `agent/router.py`, not in `Retriever.similarity_search(...)`.
- Indexing runs asynchronously from the documents route using its own DB session.
