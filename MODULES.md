# Core Modules — Indexer, Retriever, Utils

> Phase 1 documentation for the RAG pipeline internals.

---

## Modules

### `indexing.py`

Builds the document index.

- Loads PDFs
- Splits text into chunks
- Assigns deterministic `chunk_id`
- Writes vectors to Qdrant
- Builds and persists BM25

### `retrieving.py` / `backend/core/retriever.py`

Handles retrieval and answer generation.

- Sanitizes user queries
- Runs hybrid search (BM25 + vector)
- Merges results with RRF
- Reranks with CrossEncoder
- Calls the LLM
- Returns structured answer JSON

### `utils.py`

Small helper functions.

- Tokenization
- Context formatting
- Shared utility functions

---

## Important Behaviors

- `chunk_id` is the stable key across retrieval systems.
- Query sanitization runs before retrieval and before generation.
- Qdrant can fail gracefully; BM25-only mode still works.
- The LLM should answer only from provided context.

---

## Retrieval Output Shape

```python
{
  "answer": "...",
  "citations": [...],
  "chunks": [...],
  "confidence": 0.82,
  "used_vector_db": True,
  "debug": {"qdrant_error": None},
}
```

---

## What To Read First

If you are editing core behavior, start with:

1. `ARCHITECTURE.md`
2. `MODULES.md`
3. The specific source file you need to change
