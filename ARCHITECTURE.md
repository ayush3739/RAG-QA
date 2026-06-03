# 🏗️ DocuMind Architecture

> System design, data flow, and design decisions for Phase 1 (RAG Core)

---

## 🔄 End-to-End Flow

### Indexing Flow

```
User uploads PDF
    ↓
[Indexer.__init__] — check Qdrant connection, init BM25
    ↓
[index(pdf_path)] — main entry point
    ├─ _delete_existing_collection() — wipe old chunks
    ├─ _load_pdf() → extract pages (PyPDFLoader)
    ├─ _chunk_text() → split with RecursiveCharacterTextSplitter
    │  └─ chunk_size=600, overlap=150
    ├─ make_chunk_id(page_num, chunk_idx) → deterministic ID
    │  └─ format: "page_<N>_chunk_<I>"
    ├─ Embed all chunks (OpenAIEmbeddings)
    ├─ Save to Qdrant (QdrantVectorStore)
    ├─ Build BM25 from tokens
    └─ Persist BM25 to data/bm25/{collection}_bm25.pkl
    ↓
✅ Collection indexed, ready for queries
```

### Retrieval Flow

```
User asks question
    ↓
[answer(query: str, k: int = 10)]
    ├─ sanitize_query() → check length, block injections
    └─ similarity_search() → hybrid retrieval
        ├─ Parallel searches:
        │  ├─ vector_db.similarity_search(query) → ~15 Qdrant results
        │  └─ bm25.get_scores() → ~15 BM25 results
        │
        ├─ reciprocal_rank_fusion() → merge by chunk_id
        │  └─ RRF formula: 1/(k + rank) per source
        │  └─ dedup by chunk_id to avoid duplicates
        │
        ├─ rerank_(query, merged_results) → CrossEncoder
        │  └─ Predict scores for top 15 (query, chunk) pairs
        │  └─ Attach reranker_score to metadata
        │  └─ Sort by score, keep top 10
        │
        └─ structured_chunks = [{chunk_id, page, text, bm25_score, reranker_score, vector_score}]
    ↓
[generate_response(query, retrieval_result)]
    ├─ sanitize_query() — defensive sanitization
    ├─ Build context from top 6 chunks
    ├─ System prompt with rules
    │  └─ "Answer ONLY from context, cite pages, <200 words"
    ├─ Call LLM (gpt-4o-mini or Ollama)
    │  └─ LLM returns plain text (no JSON mode)
    ├─ Build citations from top 5 chunks
    │  └─ {chunk_id, source, page_label, excerpt}
    └─ Return structured dict:
       ├─ answer (str)
       ├─ citations (list)
       ├─ chunks (full metadata)
       ├─ confidence (float)
       └─ debug (errors, qdrant status)
    ↓
[Frontend pretty-printer]
    ├─ Shows "[Model-produced answer]" section
    ├─ Shows "[Sources & metadata added by retriever]" section
    └─ Prints sources with page, scores, excerpts
```

---

## 🔑 Design Decisions & Trade-offs

### 1. Hybrid Search (BM25 + Vector)

**Why:** BM25 catches exact/keyword matches; vector catches semantic similarity.
- BM25 excels at: named entities, acronyms, exact phrases
- Vector excels at: paraphrased questions, semantic concepts

**How:** RRF (Reciprocal Rank Fusion) merges results fairly:
- Formula: `score = Σ 1/(k + rank)` per result
- Avoids one method dominating

**Trade-off:** Slower than vector-only, but more reliable for diverse queries.

---

### 2. Deterministic `chunk_id`

**Format:** `"page_<N>_chunk_<I>"`  
Example: `"page_5_chunk_3"` = page 5, chunk 3 within that page

**Why:**
- Stable across BM25 and Qdrant (different systems, same ID)
- RRF deduplication: two searches returning same chunk don't duplicate
- UI traceability: trace answer back to exact location

**Trade-off:** Requires encoding page number + chunk index; small overhead.

---

### 3. CrossEncoder Reranking

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`  
**Why:** Fine-tuned on MS Marco dataset; predicts query-passage relevance directly

**vs LLM embedding distance:**
- Reranker: direct relevance prediction
- Embedding distance: proxy for similarity

**Trade-off:** Extra model inference ~10-50ms, but improves answer quality significantly.

---

### 4. Query Sanitization (Prompt Injection Defense)

**Layers:**
1. Max length: 1000 chars (prevent token bloat)
2. Pattern blocking: "ignore all instructions", "ignore previous", etc.
3. Defensive sanitization in both `answer()` and `generate_response()`

**Why layered:**
- Defense in depth (fails safely)
- Even if first layer misses, second catches it

**Trade-off:** Might block legitimate queries with keywords; acceptable for security.

---

### 5. Qdrant Fallback to BM25-Only

**Flow:**
```
try:
  vector_db = QdrantVectorStore.from_existing_collection(...)
  qdrant_available = True
except:
  vector_db = None
  qdrant_available = False
  proceed with BM25 only
```

**Why:** 
- Dev environment: Qdrant not always running
- Production: network issues shouldn't crash the app

**Trade-off:** BM25-only is slower/less semantic, but better than complete failure.

---

### 6. JSON Assembly in Python (Not LLM)

**Before:** LLM produces `response_format={"type":"json_object"}` JSON
**Now:** LLM produces plain text, Python builds JSON

**Why:**
- LLM JSON mode costs 2-3x more tokens
- Python assembly is deterministic & reliable
- Easier to add/remove fields without prompt engineering

**Trade-off:** LLM doesn't control output format; but we enforce via system prompt.

---

### 7. Confidence Scoring

**Current:** max reranker score from top-10 results
- Range: typically -15 to +20 (reranker trains on 0-5 but can exceed)
- Negative = low confidence
- Positive = high confidence

**UI Interpretation:**
- `> 5` → 🟢 High
- `0-5` → 🟡 Medium
- `< 0` → 🔴 Low

**Trade-off:** Heuristic, not calibrated; good enough for relative ordering.

---

## 🗂️ Data Structures

### Chunk Metadata

```python
{
  "chunk_id": "page_5_chunk_2",        # deterministic ID
  "page_label": "5",                   # page number
  "page_content": "...",               # actual text
  "source": "docs/report.pdf",         # filename
  "bm25_score": 12.34,                 # if from BM25
  "reranker_score": 1.82,              # if reranked
  "vector_score": None,                # TODO: capture from Qdrant
}
```

### Retrieval Result Payload

```python
{
  "chunks": [chunk_metadata, ...],
  "used_vector_db": True,
  "debug": {"qdrant_error": None},
  "confidence": 1.82,
}
```

### Final Response

```python
{
  "answer": "...",                     # LLM-produced answer
  "citations": [
    {
      "chunk_id": "page_5_chunk_2",
      "source": "report.pdf",
      "page_label": "5",
      "excerpt": "...",                # 20-30 word quote
    }
  ],
  "chunks": [full_chunk_metadata, ...], # top 10 with all scores
  "confidence": 1.82,
  "used_vector_db": True,
  "debug": {"qdrant_error": None},
}

---

## Database & Schema Placement

- Persistence infra (engine, sessions, Base) is in `backend/db/base.py`.
- ORM models (tables like `sessions` and `messages`) should be placed in `backend/models/models.py` and import `Base` from `backend.db.base`.
- Alembic (migrations) should be configured to reference `backend.models.models` as the place that defines metadata for autogeneration.
```

---

## 🔌 External Dependencies

| Service | Purpose | Fallback |
|---------|---------|----------|
| **Postgres + pgvector** | Vector store (pgvector extension) and session storage | BM25-only mode / SQLite for dev |
| **OpenAI API** | Embeddings (text-embedding-3-large) | ❌ Required |
| **GitHub Model** | LLM (gpt-4o-mini) | Ollama (local) |
| **Ollama** | Local LLM alternative | ❌ Required if no GitHub token |

---

## 📊 Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Index PDF (10 pages) | ~2-5s | Embedding + Qdrant write |
| Vector search | ~50-100ms | Qdrant similarity_search |
| BM25 search | ~10-20ms | In-memory scoring |
| RRF merge | ~5ms | Small merge operation |
| Reranking | ~50-200ms | 15 pairs × CrossEncoder |
| LLM generation | ~1-3s | Network + inference |
| **Total retrieval** | **~2-4s** | Parallel searches + LLM |

---

## 🛡️ Error Handling

| Scenario | Behavior |
|----------|----------|
| **Qdrant down** | Use BM25-only, set `qdrant_available=False` |
| **No embeddings API** | Fail at startup with clear error |
| **Query too long** | Truncate to 1000 chars |
| **Malicious query** | Raise `ValueError`, return error dict |
| **Reranker fails** | Fall back to top-10 unranked |
| **LLM timeout** | Raise `RuntimeError` to caller |

---

## 🔮 Future Improvements

1. **Vector score capture** — extract similarity distance from Qdrant
2. **Adaptive confidence** — calibrate via feedback loop
3. **Multi-language support** — detect & translate if needed
4. **Caching layer** — Redis for frequent queries
5. **Streaming retrieval** — don't wait for full reranking
6. **Graph-based retrieval** — chunk relationships via LangGraph

---

**Last Updated:** May 19, 2026
