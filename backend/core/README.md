# 🎯 Core Modules — RAG Pipeline Components

> Phase 1 core: Indexer, Retriever, Utils. These are the heart of the RAG system.

---

## 📚 Module Overview

| Module | Purpose | Status |
|--------|---------|--------|
| **indexer.py** | PDF → chunks with IDs → Qdrant + BM25 | ✅ Complete |
| **retriever.py** | Hybrid search, reranking, LLM generation | ✅ Complete |
| **utils.py** | Tokenizer, helpers | ✅ Complete |
| **config.py** | Pydantic settings (Phase 2) | 🚀 WIP |

---

## 📖 `indexer.py` — PDF Indexing

### Class: `Indexer`

```python
Indexer(file_path: str, collection_name: str = None)
```

**Attributes:**
- `file_path` — Path to PDF file
- `collection_name` — Name for Qdrant collection (defaults to filename)
- `loader` — PyPDFLoader instance
- `splitter` — RecursiveCharacterTextSplitter (600/150)
- `embedding_model` — OpenAIEmbeddings
- `vector_db` — QdrantVectorStore
- `bm25` — BM25Okapi instance

### Key Methods

#### `index() → None`
Main entry point to index a PDF.

**Steps:**
1. Delete existing collection in Qdrant (no duplicates)
2. Load PDF pages
3. Split into chunks (600 chars, 150 overlap)
4. Create deterministic chunk_id per chunk
5. Embed chunks via OpenAI API
6. Save to Qdrant
7. Build BM25 index from chunk texts
8. Pickle BM25 to `data/bm25/{collection_name}_bm25.pkl`

**Example:**
```python
indexer = Indexer("docs/report.pdf")
indexer.index()  # ~2-5s for 10-page PDF
```

#### `make_chunk_id(page_num: int, chunk_idx: int) → str`
Generate deterministic chunk identifier.

**Format:** `"page_<page_num>_chunk_<chunk_idx>"`

**Example:** `"page_5_chunk_2"` = page 5, chunk 2

**Why:** Stable across BM25 and Qdrant; enables deduplication in RRF.

#### `_delete_existing_collection() → None`
Wipe collection from Qdrant before re-indexing (prevents duplicates).

#### `_load_pdf() → list[Document]`
Load PDF pages via PyPDFLoader. Each page includes metadata: page_label, source.

#### `_chunk_text(docs: list) → list[Document]`
Split documents into chunks with RecursiveCharacterTextSplitter.
- chunk_size = 600
- chunk_overlap = 150
- Preserves metadata (page_label, source)

---

## 🔍 `retriever.py` — RAG Pipeline

### Class: `Retriever`

```python
Retriever(collection_name: str)
```

**Attributes:**
- `collection_name` — Qdrant collection to query
- `openai_client` — GitHub model API client
- `llm` — OllamaLLM for local inference
- `embedding_model` — OpenAIEmbeddings
- `vector_db` — QdrantVectorStore (or None if unavailable)
- `qdrant_available` — Boolean flag
- `bm25` — Loaded BM25Okapi model
- `bm25_texts` — Chunk texts for BM25
- `bm25_meta` — Chunk metadata for BM25
- `reranker` — CrossEncoder model

### Key Methods

#### `answer(query: str, k: int = 10) → dict`
Main entry point to get an answer.

**Flow:**
1. Sanitize query
2. Call `similarity_search()` → hybrid retrieval + reranking
3. Call `generate_response()` → LLM generation + citations
4. Return structured JSON

**Response format:**
```python
{
  "answer": "...",
  "citations": [{chunk_id, source, page_label, excerpt}, ...],
  "chunks": [full_metadata, ...],
  "confidence": 0.82,
  "used_vector_db": True,
  "debug": {"qdrant_error": None}
}
```

#### `similarity_search(query: str, k: int = 10) → dict`
Hybrid search: BM25 + vector → RRF → rerank.

**Steps:**
1. Sanitize query
2. Vector search (Qdrant): ~15 results with metadata
3. BM25 search: ~15 results with bm25_score
4. RRF merge by chunk_id (deduplication)
5. CrossEncoder reranking: top 15 → top 10
6. Build structured chunk metadata
7. Return payload with chunks + confidence

**Returns:**
```python
{
  "chunks": [structured_chunk_dict, ...],
  "used_vector_db": bool,
  "debug": {"qdrant_error": str or None},
  "confidence": float
}
```

#### `generate_response(query: str, retrieval_result: dict) → dict`
LLM generation + citation building.

**Steps:**
1. Defensively sanitize query
2. Build context from top 6 chunks
3. System prompt with grounding rules
4. Call LLM (gpt-4o-mini or Ollama)
5. LLM returns plain text (no JSON)
6. Build citations from top 5 chunks
7. Return structured dict

**System prompt enforces:**
- Answer ONLY from context
- Cite page numbers
- Keep under 200 words
- No hallucination
- If not found: "I could not find this information..."

#### `sanitize_query(query: str) → str`
Input validation and injection prevention.

**Checks:**
- Max 1000 characters (truncate if longer)
- Block injection patterns: "ignore all instructions", "ignore previous", "you are now"
- Raise ValueError if malicious

#### `reciprocal_rank_fusion(vector_results, bm25_results, k: int = 60) → list[Document]`
Merge two ranked lists fairly using RRF.

**Formula:**
```
score = Σ 1/(k + rank)  per result
```

**Deduplication:** By chunk_id (stable across sources)

**Returns:** Merged + sorted by RRF score

#### `rerank_(query: str, chunks: list, top_n: int = 5) → tuple`
CrossEncoder reranking.

**Steps:**
1. Build (query, chunk_text) pairs
2. Predict relevance scores via CrossEncoder
3. Attach reranker_score to chunk.metadata
4. Sort by score descending
5. Keep top_n

**Returns:** `(ranked_chunks, max_score)`

---

## 🛠️ `utils.py` — Helpers

### Function: `simple_tokenize(text: str) → list[str]`

Simple whitespace + lowercase tokenizer for BM25.

**Example:**
```python
tokens = simple_tokenize("Hello, World!")
# ["hello", "world"]
```

### Function: `format_context(chunks: list) → str`

Format chunk metadata into LLM context string.

---

## 🔐 Query Sanitization

**Two-layer defense:**

1. **In `answer()`:** Sanitize before retrieval & generation
2. **In `generate_response()`:** Defensive sanitization before LLM call

**Protection against:**
- Prompt injection ("ignore all instructions and...")
- Token bloat (>1000 chars)
- Malformed input

---

## 📊 Data Flow in `retriever.py`

```
Query: "What is MongoDB?"
    ↓
sanitize_query() → "what is mongodb?"
    ↓
similarity_search()
  ├─ vector_db.similarity_search() → 15 Document objects
  │  └─ each has metadata: {chunk_id, page_label, source}
  ├─ bm25.get_scores() → 15 BM25 scores
  │  └─ matched to bm25_meta list
  ├─ reciprocal_rank_fusion() → merge by chunk_id
  ├─ rerank_() → CrossEncoder top 10
  └─ structured_chunks = [
       {chunk_id, page_label, source, text, bm25_score, reranker_score, vector_score},
       ...
     ]
    ↓
generate_response()
  ├─ Build context from top 6 chunks
  ├─ System prompt
  ├─ LLM call → answer text
  ├─ Build citations from top 5
  └─ Return:
     {
       "answer": "MongoDB is a...",
       "citations": [...],
       "chunks": [...],
       "confidence": 0.82
     }
```

---

## 🔌 Environment Variables Required

```
GITHUB_TOKEN=ghp_xxx              # Must be set; raises error if missing
QDRANT_URL=http://localhost:6333  # Optional; defaults shown
OLLAMA_BASE_URL=http://localhost:11434
```

---

## 🧪 Testing

### Unit Tests (`tests/test_unit_rag.py`)

```python
# Mocks BM25, reranker, LLM
# Verifies: sanitization, RRF, reranker scores, generate_response
pytest tests/test_unit_rag.py -v
```

### Integration Test (`tests/rag_output.py`)

```python
# Real retriever, real queries
# Pretty-prints output showing [Model-produced answer] vs [Sources & metadata]
python tests/rag_output.py
```

---

## 🚀 Future Enhancements

1. **Vector score capture** — extract Qdrant similarity distance
2. **Caching** — Redis for frequent queries
3. **Multi-language** — detect query language, translate if needed
4. **Streaming** — partial results before full reranking
5. **Adaptive k** — adjust `k` based on query type/confidence

---

## 📋 Checklist for Phase 1 Completion

- [x] Indexer: PDF → chunks with deterministic IDs
- [x] Retriever: Hybrid search (BM25 + vector)
- [x] RRF merge by chunk_id
- [x] CrossEncoder reranking
- [x] Query sanitization (prompt injection defense)
- [x] LLM grounding system prompt
- [x] Citation building
- [x] Confidence scoring
- [x] Qdrant fallback
- [x] Error handling

---

**Last Updated:** May 19, 2026
