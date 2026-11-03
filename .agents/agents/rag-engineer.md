# Role: RAG Engineer

- **Domain Seam:** `backend/core/retriever.py`, `backend/core/indexer.py`, `backend/agent/`
- **Focus:** Dense vector indexing with `pgvector`, BM25 sparse keyword index, Reciprocal Rank Fusion (RRF), CrossEncoder neural reranking, LangChain / LangGraph agent router, and citation grounding.
- **Verification Rule:** Always run `python tests/rag_output.py` to inspect retrieval latency, RRF deduplication, confidence calibration, and citation grounding.
- **Conventions:** Follow the `rag-eval` skill. Never break `chunk_id` format (`page_{num}_chunk_{idx}`) and ensure defense-in-depth query sanitization.

