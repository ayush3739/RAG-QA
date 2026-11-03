---
name: rag-eval
description: Evaluation, benchmarking, and diagnosis protocol for KnowVex RAG retrieval, CrossEncoder reranking, and citation grounding. Use when evaluating answer quality, measuring retrieval accuracy, debugging low confidence scores, or tuning BM25/vector weights.
---

# KnowVex RAG Evaluation & Benchmarking Protocol

This skill guides agents in evaluating and tuning the retrieval pipeline (`backend/core/retriever.py`, `backend/agent/router.py`).

---

## 1. The 4 Golden Evaluation Metrics

1. **Context Precision (Did we find the right chunks?):**
   - High precision = the top 3-5 chunks contain the exact factual answer.
   - Diagnosing low precision: Increase BM25 weight or check chunk size/overlap in `config.py`.
2. **Context Recall (Did we miss relevant information?):**
   - Did the top 20 candidate chunks include all necessary pages?
   - Diagnosing low recall: Increase `vector_output_chunks` (e.g. from 15 to 25) before RRF fusion.
3. **CrossEncoder Rerank Delta:**
   - Did the CrossEncoder move the truly relevant chunk from rank 12 up to rank 1 or 2?
   - If not, verify query-chunk concatenation format and language/domain mismatch.
4. **Citation Faithfulness & Grounding:**
   - Is every statement in the generated answer directly supported by an excerpt in `citations`?
   - If hallucinated statements appear, tighten the system prompt: *"Answer strictly and solely based on the provided document excerpts. If unsure, state that the documents do not provide this information."*

---

## 2. Running Verification

```bash
# Execute local retrieval test and inspect structured output
python tests/rag_output.py
```

Inspect output for:
- `confidence`: Must exceed threshold (`>= 0.3` by default).
- `citations`: Must have valid `page`, `document_name`, and non-empty `excerpt`.
- `intent`: Verify if the agent router correctly selected `retrieve_doc`, `web_search`, or `direct`.

