"""
retriever_subagent.py

Current Subagent Architecture:

        check_decomposition
           (Analyze Intent)
            /           \
     [No]  /             \ [Yes]
          /               \
simple_retrieval     parallel_retrieval
  (Dual-Query)         (Dual-Query x N)
          \               /
           \             /
            ---- END ----

*Note: Semantic rewrite, BM25, Vector Search, Fusion, and Cross-Encoder Reranking 
are now encapsulated efficiently inside the Retriever class rather than the graph.*
"""
from __future__ import annotations

import json
import math

from langgraph.graph import END, StateGraph

from backend.agent.graphs.state import RetrieverState
from backend.agent.tools import retrieve_from_document_impl
from backend.core.retriever import RetrievalQuery
from backend.services.llm_provider import LLMProvider


llm = LLMProvider()


def _parse_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}


async def check_decomposition(state: RetrieverState) -> dict:
    original_query = state["query"]
    prompt = f"""Decide whether this retrieval query should be split into subqueries.
Also, optimize each query (or subquery) by creating a 'semantic_query' for vector search and a 'keyword_query' for BM25 search.

Rules for Semantic Query:
- Preserve the exact meaning of the user's question, but convert it into a declarative statement or highly descriptive natural language phrase.
- DO NOT strip out specific numbers (e.g., "three states"), nouns, or unique constraints. 
- Example: "According to the document, what are the three states a Kubernetes Job can have?" -> "The three lifecycle states of a Kubernetes Job."

Rules for Keyword Query:
- Extract ONLY the most unique, dense, and important keywords.
- ALWAYS include critical constraints like numbers ("three"), acronyms, or specific proper nouns.
- Example: "Kubernetes Job three states status"

Return JSON only in this exact format:
{{
    "needs_decomposition": true,
    "queries": [
        {{
            "original_query": "...",
            "semantic_query": "...",
            "keyword_query": "..."
        }}
    ]
}}

If it's a simple, direct question, use needs_decomposition=false and provide exactly one query object in the list.

User Query: {original_query}"""
    response = await llm.invoke([{"role": "user", "content": prompt}])
    result = _parse_json_object(response)
    
    raw_queries = result.get("queries", [])
    queries = []
    for q in raw_queries:
        try:
            queries.append(RetrievalQuery(**q))
        except Exception:
            pass
            
    if not queries:
        # Fallback if LLM fails formatting
        queries = [RetrievalQuery(
            original_query=original_query if isinstance(original_query, str) else original_query.original_query,
            semantic_query=original_query if isinstance(original_query, str) else original_query.original_query,
            keyword_query=original_query if isinstance(original_query, str) else original_query.original_query,
        )]

    print(f"\n--- DEBUG check_decomposition ---")
    print(f"Needs Decomposition: {result.get('needs_decomposition', False)}")
    for i, q in enumerate(queries):
        print(f"Query {i+1}:")
        print(f"  Semantic: {q.semantic_query}")
        print(f"  Keyword:  {q.keyword_query}")
    print(f"---------------------------------\n")

    return {
        "needs_decomposition": bool(result.get("needs_decomposition", False)),
        "queries": queries,
    }


def route_decomposition(state: RetrieverState) -> str:
    return "parallel_retrieval" if state.get("needs_decomposition") else "simple_retrieval"


async def _retrieve(query: str | RetrievalQuery, state: RetrieverState) -> dict:
    document_ids = state.get("document_ids", [])
    db = state.get("db")
    if not document_ids or db is None:
        return {
            "chunks": [],
            "confidence": 0.0,
            "used_vector_db": False,
        }
    return await retrieve_from_document_impl(
        query=query,
        document_ids=document_ids,
        db=db,
    )


async def parallel_retrieval(state: RetrieverState) -> dict:
    all_chunks: list[dict] = []
    confidence_scores: list[float] = []
    used_vector_db = False

    queries = state.get("queries") or [state["query"]]
    for q in queries:
        result = await _retrieve(q, state)
        all_chunks.extend(result.get("chunks", []))
        if result.get("confidence") is not None:
            confidence_scores.append(float(result["confidence"]))
        used_vector_db = used_vector_db or bool(result.get("used_vector_db", False))

    confidence = max(confidence_scores) if confidence_scores else 0.0
    return {
        "reranked_chunks": _deduplicate_chunks(all_chunks),
        "confidence": confidence,
        "used_vector_db": used_vector_db,
        "retriever_debug": {},
    }


async def simple_retrieval(state: RetrieverState) -> dict:
    queries = state.get("queries")
    query = queries[0] if queries else state["query"]
    
    result = await _retrieve(query, state)
    chunks = result.get("chunks", [])
    confidence = result.get("confidence")
    return {
        "reranked_chunks": chunks,
        "confidence": float(confidence) if confidence is not None else 0.0,
        "used_vector_db": bool(result.get("used_vector_db", False)),
        "retriever_debug": result.get("debug", {}),
    }


def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []
    for chunk in chunks:
        key = str(chunk.get("chunk_id") or chunk.get("text", ""))[:160]
        if key and key not in seen:
            seen.add(key)
            deduped.append(chunk)
    return deduped



builder = StateGraph(RetrieverState)

builder.add_node("check_decomposition", check_decomposition)
builder.add_node("simple_retrieval", simple_retrieval)
builder.add_node("parallel_retrieval", parallel_retrieval)

builder.set_entry_point("check_decomposition")
builder.add_conditional_edges(
    "check_decomposition",
    route_decomposition,
    {
        "simple_retrieval": "simple_retrieval",
        "parallel_retrieval": "parallel_retrieval",
    },
)
builder.add_edge("simple_retrieval", END)
builder.add_edge("parallel_retrieval", END)

retrieval_agent = builder.compile()
