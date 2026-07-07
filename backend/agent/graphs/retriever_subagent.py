from __future__ import annotations

import json
import math

from langgraph.graph import END, StateGraph

from backend.agent.graphs.state import RetrieverState
from backend.agent.tools import retrieve_from_document_impl
from backend.services.llm_provider import LLMProvider


llm = LLMProvider()


def _normalize_confidence(score: float | None) -> float:
    if score is None:
        return 0.0
    try:
        normalized = 1.0 / (1.0 + math.exp(-float(score)))
        return max(0.0, min(1.0, normalized))
    except (OverflowError, TypeError, ValueError):
        return 0.0


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
    prompt = f"""Decide whether this retrieval query should be split into subqueries.

Return JSON only:
{{"needs_decomposition": true, "subqueries": ["q1", "q2"]}}

Use needs_decomposition=false for simple, direct document questions.

Query: {state["query"]}"""
    response = await llm.invoke([{"role": "user", "content": prompt}])
    result = _parse_json_object(response)
    return {
        "needs_decomposition": bool(result.get("needs_decomposition", False)),
        "subqueries": result.get("subqueries", []),
    }


def route_decomposition(state: RetrieverState) -> str:
    return "parallel_retrieval" if state.get("needs_decomposition") else "simple_retrieval"


async def _retrieve(query: str, state: RetrieverState) -> dict:
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

    subqueries = state.get("subqueries") or [state["query"]]
    for subquery in subqueries:
        result = await _retrieve(subquery, state)
        all_chunks.extend(result.get("chunks", []))
        if result.get("confidence") is not None:
            confidence_scores.append(float(result["confidence"]))
        used_vector_db = used_vector_db or bool(result.get("used_vector_db", False))

    confidence = max(confidence_scores) if confidence_scores else 0.0
    return {
        "parallel_chunks": [all_chunks],
        "merged_chunks": all_chunks,
        "deduplicated_chunks": _deduplicate_chunks(all_chunks),
        "reranked_chunks": _deduplicate_chunks(all_chunks),
        "confidence": confidence,
        "used_vector_db": used_vector_db,
    }


async def simple_retrieval(state: RetrieverState) -> dict:
    result = await _retrieve(state["query"], state)
    chunks = result.get("chunks", [])
    confidence = result.get("confidence")
    return {
        "hybrid_chunks": chunks,
        "reranked_chunks": chunks,
        "confidence": float(confidence) if confidence is not None else 0.0,
        "used_vector_db": bool(result.get("used_vector_db", False)),
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


def check_evidence(state: RetrieverState) -> dict:
    confidence = _normalize_confidence(state.get("confidence"))
    enough = (
        len(state.get("reranked_chunks", [])) >= 3
        and confidence >= 0.4
    )
    return {"enough_evidence": enough}


def route_evidence(state: RetrieverState) -> str:
    if state.get("enough_evidence") or state.get("retry_count", 0) >= state.get("max_retries", 2):
        return END
    return "broaden_query"


async def broaden_query(state: RetrieverState) -> dict:
    prompt = f"""The query returned insufficient document evidence.
Rewrite it to be broader and more likely to find relevant chunks.
Return only the rewritten query.

Original: {state["query"]}"""
    response = await llm.invoke([{"role": "user", "content": prompt}])
    return {
        "query": response.strip(),
        "retry_count": state.get("retry_count", 0) + 1,
        "hybrid_chunks": [],
        "parallel_chunks": [],
        "merged_chunks": [],
        "deduplicated_chunks": [],
        "reranked_chunks": [],
    }


builder = StateGraph(RetrieverState)

builder.add_node("check_decomposition", check_decomposition)
builder.add_node("simple_retrieval", simple_retrieval)
builder.add_node("parallel_retrieval", parallel_retrieval)
builder.add_node("check_evidence", check_evidence)
builder.add_node("broaden_query", broaden_query)

builder.set_entry_point("check_decomposition")
builder.add_conditional_edges(
    "check_decomposition",
    route_decomposition,
    {
        "simple_retrieval": "simple_retrieval",
        "parallel_retrieval": "parallel_retrieval",
    },
)
builder.add_edge("simple_retrieval", "check_evidence")
builder.add_edge("parallel_retrieval", "check_evidence")
builder.add_conditional_edges(
    "check_evidence",
    route_evidence,
    {
        END: END,
        "broaden_query": "broaden_query",
    },
)
builder.add_edge("broaden_query", "check_decomposition")

retrieval_agent = builder.compile()
