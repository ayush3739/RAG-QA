from __future__ import annotations

import math
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from backend.agent.events import node_completed, node_started
from backend.agent.graphs.main_graph import main_graph
from backend.agent.graphs.state import create_main_state


def _normalize_confidence(score: float | None) -> float | None:
    if score is None:
        return None
    try:
        normalized = 1.0 / (1.0 + math.exp(-float(score)))
        return round(max(0.0, min(1.0, normalized)), 4)
    except (OverflowError, TypeError, ValueError):
        return None


def _coerce_confidence(score: float | None, route: str) -> float | None:
    if score is None:
        return None
    try:
        numeric = float(score)
    except (TypeError, ValueError):
        return None

    if route in {"document", "research"} or numeric < 0.0 or numeric > 1.0:
        return _normalize_confidence(numeric)
    return round(max(0.0, min(1.0, numeric)), 4)


def _doc_sources(chunks: list[dict]) -> list[dict]:
    return [
        {
            "type": "document",
            "chunk_id": chunk.get("chunk_id"),
            "page": chunk.get("page_label"),
            "source": chunk.get("source"),
            "excerpt": chunk.get("text", "")[:240],
            "reranker_score": chunk.get("reranker_score"),
            "vector_score": chunk.get("vector_score"),
            "bm25_score": chunk.get("bm25_score"),
        }
        for chunk in chunks[:8]
    ]


def _metadata_chunks(chunks: list[dict]) -> list[dict]:
    return [
        {
            "type": "document",
            "chunk_id": chunk.get("chunk_id"),
            "page": chunk.get("page_label"),
            "source": chunk.get("source"),
            "text": chunk.get("text", "")[:1000],
            "reranker_score": chunk.get("reranker_score"),
            "vector_score": chunk.get("vector_score"),
            "bm25_score": chunk.get("bm25_score"),
        }
        for chunk in chunks[:8]
    ]


def _merge_update(state: dict, update: dict) -> None:
    for key, value in update.items():
        if key == "tool_trace" and isinstance(value, list):
            state.setdefault("tool_trace", [])
            state["tool_trace"].extend(value)
        else:
            state[key] = value


def _format_graph_result(result: dict) -> dict:
    doc_chunks = result.get("retrieved_chunks", [])
    web_sources = result.get("sources", [])
    sources = []
    if doc_chunks:
        sources.extend(_doc_sources(doc_chunks))
    if web_sources:
        sources.extend(web_sources)

    raw_confidence = result.get("confidence")
    route = result.get("route", "")
    confidence = _coerce_confidence(raw_confidence, route)

    return {
        "answer": result.get("final_response", ""),
        "sources": sources,
        "confidence": confidence,
        "tool_trace": result.get("tool_trace", []),
        "used_vector_db": bool(result.get("used_vector_db", False)),
        "chunks": doc_chunks,
        "retrieved_chunks": _metadata_chunks(doc_chunks),
        "routing_reason": result.get("routing_reason", ""),
    }


async def answer_query_with_graph(
    query: str,
    document_ids: list[int],
    db: AsyncSession,
    history: list[dict] | None = None,
    include_web: bool = True,
    document_names: list[str] | None = None,
) -> dict:
    """Run one chat query through the graph agent.

    Return shape intentionally matches backend.agent.router.answer_query so
    ChatService and the frontend SSE metadata contract can stay stable.
    """
    state = create_main_state(
        query=query,
        document_ids=document_ids,
        document_names=document_names or [],
        history=history or [],
        include_web=include_web,
        db=db,
    )
    result = await main_graph.ainvoke(state)
    return _format_graph_result(result)


async def stream_answer_query_with_graph(
    query: str,
    document_ids: list[int],
    db: AsyncSession,
    history: list[dict] | None = None,
    include_web: bool = True,
    document_names: list[str] | None = None,
) -> AsyncGenerator[tuple[str, object], None]:
    """Stream graph progress events, then yield the final agent result.

    The final result is emitted as an internal "agent_result" event for
    ChatService. The API route only forwards graph_status/token/metadata/done.
    """
    state = create_main_state(
        query=query,
        document_ids=document_ids,
        document_names=document_names or [],
        history=history or [],
        include_web=include_web,
        db=db,
    )
    final_state = dict(state)

    yield node_started("main_router")

    async for update in main_graph.astream(state, stream_mode="updates"):
        for node, values in update.items():
            if isinstance(values, dict):
                _merge_update(final_state, values)

            if node == "main_router":
                route = final_state.get("route", "general")
                route_label = {
                    "general": "Direct answer",
                    "document": "Document answer",
                    "web": "Web answer",
                    "research": "Research analysis",
                }.get(route, route.replace("_", " ").title())
                next_node = {
                    "general": "direct_llm",
                    "document": "document_qa",
                    "web": "web_search",
                    "research": "research_system",
                }.get(route, "direct_llm")
                yield node_completed(
                    "main_router",
                    route=route,
                    routing_reason=final_state.get("routing_reason", ""),
                    label=route_label,
                    message=f"Using {route_label.lower()}",
                )
                yield node_started(next_node, route=route)
                if next_node == "research_system":
                    yield node_started("research_planner", route=route)
                continue

            extra = {}
            if node in {"document_qa", "call_retriever_subagent"}:
                extra["chunks_found"] = len(final_state.get("retrieved_chunks", []))
            if node == "web_search":
                extra["results_found"] = len(final_state.get("web_results", []))

            if node == "research_system":
                yield node_completed("research_planner")
                yield node_completed(
                    "call_retriever_subagent",
                    chunks_found=len(final_state.get("retrieved_chunks", [])),
                )
                if final_state.get("web_results"):
                    yield node_completed(
                        "web_search",
                        results_found=len(final_state.get("web_results", [])),
                    )
                yield node_completed("evidence_fusion")
                yield node_completed("generator")
                yield node_completed("critic")

            yield node_completed(node, **extra)
            if node in {"direct_llm", "document_qa", "web_search"}:
                yield node_started("generator")
            elif node == "generator":
                yield node_started("final_response")

    yield ("agent_result", _format_graph_result(final_state))
