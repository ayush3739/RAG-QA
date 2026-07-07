from __future__ import annotations

import json

from langgraph.graph import END, StateGraph

from backend.agent.graphs.research_graph import research_graph
from backend.agent.graphs.retriever_subagent import retrieval_agent
from backend.agent.graphs.state import MainState, create_research_state, create_retriever_state
from backend.agent.router import _synthesize
from backend.agent.tools import direct_answer_impl, summarize_document_impl, web_search_impl
from backend.services.llm_provider import LLMProvider


llm = LLMProvider()


ROUTER_SYSTEM = """Classify this user query into exactly one route.

Routes:
- "general": general knowledge, greetings, or no external context needed
- "document": asks about uploaded or linked documents
- "web": needs current/live/recent internet information
- "research": complex comparison, multi-step analysis, or document+web synthesis

If the user asks a follow-up such as "tell me details from the pages",
"from those sources", "from the above links", or "summarize these results",
use the previous conversation context to classify it. If the previous assistant
answer used web sources, classify the follow-up as "web".

Return JSON only:
{"route": "<general|document|web|research>", "reason": "<short reason>"}"""


_SOURCE_FOLLOWUP_TERMS = (
    "from the pages",
    "from those pages",
    "from these pages",
    "from the sources",
    "from those sources",
    "from these sources",
    "from the links",
    "from those links",
    "from above",
    "above sources",
    "details from",
)

_DOCUMENT_TERMS = (
    "this document",
    "the document",
    "attached document",
    "uploaded document",
    "linked document",
    "this file",
    "the file",
    "pdf",
    "docx",
    "document says",
    "document say",
    "what is this document",
    "what does this document",
    "summarize this",
    "summary",
    "summarise",
    "summarize",
)

_WEB_TERMS = (
    "latest",
    "current",
    "today",
    "recent",
    "news",
    "live",
    "now",
    "this week",
    "this month",
    "web",
    "internet",
    "online",
)

_SUMMARY_TERMS = (
    "summary",
    "summarize",
    "summarise",
    "overview",
    "what is this document about",
    "what is the document about",
    "what is this document says",
    "what does this document say",
)


_COMPLEX_ANALYSIS_TERMS = (
    "compare",
    "contrast",
    "similarities",
    "differences",
    "main themes",
    "themes",
    "identify",
    "analyze",
    "analyse",
    "analysis",
    "evaluate",
    "synthesize",
    "synthesise",
    "tradeoffs",
    "trade-offs",
    "pros and cons",
    "relationship between",
    "common patterns",
    "across the documents",
    "between the documents",
    "both documents",
    "two documents",
)


def _history_text(history: list[dict] | None) -> str:
    return "\n".join(
        f"{item.get('role', '')}: {item.get('content', '')}"
        for item in (history or [])[-4:]
        if item.get("content")
    )


def _previous_web_results(history: list[dict] | None) -> list[dict]:
    results: list[dict] = []
    for item in reversed(history or []):
        citations = item.get("citations") or {}
        sources = citations.get("sources") if isinstance(citations, dict) else citations
        if not isinstance(sources, list):
            continue
        for source in sources:
            if source.get("type") != "web":
                continue
            results.append(
                {
                    "title": source.get("title") or source.get("name"),
                    "url": source.get("url"),
                    "content": source.get("content") or source.get("excerpt") or source.get("snippet") or "",
                }
            )
        if results:
            return results
    return results


def _is_source_followup(query: str) -> bool:
    normalized = query.lower()
    return any(term in normalized for term in _SOURCE_FOLLOWUP_TERMS)


def _is_document_query(query: str) -> bool:
    normalized = query.lower()
    return any(term in normalized for term in _DOCUMENT_TERMS)


def _needs_live_web(query: str) -> bool:
    normalized = query.lower()
    return any(term in normalized for term in _WEB_TERMS)


def _is_summary_query(query: str) -> bool:
    normalized = query.lower()
    return any(term in normalized for term in _SUMMARY_TERMS)


def _is_complex_analysis_query(query: str) -> bool:
    normalized = query.lower()
    return any(term in normalized for term in _COMPLEX_ANALYSIS_TERMS)


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


async def main_router(state: MainState) -> dict:
    has_documents = bool(state.get("document_ids"))
    query = state["user_query"]
    previous_web_results = _previous_web_results(state.get("history", []))
    if _is_source_followup(query) and previous_web_results:
        return {
            "route": "web",
            "routing_reason": "follow-up asks about previous web sources",
            "tool_trace": ["main_router:web_followup"],
        }

    if has_documents and _is_document_query(query):
        route = (
            "research"
            if _is_complex_analysis_query(query)
            or (state.get("include_web", True) and _needs_live_web(query))
            else "document"
        )
        reason = (
            "complex document analysis requested"
            if route == "research"
            else "explicitly asks about the linked document"
        )
        return {
            "route": route,
            "routing_reason": reason,
            "tool_trace": [f"main_router:{route}:document_override"],
        }

    response = await llm.invoke(
        [
            {"role": "system", "content": ROUTER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"documents_attached={has_documents}\n"
                    f"web_enabled={state.get('include_web', True)}\n"
                    f"recent_history={_history_text(state.get('history', []))}\n"
                    f"query={state['user_query']}"
                ),
            },
        ]
    )
    result = _parse_json_object(response)
    route = result.get("route", "general")
    if route not in {"general", "document", "web", "research"}:
        route = "general"
    if route in {"document", "research"} and not state.get("document_ids"):
        route = "general"
    if route == "web" and not state.get("include_web", True):
        route = "general"
    return {
        "route": route,
        "routing_reason": result.get("reason", ""),
        "tool_trace": [f"main_router:{route}"],
    }


def route_query(state: MainState) -> str:
    return {
        "general": "direct_llm",
        "document": "document_qa",
        "web": "web_search",
        "research": "research_system",
    }.get(state.get("route", "general"), "direct_llm")


async def direct_llm(state: MainState) -> dict:
    result = await direct_answer_impl(
        query=state["user_query"],
        history=state.get("history", []),
        document_names=state.get("document_names", []),
    )
    return {
        "generator_output": result.get("answer", ""),
        "sources": result.get("sources", []),
        "confidence": result.get("confidence"),
        "tool_trace": ["direct_answer"],
        "used_vector_db": False,
    }


async def document_qa(state: MainState) -> dict:
    if _is_summary_query(state["user_query"]):
        result = await summarize_document_impl(
            query=state["user_query"],
            document_ids=state.get("document_ids", []),
            db=state.get("db"),
        )
        # Normalise chunks into the same shape as retrieval results so Ragas
        # and the frontend both get context data on summary queries.
        raw_chunks = result.get("chunks", [])
        retrieved = [
            {
                "chunk_id": c.get("chunk_id"),
                "page_label": c.get("page_label"),
                "source": c.get("source"),
                "text": c.get("text", ""),
            }
            for c in raw_chunks
        ]
        return {
            "generator_output": result.get("answer", ""),
            "sources": result.get("sources", []),
            "retrieved_chunks": retrieved,
            "confidence": result.get("confidence"),
            "used_vector_db": False,
            "tool_trace": ["summarize_document"],
        }


    result = await retrieval_agent.ainvoke(
        create_retriever_state(
            query=state["user_query"],
            document_ids=state.get("document_ids", []),
            db=state.get("db"),
        )
    )
    chunks = result.get("reranked_chunks", [])
    return {
        "retrieved_chunks": chunks,
        "confidence": result.get("confidence"),
        "used_vector_db": bool(result.get("used_vector_db", False)),
        "tool_trace": ["document_retrieval"],
    }


async def web_search(state: MainState) -> dict:
    previous_web_results = _previous_web_results(state.get("history", []))
    if _is_source_followup(state["user_query"]) and previous_web_results:
        return {
            "web_results": previous_web_results,
            "sources": [
                {
                    "type": "web",
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "content": item.get("content"),
                    "excerpt": (item.get("content") or "")[:500],
                }
                for item in previous_web_results
            ],
            "confidence": 0.7,
            "tool_trace": ["web_sources_from_history"],
        }

    result = await web_search_impl(query=state["user_query"])
    return {
        "web_results": result.get("results", []),
        "sources": result.get("sources", []),
        "confidence": result.get("confidence"),
        "tool_trace": ["web_search"],
    }


async def research_system(state: MainState) -> dict:
    result = await research_graph.ainvoke(
        create_research_state(
            query=state["user_query"],
            document_ids=state.get("document_ids", []),
            db=state.get("db"),
        )
    )
    return {
        "research_output": result.get("final_response", ""),
        "retrieved_chunks": result.get("retrieved_chunks", []),
        "web_results": result.get("web_results", []),
        "tool_trace": ["research_analysis"],
    }


async def generator(state: MainState) -> dict:
    if state.get("research_output"):
        return {"generator_output": state["research_output"]}

    if state.get("generator_output"):
        return {"generator_output": state["generator_output"]}

    response = await _synthesize(
        query=state["user_query"],
        doc_chunks=state.get("retrieved_chunks", []),
        web_results=state.get("web_results", []),
        history=state.get("history", []),
        document_names=state.get("document_names", []),
    )
    return {"generator_output": response}


def final_response(state: MainState) -> dict:
    return {"final_response": state.get("generator_output", "")}


builder = StateGraph(MainState)

builder.add_node("main_router", main_router)
builder.add_node("direct_llm", direct_llm)
builder.add_node("document_qa", document_qa)
builder.add_node("web_search", web_search)
builder.add_node("research_system", research_system)
builder.add_node("generator", generator)
builder.add_node("final_response", final_response)

builder.set_entry_point("main_router")
builder.add_conditional_edges(
    "main_router",
    route_query,
    {
        "direct_llm": "direct_llm",
        "document_qa": "document_qa",
        "web_search": "web_search",
        "research_system": "research_system",
    },
)
builder.add_edge("direct_llm", "generator")
builder.add_edge("document_qa", "generator")
builder.add_edge("web_search", "generator")
builder.add_edge("research_system", "generator")
builder.add_edge("generator", "final_response")
builder.add_edge("final_response", END)

main_graph = builder.compile()
