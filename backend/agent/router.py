"""Agent router for DocuMind tool-routed RAG."""

from __future__ import annotations

import json
import math

from sqlalchemy.ext.asyncio import AsyncSession

from backend.agent.tools import (
    OPENAI_TOOL_SCHEMAS,
    direct_answer_impl,
    generate_quiz_impl,
    retrieve_from_document_impl,
    summarize_document_impl,
    web_search_impl,
)
from backend.services.llm_provider import LLMProvider


CONFIDENCE_ESCALATE_THRESHOLD = 0.4
ROUTER_TOOLS = {
    "retrieve_from_document",
    "web_search",
    "summarize_document",
    "generate_quiz",
}


def _history_messages(history: list[dict] | None) -> list[dict]:
    messages = []
    for item in (history or [])[-6:]:
        role = item.get("role")
        content = item.get("content")
        if role and content:
            messages.append({"role": role, "content": content})
    return messages


def _fallback_select_tools(
    query: str,
    has_documents: bool,
    include_web: bool,
) -> list[str]:
    normalized = query.lower()
    tools = []

    if has_documents and any(
        signal in normalized
        for signal in ("summarize", "summary", "overview", "key points", "main points")
    ):
        tools.append("summarize_document")

    if has_documents and any(
        signal in normalized
        for signal in ("quiz", "mcq", "multiple choice", "test me", "practice question")
    ):
        tools.append("generate_quiz")

    if has_documents and any(
        signal in normalized
        for signal in ("document", "pdf", "report", "page", "chapter", "section")
    ):
        tools.append("retrieve_from_document")

    if include_web and any(
        signal in normalized
        for signal in (
            "today",
            "latest",
            "current",
            "recent",
            "news",
            "this week",
            "this month",
            "2026",
            "weather",
        )
    ):
        tools.append("web_search")

    return list(dict.fromkeys(tools)) or ["none"]


async def _select_tool(query: str, has_documents: bool, include_web: bool) -> dict:
    llm = LLMProvider()
    messages = [
        {
            "role": "system",
            "content": f"""You are DocuMind's routing model. Choose the tool or tools needed for the user's query.

Available tools:
- retrieve_from_document: Use only when the user asks about uploaded/indexed document content, facts likely inside the document, or refers to "this document", "the PDF", "the report", "these notes", pages, sections, clauses, chapters, tables, or document-specific entities.
- summarize_document: Use when the user asks for a summary, overview, key points, main ideas, or chapter/section summary of the uploaded document.
- generate_quiz: Use when the user asks for a quiz, MCQs, practice questions, or to be tested on the uploaded document.
- web_search: Use for current, recent, latest, today/news queries, or when live web information is required. This tool is {"enabled" if include_web else "disabled"}.

Routing rules:
- Do NOT choose retrieve_from_document merely because documents are attached.
- If the query can be answered without the document and without live web, do not call any tool. Answer directly.
- For compound questions, choose multiple tools. Example: if the user asks about the document AND today's weather, choose ["retrieve_from_document", "web_search"].
- For compound questions with a general-knowledge part, call tools only for the document/web parts. The final answer can answer the general part directly.
- If web_search is disabled, do not choose web_search.
- If no documents are attached, do not choose document tools.
- When native tool calling is available, call the selected tool or tools. If no tool is needed, answer the user directly with zero tool calls.
- If native tool calling is not available, return ONLY valid JSON with this exact shape:
  {{"tools": ["none"], "reason": "short reason"}}

Examples:
Q: what is 2+2?
{{"tools": ["none"], "reason": "simple arithmetic; no document needed"}}

Q: summarize this pdf
{{"tools": ["summarize_document"], "reason": "asks for uploaded document summary"}}

Q: what does page 4 say about cancellation?
{{"tools": ["retrieve_from_document"], "reason": "asks about a specific document page/content"}}

Q: what happened in AI news today?
{{"tools": ["web_search"], "reason": "asks for current news"}}

Q: what is this document about and also tell me the weather of today in Noida?
{{"tools": ["retrieve_from_document", "web_search"], "reason": "asks about uploaded document content and current weather"}}

Q: what is this document about and what is Newton's third law?
{{"tools": ["retrieve_from_document"], "reason": "asks about uploaded document content; general part needs no tool"}}
""",
        },
        {
            "role": "user",
            "content": (
                f"documents_attached={has_documents}\n"
                f"web_search_enabled={include_web}\n"
                f"query={query}"
            ),
        },
    ]

    if llm.supports_native_tool_calls():
        native = await llm.tool_call(messages=messages, tools=OPENAI_TOOL_SCHEMAS)
        response = native.get("content", "")
        tool_args = {}
        tools = []

        for call in native.get("tool_calls", []):
            tool_name = call.get("name")
            if tool_name not in ROUTER_TOOLS:
                continue
            if tool_name in {"retrieve_from_document", "summarize_document", "generate_quiz"} and not has_documents:
                continue
            if tool_name == "web_search" and not include_web:
                continue

            args = call.get("args") or {}
            args.setdefault("query", query)
            tool_args[tool_name] = args
            if tool_name not in tools:
                tools.append(tool_name)

        if tools:
            return {
                "tools": tools,
                "tool_args": tool_args,
                "reason": "native tool call",
            }
        if response:
            return {
                "tools": ["none"],
                "tool_args": {"none": {"query": query}},
                "reason": "model answered with zero tool calls",
                "direct_response": response,
            }
    else:
        response = ""

    if not response:
        response = await llm.invoke(messages)

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        start = response.find("{")
        end = response.rfind("}")
        if start == -1 or end == -1 or end <= start:
            tools = _fallback_select_tools(query, has_documents, include_web)
            return {
                "tools": tools,
                "tool_args": {tool: {"query": query} for tool in tools},
                "reason": "router output was not valid JSON",
            }
        try:
            parsed = json.loads(response[start : end + 1])
        except json.JSONDecodeError:
            tools = _fallback_select_tools(query, has_documents, include_web)
            return {
                "tools": tools,
                "tool_args": {tool: {"query": query} for tool in tools},
                "reason": "router output was not valid JSON",
            }

    raw_tools = parsed.get("tools")
    if not isinstance(raw_tools, list):
        raw_tools = [parsed.get("tool")]

    tools = []
    for tool in raw_tools:
        if tool == "none":
            if not tools:
                tools.append("none")
            continue
        if tool not in ROUTER_TOOLS:
            continue
        if tool in {"retrieve_from_document", "summarize_document", "generate_quiz"} and not has_documents:
            continue
        if tool == "web_search" and not include_web:
            continue
        if tool not in tools:
            tools.append(tool)

    if not tools:
        tools = _fallback_select_tools(query, has_documents, include_web)
    if len(tools) > 1 and "none" in tools:
        tools = [tool for tool in tools if tool != "none"]

    return {
        "tools": tools,
        "tool_args": {tool: {"query": query} for tool in tools},
        "reason": parsed.get("reason", ""),
    }


def _format_doc_chunks(chunks: list[dict]) -> str:
    return "\n\n".join(
        (
            f"[chunk_id={chunk.get('chunk_id')} | "
            f"page={chunk.get('page_label')} | "
            f"source={chunk.get('source')}]\n"
            f"{chunk.get('text', '')}"
        )
        for chunk in chunks[:8]
        if chunk.get("text")
    )


def _format_web_results(results: list[dict]) -> str:
    return "\n\n".join(
        (
            f"[title={result.get('title')} | url={result.get('url')}]\n"
            f"{result.get('content', '')}"
        )
        for result in results[:5]
        if result.get("content")
    )


def _normalize_confidence(score: float | None) -> float | None:
    if score is None:
        return None
    try:
        return round(1.0 / (1.0 + math.exp(-float(score))), 4)
    except (OverflowError, TypeError, ValueError):
        return None


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


def _synthesis_system_prompt(
    doc_context: str,
    web_context: str,
) -> str:
    if doc_context and not web_context:
        return f"""You are a helpful assistant that answers questions strictly based on context
retrieved from a PDF document.

Rules:
- Answer ONLY using the provided context chunks. Do not use prior knowledge.
- Exception: if the user's question also contains a clearly separate general-knowledge part that is not asking about the document, answer that part from your own knowledge and do not cite it as document-supported.
- If the answer spans multiple chunks, synthesize them into one clear response.
- Always cite the relevant page number(s) at the end, e.g., (Page 4, 12).
- You may tell the user where to read more, e.g., "You can read more on Page 4", only when that page number appears in the context metadata.
- If chunks partially relate but don't fully answer the question, say what you found and note what's missing.
- If chunks contradict each other, mention both findings and their pages.
- If the context doesn't contain the answer, respond with:
  "I could not find this information in the provided document."
- Keep answers under 200 words by default.
- If the user asks for a summary, detailed explanation, in-depth answer, or specifies a longer length, provide the requested depth up to 1000 words.
- If the user asks for more than 1000 words, keep the answer under 1000 words and focus on the most useful details.
- Do not infer or extrapolate beyond what is explicitly stated in the chunks.

CONTEXT:
{doc_context}"""

    return (
        "You are DocuMind. Use the provided document and web context when the "
        "question asks about those sources. Cite document pages as (Page X) and "
        "web sources by URL/title when used. For clearly separate general-knowledge "
        "parts of a compound question, you may answer directly from your own "
        "knowledge, but do not cite that as document-supported. If the provided "
        "context does not support a document-specific answer, say that clearly.\n\n"
        f"DOCUMENT CONTEXT:\n{doc_context or 'None'}\n\n"
        f"WEB CONTEXT:\n{web_context or 'None'}"
    )


async def _synthesize(
    query: str,
    doc_chunks: list[dict],
    web_results: list[dict],
    history: list[dict] | None,
) -> str:
    llm = LLMProvider()
    doc_context = _format_doc_chunks(doc_chunks)
    web_context = _format_web_results(web_results)
    system_prompt = _synthesis_system_prompt(
        doc_context,
        web_context,
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        *_history_messages(history),
        {"role": "user", "content": query},
    ]

    return await llm.invoke(messages)


async def answer_query(
    query: str,
    document_ids: list[int],
    db: AsyncSession,
    history: list[dict] | None = None,
    include_web: bool = True,
) -> dict:
    """Route a chat query through document, web, summary, quiz, or direct answer."""
    tool_trace: list[str] = []
    sources: list[dict] = []
    doc_chunks: list[dict] = []
    web_results: list[dict] = []
    confidence: float | None = None

    route = await _select_tool(
        query=query,
        has_documents=bool(document_ids),
        include_web=include_web,
    )
    selected_tools = route["tools"]
    tool_args = route.get("tool_args", {})
    routing_reason = route.get("reason", "")
    tool_trace.extend(selected_tools)

    if selected_tools == ["none"]:
        if route.get("direct_response"):
            result = {
                "answer": route["direct_response"],
                "sources": [],
                "confidence": None,
            }
        else:
            result = await direct_answer_impl(
                tool_args.get("none", {}).get("query", query)
            )
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
            "tool_trace": tool_trace,
            "used_vector_db": False,
            "chunks": [],
            "retrieved_chunks": [],
            "routing_reason": routing_reason,
        }

    if selected_tools == ["summarize_document"]:
        result = await summarize_document_impl(
            query=tool_args.get("summarize_document", {}).get("query", query),
            document_ids=document_ids,
            db=db,
        )
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
            "tool_trace": tool_trace,
            "used_vector_db": False,
            "chunks": [],
            "retrieved_chunks": [],
            "routing_reason": routing_reason,
        }

    if selected_tools == ["generate_quiz"]:
        quiz_args = tool_args.get("generate_quiz", {})
        result = await generate_quiz_impl(
            query=quiz_args.get("query", query),
            document_ids=document_ids,
            db=db,
            num_questions=quiz_args.get("num_questions", 5),
        )
        return {
            "answer": result.get("raw_response", ""),
            "quiz": result.get("questions", []),
            "sources": result["sources"],
            "confidence": result["confidence"],
            "tool_trace": tool_trace,
            "used_vector_db": False,
            "chunks": [],
            "retrieved_chunks": [],
            "routing_reason": routing_reason,
        }

    used_vector_db = False

    if "summarize_document" in selected_tools and "retrieve_from_document" not in selected_tools:
        selected_tools.append("retrieve_from_document")
        tool_trace.append("retrieve_from_document (document context for compound query)")

    if "retrieve_from_document" in selected_tools:
        result = await retrieve_from_document_impl(
            query=tool_args.get("retrieve_from_document", {}).get("query", query),
            document_ids=document_ids,
            db=db,
        )
        doc_chunks = result.get("chunks", [])
        confidence = _normalize_confidence(result.get("confidence"))
        used_vector_db = bool(result.get("used_vector_db"))
        sources.extend(_doc_sources(doc_chunks))

        if (
            include_web
            and "web_search" not in selected_tools
            and (confidence is None or confidence < CONFIDENCE_ESCALATE_THRESHOLD)
        ):
            web_result = await web_search_impl(
                tool_args.get("web_search", {}).get("query", query)
            )
            web_results = web_result.get("results", [])
            sources.extend(web_result.get("sources", []))
            tool_trace.append("web_search (escalated: low doc confidence)")

    if "web_search" in selected_tools:
        web_result = await web_search_impl(
            tool_args.get("web_search", {}).get("query", query)
        )
        web_results = web_result.get("results", [])
        sources.extend(web_result.get("sources", []))
        if confidence is None:
            confidence = web_result.get("confidence")

    answer = await _synthesize(
        query=query,
        doc_chunks=doc_chunks,
        web_results=web_results,
        history=history,
    )

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "tool_trace": tool_trace,
        "used_vector_db": used_vector_db,
        "chunks": doc_chunks,
        "retrieved_chunks": _metadata_chunks(doc_chunks),
        "routing_reason": routing_reason,
    }
