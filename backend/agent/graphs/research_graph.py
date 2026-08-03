"""
research_graph.py

Matches diagram 2 exactly:

    Research Query
          │
    Research Planner
          │
    Execution Plan ──────────────► Need Web?
          │                           │ Yes
          │                           ▼
    Retriever Subagent ◄──── Web Search
          │
    Evidence Fusion
          │
      Generator
          │
        Critic
          │
    Issues Found?
     │ No      │ Yes
     ▼         ▼
Final Response  Query Refiner
                    │
               (loops back to subagent)
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from langgraph.graph import END, StateGraph

from backend.agent.graphs.retriever_subagent import retrieval_agent
from backend.agent.graphs.state import ResearchState, create_retriever_state
from backend.agent.tools import web_search_impl
from backend.services.llm_provider import LLMProvider


llm = LLMProvider()


with (Path(__file__).with_name("prompts.toml")).open("rb") as f:
    prompts = tomllib.load(f)


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


def _parse_json_array(text: str) -> list[dict]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []
    return parsed if isinstance(parsed, list) else []


async def research_planner(state: ResearchState) -> dict:
    response = await llm.invoke(
        [
            {"role": "system", "content": prompts["PLANNER_SYSTEM"]},
            {"role": "user", "content": state["research_query"]},
        ]
    )
    plan = _parse_json_object(response)
    return {
        "execution_plan": plan,
        "need_web": bool(plan.get("need_web", False)),
    }


async def call_retriever_subagent(state: ResearchState) -> dict:
    query = state.get("refined_query") or state["research_query"]
    result = await retrieval_agent.ainvoke(
        create_retriever_state(
            query=query,
            document_ids=state.get("document_ids", []),
            db=state.get("db"),
        )
    )
    chunks = result.get("reranked_chunks", [])
    return {"retrieved_chunks": chunks}


def route_after_retrieval(state: ResearchState) -> str:
    return "web_search" if state.get("need_web") else "evidence_fusion"


async def web_search(state: ResearchState) -> dict:
    query = state.get("refined_query") or state["research_query"]
    result = await web_search_impl(query=query)
    return {
        "web_results": result.get("results", []),
    }


async def evidence_fusion(state: ResearchState) -> dict:
    if not state.get("web_results"):
        fused = [
            {
                "id": f"ev_{i + 1}",
                "content": chunk.get("text") or chunk.get("content", ""),
                "source": "retriever",
                "url": chunk.get("source", ""),
            }
            for i, chunk in enumerate(state.get("retrieved_chunks", []))
        ]
        return {"fused_evidence": fused}

    payload = json.dumps(
        {
            "retriever": state.get("retrieved_chunks", []),
            "web": state.get("web_results", []),
        }
    )
    response = await llm.invoke(
        [
            {"role": "system", "content": prompts["FUSION_SYSTEM"]},
            {"role": "user", "content": payload},
        ]
    )
    fused = _parse_json_array(response)
    return {"fused_evidence": fused}


async def generator(state: ResearchState) -> dict:
    evidence_text = "\n\n".join(
        f"Source: {e.get('source')}\nContent: {e.get('content')}"
        for e in state.get("fused_evidence", [])
    )
    goal = state.get("execution_plan", {}).get("goal", state["research_query"])
    payload = f"Goal: {goal}\n\nEvidence:\n{evidence_text or 'No evidence found.'}"
    response = await llm.invoke(
        [
            {"role": "system", "content": prompts["GENERATOR_SYSTEM"]},
            {"role": "user", "content": payload},
        ]
    )
    return {"draft": response}


async def critic(state: ResearchState) -> dict:
    evidence_text = "\n\n".join(
        f"Source: {e.get('source')}\nContent: {e.get('content')}"
        for e in state.get("fused_evidence", [])
    )
    payload = json.dumps(
        {
            "goal": state.get("execution_plan", {}).get("goal", state["research_query"]),
            "draft": state.get("draft", ""),
            "evidence": evidence_text,
        }
    )
    response = await llm.invoke(
        [
            {"role": "system", "content": prompts["CRITIC_SYSTEM"]},
            {"role": "user", "content": payload},
        ]
    )
    result = _parse_json_object(response)
    return {"issues_found": bool(result.get("issues_found", False))}


def route_issues(state: ResearchState) -> str:
    if not state.get("issues_found") or state.get("retry_count", 0) >= state.get("max_retries", 2):
        return "final_response"
    return "query_refiner"


async def query_refiner(state: ResearchState) -> dict:
    response = await llm.invoke(
        [
            {"role": "system", "content": prompts["REFINER_SYSTEM"]},
            {
                "role": "user",
                "content": (
                    f"Original query: {state['research_query']}\n\n"
                    f"Draft so far:\n{state.get('draft', '')}"
                ),
            },
        ]
    )
    return {
        "refined_query": response.strip(),
        "retry_count": state.get("retry_count", 0) + 1,
        "retrieved_chunks": [],
        "web_results": [],
        "fused_evidence": [],
    }


def final_response(state: ResearchState) -> dict:
    return {"final_response": state.get("draft", "")}


builder = StateGraph(ResearchState)

builder.add_node("research_planner", research_planner)
builder.add_node("call_retriever_subagent", call_retriever_subagent)
builder.add_node("web_search", web_search)
builder.add_node("evidence_fusion", evidence_fusion)
builder.add_node("generator", generator)
builder.add_node("critic", critic)
builder.add_node("query_refiner", query_refiner)
builder.add_node("final_response", final_response)

builder.set_entry_point("research_planner")
builder.add_edge("research_planner", "call_retriever_subagent")
builder.add_conditional_edges(
    "call_retriever_subagent",
    route_after_retrieval,
    {
        "web_search": "web_search",
        "evidence_fusion": "evidence_fusion",
    },
)
builder.add_edge("web_search", "evidence_fusion")
builder.add_edge("evidence_fusion", "generator")
builder.add_edge("generator", "critic")
builder.add_conditional_edges(
    "critic",
    route_issues,
    {
        "final_response": "final_response",
        "query_refiner": "query_refiner",
    },
)
builder.add_edge("query_refiner", "call_retriever_subagent")
builder.add_edge("final_response", END)

research_graph = builder.compile()
