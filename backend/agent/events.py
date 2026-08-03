from __future__ import annotations

import json
from typing import Any


GRAPH_STATUS_EVENT = "graph_status"


NODE_LABELS = {
    "main_router": "Deciding route",
    "direct_llm": "Preparing answer",
    "document_qa": "Searching documents",
    "web_search": "Searching web",
    "research_system": "Deploying research agent",
    "research_planner": "Planning research",
    "call_retriever_subagent": "Document retrieval",
    "evidence_fusion": "Evidence fusion",
    "critic": "Quality check",
    "query_refiner": "Refining query",
    "generator": "Writing answer",
    "final_response": "Finalizing",
}


ROUTE_LABELS = {
    "general": "Direct answer",
    "document": "Document answer",
    "web": "Web answer",
    "research": "Research analysis",
}


def graph_status(
    node: str,
    status: str,
    message: str | None = None,
    **extra: Any,
) -> tuple[str, str]:
    payload = {
        "node": node,
        "label": NODE_LABELS.get(node, node.replace("_", " ").title()),
        "status": status,
        "message": message or NODE_LABELS.get(node, node.replace("_", " ").title()),
        **extra,
    }
    return GRAPH_STATUS_EVENT, json.dumps(payload)


def node_started(node: str, **extra: Any) -> tuple[str, str]:
    return graph_status(node, "running", **extra)


def node_completed(node: str, **extra: Any) -> tuple[str, str]:
    return graph_status(node, "completed", **extra)


def route_selected(route: str, reason: str = "") -> tuple[str, str]:
    label = ROUTE_LABELS.get(route, route.replace("_", " ").title())
    return graph_status(
        "main_router",
        "completed",
        message=f"Using {label.lower()}",
        route=route,
        mode=f"{route}_answer" if route != "research" else "research_analysis",
        label=label,
        routing_reason=reason,
    )
