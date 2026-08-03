from operator import add
from typing import Annotated, Any
from typing_extensions import TypedDict
from backend.core.retriever import RetrievalQuery


class MainState(TypedDict, total=False):
    user_query: str
    document_ids: list[int]
    document_names: list[str]
    history: list[dict[str, str]]
    include_web: bool
    db: Any

    route: str
    routing_reason: str

    retrieved_chunks: list[dict[str, Any]]
    web_results: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    research_output: str

    generator_output: str
    final_response: str

    confidence: float | None
    tool_trace: Annotated[list[str], add]
    used_vector_db: bool
    retriever_debug: dict[str, Any]


class ResearchState(TypedDict, total=False):
    research_query: str
    document_ids: list[int]
    db: Any

    execution_plan: dict[str, Any]
    need_web: bool

    retrieved_chunks: list[dict[str, Any]]
    web_results: list[dict[str, Any]]
    fused_evidence: list[dict[str, Any]]

    draft: str
    issues_found: bool
    refined_query: str
    retry_count: int
    max_retries: int

    final_response: str


class RetrieverState(TypedDict, total=False):
    query: str | RetrievalQuery
    document_ids: list[int]
    db: Any

    needs_decomposition: bool
    queries: list[RetrievalQuery]

    reranked_chunks: list[dict[str, Any]]
    confidence: float
    used_vector_db: bool
    retriever_debug: dict[str, Any]
    
    max_retries: int


def create_main_state(
    query: str,
    document_ids: list[int],
    document_names: list[str] | None = None,
    history: list[dict[str, str]] | None = None,
    include_web: bool = True,
    db: Any | None = None,
) -> MainState:
    return {
        "user_query": query,
        "document_ids": document_ids,
        "document_names": document_names or [],
        "history": history or [],
        "include_web": include_web,
        "db": db,
        "route": "",
        "routing_reason": "",
        "retrieved_chunks": [],
        "web_results": [],
        "sources": [],
        "research_output": "",
        "generator_output": "",
        "final_response": "",
        "confidence": None,
        "tool_trace": [],
        "used_vector_db": False,
    }


def create_research_state(
    query: str,
    document_ids: list[int] | None = None,
    db: Any | None = None,
) -> ResearchState:
    return {
        "research_query": query,
        "document_ids": document_ids or [],
        "db": db,
        "execution_plan": {},
        "need_web": False,
        "retrieved_chunks": [],
        "web_results": [],
        "fused_evidence": [],
        "draft": "",
        "issues_found": False,
        "refined_query": "",
        "retry_count": 0,
        "max_retries": 2,
        "final_response": "",
    }


def create_retriever_state(
    query: str,
    document_ids: list[int] | None = None,
    db: Any | None = None,
) -> RetrieverState:
    return {
        "query": query,
        "document_ids": document_ids or [],
        "db": db,
        "needs_decomposition": False,
        "subqueries": [],
        "parallel_chunks": [],
        "merged_chunks": [],
        "deduplicated_chunks": [],
        "hybrid_chunks": [],
        "reranked_chunks": [],
        "confidence": 0.0,
        "enough_evidence": False,
        "used_vector_db": False,
        "retry_count": 0,
        "max_retries": 2,
    }
