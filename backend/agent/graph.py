"""LangGraph Agent State Machine — Phase 3."""

from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    """Agent state for LangGraph."""
    
    query: str
    collection: str
    history: List[dict]
    
    # Classification
    intent: str  # factual | summarize | web | direct | quiz
    
    # Tool tracking
    tool_used: str  # which tool was called
    
    # Data
    retrieved_chunks: list  # from RAG
    web_results: list  # from Tavily
    
    # Output
    answer: str
    confidence: float
    sources: list
    follow_up_questions: list
    
    # Transparency
    reasoning_trace: list  # for debugging
