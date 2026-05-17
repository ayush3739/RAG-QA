"""Pydantic schemas for request/response models."""

from pydantic import BaseModel
from typing import List, Optional


# Chat Models
class ChatRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = []
    k: int = 10


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[dict]
    follow_up_questions: Optional[List[str]] = []


# Research Models
class ResearchRequest(BaseModel):
    topic: str
    collection: Optional[str] = None
    include_web: bool = True
    output_format: str = "structured"  # structured | bullet | prose


class ResearchResponse(BaseModel):
    summary: str
    key_findings: List[str]
    sources: List[dict]
    confidence: float
    tool_trace: List[str]
    follow_up_questions: List[str]


# Collections
class CollectionResponse(BaseModel):
    name: str
    document_count: int
    indexed_at: str


# Feedback
class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: int  # 1-5
    confidence: float
    tool_used: Optional[str] = None
