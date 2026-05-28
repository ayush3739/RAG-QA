"""Pydantic schemas for request/response models."""

from pydantic import BaseModel
from typing import List, Optional

class Item(BaseModel):
    name: str
    description : str | None


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
class CollectionUploadResponse(BaseModel):
    status: str
    job_id: str
    collection_id: str
    filename: str

class Collectiondeleterequet(BaseModel):
    name: str

class CollectionListResponse(BaseModel):
    collections: List[str]


class CollectionDeleteResponse(BaseModel):
    status: str
    collection: str


class IndexJobStatusResponse(BaseModel):
    job_id: str
    status: str
    collection_id: str
    filename: str
    path: str
    error: Optional[str] = None


# Feedback
class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: int  # 1-5
    confidence: float
    tool_used: Optional[str] = None
