"""Pydantic schemas for request/response models."""

from pydantic import BaseModel
from typing import List, Literal, Optional
from uuid import UUID
from datetime import datetime
class Item(BaseModel):
    name: str
    description : str | None


# Chat Models
class ChatRequest(BaseModel):
    question: str


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
    output_format: Literal["structured", "bullet", "prose"] = "structured"


class ResearchResponse(BaseModel):
    summary: str
    key_findings: List[str]
    sources: List[dict]
    confidence: Optional[float] = None
    tool_trace: List[str]
    follow_up_questions: List[str]


# Collections
class DocumentUploadResponse(BaseModel):
    status: str
    job_id: str
    document_id: str
    filename: str

class Documentdeleterequest(BaseModel):
    public_id: int

class DocumentItem(BaseModel):
    public_id: str
    name: str
    chunk_count: Optional[int] = None 
    status: str
    mime_type: str
    file_size_kb: int

class DocumentListResponse(BaseModel):
    documents: List[DocumentItem]


class DocumentDeleteResponse(BaseModel):
    status: str
    document: str


class IndexJobStatusResponse(BaseModel):
    job_id: str
    status: str
    document_id: int
    filename: str
    path: str
    error: Optional[str] = None

#Sessions [session_service.py]
class SessionCreate(BaseModel):
    session_id : UUID

class SessionItem(BaseModel):
    session_id:UUID
    title: str

class SessionList(BaseModel):
    sessions: list[SessionItem]

class SessionDocumentList(BaseModel):
    document_ids : list[int]

class MessageItem(BaseModel):
    session_id : UUID
    role : str
    content : str
    citations : Optional[dict]
    chunks : Optional[dict]
    confidence : float
    tool_used : Optional[str]
    used_vector_db : Optional[bool]
    debug : Optional[dict] = None
    created_at : datetime
class MessageList(BaseModel) :
    messages : list[MessageItem]
class MessageCreateResponse(BaseModel):
    status: str
    message_id: int
class SessionDeleteResponse(BaseModel):
    status: str
    session_id : Optional[UUID] = None
    session_name : Optional[str] = None
    error : Optional[str] = None

# Feedback
class FeedbackRequest(BaseModel):
    message_id: int
    query: str
    answer: str
    rating: int  # 1-5
    comment : str
    confidence: float
    tool_used: Optional[str] = None
