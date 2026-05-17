"""Chat API — Streaming conversation."""

from fastapi import APIRouter
from backend.models.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat/{collection}")
async def chat(collection: str, request: ChatRequest):
    """Stream chat response for a collection."""
    # TODO: Implement SSE streaming
    return ChatResponse(answer="Hello!", confidence=0.9, sources=[])
