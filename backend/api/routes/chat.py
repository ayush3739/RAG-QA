"""Chat API — Streaming conversation."""

from fastapi import APIRouter
from fastapi.sse import EventSourceResponse
from backend.models.schemas import ChatRequest, ChatResponse, Item
from collections.abc import AsyncIterable, Iterable
import time,asyncio,json

router = APIRouter()

items = [
    Item(name="Plumbus", description="A multi-purpose household device."),
    Item(name="Portal Gun", description="A portal opening device."),
    Item(name="Meeseeks Box", description="A box that summons a Meeseeks."),
]

@router.post("/chat/{collection}")
async def chat(collection: str, request: ChatRequest):
    """Stream chat response for a collection."""
    # TODO: Implement SSE streaming
    return ChatResponse(answer="Hello!", confidence=0.9, sources=[])


# @router.get("/Items/stream", response_class=EventSourceResponse)
# async def see_items() -> AsyncIterable[Item]:
#     for item in items:
#         yield {
#             "event": "message",
#             "data": f"{item.name}:{item.description}",
#         }        
