"""Chat API — Streaming conversation."""

from fastapi import APIRouter
from fastapi.sse import EventSourceResponse
from backend.models.schemas import ChatRequest, ChatResponse, Item
from collections.abc import AsyncIterable, Iterable
from sse_starlette.sse import EventSourceResponse
import asyncio

router = APIRouter()

items = [
    Item(name="Plumbus", description="A multi-purpose household device."),
    Item(name="Portal Gun", description="A portal opening device."),
    Item(name="Meeseeks Box", description="A box that summons a Meeseeks."),
]



@router.post("/chat/{collection}")
async def chat(collection: str, req: ChatRequest):
    async def event_generator():
        retriever = Retriever(collection)
        async for chunk in retriever.answer_stream(req.query, req.history):
            yield {"data": chunk}
        yield {"data": "[DONE]"}
    return EventSourceResponse(event_generator())

@router.get("/stream", response_class=EventSourceResponse)
# FIX: Change the return type to AsyncIterable[dict] (or more specifically, the yielded structure)
async def see_items() -> AsyncIterable[dict]: 
    """
    Streams item data using Server-Sent Events.
    """
    for item in items:
        # Yield the dictionary structure
        await asyncio.sleep(1)
        yield {
            "event": "message",
            "data": f"{item.name}:{item.description}",
        }        


@router.post("chat/{session_id}/stream")
async def chat(session_id: int, request: ChatRequest):
    """Stream chat response for a collection."""
    # TODO: Implement SSE streaming
    return ChatResponse(answer="Hello!", confidence=0.9, sources=[])


