from uuid import UUID

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.base import get_db
from backend.services.chat_service import ChatService
from backend.services.session_service import SessionService
from backend.models.schemas import ChatRequest
from backend.api.deps import get_current_user

router = APIRouter(prefix="/chat",)

chat_service = ChatService()
session_service = SessionService()

@router.post("/{session_id}")
async def chat(
    session_id: UUID,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):

    async def event_generator():

        async for event_type, data in chat_service.stream_chat(
            session_id=session_id,
            question=request.question,
            user_id=current_user.id,
            db=db,
        ):

            yield {
                "event": event_type,
                "data": data,
            }

    return EventSourceResponse(
        event_generator()
    )

@router.get("/history/{session_id}")
async def get_chat_history(session_id: UUID,db: AsyncSession = Depends(get_db),current_user=Depends(get_current_user)):
    session = await session_service.get_session(
        session_id=session_id,
        db=db,
    )

    if not session:
        return {
            "status": "error",
            "message": "Session not found",
        }

    if session.user_id != current_user.id:
        return {
            "status": "error",
            "message": "Unauthorized",
        }

    return await session_service.get_messages(
        session_id=session_id,
        db=db,
    )