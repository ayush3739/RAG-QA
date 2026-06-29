"""Feedback API — Rate answers."""
from typing import Annotated
from fastapi import APIRouter,Depends
from backend.models.schemas import FeedbackRequest
from backend.models import models
from backend.db.base import get_db
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user

router = APIRouter(prefix="/feedback")


@router.post("")
async def submit_feedback(
    request: FeedbackRequest, 
    db : Annotated[AsyncSession,Depends(get_db)],
    current_user = Depends(get_current_user)
):
    """Submit feedback on an answer."""
    feedback = models.Feedback(
        message_id = request.message_id,
        user_id = current_user.id,
        rating = request.rating,
        comment = request.comment,
    )
    db.add(feedback)
    await db.commit()
    await db.refresh(feedback)
    return {"status": "received", "feedback_id": feedback.id}
