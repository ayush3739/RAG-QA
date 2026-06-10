"""Feedback API — Rate answers."""
from typing import Annotated
from fastapi import APIRouter,Depends
from backend.models.schemas import FeedbackRequest
from backend.models import models
from backend.db.base import get_db
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest, db : Annotated[AsyncSession,Depends(get_db)]):
    """Submit feedback on an answer."""
    # TODO: Store feedback to SQLite
    
    feedback = models.Feedback(
        message_id = request.message_id,
        user_id = 1,
        rating = request.rating,
        comment = request.comment,
    )
    db.add(feedback)
    await db.commit()
    await db.refresh
    return {"status": "received", "feedback_id": "1"}
