"""Feedback API — Rate answers."""

from fastapi import APIRouter
from backend.models.schemas import FeedbackRequest

router = APIRouter()


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback on an answer."""
    # TODO: Store feedback to SQLite
    return {"status": "received", "feedback_id": "1"}
