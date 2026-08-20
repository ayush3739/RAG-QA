from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from backend.api.deps import get_current_user, get_db
from backend.models.models import User
from backend.services.analytics_service import AnalyticsService

router = APIRouter()
analytics_service = AnalyticsService()


@router.get("/activity")
async def get_activity(
    days: int = Query(7, ge=1, le=30),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the query activity timeline for the authenticated user."""
    activity = await analytics_service.get_user_activity(
        user_id=current_user.id,
        db=db,
        days=days,
    )
    return {"activity": activity}
