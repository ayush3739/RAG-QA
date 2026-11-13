"""
backend/api/routes/users.py

User profile endpoints (all require authentication).

GET    /users/me        → return current user
PATCH  /users/me        → update name / email
DELETE /users/me        → delete account
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
import os
from sqlalchemy import select

from backend.api.deps import get_current_user, get_db
from backend.models.auth_schemas import UserResponse
from datetime import datetime, UTC
from typing import Optional
from backend.models.models import User, Document, UpgradeRequest, UserRole
from backend.services.quota_service import get_user_quota_info
router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas (small enough to live inline here)
# ---------------------------------------------------------------------------

class UserUpdate(BaseModel):
    name: str | None = None
    email: EmailStr | None = None


class CreateUpgradeRequest(BaseModel):
    reason: Optional[str] = None


class UserUpgradeStatusResponse(BaseModel):
    has_pending: bool
    status: Optional[str] = None
    created_at: Optional[datetime] = None
    message: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user


@router.patch("/me", response_model=UserResponse)
async def update_me(
    body: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if body.name is not None:
        current_user.name = body.name

    if body.email is not None:
        current_user.email = body.email

    await db.commit()
    await db.refresh(current_user)
    return current_user




@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_me(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    
    # Query all documents for the user to delete physical files
    result = await db.execute(
        select(Document).where(Document.user_id == current_user.id)
    )
    docs = result.scalars().all()
    
    for doc in docs:
        if doc.file_path and os.path.exists(doc.file_path):
            try:
                os.remove(doc.file_path)
            except Exception:
                pass
        if doc.bm25_path and os.path.exists(doc.bm25_path):
            try:
                os.remove(doc.bm25_path)
            except Exception:
                pass

    await db.delete(current_user)
    await db.commit()


@router.get("/quota")
async def get_my_quota(
    current_user : User = Depends(get_current_user),
    db : AsyncSession = Depends(get_db)
):

    return await get_user_quota_info(user = current_user,db = db)


@router.post("/upgrade-request", response_model=UserUpgradeStatusResponse)
async def request_pro_upgrade(
    payload: CreateUpgradeRequest | None = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Allow a free user to submit an internal request for Pro account access.
    """
    user_role_str = (current_user.role.value if hasattr(current_user.role, "value") else str(current_user.role)).lower()
    if user_role_str in ("pro", "admin"):
        return UserUpgradeStatusResponse(
            has_pending=False,
            status="approved",
            message=f"You already have {current_user.role} tier access.",
        )

    # Check for existing pending request
    existing_res = await db.execute(
        select(UpgradeRequest).where(
            UpgradeRequest.user_id == current_user.id,
            UpgradeRequest.status == "pending",
        )
    )
    existing = existing_res.scalar_one_or_none()
    if existing:
        return UserUpgradeStatusResponse(
            has_pending=True,
            status="pending",
            created_at=existing.created_at,
            message="Your upgrade request has already been submitted and is pending review.",
        )

    # Create new upgrade request
    reason_text = payload.reason.strip() if payload and payload.reason else "User requested Pro tier from application settings."
    new_request = UpgradeRequest(
        user_id=current_user.id,
        requested_role="Pro",
        status="pending",
        reason=reason_text,
    )
    db.add(new_request)
    await db.commit()
    await db.refresh(new_request)

    return UserUpgradeStatusResponse(
        has_pending=True,
        status="pending",
        created_at=new_request.created_at,
        message="Upgrade request submitted successfully. An administrator will review your access.",
    )


@router.get("/upgrade-request", response_model=UserUpgradeStatusResponse)
async def get_my_upgrade_request_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Check the current user's latest upgrade request status.
    """
    user_role_str = (current_user.role.value if hasattr(current_user.role, "value") else str(current_user.role)).lower()
    if user_role_str in ("pro", "admin"):
        return UserUpgradeStatusResponse(
            has_pending=False,
            status="approved",
            message=f"You currently have {current_user.role} tier access.",
        )

    result = await db.execute(
        select(UpgradeRequest)
        .where(UpgradeRequest.user_id == current_user.id)
        .order_by(UpgradeRequest.created_at.desc())
        .limit(1)
    )
    latest = result.scalar_one_or_none()

    if not latest:
        return UserUpgradeStatusResponse(has_pending=False, status=None)

    return UserUpgradeStatusResponse(
        has_pending=latest.status == "pending",
        status=latest.status,
        created_at=latest.created_at,
        message="Your request is pending review by an admin." if latest.status == "pending" else f"Request {latest.status}.",
    )

