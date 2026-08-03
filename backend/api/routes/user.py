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
from backend.models.models import User,Document

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas (small enough to live inline here)
# ---------------------------------------------------------------------------

class UserUpdate(BaseModel):
    name: str | None = None
    email: EmailStr | None = None


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