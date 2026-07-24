"""
backend/api/deps.py

FastAPI dependency functions.
Import these with Depends() in route handlers.
"""

from typing import Annotated

from sqlalchemy import select
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.security import decode_token
from backend.db.base import get_db
from backend.models.models import User
from backend.core.config import settings

# Tells FastAPI where clients send their token.
# tokenUrl is used only for the OpenAPI docs "Authorize" button.

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """
    Decode the Bearer token and return the matching User ORM object.
    Raises HTTP 401 if the token is missing, expired, or the user no longer exists.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_token(token,"access")
    if payload is None:
        raise credentials_exception

    user_id = int(payload.get("sub"))

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception

    return user

# Re-export get_db so routes only need to import from deps
__all__ = ["get_db", "get_current_user", "oauth2_scheme"]