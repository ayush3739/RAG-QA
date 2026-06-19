"""
backend/api/deps.py

FastAPI dependency functions.
Import these with Depends() in route handlers.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.security import decode_token
from backend.db.base import get_db
from backend.models.models import User
from backend.services.auth_service import AuthService
from backend.core.config import settings
from backend.core.retriever import Retriever


# Tells FastAPI where clients send their token.
# tokenUrl is used only for the OpenAPI docs "Authorize" button.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

_auth_service = AuthService()


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
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

    user_id = decode_token(token)
    if user_id is None:
        raise credentials_exception

    user = await _auth_service.get_user_by_id(user_id, db)
    if user is None:
        raise credentials_exception

    return user

async def get_settings():
    """Provide app settings."""
    return settings


async def get_retriever(collection_name: str) -> Retriever:
    """Provide retriever for a collection."""
    return Retriever(collection_name=collection_name)

# Re-export get_db so routes only need to import from deps
__all__ = ["get_db", "get_current_user", "oauth2_scheme"]