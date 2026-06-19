"""
backend/api/routes/auth.py

Authentication endpoints.

POST /auth/register   → create account, return token
POST /auth/login      → verify credentials, return token
GET  /auth/me         → return current user profile
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user, get_db
from backend.models.auth_schemas import (
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
)
from backend.models.models import User
from backend.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])

_auth_service = AuthService()


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    body: UserRegister,
    db: AsyncSession = Depends(get_db),
):
    try:
        user = await _auth_service.register_user(
            name=body.name,
            email=body.email,
            password=body.password,
            db=db,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )

    from backend.services.security import create_access_token
    token = create_access_token(subject=user.id)
    return TokenResponse(access_token=token)


@router.post("/login", response_model=TokenResponse)
async def login(
    body: UserLogin,
    db: AsyncSession = Depends(get_db),
):
    try:
        token = await _auth_service.login_user(
            email=body.email,
            password=body.password,
            db=db,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )

    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return current_user