"""
backend/api/routes/auth.py

Authentication endpoints.

POST /auth/register   → create account, return token
POST /auth/login      → verify credentials, return token
GET  /auth/me         → return current user profile
"""

from fastapi import APIRouter, Depends, Form, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated
from backend.api.deps import get_current_user, get_db
from backend.models.auth_schemas import (
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
    ForgotPasswordRequest,
    ResetPasswordRequest,
)
from backend.models.models import User
from backend.services.auth_service import AuthService
from backend.services.security import create_access_token
from backend.services.email_service import send_reset_email


router = APIRouter()

_auth_service = AuthService()


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Annotated[AsyncSession, Depends(get_db)],
):
    try:
        user = await _auth_service.register_user(
            name=name,
            email=email,
            password=password,
            db=db,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )

    token = create_access_token(subject=user.id)
    return TokenResponse(access_token=token)


@router.post("/login", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    try:
        token = await _auth_service.login_user(
            email=form_data.username,
            password=form_data.password,
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

@router.post("/forgot-password")
async def forgot_password(
    email: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    token = await _auth_service.create_password_reset_token(email, db)
    if token:
        await send_reset_email(to_email=email, reset_token=token)
    
    # Always return a generic message to prevent email enumeration
    return {"message": "If an account with that email exists, a password reset link has been sent."}

@router.post("/reset-password")
async def reset_password(
    token: str = Form(...),
    new_password: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    try:
        await _auth_service.reset_password(token=token, new_password=new_password, db=db)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    return {"message": "Password successfully reset."}