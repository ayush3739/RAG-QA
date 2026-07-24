"""
backend/api/routes/auth.py

Authentication endpoints.

POST /auth/register   → create account, return token
POST /auth/login      → verify credentials, return token
GET  /auth/me         → return current user profile
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated
from backend.api.deps import get_current_user, get_db
from backend.models.auth_schemas import (
    TokenResponse,
    UserRegister,
    UserResponse,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    RefreshTokenRequest,
)
from backend.models.models import User

from backend.services.auth_service import (
    persist_refresh_token,
    revoke_refresh_token,
    rotate_refresh_token,
)
from backend.services.security import (
    hash_password, 
    verify_password, 
    create_access_token,
    generate_reset_token,
    create_refresh_token,
    hash_reset_token,

)
from backend.models.models import PasswordResetToken
from backend.services.email_service import send_reset_email


router = APIRouter()


@router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return current_user



@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(user: UserRegister,db: Annotated[AsyncSession, Depends(get_db)],):
    # check if user already exists
    result = await db.execute(select(User).where(User.email == user.email))
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    password_hash = hash_password(user.password)
    new_user = User(email=user.email, password_hash=password_hash, name=user.name)
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    access_token = create_access_token(new_user.id, new_user.role.value)
    refresh_token = create_refresh_token(new_user.id)
    await persist_refresh_token(db, new_user.id, refresh_token)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )   
    


@router.post("/login", response_model=TokenResponse)
async def login(form_data: Annotated[OAuth2PasswordRequestForm,Depends()] ,db: Annotated[AsyncSession, Depends(get_db)],):
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,   
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified. Please check your inbox.",
        )
    if not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(user.id, user.role.value)
    refresh_token = create_refresh_token(user.id)

    # Persist refresh token with single-session semantics.
    try:
        await persist_refresh_token(db, user.id, refresh_token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error storing refresh token",
        )

    return TokenResponse(access_token=access_token,refresh_token=refresh_token,token_type="bearer")



@router.post("/logout")
async def logout(
    refresh_request: RefreshTokenRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    await revoke_refresh_token(db, refresh_request.refresh_token)

    return {"message": "Successfully logged out."}




@router.post("/forgot-password")
async def forgot_password(
    email: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.email == email.email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    reset_token = generate_reset_token()
    hashed_token = hash_reset_token(reset_token)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    reset_entry = PasswordResetToken(user_id=user.id, token_hash=hashed_token, expires_at=expires_at)
    db.add(reset_entry)
    await db.commit()

    await send_reset_email(to_email=email.email, reset_token=reset_token)

    
    return {"message": "If an account with that email exists, a password reset link has been sent."}

    


@router.post("/reset-password")
async def reset_password(
    reset_request: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    hashed_token = hash_reset_token(reset_request.token)
    result = await db.execute(select(PasswordResetToken).where(PasswordResetToken.token_hash == hashed_token))
    reset_entry = result.scalar_one_or_none()
    if not reset_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid or expired reset token",
        )
    if reset_entry.expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token has expired",
        )
    user_result = await db.execute(select(User).where(User.id == reset_entry.user_id))
    user = user_result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    user.password_hash = hash_password(reset_request.new_password)
    db.add(user)
    await db.delete(reset_entry)
    await db.commit()
    return {"message": "Password successfully reset."}


@router.post("/refresh-token", response_model=TokenResponse)
async def refresh_token(
    refresh_request: RefreshTokenRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    access_token, new_refresh_token = await rotate_refresh_token(db, refresh_request.refresh_token)

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
    )

    