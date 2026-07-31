
from datetime import datetime, timezone
from typing import Tuple

from fastapi import HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.models import RefreshToken, User, UserToken, UserTokenType
from backend.services.security import (
    TokenType,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_reset_token,
)


def _decode_refresh_payload(refresh_token: str) -> dict:
    payload = decode_token(refresh_token, TokenType.REFRESH.value)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
    return payload


def _payload_expiry(payload: dict) -> datetime:
    exp = payload.get("exp")
    if exp is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    if isinstance(exp, datetime):
        return exp if exp.tzinfo else exp.replace(tzinfo=timezone.utc)

    try:
        return datetime.fromtimestamp(exp, tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )



async def persist_refresh_token(db: AsyncSession, user_id: int, refresh_token: str) -> RefreshToken:
    """
    Persist a new refresh token in the database for the given user.
    Multi-session support: existing refresh tokens are kept.
    """
    payload = _decode_refresh_payload(refresh_token)
    jti = payload.get("jti")
    if not jti:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    refresh_entry = RefreshToken(
        user_id=user_id,
        token_hash=hash_reset_token(refresh_token),
        jti=jti,
        expires_at=_payload_expiry(payload),
    )

    try:
        db.add(refresh_entry)
        await db.commit()
        await db.refresh(refresh_entry)
        return refresh_entry
    except Exception:
        await db.rollback()
        raise


async def validate_refresh_token(db: AsyncSession, refresh_token: str) -> Tuple[RefreshToken, dict]:
    payload = _decode_refresh_payload(refresh_token)
    jti = payload.get("jti")
    if not jti:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    result = await db.execute(select(RefreshToken).where(RefreshToken.jti == jti))
    refresh_entry = result.scalar_one_or_none()
    if not refresh_entry:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    if refresh_entry.expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired",
        )

    if refresh_entry.token_hash != hash_reset_token(refresh_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    return refresh_entry, payload


async def revoke_refresh_token(db: AsyncSession, refresh_token: str) -> None:
    refresh_entry, _ = await validate_refresh_token(db, refresh_token)
    try:
        await db.delete(refresh_entry)
        await db.commit()
    except Exception:
        await db.rollback()
        raise


async def refresh_access_token(db: AsyncSession, refresh_token: str) -> str:
    _, payload = await validate_refresh_token(db, refresh_token)
    user_id = payload.get("sub")
    try:
        user_id_int = int(user_id)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    result = await db.execute(select(User).where(User.id == user_id_int))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    return create_access_token(user.id, user.role.value)


async def rotate_refresh_token(db: AsyncSession, old_refresh_token: str) -> tuple[str, str]:
    """
    Validate old refresh token, rotate it, and return new (access, refresh) tokens.

    Flow:
    1. Validate old refresh token
    2. Generate new refresh token
    3. Delete old DB row
    4. Insert new DB row
    5. Return new access + refresh token
    """
    old_entry, payload = await validate_refresh_token(db, old_refresh_token)

    user_id = payload.get("sub")
    try:
        user_id_int = int(user_id)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    user_result = await db.execute(select(User).where(User.id == user_id_int))
    user = user_result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    new_refresh_token = create_refresh_token(user.id)
    refresh_payload = _decode_refresh_payload(new_refresh_token)
    refresh_jti = refresh_payload.get("jti")
    if not refresh_jti:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    new_entry = RefreshToken(
        user_id=user.id,
        token_hash=hash_reset_token(new_refresh_token),
        jti=refresh_jti,
        expires_at=_payload_expiry(refresh_payload),
    )

    try:
        await db.delete(old_entry)
        db.add(new_entry)
        await db.commit()
    except Exception:
        await db.rollback()
        raise

    access_token = create_access_token(user.id, user.role.value)
    return access_token, new_refresh_token


async def verify_email(db: AsyncSession, verification_token: str) -> None:
    hashed_token = hash_reset_token(verification_token)
    result = await db.execute(
        select(UserToken).where(
            UserToken.token_hash == hashed_token,
            UserToken.token_type == UserTokenType.EMAIL_VERIFICATION,
        )
    )
    verification_entry = result.scalar_one_or_none()
    if not verification_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid or expired verification token",
        )
    if verification_entry.expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token has expired",
        )
    user_result = await db.execute(select(User).where(User.id == verification_entry.user_id))
    user = user_result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    try:
        user.is_verified = True
        db.add(user)
        await db.execute(
            delete(UserToken).where(
                UserToken.user_id == user.id,
                UserToken.token_type == UserTokenType.EMAIL_VERIFICATION,
            )
        )
        await db.commit()
    except Exception:
        await db.rollback()
        raise
