"""
backend/services/auth_service.py

Business logic for registration, login, and user lookup.
Keeps route handlers thin — all DB interaction lives here.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.security import (
    hash_password, 
    verify_password, 
    create_access_token,
    generate_reset_token,
    hash_reset_token
)
from backend.models.models import User, PasswordResetToken


class AuthService:

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    async def get_user_by_email(
        self,
        email: str,
        db: AsyncSession,
    ) -> User | None:
        result = await db.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def get_user_by_id(
        self,
        user_id: int,
        db: AsyncSession,
    ) -> User | None:
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # Register
    # ------------------------------------------------------------------

    async def register_user(
        self,
        name: str,
        email: str,
        password: str,
        db: AsyncSession,
    ) -> User:
        existing = await self.get_user_by_email(email, db)
        if existing:
            raise ValueError("Email already registered")

        user = User(
            name=name,
            email=email,
            password_hash=hash_password(password),
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user

    # ------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------

    async def login_user(
        self,
        email: str,
        password: str,
        db: AsyncSession,
    ) -> str:
        """
        Validates credentials and returns a JWT access token.
        Raises ValueError on bad credentials so the route can return 401.
        """
        user = await self.get_user_by_email(email, db)

        if not user or not verify_password(password, user.password_hash):
            raise ValueError("Invalid email or password")

        return create_access_token(subject=user.id)

    # ------------------------------------------------------------------
    # Password Reset
    # ------------------------------------------------------------------

    async def create_password_reset_token(
        self,
        email: str,
        db: AsyncSession,
    ) -> str | None:
        """
        Generates a token and stores its hash in the DB.
        Returns the raw token string so it can be emailed to the user.
        Returns None if user does not exist.
        """
        user = await self.get_user_by_email(email, db)
        if not user:
            return None

        # Delete any existing tokens for this user
        await db.execute(
            delete(PasswordResetToken).where(PasswordResetToken.user_id == user.id)
        )

        raw_token = generate_reset_token()
        token_hash = hash_reset_token(raw_token)
        expires = datetime.now(timezone.utc) + timedelta(hours=1)

        reset_token = PasswordResetToken(
            user_id=user.id,
            token_hash=token_hash,
            expires_at=expires,
        )
        db.add(reset_token)
        await db.commit()

        return raw_token

    async def reset_password(
        self,
        token: str,
        new_password: str,
        db: AsyncSession,
    ):
        """
        Validates token hash, checks expiration, updates user password, and deletes token.
        Raises ValueError if token is invalid or expired.
        """
        token_hash = hash_reset_token(token)
        
        result = await db.execute(
            select(PasswordResetToken).where(PasswordResetToken.token_hash == token_hash)
        )
        reset_entry = result.scalar_one_or_none()

        if not reset_entry:
            raise ValueError("Invalid or expired reset token")
            
        if reset_entry.expires_at < datetime.now(timezone.utc):
            await db.execute(
                delete(PasswordResetToken).where(PasswordResetToken.id == reset_entry.id)
            )
            await db.commit()
            raise ValueError("Invalid or expired reset token")

        # Valid token, update user password
        user = await self.get_user_by_id(reset_entry.user_id, db)
        if not user:
            raise ValueError("User not found")

        user.password_hash = hash_password(new_password)
        
        # Clean up all reset tokens for this user
        await db.execute(
            delete(PasswordResetToken).where(PasswordResetToken.user_id == user.id)
        )
        await db.commit()