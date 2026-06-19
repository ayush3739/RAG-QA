"""
backend/services/auth_service.py

Business logic for registration, login, and user lookup.
Keeps route handlers thin — all DB interaction lives here.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.security import hash_password, verify_password, create_access_token
from backend.models.models import User


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
            hashed_password=hash_password(password),
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

        if not user or not verify_password(password, user.hashed_password):
            raise ValueError("Invalid email or password")

        return create_access_token(subject=user.id)