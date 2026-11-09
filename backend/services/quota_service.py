from datetime import datetime, time,UTC
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends, HTTPException, status

from backend.models.models import User, UserRole, DataUsage

ROLE_QUOTA = {
    UserRole.FREE: 10, 
    UserRole.PRO: 500,
    UserRole.ADMIN: None,
}

class QuotaService:
    @staticmethod
    def get_today_bucket() -> datetime:
        today = datetime.now(UTC).date()
        return datetime.combine(today, time.min, tzinfo=UTC)
    async def get_usage(self, user_id: int, db: AsyncSession) -> int:
        today_start = self.get_today_bucket()
        result = await db.execute(
            select(DataUsage.query_count).where(
                DataUsage.user_id == user_id,
                DataUsage.date == today_start,
            )
        )
        return result.scalar_one_or_none() or 0
    async def check_and_consume_quota(self, user: User, db: AsyncSession) -> None:
        limit = ROLE_QUOTA.get(user.role)
        if limit is None:
            return  # Unlimited for Admin
        today_start = self.get_today_bucket()
        result = await db.execute(
            select(DataUsage).where(
                DataUsage.user_id == user.id,
                DataUsage.date == today_start,
            )
        )
        usage = result.scalars().first()
        current_count = usage.query_count if usage else 0
        if current_count >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "QUOTA_EXCEEDED",
                    "message": f"Daily limit of {limit} queries reached for your plan.",
                    "limit": limit,
                    "used": current_count,
                    "upgrade_url": "/pricing",
                },
            )
        # Atomic increment or insert
        if usage:
            usage.query_count += 1
        else:
            usage = DataUsage(user_id=user.id, date=today_start, query_count=1)
            db.add(usage)
        await db.commit()
