from datetime import datetime, time, timedelta, UTC
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.models import DataUsage


class AnalyticsService:

    async def record_user_query(self, user_id: int, db: AsyncSession) -> None:
        """Increment the query count for the user for today's UTC date bucket."""
        today = datetime.now(UTC).date()
        today_start = datetime.combine(today, time.min, tzinfo=UTC)

        result = await db.execute(
            select(DataUsage).where(
                and_(
                    DataUsage.user_id == user_id,
                    DataUsage.date == today_start,
                )
            )
        )
        usage = result.scalars().first()

        if usage:
            usage.query_count += 1
        else:
            usage = DataUsage(
                user_id=user_id,
                date=today_start,
                query_count=1,
            )
            db.add(usage)

        await db.commit()

    async def get_user_activity(
        self, user_id: int, db: AsyncSession, days: int = 7
    ) -> list[dict]:
        """
        Return query activity for the past `days` days,
        including zero-count entries for days with no activity.
        """
        today = datetime.now(UTC).date()
        start_date = today - timedelta(days=days - 1)
        start_datetime = datetime.combine(start_date, time.min, tzinfo=UTC)

        result = await db.execute(
            select(DataUsage)
            .where(
                and_(
                    DataUsage.user_id == user_id,
                    DataUsage.date >= start_datetime,
                )
            )
            .order_by(DataUsage.date.asc())
        )
        records = result.scalars().all()

        usage_map = {
            r.date.date().isoformat(): r.query_count for r in records if r.date
        }

        activity = []
        for i in range(days):
            current_day = start_date + timedelta(days=i)
            iso_day = current_day.isoformat()
            day_label = current_day.strftime("%a")  # e.g., 'Mon', 'Tue'
            activity.append(
                {
                    "name": day_label,
                    "date": iso_day,
                    "queries": usage_map.get(iso_day, 0),
                }
            )

        return activity
