import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import HTTPException, status

from backend.models.models import User, UserRole, DataUsage
from backend.services.quota_service import (
    get_quota_limit,
    get_today_bucket,
    check_and_consume_quota,
)


def test_get_quota_limit_by_role():
    """Verify tier limits return expected values."""
    assert get_quota_limit(UserRole.USER) == 10
    assert get_quota_limit(UserRole.PRO) == 500
    assert get_quota_limit(UserRole.ADMIN) is None


def test_first_query_of_the_day_inserts_record():
    """When no DataUsage row exists today, check_and_consume_quota creates one with count=1."""
    async def run():
        user = User(id=1, role=UserRole.USER, email="user@example.com", name="Regular User")
        mock_db = AsyncMock()
        mock_db.add = MagicMock()

        mock_exec = MagicMock()
        mock_exec.scalars.return_value.first.return_value = None
        mock_db.execute.return_value = mock_exec

        await check_and_consume_quota(user=user, db=mock_db)

        assert mock_db.add.called
        added_obj = mock_db.add.call_args[0][0]
        assert isinstance(added_obj, DataUsage)
        assert added_obj.user_id == 1
        assert added_obj.query_count == 1
        assert mock_db.commit.called

    asyncio.run(run())


def test_existing_query_within_limit_increments():
    """When usage is under the limit, count is incremented."""
    async def run():
        user = User(id=2, role=UserRole.USER, email="user2@example.com", name="Regular User")
        mock_db = AsyncMock()
        mock_db.add = MagicMock()

        usage_row = DataUsage(id=10, user_id=2, date=get_today_bucket(), query_count=5)
        mock_exec = MagicMock()
        mock_exec.scalars.return_value.first.return_value = usage_row
        mock_db.execute.return_value = mock_exec

        await check_and_consume_quota(user=user, db=mock_db)

        assert usage_row.query_count == 6
        assert mock_db.commit.called

    asyncio.run(run())


def test_quota_exceeded_raises_429():
    """When query count reaches or exceeds the role limit, raise HTTP 429."""
    async def run():
        user = User(id=3, role=UserRole.USER, email="user3@example.com", name="Regular User")
        mock_db = AsyncMock()
        mock_db.add = MagicMock()

        usage_row = DataUsage(id=11, user_id=3, date=get_today_bucket(), query_count=10)
        mock_exec = MagicMock()
        mock_exec.scalars.return_value.first.return_value = usage_row
        mock_db.execute.return_value = mock_exec

        with pytest.raises(HTTPException) as exc_info:
            await check_and_consume_quota(user=user, db=mock_db)

        assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert exc_info.value.detail["error"] == "QUOTA_EXCEEDED"
        assert exc_info.value.detail["limit"] == 10
        assert exc_info.value.detail["used"] == 10

    asyncio.run(run())


def test_admin_is_unlimited_and_tracked():
    """Admin has no limit but their query count is still tracked in DataUsage."""
    async def run():
        admin = User(id=99, role=UserRole.ADMIN, email="admin@example.com", name="Admin User")
        mock_db = AsyncMock()
        mock_db.add = MagicMock()

        usage_row = DataUsage(id=20, user_id=99, date=get_today_bucket(), query_count=100)
        mock_exec = MagicMock()
        mock_exec.scalars.return_value.first.return_value = usage_row
        mock_db.execute.return_value = mock_exec

        # Should not raise 429 despite count=100
        await check_and_consume_quota(user=admin, db=mock_db)

        assert usage_row.query_count == 101
        assert mock_db.commit.called

    asyncio.run(run())


def test_pro_user_allows_up_to_500():
    """PRO user is not blocked at 10 queries, but blocked at 500."""
    async def run():
        pro_user = User(id=4, role=UserRole.PRO, email="pro@example.com", name="Pro User")
        mock_db = AsyncMock()
        mock_db.add = MagicMock()

        usage_row = DataUsage(id=30, user_id=4, date=get_today_bucket(), query_count=499)
        mock_exec = MagicMock()
        mock_exec.scalars.return_value.first.return_value = usage_row
        mock_db.execute.return_value = mock_exec

        # 499th query -> increments to 500
        await check_and_consume_quota(user=pro_user, db=mock_db)
        assert usage_row.query_count == 500

        # 500th query -> raises 429
        with pytest.raises(HTTPException) as exc_info:
            await check_and_consume_quota(user=pro_user, db=mock_db)

        assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert exc_info.value.detail["limit"] == 500

    asyncio.run(run())
