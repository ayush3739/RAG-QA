from datetime import datetime, time, timedelta, UTC
from sqlalchemy import select,func
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends, HTTPException, status

from backend.models.models import User, UserRole, DataUsage, Document

ROLE_QUOTA = {
    UserRole.USER: 10, 
    UserRole.PRO: 500,
    UserRole.ADMIN: None,
}

ROLE_DOCUMENT_LIMITS = {
    UserRole.USER: 3,
    UserRole.PRO: 50,
    UserRole.ADMIN: None,
}

ROLE_PAGE_LIMITS = {
    UserRole.USER: 15,
    UserRole.PRO: 300,
    UserRole.ADMIN: None,
}

ROLE_RESEARCH_LIMITS = {
    UserRole.USER: 1,
    UserRole.PRO: 25,
    UserRole.ADMIN: None,
}

def get_today_bucket() -> datetime:
    today = datetime.now(UTC).date()
    return datetime.combine(today, time.min, tzinfo=UTC)


def get_quota_limit(role: UserRole) -> int | None:
    return ROLE_QUOTA.get(role)


async def get_usage(user_id: int, db: AsyncSession) -> int:
    today_start = get_today_bucket()
    result = await db.execute(
        select(DataUsage.query_count).where(
            DataUsage.user_id == user_id,
            DataUsage.date == today_start,
        )
    )
    return result.scalar_one_or_none() or 0


async def get_user_quota_info(user: User, db: AsyncSession) -> dict:
    today_start = get_today_bucket()
    result = await db.execute(
        select(DataUsage).where(
            DataUsage.user_id == user.id,
            DataUsage.date == today_start,
        )
    )
    usage = result.scalars().first()
    queries_used = usage.query_count if usage else 0
    research_used = getattr(usage, "research_count", 0) if usage else 0

    query_limit = ROLE_QUOTA.get(user.role)
    research_limit = ROLE_RESEARCH_LIMITS.get(user.role)

    query_remaining = max(0, query_limit - queries_used) if query_limit is not None else None
    research_remaining = max(0, research_limit - research_used) if research_limit is not None else None
    reset_at = today_start + timedelta(days=1)

    return {
        "role": user.role.value if hasattr(user.role, "value") else str(user.role),
        "queries_used": queries_used,
        "query_limit": query_limit,
        "queries_remaining": query_remaining,
        "research_used": research_used,
        "research_limit": research_limit,
        "research_remaining": research_remaining,
        "reset_at": reset_at,
    }


async def check_and_consume_quota(user: User, db: AsyncSession) -> None:
    today_start = get_today_bucket()
    result = await db.execute(
        select(DataUsage).where(
            DataUsage.user_id == user.id,
            DataUsage.date == today_start,
        )
    )
    usage = result.scalars().first()
    current_count = usage.query_count if usage else 0

    limit = ROLE_QUOTA.get(user.role)
    if limit is not None and current_count >= limit:
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


async def check_document_upload_quota(user: User, db: AsyncSession) -> None:
    limit = ROLE_DOCUMENT_LIMITS.get(user.role)
    if limit is None:
        return  # No limit for this role

    result = await db.execute(
        select(func.count(Document.id)).where(Document.user_id == user.id)
    )
    current_count = result.scalar_one() or 0

    if current_count >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "DOCUMENT_UPLOAD_QUOTA_EXCEEDED",
                "message": f"Daily document upload limit of {limit} reached for your plan.",
                "limit": limit,
                "used": current_count,
                "upgrade_url": "/pricing",
            },
        )


async def  check_and_consume_research_quota(user:User,db : AsyncSession) -> None:
    limit = ROLE_RESEARCH_LIMITS.get(user.role)
    today_start = get_today_bucket()

    result = await db.execute(
        select(DataUsage).where(
            DataUsage.user_id == user.id,
            DataUsage.date == today_start,
        )
    )
    usage = result.scalars().first()
    current_count = usage.research_count if usage else 0

    if limit is not None and current_count >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "RESEARCH_QUOTA_EXCEEDED",
                "message": f"Daily deep research limit of {limit} reached for your plan.",
                "limit": limit,
                "used": current_count,
            },
        )

    # Atomic increment or insert
    if usage:
        usage.research_count += 1
    else:
        usage = DataUsage(user_id=user.id, date=today_start,query_count = 0, research_count=1)
        db.add(usage)
    await db.commit()


def validate_document_pages(num_pages:int,role: UserRole) -> None:
    limit = ROLE_PAGE_LIMITS.get(role)
    if limit is not None and num_pages > limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "PAGE_LIMIT_EXCEEDED",
                "message": f"Document has {num_pages} pages, which exceeds your tier limit of {limit} pages.",
                "pages": num_pages,
                "limit": limit,
            },
        )

