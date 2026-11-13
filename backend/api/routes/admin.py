"""
backend/api/routes/admin.py

Internal Admin portal endpoints for user access management, role promotion, and Pro upgrade requests.
All endpoints require ADMIN role (require_admin).
"""

from datetime import datetime, UTC
from typing import Annotated, Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select, or_, func, desc, cast, String
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user, get_db
from backend.core.rbac import require_admin
from backend.models.models import User, UserRole, Document, DataUsage, UpgradeRequest

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AdminStatsResponse(BaseModel):
    total_users: int
    pro_users: int
    admin_users: int
    pending_requests: int


class AdminUserItem(BaseModel):
    id: int
    name: str
    email: str
    role: str
    is_verified: bool
    created_at: datetime
    document_count: int = 0
    query_count: int = 0
    has_pending_upgrade: bool = False

    model_config = {"from_attributes": True}


class AdminUserListResponse(BaseModel):
    users: List[AdminUserItem]
    total: int
    limit: int
    skip: int


class UpdateRoleRequest(BaseModel):
    role: str = Field(..., description="Target role: 'user', 'pro', or 'admin' (case-insensitive)")


class UpgradeRequestItem(BaseModel):
    id: int
    user_id: int
    user_name: str
    user_email: str
    current_role: str
    requested_role: str
    status: str
    reason: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Dependency: ensure authenticated user is admin
# ---------------------------------------------------------------------------

async def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    return require_admin(current_user)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/stats", response_model=AdminStatsResponse)
async def get_admin_stats(
    admin_user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get high-level administrator metrics (total users, tier breakdowns, pending requests).
    Runs fast aggregate queries to eliminate N+1 resource exhaustion.
    """
    total_res = await db.execute(select(func.count(User.id)))
    total_users = total_res.scalar() or 0

    pro_res = await db.execute(
        select(func.count(User.id)).where(func.lower(cast(User.role, String)) == "pro")
    )
    pro_users = pro_res.scalar() or 0

    admin_res = await db.execute(
        select(func.count(User.id)).where(func.lower(cast(User.role, String)) == "admin")
    )
    admin_users = admin_res.scalar() or 0

    pending_res = await db.execute(
        select(func.count(UpgradeRequest.id)).where(UpgradeRequest.status == "pending")
    )
    pending_requests = pending_res.scalar() or 0

    return AdminStatsResponse(
        total_users=total_users,
        pro_users=pro_users,
        admin_users=admin_users,
        pending_requests=pending_requests,
    )


@router.get("/users", response_model=AdminUserListResponse)
async def list_admin_users(
    search: Optional[str] = Query(None, description="Search by name or email"),
    role_filter: Optional[str] = Query(None, alias="role", description="Filter by role"),
    limit: int = Query(50, ge=1, le=100),
    skip: int = Query(0, ge=0),
    admin_user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Search and list users with pagination (default 50 users per page).
    Uses single outer-joined subqueries to completely prevent N+1 DB calls.
    """
    # 1. Base filter conditions
    filter_conditions = []
    if search and search.strip():
        search_term = f"%{search.strip()}%"
        filter_conditions.append(
            or_(
                User.email.ilike(search_term),
                User.name.ilike(search_term),
            )
        )

    if role_filter and role_filter.strip() and role_filter.strip().lower() != "all":
        clean_role = role_filter.strip().lower()
        if clean_role in ("pro", "admin", "user"):
            filter_conditions.append(func.lower(cast(User.role, String)) == clean_role)

    # 2. Total Count (Single aggregate query)
    count_query = select(func.count(User.id))
    if filter_conditions:
        count_query = count_query.where(*filter_conditions)
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # 3. Subqueries for user metrics (aggregated once in DB)
    doc_subq = (
        select(Document.user_id, func.count(Document.id).label("doc_count"))
        .group_by(Document.user_id)
        .subquery()
    )
    usage_subq = (
        select(DataUsage.user_id, func.coalesce(func.sum(DataUsage.query_count), 0).label("query_count"))
        .group_by(DataUsage.user_id)
        .subquery()
    )
    upgrade_subq = (
        select(UpgradeRequest.user_id, func.count(UpgradeRequest.id).label("pending_count"))
        .where(UpgradeRequest.status == "pending")
        .group_by(UpgradeRequest.user_id)
        .subquery()
    )

    # 4. Paged query joined with subqueries
    paged_query = (
        select(
            User,
            func.coalesce(doc_subq.c.doc_count, 0).label("doc_count"),
            func.coalesce(usage_subq.c.query_count, 0).label("query_count"),
            func.coalesce(upgrade_subq.c.pending_count, 0).label("pending_count"),
        )
        .outerjoin(doc_subq, User.id == doc_subq.c.user_id)
        .outerjoin(usage_subq, User.id == usage_subq.c.user_id)
        .outerjoin(upgrade_subq, User.id == upgrade_subq.c.user_id)
    )

    if filter_conditions:
        paged_query = paged_query.where(*filter_conditions)

    paged_query = paged_query.order_by(desc(User.created_at)).offset(skip).limit(limit)
    users_result = await db.execute(paged_query)
    rows = users_result.all()

    items: List[AdminUserItem] = []
    for u, doc_cnt, q_cnt, pend_cnt in rows:
        role_str = u.role.value if hasattr(u.role, "value") else str(u.role)
        items.append(
            AdminUserItem(
                id=u.id,
                name=u.name,
                email=u.email,
                role=role_str,
                is_verified=u.is_verified,
                created_at=u.created_at,
                document_count=int(doc_cnt),
                query_count=int(q_cnt),
                has_pending_upgrade=bool(pend_cnt > 0),
            )
        )

    return AdminUserListResponse(users=items, total=total, limit=limit, skip=skip)


@router.patch("/users/{user_id}/role", response_model=AdminUserItem)
async def update_user_role(
    user_id: int,
    payload: UpdateRoleRequest,
    admin_user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update a user's role to 'user', 'pro', or 'admin'.
    Promoting to Pro or Admin automatically marks any pending upgrade request as approved.
    Prevents an admin from accidentally demoting themselves.
    """
    clean_role = payload.role.strip().lower()
    role_map = {
        "user": UserRole.USER,
        "pro": UserRole.PRO,
        "admin": UserRole.ADMIN,
    }

    if clean_role not in role_map:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role '{payload.role}'. Must be one of: user, pro, admin.",
        )

    if user_id == admin_user.id and clean_role != "admin":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot demote your own active admin account.",
        )

    target_role = role_map[clean_role]

    user_result = await db.execute(select(User).where(User.id == user_id))
    target_user = user_result.scalar_one_or_none()
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found.",
        )

    target_user.role = target_role
    target_user.updated_at = datetime.now(UTC)

    # If promoted to Pro or Admin, resolve any pending upgrade requests
    if clean_role in ("pro", "admin"):
        requests_res = await db.execute(
            select(UpgradeRequest).where(
                UpgradeRequest.user_id == user_id,
                UpgradeRequest.status == "pending",
            )
        )
        for req in requests_res.scalars().all():
            req.status = "approved"
            req.updated_at = datetime.now(UTC)

    await db.commit()
    await db.refresh(target_user)

    # Document & Query counts for response
    doc_count_res = await db.execute(
        select(func.count()).select_from(Document).where(Document.user_id == target_user.id)
    )
    doc_count = doc_count_res.scalar_one_or_none() or 0

    query_sum_res = await db.execute(
        select(func.coalesce(func.sum(DataUsage.query_count), 0)).where(DataUsage.user_id == target_user.id)
    )
    query_count = query_sum_res.scalar_one_or_none() or 0

    role_str = target_user.role.value if hasattr(target_user.role, "value") else str(target_user.role)
    return AdminUserItem(
        id=target_user.id,
        name=target_user.name,
        email=target_user.email,
        role=role_str,
        is_verified=target_user.is_verified,
        created_at=target_user.created_at,
        document_count=doc_count,
        query_count=query_count,
        has_pending_upgrade=False,
    )


@router.get("/upgrade-requests", response_model=List[UpgradeRequestItem])
async def list_upgrade_requests(
    status_filter: Optional[str] = Query("pending", description="Filter by status: 'pending', 'approved', 'rejected', or 'all'"),
    admin_user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List user requests to be upgraded to Pro account.
    """
    query = (
        select(UpgradeRequest, User)
        .join(User, UpgradeRequest.user_id == User.id)
        .order_by(desc(UpgradeRequest.created_at))
    )

    if status_filter and status_filter.lower() != "all":
        query = query.where(UpgradeRequest.status == status_filter.lower())

    results = await db.execute(query)
    rows = results.all()

    items: List[UpgradeRequestItem] = []
    for req, u in rows:
        role_str = u.role.value if hasattr(u.role, "value") else str(u.role)
        items.append(
            UpgradeRequestItem(
                id=req.id,
                user_id=u.id,
                user_name=u.name,
                user_email=u.email,
                current_role=role_str,
                requested_role=req.requested_role,
                status=req.status,
                reason=req.reason,
                created_at=req.created_at,
                updated_at=req.updated_at,
            )
        )

    return items


@router.post("/upgrade-requests/{request_id}/approve")
async def approve_upgrade_request(
    request_id: int,
    admin_user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Approve an upgrade request, instantly granting the user Pro status.
    """
    result = await db.execute(select(UpgradeRequest).where(UpgradeRequest.id == request_id))
    req = result.scalar_one_or_none()
    if not req:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upgrade request {request_id} not found.",
        )

    user_res = await db.execute(select(User).where(User.id == req.user_id))
    target_user = user_res.scalar_one_or_none()
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Associated user not found.",
        )

    target_user.role = UserRole.PRO
    target_user.updated_at = datetime.now(UTC)
    req.status = "approved"
    req.updated_at = datetime.now(UTC)

    await db.commit()

    return {
        "message": f"Upgrade request approved. User {target_user.email} is now Pro.",
        "user_id": target_user.id,
        "new_role": target_user.role.value if hasattr(target_user.role, "value") else str(target_user.role),
        "status": "approved",
    }


@router.post("/upgrade-requests/{request_id}/reject")
async def reject_upgrade_request(
    request_id: int,
    admin_user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Reject an upgrade request.
    """
    result = await db.execute(select(UpgradeRequest).where(UpgradeRequest.id == request_id))
    req = result.scalar_one_or_none()
    if not req:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upgrade request {request_id} not found.",
        )

    req.status = "rejected"
    req.updated_at = datetime.now(UTC)
    await db.commit()

    return {"message": "Upgrade request rejected.", "status": "rejected"}
