from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, UTC
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.api.deps import get_current_user, get_db
from backend.models.models import User, UserRole

client = TestClient(app)


@pytest.fixture
def mock_admin_user():
    user = MagicMock(spec=User)
    user.id = 1
    user.name = "Admin User"
    user.email = "admin@knowvex.ai"
    user.role = UserRole.ADMIN
    user.is_verified = True
    user.created_at = datetime.now(UTC)
    return user


@pytest.fixture
def mock_regular_user():
    user = MagicMock(spec=User)
    user.id = 2
    user.name = "Regular User"
    user.email = "user@knowvex.ai"
    user.role = UserRole.USER
    user.is_verified = True
    user.created_at = datetime.now(UTC)
    return user


def test_regular_user_forbidden_from_admin_portal(mock_regular_user):
    """A regular user must be rejected with 403 Forbidden on admin endpoints."""
    app.dependency_overrides[get_current_user] = lambda: mock_regular_user
    mock_db = AsyncMock()
    app.dependency_overrides[get_db] = lambda: mock_db

    response = client.get("/api/v1/admin/users")
    assert response.status_code == 403
    assert "Action requires" in response.json()["detail"]

    app.dependency_overrides.clear()


def test_admin_can_access_user_search(mock_admin_user):
    """An admin user can search and list users."""
    app.dependency_overrides[get_current_user] = lambda: mock_admin_user
    mock_db = AsyncMock()
    
    target_user = MagicMock(spec=User)
    target_user.id = 2
    target_user.name = "Searched User"
    target_user.email = "search@knowvex.ai"
    target_user.role = UserRole.USER
    target_user.is_verified = True
    target_user.created_at = datetime.now(UTC)

    mock_total_result = MagicMock()
    mock_total_result.scalar_one_or_none.return_value = 1

    mock_users_result = MagicMock()
    mock_users_result.scalars.return_value.all.return_value = [target_user]

    mock_count_result = MagicMock()
    mock_count_result.scalar_one_or_none.return_value = 0

    mock_db.execute = AsyncMock(side_effect=[
        mock_total_result,  # count
        mock_users_result,  # users
        mock_count_result,  # doc count
        mock_count_result,  # query count
        mock_count_result,  # pending upgrade count
    ])
    app.dependency_overrides[get_db] = lambda: mock_db

    response = client.get("/api/v1/admin/users?search=search@knowvex.ai")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["users"]) == 1
    assert data["users"][0]["email"] == "search@knowvex.ai"

    app.dependency_overrides.clear()


def test_admin_can_promote_user_to_pro(mock_admin_user):
    """An admin can update a user's role to pro."""
    app.dependency_overrides[get_current_user] = lambda: mock_admin_user
    mock_db = AsyncMock()

    target_user = MagicMock(spec=User)
    target_user.id = 2
    target_user.name = "Target"
    target_user.email = "target@knowvex.ai"
    target_user.role = UserRole.USER
    target_user.is_verified = True
    target_user.created_at = datetime.now(UTC)

    user_query_res = MagicMock()
    user_query_res.scalar_one_or_none.return_value = target_user

    pending_reqs_res = MagicMock()
    pending_reqs_res.scalars.return_value.all.return_value = []

    mock_count_res = MagicMock()
    mock_count_res.scalar_one_or_none.return_value = 0

    mock_db.execute = AsyncMock(side_effect=[
        user_query_res,     # find user
        pending_reqs_res,   # pending requests
        mock_count_res,     # doc count
        mock_count_res,     # query count
    ])
    mock_db.commit = AsyncMock()
    mock_db.refresh = AsyncMock()
    app.dependency_overrides[get_db] = lambda: mock_db

    response = client.patch("/api/v1/admin/users/2/role", json={"role": "pro"})
    assert response.status_code == 200
    assert target_user.role == UserRole.PRO

    app.dependency_overrides.clear()


def test_user_can_submit_upgrade_request(mock_regular_user):
    """A regular user can submit an upgrade request."""
    app.dependency_overrides[get_current_user] = lambda: mock_regular_user
    mock_db = AsyncMock()

    existing_res = MagicMock()
    existing_res.scalar_one_or_none.return_value = None
    mock_db.execute = AsyncMock(return_value=existing_res)
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    
    async def fake_refresh(obj):
        obj.created_at = datetime.now(UTC)
    mock_db.refresh = AsyncMock(side_effect=fake_refresh)

    app.dependency_overrides[get_db] = lambda: mock_db

    response = client.post("/api/v1/user/upgrade-request", json={"reason": "Need higher query limits"})
    assert response.status_code == 200
    data = response.json()
    assert data["has_pending"] is True
    assert data["status"] == "pending"

    app.dependency_overrides.clear()
