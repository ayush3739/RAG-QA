import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import status

from backend.main import app
from backend.api.deps import get_db
from backend.models.models import User, UserToken, UserTokenType, UserRole
from backend.services.security import hash_password, create_access_token

client = TestClient(app)

DUMMY_USER_ID = 1
DUMMY_EMAIL = "test@example.com"
DUMMY_PASSWORD = "securepassword123"
DUMMY_TOKEN = "dummy_access_token"
DUMMY_RESET_TOKEN = "dummy_reset_token"

@pytest.fixture(autouse=True)
def override_db_dependency():
    mock_db = AsyncMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    yield mock_db
    app.dependency_overrides.clear()


def test_register_success(override_db_dependency):
    mock_db = override_db_dependency

    # Mock DB query for existing user (returns None)
    mock_exec_1 = MagicMock()
    mock_exec_1.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_exec_1

    with patch("backend.api.routes.auth.send_verification_email", new_callable=AsyncMock) as mock_send_email:
        response = client.post(
            "/api/v1/auth/register",
            json={
                "name": "Test User",
                "email": DUMMY_EMAIL,
                "password": DUMMY_PASSWORD
            }
        )

    assert response.status_code == status.HTTP_201_CREATED
    assert "verify your email" in response.json()["message"]


def test_register_duplicate_email(override_db_dependency):
    mock_db = override_db_dependency

    # Mock DB query for existing user (returns existing user)
    existing_user = User(id=1, email=DUMMY_EMAIL, name="Existing")
    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.return_value = existing_user
    mock_db.execute.return_value = mock_exec

    response = client.post(
        "/api/v1/auth/register",
        json={
            "name": "Test User",
            "email": DUMMY_EMAIL,
            "password": DUMMY_PASSWORD
        }
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "already registered" in response.json()["detail"].lower()


def test_login_success(override_db_dependency):
    mock_db = override_db_dependency

    hashed_pw = hash_password(DUMMY_PASSWORD)
    user = User(id=DUMMY_USER_ID, email=DUMMY_EMAIL, password_hash=hashed_pw, is_verified=True, role=UserRole.USER)

    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.return_value = user
    mock_db.execute.return_value = mock_exec

    with patch("backend.api.routes.auth.persist_refresh_token", new_callable=AsyncMock):
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": DUMMY_EMAIL,
                "password": DUMMY_PASSWORD
            }
        )

    assert response.status_code == status.HTTP_200_OK
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"


def test_login_unverified_user(override_db_dependency):
    mock_db = override_db_dependency

    hashed_pw = hash_password(DUMMY_PASSWORD)
    user = User(id=DUMMY_USER_ID, email=DUMMY_EMAIL, password_hash=hashed_pw, is_verified=False, role=UserRole.USER)

    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.return_value = user
    mock_db.execute.return_value = mock_exec

    response = client.post(
        "/api/v1/auth/login",
        data={
            "username": DUMMY_EMAIL,
            "password": DUMMY_PASSWORD
        }
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert "not verified" in response.json()["detail"].lower()


def test_login_invalid_credentials(override_db_dependency):
    mock_db = override_db_dependency

    # Return None for non-existent user
    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_exec

    response = client.post(
        "/api/v1/auth/login",
        data={
            "username": DUMMY_EMAIL,
            "password": "wrongpassword"
        }
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "invalid email or password" in response.json()["detail"].lower()


def test_forgot_password_success(override_db_dependency):
    mock_db = override_db_dependency

    user = User(id=DUMMY_USER_ID, email=DUMMY_EMAIL, name="Test User")
    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.return_value = user
    mock_db.execute.return_value = mock_exec

    with patch("backend.api.routes.auth.send_reset_email", new_callable=AsyncMock) as mock_send_email:
        response = client.post(
            "/api/v1/auth/forgot-password",
            json={"email": DUMMY_EMAIL}
        )

    assert response.status_code == status.HTTP_200_OK
    assert "password reset link has been sent" in response.json()["message"]


def test_forgot_password_nonexistent_email(override_db_dependency):
    mock_db = override_db_dependency

    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_exec

    response = client.post(
        "/api/v1/auth/forgot-password",
        json={"email": "nonexistent@example.com"}
    )

    assert response.status_code == status.HTTP_200_OK
    # Generic security message returned regardless
    assert "password reset link has been sent" in response.json()["message"]
