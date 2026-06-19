import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from backend.main import app
from backend.models.models import User
from backend.models.auth_schemas import TokenResponse

client = TestClient(app)

# Dummy data
DUMMY_USER_ID = 1
DUMMY_EMAIL = "test@example.com"
DUMMY_PASSWORD = "securepassword"
DUMMY_TOKEN = "dummy_access_token"
DUMMY_RESET_TOKEN = "dummy_reset_token"


@pytest.fixture
def mock_auth_service():
    with patch("backend.api.routes.auth._auth_service") as mock_service:
        yield mock_service

@pytest.fixture
def mock_email_service():
    with patch("backend.api.routes.auth.send_reset_email") as mock_send:
        yield mock_send


def test_register_success(mock_auth_service):
    # Setup mock
    mock_user = User(id=DUMMY_USER_ID, name="Test User", email=DUMMY_EMAIL)
    mock_auth_service.register_user = AsyncMock(return_value=mock_user)
    
    # We also need to mock create_access_token since it's used in the route
    with patch("backend.api.routes.auth.create_access_token", return_value=DUMMY_TOKEN):
        response = client.post(
            "/api/v1/register",
            data={
                "name": "Test User",
                "email": DUMMY_EMAIL,
                "password": DUMMY_PASSWORD
            }
        )
        
    assert response.status_code == 201
    assert response.json()["access_token"] == DUMMY_TOKEN
    assert response.json()["token_type"] == "bearer"


def test_register_duplicate_email(mock_auth_service):
    # Setup mock to raise ValueError
    mock_auth_service.register_user = AsyncMock(side_effect=ValueError("Email already registered"))
    
    response = client.post(
        "/api/v1/register",
        data={
            "name": "Test User",
            "email": DUMMY_EMAIL,
            "password": DUMMY_PASSWORD
        }
    )
    
    assert response.status_code == 409
    assert response.json()["detail"] == "Email already registered"


def test_login_success(mock_auth_service):
    # Setup mock
    mock_auth_service.login_user = AsyncMock(return_value=DUMMY_TOKEN)
    
    response = client.post(
        "/api/v1/login",
        data={
            "username": DUMMY_EMAIL,
            "password": DUMMY_PASSWORD
        }
    )
    
    assert response.status_code == 200
    assert response.json()["access_token"] == DUMMY_TOKEN


def test_login_invalid_credentials(mock_auth_service):
    # Setup mock
    mock_auth_service.login_user = AsyncMock(side_effect=ValueError("Invalid email or password"))
    
    response = client.post(
        "/api/v1/login",
        data={
            "username": DUMMY_EMAIL,
            "password": "wrongpassword"
        }
    )
    
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password"


def test_forgot_password_success(mock_auth_service, mock_email_service):
    # Setup mock
    mock_auth_service.create_password_reset_token = AsyncMock(return_value=DUMMY_RESET_TOKEN)
    mock_email_service.return_value = None  # It's an async function returning None
    
    response = client.post(
        "/api/v1/forgot-password",
        json={"email": DUMMY_EMAIL}
    )
    
    assert response.status_code == 200
    assert "password reset link has been sent" in response.json()["message"]
    
    # Verify email was "sent"
    mock_email_service.assert_awaited_once_with(to_email=DUMMY_EMAIL, reset_token=DUMMY_RESET_TOKEN)


def test_forgot_password_nonexistent_email(mock_auth_service, mock_email_service):
    # Setup mock to return None (user not found)
    mock_auth_service.create_password_reset_token = AsyncMock(return_value=None)
    
    response = client.post(
        "/api/v1/forgot-password",
        json={"email": "nonexistent@example.com"}
    )
    
    assert response.status_code == 200
    # The message should be exactly the same for security reasons
    assert "password reset link has been sent" in response.json()["message"]
    
    # Verify email was NOT sent
    mock_email_service.assert_not_awaited()


def test_reset_password_success(mock_auth_service):
    # Setup mock
    mock_auth_service.reset_password = AsyncMock(return_value=None)
    
    response = client.post(
        "/api/v1/reset-password",
        json={
            "token": DUMMY_RESET_TOKEN,
            "new_password": "new_secure_password"
        }
    )
    
    assert response.status_code == 200
    assert response.json()["message"] == "Password successfully reset."


def test_reset_password_invalid_token(mock_auth_service):
    # Setup mock
    mock_auth_service.reset_password = AsyncMock(side_effect=ValueError("Invalid or expired reset token"))
    
    response = client.post(
        "/api/v1/reset-password",
        json={
            "token": "invalid_token",
            "new_password": "new_secure_password"
        }
    )
    
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid or expired reset token"
