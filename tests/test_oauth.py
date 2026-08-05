import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException, status
from fastapi.testclient import TestClient
from jose import jwt

from backend.main import app
from backend.core.config import settings
from backend.api.deps import get_db
from backend.models.auth_schemas import OAuthProfile, TokenResponse
from backend.models.models import OAuthAccount, OAuthProvider, User

from backend.services.oauth_service import (
    build_github_authorization_url,
    build_google_authorization_url,
    create_github_oauth_state,
    create_google_oauth_state,
    exchange_github_code,
    exchange_google_code,
    fetch_github_user_info,
    fetch_google_user_info,
    login_or_create_oauth_user,
    validate_github_oauth_state,
    validate_google_oauth_state,
)

client = TestClient(app)

@pytest.fixture(autouse=True)
def override_db_dependency():
    mock_db = AsyncMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    yield mock_db
    app.dependency_overrides.clear()

@pytest.fixture
def anyio_backend():
    return 'asyncio'

# ─────────────────────────────────────────────────────────────────────────────
# 1. STATE & AUTHORIZATION URL TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_github_authorization_url():
    state = "test_github_state_123"
    url = build_github_authorization_url(state)
    assert "https://github.com/login/oauth/authorize" in url
    assert f"state={state}" in url


def test_github_state_create_and_validate():
    state = create_github_oauth_state()
    assert isinstance(state, str)
    validate_github_oauth_state(state)


def test_github_state_invalid_signature():
    invalid_state = "invalid.jwt.token"
    with pytest.raises(HTTPException) as exc:
        validate_github_oauth_state(invalid_state)
    assert exc.value.status_code == status.HTTP_400_BAD_REQUEST


def test_google_authorization_url():
    state = "test_google_state_456"
    url = build_google_authorization_url(state)
    assert "https://accounts.google.com/o/oauth2/v2/auth" in url
    assert "response_type=code" in url
    assert f"state={state}" in url


def test_google_state_create_and_validate():
    state = create_google_oauth_state()
    assert isinstance(state, str)
    validate_google_oauth_state(state)


def test_google_state_invalid_provider():
    payload = {
        "iss": settings.issuer,
        "type": "oauth_state",
        "provider": "github",
    }
    state = jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)
    with pytest.raises(HTTPException) as exc:
        validate_google_oauth_state(state)
    assert exc.value.status_code == status.HTTP_400_BAD_REQUEST


# ─────────────────────────────────────────────────────────────────────────────
# 2. HTTP EXCHANGE & PROFILE FETCHING TESTS (MOCKED HTTPX)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_exchange_github_code_success():
    mock_response = MagicMock()
    mock_response.json.return_value = {"access_token": "gho_dummy_token_123"}
    mock_response.raise_for_status.return_value = None

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        token = await exchange_github_code("dummy_code")
        assert token == "gho_dummy_token_123"


@pytest.mark.anyio
async def test_fetch_github_user_info_success():
    mock_user_res = MagicMock()
    mock_user_res.json.return_value = {
        "id": 98765,
        "email": "octocat@github.com",
        "name": "Mona Lisa Octocat",
        "avatar_url": "https://github.com/images/error/octocat_happy.gif",
    }
    mock_user_res.raise_for_status.return_value = None

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_user_res
        profile = await fetch_github_user_info("gho_dummy_token_123")
        assert profile.provider_id == "98765"
        assert profile.email == "octocat@github.com"
        assert profile.name == "Mona Lisa Octocat"


@pytest.mark.anyio
async def test_exchange_google_code_success():
    mock_response = MagicMock()
    mock_response.json.return_value = {"access_token": "ya29.dummy_google_token"}
    mock_response.raise_for_status.return_value = None

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        token = await exchange_google_code("dummy_google_code")
        assert token == "ya29.dummy_google_token"


@pytest.mark.anyio
async def test_fetch_google_user_info_success():
    mock_res = MagicMock()
    mock_res.json.return_value = {
        "id": "1122334455",
        "email": "user@gmail.com",
        "name": "Google User",
        "picture": "https://lh3.googleusercontent.com/a/dummy",
    }
    mock_res.raise_for_status.return_value = None

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_res
        profile = await fetch_google_user_info("ya29.dummy_google_token")
        assert profile.provider_id == "1122334455"
        assert profile.email == "user@gmail.com"
        assert profile.name == "Google User"


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOGIN OR CREATE OAUTH USER SERVICE TESTS
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_login_or_create_oauth_user_existing_account():
    mock_db = AsyncMock()

    mock_oauth_acc = OAuthAccount(id=10, user_id=1, provider=OAuthProvider.GITHUB, provider_account_id="98765")
    mock_user = User(id=1, email="octocat@github.com", name="Mona", role=MagicMock(value="user"))

    mock_exec_1 = MagicMock()
    mock_exec_1.scalar_one_or_none.return_value = mock_oauth_acc

    mock_exec_2 = MagicMock()
    mock_exec_2.scalar_one_or_none.return_value = mock_user

    mock_db.execute.side_effect = [mock_exec_1, mock_exec_2]

    profile = OAuthProfile(
        provider_id="98765",
        email="octocat@github.com",
        name="Mona",
        avatar_url=None,
    )

    with patch("backend.services.oauth_service.create_access_token", return_value="acc_token"), \
         patch("backend.services.oauth_service.create_refresh_token", return_value="ref_token"), \
         patch("backend.services.oauth_service.persist_refresh_token", new_callable=AsyncMock):

        response = await login_or_create_oauth_user(mock_db, profile, "github")
        assert isinstance(response, TokenResponse)
        assert response.access_token == "acc_token"
        assert response.refresh_token == "ref_token"


@pytest.mark.anyio
async def test_login_or_create_oauth_user_unsupported_provider():
    mock_db = AsyncMock()
    profile = OAuthProfile(provider_id="1", email="test@test.com", name="Test", avatar_url=None)

    with pytest.raises(HTTPException) as exc:
        await login_or_create_oauth_user(mock_db, profile, "twitter")
    assert exc.value.status_code == status.HTTP_400_BAD_REQUEST


# ─────────────────────────────────────────────────────────────────────────────
# 4. FASTAPI AUTH ROUTE ENDPOINT TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_github_login_endpoint():
    response = client.get("/api/v1/auth/github/login", follow_redirects=False)
    assert response.status_code == status.HTTP_302_FOUND
    assert "https://github.com/login/oauth/authorize" in response.headers["location"]


def test_google_login_endpoint():
    response = client.get("/api/v1/auth/google/login", follow_redirects=False)
    assert response.status_code == status.HTTP_302_FOUND
    assert "https://accounts.google.com/o/oauth2/v2/auth" in response.headers["location"]


def test_github_callback_endpoint_redirect():
    dummy_profile = OAuthProfile(provider_id="123", email="user@github.com", name="GitHub User", avatar_url=None)
    dummy_tokens = TokenResponse(access_token="access_123", refresh_token="refresh_123", token_type="bearer")

    with patch("backend.api.routes.auth.validate_github_oauth_state") as mock_val, \
         patch("backend.api.routes.auth.exchange_github_code", new_callable=AsyncMock, return_value="gh_code"), \
         patch("backend.api.routes.auth.fetch_github_user_info", new_callable=AsyncMock, return_value=dummy_profile), \
         patch("backend.api.routes.auth.login_or_create_oauth_user", new_callable=AsyncMock, return_value=dummy_tokens):

        response = client.get("/api/v1/auth/github/callback?code=testcode&state=teststate", follow_redirects=False)
        assert response.status_code == status.HTTP_302_FOUND
        redirect_url = response.headers["location"]
        assert "http://localhost:3000/#/oauth/callback" in redirect_url
        assert "access_token=access_123" in redirect_url
        assert "refresh_token=refresh_123" in redirect_url


def test_google_callback_endpoint_redirect():
    dummy_profile = OAuthProfile(provider_id="456", email="user@gmail.com", name="Google User", avatar_url=None)
    dummy_tokens = TokenResponse(access_token="google_acc", refresh_token="google_ref", token_type="bearer")

    with patch("backend.api.routes.auth.validate_google_oauth_state") as mock_val, \
         patch("backend.api.routes.auth.exchange_google_code", new_callable=AsyncMock, return_value="goog_code"), \
         patch("backend.api.routes.auth.fetch_google_user_info", new_callable=AsyncMock, return_value=dummy_profile), \
         patch("backend.api.routes.auth.login_or_create_oauth_user", new_callable=AsyncMock, return_value=dummy_tokens):

        response = client.get("/api/v1/auth/google/callback?code=testcode&state=teststate", follow_redirects=False)
        assert response.status_code == status.HTTP_302_FOUND
        redirect_url = response.headers["location"]
        assert "http://localhost:3000/#/oauth/callback" in redirect_url
        assert "access_token=google_acc" in redirect_url
        assert "refresh_token=google_ref" in redirect_url
