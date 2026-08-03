from backend.core.config import settings
from backend.models.auth_schemas import OAuthProfile
from fastapi import HTTPException, status
from jose import JWTError, jwt
from urllib.parse import urlencode
from datetime import datetime, timedelta, timezone


def build_github_authorization_url(state: str) -> str:
    query = urlencode(
        {
            "client_id": settings.GITHUB_CLIENT_ID.get_secret_value(),
            "redirect_uri": settings.GITHUB_REDIRECT_URI,
            "scope": "read:user user:email",
            "state": state,
        }
    )
    return f"https://github.com/login/oauth/authorize?{query}"


def create_github_oauth_state() -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "iss": settings.issuer,
        "iat": now,
        "exp": now + timedelta(minutes=10),
        "type": "oauth_state",
        "provider": "github",
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def validate_github_oauth_state(state: str) -> None:
    try:
        payload = jwt.decode(state, settings.secret_key, algorithms=[settings.algorithm])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid OAuth state",
        ) from exc

    if payload.get("iss") != settings.issuer:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid OAuth state",
        )

    if payload.get("type") != "oauth_state" or payload.get("provider") != "github":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid OAuth state",
        )

async def exchange_github_code(code: str) -> str:
    """
    Exchange the authorization code for an access token.
    """
    import httpx

    url = "https://github.com/login/oauth/access_token"
    headers = {"Accept": "application/json"}
    data = {
        "client_id": settings.GITHUB_CLIENT_ID.get_secret_value(),
        "client_secret": settings.GITHUB_CLIENT_SECRET.get_secret_value(),
        "code": code,
        "redirect_uri": settings.GITHUB_REDIRECT_URI,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, data=data)
        response.raise_for_status()
        payload = response.json()
        access_token = payload.get("access_token")
        if not access_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=payload.get("error_description") or "Unable to exchange GitHub OAuth code",
            )
        return access_token


async def _fetch_github_primary_email(client, headers: dict) -> str | None:
    response = await client.get("https://api.github.com/user/emails", headers=headers)
    response.raise_for_status()
    emails = response.json()
    if not isinstance(emails, list):
        return None

    primary_verified = next(
        (
            entry.get("email")
            for entry in emails
            if entry.get("primary") and entry.get("verified") and entry.get("email")
        ),
        None,
    )
    if primary_verified:
        return primary_verified

    return next((entry.get("email") for entry in emails if entry.get("email")), None)

async def fetch_github_user_info(access_token: str) -> OAuthProfile:
    """
    Fetch the user's GitHub profile information using the access token.
    """
    import httpx

    url = "https://api.github.com/user"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github+json",
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        oauth_profile = response.json()
        email = oauth_profile.get("email")
        if not email:
            email = await _fetch_github_primary_email(client, headers)
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="GitHub account does not expose an email address",
            )

        display_name = oauth_profile.get("name") or oauth_profile.get("login") or email.split("@")[0]
        return OAuthProfile(
            provider_id=str(oauth_profile["id"]),
            email=email,
            name=display_name,
            avatar_url=oauth_profile.get("avatar_url"),
        )
