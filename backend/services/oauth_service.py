from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional
from urllib.parse import urlencode
from fastapi import Depends, HTTPException, status
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import settings
from backend.db.base import get_db
from backend.models.auth_schemas import OAuthProfile, TokenResponse
from backend.models.models import OAuthAccount, OAuthProvider, User
from backend.services.auth_service import persist_refresh_token
from backend.services.security import create_access_token, create_refresh_token


# GITHUB OAUTH HELPERS

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
    Exchange the GitHub authorization code for an access token.
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


# GOOGLE OAUTH HELPERS

def build_google_authorization_url(state: str) -> str:
    query = urlencode(
        {
            "client_id": settings.GOOGLE_CLIENT_ID.get_secret_value(),
            "redirect_uri": settings.GOOGLE_REDIRECT_URI,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        }
    )
    return f"https://accounts.google.com/o/oauth2/v2/auth?{query}"


def create_google_oauth_state() -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "iss": settings.issuer,
        "iat": now,
        "exp": now + timedelta(minutes=10),
        "type": "oauth_state",
        "provider": "google",
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def validate_google_oauth_state(state: str) -> None:
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

    if payload.get("type") != "oauth_state" or payload.get("provider") != "google":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid OAuth state",
        )


async def exchange_google_code(code: str) -> str:
    """
    Exchange the Google authorization code for an access token.
    """
    import httpx

    url = "https://oauth2.googleapis.com/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "client_id": settings.GOOGLE_CLIENT_ID.get_secret_value(),
        "client_secret": settings.GOOGLE_CLIENT_SECRET.get_secret_value(),
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, data=data)
        response.raise_for_status()
        payload = response.json()
        access_token = payload.get("access_token")
        if not access_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=payload.get("error_description") or "Unable to exchange Google OAuth code",
            )
        return access_token


async def fetch_google_user_info(access_token: str) -> OAuthProfile:
    """
    Fetch the user's Google profile information using the access token.
    """
    import httpx

    url = "https://www.googleapis.com/oauth2/v2/userinfo"
    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        profile_data = response.json()
        email = profile_data.get("email")
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Google account does not expose an email address",
            )

        display_name = profile_data.get("name") or email.split("@")[0]
        return OAuthProfile(
            provider_id=str(profile_data["id"]),
            email=email,
            name=display_name,
            avatar_url=profile_data.get("picture"),
        )


# REUSABLE OAUTH USER LOGIN / CREATION SERVICE

async def login_or_create_oauth_user(
    db: AsyncSession,
    profile: OAuthProfile,
    provider: str,
) -> TokenResponse:
    """
    Log in the user if they exist, or create a new user based on the OAuth profile.
    Supports GitHub, Google, and any configured OAuth provider.
    """
    # Map string provider to OAuthProvider enum
    try:
        provider_enum = OAuthProvider(provider.lower())
    except ValueError:
        p_lower = provider.lower()
        if p_lower in ("github", "git"):
            provider_enum = OAuthProvider.GITHUB
        elif p_lower in ("google", "goog"):
            provider_enum = OAuthProvider.GOOGLE
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported OAuth provider: {provider}",
            )

    # 1. Search for existing linked OAuthAccount
    oauth_result = await db.execute(
        select(OAuthAccount).where(
            OAuthAccount.provider == provider_enum,
            OAuthAccount.provider_account_id == profile.provider_id,
        )
    )
    oauth_account = oauth_result.scalar_one_or_none()

    if oauth_account:
        user_result = await db.execute(select(User).where(User.id == oauth_account.user_id))
        user = user_result.scalar_one_or_none()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Linked OAuth user not found",
            )
    else:
        # 2. Search for existing user by email
        user_result = await db.execute(select(User).where(User.email == profile.email))
        user = user_result.scalar_one_or_none()

        try:
            if not user:
                user = User(
                    email=profile.email,
                    name=profile.name,
                    is_verified=True,
                )
                db.add(user)
                await db.flush()
            else:
                user.is_verified = True
                if not user.name:
                    user.name = profile.name
                db.add(user)

            oauth_account = OAuthAccount(
                user_id=user.id,
                provider=provider_enum,
                provider_account_id=profile.provider_id,
                provider_email=profile.email,
                provider_username=profile.name,
            )
            db.add(oauth_account)
            await db.commit()
            await db.refresh(user)
        except Exception:
            await db.rollback()
            raise

    access_token = create_access_token(user.id, user.role.value)
    refresh_token = create_refresh_token(user.id)

    # Persist this login's refresh token as a separate session entry.
    try:
        await persist_refresh_token(db, user.id, refresh_token)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error storing refresh token",
        )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
    )