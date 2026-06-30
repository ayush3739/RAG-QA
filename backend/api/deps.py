"""
backend/api/deps.py

FastAPI dependency functions.
Import these with Depends() in route handlers.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.security import decode_token
from backend.db.base import get_db
from backend.models.models import User
from backend.services.auth_service import AuthService
from backend.core.config import settings
from backend.core.retriever import Retriever


import base64
import time
import httpx,logging
from jose import jwt, jwk
from jose.exceptions import JWTError, ExpiredSignatureError
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select

security_scheme = HTTPBearer()

_JWKS_CACHE = {}
_JWKS_LAST_FETCH = None

def get_clerk_jwks_url() -> str:
    pub_key = settings.clerk_publishable_key
    if not pub_key:
        raise RuntimeError("CLERK_PUBLISHABLE_KEY is not configured in backend settings. Please configure it in .env")
    parts = pub_key.split('_')
    if len(parts) < 3:
        raise ValueError("Invalid Clerk Publishable Key format.")
    encoded = parts[2]
    # Strip suffix symbol $ if present in base64 string or the decoded result
    encoded_clean = encoded.rstrip('$')
    padding = len(encoded_clean) % 4
    if padding:
        encoded_clean += '=' * (4 - padding)
    decoded = base64.b64decode(encoded_clean).decode('utf-8').rstrip('$')
    return f"https://{decoded}/.well-known/jwks.json"

async def fetch_jwks(jwks_url: str):
    global _JWKS_CACHE, _JWKS_LAST_FETCH
    now = time.time()
    if not _JWKS_CACHE or not _JWKS_LAST_FETCH or (now - _JWKS_LAST_FETCH) > 86400:
        async with httpx.AsyncClient() as client:
            res = await client.get(jwks_url)
            if res.status_code == 200:
                _JWKS_CACHE = res.json()
                _JWKS_LAST_FETCH = now
            else:
                raise RuntimeError(f"Failed to fetch Clerk JWKS from {jwks_url}")
    return _JWKS_CACHE

async def verify_clerk_token(token: str) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        url = get_clerk_jwks_url()
        jwks = await fetch_jwks(url)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clerk JWKS configuration error: {e}"
        )

    try:
        headers = jwt.get_unverified_header(token)
    except JWTError:
        raise credentials_exception

    kid = headers.get("kid")
    if not kid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token header missing 'kid'")

    public_key = None
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            public_key = jwk.construct(key)
            break

    if not public_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Matching key not found in JWKS")

    try:
        decoded = jwt.decode(
            token,
            public_key.to_pem().decode('utf-8'),
            algorithms=["RS256"],
            options={"verify_aud": False, "verify_nbf": False, "leeway": 60}
        )
        return decoded
    except ExpiredSignatureError as e:
        logging.error(f"Clerk JWT token expired: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError as e:
        import traceback
        logging.error(f"Clerk JWT verification failed: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Verify Clerk JWT token and return/auto-create the matching User ORM object.
    """
    token = credentials.credentials
    decoded = await verify_clerk_token(token)
    
    clerk_id = decoded.get("sub")
    if not clerk_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token claims")

    # 1. Look up by clerk_id
    result = await db.execute(
        select(User).where(User.clerk_id == clerk_id)
    )
    user = result.scalar_one_or_none()

    if not user:
        # 2. Look up by email for existing users transitioning to Clerk
        email = decoded.get("email")
        if email:
            result = await db.execute(
                select(User).where(User.email == email)
            )
            user = result.scalar_one_or_none()
            if user:
                # Link account
                user.clerk_id = clerk_id
                await db.commit()
                await db.refresh(user)

        if not user:
            # 3. Auto-create user since they logged in via Clerk
            name = decoded.get("name", "Clerk User")
            if not email:
                email = f"{clerk_id}@clerk.local"
            
            user = User(
                name=name,
                email=email,
                clerk_id=clerk_id,
                password_hash=None
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)

    return user

async def get_settings():
    return settings

__all__ = ["get_db", "get_current_user", "security_scheme"]