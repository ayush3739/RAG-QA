"""
backend/services/security.py

Password hashing and JWT token utilities.
No business logic here — pure crypto primitives used by auth_service.
"""

from uuid import uuid4
from datetime import datetime, timedelta, timezone
import secrets
import hashlib

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import Enum

from backend.core.config import settings

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class TokenType(str,Enum):
    ACCESS = "access"
    REFRESH = "refresh"
    


def hash_password(plain: str) -> str:
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


#JWT token utilities

def create_access_token(user_id: str, role: str, expires_delta:timedelta | None = None) -> str:

    
    now = datetime.now(timezone.utc)
    
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.access_token_expire_minutes)
     
    to_encode = {
        "sub": user_id,
        "role": role,
        "iss": settings.issuer,
        "exp": expire,
        "iat": now,
        "type": TokenType.ACCESS.value,
    }

    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    
    return encoded_jwt

def create_refresh_token(user_id: str, expires_delta:timedelta | None = None) -> str:

    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(days=settings.refresh_token_expire_days)

    to_encode = {
        "sub": user_id,
        "iss": settings.issuer,
        "iat": now,
        "exp": expire,
        "type": TokenType.REFRESH.value,
        "jti": str(uuid4()),
    }

    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    
    return encoded_jwt

def decode_token(token: str,expected_type: str) -> dict | None:
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
        )
        if payload is None:
            return None
        if payload.get("iss") != settings.issuer:
            return None
        

        if payload.get("type") != expected_type:
            return None

        return payload

    except JWTError:
        return None
    


# ---------------------------------------------------------------------------
# Password Reset Tokens
# ---------------------------------------------------------------------------

def generate_reset_token() -> str:
    """Generate a secure, random token for password reset."""
    return secrets.token_urlsafe(32)

def hash_reset_token(token: str) -> str:
    """Hash the token using SHA-256 for secure database storage."""
    return hashlib.sha256(token.encode()).hexdigest()