"""
backend/models/auth_schemas.py

Pydantic schemas for authentication and user endpoints.
These are request/response shapes only — ORM models live in models/models.py.
"""




from typing import Optional
from pydantic import BaseModel, EmailStr, Field


# ---------------------------------------------------------------------------
# Register / Login
# ---------------------------------------------------------------------------

class UserBase(BaseModel):
    email: EmailStr
    password: str

class UserRegister(UserBase):
    name: str = Field(..., min_length=1, max_length=50)


class UserLogin(UserBase):
    pass 


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    name: str
    email: str

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Password reset  (wired up later)
# ---------------------------------------------------------------------------

class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str
    
class VerifyEmailRequest(BaseModel):
    token: str

class RefreshTokenRequest(BaseModel):
    refresh_token: Optional[str] = None




class MessageResponse(BaseModel):
    message: str

class OAuthProfile(BaseModel):
    provider_id: str
    email: str
    name: str
    avatar_url: str | None