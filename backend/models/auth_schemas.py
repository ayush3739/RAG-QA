"""
backend/models/auth_schemas.py

Pydantic schemas for authentication and user endpoints.
These are request/response shapes only — ORM models live in models/models.py.
"""

from pydantic import BaseModel, EmailStr


# ---------------------------------------------------------------------------
# Register / Login
# ---------------------------------------------------------------------------

class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class TokenResponse(BaseModel):
    access_token: str
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