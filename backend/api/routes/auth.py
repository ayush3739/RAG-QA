"""
backend/api/routes/auth.py

Authentication endpoints.

POST /auth/register   → create account, return token
POST /auth/login      → verify credentials, return token
GET  /auth/me         → return current user profile
"""

from datetime import datetime, timedelta, timezone
from sqlalchemy import delete, select
from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, status
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated, Optional
from urllib.parse import quote_plus
from backend.api.deps import get_current_user, get_db
from backend.core.config import settings
from backend.models.auth_schemas import (
    MessageResponse,
    TokenResponse,
    UserRegister,
    UserResponse,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    RefreshTokenRequest,
    VerifyEmailRequest,
)
from backend.models.models import RefreshToken, User, UserToken, UserTokenType
from backend.services.oauth_service import (
    build_github_authorization_url,
    create_github_oauth_state,
    exchange_github_code,
    fetch_github_user_info,
    build_google_authorization_url,
    create_google_oauth_state,
    exchange_google_code,
    fetch_google_user_info,
    login_or_create_oauth_user,
    validate_github_oauth_state,
    validate_google_oauth_state,
)

from backend.services.auth_service import (
    persist_refresh_token,
    revoke_refresh_token,
    revoke_all_refresh_tokens,
    rotate_refresh_token,
    verify_email as verify_email_token,
)
from backend.services.security import (
    hash_password, 
    verify_password, 
    create_access_token,
    generate_reset_token,
    create_refresh_token,
    hash_reset_token,

)
from backend.services.email_service import send_reset_email,send_verification_email


router = APIRouter()


def set_refresh_cookie(response: Response, refresh_token: str) -> None:
    """Attach secure HttpOnly refresh token cookie to HTTP response."""
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        max_age=60 * 60 * 24 * settings.refresh_token_expire_days,
        domain=settings.cookie_domain,
        path="/api/v1/auth",
    )


def clear_refresh_cookie(response: Response) -> None:
    """Delete HttpOnly refresh token cookie on HTTP response."""
    response.delete_cookie(
        key="refresh_token",
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        domain=settings.cookie_domain,
        path="/api/v1/auth",
    )


@router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return current_user



@router.post(
    "/register",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(user: UserRegister,db: Annotated[AsyncSession, Depends(get_db)],):
    # check if user already exists
    result = await db.execute(select(User).where(User.email == user.email))
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    try:
        password_hash = hash_password(user.password)
        new_user = User(email=user.email, password_hash=password_hash, name=user.name)
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
    except Exception:
        await db.rollback()
        raise

    try:
        verification_token = generate_reset_token()
        hashed_token = hash_reset_token(verification_token)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=15)  # Token valid for 15 minutes
        verification_entry = UserToken(
            user_id=new_user.id,
            token_hash=hashed_token,
            expires_at=expires_at,
            token_type=UserTokenType.EMAIL_VERIFICATION,
        )
        db.add(verification_entry)
        await db.commit()
        await send_verification_email(
            to_email=new_user.email,
            verification_token=verification_token,
            username=new_user.name,
        )
    except Exception:
        await db.rollback()
        await db.execute(delete(UserToken).where(UserToken.token_hash == hashed_token))
        await db.execute(delete(User).where(User.id == new_user.id))
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to send verification email",
        )
    




    return {"message": "User registered successfully. Please verify your email before logging in."} 
    


@router.post("/login", response_model=TokenResponse)
async def login(
    response: Response,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,   
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified. Please check your inbox.",
        )
    if not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
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

    set_refresh_cookie(response, refresh_token)
    return TokenResponse(access_token=access_token, token_type="bearer")



@router.post("/logout")
async def logout(
    response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
    refresh_token_cookie: Optional[str] = Cookie(None, alias="refresh_token"),
    refresh_request: Optional[RefreshTokenRequest] = None,
):
    token_to_use = refresh_request.refresh_token if (refresh_request and refresh_request.refresh_token) else refresh_token_cookie
    if token_to_use:
        await revoke_refresh_token(db, token_to_use)

    clear_refresh_cookie(response)
    return {"message": "Successfully logged out."}


@router.post("/logout-all")
async def logout_all(
    response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
    refresh_token_cookie: Optional[str] = Cookie(None, alias="refresh_token"),
    refresh_request: Optional[RefreshTokenRequest] = None,
):
    token_to_use = refresh_request.refresh_token if (refresh_request and refresh_request.refresh_token) else refresh_token_cookie
    if token_to_use:
        await revoke_all_refresh_tokens(db, token_to_use)

    clear_refresh_cookie(response)
    return {"message": "Successfully logged out of all sessions."}


@router.post("/forgot-password")
async def forgot_password(
    email: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.email == email.email))
    user = result.scalar_one_or_none()
    if not user:
        return {"message": "If an account with that email exists, a password reset link has been sent."}

    reset_token = generate_reset_token()
    hashed_token = hash_reset_token(reset_token)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=15)  # Token valid for 15 minutes
    reset_entry = UserToken(
        user_id=user.id,
        token_hash=hashed_token,
        expires_at=expires_at,
        token_type=UserTokenType.PASSWORD_RESET,
    )
    try:
        db.add(reset_entry)
        await db.commit()
        await send_reset_email(
            to_email=email.email,
            reset_token=reset_token,
            username=user.name,
        )
    except Exception:
        await db.rollback()
        await db.execute(delete(UserToken).where(UserToken.token_hash == hashed_token))
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to send password reset email",
        )

    
    return {"message": "If an account with that email exists, a password reset link has been sent."}

    


@router.post("/reset-password")
async def reset_password(
    reset_request: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    hashed_token = hash_reset_token(reset_request.token)
    result = await db.execute(select(UserToken).where(UserToken.token_hash == hashed_token, UserToken.token_type == UserTokenType.PASSWORD_RESET))
    reset_entry = result.scalar_one_or_none()
    if not reset_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid or expired reset token",
        )
    if reset_entry.expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token has expired",
        )
    user_result = await db.execute(select(User).where(User.id == reset_entry.user_id))
    user = user_result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    try:
        user.password_hash = hash_password(reset_request.new_password)
        db.add(user)
        await db.execute(delete(UserToken).where(UserToken.user_id == user.id, UserToken.token_type == UserTokenType.PASSWORD_RESET))
        await db.execute(delete(RefreshToken).where(RefreshToken.user_id == user.id))
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    return {"message": "Password successfully reset. Please log in with your new password."}


@router.post("/refresh-token", response_model=TokenResponse)
async def refresh_token(
    response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
    refresh_token_cookie: Optional[str] = Cookie(None, alias="refresh_token"),
    refresh_request: Optional[RefreshTokenRequest] = None,
):
    token_to_use = refresh_request.refresh_token if (refresh_request and refresh_request.refresh_token) else refresh_token_cookie
    if not token_to_use:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing",
        )

    access_token, new_refresh_token = await rotate_refresh_token(db, token_to_use)

    set_refresh_cookie(response, new_refresh_token)

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
    )


@router.post("/verify-email", response_model=MessageResponse)
async def verify_email(
    verification_request: VerifyEmailRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    await verify_email_token(db, verification_request.token)
    return {"message": "Email successfully verified. You can now log in."}


@router.get("/github/login")
async def github_login():
    state = create_github_oauth_state()
    return RedirectResponse(
        url=build_github_authorization_url(state),
        status_code=status.HTTP_302_FOUND,
    )


@router.get("/github/callback")
async def github_callback(
    db: Annotated[AsyncSession, Depends(get_db)],
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
):
    if error or not code or not state:
        msg = quote_plus(error_description or "GitHub authentication was cancelled or failed.")
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/#/auth?oauth_error=github_cancelled&message={msg}",
            status_code=status.HTTP_302_FOUND,
        )

    try:
        validate_github_oauth_state(state)
        github_access_token = await exchange_github_code(code)
        profile = await fetch_github_user_info(github_access_token)
        token_data = await login_or_create_oauth_user(db, profile, "github")
        redirect_res = RedirectResponse(
            url=f"{settings.FRONTEND_URL}/#/oauth/callback?access_token={token_data.access_token}",
            status_code=status.HTTP_302_FOUND,
        )
        set_refresh_cookie(redirect_res, token_data.refresh_token)
        return redirect_res
    except HTTPException as exc:
        msg = quote_plus(str(exc.detail))
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/#/auth?oauth_error=github_failed&message={msg}",
            status_code=status.HTTP_302_FOUND,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        msg = quote_plus(f"An unexpected error occurred during GitHub login: {str(e)}")
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/#/auth?oauth_error=github_failed&message={msg}",
            status_code=status.HTTP_302_FOUND,
        )


@router.get("/google/login")
async def google_login():
    state = create_google_oauth_state()
    return RedirectResponse(
        url=build_google_authorization_url(state),
        status_code=status.HTTP_302_FOUND,
    )


@router.get("/google/callback")
async def google_callback(
    db: Annotated[AsyncSession, Depends(get_db)],
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
):
    if error or not code or not state:
        msg = quote_plus(error_description or "Google authentication was cancelled or failed.")
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/#/auth?oauth_error=google_cancelled&message={msg}",
            status_code=status.HTTP_302_FOUND,
        )

    try:
        validate_google_oauth_state(state)
        google_access_token = await exchange_google_code(code)
        profile = await fetch_google_user_info(google_access_token)
        token_data = await login_or_create_oauth_user(db, profile, "google")
        redirect_res = RedirectResponse(
            url=f"{settings.FRONTEND_URL}/#/oauth/callback?access_token={token_data.access_token}",
            status_code=status.HTTP_302_FOUND,
        )
        set_refresh_cookie(redirect_res, token_data.refresh_token)
        return redirect_res
    except HTTPException as exc:
        msg = quote_plus(str(exc.detail))
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/#/auth?oauth_error=google_failed&message={msg}",
            status_code=status.HTTP_302_FOUND,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Fallback for unexpected errors
        msg = quote_plus(f"An unexpected error occurred during Google login: {str(e)}")
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/#/auth?oauth_error=google_failed&message={msg}",
            status_code=status.HTTP_302_FOUND,
        )