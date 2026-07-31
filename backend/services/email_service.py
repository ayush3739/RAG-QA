from email.message import EmailMessage
from pathlib import Path

import aiosmtplib
from fastapi.templating import Jinja2Templates

from backend.core.config import settings


TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _frontend_base_url() -> str:
    frontend_url = getattr(settings, "frontend_url", None)
    if frontend_url:
        return frontend_url.rstrip("/")
    return "http://localhost:3000"


def _render_template(template_name: str, **context: str) -> str:
    template = templates.env.get_template(template_name)
    return template.render(**context)


async def send_email(
    to_email: str,
    subject: str,
    plain_text: str,
    html_content: str | None = None,
) -> None:
    message = EmailMessage()
    message["From"] = settings.mail_from
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(plain_text)

    if html_content:
        message.add_alternative(html_content, subtype="html")

    await aiosmtplib.send(
        message,
        hostname=settings.mail_server,
        port=settings.mail_port,
        username=settings.mail_username or None,
        password=settings.mail_password.get_secret_value() or None,
        start_tls=settings.mail_use_tls,
    )


async def send_reset_email(to_email: str, reset_token: str, username: str = "there") -> None:
    reset_url = f"{_frontend_base_url()}/reset-password?token={reset_token}"
    html_content = _render_template(
        "password_reset.html",
        username=username,
        action_url=reset_url,
        action_text="Reset password",
        token=reset_token,
        expiry_text="This link will expire in 15 minutes.",
    )
    plain_text = (
        f"Hello {username},\n\n"
        "We received a request to reset your password.\n\n"
        f"Reset your password: {reset_url}\n\n"
        f"Reset token: {reset_token}\n\n"
        "This link will expire in 15 minutes.\n"
        "If you did not request this, you can ignore this email.\n"
    )

    await send_email(
        to_email=to_email,
        subject="Reset your password",
        plain_text=plain_text,
        html_content=html_content,
    )


async def send_verification_email(
    to_email: str,
    verification_token: str,
    username: str = "there",
) -> None:
    verify_url = f"{_frontend_base_url()}/verify-email?token={verification_token}"
    html_content = _render_template(
        "email_verify.html",
        username=username,
        action_url=verify_url,
        action_text="Verify email",
        token=verification_token,
        expiry_text="Use this link to confirm your email address.",
    )
    plain_text = (
        f"Hello {username},\n\n"
        "Please verify your email address to continue.\n\n"
        f"Verify your email: {verify_url}\n\n"
        f"Verification token: {verification_token}\n\n"
        "If you did not create this account, you can ignore this email.\n"
    )

    await send_email(
        to_email=to_email,
        subject="Verify your email",
        plain_text=plain_text,
        html_content=html_content,
    )
