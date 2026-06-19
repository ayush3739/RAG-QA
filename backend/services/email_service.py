"""
backend/services/email_service.py

Handles sending emails via SMTP.
"""

import aiosmtplib
from email.message import EmailMessage

from backend.core.config import settings

async def send_reset_email(to_email: str, reset_token: str):
    """
    Sends a password reset email to the user with the provided token.
    Uses SMTP settings defined in the config.
    """
    message = EmailMessage()
    message["From"] = settings.mail_from
    message["To"] = to_email
    message["Subject"] = "Password Reset Request - DocuMind"
    
    # In a real app, this would be a link to a frontend reset page:
    # http://localhost:3000/reset-password?token={reset_token}
    # For now, we'll just send the raw token.
    
    content = f"""
Hello,

You have requested to reset your password.
Please use the following token to reset your password:

{reset_token}

If you did not request this, please ignore this email.
"""
    message.set_content(content)
    
    try:
        if settings.mail_use_tls:
            await aiosmtplib.send(
                message,
                hostname=settings.mail_server,
                port=settings.mail_port,
                username=settings.mail_username,
                password=settings.mail_password.get_secret_value(),
                start_tls=True,
            )
        else:
            await aiosmtplib.send(
                message,
                hostname=settings.mail_server,
                port=settings.mail_port,
                username=settings.mail_username,
                password=settings.mail_password.get_secret_value(),
            )
    except Exception as e:
        print(f"Failed to send email to {to_email}: {e}")
        # In a production environment, you might want to raise this or log it properly
        pass
