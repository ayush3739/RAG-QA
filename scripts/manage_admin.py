#!/usr/bin/env python3
"""
scripts/manage_admin.py

Command-line tool to manage administrative privileges in KnowVex.
Allows promoting an existing user to Admin, creating a new Admin user,
or listing registered users directly in PostgreSQL.

Usage examples:
    # Interactive mode:
    python scripts/manage_admin.py

    # Promote an existing user:
    python scripts/manage_admin.py --email ayushmaurya21086@gmail.com

    # Create or update with password:
    python scripts/manage_admin.py --email admin@knowvex.ai --password "SuperSecretPass123"

    # List all users and roles:
    python scripts/manage_admin.py --list

    # Target a remote deployed database:
    python scripts/manage_admin.py --db-url "postgresql://user:pass@remote-host:5432/db" --email admin@company.com
"""

import sys
import argparse
import asyncio
import getpass
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from backend.core.config import settings
from backend.models.models import User, UserRole
from backend.services.security import hash_password


def normalize_database_url(url: str) -> str:
    """Ensure the URL uses the asyncpg driver."""
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql://") and not url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


async def list_users(session: AsyncSession) -> None:
    """Print all registered users, their roles, and verification status."""
    stmt = select(User).order_by(User.id.asc())
    result = await session.execute(stmt)
    users = result.scalars().all()

    print("\n" + "=" * 70)
    print(f"{'ID':<6} | {'EMAIL':<32} | {'ROLE':<10} | {'VERIFIED':<8} | {'NAME'}")
    print("-" * 70)
    if not users:
        print("  No users found in database.")
    for u in users:
        role_str = u.role.value if hasattr(u.role, "value") else str(u.role)
        ver_str = "Yes" if u.is_verified else "No"
        print(f"{u.id:<6} | {u.email:<32} | {role_str:<10} | {ver_str:<8} | {u.name}")
    print("=" * 70 + "\n")


async def promote_or_create_admin(
    session: AsyncSession,
    email: str,
    password: str | None = None,
    name: str | None = None,
    role_to_set: str = "admin",
) -> None:
    """Promote an existing user or create a new user with the specified role."""
    clean_email = email.strip().lower()

    stmt = select(User).where(User.email == clean_email)
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()

    target_role = UserRole.ADMIN if role_to_set.lower() == "admin" else (
        UserRole.PRO if role_to_set.lower() == "pro" else UserRole.USER
    )

    if user:
        previous_role = user.role.value if hasattr(user.role, "value") else str(user.role)
        user.role = target_role
        user.is_verified = True  # Ensure admin is marked verified
        if name:
            user.name = name.strip()
        if password:
            user.password_hash = hash_password(password)

        await session.commit()
        await session.refresh(user)

        print("\n" + "─" * 60)
        print(f"✅ User Updated Successfully!")
        print(f"   • Email:       {user.email}")
        print(f"   • Name:        {user.name}")
        print(f"   • Previous:    {previous_role}")
        print(f"   • New Role:    {target_role.value}")
        print(f"   • Verified:    {user.is_verified}")
        if password:
            print("   • Password:    Updated")
        print("─" * 60 + "\n")
    else:
        # User does not exist, create new
        if not password:
            password = getpass.getpass(f"Enter password for new admin ({clean_email}): ")
            if not password:
                print("❌ Error: Password cannot be empty when creating a new user.")
                sys.exit(1)

        display_name = name or clean_email.split("@")[0].capitalize()
        new_user = User(
            email=clean_email,
            name=display_name,
            password_hash=hash_password(password),
            role=target_role,
            is_verified=True,
        )
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)

        print("\n" + "─" * 60)
        print(f"🎉 New Admin Account Created Successfully!")
        print(f"   • User ID:     {new_user.id}")
        print(f"   • Email:       {new_user.email}")
        print(f"   • Name:        {new_user.name}")
        print(f"   • Role:        {new_user.role.value}")
        print(f"   • Verified:    {new_user.is_verified}")
        print("─" * 60 + "\n")


async def main_async(args: argparse.Namespace) -> None:
    db_url = normalize_database_url(args.db_url or settings.DATABASE_URL)
    engine = create_async_engine(db_url, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with async_session() as session:
            if args.list:
                await list_users(session)
                return

            email = args.email
            if not email:
                print("\n🔐 KnowVex Admin Management CLI")
                email = input("Enter user email to promote or create: ").strip()
                if not email:
                    print("❌ Error: Email is required.")
                    sys.exit(1)

            password = args.password
            await promote_or_create_admin(
                session=session,
                email=email,
                password=password,
                name=args.name,
                role_to_set=args.role,
            )
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KnowVex Admin Account & Privilege Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--email",
        "-e",
        help="Email address of the user to promote or create.",
    )
    parser.add_argument(
        "--password",
        "-p",
        help="Password for the user (optional if promoting an existing user).",
    )
    parser.add_argument(
        "--name",
        "-n",
        help="Display name for the user.",
    )
    parser.add_argument(
        "--role",
        "-r",
        default="admin",
        choices=["admin", "Pro", "user"],
        help="Target role to assign (default: admin).",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all users currently registered in the database.",
    )
    parser.add_argument(
        "--db-url",
        help="Custom database connection URL (defaults to DATABASE_URL from .env).",
    )

    args = parser.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n❌ Database error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
