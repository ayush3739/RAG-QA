"""add users role and data usage

Revision ID: b4c712023aeb
Revises: 949c30d9b6b0
Create Date: 2026-07-21 13:57:09.741497

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b4c712023aeb'
down_revision: Union[str, Sequence[str], None] = '949c30d9b6b0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None
user_role_enum = sa.Enum(
    "USER",
    "PRO",
    "ADMIN",
    name="userrole",
)

user_role_enum = sa.Enum(
    "USER",
    "PRO",
    "ADMIN",
    name="userrole",
)

def upgrade() -> None:
    """Upgrade schema."""

    # Create PostgreSQL enum
    user_role_enum.create(op.get_bind(), checkfirst=True)

    # ------------------------------------------------------------------
    # Data Usage Table
    # ------------------------------------------------------------------

    op.create_table(
        "data_usages",

        sa.Column("id", sa.Integer(), primary_key=True),

        sa.Column(
            "user_id",
            sa.Integer(),
            nullable=False,
        ),

        sa.Column(
            "date",
            sa.Date(),
            nullable=False,
        ),

        sa.Column(
            "query_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),

        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            ondelete="CASCADE",
        ),

        sa.UniqueConstraint(
            "user_id",
            "date",
            name="uq_data_usage_user_date",
        ),
    )

    op.create_index(
        "ix_data_usages_user_id",
        "data_usages",
        ["user_id"],
    )

    op.create_index(
        "ix_data_usages_date",
        "data_usages",
        ["date"],
    )

    # ------------------------------------------------------------------
    # Users Table
    # ------------------------------------------------------------------

    op.add_column(
        "users",
        sa.Column(
            "role",
            user_role_enum,
            nullable=False,
            server_default="USER",
        ),
    )

    op.add_column(
        "users",
        sa.Column(
            "is_verified",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )

    op.add_column(
        "users",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.add_column(
        "users",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    # Remove temporary defaults

    op.alter_column(
        "users",
        "role",
        server_default=None,
    )

    op.alter_column(
        "users",
        "is_verified",
        server_default=None,
    )

    op.alter_column(
        "users",
        "created_at",
        server_default=None,
    )
def downgrade() -> None:
    """Downgrade schema."""

    op.drop_column("users", "updated_at")
    op.drop_column("users", "created_at")
    op.drop_column("users", "is_verified")
    op.drop_column("users", "role")

    op.drop_index(
        "ix_data_usages_date",
        table_name="data_usages",
    )

    op.drop_index(
        "ix_data_usages_user_id",
        table_name="data_usages",
    )

    op.drop_table("data_usages")

    user_role_enum.drop(
        op.get_bind(),
        checkfirst=True,
    )