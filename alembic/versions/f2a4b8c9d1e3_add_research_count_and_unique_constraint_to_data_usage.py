"""add research_count and unique constraint to data_usage

Revision ID: f2a4b8c9d1e3
Revises: 861a3d4ea8eb
Create Date: 2026-11-10 22:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f2a4b8c9d1e3'
down_revision: Union[str, Sequence[str], None] = '861a3d4ea8eb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add research_count column to data_usages with default 0
    op.add_column(
        'data_usages',
        sa.Column('research_count', sa.Integer(), nullable=False, server_default='0')
    )

    # 2. Add unique constraint (user_id, date) to prevent concurrency duplicates
    op.create_unique_constraint(
        'uq_user_daily_usage',
        'data_usages',
        ['user_id', 'date']
    )


def downgrade() -> None:
    op.drop_constraint('uq_user_daily_usage', 'data_usages', type_='unique')
    op.drop_column('data_usages', 'research_count')
