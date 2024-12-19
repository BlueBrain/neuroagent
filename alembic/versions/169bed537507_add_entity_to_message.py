"""add entity to message

Revision ID: 169bed537507
Revises: b57e558cf11f
Create Date: 2024-11-13 10:27:11.640265

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "169bed537507"
down_revision: Union[str, None] = "b57e558cf11f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create the enum type first
    entity_enum = sa.Enum("USER", "AI_TOOL", "TOOL", "AI_MESSAGE", name="entity")
    entity_enum.create(op.get_bind())

    # Then add the column using the enum
    op.add_column("messages", sa.Column("entity", entity_enum, nullable=False))


def downgrade() -> None:
    # Drop the column first
    op.drop_column("messages", "entity")

    # Then drop the enum type
    entity_enum = sa.Enum(name="entity")
    entity_enum.drop(op.get_bind())
