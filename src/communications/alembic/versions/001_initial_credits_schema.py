"""Initial credits ledger schema

Revision ID: 001
Revises:
Create Date: 2025-01-16 10:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("node_id", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
        sa.UniqueConstraint("node_id"),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=False)

    # Create wallets table
    op.create_table(
        "wallets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("balance", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id"),
    )
    op.create_index("idx_wallet_user", "wallets", ["user_id"], unique=False)

    # Create transactions table
    op.create_table(
        "transactions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("from_user_id", sa.Integer(), nullable=False),
        sa.Column("to_user_id", sa.Integer(), nullable=False),
        sa.Column("amount", sa.Integer(), nullable=False),
        sa.Column("burn_amount", sa.Integer(), nullable=False),
        sa.Column("net_amount", sa.Integer(), nullable=False),
        sa.Column("transaction_type", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["from_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["to_user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_transaction_created", "transactions", ["created_at"], unique=False
    )
    op.create_index(
        "idx_transaction_from_user", "transactions", ["from_user_id"], unique=False
    )
    op.create_index(
        "idx_transaction_to_user", "transactions", ["to_user_id"], unique=False
    )

    # Create earnings table
    op.create_table(
        "earnings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("scrape_timestamp", sa.DateTime(), nullable=False),
        sa.Column("uptime_seconds", sa.Integer(), nullable=False),
        sa.Column("flops", sa.Integer(), nullable=False),
        sa.Column("bandwidth_bytes", sa.Integer(), nullable=False),
        sa.Column("credits_earned", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "scrape_timestamp", name="unique_user_scrape"),
    )
    op.create_index(
        "idx_earning_scrape", "earnings", ["scrape_timestamp"], unique=False
    )
    op.create_index("idx_earning_user", "earnings", ["user_id"], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table("earnings")
    op.drop_table("transactions")
    op.drop_table("wallets")
    op.drop_table("users")
