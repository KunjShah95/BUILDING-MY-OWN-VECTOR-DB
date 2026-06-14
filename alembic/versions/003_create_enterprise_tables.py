"""create data_retention_policies, query_budgets, and compliance_reports tables

Revision ID: 003
Revises: 002
Create Date: 2026-06-14

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "data_retention_policies",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("collection_id", sa.String(), nullable=False, index=True),
        sa.Column("ttl_days", sa.Integer(), nullable=False, server_default="365"),
        sa.Column("archive_after_days", sa.Integer(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_drp_collection", "data_retention_policies", ["collection_id"])

    op.create_table(
        "query_budgets",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("tenant_id", sa.String(), nullable=False, index=True),
        sa.Column("max_vectors_scanned", sa.Integer(), server_default="100000"),
        sa.Column("max_ef_search", sa.Integer(), server_default="800"),
        sa.Column("max_concurrent_queries", sa.Integer(), server_default="50"),
        sa.Column("cost_limit_per_query", sa.Integer(), server_default="1000"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_qb_tenant", "query_budgets", ["tenant_id"])

    op.create_table(
        "compliance_reports",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("report_type", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False, index=True),
        sa.Column("status", sa.String(), server_default="pending"),
        sa.Column("report_data", sa.JSON(), nullable=True),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_cr_tenant_type", "compliance_reports", ["tenant_id", "report_type"])


def downgrade() -> None:
    op.drop_index("idx_cr_tenant_type", table_name="compliance_reports")
    op.drop_table("compliance_reports")
    op.drop_index("idx_qb_tenant", table_name="query_budgets")
    op.drop_table("query_budgets")
    op.drop_index("idx_drp_collection", table_name="data_retention_policies")
    op.drop_table("data_retention_policies")
