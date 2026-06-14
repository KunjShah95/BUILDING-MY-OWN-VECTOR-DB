"""Enterprise compliance, data retention, and query budget services."""
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from database.schema import DataRetentionPolicy, QueryBudget, ComplianceReport

logger = logging.getLogger(__name__)

class ComplianceService:
    def __init__(self, db: Session):
        self.db = db

    # --- Data Retention ---
    def set_retention_policy(self, collection_id: str, ttl_days: int = 365,
                             archive_after_days: Optional[int] = None,
                             enabled: bool = True) -> Dict[str, Any]:
        policy = self.db.query(DataRetentionPolicy).filter(
            DataRetentionPolicy.collection_id == collection_id
        ).first()
        if not policy:
            policy = DataRetentionPolicy(collection_id=collection_id)
            self.db.add(policy)
        policy.ttl_days = ttl_days
        policy.archive_after_days = archive_after_days
        policy.enabled = enabled
        self.db.commit()
        return {"success": True, "policy": {
            "collection_id": policy.collection_id,
            "ttl_days": policy.ttl_days,
            "archive_after_days": policy.archive_after_days,
            "enabled": policy.enabled,
        }}

    def get_retention_policy(self, collection_id: str) -> Dict[str, Any]:
        policy = self.db.query(DataRetentionPolicy).filter(
            DataRetentionPolicy.collection_id == collection_id
        ).first()
        if not policy:
            return {"success": True, "policy": None}
        return {"success": True, "policy": {
            "collection_id": policy.collection_id,
            "ttl_days": policy.ttl_days,
            "archive_after_days": policy.archive_after_days,
            "enabled": policy.enabled,
        }}

    def list_expired_vectors(self, collection_id: str) -> List[str]:
        """Return vector IDs that have expired based on retention policy."""
        policy = self.db.query(DataRetentionPolicy).filter(
            DataRetentionPolicy.collection_id == collection_id
        ).first()
        if not policy or not policy.enabled:
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(days=policy.ttl_days)
        from database.schema import Vector  # noqa: E501 — assuming Vector model exists
        expired = self.db.query(Vector.vector_id).filter(
            Vector.collection_id == collection_id,
            Vector.created_at < cutoff
        ).all()
        return [v[0] for v in expired]

    # --- Query Budget ---
    def set_query_budget(self, tenant_id: str, max_vectors_scanned: int = 100000,
                         max_ef_search: int = 800, max_concurrent: int = 50,
                         cost_limit: int = 1000) -> Dict[str, Any]:
        budget = self.db.query(QueryBudget).filter(
            QueryBudget.tenant_id == tenant_id
        ).first()
        if not budget:
            budget = QueryBudget(tenant_id=tenant_id)
            self.db.add(budget)
        budget.max_vectors_scanned = max_vectors_scanned
        budget.max_ef_search = max_ef_search
        budget.max_concurrent_queries = max_concurrent
        budget.cost_limit_per_query = cost_limit
        self.db.commit()
        return {"success": True, "budget": {
            "tenant_id": budget.tenant_id,
            "max_vectors_scanned": budget.max_vectors_scanned,
            "max_ef_search": budget.max_ef_search,
            "max_concurrent_queries": budget.max_concurrent_queries,
            "cost_limit_per_query": budget.cost_limit_per_query,
        }}

    def get_query_budget(self, tenant_id: str) -> Dict[str, Any]:
        budget = self.db.query(QueryBudget).filter(
            QueryBudget.tenant_id == tenant_id
        ).first()
        if not budget:
            return {"success": True, "budget": None}
        return {"success": True, "budget": {
            "tenant_id": budget.tenant_id,
            "max_vectors_scanned": budget.max_vectors_scanned,
            "max_ef_search": budget.max_ef_search,
            "max_concurrent_queries": budget.max_concurrent_queries,
            "cost_limit_per_query": budget.cost_limit_per_query,
        }}

    # --- Compliance Reporting ---
    def generate_report(self, report_type: str, tenant_id: str,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> Dict[str, Any]:
        report = ComplianceReport(report_type=report_type, tenant_id=tenant_id)
        self.db.add(report)
        self.db.flush()
        report_data = {"report_type": report_type, "tenant_id": tenant_id}
        # Gather data from existing audit logs
        try:
            from utils.audit_log import query_logs
            logs = query_logs(action=None, tenant_id=tenant_id, limit=1000)
            report_data["total_audit_events"] = len(logs.get("logs", []))
            report_data["events_sampled"] = min(len(logs.get("logs", [])), 5)
        except Exception:
            report_data["audit_log_error"] = "Audit logs unavailable"
        # Add encryption status
        try:
            from config.settings import get_settings
            settings = get_settings()
            report_data["encryption_at_rest"] = bool(settings.ENCRYPTION_KEY)
        except Exception:
            report_data["encryption_at_rest"] = "unknown"
        # Add tenant isolation info
        try:
            tenants = self.db.query(QueryBudget).filter(
                QueryBudget.tenant_id == tenant_id
            ).count()
            report_data["tenant_budgets_configured"] = tenants
        except Exception:
            report_data["tenant_budgets_configured"] = 0
        report.report_data = report_data
        report.status = "generated"
        self.db.commit()
        return {"success": True, "report": {
            "id": report.id,
            "report_type": report.report_type,
            "tenant_id": report.tenant_id,
            "status": report.status,
            "data": report.report_data,
            "generated_at": str(report.generated_at),
        }}

    def list_reports(self, tenant_id: str) -> Dict[str, Any]:
        reports = self.db.query(ComplianceReport).filter(
            ComplianceReport.tenant_id == tenant_id
        ).order_by(ComplianceReport.created_at.desc()).limit(50).all()
        return {"success": True, "reports": [
            {"id": r.id, "report_type": r.report_type, "status": r.status,
             "generated_at": str(r.generated_at)} for r in reports
        ]}
