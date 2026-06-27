"""Enterprise and compliance API endpoints."""
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from config.database import get_db
from services.compliance_service import ComplianceService
from services.audit_service import AuditService
from models.pydantic_models import (
    DataRetentionPolicyCreate, DataRetentionPolicyResponse,
    QueryBudgetCreate, QueryBudgetResponse,
    ComplianceReportRequest, ComplianceReportResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Enterprise"])

_audit_service = AuditService()

def _get_compliance_service(db: Session = Depends(get_db)):
    return ComplianceService(db)

@router.post("/admin/retention", response_model=DataRetentionPolicyResponse)
def set_retention_policy(body: DataRetentionPolicyCreate,
                         svc: ComplianceService = Depends(_get_compliance_service)):
    return svc.set_retention_policy(**body.model_dump())

@router.get("/admin/retention/{collection_id}")
def get_retention_policy(collection_id: str,
                         svc: ComplianceService = Depends(_get_compliance_service)):
    return svc.get_retention_policy(collection_id)

@router.get("/admin/retention/{collection_id}/expired")
def list_expired(collection_id: str,
                 svc: ComplianceService = Depends(_get_compliance_service)):
    expired = svc.list_expired_vectors(collection_id)
    return {"success": True, "collection_id": collection_id, "expired_count": len(expired), "expired_ids": expired}

@router.post("/admin/query-budgets", response_model=QueryBudgetResponse)
def set_query_budget(body: QueryBudgetCreate,
                     svc: ComplianceService = Depends(_get_compliance_service)):
    return svc.set_query_budget(**body.model_dump())

@router.get("/admin/query-budgets/{tenant_id}")
def get_query_budget(tenant_id: str,
                     svc: ComplianceService = Depends(_get_compliance_service)):
    return svc.get_query_budget(tenant_id)

@router.post("/admin/compliance/reports", response_model=ComplianceReportResponse)
def generate_report(body: ComplianceReportRequest,
                    svc: ComplianceService = Depends(_get_compliance_service)):
    return svc.generate_report(**body.model_dump())

@router.get("/admin/compliance/reports/{tenant_id}")
def list_reports(tenant_id: str,
                 svc: ComplianceService = Depends(_get_compliance_service)):
    return svc.list_reports(tenant_id)


# ==================== Audit Export (SOC2 / GDPR) ====================

@router.get("/api/enterprise/audit/export")
def audit_export(
    tenant_id: Optional[str] = Query(None, description="Tenant ID for SOC2 export"),
    user_id: Optional[str] = Query(None, description="User ID for GDPR export"),
    format: str = Query("soc2", description="Export format: soc2 or gdpr"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD) for SOC2"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD) for SOC2"),
):
    """
    Export audit log entries.

    - format=soc2: requires tenant_id; optionally filter by start_date/end_date
    - format=gdpr:  requires user_id; returns all events for that user (DSAR)
    """
    if format == "soc2":
        if not tenant_id:
            raise HTTPException(status_code=400, detail="tenant_id is required for SOC2 export")
        events = _audit_service.export_soc2(tenant_id, start_date=start_date, end_date=end_date)
        return {"success": True, "format": "soc2", "tenant_id": tenant_id, "count": len(events), "events": events}

    elif format == "gdpr":
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required for GDPR export")
        events = _audit_service.export_gdpr(user_id)
        return {"success": True, "format": "gdpr", "user_id": user_id, "count": len(events), "events": events}

    else:
        raise HTTPException(status_code=400, detail=f"Unknown format '{format}'. Use 'soc2' or 'gdpr'.")


@router.post("/api/enterprise/audit/log")
def audit_log_event(
    tenant_id: str = Query(...),
    user_id: str = Query(...),
    action: str = Query(...),
    resource: str = Query(...),
):
    """Append an audit event to the log (for testing / manual ingestion)."""
    entry = _audit_service.log_event(tenant_id=tenant_id, user_id=user_id, action=action, resource=resource)
    return {"success": True, "entry": entry}


@router.delete("/api/enterprise/audit/purge-user")
def audit_purge_user(user_id: str = Query(..., description="User ID whose PII to redact")):
    """GDPR right-to-erasure: redact PII for a user from the audit log."""
    count = _audit_service.purge_user_data(user_id)
    return {"success": True, "user_id": user_id, "redacted_count": count}
