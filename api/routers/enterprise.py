"""Enterprise and compliance API endpoints."""
import logging
from typing import Optional
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from config.database import get_db
from services.compliance_service import ComplianceService
from models.pydantic_models import (
    DataRetentionPolicyCreate, DataRetentionPolicyResponse,
    QueryBudgetCreate, QueryBudgetResponse,
    ComplianceReportRequest, ComplianceReportResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Enterprise"])

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
