from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from config.database import get_db
from services.tenant_service import TenantService

router = APIRouter(prefix="/api/tenants", tags=["Tenants"])


def get_tenant_service(db: Session = Depends(get_db)) -> TenantService:
    return TenantService(db_session=db)


async def admin_required(request: Request):
    """Dependency: ensure the request has admin-level permissions."""
    info = getattr(request.state, "api_key_info", None)
    if not info or not isinstance(info, dict):
        raise HTTPException(
            status_code=403,
            detail={"success": False, "message": "Admin access required"},
        )
    permissions = info.get("permissions", "")
    if permissions != "admin":
        raise HTTPException(
            status_code=403,
            detail={"success": False, "message": "Admin access required"},
        )
    return True


@router.post("/create", dependencies=[Depends(admin_required)])
async def create_tenant(
    tenant_id: str = Query(..., description="Unique tenant identifier"),
    name: str = Query(..., description="Tenant display name"),
    rate_limit_per_minute: int = Query(100, ge=1, le=10000),
    service: TenantService = Depends(get_tenant_service),
):
    result = service.create_tenant(
        tenant_id=tenant_id,
        name=name,
        rate_limit_per_minute=rate_limit_per_minute,
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result)
    return result


@router.get("/{tenant_id}")
async def get_tenant(
    tenant_id: str,
    service: TenantService = Depends(get_tenant_service),
):
    result = service.get_tenant(tenant_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result)
    return result


@router.get("")
async def list_tenants(
    service: TenantService = Depends(get_tenant_service),
):
    return service.list_tenants()


@router.put("/{tenant_id}", dependencies=[Depends(admin_required)])
async def update_tenant(
    tenant_id: str,
    name: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    rate_limit_per_minute: Optional[int] = Query(None, ge=1, le=10000),
    service: TenantService = Depends(get_tenant_service),
):
    result = service.update_tenant(
        tenant_id=tenant_id,
        name=name,
        is_active=is_active,
        rate_limit_per_minute=rate_limit_per_minute,
    )
    if not result.get("success"):
        raise HTTPException(status_code=404 if "not found" in result.get("message", "") else 400, detail=result)
    return result


@router.delete("/{tenant_id}", dependencies=[Depends(admin_required)])
async def delete_tenant(
    tenant_id: str,
    service: TenantService = Depends(get_tenant_service),
):
    result = service.delete_tenant(tenant_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result)
    return result


@router.get("/{tenant_id}/rate-limit")
async def rate_limit_info(
    tenant_id: str,
    service: TenantService = Depends(get_tenant_service),
):
    return service.get_rate_limit_info(tenant_id)
