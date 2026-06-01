"""Tenant management API endpoints.

All endpoints require the admin API key (settings.API_KEY) and are
excluded from the standard auth middleware check.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional

from config.database import get_db
from services.tenant_service import TenantService
from services.auth_service import APIKeyManager

router = APIRouter(prefix="/api/tenants", tags=["Tenants"])


def get_tenant_service(db: Session = Depends(get_db)) -> TenantService:
    return TenantService(db_session=db)


def get_key_manager(db: Session = Depends(get_db)) -> APIKeyManager:
    return APIKeyManager(db_session=db)


# ==================== Tenant CRUD ====================


@router.post("", status_code=201)
async def create_tenant(
    name: str,
    tenant_id: Optional[str] = None,
    rate_limit_per_minute: Optional[int] = None,
    service: TenantService = Depends(get_tenant_service),
):
    """Create a new tenant (requires admin API key)."""
    result = service.create_tenant(
        name=name,
        tenant_id=tenant_id,
        rate_limit_per_minute=rate_limit_per_minute,
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result)
    return result


@router.get("")
async def list_tenants(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    service: TenantService = Depends(get_tenant_service),
):
    """List all tenants (requires admin API key)."""
    return service.list_tenants(limit=limit, offset=offset)


@router.get("/{tenant_id}")
async def get_tenant(
    tenant_id: str,
    service: TenantService = Depends(get_tenant_service),
):
    """Get a tenant by ID (requires admin API key)."""
    result = service.get_tenant(tenant_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result)
    return result


@router.patch("/{tenant_id}")
async def update_tenant(
    tenant_id: str,
    name: Optional[str] = None,
    is_active: Optional[bool] = None,
    rate_limit_per_minute: Optional[int] = None,
    service: TenantService = Depends(get_tenant_service),
):
    """Update tenant properties (name, active status, rate limit)."""
    result = service.update_tenant(
        tenant_id=tenant_id,
        name=name,
        is_active=is_active,
        rate_limit_per_minute=rate_limit_per_minute,
    )
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result)
    return result


@router.delete("/{tenant_id}")
async def delete_tenant(
    tenant_id: str,
    service: TenantService = Depends(get_tenant_service),
):
    """Delete a tenant and cascade all associated resources."""
    result = service.delete_tenant(tenant_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result)
    return result


# ==================== Per-Tenant API Keys ====================


@router.post("/{tenant_id}/keys", status_code=201)
async def create_tenant_api_key(
    tenant_id: str,
    name: str = "default",
    permissions: str = "read_write",
    collection_id: Optional[str] = None,
    manager: APIKeyManager = Depends(get_key_manager),
):
    """Create an API key scoped to a tenant."""
    result = manager.create_key(
        tenant_id=tenant_id,
        name=name,
        permissions=permissions,
        collection_id=collection_id,
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result)
    return result


@router.get("/{tenant_id}/keys")
async def list_tenant_api_keys(
    tenant_id: str,
    manager: APIKeyManager = Depends(get_key_manager),
):
    """List all API keys for a tenant."""
    return manager.list_keys_by_tenant(tenant_id)


@router.delete("/{tenant_id}/keys/{api_key}")
async def revoke_tenant_api_key(
    tenant_id: str,
    api_key: str,
    manager: APIKeyManager = Depends(get_key_manager),
):
    """Revoke a specific API key (must belong to the tenant)."""
    # Verify the key belongs to this tenant first
    result = manager.validate_key(api_key)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result)
    if result.get("tenant_id") != tenant_id:
        raise HTTPException(
            status_code=403,
            detail={"success": False, "message": "Key does not belong to this tenant"},
        )
    revoke_result = manager.revoke_key(api_key)
    if not revoke_result.get("success"):
        raise HTTPException(status_code=404, detail=revoke_result)
    return revoke_result
