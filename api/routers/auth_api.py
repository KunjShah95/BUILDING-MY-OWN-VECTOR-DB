from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from config.database import get_db
from services.auth_service import APIKeyManager

router = APIRouter(prefix="/api/keys", tags=["Auth"])


def get_key_manager(db: Session = Depends(get_db)) -> APIKeyManager:
    return APIKeyManager(db_session=db)


@router.post("/create")
async def create_api_key(
    tenant_id: str,
    name: str = "default",
    permissions: str = "read_write",
    collection_id: Optional[str] = None,
    manager: APIKeyManager = Depends(get_key_manager),
):
    result = manager.create_key(
        tenant_id=tenant_id,
        name=name,
        permissions=permissions,
        collection_id=collection_id,
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result)
    return result


@router.delete("/revoke")
async def revoke_api_key(
    api_key: str,
    manager: APIKeyManager = Depends(get_key_manager),
):
    result = manager.revoke_key(api_key)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result)
    return result


@router.get("/list/{tenant_id}")
async def list_api_keys(
    tenant_id: str,
    manager: APIKeyManager = Depends(get_key_manager),
):
    return manager.list_keys_by_tenant(tenant_id)
