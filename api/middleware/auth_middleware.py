"""Auth middleware: validates API keys, attaches tenant info, applies per-tenant rate limits."""
from fastapi import Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Callable
import logging

from config.database import get_db
from services.auth_service import APIKeyManager
from services.tenant_service import TenantService
from utils.rbac import policy_from_validation, Permission

logger = logging.getLogger(__name__)

PUBLIC_PATHS = {
    "/health", "/ready", "/docs", "/redoc", "/openapi.json",
    "/dashboard", "/dashboard/static",
    "/ws/health",
}


def get_api_key(request: Request):
    key = request.headers.get("X-API-Key") or request.headers.get(
        "Authorization", ""
    ).replace("Bearer ", "")
    if not key:
        key = request.query_params.get("api_key")
    return key if key else None


async def auth_middleware(request: Request, call_next: Callable):
    path = request.url.path

    if any(path.startswith(p) for p in PUBLIC_PATHS) or path == "/":
        return await call_next(request)

    api_key = get_api_key(request)

    if not api_key:
        return JSONResponse(
            status_code=401,
            content={
                "success": False,
                "message": "API key required. Provide via X-API-Key header or ?api_key=...",
            },
        )

    from config.settings import get_settings
    settings = get_settings()

    # ---- Admin key: full access, no rate limit ------------------------------
    if api_key == settings.API_KEY:
        request.state.api_key_info = {
            "success": True,
            "permissions": "admin",
            "tenant_id": None,
        }
        request.state.tenant_id = None
        return await call_next(request)

    # ---- Tenant-bound API key -----------------------------------------------
    db: Session = next(get_db())
    try:
        manager = APIKeyManager(db_session=db)
        result = manager.validate_key(api_key)

        if not result.get("success"):
            status = 401 if "Invalid" in str(result.get("message", "")) else 403
            return JSONResponse(status_code=status, content=result)

        tenant_id = result.get("tenant_id")

        # Build RBAC policy from the key record
        from utils.rbac import policy_from_validation
        rbac_policy = policy_from_validation(result)

        # Attach tenant and RBAC context to request state
        request.state.api_key_info = result
        request.state.tenant_id = tenant_id
        request.state.rbac_policy = rbac_policy  # available for endpoint enforcement

        # ---- Operation-level RBAC check -------------------------------------
        # Check write operations
        write_methods = {"POST", "PUT", "PATCH"}
        delete_methods = {"DELETE"}

        if request.method in delete_methods and not rbac_policy.allows("delete"):
            from utils.rbac import Permission
            if not rbac_policy.allows(Permission.DELETE):
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "message": "API key does not have delete permission",
                    },
                )

        if request.method in write_methods:
            from utils.rbac import Permission
            if not rbac_policy.allows(Permission.WRITE):
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "message": "API key does not have write permission",
                    },
                )

        # ---- Per-tenant rate limiting ---------------------------------------
        if tenant_id:
            ts = TenantService(db_session=db)
            if not ts.check_rate_limit(tenant_id):
                logger.warning("Rate limit hit for tenant %s", tenant_id)
                return JSONResponse(
                    status_code=429,
                    content={
                        "success": False,
                        "message": "Rate limit exceeded. Try again later.",
                    },
                )

        return await call_next(request)
    except Exception as e:
        logger.error(f"Auth error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Auth service error"},
        )
    finally:
        db.close()
