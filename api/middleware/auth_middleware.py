import logging
from typing import Callable
from fastapi import Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from config.database import get_db
from services.auth_service import APIKeyManager
from services.tenant_service import TenantService

logger = logging.getLogger(__name__)

PUBLIC_PATHS = {
    "/health",
    "/ready",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/metrics",
    "/",
    "/api/keys/create",
}


def get_api_key(request: Request) -> str:
    key = request.headers.get("X-API-Key") or request.headers.get(
        "Authorization", ""
    ).replace("Bearer ", "")
    if not key:
        key = request.query_params.get("api_key", "")
    return key


async def auth_middleware(request: Request, call_next: Callable):
    """Authenticate requests via API key and attach tenant context."""
    from config.settings import get_settings

    settings = get_settings()
    path = request.url.path

    # Skip auth for public paths and OPTIONS requests
    if path in PUBLIC_PATHS or request.method == "OPTIONS" or path.startswith("/static/"):
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

    # Check against master admin key
    if api_key == settings.API_KEY:
        request.state.api_key_info = {"success": True, "permissions": "admin"}
        request.state.tenant_id = None
        return await call_next(request)

    # Validate against database
    try:
        db: Session = next(get_db())
        try:
            manager = APIKeyManager(db_session=db)
            result = manager.validate_key(api_key)

            if not result.get("success"):
                return JSONResponse(
                    status_code=401,
                    content={"success": False, "message": result.get("message", "Invalid API key")},
                )

            tenant_id = result.get("tenant_id")
            request.state.api_key_info = result
            request.state.tenant_id = tenant_id

            # Per-tenant rate limiting
            if tenant_id:
                ts = TenantService(db_session=db)
                if not ts.check_rate_limit(tenant_id):
                    return JSONResponse(
                        status_code=429,
                        content={
                            "success": False,
                            "message": "Rate limit exceeded for tenant. Try again later.",
                        },
                    )

            return await call_next(request)
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Auth error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Auth service error"},
        )
