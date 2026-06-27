from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from config.database import get_db
from services.auth_service import APIKeyManager
from services.sso_service import OIDCConfig, OIDCProvider, SSOSession, SSOSessionData, TokenResponse
import time
import os

router = APIRouter(prefix="/api/keys", tags=["Auth"])

# ---- OIDC provider (configured from environment) ----
_oidc_config = OIDCConfig(
    issuer=os.environ.get("OIDC_ISSUER", "https://accounts.example.com"),
    client_id=os.environ.get("OIDC_CLIENT_ID", "vectordb-client"),
    client_secret=os.environ.get("OIDC_CLIENT_SECRET", "secret"),
    redirect_uri=os.environ.get("OIDC_REDIRECT_URI", "http://localhost:8000/api/auth/sso/callback"),
)
_oidc_provider = OIDCProvider(_oidc_config)

# Temporary in-memory store for PKCE verifiers keyed by state
_pkce_store: dict = {}


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


# ==================== SSO / OIDC Routes ====================

sso_router = APIRouter(prefix="/api/auth/sso", tags=["Auth"])


@sso_router.get("/login")
async def sso_login():
    """
    Initiate OIDC login with PKCE.
    Redirects the user to the identity provider's authorization endpoint.
    """
    auth_url, code_verifier, state = _oidc_provider.get_auth_url()
    _pkce_store[state] = code_verifier
    return RedirectResponse(url=auth_url)


@sso_router.get("/callback")
async def sso_callback(
    code: str = Query(..., description="Authorization code from IdP"),
    state: str = Query(..., description="State parameter for CSRF protection"),
):
    """
    Handle OIDC callback: exchange code for tokens, verify id_token, create session.
    Returns session_id that the client should store and send as a Bearer token.
    """
    code_verifier = _pkce_store.pop(state, None)
    if code_verifier is None:
        raise HTTPException(status_code=400, detail="Invalid or expired state parameter")

    try:
        tokens: TokenResponse = await _oidc_provider.exchange_code(code, code_verifier)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Token exchange failed: {exc}") from exc

    try:
        claims = await _oidc_provider.verify_token(tokens.id_token)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {exc}") from exc

    session_data = SSOSessionData(
        user_id=claims.get("sub", ""),
        email=claims.get("email", ""),
        roles=claims.get("roles", claims.get("groups", [])),
        expiry=time.time() + tokens.expires_in,
    )
    session_id = SSOSession.generate_session_id()
    SSOSession.save(session_id, session_data)

    return {
        "success": True,
        "session_id": session_id,
        "user_id": session_data.user_id,
        "email": session_data.email,
        "roles": session_data.roles,
        "expires_in": tokens.expires_in,
    }


@sso_router.get("/session")
async def get_sso_session(session_id: str = Query(...)):
    """Look up an active SSO session by session_id."""
    data = SSOSession.get(session_id)
    if not data:
        raise HTTPException(status_code=401, detail="Session not found or expired")
    return {
        "success": True,
        "user_id": data.user_id,
        "email": data.email,
        "roles": data.roles,
        "expiry": data.expiry,
    }
