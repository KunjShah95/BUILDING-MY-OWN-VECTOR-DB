import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from database.schema import ApiKey, Tenant


class APIKeyManager:
    """Manages API key creation, validation, and revocation with tenant support."""

    def __init__(self, db_session: Optional[Session] = None):
        self.db = db_session

    def _hash_key(self, raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def _generate_key(self) -> str:
        return f"vd_{secrets.token_hex(24)}"

    def create_key(
        self,
        tenant_id: Optional[str] = None,
        name: str = "default",
        permissions: str = "read_write",
        collection_id: Optional[str] = None,
        expires_in_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a new API key for a tenant."""
        try:
            if not self.db:
                return {"success": False, "message": "No database session"}

            raw_key = self._generate_key()
            key_hash = self._hash_key(raw_key)

            api_key = ApiKey(
                key_hash=key_hash,
                name=name,
                permissions=permissions,
                tenant_id=tenant_id,
                collection_id=collection_id,
                is_active=True,
                expires_at=(
                    datetime.utcnow() + timedelta(days=expires_in_days)
                    if expires_in_days
                    else None
                ),
            )
            self.db.add(api_key)
            self.db.commit()
            self.db.refresh(api_key)

            return {
                "success": True,
                "api_key": raw_key,
                "key_id": api_key.id,
                "tenant_id": tenant_id,
                "permissions": permissions,
            }
        except Exception as e:
            if self.db:
                self.db.rollback()
            return {"success": False, "message": f"Error creating key: {e}"}

    def validate_key(self, api_key: str) -> Dict[str, Any]:
        """Validate an API key and return tenant info if valid."""
        try:
            if not self.db:
                return {"success": False, "message": "No database session"}

            key_hash = self._hash_key(api_key)
            record = (
                self.db.query(ApiKey)
                .filter(ApiKey.key_hash == key_hash, ApiKey.is_active.is_(True))
                .first()
            )

            if not record:
                return {"success": False, "message": "Invalid API key"}

            # Check expiration
            if record.expires_at and record.expires_at < datetime.utcnow():
                return {"success": False, "message": "API key expired"}

            # Check tenant is active
            if record.tenant_id:
                tenant = (
                    self.db.query(Tenant)
                    .filter(
                        Tenant.tenant_id == record.tenant_id,
                        Tenant.is_active.is_(True),
                    )
                    .first()
                )
                if not tenant:
                    return {"success": False, "message": "Tenant is inactive"}

            result = {
                "success": True,
                "tenant_id": record.tenant_id,
                "permissions": record.permissions,
                "collection_id": record.collection_id,
                "key_name": record.name,
            }

            return result
        except Exception as e:
            return {"success": False, "message": f"Error validating key: {e}"}

    def revoke_key(self, api_key: str) -> Dict[str, Any]:
        """Revoke an API key."""
        try:
            if not self.db:
                return {"success": False, "message": "No database session"}

            key_hash = self._hash_key(api_key)
            record = self.db.query(ApiKey).filter(ApiKey.key_hash == key_hash).first()

            if not record:
                return {"success": False, "message": "API key not found"}

            record.is_active = False
            self.db.commit()

            return {"success": True, "message": "API key revoked"}
        except Exception as e:
            if self.db:
                self.db.rollback()
            return {"success": False, "message": f"Error revoking key: {e}"}

    def list_keys_by_tenant(self, tenant_id: str) -> Dict[str, Any]:
        """List all API keys for a tenant."""
        try:
            if not self.db:
                return {"success": False, "message": "No database session"}

            keys = (
                self.db.query(ApiKey)
                .filter(ApiKey.tenant_id == tenant_id)
                .order_by(ApiKey.created_at.desc())
                .all()
            )

            return {
                "success": True,
                "keys": [
                    {"id": k.id, "name": k.name, "permissions": k.permissions, "is_active": k.is_active, "created_at": k.created_at}
                    for k in keys
                ],
            }
        except Exception as e:
            return {"success": False, "message": f"Error listing keys: {e}"}
