from typing import Optional, Dict, Any, List
import secrets
import hashlib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class APIKeyManager:

    def __init__(self, db_session=None):
        self.db = db_session

    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def _generate_key(self) -> str:
        return f"vk_{secrets.token_hex(24)}"

    def create_key(
        self,
        tenant_id: str,
        name: str = "default",
        permissions: str = "read_write",
        collection_id: Optional[str] = None,
        row_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new API key with optional row-level security filter.

        Args:
            tenant_id: Tenant that owns this key.
            name: Human-readable key name.
            permissions: Permission level (read, write, read_write, admin).
            collection_id: Optional collection scope.
            row_filter: Optional metadata predicate DSL for row-level security,
                e.g. "department = 'eng' AND clearance < 3".
        """
        raw_key = self._generate_key()
        key_hash = self._hash_key(raw_key)

        try:
            from database.schema import ApiKey
            record = ApiKey(
                key_hash=key_hash,
                tenant_id=tenant_id,
                collection_id=collection_id,
                name=name,
                permissions=permissions,
                row_filter=row_filter,
            )
            self.db.add(record)
            self.db.commit()

            return {
                "success": True,
                "api_key": raw_key,
                "tenant_id": tenant_id,
                "collection_id": collection_id,
                "name": name,
                "permissions": permissions,
                "row_filter": row_filter,
            }
        except Exception as e:
            self.db.rollback()
            return {"success": False, "message": str(e)}

    def validate_key(self, api_key: str) -> Dict[str, Any]:
        from database.schema import ApiKey
        key_hash = self._hash_key(api_key)

        record = self.db.query(ApiKey).filter(ApiKey.key_hash == key_hash).first()
        if not record:
            return {"success": False, "message": "Invalid API key"}

        if record.expires_at and record.expires_at < datetime.utcnow():
            return {"success": False, "message": "API key expired"}

        if record.is_active is False:
            return {"success": False, "message": "API key is disabled"}

        # Verify tenant is still active
        from database.schema import Tenant
        tenant = self.db.query(Tenant).filter(
            Tenant.tenant_id == record.tenant_id,
            Tenant.is_active.is_(True),
        ).first() if record.tenant_id else None
        if record.tenant_id and not tenant:
            return {"success": False, "message": "Tenant is disabled or not found"}

        return {
            "success": True,
            "tenant_id": record.tenant_id,
            "collection_id": record.collection_id,
            "permissions": record.permissions,
            "name": record.name,
            "row_filter": record.row_filter,
        }

    def revoke_key(self, api_key: str) -> Dict[str, Any]:
        from database.schema import ApiKey
        key_hash = self._hash_key(api_key)
        record = self.db.query(ApiKey).filter(ApiKey.key_hash == key_hash).first()
        if not record:
            return {"success": False, "message": "Key not found"}
        self.db.delete(record)
        self.db.commit()
        return {"success": True, "message": "Key revoked"}

    def list_keys(
        self,
        tenant_id: Optional[str] = None,
        collection_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        from database.schema import ApiKey
        query = self.db.query(ApiKey)
        if tenant_id:
            query = query.filter(ApiKey.tenant_id == tenant_id)
        if collection_id:
            query = query.filter(ApiKey.collection_id == collection_id)
        records: List[Any] = query.all()
        return {
            "success": True,
            "keys": [
                {
                    "name": r.name,
                    "permissions": r.permissions,
                    "tenant_id": r.tenant_id,
                    "collection_id": r.collection_id,
                    "created_at": r.created_at,
                    "expires_at": r.expires_at,
                }
                for r in records
            ]
        }

    def list_keys_by_tenant(self, tenant_id: str) -> Dict[str, Any]:
        return self.list_keys(tenant_id=tenant_id)
