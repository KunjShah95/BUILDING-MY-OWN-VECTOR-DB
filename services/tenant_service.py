"""Tenant management service with in-memory rate limiting."""
import secrets
import time
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from config.settings import get_settings

logger = __import__("logging").getLogger(__name__)

# Module-level token-bucket store shared across all TenantService instances.
# This ensures rate-limit state persists across requests.
_buckets: Dict[str, "TokenBucket"] = {}


class TokenBucket:
    """Simple in-memory token bucket for per-tenant rate limiting."""

    def __init__(self, rate: int, burst: Optional[int] = None) -> None:
        self.rate = rate  # tokens per minute
        self.burst = burst or rate
        self.tokens = float(self.burst)
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        new_tokens = elapsed * (self.rate / 60.0)
        if new_tokens > 0:
            self.tokens = min(self.burst, self.tokens + new_tokens)
            self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class TenantService:
    """CRUD for tenants and in-memory rate-limit tracking."""

    def __init__(self, db_session: Session) -> None:
        self.db = db_session

    # ---- In-memory rate limiting -------------------------------------------

    def _get_bucket(self, tenant_id: str, rate_limit: Optional[int] = None) -> TokenBucket:
        if tenant_id not in _buckets:
            rate = rate_limit or self._default_rate_for(tenant_id)
            _buckets[tenant_id] = TokenBucket(rate=rate)
        return _buckets[tenant_id]

    def _default_rate_for(self, tenant_id: str) -> int:
        """Look up the DB-stored rate limit for a tenant."""
        from database.schema import Tenant
        record = self.db.query(Tenant).filter(
            Tenant.tenant_id == tenant_id,
            Tenant.is_active.is_(True),
        ).first()
        return record.rate_limit_per_minute if record else 100

    def check_rate_limit(self, tenant_id: str) -> bool:
        """Return True if the request is allowed, False if rate-limited."""
        bucket = self._get_bucket(tenant_id)
        return bucket.consume()

    # ---- Tenant CRUD -------------------------------------------------------

    def _generate_tenant_id(self) -> str:
        return f"tenant_{uuid.uuid4().hex[:12]}"

    def create_tenant(
        self,
        name: str,
        tenant_id: Optional[str] = None,
        rate_limit_per_minute: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            from database.schema import Tenant

            slug = (tenant_id or self._generate_tenant_id()).strip().lower()
            existing = self.db.query(Tenant).filter(
                Tenant.tenant_id == slug,
            ).first()
            if existing:
                return {"success": False, "message": f"Tenant '{slug}' already exists"}

            record = Tenant(
                tenant_id=slug,
                name=name.strip(),
                rate_limit_per_minute=rate_limit_per_minute or 100,
                is_active=True,
            )
            self.db.add(record)
            self.db.commit()
            self.db.refresh(record)
            return {
                "success": True,
                "message": "Tenant created",
                "tenant": record.to_dict(),
            }
        except Exception as exc:
            self.db.rollback()
            return {"success": False, "message": f"Error creating tenant: {exc}"}

    def get_tenant(self, tenant_id: str) -> Dict[str, Any]:
        try:
            from database.schema import Tenant
            record = self.db.query(Tenant).filter(
                Tenant.tenant_id == tenant_id.strip().lower(),
            ).first()
            if not record:
                return {"success": False, "message": "Tenant not found"}
            return {"success": True, "tenant": record.to_dict()}
        except Exception as exc:
            return {"success": False, "message": f"Error getting tenant: {exc}"}

    def list_tenants(self, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        try:
            from database.schema import Tenant
            rows = self.db.query(Tenant).order_by(
                Tenant.created_at.desc(),
            ).offset(offset).limit(limit).all()
            return {
                "success": True,
                "tenants": [r.to_dict() for r in rows],
                "count": len(rows),
            }
        except Exception as exc:
            return {"success": False, "message": f"Error listing tenants: {exc}"}

    def update_tenant(
        self,
        tenant_id: str,
        name: Optional[str] = None,
        is_active: Optional[bool] = None,
        rate_limit_per_minute: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            from database.schema import Tenant
            slug = tenant_id.strip().lower()
            record = self.db.query(Tenant).filter(
                Tenant.tenant_id == slug,
            ).first()
            if not record:
                return {"success": False, "message": "Tenant not found"}

            if name is not None:
                record.name = name.strip()
            if is_active is not None:
                record.is_active = is_active
            if rate_limit_per_minute is not None:
                record.rate_limit_per_minute = rate_limit_per_minute
                # Reset bucket so new rate takes effect immediately
                _buckets.pop(slug, None)

            self.db.commit()
            self.db.refresh(record)
            return {"success": True, "message": "Tenant updated", "tenant": record.to_dict()}
        except Exception as exc:
            self.db.rollback()
            return {"success": False, "message": f"Error updating tenant: {exc}"}

    def delete_tenant(self, tenant_id: str) -> Dict[str, Any]:
        try:
            from database.schema import Tenant
            slug = tenant_id.strip().lower()
            record = self.db.query(Tenant).filter(
                Tenant.tenant_id == slug,
            ).first()
            if not record:
                return {"success": False, "message": "Tenant not found"}

            self.db.delete(record)
            self.db.commit()
            _buckets.pop(slug, None)
            return {"success": True, "message": f"Tenant '{slug}' deleted"}
        except Exception as exc:
            self.db.rollback()
            return {"success": False, "message": f"Error deleting tenant: {exc}"}
