import time
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from database.schema import Collection, Tenant

# Module-level bucket store so rate-limit state persists across requests
_bucket_store: Dict[str, "TokenBucket"] = {}


class TokenBucket:
    """Simple token bucket rate limiter (thread-safe for single-process)."""

    def __init__(self, rate: int, per: float = 60.0):
        self.rate = float(rate)
        self.per = per
        self.tokens = float(rate)
        self.last_refill = time.monotonic()

    def consume(self, tokens: float = 1.0) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.rate, self.tokens + elapsed * (self.rate / self.per))
        self.last_refill = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class TenantService:
    """CRUD and rate limiting for tenants."""

    def __init__(self, db_session: Session):
        self.db = db_session

    # ── CRUD ──────────────────────────────────────────

    def create_tenant(
        self,
        tenant_id: str,
        name: str,
        rate_limit_per_minute: int = 100,
    ) -> Dict[str, Any]:
        try:
            existing = self.get_tenant(tenant_id)
            if existing.get("success"):
                return {"success": False, "message": f"Tenant '{tenant_id}' already exists"}

            record = Tenant(
                tenant_id=tenant_id.strip().lower(),
                name=name.strip(),
                is_active=True,
                rate_limit_per_minute=rate_limit_per_minute,
            )
            self.db.add(record)
            self.db.commit()
            self.db.refresh(record)

            return {"success": True, "message": "Tenant created", "tenant": record.to_dict()}
        except Exception as e:
            self.db.rollback()
            return {"success": False, "message": f"Error creating tenant: {e}"}

    def get_tenant(self, tenant_id: str) -> Dict[str, Any]:
        try:
            record = (
                self.db.query(Tenant)
                .filter(Tenant.tenant_id == tenant_id.strip().lower())
                .first()
            )
            if not record:
                return {"success": False, "message": "Tenant not found"}
            return {"success": True, "tenant": record.to_dict()}
        except Exception as e:
            return {"success": False, "message": f"Error getting tenant: {e}"}

    def list_tenants(self) -> Dict[str, Any]:
        try:
            records = self.db.query(Tenant).order_by(Tenant.created_at.desc()).all()
            return {
                "success": True,
                "tenants": [r.to_dict() for r in records],
                "count": len(records),
            }
        except Exception as e:
            return {"success": False, "message": f"Error listing tenants: {e}"}

    def update_tenant(
        self,
        tenant_id: str,
        name: Optional[str] = None,
        is_active: Optional[bool] = None,
        rate_limit_per_minute: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            slug = tenant_id.strip().lower()
            record = self.db.query(Tenant).filter(Tenant.tenant_id == slug).first()
            if not record:
                return {"success": False, "message": "Tenant not found"}

            if name is not None:
                record.name = name.strip()
            if is_active is not None:
                record.is_active = is_active
            if rate_limit_per_minute is not None:
                record.rate_limit_per_minute = rate_limit_per_minute
                # Reset bucket so new rate takes effect immediately
                _bucket_store.pop(slug, None)

            self.db.commit()
            self.db.refresh(record)
            return {"success": True, "message": "Tenant updated", "tenant": record.to_dict()}
        except Exception as e:
            self.db.rollback()
            return {"success": False, "message": f"Error updating tenant: {e}"}

    def delete_tenant(self, tenant_id: str) -> Dict[str, Any]:
        try:
            slug = tenant_id.strip().lower()
            record = self.db.query(Tenant).filter(Tenant.tenant_id == slug).first()
            if not record:
                return {"success": False, "message": "Tenant not found"}

            # Clean up associated collections
            self.db.query(Collection).filter(Collection.tenant_id == slug).update(
                {Collection.tenant_id: None}
            )

            self.db.delete(record)
            self.db.commit()

            # Clean up bucket
            _bucket_store.pop(slug, None)

            return {"success": True, "message": f"Tenant '{slug}' deleted"}
        except Exception as e:
            self.db.rollback()
            return {"success": False, "message": f"Error deleting tenant: {e}"}

    # ── Rate Limiting ─────────────────────────────────

    def _default_rate_for(self, tenant_id: str) -> int:
        try:
            record = self.db.query(Tenant).filter(Tenant.tenant_id == tenant_id).first()
            if record:
                return record.rate_limit_per_minute
        except Exception:
            pass
        return 100

    def check_rate_limit(self, tenant_id: str, tokens: float = 1.0) -> bool:
        slug = tenant_id.strip().lower()
        if slug not in _bucket_store:
            rate = self._default_rate_for(slug)
            _bucket_store[slug] = TokenBucket(rate=rate, per=60.0)
        return _bucket_store[slug].consume(tokens)

    def get_rate_limit_info(self, tenant_id: str) -> Dict[str, Any]:
        slug = tenant_id.strip().lower()
        rate = self._default_rate_for(slug)
        bucket = _bucket_store.get(slug)
        return {
            "tenant_id": slug,
            "rate_limit_per_minute": rate,
            "tokens_remaining": bucket.tokens if bucket else rate,
            "bucket_exists": bucket is not None,
        }
