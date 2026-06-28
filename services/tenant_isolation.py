"""Tenant-level schema isolation with namespace prefixing and quota enforcement."""
import shutil
import threading
import time
from pathlib import Path
from typing import List, Optional

import logging

logger = logging.getLogger(__name__)


class QuotaExceededError(Exception):
    """Raised when a tenant exceeds their resource quota."""
    pass


class TenantQuota:
    """Resource quota limits for a tenant."""

    def __init__(
        self,
        max_vectors: int = 100_000,
        max_collections: int = 50,
        max_qps: int = 100,
    ) -> None:
        self.max_vectors = max_vectors
        self.max_collections = max_collections
        self.max_qps = max_qps


class QuotaEnforcer:
    """Thread-safe quota counter that tracks usage and enforces limits."""

    def __init__(self, quota: TenantQuota) -> None:
        self.quota = quota
        self._lock = threading.Lock()
        self._vector_count: int = 0
        self._collection_count: int = 0
        # QPS tracking: timestamps in the last 1-second sliding window
        self._request_times: List[float] = []

    def check_and_increment(
        self,
        vectors: int = 0,
        collections: int = 0,
        check_qps: bool = False,
    ) -> None:
        """Check quotas and increment counters. Raises QuotaExceededError on violation."""
        with self._lock:
            if check_qps:
                now = time.monotonic()
                self._request_times = [t for t in self._request_times if now - t < 1.0]
                if len(self._request_times) >= self.quota.max_qps:
                    raise QuotaExceededError(
                        f"QPS limit of {self.quota.max_qps} exceeded"
                    )
                self._request_times.append(now)

            if vectors > 0:
                if self._vector_count + vectors > self.quota.max_vectors:
                    raise QuotaExceededError(
                        f"Vector limit of {self.quota.max_vectors} exceeded "
                        f"(current: {self._vector_count}, adding: {vectors})"
                    )
                self._vector_count += vectors

            if collections > 0:
                if self._collection_count + collections > self.quota.max_collections:
                    raise QuotaExceededError(
                        f"Collection limit of {self.quota.max_collections} exceeded "
                        f"(current: {self._collection_count})"
                    )
                self._collection_count += collections

    def reset_counters(self) -> None:
        """Reset all usage counters."""
        with self._lock:
            self._vector_count = 0
            self._collection_count = 0
            self._request_times = []

    @property
    def vector_count(self) -> int:
        with self._lock:
            return self._vector_count

    @property
    def collection_count(self) -> int:
        with self._lock:
            return self._collection_count


class TenantIsolation:
    """
    Provides per-tenant namespace isolation by prefixing all collection names.

    Collection names become: tenant_{tenant_id}_{original_name}
    """

    def __init__(self, index_dir: Optional[str] = None) -> None:
        self.index_dir = Path(index_dir or "indexes")

    @staticmethod
    def get_tenant_prefix(tenant_id: str) -> str:
        """Return the namespace prefix for a given tenant."""
        return f"tenant_{tenant_id}_"

    def namespaced_collection(self, tenant_id: str, collection_name: str) -> str:
        """Return the fully-qualified (prefixed) collection name."""
        prefix = self.get_tenant_prefix(tenant_id)
        if collection_name.startswith(prefix):
            return collection_name
        return f"{prefix}{collection_name}"

    def strip_prefix(self, tenant_id: str, collection_name: str) -> str:
        """Remove the tenant prefix, returning the bare collection name."""
        prefix = self.get_tenant_prefix(tenant_id)
        if collection_name.startswith(prefix):
            return collection_name[len(prefix):]
        return collection_name

    def list_tenant_collections(self, tenant_id: str) -> List[str]:
        """Return all collections that belong to this tenant (filtered by prefix)."""
        prefix = self.get_tenant_prefix(tenant_id)
        collections: List[str] = []
        if not self.index_dir.exists():
            return collections
        for entry in self.index_dir.iterdir():
            if entry.name.startswith(prefix):
                collections.append(entry.name)
        return sorted(collections)

    def purge_tenant(self, tenant_id: str) -> int:
        """Delete all collections belonging to this tenant. Returns count removed."""
        collections = self.list_tenant_collections(tenant_id)
        removed = 0
        for col in collections:
            target = self.index_dir / col
            try:
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.is_file():
                    target.unlink()
                removed += 1
                logger.info("Purged tenant collection: %s", target)
            except Exception:
                logger.exception("Failed to purge %s", target)
        return removed


# Module-level quota registry
_quota_registry: dict = {}
_registry_lock = threading.Lock()


def get_quota_enforcer(tenant_id: str, quota: Optional[TenantQuota] = None) -> QuotaEnforcer:
    """Return (or create) the QuotaEnforcer for a tenant."""
    with _registry_lock:
        if tenant_id not in _quota_registry:
            _quota_registry[tenant_id] = QuotaEnforcer(quota or TenantQuota())
        return _quota_registry[tenant_id]
