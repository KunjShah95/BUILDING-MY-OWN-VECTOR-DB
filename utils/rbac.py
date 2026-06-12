"""
Role-Based Access Control with row-level (document-level) security.

Builds on the existing API-key scoping (tenant + collection + permission) by
adding two enforcement layers:

1. **Operation permissions** — read / write / delete / admin gates derived from
   the key's ``permissions`` string ("read", "write", "read_write", "admin").
2. **Row-level filters** — an optional metadata predicate (expressed in the
   same hybrid-query DSL as the query planner) that every returned/affected row
   must satisfy. E.g. a key scoped to ``department = 'eng'`` can only see rows
   whose metadata matches, even within a shared collection.

The row filter compiles to a callable usable directly as the
``metadata_filter`` argument of ``HNSWIndex.search`` so enforcement happens
during traversal, not just as a post-filter.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from utils.query_planner import parse_query, to_predicate_fn


class Permission(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


# What each permission string grants.
_GRANTS: Dict[str, set] = {
    "read": {Permission.READ},
    "write": {Permission.READ, Permission.WRITE},
    "read_write": {Permission.READ, Permission.WRITE, Permission.DELETE},
    "admin": {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN},
}


class AccessDenied(Exception):
    """Raised when an operation violates the active RBAC policy."""


@dataclass
class RBACPolicy:
    """
    Effective access policy resolved from a validated API key.

    Args:
        tenant_id: Tenant the key belongs to (None = global/admin).
        collection_id: Collection the key is scoped to (None = all in tenant).
        permissions: Permission string from the key.
        row_filter_dsl: Optional metadata predicate the key is restricted to,
            e.g. "department = 'eng' AND clearance < 3".
    """
    tenant_id: Optional[str] = None
    collection_id: Optional[str] = None
    permissions: str = "read"
    row_filter_dsl: Optional[str] = None

    def __post_init__(self):
        self._granted = _GRANTS.get(self.permissions, {Permission.READ})
        self._row_pred: Optional[Callable[[Optional[Dict[str, Any]]], bool]] = None
        if self.row_filter_dsl:
            self._row_pred = to_predicate_fn(parse_query(self.row_filter_dsl))

    # ---- operation gates ----

    def allows(self, perm: Permission) -> bool:
        return perm in self._granted

    def require(self, perm: Permission) -> None:
        """Raise AccessDenied if the permission is not granted."""
        if not self.allows(perm):
            raise AccessDenied(f"permission '{perm.value}' not granted by key ({self.permissions})")

    def require_collection(self, collection_id: Optional[str]) -> None:
        """Raise AccessDenied if the key is scoped to a different collection."""
        if self.collection_id is not None and collection_id != self.collection_id:
            raise AccessDenied(
                f"key scoped to collection '{self.collection_id}', not '{collection_id}'"
            )

    # ---- row-level enforcement ----

    def row_allowed(self, metadata: Optional[Dict[str, Any]]) -> bool:
        """True if a row's metadata satisfies the key's row filter (or none set)."""
        if self._row_pred is None:
            return True
        return self._row_pred(metadata)

    def metadata_filter(self) -> Optional[Callable[[Optional[Dict[str, Any]]], bool]]:
        """
        Return a predicate for use as ``HNSWIndex.search(metadata_filter=...)``,
        or None when the key imposes no row filter.
        """
        return self._row_pred

    def filter_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop result rows the key is not allowed to see."""
        if self._row_pred is None:
            return results
        return [r for r in results if self._row_pred(r.get("metadata"))]


def policy_from_validation(validation: Dict[str, Any]) -> RBACPolicy:
    """
    Build an RBACPolicy from an ``APIKeyManager.validate_key`` result.
    Looks for an optional ``row_filter`` field (forward-compatible if the key
    record gains one); absent it, the policy enforces only permission + scope.
    """
    return RBACPolicy(
        tenant_id=validation.get("tenant_id"),
        collection_id=validation.get("collection_id"),
        permissions=validation.get("permissions", "read"),
        row_filter_dsl=validation.get("row_filter"),
    )
