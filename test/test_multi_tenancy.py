"""Tests for multi-tenancy: Tenant CRUD, rate limiting, auth, collection scoping."""

import time
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON

from database.schema import Base, Tenant, ApiKey, Collection
from services.auth_service import APIKeyManager
from services.tenant_service import TenantService, TokenBucket
from services.collection_service import CollectionService


# Make ARRAY type compile to JSON on SQLite so we can create all tables
from sqlalchemy import ARRAY as _SQLA_ARRAY

@compiles(_SQLA_ARRAY, "sqlite")
def _compile_array_sqlite(type_, compiler, **kw):
    return compiler.process(SQLiteJSON())


# ---------------------------------------------------------------------------
# SQLite in-memory DB fixture (no PostgreSQL dependency)
# ---------------------------------------------------------------------------

@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()


# ---------------------------------------------------------------------------
# TokenBucket
# ---------------------------------------------------------------------------

class TestTokenBucket:
    def test_allow_within_limit(self):
        bucket = TokenBucket(rate=5, per=60.0)
        for _ in range(5):
            assert bucket.consume()

    def test_deny_when_exhausted(self):
        bucket = TokenBucket(rate=3, per=60.0)
        for _ in range(3):
            bucket.consume()
        assert not bucket.consume()

    def test_refill_over_time(self):
        bucket = TokenBucket(rate=2, per=60.0)  # 2 tokens per 60 sec = 1 token/30sec
        bucket.consume()
        bucket.consume()
        assert not bucket.consume()
        # Simulate time passing
        bucket.last_refill = time.monotonic() - 31  # > 30 sec
        assert bucket.consume()

    def test_allow_returns_true_when_below_capacity(self):
        bucket = TokenBucket(rate=10, per=60.0)
        bucket.tokens = 5.0
        assert bucket.consume() is True

    def test_refill_does_not_exceed_capacity(self):
        bucket = TokenBucket(rate=5, per=60.0)
        bucket.tokens = 4.0
        bucket.last_refill = time.monotonic()
        bucket.consume()
        # tokens should be approx 3.0 (tiny refill from elapsed time)
        assert bucket.tokens == pytest.approx(3.0, abs=0.01)

    def test_negative_tokens(self):
        bucket = TokenBucket(rate=5, per=60.0)
        bucket.tokens = -1.0
        assert not bucket.consume()


# ---------------------------------------------------------------------------
# TenantService
# ---------------------------------------------------------------------------

class TestTenantService:
    """Uses SQLite in-memory database."""

    @pytest.fixture(autouse=True)
    def _db(self, db_session: Session):
        self.db = db_session
        self.svc = TenantService(db_session=self.db)
        self.col_svc = CollectionService(db_session=self.db)

    def test_create_tenant(self):
        tenant = self.svc.create_tenant("test-tenant", "test-tenant", rate_limit_per_minute=30)
        assert tenant["success"] is True
        assert tenant["tenant"]["tenant_id"] == "test-tenant"
        assert tenant["tenant"]["name"] == "test-tenant"
        assert tenant["tenant"]["is_active"] is True
        assert tenant["tenant"]["rate_limit_per_minute"] == 30

    def test_create_duplicate_tenant(self):
        self.svc.create_tenant("dup-tenant", "dup-tenant")
        result = self.svc.create_tenant("dup-tenant", "dup-tenant")
        assert result["success"] is False
        assert "already exists" in result["message"]

    def test_get_tenant(self):
        self.svc.create_tenant("get-me", "get-me")
        result = self.svc.get_tenant("get-me")
        assert result["success"] is True
        assert result["tenant"]["tenant_id"] == "get-me"

    def test_get_nonexistent_tenant(self):
        result = self.svc.get_tenant("nobody")
        assert result["success"] is False

    def test_list_tenants(self):
        self.svc.create_tenant("list-a", "list-a")
        self.svc.create_tenant("list-b", "list-b")
        result = self.svc.list_tenants()
        assert result["success"] is True
        ids = [t["tenant_id"] for t in result["tenants"]]
        assert "list-a" in ids
        assert "list-b" in ids

    def test_update_tenant(self):
        self.svc.create_tenant("updatable", "updatable")
        result = self.svc.update_tenant("updatable", name="Updated", rate_limit_per_minute=60)
        assert result["success"] is True
        assert result["tenant"]["name"] == "Updated"
        assert result["tenant"]["rate_limit_per_minute"] == 60

    def test_update_tenant_deactivate(self):
        self.svc.create_tenant("to-deactivate", "to-deactivate")
        result = self.svc.update_tenant("to-deactivate", is_active=False)
        assert result["success"] is True
        assert result["tenant"]["is_active"] is False

    def test_update_nonexistent_tenant(self):
        result = self.svc.update_tenant("ghost", name="Nope")
        assert result["success"] is False

    def test_delete_tenant(self):
        self.svc.create_tenant("delete-me", "delete-me")
        result = self.svc.delete_tenant("delete-me")
        assert result["success"] is True
        get_result = self.svc.get_tenant("delete-me")
        assert get_result["success"] is False

    def test_delete_nonexistent_tenant(self):
        result = self.svc.delete_tenant("ghost")
        assert result["success"] is False

    # -- Rate limiting via TenantService --

    def test_check_rate_limit_within_limit(self):
        self.svc.create_tenant("rl-ok", "rl-ok", rate_limit_per_minute=10)
        for _ in range(10):
            assert self.svc.check_rate_limit("rl-ok") is True

    def test_check_rate_limit_exceeded(self):
        self.svc.create_tenant("rl-burst", "rl-burst", rate_limit_per_minute=3)
        for _ in range(3):
            self.svc.check_rate_limit("rl-burst")
        assert self.svc.check_rate_limit("rl-burst") is False

    def test_check_rate_limit_nonexistent_tenant(self):
        # Non-existent tenants get a default rate limit bucket
        assert self.svc.check_rate_limit("ghost") is True

    def test_check_rate_limit_default_to_100(self):
        """Tenants created without explicit rate_limit get the default of 100 RPM."""
        self.svc.create_tenant("default-rl", "default-rl")
        for _ in range(100):
            assert self.svc.check_rate_limit("default-rl") is True
        assert self.svc.check_rate_limit("default-rl") is False

    def test_check_rate_limit_clears_bucket_on_tenant_delete(self):
        self.svc.create_tenant("rl-cleanup", "rl-cleanup", rate_limit_per_minute=2)
        self.svc.check_rate_limit("rl-cleanup")
        self.svc.check_rate_limit("rl-cleanup")
        self.svc.delete_tenant("rl-cleanup")
        # deleted tenant gets a fresh bucket with default rate
        assert self.svc.check_rate_limit("rl-cleanup") is True


# ---------------------------------------------------------------------------
# APIKeyManager — tenant validation
# ---------------------------------------------------------------------------

class TestAPIKeyManager:
    @pytest.fixture(autouse=True)
    def _db(self, db_session: Session):
        self.db = db_session
        self.ts = TenantService(db_session=self.db)
        self.mgr = APIKeyManager(db_session=self.db)

    def test_validate_key_tenant_active(self):
        """A valid key bound to an active tenant should succeed."""
        self.ts.create_tenant("active-tenant", "active-tenant")
        create_result = self.mgr.create_key(
            name="my-key", tenant_id="active-tenant"
        )
        raw_key = create_result["api_key"]
        result = self.mgr.validate_key(raw_key)
        assert result["success"] is True
        assert result["tenant_id"] == "active-tenant"

    def test_validate_key_inactive_tenant(self):
        """A key bound to an inactive tenant should be rejected."""
        self.ts.create_tenant("inactive-tenant", "inactive-tenant")
        create_result = self.mgr.create_key(
            name="key-for-disabled", tenant_id="inactive-tenant"
        )
        raw_key = create_result["api_key"]
        self.ts.update_tenant("inactive-tenant", is_active=False)
        result = self.mgr.validate_key(raw_key)
        assert result["success"] is False
        # inactive tenant should not return tenant_id
        assert result.get("tenant_id") is None

    def test_validate_key_no_tenant(self):
        """A key without a tenant (admin-like) should still succeed."""
        create_result = self.mgr.create_key(name="no-tenant-key")
        raw_key = create_result["api_key"]
        result = self.mgr.validate_key(raw_key)
        assert result["success"] is True
        assert result.get("tenant_id") is None

    def test_create_key_with_collection_binding(self):
        create_result = self.mgr.create_key(
            name="coll-bound",
            collection_id="my-collection",
        )
        raw_key = create_result["api_key"]
        result = self.mgr.validate_key(raw_key)
        assert result["success"] is True

    def test_create_key_missing_tenant(self):
        """Creating a key for a non-existent tenant still succeeds (existence validated at auth time)."""
        result = self.mgr.create_key(name="ghost-key", tenant_id="ghost-tenant")
        assert result["success"] is True
        assert result.get("tenant_id") == "ghost-tenant"

    def test_list_keys_respects_tenant_id(self):
        self.ts.create_tenant("list-tenant", "list-tenant")
        self.mgr.create_key(name="k1", tenant_id="list-tenant")
        self.mgr.create_key(name="k2", tenant_id="list-tenant")
        self.mgr.create_key(name="k3")  # no tenant

        keys_t = self.mgr.list_keys_by_tenant("list-tenant")
        assert keys_t["success"] is True
        assert len(keys_t["keys"]) == 2

        # list all keys (no tenant filter available via APIKeyManager)
        # Only verify tenant-specific list works
        assert len(keys_t["keys"]) == 2


# ---------------------------------------------------------------------------
# CollectionService — tenant scoping
# ---------------------------------------------------------------------------

class TestCollectionServiceTenantScoped:
    @pytest.fixture(autouse=True)
    def _db(self, db_session: Session):
        self.db = db_session
        self.ts = TenantService(db_session=self.db)
        self.svc = CollectionService(db_session=self.db)

    def test_create_collection_with_tenant(self):
        self.ts.create_tenant("coll-tenant", "coll-tenant")
        result = self.svc.create_collection(
            "my-collection",
            dimension=128,
            tenant_id="coll-tenant",
        )
        assert result["success"] is True
        assert result["collection"]["tenant_id"] == "coll-tenant"

    def test_get_collection_no_tenant(self):
        """Getting a collection without tenant_id should still work (backward compat)."""
        self.svc.create_collection(
            "backward-compat-col",
            collection_id="backward-compat-col",
            dimension=128,
        )
        result = self.svc.get_collection("backward-compat-col")
        assert result["success"] is True

    def test_get_collection_wrong_tenant_blocked(self):
        """A collection created by tenant A should not be visible to tenant B."""
        self.ts.create_tenant("tenant-a", "tenant-a")
        self.ts.create_tenant("tenant-b", "tenant-b")
        self.svc.create_collection(
            "secret", collection_id="secret", dimension=128, tenant_id="tenant-a"
        )
        result = self.svc.get_collection("secret", tenant_id="tenant-b")
        assert result["success"] is False

    def test_list_collections_filters_by_tenant(self):
        self.ts.create_tenant("list-a", "list-a")
        self.ts.create_tenant("list-b", "list-b")
        self.svc.create_collection("coll-a1", collection_id="coll-a1", dimension=128, tenant_id="list-a")
        self.svc.create_collection("coll-a2", collection_id="coll-a2", dimension=128, tenant_id="list-a")
        self.svc.create_collection("coll-b1", collection_id="coll-b1", dimension=128, tenant_id="list-b")

        result_a = self.svc.list_collections(tenant_id="list-a")
        assert result_a["success"] is True
        assert len(result_a["collections"]) == 2

        result_b = self.svc.list_collections(tenant_id="list-b")
        assert result_b["success"] is True
        assert len(result_b["collections"]) == 1

    def test_list_collections_no_tenant_returns_all(self):
        self.ts.create_tenant("all-tenant", "all-tenant")
        self.svc.create_collection("c1", collection_id="c1", dimension=128, tenant_id="all-tenant")
        self.svc.create_collection("c2", collection_id="c2", dimension=128)  # no tenant
        result = self.svc.list_collections()
        assert result["success"] is True
        all_ids = [c["collection_id"] for c in result["collections"]]
        assert "c1" in all_ids
        assert "c2" in all_ids

    def test_delete_collection_scoped_by_tenant(self):
        self.ts.create_tenant("del-tenant", "del-tenant")
        self.svc.create_collection("to-delete", collection_id="to-delete", dimension=128, tenant_id="del-tenant")
        result = self.svc.delete_collection("to-delete", tenant_id="del-tenant")
        assert result["success"] is True

    def test_delete_collection_wrong_tenant_blocked(self):
        self.ts.create_tenant("del-a", "del-a")
        self.ts.create_tenant("del-b", "del-b")
        self.svc.create_collection(
            "shared", collection_id="shared", dimension=128, tenant_id="del-a"
        )
        result = self.svc.delete_collection("shared", tenant_id="del-b")
        assert result["success"] is False

    def test_validate_vector_dimension_with_tenant(self):
        self.ts.create_tenant("dim-tenant", "dim-tenant")
        self.svc.create_collection("dim-coll", collection_id="dim-coll", dimension=128, tenant_id="dim-tenant")
        result = self.svc.validate_vector_dimension(
            "dim-coll", [0.1] * 128, tenant_id="dim-tenant"
        )
        assert result["success"] is True

    def test_validate_vector_dimension_wrong_tenant_blocked(self):
        self.ts.create_tenant("dim-a", "dim-a")
        self.ts.create_tenant("dim-b", "dim-b")
        self.svc.create_collection("dim-coll", collection_id="dim-coll", dimension=128, tenant_id="dim-a")
        result = self.svc.validate_vector_dimension(
            "dim-coll", [0.1] * 128, tenant_id="dim-b"
        )
        assert result["success"] is False

    def test_collection_points_to_tenant(self):
        """Verify collection.tenant_id is stored correctly."""
        self.ts.create_tenant("fk-tenant", "fk-tenant")
        self.svc.create_collection("fk-coll", collection_id="fk-coll", dimension=128, tenant_id="fk-tenant")
        rec = self.db.query(Collection).filter(
            Collection.collection_id == "fk-coll"
        ).first()
        assert rec is not None
        assert rec.tenant_id == "fk-tenant"


# ---------------------------------------------------------------------------
# admin_required dependency — permission enforcement
# ---------------------------------------------------------------------------

class TestAdminRequired:
    """Test the admin_required permission check used on tenant management endpoints."""

    @staticmethod
    def _make_request(api_key_info):
        """Helper: build a mock request with the given api_key_info on state."""
        request = MagicMock()
        # Set up state as a nested object so getattr(request.state, ...) works
        class State:
            pass
        state = State()
        state.api_key_info = api_key_info
        request.state = state
        return request

    def test_admin_allows_master_key_permissions(self):
        """Admin permission set in request state should pass."""
        from api.routers.tenants import admin_required
        request = self._make_request({"success": True, "permissions": "admin"})
        import asyncio
        result = asyncio.run(admin_required(request))
        assert result is True

    def test_admin_rejects_read_write_permissions(self):
        """read_write permissions should be rejected with 403."""
        from api.routers.tenants import admin_required
        from fastapi import HTTPException
        request = self._make_request({"success": True, "permissions": "read_write"})
        import asyncio
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(admin_required(request))
        assert exc_info.value.status_code == 403
        assert "Admin access required" in str(exc_info.value.detail["message"])

    def test_admin_rejects_read_only_permissions(self):
        """read_only permissions should be rejected with 403."""
        from api.routers.tenants import admin_required
        from fastapi import HTTPException
        request = self._make_request({"success": True, "permissions": "read_only"})
        import asyncio
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(admin_required(request))
        assert exc_info.value.status_code == 403

    def test_admin_rejects_missing_permissions_key(self):
        """api_key_info without a 'permissions' key should be rejected."""
        from api.routers.tenants import admin_required
        from fastapi import HTTPException
        request = self._make_request({"success": True})
        import asyncio
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(admin_required(request))
        assert exc_info.value.status_code == 403

    def test_admin_rejects_none_api_key_info(self):
        """api_key_info set to None should be rejected."""
        from api.routers.tenants import admin_required
        from fastapi import HTTPException
        request = self._make_request(None)
        import asyncio
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(admin_required(request))
        assert exc_info.value.status_code == 403

    def test_admin_rejects_no_state_attribute(self):
        """Request without api_key_info on state should be rejected."""
        from api.routers.tenants import admin_required
        from fastapi import HTTPException
        request = MagicMock()
        # Don't set state at all — getattr(request, "state", None) would return a MagicMock
        # So we set state to an object without api_key_info
        class BareState:
            pass
        request.state = BareState()
        import asyncio
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(admin_required(request))
        assert exc_info.value.status_code == 403

    def test_admin_rejects_non_dict_info(self):
        """api_key_info that is not a dict should be rejected."""
        from api.routers.tenants import admin_required
        from fastapi import HTTPException
        request = self._make_request("not-a-dict")
        import asyncio
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(admin_required(request))
        assert exc_info.value.status_code == 403
