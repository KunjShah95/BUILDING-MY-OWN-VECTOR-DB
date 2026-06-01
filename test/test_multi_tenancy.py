"""Tests for multi-tenancy: tenants, per-tenant API keys, rate limiting, and collection scoping."""
import time
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from fastapi import Request
from fastapi.responses import JSONResponse


# ==================== TokenBucket ====================


class TestTokenBucket:
    def test_initial_tokens_full(self):
        from services.tenant_service import TokenBucket
        tb = TokenBucket(rate=10)
        assert tb.tokens == 10

    def test_consume_allowed(self):
        from services.tenant_service import TokenBucket
        tb = TokenBucket(rate=10)
        assert tb.consume() is True
        assert tb.tokens == 9

    def test_consume_exhausted(self):
        from services.tenant_service import TokenBucket
        tb = TokenBucket(rate=3)
        assert tb.consume() is True
        assert tb.consume() is True
        assert tb.consume() is True
        assert tb.consume() is False  # exhausted

    def test_burst_config(self):
        from services.tenant_service import TokenBucket
        tb = TokenBucket(rate=5, burst=10)
        assert tb.tokens == 10  # burst tokens available immediately

    def test_refill_over_time(self):
        from services.tenant_service import TokenBucket
        tb = TokenBucket(rate=60)  # 1 per second
        tb.tokens = 0
        tb.last_refill = time.monotonic() - 2  # 2 seconds elapsed
        # Should have ~2 tokens now
        assert tb.consume() is True
        assert tb.tokens < 2  # consumed 1


# ==================== TenantService ====================


class TestTenantService:
    @pytest.fixture
    def mock_db(self):
        return MagicMock()

    def test_create_tenant(self, mock_db):
        from services.tenant_service import TenantService
        mock_db.query.return_value.filter.return_value.first.return_value = None

        ts = TenantService(db_session=mock_db)
        result = ts.create_tenant(name="Test Corp", tenant_id="test-corp")

        assert result["success"] is True
        assert result["tenant"]["tenant_id"] == "test-corp"
        assert result["tenant"]["name"] == "Test Corp"
        assert result["tenant"]["rate_limit_per_minute"] == 100
        assert result["tenant"]["is_active"] is True
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    def test_create_tenant_generates_id(self, mock_db):
        from services.tenant_service import TenantService
        mock_db.query.return_value.filter.return_value.first.return_value = None

        ts = TenantService(db_session=mock_db)
        result = ts.create_tenant(name="Auto ID")
        assert result["success"] is True
        assert result["tenant"]["tenant_id"].startswith("tenant_")

    def test_create_tenant_duplicate(self, mock_db):
        from services.tenant_service import TenantService
        from database.schema import Tenant

        mock_db.query.return_value.filter.return_value.first.return_value = Tenant(
            tenant_id="dup", name="Existing"
        )

        ts = TenantService(db_session=mock_db)
        result = ts.create_tenant(name="New", tenant_id="dup")
        assert result["success"] is False
        assert "already exists" in result["message"]

    def test_get_tenant_found(self, mock_db):
        from services.tenant_service import TenantService
        from database.schema import Tenant

        mock_db.query.return_value.filter.return_value.first.return_value = Tenant(
            tenant_id="acme", name="Acme Inc", is_active=True, rate_limit_per_minute=200,
        )

        ts = TenantService(db_session=mock_db)
        result = ts.get_tenant("acme")
        assert result["success"] is True
        assert result["tenant"]["tenant_id"] == "acme"

    def test_get_tenant_not_found(self, mock_db):
        from services.tenant_service import TenantService
        mock_db.query.return_value.filter.return_value.first.return_value = None

        ts = TenantService(db_session=mock_db)
        result = ts.get_tenant("nonexistent")
        assert result["success"] is False

    def test_update_tenant_rate_limit(self, mock_db):
        from services.tenant_service import TenantService
        from database.schema import Tenant

        tenant = Tenant(tenant_id="acme", name="Acme Inc", rate_limit_per_minute=100)
        mock_db.query.return_value.filter.return_value.first.return_value = tenant

        ts = TenantService(db_session=mock_db)
        result = ts.update_tenant("acme", rate_limit_per_minute=500)
        assert result["success"] is True
        assert tenant.rate_limit_per_minute == 500

    def test_update_tenant_deactivate(self, mock_db):
        from services.tenant_service import TenantService
        from database.schema import Tenant

        tenant = Tenant(tenant_id="acme", name="Acme Inc", is_active=True)
        mock_db.query.return_value.filter.return_value.first.return_value = tenant

        ts = TenantService(db_session=mock_db)
        result = ts.update_tenant("acme", is_active=False)
        assert result["success"] is True
        assert tenant.is_active is False

    def test_delete_tenant(self, mock_db):
        from services.tenant_service import TenantService
        from database.schema import Tenant

        mock_db.query.return_value.filter.return_value.first.return_value = Tenant(
            tenant_id="acme", name="Acme Inc"
        )

        ts = TenantService(db_session=mock_db)
        result = ts.delete_tenant("acme")
        assert result["success"] is True
        mock_db.delete.assert_called_once()
        mock_db.commit.assert_called_once()

    def test_list_tenants(self, mock_db):
        from services.tenant_service import TenantService
        from database.schema import Tenant

        mock_db.query.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [
            Tenant(tenant_id="a", name="A"),
            Tenant(tenant_id="b", name="B"),
        ]

        ts = TenantService(db_session=mock_db)
        result = ts.list_tenants()
        assert result["success"] is True
        assert result["count"] == 2

    def test_check_rate_limit_allows_first_request(self, mock_db):
        from services.tenant_service import TenantService
        from database.schema import Tenant

        mock_db.query.return_value.filter.return_value.first.return_value = Tenant(
            tenant_id="acme", name="Acme Inc", rate_limit_per_minute=60,
        )

        ts = TenantService(db_session=mock_db)
        assert ts.check_rate_limit("acme") is True

    def test_check_rate_limit_exhausted(self, mock_db):
        from services.tenant_service import TenantService
        from database.schema import Tenant

        mock_db.query.return_value.filter.return_value.first.return_value = Tenant(
            tenant_id="limited", name="Limited", rate_limit_per_minute=2,
        )

        ts = TenantService(db_session=mock_db)
        assert ts.check_rate_limit("limited") is True
        assert ts.check_rate_limit("limited") is True
        assert ts.check_rate_limit("limited") is False  # exhausted


# ==================== APIKeyManager with Tenants ====================


class TestAPIKeyManagerTenants:
    @pytest.fixture
    def mock_db(self):
        return MagicMock()

    def test_create_key_with_tenant(self, mock_db):
        from services.auth_service import APIKeyManager

        mock_db.add.return_value = None
        mock_db.commit.return_value = None

        manager = APIKeyManager(db_session=mock_db)
        result = manager.create_key(
            tenant_id="acme",
            name="my-key",
            permissions="read_only",
        )
        assert result["success"] is True
        assert result["tenant_id"] == "acme"
        assert result["api_key"].startswith("vk_")
        assert result["permissions"] == "read_only"

    def test_create_key_with_collection(self, mock_db):
        from services.auth_service import APIKeyManager

        manager = APIKeyManager(db_session=mock_db)
        result = manager.create_key(
            tenant_id="acme",
            name="coll-key",
            collection_id="my-docs",
        )
        assert result["success"] is True
        assert result["collection_id"] == "my-docs"

    def test_validate_key_success(self, mock_db):
        from services.auth_service import APIKeyManager
        from database.schema import ApiKey, Tenant

        mock_db.query.return_value.filter.return_value.first.side_effect = [
            ApiKey(key_hash="abc", tenant_id="acme", permissions="read_write", name="k1"),
            Tenant(tenant_id="acme", name="Acme Inc", is_active=True),
        ]

        manager = APIKeyManager(db_session=mock_db)
        result = manager.validate_key("test-key-123")
        assert result["success"] is True
        assert result["tenant_id"] == "acme"

    def test_validate_key_inactive_tenant(self, mock_db):
        from services.auth_service import APIKeyManager
        from database.schema import ApiKey, Tenant

        # The validate_key method filters by is_active=True, so a Tenant with
        # is_active=False should NOT be returned by `.first()`. Return None to
        # simulate the filter not matching the inactive tenant.
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            ApiKey(key_hash="abc", tenant_id="disabled", permissions="read_write", name="k1"),
            None,  # no active tenant found
        ]

        manager = APIKeyManager(db_session=mock_db)
        result = manager.validate_key("test-key-123")
        assert result["success"] is False
        assert "disabled" in result["message"].lower()

    def test_validate_key_not_found(self, mock_db):
        from services.auth_service import APIKeyManager

        mock_db.query.return_value.filter.return_value.first.return_value = None

        manager = APIKeyManager(db_session=mock_db)
        result = manager.validate_key("bad-key")
        assert result["success"] is False
        assert "Invalid" in result["message"]

    def test_list_keys_by_tenant(self, mock_db):
        from services.auth_service import APIKeyManager
        from database.schema import ApiKey

        mock_db.query.return_value.filter.return_value.all.return_value = [
            ApiKey(tenant_id="acme", name="k1", permissions="read_write"),
            ApiKey(tenant_id="acme", name="k2", permissions="read_only"),
        ]

        manager = APIKeyManager(db_session=mock_db)
        result = manager.list_keys_by_tenant("acme")
        assert result["success"] is True
        assert len(result["keys"]) == 2


# ==================== Auth Middleware ====================


class TestAuthMiddlewareTenants:
    @pytest.mark.asyncio
    @patch("api.middleware.auth_middleware.get_db")
    @patch("api.middleware.auth_middleware.APIKeyManager.validate_key")
    @patch("api.middleware.auth_middleware.TenantService.check_rate_limit")
    async def test_auth_middleware_attaches_tenant(
        self, mock_rate_limit, mock_validate, mock_get_db
    ):
        from api.middleware.auth_middleware import auth_middleware

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/vectors"
        mock_request.headers = {"X-API-Key": "valid-tenant-key"}
        mock_request.query_params = {}

        mock_validate.return_value = {
            "success": True,
            "tenant_id": "acme",
            "permissions": "read_write",
            "name": "test-key",
        }
        mock_rate_limit.return_value = True

        mock_db_session = MagicMock()
        mock_get_db.return_value = iter([mock_db_session])

        async def fake_call_next(req):
            assert req.state.tenant_id == "acme"
            assert req.state.api_key_info["tenant_id"] == "acme"
            return JSONResponse(status_code=200, content={"ok": True})

        response = await auth_middleware(mock_request, fake_call_next)
        assert response.status_code == 200

    @pytest.mark.asyncio
    @patch("config.settings.get_settings")
    async def test_admin_key_skips_tenant(self, mock_settings):
        from api.middleware.auth_middleware import auth_middleware

        settings = MagicMock()
        settings.API_KEY = "admin-key-123"
        mock_settings.return_value = settings

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/vectors"
        mock_request.headers = {"X-API-Key": "admin-key-123"}
        mock_request.query_params = {}

        async def fake_call_next(req):
            assert req.state.tenant_id is None
            assert req.state.api_key_info["permissions"] == "admin"
            return JSONResponse(status_code=200, content={"ok": True})

        response = await auth_middleware(mock_request, fake_call_next)
        assert response.status_code == 200

    @pytest.mark.asyncio
    @patch("api.middleware.auth_middleware.get_db")
    @patch("api.middleware.auth_middleware.APIKeyManager")
    @patch("api.middleware.auth_middleware.TenantService")
    async def test_auth_middleware_rate_limits_tenant(
        self, mock_ts_class, mock_akm_class, mock_get_db
    ):
        from api.middleware.auth_middleware import auth_middleware

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/vectors"
        mock_request.headers = {"X-API-Key": "valid-key"}
        mock_request.query_params = {}

        # Set up mock chain for validate_key
        mock_akm_instance = mock_akm_class.return_value
        mock_akm_instance.validate_key.return_value = {
            "success": True,
            "tenant_id": "rate-limited",
            "permissions": "read_write",
        }

        # Rate limit returns False → 429
        mock_ts_instance = mock_ts_class.return_value
        mock_ts_instance.check_rate_limit.return_value = False

        mock_db_session = MagicMock()
        mock_get_db.return_value = iter([mock_db_session])

        response = await auth_middleware(mock_request, None)
        assert response.status_code == 429

    @pytest.mark.asyncio
    @patch("config.settings.get_settings")
    async def test_auth_middleware_public_paths_skip_auth(self, mock_settings):
        from api.middleware.auth_middleware import auth_middleware

        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/health"
        mock_request.headers = {}

        async def fake_call_next(req):
            return JSONResponse(status_code=200, content={"ok": True})

        response = await auth_middleware(mock_request, fake_call_next)
        assert response.status_code == 200


# ==================== CollectionService Tenant Scoping ====================


class TestCollectionServiceTenants:
    @pytest.fixture
    def mock_db(self):
        return MagicMock()

    def test_create_collection_with_tenant(self, mock_db):
        from services.collection_service import CollectionService

        mock_db.query.return_value.filter.return_value.first.return_value = None

        cs = CollectionService(db_session=mock_db)
        result = cs.create_collection(
            name="My Collection",
            collection_id="my-coll",
            tenant_id="acme",
        )
        assert result["success"] is True
        # Verify tenant_id was passed to the Collection model
        added = mock_db.add.call_args[0][0]
        assert added.tenant_id == "acme"
        assert added.collection_id == "my-coll"

    def test_get_collection_scoped_by_tenant(self, mock_db):
        from services.collection_service import CollectionService
        from database.schema import Collection

        mock_db.query.return_value.filter.return_value.filter.return_value.first.return_value = (
            Collection(collection_id="my-coll", tenant_id="acme", name="My Coll")
        )

        cs = CollectionService(db_session=mock_db)
        # This should succeed because the collection belongs to acme
        result = cs.get_collection("my-coll", tenant_id="acme")
        assert result["success"] is True

        # When called without tenant_id, should still work (backward compat)
        # Reset mock since the previous call already consumed the result
        mock_db.query.return_value.filter.return_value.first.return_value = (
            Collection(collection_id="my-coll", tenant_id="acme", name="My Coll")
        )
        result2 = cs.get_collection("my-coll")
        assert result2["success"] is True

    def test_get_collection_wrong_tenant_returns_not_found(self, mock_db):
        from services.collection_service import CollectionService

        # No collection matches for "other-tenant"
        mock_db.query.return_value.filter.return_value.filter.return_value.first.return_value = None

        cs = CollectionService(db_session=mock_db)
        result = cs.get_collection("my-coll", tenant_id="other-tenant")
        assert result["success"] is False

    def test_list_collections_scoped_by_tenant(self, mock_db):
        from services.collection_service import CollectionService

        mock_db.query.return_value.order_by.return_value.filter.return_value.offset.return_value.limit.return_value.all.return_value = [
            MagicMock(to_dict=lambda: {"collection_id": "c1", "tenant_id": "acme"}),
        ]

        cs = CollectionService(db_session=mock_db)
        result = cs.list_collections(tenant_id="acme")
        assert result["success"] is True
        assert result["count"] == 1


# ==================== Integration: Tenant → API Key → Scoped Collection ====================


class TestTenantIntegration:
    """Lightweight integration tests using mocked DB."""

    def test_full_tenant_lifecycle(self):
        """Create tenant → create API key → validate key → verify tenant info."""
        from services.auth_service import APIKeyManager
        from services.tenant_service import TenantService

        db = MagicMock()

        # Step 1: Create tenant
        db.query.return_value.filter.return_value.first.return_value = None
        ts = TenantService(db_session=db)
        tenant_result = ts.create_tenant(name="Integration Test", tenant_id="integration-test")
        assert tenant_result["success"] is True

        # Step 2: Create API key for tenant
        db.add.reset_mock()
        db.commit.reset_mock()
        manager = APIKeyManager(db_session=db)
        key_result = manager.create_key(
            tenant_id="integration-test",
            name="integration-key",
        )
        assert key_result["success"] is True
        assert key_result["tenant_id"] == "integration-test"
        raw_key = key_result["api_key"]

        # Step 3: Validate key and verify tenant info
        from database.schema import ApiKey, Tenant
        # First call to validate_key queries ApiKey, second queries Tenant
        db.query.return_value.filter.return_value.first.side_effect = [
            ApiKey(
                key_hash=manager._hash_key(raw_key),
                tenant_id="integration-test",
                permissions="read_write",
                name="integration-key",
            ),
            Tenant(tenant_id="integration-test", name="Integration Test", is_active=True, rate_limit_per_minute=100),
        ]
        validate_result = manager.validate_key(raw_key)
        assert validate_result["success"] is True
        assert validate_result["tenant_id"] == "integration-test"
