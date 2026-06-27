"""Tests for Phase 12: Enterprise Hardening."""
import json
import os
import tempfile
import time
import pytest

# ============================================================
# 1. Tenant Isolation
# ============================================================

class TestTenantIsolation:
    def test_get_tenant_prefix(self):
        from services.tenant_isolation import TenantIsolation
        ti = TenantIsolation()
        assert ti.get_tenant_prefix("acme") == "tenant_acme_"

    def test_namespaced_collection(self):
        from services.tenant_isolation import TenantIsolation
        ti = TenantIsolation()
        assert ti.namespaced_collection("acme", "products") == "tenant_acme_products"

    def test_namespaced_collection_idempotent(self):
        from services.tenant_isolation import TenantIsolation
        ti = TenantIsolation()
        already = "tenant_acme_products"
        assert ti.namespaced_collection("acme", already) == already

    def test_strip_prefix(self):
        from services.tenant_isolation import TenantIsolation
        ti = TenantIsolation()
        assert ti.strip_prefix("acme", "tenant_acme_products") == "products"

    def test_list_tenant_collections(self, tmp_path):
        from services.tenant_isolation import TenantIsolation
        # Create fake collection dirs
        (tmp_path / "tenant_acme_col1").mkdir()
        (tmp_path / "tenant_acme_col2").mkdir()
        (tmp_path / "tenant_other_col1").mkdir()

        ti = TenantIsolation(index_dir=str(tmp_path))
        cols = ti.list_tenant_collections("acme")
        assert "tenant_acme_col1" in cols
        assert "tenant_acme_col2" in cols
        assert "tenant_other_col1" not in cols

    def test_purge_tenant(self, tmp_path):
        from services.tenant_isolation import TenantIsolation
        (tmp_path / "tenant_acme_col1").mkdir()
        (tmp_path / "tenant_acme_col2").mkdir()
        (tmp_path / "tenant_other_col1").mkdir()

        ti = TenantIsolation(index_dir=str(tmp_path))
        removed = ti.purge_tenant("acme")
        assert removed == 2
        assert not (tmp_path / "tenant_acme_col1").exists()
        assert (tmp_path / "tenant_other_col1").exists()


# ============================================================
# 2. Quota Enforcer
# ============================================================

class TestQuotaEnforcer:
    def test_vector_quota_enforced(self):
        from services.tenant_isolation import TenantQuota, QuotaEnforcer, QuotaExceededError
        quota = TenantQuota(max_vectors=5, max_collections=10, max_qps=100)
        enforcer = QuotaEnforcer(quota)
        enforcer.check_and_increment(vectors=5)
        with pytest.raises(QuotaExceededError):
            enforcer.check_and_increment(vectors=1)

    def test_collection_quota_enforced(self):
        from services.tenant_isolation import TenantQuota, QuotaEnforcer, QuotaExceededError
        quota = TenantQuota(max_vectors=100, max_collections=2, max_qps=100)
        enforcer = QuotaEnforcer(quota)
        enforcer.check_and_increment(collections=2)
        with pytest.raises(QuotaExceededError):
            enforcer.check_and_increment(collections=1)

    def test_reset_counters(self):
        from services.tenant_isolation import TenantQuota, QuotaEnforcer
        quota = TenantQuota(max_vectors=5, max_collections=10, max_qps=100)
        enforcer = QuotaEnforcer(quota)
        enforcer.check_and_increment(vectors=5)
        enforcer.reset_counters()
        # Should not raise after reset
        enforcer.check_and_increment(vectors=3)
        assert enforcer.vector_count == 3

    def test_qps_quota_enforced(self):
        from services.tenant_isolation import TenantQuota, QuotaEnforcer, QuotaExceededError
        quota = TenantQuota(max_vectors=100, max_collections=10, max_qps=2)
        enforcer = QuotaEnforcer(quota)
        enforcer.check_and_increment(check_qps=True)
        enforcer.check_and_increment(check_qps=True)
        with pytest.raises(QuotaExceededError):
            enforcer.check_and_increment(check_qps=True)


# ============================================================
# 3. Audit Service
# ============================================================

class TestAuditService:
    def _make_service(self, tmp_path):
        from services.audit_service import AuditService
        return AuditService(log_path=str(tmp_path / "audit.log"))

    def test_log_event(self, tmp_path):
        svc = self._make_service(tmp_path)
        entry = svc.log_event("tenant1", "user1", "READ", "/collections/c1")
        assert entry["tenant_id"] == "tenant1"
        assert entry["user_id"] == "user1"
        assert entry["action"] == "READ"
        assert "timestamp" in entry

    def test_export_soc2_filters_tenant(self, tmp_path):
        svc = self._make_service(tmp_path)
        svc.log_event("tenant1", "user1", "READ", "/a")
        svc.log_event("tenant2", "user2", "WRITE", "/b")
        results = svc.export_soc2("tenant1")
        assert len(results) == 1
        assert results[0]["tenant_id"] == "tenant1"

    def test_export_soc2_date_filter(self, tmp_path):
        svc = self._make_service(tmp_path)
        svc.log_event("t1", "u1", "READ", "/a")
        results = svc.export_soc2("t1", start_date="2000-01-01", end_date="2099-12-31")
        assert len(results) == 1

    def test_export_gdpr(self, tmp_path):
        svc = self._make_service(tmp_path)
        svc.log_event("t1", "alice", "READ", "/a")
        svc.log_event("t1", "bob", "WRITE", "/b")
        svc.log_event("t2", "alice", "DELETE", "/c")
        results = svc.export_gdpr("alice")
        assert len(results) == 2
        assert all(e["user_id"] == "alice" for e in results)

    def test_purge_user_data_redacts_pii(self, tmp_path):
        svc = self._make_service(tmp_path)
        svc.log_event("t1", "alice", "READ", "/a", metadata={"email": "alice@example.com"})
        svc.log_event("t1", "bob", "WRITE", "/b")
        count = svc.purge_user_data("alice")
        assert count == 1
        # alice's entry should be redacted
        entries = svc.export_gdpr("[REDACTED]")
        assert len(entries) == 1
        assert entries[0]["user_id"] == "[REDACTED]"
        assert entries[0]["metadata"]["email"] == "[REDACTED]"
        # bob's entry untouched
        bob_entries = svc.export_gdpr("bob")
        assert len(bob_entries) == 1


# ============================================================
# 4. SSO Service
# ============================================================

class TestSSOService:
    def test_get_auth_url_contains_pkce(self):
        from services.sso_service import OIDCConfig, OIDCProvider
        config = OIDCConfig(
            issuer="https://idp.example.com",
            client_id="client123",
            client_secret="secret",
            redirect_uri="https://app.example.com/callback",
        )
        provider = OIDCProvider(config)
        url, code_verifier, state = provider.get_auth_url()
        assert "code_challenge" in url
        assert "code_challenge_method=S256" in url
        assert len(code_verifier) > 0
        assert len(state) > 0

    def test_sso_session_roundtrip(self):
        from services.sso_service import SSOSession, SSOSessionData
        data = SSOSessionData(
            user_id="u1",
            email="u1@example.com",
            roles=["admin"],
            expiry=time.time() + 3600,
        )
        sid = SSOSession.generate_session_id()
        SSOSession.save(sid, data)
        retrieved = SSOSession.get(sid)
        assert retrieved is not None
        assert retrieved.email == "u1@example.com"
        SSOSession.delete(sid)
        assert SSOSession.get(sid) is None

    def test_sso_session_expired(self):
        from services.sso_service import SSOSession, SSOSessionData
        data = SSOSessionData(
            user_id="u2",
            email="u2@example.com",
            roles=[],
            expiry=time.time() - 1,  # already expired
        )
        sid = SSOSession.generate_session_id()
        SSOSession.save(sid, data)
        assert SSOSession.get(sid) is None

    def test_verify_token_without_pyjwt(self, monkeypatch):
        """Without PyJWT, verify_token should fall back to base64 decode."""
        import services.sso_service as sso_mod
        monkeypatch.setattr(sso_mod, "_JWT_AVAILABLE", False)

        import base64, json as _json
        payload = {"sub": "user99", "email": "test@example.com"}
        encoded = base64.urlsafe_b64encode(_json.dumps(payload).encode()).rstrip(b"=").decode()
        fake_jwt = f"header.{encoded}.sig"

        from services.sso_service import OIDCConfig, OIDCProvider
        config = OIDCConfig(
            issuer="https://idp.example.com",
            client_id="client",
            client_secret="s",
            redirect_uri="https://app.example.com/cb",
        )
        import asyncio
        provider = OIDCProvider(config)
        claims = asyncio.get_event_loop().run_until_complete(provider.verify_token(fake_jwt))
        assert claims["sub"] == "user99"


# ============================================================
# 5. PgWire Server (unit-level, no TCP connection)
# ============================================================

class TestPgvectorCompat:
    def test_build_row_description(self):
        from services.pgvector_compat import _build_row_description
        data = _build_row_description(["id", "score"])
        assert data[0:1] == b"T"

    def test_build_data_row(self):
        from services.pgvector_compat import _build_data_row
        data = _build_data_row(["vec1", "0.95", "{}"])
        assert data[0:1] == b"D"

    def test_build_error_response(self):
        from services.pgvector_compat import _build_error_response
        data = _build_error_response("something went wrong")
        assert data[0:1] == b"E"
        assert b"something went wrong" in data

    def test_parse_vector(self):
        from services.pgvector_compat import _parse_vector
        vec = _parse_vector("[1.0, 2.0, 3.0]")
        assert vec == [1.0, 2.0, 3.0]

    def test_pgwire_server_instantiation(self):
        from services.pgvector_compat import PgWireServer
        server = PgWireServer(vector_service=None)
        assert server._server is None

    def test_command_complete(self):
        from services.pgvector_compat import _build_command_complete
        data = _build_command_complete("SELECT 5")
        assert data[0:1] == b"C"
        assert b"SELECT 5" in data
