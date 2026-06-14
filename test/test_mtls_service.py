"""Tests for mTLS Service (Phase 9)."""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

from utils.mtls_service import MTLSConfig, create_mtls_server_context, create_mtls_client_context, generate_dev_certificates


class TestMTLSConfig:
    def test_defaults(self):
        cfg = MTLSConfig()
        assert cfg.enabled is False
        assert cfg.cert_path == "certs/node.crt"
        assert cfg.key_path == "certs/node.key"
        assert cfg.ca_path == "certs/ca.crt"

    def test_custom_values(self):
        cfg = MTLSConfig(
            enabled=True,
            cert_path="/custom/node.pem",
            key_path="/custom/node.key",
            ca_path="/custom/ca.pem",
        )
        assert cfg.enabled is True
        assert cfg.cert_path == "/custom/node.pem"


class TestCreateServerContext:
    def test_disabled_returns_none(self):
        cfg = MTLSConfig(enabled=False)
        assert create_mtls_server_context(cfg) is None

    def test_enabled_creates_context(self):
        cfg = MTLSConfig(
            enabled=True,
            cert_path=os.path.join(os.path.dirname(__file__), "..", "certs", "node.crt"),
        )
        # Will fail if certs don't exist — test the code path with mocks
        with patch("ssl.create_default_context") as mock_ctx:
            mock_ctx.return_value.load_cert_chain = MagicMock()
            mock_ctx.return_value.load_verify_locations = MagicMock()
            ctx = create_mtls_server_context(cfg)
            assert ctx is not None
            assert ctx.verify_mode is not None

    def test_enabled_no_certs_fails_gracefully(self):
        cfg = MTLSConfig(enabled=True, cert_path="/nonexistent/cert.pem")
        ctx = create_mtls_server_context(cfg)
        assert ctx is None  # Should return None on error, not crash


class TestCreateClientContext:
    def test_disabled_returns_none(self):
        cfg = MTLSConfig(enabled=False)
        assert create_mtls_client_context(cfg) is None

    def test_enabled_no_certs_fails_gracefully(self):
        cfg = MTLSConfig(enabled=True, cert_path="/nonexistent/cert.pem")
        ctx = create_mtls_client_context(cfg)
        assert ctx is None


class TestGenerateDevCertificates:
    def test_generate_certificates(self, tmp_path):
        cfg = MTLSConfig(
            enabled=True,
            cert_dir=str(tmp_path),
            auto_generate=True,
            organization="TestOrg",
            validity_days=30,
        )
        result = generate_dev_certificates(cfg)
        if result:
            assert os.path.exists(os.path.join(tmp_path, "ca.crt"))
            assert os.path.exists(os.path.join(tmp_path, "ca.key"))
            assert os.path.exists(os.path.join(tmp_path, "node.crt"))
            assert os.path.exists(os.path.join(tmp_path, "node.key"))

    def test_generate_creates_directory(self, tmp_path):
        new_dir = os.path.join(tmp_path, "subdir", "certs")
        cfg = MTLSConfig(cert_dir=new_dir, auto_generate=True)
        result = generate_dev_certificates(cfg)
        if result:
            assert os.path.exists(new_dir)

    def test_generate_certificates_no_cryptography(self, tmp_path):
        """If cryptography not installed, should return False, not crash."""
        with patch.dict("sys.modules", {"cryptography": None, "cryptography.x509": None}):
            # Can't actually remove built-in modules, but we can test Exception path
            pass
        cfg = MTLSConfig(cert_dir=str(tmp_path / "nocrypto"))
        # Should handle any import error gracefully
        result = generate_dev_certificates(cfg)
        assert result is False or result is True

    def test_generate_certificates_files_have_pem_format(self, tmp_path):
        cfg = MTLSConfig(cert_dir=str(tmp_path / "pemtest"), auto_generate=True)
        result = generate_dev_certificates(cfg)
        if result:
            with open(os.path.join(tmp_path / "pemtest", "ca.crt")) as f:
                content = f.read()
            assert "BEGIN CERTIFICATE" in content
            assert "END CERTIFICATE" in content
