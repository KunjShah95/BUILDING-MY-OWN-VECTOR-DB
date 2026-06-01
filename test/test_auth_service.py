"""Tests for auth service."""
import pytest
from unittest.mock import MagicMock, patch


class TestAPIKeyManager:
    def test_generate_key_format(self):
        from services.auth_service import APIKeyManager
        manager = APIKeyManager(db_session=None)
        key = manager._generate_key()
        assert key.startswith("vk_")
        assert len(key) > 20

    def test_hash_key(self):
        from services.auth_service import APIKeyManager
        manager = APIKeyManager(db_session=None)
        h1 = manager._hash_key("test-key-123")
        h2 = manager._hash_key("test-key-123")
        assert h1 == h2
        assert len(h1) == 64

    @patch("services.auth_service.APIKeyManager.create_key")
    def test_create_key(self, mock_create):
        from services.auth_service import APIKeyManager
        mock_create.return_value = {"success": True, "api_key": "vk_test123"}
        manager = APIKeyManager(db_session=None)
        result = manager.create_key("coll1", "test")
        assert result["success"]
        assert result["api_key"].startswith("vk_")

    def test_list_keys_format(self):
        from services.auth_service import APIKeyManager
        mock_db = MagicMock()
        manager = APIKeyManager(db_session=mock_db)
        result = manager.list_keys("coll1")
        assert result["success"]
        assert "keys" in result
