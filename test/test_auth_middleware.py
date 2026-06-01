"""Tests for auth middleware."""
import pytest
from unittest.mock import MagicMock, patch
from fastapi import Request
from fastapi.responses import JSONResponse


class TestAuthMiddleware:
    def test_get_api_key_from_header(self):
        from api.middleware.auth_middleware import get_api_key
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"X-API-Key": "test-key-123"}
        mock_request.query_params = {}
        result = get_api_key(mock_request)
        assert result == "test-key-123"

    def test_get_api_key_from_bearer(self):
        from api.middleware.auth_middleware import get_api_key
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"Authorization": "Bearer bearer-key-456"}
        mock_request.query_params = {}
        result = get_api_key(mock_request)
        assert result == "bearer-key-456"

    def test_get_api_key_from_query(self):
        from api.middleware.auth_middleware import get_api_key
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.query_params = {"api_key": "query-key-789"}
        result = get_api_key(mock_request)
        assert result == "query-key-789"

    def test_get_api_key_missing(self):
        from api.middleware.auth_middleware import get_api_key
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.query_params = {}
        result = get_api_key(mock_request)
        assert result is None

    def test_public_paths_skip_auth(self):
        from api.middleware.auth_middleware import PUBLIC_PATHS
        assert "/health" in PUBLIC_PATHS
        assert "/docs" in PUBLIC_PATHS
        assert "/ws/health" in PUBLIC_PATHS

    @pytest.mark.asyncio
    async def test_auth_middleware_rejects_no_key(self):
        from api.middleware.auth_middleware import auth_middleware
        mock_request = MagicMock(spec=Request)
        mock_request.url.path = "/vectors"
        mock_request.headers = {}
        mock_request.query_params = {}

        response = await auth_middleware(mock_request, None)
        assert response.status_code == 401
