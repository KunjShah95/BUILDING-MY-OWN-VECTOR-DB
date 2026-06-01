"""Tests for WebSocket search (connection-level only)."""
import pytest


class TestWebSocketSearch:
    def test_ws_router_exists(self):
        from api.routers.ws_search import router
        routes = [r.path for r in router.routes]
        assert "/ws/search/{collection_id}" in routes
        assert "/ws/health" in routes

    def test_ws_route_details(self):
        from api.routers.ws_search import router
        for route in router.routes:
            if hasattr(route, "path"):
                assert route.path.startswith("/ws/")
