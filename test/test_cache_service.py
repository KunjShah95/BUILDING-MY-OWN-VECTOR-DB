"""Tests for cache service (works without Redis - falls back gracefully)."""
import pytest
from services.cache_service import CacheService


class TestCacheService:
    def test_available(self):
        svc = CacheService()
        assert isinstance(svc.available, bool)

    def test_cache_search_no_redis(self):
        svc = CacheService()
        svc.cache_search([0.1, 0.2], 5, "test", {"results": []})
        cached = svc.get_cached_search([0.1, 0.2], 5, "test")
        assert cached is None

    def test_cache_embedding_no_redis(self):
        svc = CacheService()
        svc.cache_embedding("hello", "model", [0.1, 0.2])
        cached = svc.get_cached_embedding("hello", "model")
        assert cached is None

    def test_stats_no_redis(self):
        svc = CacheService()
        stats = svc.get_stats()
        assert "available" in stats

    def test_stats_contains_available_key(self):
        svc = CacheService()
        stats = svc.get_stats()
        assert stats["available"] is False

    def test_invalidate_search_cache(self):
        svc = CacheService()
        svc.invalidate_search_cache("test")
        stats = svc.get_stats()
        assert "available" in stats
