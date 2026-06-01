from typing import Optional, Any, Dict, List
import json
import hashlib
import os
import time
import logging

logger = logging.getLogger(__name__)

_redis_client = None


def _get_redis():
    """Get Redis client (lazy init)."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    redis_url = os.getenv("REDIS_URL", "")
    if not redis_url:
        return None

    try:
        import redis
        _redis_client = redis.from_url(redis_url, decode_responses=True)
        _redis_client.ping()
        logger.info("Redis connected")
        return _redis_client
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}")
        return None


def _make_key(prefix: str, *parts) -> str:
    key = ":".join(str(p) for p in parts)
    return f"vectordb:{prefix}:{hashlib.md5(key.encode()).hexdigest()}"


class CacheService:
    """Caching for search results and embeddings."""

    DEFAULT_TTL = 300

    def __init__(self):
        self._client = _get_redis()

    @property
    def available(self) -> bool:
        return self._client is not None

    def cache_search(self, query_vector: list, k: int, collection_id: str,
                     results: Dict[str, Any], ttl: int = DEFAULT_TTL,
                     filters: Optional[Dict] = None):
        """Cache search results keyed by query hash + k + collection."""
        if not self.available:
            return
        parts = [collection_id, str(k), str(hash(tuple(query_vector)))]
        if filters:
            parts.append(str(hash(json.dumps(filters, sort_keys=True))))
        key = _make_key("search", *parts)
        try:
            self._client.setex(key, ttl, json.dumps(results))
        except Exception as e:
            logger.debug(f"Cache set failed: {e}")

    def get_cached_search(self, query_vector: list, k: int,
                          collection_id: str,
                          filters: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Get cached search results."""
        if not self.available:
            return None
        parts = [collection_id, str(k), str(hash(tuple(query_vector)))]
        if filters:
            parts.append(str(hash(json.dumps(filters, sort_keys=True))))
        key = _make_key("search", *parts)
        try:
            data = self._client.get(key)
            if data:
                return json.loads(data)
        except Exception:
            pass
        return None

    def cache_embedding(self, text: str, model: str,
                        vector: List[float], ttl: int = 3600):
        """Cache embedding vectors."""
        if not self.available:
            return
        key = _make_key("embed", model, hashlib.md5(text.encode()).hexdigest())
        try:
            self._client.setex(key, ttl, json.dumps(vector))
        except Exception as e:
            logger.debug(f"Cache set failed: {e}")

    def get_cached_embedding(self, text: str, model: str) -> Optional[List[float]]:
        if not self.available:
            return None
        key = _make_key("embed", model, hashlib.md5(text.encode()).hexdigest())
        try:
            data = self._client.get(key)
            if data:
                return json.loads(data)
        except Exception:
            pass
        return None

    def invalidate_search_cache(self, collection_id: str):
        """Invalidate all cached searches for a collection."""
        if not self.available:
            return
        pattern = f"vectordb:search:{collection_id}:*"
        try:
            cursor = 0
            while True:
                cursor, keys = self._client.scan(cursor=cursor, match=pattern, count=100)
                if keys:
                    self._client.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.debug(f"Cache invalidation failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        if not self.available:
            return {"available": False}
        try:
            info = self._client.info()
            return {
                "available": True,
                "used_memory": info.get("used_memory_human", "unknown"),
                "uptime_days": info.get("uptime_in_days", 0),
                "total_keys": self._client.dbsize(),
            }
        except Exception as e:
            return {"available": False, "error": str(e)}
