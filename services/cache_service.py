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

# =========================================================================
# Async Cache Manager (Ported from ANN Search Engine)
# =========================================================================
import pickle
from typing import Callable
from functools import wraps

class AsyncCacheManager:
    """
    Redis cache manager with serialization support (Async)
    """
    
    def __init__(self):
        self._redis = None
        self._enabled = False
        self._default_ttl = 300  # 5 minutes
    
    async def init(self, redis_url: str = None, enabled: bool = True):
        """Initialize Redis connection"""
        if not enabled:
            self._enabled = False
            logger.info("Async Cache is disabled")
            return
        
        try:
            import redis.asyncio as redis
            self._redis = redis.from_url(redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0"))
            await self._redis.ping()
            self._enabled = True
            logger.info("Async Redis cache initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize async Redis cache: {e}")
            self._enabled = False
            self._redis = None
    
    async def close(self):
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Async Redis cache connection closed")
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = f"{prefix}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self._enabled or not self._redis:
            return None
        
        try:
            data = await self._redis.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"Async cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        if not self._enabled or not self._redis:
            return False
        
        try:
            serialized = pickle.dumps(value)
            await self._redis.setex(key, ttl or self._default_ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Async cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if not self._enabled or not self._redis:
            return False
        
        try:
            await self._redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Async cache delete error: {e}")
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern"""
        if not self._enabled or not self._redis:
            return 0
        
        try:
            keys = await self._redis.keys(pattern)
            if keys:
                return await self._redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Async cache delete pattern error: {e}")
            return 0


# Global cache manager instance
async_cache_manager = AsyncCacheManager()

async def init_async_cache(redis_url: str = None, enabled: bool = True):
    """Initialize global async cache"""
    await async_cache_manager.init(redis_url, enabled)

async def close_async_cache():
    """Close global async cache"""
    await async_cache_manager.close()

def async_cached(prefix: str, ttl: int = 300):
    """
    Decorator to cache async function results
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = async_cache_manager._generate_key(prefix, args, kwargs)
            
            # Try to get from cache
            cached_value = await async_cache_manager.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {prefix}")
                return cached_value
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await async_cache_manager.set(cache_key, result, ttl)
            logger.debug(f"Cache set for {prefix}")
            
            return result
        return async_wrapper
    return decorator

def invalidate_async_cache(pattern: str):
    """
    Decorator to invalidate cache after async function execution
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            await async_cache_manager.delete_pattern(pattern)
            return result
        return async_wrapper
    return decorator
