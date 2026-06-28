"""
Multi-level vector cache hierarchy: L1 RAM → L2 NVMe (mmap) → L3 S3.

L1  In-process LRU dict.  Sub-millisecond.  Bounded by max_l1 entries.
L2  Memory-mapped files on local SSD.  ~0.1 ms.  Bounded by max_l2 entries.
L3  S3-compatible object storage.  ~10 ms.  Unbounded (cold tier).

Promotion:  L2 hit → L1.  L3 hit → L2 → L1.
Eviction:   L1 overflow → demote LRU entry to L2.
            L2 overflow → demote LRU entry to L3 (if S3 configured).

Thread-safe for concurrent reads/writes.
"""

from __future__ import annotations

import hashlib
import logging
import os
import struct
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# L2 mmap file layout: 8-byte float64 timestamp + N × 4-byte float32 values
_HEADER_SIZE = 8  # bytes: float64 unix timestamp


class MultilevelCache:
    """Three-tier cache for vector data and search results.

    Parameters
    ----------
    max_l1 : int
        Max entries in the in-process RAM LRU cache. Default 10_000.
    l2_dir : str
        Directory for L2 mmap/NVMe files. Default "cache_l2".
    max_l2 : int
        Max L2 files to keep. Oldest evicted to L3 or discarded. Default 100_000.
    l3_bucket : str or None
        S3 bucket name for L3 cold storage. None disables L3. Default None.
    l3_prefix : str
        Key prefix for S3 objects. Default "vectordb/cache/".
    dim : int or None
        Vector dimension (needed for L2 binary layout). None = result cache mode
        (values are serialised with pickle instead of raw float32).
    """

    def __init__(
        self,
        max_l1: int = 10_000,
        l2_dir: str = "cache_l2",
        max_l2: int = 100_000,
        l3_bucket: Optional[str] = None,
        l3_prefix: str = "vectordb/cache/",
        dim: Optional[int] = None,
    ):
        self.max_l1 = max_l1
        self.l2_dir = l2_dir
        self.max_l2 = max_l2
        self.l3_bucket = l3_bucket
        self.l3_prefix = l3_prefix
        self.dim = dim

        os.makedirs(l2_dir, exist_ok=True)

        # L1: OrderedDict used as LRU (move-to-end on access)
        self._l1: OrderedDict[str, Any] = OrderedDict()
        self._l1_lock = threading.RLock()

        # L2: set of known keys (actual data on disk)
        self._l2_keys: OrderedDict[str, float] = OrderedDict()  # key -> mtime
        self._l2_lock = threading.RLock()

        # L3: lazy boto3 client
        self._s3: Any = None
        self._s3_lock = threading.Lock()

        # Stats
        self._hits = {"l1": 0, "l2": 0, "l3": 0}
        self._misses = 0
        self._stat_lock = threading.Lock()

    # ------------------------------------------------------------------ public

    def get(self, key: str) -> Optional[Any]:
        # L1
        with self._l1_lock:
            if key in self._l1:
                self._l1.move_to_end(key)
                self._inc_hit("l1")
                return self._l1[key]

        # L2
        value = self._l2_get(key)
        if value is not None:
            self._promote_to_l1(key, value)
            self._inc_hit("l2")
            return value

        # L3
        value = self._l3_get(key)
        if value is not None:
            self._l2_put(key, value)
            self._promote_to_l1(key, value)
            self._inc_hit("l3")
            return value

        with self._stat_lock:
            self._misses += 1
        return None

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self._promote_to_l1(key, value)
        self._l2_put(key, value)

    def delete(self, key: str) -> None:
        with self._l1_lock:
            self._l1.pop(key, None)
        self._l2_delete(key)
        self._l3_delete(key)

    def clear(self) -> None:
        with self._l1_lock:
            self._l1.clear()
        # L2: remove all files
        with self._l2_lock:
            for k in list(self._l2_keys):
                self._l2_delete_file(k)
            self._l2_keys.clear()

    def stats(self) -> Dict[str, Any]:
        with self._stat_lock:
            total_hits = sum(self._hits.values())
            total = total_hits + self._misses
            return {
                "l1_size": len(self._l1),
                "l2_size": len(self._l2_keys),
                "hits": dict(self._hits),
                "misses": self._misses,
                "hit_rate": round(total_hits / total, 4) if total else 0.0,
            }

    # --------------------------------------------------------------- L1 helpers

    def _promote_to_l1(self, key: str, value: Any) -> None:
        with self._l1_lock:
            if key in self._l1:
                self._l1.move_to_end(key)
                self._l1[key] = value
                return
            self._l1[key] = value
            self._l1.move_to_end(key)
            # Evict LRU if over capacity
            while len(self._l1) > self.max_l1:
                evicted_key, evicted_val = self._l1.popitem(last=False)
                # Demote to L2 asynchronously (best-effort; already in L2 from put)
                # No-op here: L2 write happens in put(); eviction just drops from L1

    # --------------------------------------------------------------- L2 helpers

    def _l2_path(self, key: str) -> str:
        hk = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.l2_dir, hk[:2], hk[2:] + ".bin")

    def _l2_put(self, key: str, value: Any) -> None:
        path = self._l2_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            data = self._serialize(value)
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                ts = struct.pack(">d", time.time())
                f.write(ts + data)
            os.replace(tmp, path)
            with self._l2_lock:
                self._l2_keys[key] = time.time()
                self._l2_keys.move_to_end(key)
                # Evict oldest if over max_l2
                while len(self._l2_keys) > self.max_l2:
                    old_key, _ = self._l2_keys.popitem(last=False)
                    self._l3_demote(old_key)
                    self._l2_delete_file(old_key)
        except Exception as e:
            logger.warning("L2 put failed for key %s: %s", key[:16], e)

    def _l2_get(self, key: str) -> Optional[Any]:
        path = self._l2_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                raw = f.read()
            if len(raw) <= _HEADER_SIZE:
                return None
            return self._deserialize(raw[_HEADER_SIZE:])
        except Exception as e:
            logger.warning("L2 get failed for key %s: %s", key[:16], e)
            return None

    def _l2_delete(self, key: str) -> None:
        with self._l2_lock:
            self._l2_keys.pop(key, None)
        self._l2_delete_file(key)

    def _l2_delete_file(self, key: str) -> None:
        path = self._l2_path(key)
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

    # --------------------------------------------------------------- L3 helpers

    def _s3_client(self):
        if self._s3 is not None:
            return self._s3
        if not self.l3_bucket:
            return None
        with self._s3_lock:
            if self._s3 is None:
                try:
                    import boto3
                    self._s3 = boto3.client("s3")
                except Exception as e:
                    logger.warning("S3 init failed: %s", e)
            return self._s3

    def _l3_key(self, key: str) -> str:
        return self.l3_prefix + hashlib.md5(key.encode()).hexdigest()

    def _l3_get(self, key: str) -> Optional[Any]:
        s3 = self._s3_client()
        if s3 is None:
            return None
        try:
            obj = s3.get_object(Bucket=self.l3_bucket, Key=self._l3_key(key))
            return self._deserialize(obj["Body"].read())
        except Exception:
            return None

    def _l3_demote(self, key: str) -> None:
        """Demote an L2 entry to L3 before eviction."""
        s3 = self._s3_client()
        if s3 is None:
            return
        value = self._l2_get(key)
        if value is None:
            return
        try:
            s3.put_object(
                Bucket=self.l3_bucket,
                Key=self._l3_key(key),
                Body=self._serialize(value),
            )
        except Exception as e:
            logger.warning("L3 demote failed for key %s: %s", key[:16], e)

    def _l3_delete(self, key: str) -> None:
        s3 = self._s3_client()
        if s3 is None:
            return
        try:
            s3.delete_object(Bucket=self.l3_bucket, Key=self._l3_key(key))
        except Exception:
            pass

    # --------------------------------------------------------------- serde

    def _serialize(self, value: Any) -> bytes:
        if self.dim is not None and isinstance(value, np.ndarray):
            return value.astype(np.float32).tobytes()
        import pickle
        return pickle.dumps(value, protocol=4)

    def _deserialize(self, data: bytes) -> Any:
        if self.dim is not None:
            arr = np.frombuffer(data, dtype=np.float32)
            if arr.shape == (self.dim,):
                return arr.copy()
        try:
            import pickle
            return pickle.loads(data)
        except Exception:
            return None

    # --------------------------------------------------------------- stats util

    def _inc_hit(self, tier: str) -> None:
        with self._stat_lock:
            self._hits[tier] += 1
