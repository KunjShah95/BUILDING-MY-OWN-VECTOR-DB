import json
import os
import time
import threading
from typing import Dict, Any, List, Optional, Callable

from filelock import FileLock


class WriteAheadLog:
    def __init__(
        self,
        collection_id: str,
        log_dir: str = "wal_logs",
        max_size_mb: int = 64,
        flush_interval: float = 0.5,
        max_rotated_keep: int = 5,
    ):
        self.collection_id = collection_id
        self.log_dir = log_dir
        self.max_size = max_size_mb * 1024 * 1024
        self.flush_interval = flush_interval
        self.max_rotated_keep = max_rotated_keep

        os.makedirs(self.log_dir, exist_ok=True)
        self.wal_path = os.path.join(self.log_dir, f"{collection_id}.wal")
        self.checkpoint_path = os.path.join(self.log_dir, f"{collection_id}.checkpoint")

        self._buffer: List[str] = []
        self._buf_lock = threading.Lock()
        self._stop = threading.Event()

        if self.flush_interval > 0 and not os.environ.get("WAL_NO_BACKGROUND"):
            self._flusher = threading.Thread(target=self._flush_loop, daemon=True, name=f"wal-flush-{collection_id}")
            self._flusher.start()
        else:
            self._flusher = None

    def log_insert(self, vector_id: str, vector: List[float], metadata: Optional[Dict] = None):
        self._enqueue("INSERT", {"id": vector_id, "vec": vector, "meta": metadata})

    def log_delete(self, vector_id: str):
        self._enqueue("DELETE", {"id": vector_id})

    def log_update_metadata(self, vector_id: str, metadata: Dict):
        self._enqueue("UPDATE_META", {"id": vector_id, "meta": metadata})

    def log_batch_insert(self, vectors: List[Dict[str, Any]]):
        batch_data = [
            {"id": v["vector_id"], "vec": v["vector"], "meta": v.get("metadata")}
            for v in vectors
        ]
        self._enqueue("BATCH_INSERT", {"vectors": batch_data})

    def flush(self):
        entries = self._drain_buffer()
        if entries:
            self._write_lines(entries)

    def close(self):
        self._stop.set()
        self.flush()
        if self._flusher and self._flusher.is_alive():
            self._flusher.join(timeout=3)

    def _enqueue(self, operation: str, data: Dict[str, Any]):
        entry = json.dumps({"ts": time.time(), "op": operation, "data": data})
        with self._buf_lock:
            self._buffer.append(entry)

    def _drain_buffer(self) -> List[str]:
        with self._buf_lock:
            batch = self._buffer
            self._buffer = []
        return batch

    def _flush_loop(self):
        while not self._stop.is_set():
            time.sleep(self.flush_interval)
            entries = self._drain_buffer()
            if entries:
                self._write_lines(entries)

    def _write_lines(self, lines: List[str]):
        text = "\n".join(lines) + "\n"
        lock = FileLock(self.wal_path + ".lock")
        with lock:
            with open(self.wal_path, "a") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
        self._maybe_rotate()

    def _maybe_rotate(self):
        if self.max_size <= 0:
            return
        try:
            if os.path.getsize(self.wal_path) >= self.max_size:
                ts = int(time.time())
                rotated = f"{self.wal_path}.{ts}.rotated"
                os.rename(self.wal_path, rotated)
                self._cleanup_rotated()
        except OSError:
            pass

    def _cleanup_rotated(self):
        prefix = f"{self.collection_id}.wal."
        try:
            files = sorted(
                [f for f in os.listdir(self.log_dir) if f.startswith(prefix) and f.endswith(".rotated")],
            )
            while len(files) > self.max_rotated_keep:
                os.remove(os.path.join(self.log_dir, files.pop(0)))
        except OSError:
            pass

    def truncate(self):
        lock = FileLock(self.wal_path + ".lock")
        with lock:
            with open(self.wal_path, "w") as f:
                f.truncate(0)

    def checkpoint(self):
        self.flush()
        lock = FileLock(self.wal_path + ".lock")
        with lock:
            with open(self.checkpoint_path, "w") as f:
                json.dump({"ts": time.time()}, f)
                f.flush()
                os.fsync(f.fileno())
            with open(self.wal_path, "w") as f:
                f.truncate(0)

    def last_checkpoint_ts(self) -> float:
        if not os.path.exists(self.checkpoint_path):
            return 0.0
        try:
            with open(self.checkpoint_path) as f:
                return float(json.load(f).get("ts", 0.0))
        except (json.JSONDecodeError, ValueError, OSError):
            return 0.0

    def read_all(self) -> List[Dict[str, Any]]:
        self.flush()
        if not os.path.exists(self.wal_path):
            return []
        entries = []
        with open(self.wal_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def pending_entries(self) -> List[Dict[str, Any]]:
        cutoff = self.last_checkpoint_ts()
        return [e for e in self.read_all() if e.get("ts", 0.0) > cutoff]

    @staticmethod
    def _index_insert(index, vec, vid, meta):
        if hasattr(index, "insert"):
            index.insert(vec, vid, meta)
        elif hasattr(index, "add"):
            index.add(vec, vid, meta)
        else:
            raise TypeError("index exposes neither insert() nor add()")

    @staticmethod
    def _index_delete(index, vid):
        if hasattr(index, "delete"):
            index.delete(vid)
        elif hasattr(index, "delete_vector"):
            index.delete_vector(vid)
        else:
            raise TypeError("index exposes neither delete() nor delete_vector()")

    def replay(self, index, on_apply: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        applied = {"INSERT": 0, "DELETE": 0, "UPDATE_META": 0}
        skipped = 0

        for entry in self.pending_entries():
            op = entry.get("op")
            data = entry.get("data", {})
            try:
                if op == "BATCH_INSERT":
                    for vec in data.get("vectors", []):
                        self._index_insert(index, vec["vec"], vec["id"], vec.get("meta"))
                        applied["INSERT"] += 1
                elif op == "INSERT":
                    self._index_insert(index, data["vec"], data["id"], data.get("meta"))
                    applied["INSERT"] += 1
                elif op == "DELETE":
                    self._index_delete(index, data["id"])
                    applied["DELETE"] += 1
                elif op == "UPDATE_META":
                    meta_store = getattr(index, "metadata", None)
                    if meta_store is not None and data["id"] in meta_store:
                        meta_store[data["id"]] = data["meta"]
                        applied["UPDATE_META"] += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1
            except (KeyError, ValueError, TypeError):
                skipped += 1
                continue
            if on_apply is not None:
                on_apply(entry)

        return {
            "replayed": sum(applied.values()),
            "by_op": applied,
            "skipped": skipped,
        }
