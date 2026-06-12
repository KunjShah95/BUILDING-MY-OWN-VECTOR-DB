import json
import os
import time
from typing import Dict, Any, List, Optional, Callable

from filelock import FileLock


class WriteAheadLog:
    """
    Append-only Write-Ahead Log (WAL) for durability.
    Records all index mutations (inserts, deletes) synchronously
    before acknowledging the request, ensuring crash recovery.

    Lifecycle:
        1. Every mutation is appended (and fsynced) before the request is
           acknowledged.
        2. On startup, ``replay()`` re-applies any entries written after the
           last checkpoint to the freshly loaded index.
        3. After the index is successfully persisted to disk,
           ``checkpoint()`` truncates the log so replay starts clean.
    """

    def __init__(self, collection_id: str, log_dir: str = "wal_logs"):
        self.collection_id = collection_id
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.wal_path = os.path.join(self.log_dir, f"{collection_id}.wal")
        self.checkpoint_path = os.path.join(self.log_dir, f"{collection_id}.checkpoint")

    def _write_entry(self, operation: str, data: Dict[str, Any]):
        entry = {
            "ts": time.time(),
            "op": operation,
            "data": data
        }
        entry_str = json.dumps(entry) + "\n"

        # Open file in append mode and acquire an exclusive lock
        # to ensure concurrent inserts don't corrupt the WAL
        lock = FileLock(self.wal_path + ".lock")
        with lock:
            with open(self.wal_path, "a") as f:
                f.write(entry_str)
                f.flush()
                os.fsync(f.fileno())  # Force write to physical disk

    def log_insert(self, vector_id: str, vector: List[float], metadata: Optional[Dict] = None):
        """Log a vector insertion."""
        self._write_entry("INSERT", {
            "id": vector_id,
            "vec": vector,
            "meta": metadata
        })

    def log_delete(self, vector_id: str):
        """Log a vector deletion."""
        self._write_entry("DELETE", {
            "id": vector_id
        })

    def log_update_metadata(self, vector_id: str, metadata: Dict):
        """Log a metadata update."""
        self._write_entry("UPDATE_META", {
            "id": vector_id,
            "meta": metadata
        })

    def truncate(self):
        """
        Clears the WAL. Typically called after a successful background
        snapshot/compaction of the main index to disk.
        """
        lock = FileLock(self.wal_path + ".lock")
        with lock:
            with open(self.wal_path, "w") as f:
                f.truncate(0)

    def checkpoint(self):
        """
        Mark all current WAL entries as durably persisted in the index
        snapshot and truncate the log.

        Call this only AFTER the index has been successfully saved to disk;
        truncating earlier would lose the only durable copy of recent writes.
        """
        lock = FileLock(self.wal_path + ".lock")
        with lock:
            with open(self.checkpoint_path, "w") as f:
                json.dump({"ts": time.time()}, f)
                f.flush()
                os.fsync(f.fileno())
            with open(self.wal_path, "w") as f:
                f.truncate(0)

    def last_checkpoint_ts(self) -> float:
        """Timestamp of the last checkpoint, or 0.0 if never checkpointed."""
        if not os.path.exists(self.checkpoint_path):
            return 0.0
        try:
            with open(self.checkpoint_path, "r") as f:
                return float(json.load(f).get("ts", 0.0))
        except (json.JSONDecodeError, ValueError, OSError):
            return 0.0

    def read_all(self) -> List[Dict[str, Any]]:
        """Reads all operations for crash recovery playback."""
        if not os.path.exists(self.wal_path):
            return []

        entries = []
        with open(self.wal_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Log corruption detected at the tail end
                        continue
        return entries

    def pending_entries(self) -> List[Dict[str, Any]]:
        """Entries written after the last checkpoint (those needing replay)."""
        cutoff = self.last_checkpoint_ts()
        return [e for e in self.read_all() if e.get("ts", 0.0) > cutoff]

    @staticmethod
    def _index_insert(index, vec, vid, meta):
        """Insert via whichever method the index exposes (HNSW: insert, IVF: add)."""
        if hasattr(index, "insert"):
            index.insert(vec, vid, meta)
        elif hasattr(index, "add"):
            index.add(vec, vid, meta)
        else:
            raise TypeError("index exposes neither insert() nor add()")

    @staticmethod
    def _index_delete(index, vid):
        """Delete via whichever method the index exposes (HNSW: delete, IVF: delete_vector)."""
        if hasattr(index, "delete"):
            index.delete(vid)
        elif hasattr(index, "delete_vector"):
            index.delete_vector(vid)
        else:
            raise TypeError("index exposes neither delete() nor delete_vector()")

    def replay(self, index, on_apply: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        Re-apply pending WAL entries to an index after a crash.

        Works with both HNSWIndex (``insert``/``delete``) and IVFIndex
        (``add``/``delete_vector``); metadata updates are applied to the
        index's ``metadata`` dict when present.

        Returns a summary dict: counts per operation and number of skipped
        (corrupt/unknown) entries.
        """
        applied = {"INSERT": 0, "DELETE": 0, "UPDATE_META": 0}
        skipped = 0

        for entry in self.pending_entries():
            op = entry.get("op")
            data = entry.get("data", {})
            try:
                if op == "INSERT":
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
