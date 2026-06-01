import time
import threading
from typing import Dict
from dataclasses import dataclass


@dataclass
class IndexStatus:
    collection_id: str
    vector_count: int
    last_rebuild: float
    pending_changes: int


class AutoReindexService:
    def __init__(self, threshold: int = 1000, cooldown: int = 300):
        self.threshold = threshold
        self.cooldown = cooldown
        self._statuses: Dict[str, IndexStatus] = {}
        self._timer = None

    def register_collection(self, collection_id: str, vector_count: int = 0):
        if collection_id not in self._statuses:
            self._statuses[collection_id] = IndexStatus(
                collection_id=collection_id,
                vector_count=vector_count,
                last_rebuild=0.0,
                pending_changes=0,
            )

    def record_change(self, collection_id: str, delta: int = 1):
        if collection_id not in self._statuses:
            self.register_collection(collection_id)
        self._statuses[collection_id].pending_changes += delta
        self._statuses[collection_id].vector_count += delta
        self._maybe_schedule_rebuild(collection_id)

    def _maybe_schedule_rebuild(self, collection_id: str):
        status = self._statuses[collection_id]
        if (status.pending_changes >= self.threshold and
                time.time() - status.last_rebuild >= self.cooldown):
            self._schedule_rebuild(collection_id)

    def _schedule_rebuild(self, collection_id: str):
        print(f"Scheduling index rebuild for {collection_id}")
        status = self._statuses[collection_id]
        status.pending_changes = 0
        status.last_rebuild = time.time()

    def get_status(self, collection_id: str) -> Dict:
        status = self._statuses.get(collection_id)
        if not status:
            return {}
        return {
            "collection_id": status.collection_id,
            "vector_count": status.vector_count,
            "pending_changes": status.pending_changes,
            "needs_rebuild": status.pending_changes >= self.threshold,
            "last_rebuild": status.last_rebuild,
        }

    def get_all_statuses(self) -> Dict[str, Dict]:
        return {
            cid: self.get_status(cid)
            for cid in self._statuses
        }
