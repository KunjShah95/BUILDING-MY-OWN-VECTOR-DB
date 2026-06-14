"""Materialized views for pre-computing common query results."""
import json
import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Callable
import uuid

logger = logging.getLogger(__name__)

class MaterializedView:
    def __init__(self, view_id: str, name: str, collection_id: str,
                 query_embedding: List[float], k: int = 100,
                 refresh_interval: int = 300):
        self.view_id = view_id
        self.name = name
        self.collection_id = collection_id
        self.query_embedding = query_embedding
        self.k = k
        self.refresh_interval = refresh_interval
        self.cached_results: List[Dict[str, Any]] = []
        self.last_refreshed: Optional[datetime] = None
        self.refresh_count = 0

class MaterializedViewService:
    def __init__(self):
        self._views: Dict[str, MaterializedView] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def create_view(self, name: str, collection_id: str, query_embedding: List[float],
                    k: int = 100, refresh_interval: int = 300) -> str:
        view_id = str(uuid.uuid4())
        view = MaterializedView(view_id, name, collection_id, query_embedding, k, refresh_interval)
        with self._lock:
            self._views[view_id] = view
        logger.info("Created materialized view %s (%s)", view_id, name)
        return view_id

    def get_view(self, view_id: str) -> Optional[MaterializedView]:
        with self._lock:
            return self._views.get(view_id)

    def list_views(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [{
                "view_id": v.view_id, "name": v.name,
                "collection_id": v.collection_id, "k": v.k,
                "refresh_interval": v.refresh_interval,
                "last_refreshed": str(v.last_refreshed) if v.last_refreshed else None,
                "refresh_count": v.refresh_count,
            } for v in self._views.values()]

    def delete_view(self, view_id: str):
        with self._lock:
            self._views.pop(view_id, None)

    def refresh_view(self, view_id: str, search_fn: Optional[Callable] = None):
        view = self.get_view(view_id)
        if not view or not search_fn:
            return
        try:
            result = search_fn(view.query_embedding, k=view.k)
            results = result.get("results", []) if isinstance(result, dict) else []
            view.cached_results = results
            view.last_refreshed = datetime.now(timezone.utc)
            view.refresh_count += 1
        except Exception as e:
            logger.error("Failed to refresh view %s: %s", view_id, e)

    def start_auto_refresh(self, search_fn: Callable, interval: int = 60):
        self._running = True
        self._thread = threading.Thread(
            target=self._refresh_loop, args=(search_fn, interval), daemon=True
        )
        self._thread.start()
        logger.info("Auto-refresh started (interval=%ds)", interval)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    def _refresh_loop(self, search_fn: Callable, interval: int):
        while self._running:
            time.sleep(interval)
            with self._lock:
                view_ids = list(self._views.keys())
            for vid in view_ids:
                self.refresh_view(vid, search_fn)

mat_view_service = MaterializedViewService()
