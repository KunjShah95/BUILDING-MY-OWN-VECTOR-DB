"""Semantic Kernel memory store connector."""
import logging
from typing import Any, Dict, List, Optional, Tuple
import httpx

logger = logging.getLogger(__name__)

class SemanticKernelMemoryStore:
    """SK memory store connector wrapping the Vector DB API."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(base_url=self.base_url, headers=headers)

    def save_information(self, collection: str, text: str, id: str,
                         description: Optional[str] = None,
                         metadata: Optional[Dict] = None) -> str:
        """Save a memory record."""
        payload = {
            "text": text,
            "metadata": {
                "id": id,
                "description": description or "",
                "collection": collection,
                **(metadata or {}),
            }
        }
        resp = self.client.post("/ingest/text", json=payload)
        resp.raise_for_status()
        return id

    def get_nearest_match(self, collection: str, query: str,
                          k: int = 1) -> List[Tuple[str, str, float]]:
        """Find nearest matching memories."""
        payload = {"query": query, "k": k}
        resp = self.client.post("/search/text", json=payload)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("results", []):
            meta = r.get("metadata", {})
            results.append((
                meta.get("id", r.get("vector_id", "")),
                meta.get("text", r.get("text", "")),
                1 - r.get("distance", 1.0),
            ))
        return results

    def remove(self, collection: str, id: str):
        self.client.delete(f"/vectors/{id}")

    def close(self):
        self.client.close()
