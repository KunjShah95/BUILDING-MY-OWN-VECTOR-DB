"""Haystack 2.x Document Store integration for the Vector DB."""
import logging
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger(__name__)

class HaystackVectorStore:
    """A Haystack 2.x-compatible document store wrapping the Vector DB API."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(base_url=self.base_url, headers=headers)

    def write_documents(self, documents: List[Dict[str, Any]], collection_id: Optional[str] = None):
        """Write documents to the vector database."""
        for doc in documents:
            text = doc.get("content", doc.get("text", ""))
            metadata = doc.get("metadata", {})
            payload = {"text": text, "metadata": metadata}
            if collection_id:
                payload["collection_id"] = collection_id
            resp = self.client.post("/ingest/text", json=payload)
            resp.raise_for_status()
        return {"success": True, "documents_written": len(documents)}

    def search(self, query: str, k: int = 10, collection_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search documents by text query."""
        payload = {"query": query, "k": k}
        if collection_id:
            payload["collection_id"] = collection_id
        resp = self.client.post("/search/text", json=payload)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("results", []):
            results.append({
                "id": r.get("vector_id") or r.get("id"),
                "content": r.get("text") or r.get("metadata", {}).get("text", ""),
                "score": 1 - r.get("distance", 0),
                "metadata": r.get("metadata", {}),
            })
        return results

    def delete_documents(self, document_ids: List[str]):
        for doc_id in document_ids:
            self.client.delete(f"/vectors/{doc_id}")
        return {"success": True, "deleted": len(document_ids)}

    def close(self):
        self.client.close()
