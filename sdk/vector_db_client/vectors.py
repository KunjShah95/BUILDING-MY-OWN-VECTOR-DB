from typing import Any, Dict, List, Optional

from vector_db_client._http import json_post, raise_for_status


class VectorsAPI:
    def __init__(self, client):
        self._client = client

    def create(
        self,
        vector: List[float],
        *,
        metadata: Optional[Dict[str, Any]] = None,
        vector_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"vector": vector}
        if metadata:
            body["metadata"] = metadata
        if vector_id:
            body["vector_id"] = vector_id
        return json_post(self._client, "/vectors", body)

    def get(self, vector_id: str) -> Dict[str, Any]:
        response = self._client.get(f"/vectors/{vector_id}")
        return raise_for_status(response)

    def delete(self, vector_id: str) -> Dict[str, Any]:
        response = self._client.delete(f"/vectors/{vector_id}")
        return raise_for_status(response)

    def search(
        self,
        query_vector: List[float],
        *,
        k: int = 5,
        method: str = "hnsw",
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "query_vector": query_vector,
            "k": k,
            "method": method,
        }
        if filters:
            body["filters"] = filters
        return json_post(self._client, "/search", body)
