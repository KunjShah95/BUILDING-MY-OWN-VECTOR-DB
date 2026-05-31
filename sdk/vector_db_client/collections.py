from typing import Any, Dict, List, Optional

from vector_db_client._http import json_post, raise_for_status
from vector_db_client.models import Collection


class CollectionsAPI:
    def __init__(self, client):
        self._client = client

    def create(
        self,
        name: str,
        *,
        collection_id: Optional[str] = None,
        modality: str = "text",
        dimension: Optional[int] = None,
        embedding_model: Optional[str] = None,
        description: Optional[str] = None,
        distance_metric: str = "cosine",
    ) -> Collection:
        body: Dict[str, Any] = {
            "name": name,
            "modality": modality,
            "distance_metric": distance_metric,
        }
        if collection_id:
            body["collection_id"] = collection_id
        if dimension is not None:
            body["dimension"] = dimension
        if embedding_model:
            body["embedding_model"] = embedding_model
        if description:
            body["description"] = description

        data = json_post(self._client, "/collections", body)
        return Collection.from_api(data["collection"])

    def list(self, limit: int = 100, offset: int = 0) -> List[Collection]:
        response = self._client.get(
            "/collections", params={"limit": limit, "offset": offset}
        )
        data = raise_for_status(response)
        return [Collection.from_api(c) for c in data.get("collections", [])]

    def get(self, collection_id: str) -> Collection:
        response = self._client.get(f"/collections/{collection_id}")
        data = raise_for_status(response)
        return Collection.from_api(data["collection"])

    def delete(self, collection_id: str) -> Dict[str, Any]:
        response = self._client.delete(f"/collections/{collection_id}")
        return raise_for_status(response)

    def build_index(
        self,
        collection_id: str,
        *,
        method: str = "hnsw",
        m: int = 16,
        m0: Optional[int] = None,
        ef_construction: int = 200,
        n_clusters: int = 100,
        n_probes: int = 10,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "method": method,
            "m": m,
            "ef_construction": ef_construction,
            "n_clusters": n_clusters,
            "n_probes": n_probes,
        }
        if m0 is not None:
            body["m0"] = m0
        return json_post(self._client, f"/collections/{collection_id}/index", body)

    def index_stats(self, collection_id: str) -> Dict[str, Any]:
        response = self._client.get(f"/collections/{collection_id}/index/stats")
        return raise_for_status(response)
