from typing import Any, Dict, List, Optional

from vector_db_client._http import json_post, raise_for_status


class AnnAPI:
    def __init__(self, client):
        self._client = client

    def create_index(self, index_type: str, **kwargs) -> Dict[str, Any]:
        """
        Build and populate an index from all vectors currently stored in the DB.
        
        Parameters:
            index_type: 'hnsw', 'ivf', or 'brute'
            metric: DistanceMetric (brute force only)
            m: neighbors per node (HNSW only)
            m0: layer-0 neighbors (HNSW only)
            ef_construction: construction breadth (HNSW only)
            n_clusters: cluster count (IVF only)
            n_probes: clusters to probe at search time (IVF only)
        """
        body: Dict[str, Any] = {"index_type": index_type}
        body.update(kwargs)
        return json_post(self._client, "/api/v1/ann/index", body)

    def get_index_info(self, index_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Return stats for all loaded indexes, or just one if index_type is given.
        """
        params = {}
        if index_type:
            params["index_type"] = index_type
        response = self._client.get("/api/v1/ann/index", params=params)
        return raise_for_status(response)

    def save_index(self, index_type: str) -> Dict[str, Any]:
        """
        Persist an in-memory index to disk (indexes/ann/<type>_index.json).
        """
        response = self._client.post("/api/v1/ann/index/save", params={"index_type": index_type})
        return raise_for_status(response)

    def load_index(self, index_type: str) -> Dict[str, Any]:
        """
        Restore a previously saved index from disk into memory.
        """
        response = self._client.post("/api/v1/ann/index/load", params={"index_type": index_type})
        return raise_for_status(response)

    def compare_search(self, query_vector: List[float], k: int = 10) -> Dict[str, Any]:
        """
        Run the same query against every loaded index (HNSW, IVF, BruteForce)
        and return results + wall-clock search time for each.
        """
        vec_str = ",".join(str(x) for x in query_vector)
        response = self._client.get(
            "/api/v1/ann/search/compare",
            params={"query_vector": vec_str, "k": k},
        )
        return raise_for_status(response)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return DB vector count and stats for all loaded indexes.
        """
        response = self._client.get("/api/v1/ann/stats")
        return raise_for_status(response)
