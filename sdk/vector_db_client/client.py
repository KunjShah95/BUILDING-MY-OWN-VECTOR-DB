from typing import Optional

import httpx

from vector_db_client.ann import AnnAPI
from vector_db_client.collections import CollectionsAPI
from vector_db_client.multimodal import MultimodalAPI
from vector_db_client.vectors import VectorsAPI


class VectorDBClient:
    """Sync HTTP client for the Vector Database API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        timeout: float = 60.0,
        headers: Optional[dict] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers=headers or {},
        )
        self.collections = CollectionsAPI(self)
        self.vectors = VectorsAPI(self)
        self.multimodal = MultimodalAPI(self)
        self.ann = AnnAPI(self)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "VectorDBClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def request(self, method: str, path: str, **kwargs):
        return self._client.request(method, path, **kwargs)

    def get(self, path: str, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs):
        return self.request("POST", path, **kwargs)

    def delete(self, path: str, **kwargs):
        return self.request("DELETE", path, **kwargs)
