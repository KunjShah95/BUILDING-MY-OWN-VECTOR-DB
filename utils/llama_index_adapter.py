"""
LlamaIndex VectorStore Adapter (Phase 3: Agentic Connectors).

Provides a ``LlamaIndexVectorStore`` class that implements the
``llama_index.vector_stores.types.VectorStore`` protocol, allowing the vector
database to be used as a retriever/index backend in LlamaIndex applications.

Usage::

    from utils.llama_index_adapter import LlamaIndexVectorStore
    from llama_index.core import VectorStoreIndex

    store = LlamaIndexVectorStore(
        collection_id="my_docs",
        api_url="http://localhost:8000",
        api_key="...",
    )
    index = VectorStoreIndex.from_vector_store(store)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class LlamaIndexVectorStore:
    """LlamaIndex-compatible vector store adapter.

    Connects to the vector database via its REST API, allowing LlamaIndex
    to use it as a storage backend for indices and retrievers.
    """

    def __init__(
        self,
        collection_id: str,
        api_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
    ):
        self.collection_id = collection_id
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request to the vector database API."""
        import httpx
        url = f"{self.api_url}{path}"
        resp = httpx.request(method, url, headers=self._headers(), **kwargs)
        resp.raise_for_status()
        return resp.json()

    # ---- LlamaIndex VectorStore protocol -----------------------------------

    @property
    def client(self) -> Any:
        return self

    def add(self, nodes: Sequence[Any]) -> List[str]:
        """Add nodes (text + embedding) to the vector store.

        Args:
            nodes: List of llama_index NodeWithEmbedding objects.

        Returns:
            List of vector IDs for the added nodes.
        """
        vectors = []
        for node in nodes:
            vectors.append({
                "vector": node.embedding,
                "metadata": {
                    "text": node.text or "",
                    "node_id": node.node_id,
                    **(node.metadata or {}),
                },
                "vector_id": f"llama_{node.node_id}",
            })

        result = self._request(
            "POST", "/vectors/batch",
            json={"vectors": vectors},
        )
        if result.get("success"):
            return result.get("vector_ids", [])
        logger.error("LlamaIndex add failed: %s", result.get("message"))
        return []

    def delete(self, ref_doc_id: str, **delete_kwargs) -> None:
        """Delete a document from the vector store."""
        vector_id = f"llama_{ref_doc_id}"
        self._request("DELETE", f"/vectors/{vector_id}")

    def query(self, query: Any, **kwargs: Any) -> Any:
        """Query the vector store.

        Args:
            query: A VectorStoreQuery object with query_embedding, similarity_top_k, etc.

        Returns:
            A VectorStoreQueryResult object.
        """
        try:
            from llama_index.core.vector_stores.types import (
                VectorStoreQuery,
                VectorStoreQueryResult,
            )
        except ImportError:
            logger.error("llama_index not installed")
            return type("EmptyResult", (), {"nodes": [], "similarities": [], "ids": []})()

        embedding = query.query_embedding
        k = query.similarity_top_k or 10

        result = self._request(
            "POST", "/search",
            json={
                "query_vector": embedding,
                "k": k,
                "method": "hnsw",
            },
        )

        nodes = []
        similarities = []
        ids = []

        for r in result.get("results", []):
            try:
                from llama_index.core.schema import TextNode
                meta = r.get("metadata", {}) or {}
                node = TextNode(
                    text=meta.get("text", ""),
                    node_id=r.get("vector_id", ""),
                    metadata=meta,
                )
                nodes.append(node)
                similarities.append(1.0 - float(r.get("distance", 0)))
                ids.append(r.get("vector_id", ""))
            except Exception:
                pass

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    def persist(self, persist_path: str, fs: Any = None) -> None:
        """Persist is handled server-side; this is a no-op."""
        pass
