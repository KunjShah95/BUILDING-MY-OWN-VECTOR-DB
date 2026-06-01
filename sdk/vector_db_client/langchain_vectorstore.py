"""LangChain VectorStore integration for the Vector DB.

Wraps the VectorDBClient so that LangChain's ecosystem (chains, retrievers,
agents) can use this database as a drop-in vector store.

Usage
-----
.. code-block:: python

    from langchain.embeddings import OpenAIEmbeddings
    from vector_db_client.langchain_vectorstore import VectorDBVectorStore

    embeddings = OpenAIEmbeddings()
    store = VectorDBVectorStore(
        embedding=embeddings,
        base_url="http://localhost:8000",
        collection_id="my_docs",
    )

    # Add documents
    store.add_texts(["Hello world", "Foo bar"])

    # Search
    docs = store.similarity_search("hello", k=5)

    # Use as a retriever in a chain
    retriever = store.as_retriever()
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import httpx
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from vector_db_client import VectorDBClient
from vector_db_client._http import raise_for_status


class VectorDBVectorStore(VectorStore):
    """LangChain VectorStore backed by the Vector Database API.

    Parameters
    ----------
    embedding : Embeddings
        LangChain ``Embeddings`` instance that converts texts into vectors.
    base_url : str
        Base URL of the running Vector DB API (default ``http://localhost:8000``).
    collection_id : str, optional
        If provided, all operations are scoped to this collection namespace.
        When absent, global ``/vectors`` and ``/search`` endpoints are used.

        .. note::
           When ``collection_id`` is set, ``add_texts`` and ``similarity_search``
           delegate embedding to **the server's embedder** via the collection
           ingest / search endpoints rather than using the local ``Embeddings``
           instance for that step. This means the server must have the same
           embedding model registered. If you need to use a custom LangChain
           ``Embeddings``, omit ``collection_id`` to use the global endpoints,
           which accept pre-computed vectors.
    **kwargs
        Extra arguments forwarded to :class:`VectorDBClient`
        (e.g. ``timeout``, ``headers``).
    """

    def __init__(
        self,
        embedding: Embeddings,
        base_url: str = "http://localhost:8000",
        collection_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.base_url = base_url
        self.collection_id = collection_id
        self._client = VectorDBClient(base_url=base_url, **kwargs)

    # ---- convenience helpers ------------------------------------------------

    def _to_documents(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Document]:
        """Convert API search result rows into LangChain ``Document``\\ s."""
        docs: List[Document] = []
        for hit in results:
            metadata = dict(hit.get("metadata") or {})  # shallow copy
            # Extract text from common metadata keys
            text = (
                metadata.pop("text", None)
                or metadata.pop("content", None)
                or metadata.pop("chunk_text", None)
                or ""
            )
            if not text:
                text = hit.get("vector_id", "")

            metadata["vector_id"] = hit.get("vector_id", "")
            metadata["distance"] = hit.get("distance", 0.0)
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

    # ---- required sync methods ----------------------------------------------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed texts and store them in the vector database.

        Parameters
        ----------
        texts : Iterable[str]
            Texts to embed and store.
        metadatas : list of dict, optional
            One metadata dict per text.
        **kwargs
            - ``method``: search method for collection-based ingest
              (default ``"brute"``).

        Returns
        -------
        list of str
            Vector IDs assigned to each inserted text.
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        ids: List[str] = []

        # Pre-compute embeddings only when global mode is active (server doesn't embed)
        embeddings = (
            self.embedding.embed_documents(texts_list)
            if not self.collection_id
            else None
        )

        for i, text in enumerate(texts_list):
            metadata = dict(metadatas[i]) if metadatas and i < len(metadatas) else {}
            # Ensure the text is stored in metadata so we can retrieve it later
            metadata["text"] = text

            if self.collection_id:
                # Use the collection-aware ingest endpoint (server embeds)
                resp = self._client.post(
                    f"/collections/{self.collection_id}/ingest/text",
                    json={
                        "text": text,
                        "metadata": metadata,
                    },
                )
                result = raise_for_status(resp)
            else:
                # Use the global /vectors endpoint (client pre-computed embeddings)
                result = self._client.vectors.create(
                    vector=embeddings[i],
                    metadata=metadata,
                )

            vid = (
                result.get("vector_id")
                or result.get("vector", {}).get("vector_id")
                or ""
            )
            ids.append(vid)

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Return the ``k`` most similar documents to ``query``.

        Parameters
        ----------
        query : str
            Query text.
        k : int
            Number of results (default 4).
        **kwargs
            - ``method``: search method (default ``"hnsw"``).
            - ``filters``: optional metadata filter dict.
            - ``ef_search``: HNSW parameter.
            - ``n_probes``: IVF parameter.
            - ``use_rerank``: IVF rerank flag.

        Returns
        -------
        list of Document
            Matched documents sorted by relevance (closest first).
        """
        if self.collection_id:
            # Use collection-scoped search (server embeds the query)
            body: Dict[str, Any] = {
                "query": query,
                "k": k,
                "method": kwargs.get("method", "brute"),
            }
            if kwargs.get("ef_search") is not None:
                body["ef_search"] = kwargs["ef_search"]
            if kwargs.get("n_probes") is not None:
                body["n_probes"] = kwargs["n_probes"]
            if kwargs.get("use_rerank") is not None:
                body["use_rerank"] = kwargs["use_rerank"]
            if kwargs.get("filters"):
                body["filters"] = kwargs["filters"]

            resp = self._client.post(
                f"/collections/{self.collection_id}/search/text",
                json=body,
            )
            result = raise_for_status(resp)
        else:
            # Global search using raw vectors (client embeds the query)
            query_vector = self.embedding.embed_query(query)
            body = {
                "query_vector": query_vector,
                "k": k,
                "method": kwargs.get("method", "hnsw"),
            }
            if kwargs.get("filters"):
                body["filters"] = kwargs["filters"]

            result = self._client.vectors.search(**body)

        results_list = result.get("results") or result.get("data", [])
        return self._to_documents(results_list)

    # ---- class factory methods ----------------------------------------------

    def close(self) -> None:
        """Release the underlying HTTP client resources."""
        self._client.close()

    def __enter__(self) -> "VectorDBVectorStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VectorDBVectorStore:
        """Create the store, embed the texts, and insert them.

        Parameters
        ----------
        texts : list of str
            Texts to store.
        embedding : Embeddings
            LangChain embedding model.
        metadatas : list of dict, optional
            One metadata dict per text.
        **kwargs
            Forwarded to the constructor and then to :meth:`add_texts`.

        Returns
        -------
        VectorDBVectorStore
            A populated store ready for search.
        """
        base_url = kwargs.pop("base_url", "http://localhost:8000")
        collection_id = kwargs.pop("collection_id", None)

        store = cls(
            embedding=embedding,
            base_url=base_url,
            collection_id=collection_id,
            **kwargs,
        )
        store.add_texts(texts, metadatas=metadatas)
        return store

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> VectorDBVectorStore:
        """Create the store from ``Document`` objects.

        Parameters
        ----------
        documents : list of Document
            Documents to store.
        embedding : Embeddings
            LangChain embedding model.
        **kwargs
            Forwarded to :meth:`from_texts`.

        Returns
        -------
        VectorDBVectorStore
            A populated store ready for search.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(texts, embedding, metadatas=metadatas, **kwargs)

    # ---- optional -----------------------------------------------------------

    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete vectors by their IDs.

        Parameters
        ----------
        ids : list of str, optional
            Vector IDs to remove.
        **kwargs
            Unused.

        Returns
        -------
        bool or None
            ``True`` if at least one vector was successfully deleted.
            Errors are logged individually.
        """
        if not ids:
            return False
        any_success = False
        for vid in ids:
            try:
                self._client.vectors.delete(vid)
                any_success = True
            except Exception:
                import logging

                logging.getLogger(__name__).warning(
                    "Failed to delete vector %s", vid
                )
        return any_success
