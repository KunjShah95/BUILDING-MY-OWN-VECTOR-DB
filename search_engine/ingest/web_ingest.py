"""Web-ingest adapter: bridge crawled pages into the existing dual index.

For each page we write BOTH:
  - a dense vector via the injected vector_service (HNSW/ANN core), and
  - a BM25 sparse posting via utils.bm25_index.BM25Index,

using the SAME doc_id for both so SearchEngineService's Reciprocal Rank Fusion
can join them (it fuses on `vector_id`). The BM25 index is persisted to
`indexes/{collection_id}/sparse.json`, exactly where SearchEngineService loads it.
"""

from __future__ import annotations

import hashlib
import os
from typing import Callable, Dict, List, Optional

from utils.bm25_index import BM25Index


def doc_id_for(url: str) -> str:
    """Stable doc id derived from the canonical URL (re-crawl overwrites, no dup)."""
    return "web_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:24]


# Default embedder is imported lazily so tests / crawl-only paths don't need
# sentence-transformers installed.
def _default_embed(text: str) -> List[float]:
    from services.embedding_service import embed_text

    return embed_text(text)


class WebIngestor:
    """Accumulates crawled pages into the dense + sparse indexes for a collection."""

    def __init__(
        self,
        vector_service,
        collection_id: str,
        tenant_id: Optional[str] = None,
        index_dir: str = "indexes",
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        max_embed_chars: int = 8000,
    ):
        self.vector_service = vector_service
        self.collection_id = collection_id
        self.tenant_id = tenant_id
        self.index_dir = index_dir
        self.embed_fn = embed_fn or _default_embed
        self.max_embed_chars = max_embed_chars
        self.bm25 = self._load_or_new_bm25()
        self.ingested = 0

    @property
    def sparse_path(self) -> str:
        return os.path.join(self.index_dir, self.collection_id, "sparse.json")

    def _load_or_new_bm25(self) -> BM25Index:
        if os.path.exists(self.sparse_path):
            try:
                return BM25Index.load(self.sparse_path)
            except Exception:
                pass
        return BM25Index()

    def ingest_page(self, url: str, title: str, text: str,
                    metadata: Optional[Dict] = None) -> str:
        """Index one page. Returns the doc_id used."""
        did = doc_id_for(url)
        # Embed title + body (title repeated lightly to boost its weight).
        embed_input = (title + ". " + text)[: self.max_embed_chars]
        vector = self.embed_fn(embed_input)

        meta = {"url": url, "title": title,
                "snippet": text[:300].replace("\n", " ").strip(),
                "source": "web"}
        if metadata:
            meta.update(metadata)

        self.vector_service.create_vector(
            vector_data=vector,
            metadata=meta,
            vector_id=did,
            collection_id=self.collection_id,
            tenant_id=self.tenant_id,
        )
        # BM25 doc text = title + body so keyword hits on titles score well.
        self.bm25.add_document(f"{title}\n{text}", did)
        self.ingested += 1
        return did

    def flush(self) -> str:
        """Persist the BM25 index to where SearchEngineService reads it."""
        os.makedirs(os.path.dirname(self.sparse_path), exist_ok=True)
        self.bm25.save(self.sparse_path)
        return self.sparse_path
