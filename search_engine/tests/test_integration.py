"""End-to-end: crawl (fake site) -> ingest -> hybrid search via SearchEngineService.

Proves the whole pipeline wires together without network, DB, or an embedding
model: a fake fetcher serves HTML, WebIngestor writes the BM25 file + records
dense vectors in a fake vector_service, and SearchEngineService fuses both.
"""

import asyncio
import os

import pytest

from search_engine.crawler import Crawler, CrawlConfig, Frontier
from search_engine.crawler.fetcher import FetchResult
from search_engine.ingest import WebIngestor, doc_id_for


class FakeFetcher:
    def __init__(self, pages):
        self.pages = pages

    async def fetch(self, url):
        if url.endswith("/robots.txt"):
            return FetchResult(url, 404, "", "", False, "no robots")
        html = self.pages.get(url)
        if html is None:
            return FetchResult(url, 404, "", "text/html", False, "404")
        return FetchResult(url, 200, html, "text/html", True)

    async def fetch_text(self, url):
        r = await self.fetch(url)
        return r.text if r.ok else None


class FakeVectorService:
    """Records dense vectors; search_vectors returns naive cosine ranking."""

    def __init__(self):
        self.store = {}  # doc_id -> (vector, metadata)

    def create_vector(self, vector_data, metadata=None, vector_id=None,
                      collection_id=None, tenant_id=None):
        self.store[vector_id] = (vector_data, metadata or {})
        return {"success": True, "vector_id": vector_id}

    def search_vectors(self, query_vector, k=10, collection_id=None,
                       tenant_id=None, **kw):
        def dot(a, b):
            return sum(x * y for x, y in zip(a, b))
        ranked = sorted(self.store.items(), key=lambda kv: -dot(query_vector, kv[1][0]))
        results = [{"vector_id": did, "score": dot(query_vector, vec),
                    "metadata": meta} for did, (vec, meta) in ranked[:k]]
        return {"results": results}


def _page(title, body):
    return f"<html><head><title>{title}</title></head><body><p>{body}</p></body></html>"


def test_crawl_ingest_search_end_to_end(tmp_path, monkeypatch):
    # Run inside tmp so SearchEngineService's relative indexes/ path resolves here.
    monkeypatch.chdir(tmp_path)

    site = {
        "https://kb.com/": _page(
            "HNSW indexing",
            "Hierarchical navigable small world graphs power fast vector search "
            "with high recall. " * 8),
        "https://kb.com/pq": _page(
            "Product quantization",
            "Product quantization compresses vectors into compact codes to cut "
            "memory by large factors. " * 8),
    }
    # Distinct embeddings so dense ranking is meaningful.
    embeds = {
        "https://kb.com/": [1.0, 0.0],
        "https://kb.com/pq": [0.0, 1.0],
    }
    vs = FakeVectorService()
    ingestor = WebIngestor(vs, collection_id="web", index_dir="indexes",
                           embed_fn=lambda text: embeds[_current_url[0]])

    _current_url = [None]

    def on_page(page):
        _current_url[0] = page.url
        ingestor.ingest_page(page.url, page.title, page.text)

    cfg = CrawlConfig(max_pages=10, respect_robots=False, min_text_len=50,
                      default_delay=0.0)
    crawler = Crawler(config=cfg, frontier=Frontier(), fetcher=FakeFetcher(site))
    asyncio.run(crawler.crawl(["https://kb.com/", "https://kb.com/pq"], on_page=on_page))
    ingestor.flush()

    assert ingestor.ingested == 2
    assert os.path.exists(os.path.join("indexes", "web", "sparse.json"))

    # Now search via the real engine, fusing dense (fake) + BM25 (real file).
    from services.search_engine_service import SearchEngineService
    engine = SearchEngineService(vs)

    # Query closest to the PQ page's embedding -> PQ should rank first.
    result = engine.search(query="quantization compresses vectors",
                           query_vector=[0.0, 1.0], collection_id="web", top_k=5)
    assert result["success"] is True
    assert result["results"]
    top_id = result["results"][0]["vector_id"]
    assert top_id == doc_id_for("https://kb.com/pq")
