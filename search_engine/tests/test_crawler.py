"""Offline tests for the crawler stack: no network, no embedding model."""

import asyncio

import pytest

from search_engine.crawler import (
    Crawler, CrawlConfig, Frontier, RobotsRules, SimHashDedup, extract,
)
from search_engine.crawler.fetcher import FetchResult
from search_engine.crawler.frontier import canonicalize


# ---- Frontier --------------------------------------------------------------

def test_frontier_dedup_and_order():
    f = Frontier()
    assert f.add("https://a.com/", priority=1.0) is True
    assert f.add("https://a.com/#frag") is False        # same after canonicalize
    assert f.add("https://a.com/page", priority=0.5, depth=1) is True
    url, depth = f.next()
    assert url == "https://a.com/" and depth == 0       # higher priority first
    f.mark_done(url)
    url2, _ = f.next()
    assert url2 == "https://a.com/page"
    assert f.pending_count() == 0                        # both popped


def test_canonicalize_strips_tracking():
    a = canonicalize("https://X.com/p?utm_source=x&b=2&a=1#frag")
    assert a == "https://x.com/p?a=1&b=2"


# ---- Dedup -----------------------------------------------------------------

def test_simhash_detects_near_dup():
    d = SimHashDedup(threshold=3)
    base = "the quick brown fox jumps over the lazy dog " * 5
    assert d.check_and_add(base) is False                # first time: novel
    assert d.is_duplicate(base + " and runs") is True    # near-identical
    assert d.is_duplicate("completely different text about vector databases") is False


# ---- Parser ----------------------------------------------------------------

def test_extract_title_text_links():
    html = """
    <html><head><title>Hello World</title></head>
    <body><script>ignore()</script>
    <h1>Heading</h1><p>Some meaningful body content here.</p>
    <a href="/next">next</a><a href="https://other.com/x">ext</a>
    </body></html>
    """
    out = extract(html, url="https://site.com/page")
    assert out["title"] == "Hello World"
    assert "meaningful body content" in out["text"]
    assert "ignore()" not in out["text"]
    assert "https://site.com/next" in out["links"]
    assert "https://other.com/x" in out["links"]


# ---- Robots ----------------------------------------------------------------

def test_robots_blocks_disallowed():
    r = RobotsRules(user_agent="VectorDBBot")
    r.load("https://site.com/", "User-agent: *\nDisallow: /private\n")
    assert r.allowed("https://site.com/public") is True
    assert r.allowed("https://site.com/private/x") is False


# ---- Crawler (with fake fetcher) ------------------------------------------

class FakeFetcher:
    """In-memory site. Matches the Fetcher interface used by Crawler."""

    def __init__(self, pages: dict):
        self.pages = pages  # url -> html

    async def fetch(self, url: str) -> FetchResult:
        if url.endswith("/robots.txt"):
            return FetchResult(url, 404, "", "", False, "no robots")
        html = self.pages.get(url)
        if html is None:
            return FetchResult(url, 404, "", "text/html", False, "404")
        return FetchResult(url, 200, html, "text/html", True)

    async def fetch_text(self, url: str):
        res = await self.fetch(url)
        return res.text if res.ok else None


def _page(title, body, links):
    a = "".join(f'<a href="{l}">x</a>' for l in links)
    return f"<html><head><title>{title}</title></head><body><p>{body}</p>{a}</body></html>"


def test_crawler_bfs_respects_limits_and_callback():
    def body(topic):
        return f"This page discusses {topic} in detail with enough unique words. " * 6
    site = {
        "https://demo.com/": _page("Home", body("indexing graphs"), ["/a", "/b", "https://evil.com/x"]),
        "https://demo.com/a": _page("A", body("quantization product codes"), ["/c"]),
        "https://demo.com/b": _page("B", body("sharding replication consensus"), []),
        "https://demo.com/c": _page("C", body("reranking cross encoders latency"), []),
    }
    fetcher = FakeFetcher(site)
    cfg = CrawlConfig(max_pages=10, max_depth=2, same_domain_only=True,
                      respect_robots=False, min_text_len=50, default_delay=0.0)
    crawler = Crawler(config=cfg, frontier=Frontier(), fetcher=fetcher)

    collected = []
    asyncio.run(crawler.crawl(["https://demo.com/"], on_page=collected.append))

    urls = {p.url for p in collected}
    assert "https://demo.com/" in urls
    assert "https://demo.com/a" in urls
    assert "https://demo.com/c" in urls           # reached via /a at depth 2
    assert all("evil.com" not in u for u in urls)  # off-domain blocked
    assert crawler.pages_crawled == 4


def test_crawler_skips_duplicates():
    body = "Identical content paragraph repeated across two different urls. " * 6
    site = {
        "https://dup.com/": _page("One", body, ["/copy"]),
        "https://dup.com/copy": _page("Two", body, []),
    }
    cfg = CrawlConfig(max_pages=10, respect_robots=False, min_text_len=50,
                      default_delay=0.0)
    crawler = Crawler(config=cfg, fetcher=FakeFetcher(site))
    collected = []
    asyncio.run(crawler.crawl(["https://dup.com/"], on_page=collected.append))
    assert crawler.pages_crawled == 1
    assert crawler.stats["skipped_dup"] == 1


# ---- Web ingest (fake vector_service, fake embedder) ----------------------

class FakeVectorService:
    def __init__(self):
        self.created = []

    def create_vector(self, vector_data, metadata=None, vector_id=None,
                      collection_id=None, tenant_id=None):
        self.created.append({"id": vector_id, "vec": vector_data,
                             "meta": metadata, "col": collection_id})
        return {"success": True, "vector_id": vector_id}


def test_web_ingest_writes_dense_and_sparse(tmp_path):
    from search_engine.ingest import WebIngestor, doc_id_for

    vs = FakeVectorService()
    ing = WebIngestor(vs, collection_id="web", index_dir=str(tmp_path),
                      embed_fn=lambda t: [0.1, 0.2, 0.3])
    did = ing.ingest_page("https://x.com/p", "Title", "lots of body text here")
    assert did == doc_id_for("https://x.com/p")
    assert vs.created[0]["id"] == did
    assert vs.created[0]["meta"]["url"] == "https://x.com/p"

    path = ing.flush()
    assert path.endswith("sparse.json")

    # Reload BM25 and confirm the doc is searchable by keyword.
    from utils.bm25_index import BM25Index
    idx = BM25Index.load(path)
    hits = idx.search("body", k=5)
    assert hits and hits[0][0] == did
