"""Web search endpoints: crawl the open web into the index, then search it.

Exa-style (neural) + SerpAPI-style (keyword) search over a self-crawled corpus,
served entirely from our own indexes via SearchEngineService. No external APIs.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import uuid
from collections import Counter, deque
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from config.database import get_db
from services.vector_service import VectorService
from services.search_engine_service import SearchEngineService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/web", tags=["Web Search"])

# ── Module-level singletons ──────────────────────────────────────────────────

_reranker = None
_reranker_failed = False

_search_cache: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL = 300.0

# Recent queries for autocomplete (bounded FIFO, newest first).
_recent_queries: deque = deque(maxlen=1000)

# Crawl job registry for progress polling.
_crawl_jobs: Dict[str, Dict[str, Any]] = {}

# Freshness tracker singleton.
_freshness_tracker = None
_FRESHNESS_DB = "freshness.db"

# Analytics counters (in-process; reset on server restart).
_analytics: Dict[str, Any] = {
    "total_queries": 0,
    "zero_result_queries": 0,
    "zero_result_examples": [],   # last 20
    "latency_samples_ms": [],     # last 500 samples (circular)
    "clicks": [],                 # last 200 click events
}
_MAX_LATENCY_SAMPLES = 500
_MAX_CLICK_LOG = 200


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cache_key(q: str, collection_id: str, k: int, page: int,
               expand: bool, rerank: bool) -> str:
    raw = f"{q}|{collection_id}|{k}|{page}|{expand}|{rerank}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    entry = _search_cache.get(key)
    if entry and time.time() - entry["ts"] < _CACHE_TTL:
        return entry["val"]
    if entry:
        del _search_cache[key]
    return None


def _cache_set(key: str, value: Dict[str, Any]) -> None:
    _search_cache[key] = {"val": value, "ts": time.time()}


def _get_reranker():
    global _reranker, _reranker_failed
    if _reranker is not None or _reranker_failed:
        return _reranker
    try:
        from services.reranker_service import RerankerService
        _reranker = RerankerService()
    except Exception:  # noqa: BLE001
        _reranker_failed = True
    return _reranker


def _get_freshness_tracker():
    global _freshness_tracker
    if _freshness_tracker is None:
        from search_engine.recrawl import FreshnessTracker
        _freshness_tracker = FreshnessTracker(db_path=_FRESHNESS_DB)
    return _freshness_tracker


def _run_crawl(seeds: List[str], collection_id: str, max_pages: int,
               max_depth: int, same_domain_only: bool,
               job_id: Optional[str] = None) -> dict:
    """Synchronous crawl+ingest driver. Updates _crawl_jobs[job_id] if given."""
    import asyncio

    from config.database import SessionLocal
    from search_engine.crawler import Crawler, CrawlConfig
    from search_engine.ingest import WebIngestor
    from services.embedding_service import embed_text

    db = SessionLocal()
    try:
        vs = VectorService(db)
        ingestor = WebIngestor(vs, collection_id=collection_id, embed_fn=embed_text)
        tracker = _get_freshness_tracker()

        def on_page(page):
            try:
                ingestor.ingest_page(page.url, page.title, page.text)
                from search_engine.crawler.dedup import simhash
                tracker.record(page.url, content_hash=simhash(page.text))
                if job_id and job_id in _crawl_jobs:
                    _crawl_jobs[job_id]["pages"] += 1
                    _crawl_jobs[job_id]["ingested"] = ingestor.ingested
                    _crawl_jobs[job_id]["last_url"] = page.url
            except Exception as e:  # noqa: BLE001
                logger.warning("ingest failed for %s: %s", page.url, e)

        cfg = CrawlConfig(max_pages=max_pages, max_depth=max_depth,
                          same_domain_only=same_domain_only)
        crawler = Crawler(config=cfg)
        stats = asyncio.run(crawler.crawl(seeds, on_page=on_page))
        ingestor.flush()
        stats["ingested"] = ingestor.ingested
        stats["collection_id"] = collection_id
        if job_id and job_id in _crawl_jobs:
            _crawl_jobs[job_id].update({"status": "done", **stats})
        logger.info("crawl complete: %s", stats)
        return stats
    except Exception as exc:
        if job_id and job_id in _crawl_jobs:
            _crawl_jobs[job_id]["status"] = "failed"
            _crawl_jobs[job_id]["error"] = str(exc)
        raise
    finally:
        db.close()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/crawl")
async def crawl(
    background_tasks: BackgroundTasks,
    seeds: List[str] = Body(..., embed=True),
    collection_id: str = Body("web"),
    max_pages: int = Body(50, ge=1, le=5000),
    max_depth: int = Body(3, ge=0, le=10),
    same_domain_only: bool = Body(True),
    run_in_background: bool = Body(True),
):
    """Crawl seed URLs into the search index. Returns job_id for progress polling."""
    if not seeds:
        raise HTTPException(status_code=400, detail="seeds required")
    job_id = uuid.uuid4().hex[:8]
    _crawl_jobs[job_id] = {
        "status": "running", "pages": 0, "ingested": 0,
        "seeds": seeds, "collection_id": collection_id,
        "started_at": time.time(), "last_url": None,
    }
    if run_in_background:
        background_tasks.add_task(
            _run_crawl, seeds, collection_id, max_pages, max_depth,
            same_domain_only, job_id,
        )
        return {"success": True, "status": "crawl_started",
                "job_id": job_id, "collection_id": collection_id, "seeds": seeds}
    stats = _run_crawl(seeds, collection_id, max_pages, max_depth, same_domain_only, job_id)
    return {"success": True, "status": "completed", "job_id": job_id, "stats": stats}


@router.get("/crawl/status/{job_id}")
async def crawl_job_status(job_id: str):
    """Poll crawl progress by job_id returned from POST /crawl."""
    job = _crawl_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@router.get("/crawl/jobs")
async def crawl_jobs():
    """List recent crawl jobs."""
    return {"jobs": list(_crawl_jobs.values())[-20:]}


@router.get("/search")
async def web_search(
    request: Request,
    q: str = Query(..., min_length=1),
    k: int = Query(10, ge=1, le=100),
    page: int = Query(1, ge=1, le=20),
    collection_id: str = Query("web"),
    rerank: bool = Query(True),
    expand: bool = Query(True),
    db: Session = Depends(get_db),
):
    """Hybrid web search with pagination, query expansion, rerank, and L1 cache."""
    from services.embedding_service import embed_text
    from search_engine.query import expand_query, route

    _analytics["total_queries"] += 1
    t0 = time.time()

    detected = route(q)
    effective_expand = expand and detected != "keyword"
    effective_rerank = rerank and detected != "keyword"

    ck = _cache_key(q, collection_id, k, page, effective_expand, effective_rerank)
    cached_val = _cache_get(ck)
    if cached_val is not None:
        cached_val["cached"] = True
        return cached_val

    try:
        query_text = expand_query(q) if effective_expand else q
        query_vector = embed_text(query_text)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"embedding unavailable: {e}")

    # Fetch enough for pagination without unbounded over-fetch.
    top_k_internal = min(k * page, 200)
    reranker = _get_reranker() if effective_rerank else None
    engine = SearchEngineService(VectorService(db), reranker_service=reranker)
    raw = engine.search(query=q, query_vector=query_vector,
                        collection_id=collection_id, top_k=top_k_internal)

    all_results = raw.get("results", [])
    offset = (page - 1) * k
    paged_results = all_results[offset: offset + k]
    total = len(all_results)
    has_next = offset + k < total

    # Track analytics.
    elapsed_ms = (time.time() - t0) * 1000
    samples = _analytics["latency_samples_ms"]
    samples.append(elapsed_ms)
    if len(samples) > _MAX_LATENCY_SAMPLES:
        _analytics["latency_samples_ms"] = samples[-_MAX_LATENCY_SAMPLES:]
    if not all_results:
        _analytics["zero_result_queries"] += 1
        ex = _analytics["zero_result_examples"]
        ex.append(q)
        if len(ex) > 20:
            _analytics["zero_result_examples"] = ex[-20:]

    result = {
        "success": raw.get("success", True),
        "query": q,
        "expanded_query": query_text,
        "route": detected,
        "page": page,
        "page_size": k,
        "total_fetched": total,
        "has_next": has_next,
        "results": paged_results,
        "cached": False,
    }

    _cache_set(ck, result)
    _recent_queries.appendleft(q)
    return result


@router.get("/suggest")
async def suggest(
    q: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=10),
):
    """Autocomplete suggestions from recent search history."""
    q_lower = q.lower().strip()
    seen = set()
    suggestions: List[str] = []
    for rq in _recent_queries:
        rq_l = rq.lower()
        if rq_l.startswith(q_lower) and rq_l != q_lower and rq not in seen:
            suggestions.append(rq)
            seen.add(rq)
            if len(suggestions) >= limit:
                break
    return {"q": q, "suggestions": suggestions}


@router.post("/recrawl/start")
async def recrawl_start(
    db_path: str = Body("freshness.db"),
    poll_interval: float = Body(60.0, ge=10),
    batch_size: int = Body(20, ge=1, le=200),
    collection_id: str = Body("web"),
):
    """Start the background recrawl worker."""
    from search_engine.worker import start
    return start(db_path=db_path, poll_interval=poll_interval,
                 batch_size=batch_size, collection_id=collection_id)


@router.post("/recrawl/stop")
async def recrawl_stop():
    """Signal the background recrawl worker to stop."""
    from search_engine.worker import stop
    return stop()


@router.get("/recrawl/status")
async def recrawl_status():
    """Return recrawl worker state and freshness tracker stats."""
    from search_engine.worker import status
    return status()


@router.delete("/cache")
async def cache_clear():
    """Flush the in-process search result cache."""
    _search_cache.clear()
    return {"success": True, "message": "cache cleared"}


@router.get("/index/stats")
async def index_stats(collection_id: str = Query("web")):
    """Return freshness tracker stats + cache size for the collection."""
    tracker = _get_freshness_tracker()
    return {
        "collection_id": collection_id,
        "freshness": tracker.stats(),
        "cache_entries": len(_search_cache),
        "cache_ttl_seconds": _CACHE_TTL,
        "recent_queries": len(_recent_queries),
    }


@router.post("/click")
async def record_click(
    query: str = Body(...),
    result_url: str = Body(...),
    position: int = Body(..., ge=0),
    session_id: str = Body(""),
):
    """Record a click signal for CTR analytics and future learning-to-rank."""
    event = {
        "query": query,
        "result_url": result_url,
        "position": position,
        "session_id": session_id,
        "ts": time.time(),
    }
    clicks = _analytics["clicks"]
    clicks.append(event)
    if len(clicks) > _MAX_CLICK_LOG:
        _analytics["clicks"] = clicks[-_MAX_CLICK_LOG:]
    return {"success": True}


@router.get("/analytics")
async def analytics():
    """Search analytics: query counts, zero-result rate, p50/p95 latency, CTR."""
    samples = _analytics["latency_samples_ms"]
    total_q = _analytics["total_queries"]
    zero_q = _analytics["zero_result_queries"]

    if samples:
        sorted_s = sorted(samples)
        n = len(sorted_s)
        p50 = sorted_s[int(n * 0.50)]
        p95 = sorted_s[min(int(n * 0.95), n - 1)]
        avg = sum(sorted_s) / n
    else:
        p50 = p95 = avg = 0.0

    click_count = len(_analytics["clicks"])
    ctr = round(click_count / total_q, 4) if total_q else 0.0

    return {
        "total_queries": total_q,
        "zero_result_queries": zero_q,
        "zero_result_rate": round(zero_q / total_q, 4) if total_q else 0.0,
        "zero_result_examples": _analytics["zero_result_examples"][-5:],
        "latency_ms": {"p50": round(p50, 1), "p95": round(p95, 1), "avg": round(avg, 1)},
        "click_count": click_count,
        "ctr": ctr,
        "latency_samples": len(samples),
    }


# ── Answer synthesis ──────────────────────────────────────────────────────────

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "it", "this", "that", "be", "as", "are",
})


def _highlight(text: str, query_tokens: List[str]) -> str:
    """Wrap query terms in **bold** markers."""
    if not query_tokens:
        return text
    pat = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in query_tokens) + r")\b",
        re.IGNORECASE,
    )
    return pat.sub(r"**\1**", text)


def _extract_answer(results: List[Dict[str, Any]], query: str,
                    max_sentences: int = 3) -> Dict[str, Any]:
    """Extractive answer: score sentences by query-term overlap, pick top ones."""
    q_tokens = [
        t.lower() for t in re.findall(r"\b\w+\b", query)
        if t.lower() not in _STOP_WORDS and len(t) > 2
    ]
    if not q_tokens:
        return {"answer": "", "citations": []}

    q_set = set(q_tokens)
    candidates: List[tuple] = []  # (score, sentence, url, title)

    for rank, res in enumerate(results[:5]):
        snippet = res.get("snippet") or res.get("text") or ""
        url = res.get("url", "")
        title = res.get("title", "")
        sentences = _SENT_RE.split(snippet)
        for sent in sentences:
            words = {w.lower() for w in re.findall(r"\b\w+\b", sent)}
            overlap = len(q_set & words)
            if overlap == 0 or len(sent.split()) < 5:
                continue
            # Penalise lower-ranked results slightly
            score = overlap / (1 + 0.1 * rank)
            candidates.append((score, sent.strip(), url, title))

    candidates.sort(key=lambda x: x[0], reverse=True)
    seen_sents: set = set()
    chosen: List[tuple] = []
    citations: List[Dict[str, str]] = []
    for score, sent, url, title in candidates:
        if sent in seen_sents:
            continue
        seen_sents.add(sent)
        chosen.append((score, sent, url, title))
        if url not in {c["url"] for c in citations}:
            citations.append({"url": url, "title": title})
        if len(chosen) >= max_sentences:
            break

    answer_text = " ".join(
        _highlight(s, q_tokens) for _, s, _, _ in chosen
    )
    return {"answer": answer_text, "citations": citations[:3]}


@router.get("/answer")
async def answer(
    q: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=20),
    collection_id: str = Query("web"),
    db: Session = Depends(get_db),
):
    """Extractive answer synthesised from top search result snippets."""
    from services.embedding_service import embed_text
    from search_engine.query import expand_query

    try:
        query_vector = embed_text(expand_query(q))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"embedding unavailable: {e}")

    engine = SearchEngineService(VectorService(db))
    raw = engine.search(query=q, query_vector=query_vector,
                        collection_id=collection_id, top_k=k)
    results = raw.get("results", [])
    synthesis = _extract_answer(results, q)
    return {
        "success": True,
        "query": q,
        "answer": synthesis["answer"],
        "citations": synthesis["citations"],
        "result_count": len(results),
    }


# ── Highlighted search ────────────────────────────────────────────────────────

@router.get("/search/highlighted")
async def search_highlighted(
    q: str = Query(..., min_length=1),
    k: int = Query(10, ge=1, le=100),
    collection_id: str = Query("web"),
    db: Session = Depends(get_db),
):
    """Same as /search but with query terms **bolded** in result snippets."""
    from services.embedding_service import embed_text
    from search_engine.query import expand_query, route

    q_tokens = [
        t.lower() for t in re.findall(r"\b\w+\b", q)
        if t.lower() not in _STOP_WORDS and len(t) > 2
    ]

    try:
        query_vector = embed_text(expand_query(q))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"embedding unavailable: {e}")

    engine = SearchEngineService(VectorService(db))
    raw = engine.search(query=q, query_vector=query_vector,
                        collection_id=collection_id, top_k=k)
    results = raw.get("results", [])
    for res in results:
        snippet = res.get("snippet") or res.get("text") or ""
        res["snippet_highlighted"] = _highlight(snippet, q_tokens)

    return {
        "success": True,
        "query": q,
        "route": route(q),
        "results": results,
    }


# ── Feed ingestion ────────────────────────────────────────────────────────────

@router.post("/feed")
async def ingest_feed(
    background_tasks: BackgroundTasks,
    feed_url: str = Body(..., embed=True),
    collection_id: str = Body("web"),
    run_in_background: bool = Body(True),
):
    """Ingest an RSS 2.0 or Atom 1.0 feed URL into the search index."""
    import asyncio
    from search_engine.crawler.feed import ingest_feed as _ingest_feed
    from search_engine.crawler.fetcher import AsyncFetcher

    job_id = uuid.uuid4().hex[:8]
    _crawl_jobs[job_id] = {
        "type": "feed", "status": "running",
        "feed_url": feed_url, "collection_id": collection_id,
        "pages": 0, "ingested": 0, "started_at": time.time(),
    }

    async def _run():
        from config.database import SessionLocal
        from search_engine.ingest import WebIngestor
        from services.embedding_service import embed_text

        db = SessionLocal()
        try:
            vs = VectorService(db)
            ingestor = WebIngestor(vs, collection_id=collection_id, embed_fn=embed_text)

            async with AsyncFetcher() as fetcher:
                count = await _ingest_feed(feed_url, fetcher, ingestor)

            ingestor.flush()
            _crawl_jobs[job_id].update({
                "status": "done", "ingested": count, "pages": count,
            })
        except Exception as exc:
            _crawl_jobs[job_id].update({"status": "failed", "error": str(exc)})
            logger.error("feed ingest error: %s", exc)
        finally:
            db.close()

    if run_in_background:
        background_tasks.add_task(asyncio.run, _run())
        return {"success": True, "status": "feed_ingest_started",
                "job_id": job_id, "feed_url": feed_url}
    import asyncio as _async
    _async.run(_run())
    return {"success": True, "status": "completed",
            "job_id": job_id, **_crawl_jobs[job_id]}


# ── Sitemap crawl ─────────────────────────────────────────────────────────────

@router.post("/sitemap")
async def crawl_sitemap(
    background_tasks: BackgroundTasks,
    site_url: str = Body(..., embed=True),
    collection_id: str = Body("web"),
    max_urls: int = Body(500, ge=1, le=10000),
    run_in_background: bool = Body(True),
):
    """Discover URLs from sitemap.xml, then crawl and index them all."""
    job_id = uuid.uuid4().hex[:8]
    _crawl_jobs[job_id] = {
        "type": "sitemap", "status": "running",
        "site_url": site_url, "collection_id": collection_id,
        "pages": 0, "ingested": 0, "started_at": time.time(),
    }

    def _run():
        import asyncio
        from search_engine.crawler.sitemap import discover_urls
        from search_engine.crawler.fetcher import AsyncFetcher

        async def _async():
            async with AsyncFetcher() as fetcher:
                urls = await discover_urls(fetcher, site_url, max_urls=max_urls)
            _crawl_jobs[job_id]["discovered_urls"] = len(urls)
            if urls:
                stats = _run_crawl(
                    list(urls), collection_id,
                    max_pages=max_urls, max_depth=0,
                    same_domain_only=False, job_id=job_id,
                )
                _crawl_jobs[job_id].update({"status": "done", **stats})
            else:
                _crawl_jobs[job_id]["status"] = "done"

        asyncio.run(_async())

    if run_in_background:
        background_tasks.add_task(_run)
        return {"success": True, "status": "sitemap_crawl_started",
                "job_id": job_id, "site_url": site_url}
    _run()
    return {"success": True, "status": "completed",
            "job_id": job_id, **_crawl_jobs[job_id]}


# ── Collection management ─────────────────────────────────────────────────────

_INDEX_ROOT = "indexes"


def _list_collections() -> List[Dict[str, Any]]:
    """Scan index directories and return collection metadata."""
    collections = []
    if not os.path.isdir(_INDEX_ROOT):
        return collections
    for name in os.listdir(_INDEX_ROOT):
        col_dir = os.path.join(_INDEX_ROOT, name)
        if not os.path.isdir(col_dir):
            continue
        sparse_path = os.path.join(col_dir, "sparse.json")
        size_bytes = sum(
            os.path.getsize(os.path.join(col_dir, f))
            for f in os.listdir(col_dir)
            if os.path.isfile(os.path.join(col_dir, f))
        )
        collections.append({
            "collection_id": name,
            "has_sparse": os.path.isfile(sparse_path),
            "size_bytes": size_bytes,
            "files": os.listdir(col_dir),
        })
    return collections


@router.get("/collections")
async def list_collections():
    """List all indexed collections with size and index presence."""
    return {"success": True, "collections": _list_collections()}


@router.delete("/collections/{collection_id}")
async def delete_collection(collection_id: str):
    """Delete all index files for a collection (irreversible)."""
    import shutil

    if not collection_id or collection_id in (".", ".."):
        raise HTTPException(status_code=400, detail="invalid collection_id")
    col_dir = os.path.join(_INDEX_ROOT, collection_id)
    if not os.path.isdir(col_dir):
        raise HTTPException(status_code=404, detail=f"collection '{collection_id}' not found")

    shutil.rmtree(col_dir)
    # Also flush related cache entries
    to_del = [k for k, v in _search_cache.items()
              if collection_id in str(v.get("val", {}))]
    for k in to_del:
        _search_cache.pop(k, None)

    return {"success": True, "deleted": collection_id}


# ── SPLADE sparse encode ──────────────────────────────────────────────────────

@router.post("/splade/encode")
async def splade_encode(
    text: str = Body(..., embed=True),
    collection_id: str = Body("web"),
):
    """Return SPLADE-lite sparse vector for a text (top-50 term weights)."""
    splade_dir = os.path.join(_INDEX_ROOT, collection_id, "splade")
    if not os.path.isdir(splade_dir):
        raise HTTPException(
            status_code=404,
            detail=f"No SPLADE index for collection '{collection_id}'. "
                   "POST /api/web/splade/build first."
        )
    from utils.splade_index import SPLADEIndex
    idx = SPLADEIndex.load(splade_dir)
    return {"success": True, "sparse_vector": idx.encode_document(text)}


@router.post("/splade/search")
async def splade_search(
    q: str = Body(..., embed=True),
    k: int = Body(10, ge=1, le=100),
    collection_id: str = Body("web"),
):
    """Sparse dot-product search via SPLADE index."""
    splade_dir = os.path.join(_INDEX_ROOT, collection_id, "splade")
    if not os.path.isdir(splade_dir):
        raise HTTPException(status_code=404,
                            detail=f"No SPLADE index for '{collection_id}'")
    from utils.splade_index import SPLADEIndex
    idx = SPLADEIndex.load(splade_dir)
    results = idx.search(q, k=k)
    return {"success": True, "query": q, "results": results,
            "stats": idx.get_stats()}


# ── Internet-scale (global) search ────────────────────────────────────────────

@router.get("/global")
async def global_search(
    q: str = Query(..., min_length=1, description="Search query"),
    k: int = Query(10, ge=1, le=50),
    collection_id: str = Query("web", description="Local index collection to fuse"),
    providers: Optional[str] = Query(None, description="Comma-separated: brave,exa,serpapi"),
    local_only: bool = Query(False, description="Skip internet providers, local index only"),
    db: Session = Depends(get_db),
):
    """Search the entire internet via Brave/Exa/SerpAPI, fused with local index via RRF.

    Configure providers by setting env vars:
      BRAVE_SEARCH_API_KEY  — Brave Search (recommended, privacy-first)
      EXA_API_KEY           — Exa neural search
      SERPAPI_KEY           — Google SERP proxy

    Falls back to local-only search when no keys are configured.
    """
    from services.internet_search_service import internet_search, available_providers
    from services.embedding_service import embed_text
    from search_engine.query import expand_query

    # Fetch local results to fuse in
    local_results: List[Dict[str, Any]] = []
    try:
        qvec = embed_text(expand_query(q))
        engine = SearchEngineService(VectorService(db))
        raw = engine.search(query=q, query_vector=qvec,
                            collection_id=collection_id, top_k=k)
        local_results = raw.get("results", [])
    except Exception as exc:
        logger.warning("Local search failed (continuing with internet only): %s", exc)

    if local_only:
        return {
            "success": True, "query": q,
            "results": local_results[:k],
            "providers_used": ["local"],
            "total": len(local_results[:k]),
        }

    want = [p.strip() for p in providers.split(",")] if providers else None
    result = internet_search(
        query=q, count=k,
        providers=want,
        local_results=local_results,
    )

    # Track in analytics
    _analytics["total_queries"] += 1
    _recent_queries.appendleft(q)
    if not result["results"]:
        _analytics["zero_result_queries"] += 1

    configured = available_providers()
    if not configured and not local_only:
        result["warning"] = (
            "No internet search providers configured. "
            "Set BRAVE_SEARCH_API_KEY, EXA_API_KEY, or SERPAPI_KEY for real web search. "
            "Showing local index results only."
        )

    return result


@router.get("/providers")
async def search_providers():
    """List all search providers with status and setup instructions."""
    from services.internet_search_service import available_providers
    configured = available_providers()
    all_providers = {
        "duckduckgo": {
            "configured": "duckduckgo" in configured,
            "env_var": None,
            "setup": "pip install duckduckgo-search",
            "free_tier": "unlimited (no key)",
            "notes": "No API key needed. Install duckduckgo-search package.",
        },
        "searxng": {
            "configured": "searxng" in configured,
            "env_var": "SEARXNG_URL",
            "setup": "docker run -p 8080:8080 searxng/searxng  →  SEARXNG_URL=http://localhost:8080",
            "free_tier": "unlimited (self-hosted)",
            "notes": "Included in docker-compose. Aggregates Google, Bing, DDG.",
        },
        "serper": {
            "configured": "serper" in configured,
            "env_var": "SERPER_API_KEY",
            "setup": "serper.dev → free 2 500 queries/month",
            "free_tier": "2500 queries/month",
        },
        "tavily": {
            "configured": "tavily" in configured,
            "env_var": "TAVILY_API_KEY",
            "setup": "tavily.com → free 1 000 queries/month",
            "free_tier": "1000 queries/month",
            "notes": "AI-optimised results, great for RAG.",
        },
        "exa": {
            "configured": "exa" in configured,
            "env_var": "EXA_API_KEY",
            "setup": "exa.ai → free 1 000 queries/month",
            "free_tier": "1000 queries/month",
            "notes": "Neural/semantic search.",
        },
        "brave": {
            "configured": "brave" in configured,
            "env_var": "BRAVE_SEARCH_API_KEY",
            "setup": "api.search.brave.com → free 2 000 queries/month",
            "free_tier": "2000 queries/month",
        },
        "serpapi": {
            "configured": "serpapi" in configured,
            "env_var": "SERPAPI_KEY",
            "setup": "serpapi.com → free 100 queries/month",
            "free_tier": "100 queries/month",
        },
    }
    return {
        "success": True,
        "configured": configured,
        "all_providers": all_providers,
        "recommendation": (
            "Best zero-config setup: SearXNG (self-hosted, already in docker-compose) "
            "+ DuckDuckGo (pip install duckduckgo-search). No API keys needed."
        ),
    }
