"""Web search endpoints: crawl the open web into the index, then search it.

Exa-style (neural) + SerpAPI-style (keyword) search over a self-crawled corpus,
served entirely from our own indexes via SearchEngineService. No external APIs.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections import deque
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
