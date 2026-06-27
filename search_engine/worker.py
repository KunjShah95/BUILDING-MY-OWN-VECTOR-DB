"""Background recrawl worker.

Polls FreshnessTracker for URLs past their recrawl deadline and re-ingests them
using the existing _run_crawl driver from the web_search router. Runs as a
daemon thread so it does not block FastAPI startup/shutdown.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from search_engine.recrawl import FreshnessTracker

logger = logging.getLogger(__name__)

_thread: Optional[threading.Thread] = None
_stop = threading.Event()
_tracker: Optional[FreshnessTracker] = None


def start(
    db_path: str = "freshness.db",
    poll_interval: float = 60.0,
    batch_size: int = 20,
    collection_id: str = "web",
) -> dict:
    """Start the recrawl worker thread (idempotent — safe to call multiple times)."""
    global _thread, _tracker
    if _thread and _thread.is_alive():
        return {"status": "already_running"}
    _stop.clear()
    _tracker = FreshnessTracker(db_path=db_path)
    _thread = threading.Thread(
        target=_loop,
        args=(poll_interval, batch_size, collection_id),
        daemon=True,
        name="recrawl-worker",
    )
    _thread.start()
    logger.info("recrawl worker started (poll=%.0fs, batch=%d)", poll_interval, batch_size)
    return {"status": "started", "poll_interval": poll_interval, "batch_size": batch_size}


def stop() -> dict:
    """Signal the worker to stop after its current poll cycle."""
    _stop.set()
    return {"status": "stopping"}


def status() -> dict:
    running = bool(_thread and _thread.is_alive())
    return {
        "running": running,
        "tracker": _tracker.stats() if _tracker else None,
    }


def _loop(poll_interval: float, batch_size: int, collection_id: str) -> None:
    from api.routers.web_search import _run_crawl

    while not _stop.is_set():
        try:
            if _tracker:
                due = _tracker.due(limit=batch_size)
                if due:
                    logger.info("recrawl: %d URLs due", len(due))
                    _run_crawl(
                        due, collection_id,
                        max_pages=len(due), max_depth=0, same_domain_only=False,
                    )
                    for url in due:
                        _tracker.record(url)
        except Exception as exc:
            logger.warning("recrawl loop error: %s", exc)
        _stop.wait(poll_interval)
