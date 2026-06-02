"""Async ingestion queue API router.

Provides REST endpoints for enqueuing vectors for batched, async ingestion
and monitoring queue status.
"""
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Any, Dict, List, Optional

from config.database import get_db
from services.ingestion_service import BulkIngestionQueue

router = APIRouter(tags=["Ingestion Queue"])

# Module-level singleton queue
_queue: Optional[BulkIngestionQueue] = None


def get_queue() -> BulkIngestionQueue:
    """Get (or create) the global BulkIngestionQueue singleton."""
    global _queue
    if _queue is None:
        from config.settings import get_settings
        settings = get_settings()
        _queue = BulkIngestionQueue(
            batch_size=min(settings.RATE_LIMIT_REQUESTS, 100),
            flush_interval=5.0,
        )
    return _queue


@router.post("/ingest/enqueue", status_code=202)
async def enqueue_vector(
    vector: List[float] = Body(...),
    collection_id: str = Body(...),
    metadata: Optional[Dict[str, Any]] = Body(None),
    queue: BulkIngestionQueue = Depends(get_queue),
):
    """Enqueue a single vector for async batch ingestion.

    The vector will be stored in an in-memory queue and flushed to the
    database in batches (when batch_size is reached or flush_interval elapses).
    Returns immediately with a 202 Accepted status.
    """
    import numpy as np
    try:
        await queue.enqueue(collection_id, np.array(vector, dtype=np.float32), metadata)
        return {
            "success": True,
            "message": "Vector enqueued for batch ingestion",
            "queue_size": queue.size,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"success": False, "message": str(exc)})


@router.post("/ingest/enqueue-many", status_code=202)
async def enqueue_many(
    vectors: List[List[float]] = Body(...),
    collection_id: str = Body(...),
    metadata_list: Optional[List[Optional[Dict[str, Any]]]] = Body(None),
    queue: BulkIngestionQueue = Depends(get_queue),
):
    """Enqueue multiple vectors for async batch ingestion.

    Accepts a list of vectors and an optional parallel list of metadata dicts.
    """
    import numpy as np
    try:
        vecs = [np.array(v, dtype=np.float32) for v in vectors]
        await queue.enqueue_many(collection_id, vecs, metadata_list)
        return {
            "success": True,
            "message": f"{len(vectors)} vectors enqueued for batch ingestion",
            "queue_size": queue.size,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"success": False, "message": str(exc)})


@router.post("/ingest/flush")
async def flush_queue(
    queue: BulkIngestionQueue = Depends(get_queue),
):
    """Force-flush all pending items in the ingestion queue to the database."""
    await queue.flush_all()
    return {
        "success": True,
        "message": "Queue flushed",
        "queue_size": queue.size,
    }


@router.get("/ingest/status")
async def ingestion_status(
    queue: BulkIngestionQueue = Depends(get_queue),
):
    """Get the current ingestion queue status."""
    return {
        "success": True,
        "queue_size": queue.size,
        "batch_size": queue.batch_size,
        "flush_interval": queue.flush_interval,
        "flush_needed": queue.is_flush_needed,
    }
