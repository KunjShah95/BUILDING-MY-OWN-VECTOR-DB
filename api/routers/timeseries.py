"""Time-series vector support API router.

Provides endpoints for storing and querying time-series vectors
(indexed by timestamp, grouped into logical series).
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from config.database import get_db
from database.schema import Vector

router = APIRouter(tags=["Time Series"])


@router.post("/collections/{collection_id}/vectors/timeseries", status_code=201)
async def insert_timeseries_vector(
    collection_id: str,
    vector: List[float] = Body(...),
    series_id: str = Body(...),
    timestamp: Optional[datetime] = Body(None),
    metadata: Optional[Dict[str, Any]] = Body(None),
    db: Session = Depends(get_db),
):
    """Insert a time-series vector with a timestamp and series_id.

    The vector is stored in the specified collection with time-series
    metadata that enables time-range queries and per-series aggregation.
    """
    from models.vector_model import VectorModel

    model = VectorModel(db)
    ts = timestamp or datetime.utcnow()
    try:
        vec = model.create_vector(
            vector_data=vector,
            metadata=metadata,
            collection_id=collection_id,
        )
        # Attach time-series fields
        vec.timestamp = ts
        vec.series_id = series_id
        db.commit()
        db.refresh(vec)

        return {
            "success": True,
            "message": "Time-series vector inserted",
            "vector_id": vec.vector_id,
            "series_id": series_id,
            "timestamp": ts.isoformat(),
        }
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail={"success": False, "message": str(exc)})


@router.get("/collections/{collection_id}/search/timeseries")
async def search_timeseries(
    collection_id: str,
    query_vector: str = Query(..., description="Query vector as comma-separated values"),
    series_id: Optional[str] = Query(None, description="Filter by series"),
    from_date: Optional[datetime] = Query(None, alias="from"),
    to_date: Optional[datetime] = Query(None, alias="to"),
    k: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Time-range vector search.

    Filters vectors by the time window and/or series_id before performing
    ANN search. Supports pre-filtering at the SQL level.
    """
    try:
        query = [float(x) for x in query_vector.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid query vector format")

    from models.vector_model import VectorModel

    model = VectorModel(db)

    # Build time-series filter query
    ts_query = db.query(Vector).filter(Vector.collection_id == collection_id)
    if series_id:
        ts_query = ts_query.filter(Vector.series_id == series_id)
    if from_date:
        ts_query = ts_query.filter(Vector.timestamp >= from_date)
    if to_date:
        ts_query = ts_query.filter(Vector.timestamp <= to_date)

    # Get filtered vectors and compute distances
    vectors = ts_query.all()
    if not vectors:
        return {"success": True, "results": [], "total_results": 0, "method": "timeseries"}

    all_results = []
    for v in vectors:
        v1 = np.array(query)
        v2 = np.array(v.vector_data)
        dist = float(1.0 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
        all_results.append({
            "distance": dist,
            "vector_id": v.vector_id,
            "metadata": v.meta_data,
            "timestamp": v.timestamp.isoformat() if v.timestamp else None,
            "series_id": v.series_id,
        })

    all_results.sort(key=lambda x: x["distance"])
    results = all_results[:k]

    return {
        "success": True,
        "results": results,
        "total_results": len(results),
        "method": "timeseries",
        "filter": {
            "series_id": series_id,
            "from": from_date.isoformat() if from_date else None,
            "to": to_date.isoformat() if to_date else None,
        },
    }


@router.get("/collections/{collection_id}/series")
async def list_series(
    collection_id: str,
    db: Session = Depends(get_db),
):
    """List all distinct time-series in a collection."""
    rows = (
        db.query(Vector.series_id)
        .filter(Vector.collection_id == collection_id)
        .filter(Vector.series_id.isnot(None))
        .distinct()
        .all()
    )
    series_ids = [r[0] for r in rows]
    return {"success": True, "series": series_ids, "count": len(series_ids)}


@router.get("/collections/{collection_id}/series/{series_id}/latest")
async def get_latest_per_series(
    collection_id: str,
    series_id: str,
    k: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Get the most recent vectors in a time-series."""
    rows = (
        db.query(Vector)
        .filter(Vector.collection_id == collection_id)
        .filter(Vector.series_id == series_id)
        .filter(Vector.timestamp.isnot(None))
        .order_by(desc(Vector.timestamp))
        .limit(k)
        .all()
    )
    return {
        "success": True,
        "results": [
            {
                "vector_id": r.vector_id,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "metadata": r.meta_data,
            }
            for r in rows
        ],
        "count": len(rows),
    }


@router.get("/collections/{collection_id}/series/{series_id}/aggregate")
async def aggregate_series(
    collection_id: str,
    series_id: str,
    window: str = Query("1h", description="Aggregation window (e.g. 1h, 30m, 1d)"),
    db: Session = Depends(get_db),
):
    """Aggregate vector statistics over time windows for a series."""
    rows = (
        db.query(Vector)
        .filter(Vector.collection_id == collection_id)
        .filter(Vector.series_id == series_id)
        .filter(Vector.timestamp.isnot(None))
        .order_by(Vector.timestamp)
        .all()
    )

    if not rows:
        return {"success": True, "aggregations": [], "count": 0}

    # Parse window into seconds
    unit = window[-1]
    value = int(window[:-1])
    window_sec = value * {"h": 3600, "m": 60, "d": 86400}.get(unit, 3600)

    aggregations = []
    current_window_start = rows[0].timestamp
    window_vectors = []

    for r in rows:
        if (r.timestamp - current_window_start).total_seconds() > window_sec:
            if window_vectors:
                arr = np.array([w.vector_data for w in window_vectors])
                aggregations.append({
                    "window_start": current_window_start.isoformat(),
                    "window_end": r.timestamp.isoformat(),
                    "count": len(window_vectors),
                    "mean_vector": arr.mean(axis=0).tolist(),
                    "std_vector": arr.std(axis=0).tolist() if len(window_vectors) > 1 else None,
                })
            current_window_start = r.timestamp
            window_vectors = [r]
        else:
            window_vectors.append(r)

    # Last window
    if window_vectors:
        arr = np.array([w.vector_data for w in window_vectors])
        aggregations.append({
            "window_start": current_window_start.isoformat(),
            "window_end": rows[-1].timestamp.isoformat(),
            "count": len(window_vectors),
            "mean_vector": arr.mean(axis=0).tolist(),
            "std_vector": arr.std(axis=0).tolist() if len(window_vectors) > 1 else None,
        })

    return {"success": True, "aggregations": aggregations, "count": len(aggregations), "window": window}
