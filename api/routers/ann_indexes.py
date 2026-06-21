"""
ANN index management routes.
Merged from C:\\ann search engine on 2026-06-21.

Prefix: /api/v1/ann
Tags:   ANN Index Management
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from config.database import get_db
from models.pydantic_models import (
    AnnIndexCreateRequest,
    AnnIndexStatsResponse,
    AnnIndexType,
)
from services.ann_index_service import AnnIndexService

router = APIRouter(prefix="/api/v1/ann", tags=["ANN Index Management"])


def _get_service(db: Session = Depends(get_db)) -> AnnIndexService:
    return AnnIndexService(db)


# ── build ─────────────────────────────────────────────────────────────────────

@router.post("/index")
async def create_index(
    body: AnnIndexCreateRequest,
    svc: AnnIndexService = Depends(_get_service),
):
    """
    Build and populate an index from all vectors currently stored in the DB.

    Pass `index_type` = hnsw | ivf | brute.  Extra parameters (m, ef_construction,
    n_clusters, etc.) are forwarded to the index constructor.
    """
    kwargs = body.model_dump(exclude={"index_type"}, exclude_none=True)
    result = svc.create_index(body.index_type.value, **kwargs)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


# ── info ──────────────────────────────────────────────────────────────────────

@router.get("/index", response_model=AnnIndexStatsResponse)
async def get_index_info(
    index_type: Optional[AnnIndexType] = Query(None, description="Filter to one index type"),
    svc: AnnIndexService = Depends(_get_service),
):
    """Return stats for all loaded indexes, or just one if `index_type` is given."""
    result = svc.get_index_info(index_type.value if index_type else None)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    return AnnIndexStatsResponse(**result)


# ── persist ───────────────────────────────────────────────────────────────────

@router.post("/index/save")
async def save_index(
    index_type: AnnIndexType = Query(..., description="Index type to persist"),
    svc: AnnIndexService = Depends(_get_service),
):
    """Persist an in-memory index to disk (indexes/ann/<type>_index.json)."""
    result = svc.save_index(index_type.value)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@router.post("/index/load")
async def load_index(
    index_type: AnnIndexType = Query(..., description="Index type to restore"),
    svc: AnnIndexService = Depends(_get_service),
):
    """Restore a previously saved index from disk into memory."""
    result = svc.load_index(index_type.value)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    return result


# ── compare ───────────────────────────────────────────────────────────────────

@router.get("/search/compare")
async def compare_search_methods(
    query_vector: str = Query(
        ..., description="Comma-separated float values, e.g. 0.1,0.2,0.3"
    ),
    k: int = Query(10, ge=1, le=100, description="Results per index"),
    svc: AnnIndexService = Depends(_get_service),
):
    """
    Run the same query against every loaded index (HNSW, IVF, BruteForce)
    and return results + wall-clock search time for each.  Useful for
    recall and latency benchmarking.
    """
    try:
        vec = [float(x.strip()) for x in query_vector.split(",")]
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="query_vector must be comma-separated floats, e.g. 0.1,0.2,0.3",
        )

    result = svc.compare_search(vec, k=k)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


# ── overall stats ─────────────────────────────────────────────────────────────

@router.get("/stats")
async def get_statistics(svc: AnnIndexService = Depends(_get_service)):
    """Return DB vector count and stats for all loaded indexes."""
    return svc.get_statistics()
