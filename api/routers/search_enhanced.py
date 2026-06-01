"""Enhanced search endpoints: hybrid, rerank."""
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any, List

from config.database import get_db
from services.hybrid_search_service import HybridSearchService
from services.reranker_service import RerankerService

router = APIRouter(tags=["Search Enhanced"])


@router.post("/collections/{collection_id}/search/hybrid")
async def hybrid_search(
    collection_id: str,
    query: str = Body(..., embed=True),
    k: int = Body(10),
    alpha: float = Body(0.5, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
):
    """Hybrid dense+sparse search using RRF fusion."""
    svc = HybridSearchService(db_session=db)
    result = svc.search(collection_id=collection_id, query=query, k=k, alpha=alpha)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result)
    return result


@router.post("/collections/{collection_id}/search/rerank")
async def rerank_search(
    collection_id: str,
    query: str = Body(...),
    k: int = Body(10),
    db: Session = Depends(get_db),
):
    """Search then re-rank with cross-encoder."""
    from services.collection_index_service import CollectionIndexService
    from services.embedding_service import embed_text

    query_vector = embed_text(query)
    index_svc = CollectionIndexService(db)
    results = index_svc.search_collection_indexed(
        collection_id=collection_id, query_vector=query_vector,
        k=k * 2, method="brute", distance_metric="cosine",
    )

    if not results.get("success") or not results.get("results"):
        return results

    reranker = RerankerService()
    try:
        reranked = reranker.rerank(query, results["results"], top_k=k)
        results["results"] = reranked
        results["method"] = "reranked"
    except RuntimeError as e:
        results["message"] = f"Rerank unavailable: {e}"

    return results


@router.post("/collections/{collection_id}/index/sparse")
async def build_sparse_index(
    collection_id: str,
    db: Session = Depends(get_db),
):
    """Build a BM25 sparse index for a collection."""
    svc = HybridSearchService(db_session=db)
    result = svc.build_sparse_index(collection_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result)
    return result
