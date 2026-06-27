"""Phase 14 — Intelligent Query Mesh router."""
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from services.query_cost_predictor import query_cost_predictor, CostEstimate
from services.query_scheduler import query_scheduler
from services.fusion_telemetry import fusion_telemetry
from services.view_recommender import view_recommender

router = APIRouter(prefix="/api/query", tags=["Query Mesh"])


# ── Pydantic request/response models ─────────────────────────────────────────

class CostEstimateRequest(BaseModel):
    query_vector: List[float]
    k: int = 10
    collection_id: str
    index_type: str = "hnsw"
    filters: Optional[Dict[str, Any]] = None
    index_params: Optional[Dict[str, Any]] = None


class RegisterTenantRequest(BaseModel):
    tenant_id: str
    credits_per_second: float = 10.0
    max_burst: float = 50.0


class FusedSearchRequest(BaseModel):
    results_per_index: Dict[str, List[Dict[str, Any]]]
    k: int = 10
    index_latencies: Optional[Dict[str, float]] = None  # name -> ms, for recording


class RecordActualRequest(BaseModel):
    query_id: str
    actual_latency_ms: float
    actual_scan_count: int


class RecordQueryRequest(BaseModel):
    query_vector: List[float]
    k: int = 10
    filters: Optional[Dict[str, Any]] = None
    collection_id: str


# ── Cost estimator endpoints ──────────────────────────────────────────────────

@router.get("/cost-estimate")
async def get_cost_estimate(
    collection_id: str = Query(...),
    k: int = Query(10, ge=1),
    index_type: str = Query("hnsw"),
    collection_size: Optional[int] = Query(None),
):
    """Estimate query cost for a zero vector (quick sanity check)."""
    if collection_size is not None:
        query_cost_predictor.set_collection_size(collection_id, collection_size)
    estimate: CostEstimate = query_cost_predictor.estimate(
        query_vector=[0.0] * 128,
        k=k,
        collection_id=collection_id,
        index_type=index_type,
    )
    return {"success": True, "estimate": estimate.__dict__}


@router.post("/cost-estimate")
async def post_cost_estimate(body: CostEstimateRequest):
    """Estimate query cost for a real query vector."""
    estimate: CostEstimate = query_cost_predictor.estimate(
        query_vector=body.query_vector,
        k=body.k,
        collection_id=body.collection_id,
        index_type=body.index_type,
        filters=body.filters,
        index_params=body.index_params,
    )
    return {"success": True, "estimate": estimate.__dict__}


@router.post("/cost-estimate/record-actual")
async def record_actual(body: RecordActualRequest):
    """Record observed latency/scan stats to improve calibration."""
    query_cost_predictor.record_actual(
        query_id=body.query_id,
        actual_latency_ms=body.actual_latency_ms,
        actual_scan_count=body.actual_scan_count,
    )
    return {"success": True}


@router.post("/cost-estimate/calibrate")
async def calibrate():
    """Trigger coefficient calibration from recorded actuals."""
    result = query_cost_predictor.calibrate()
    return {"success": True, "calibration": result}


# ── Scheduler endpoints ───────────────────────────────────────────────────────

@router.post("/scheduler/register-tenant")
async def register_tenant(body: RegisterTenantRequest):
    query_scheduler.register_tenant(
        tenant_id=body.tenant_id,
        credits_per_second=body.credits_per_second,
        max_burst=body.max_burst,
    )
    return {"success": True, "tenant_id": body.tenant_id}


@router.get("/scheduler/queue-depth")
async def get_queue_depth(tenant_id: str = Query(...)):
    depth = query_scheduler.get_queue_depth(tenant_id)
    return {"success": True, "tenant_id": tenant_id, "queue_depth": depth}


# ── Fusion telemetry endpoints ────────────────────────────────────────────────

@router.get("/fusion-telemetry")
async def get_fusion_telemetry(window_seconds: int = Query(60, ge=1)):
    snapshot = fusion_telemetry.get_telemetry(window_seconds=window_seconds)
    return {"success": True, "telemetry": snapshot}


@router.post("/fused-search")
async def fused_search(body: FusedSearchRequest):
    """Fuse results from multiple indexes using RRF and return combined results + telemetry."""
    t0 = time.time()
    fused = fusion_telemetry.fuse(results_per_index=body.results_per_index, k=body.k)

    # Record per-index latencies if provided
    if body.index_latencies:
        for index_name, latency_ms in body.index_latencies.items():
            result_count = len(body.results_per_index.get(index_name, []))
            fusion_telemetry.record_index_latency(index_name, latency_ms, result_count)

    total_ms = (time.time() - t0) * 1000
    snapshot = fusion_telemetry.get_telemetry()
    return {
        "success": True,
        "results": fused,
        "count": len(fused),
        "fusion_ms": round(total_ms, 3),
        "telemetry": snapshot,
    }


@router.get("/fusion-telemetry/recommend-index")
async def recommend_index():
    """Suggest the best single index based on recent telemetry."""
    recommended = fusion_telemetry.recommend_index({})
    return {"success": True, "recommended_index": recommended}


# ── View recommender endpoints ────────────────────────────────────────────────

@router.post("/view-recommendations/record")
async def record_query_for_recommendation(body: RecordQueryRequest):
    view_recommender.record_query(
        query_vector=body.query_vector,
        k=body.k,
        filters=body.filters,
        collection_id=body.collection_id,
    )
    return {"success": True}


@router.get("/view-recommendations")
async def list_view_recommendations(min_frequency: int = Query(10, ge=1)):
    recs = view_recommender.analyze(min_frequency=min_frequency)
    return {
        "success": True,
        "recommendations": [
            {
                "recommendation_id": r.recommendation_id,
                "collection_id": r.collection_id,
                "filter_pattern": r.filter_pattern,
                "estimated_speedup": r.estimated_speedup,
                "query_count": r.query_count,
                "applied": r.applied,
            }
            for r in recs
        ],
    }


@router.post("/view-recommendations/{recommendation_id}/apply")
async def apply_view_recommendation(recommendation_id: str):
    rec = view_recommender.get_recommendation(recommendation_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Recommendation {recommendation_id} not found")
    if rec.applied:
        return {"success": True, "message": "Already applied", "recommendation_id": recommendation_id}
    view_id = view_recommender.auto_create(rec)
    return {"success": True, "view_id": view_id, "recommendation_id": recommendation_id}
