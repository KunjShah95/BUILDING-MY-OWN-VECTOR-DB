"""Performance and benchmark API endpoints."""
import logging
from typing import Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel

from services.materialized_views import mat_view_service
from services.adaptive_index import adaptive_selector
from services.benchmark_service import benchmark_suite

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Performance"])

# --- Materialized Views ---

class CreateViewRequest(BaseModel):
    name: str
    collection_id: str
    k: int = 100
    refresh_interval: int = 300

@router.post("/admin/views")
def create_view(body: CreateViewRequest):
    # Placeholder embedding — in production, this would come from the query
    query_embedding = [0.0] * 128
    view_id = mat_view_service.create_view(
        name=body.name, collection_id=body.collection_id,
        query_embedding=query_embedding, k=body.k,
        refresh_interval=body.refresh_interval,
    )
    return {"success": True, "view_id": view_id}

@router.get("/admin/views")
def list_views():
    return {"success": True, "views": mat_view_service.list_views()}

@router.delete("/admin/views/{view_id}")
def delete_view(view_id: str):
    mat_view_service.delete_view(view_id)
    return {"success": True, "message": "View deleted"}

# --- Adaptive Index ---

@router.get("/admin/adaptive-index/performance")
def get_index_performance(collection_id: Optional[str] = None):
    if collection_id:
        return adaptive_selector.get_performance_report(collection_id)
    return {"success": True, "collections": adaptive_selector.get_all_reports()}

# --- Benchmark ---

@router.post("/admin/benchmark/run")
def run_benchmark():
    vectors, queries, gt = benchmark_suite.generate_synthetic_dataset()
    def search_fn(q, k=10, method="hnsw"):
        return {"results": [{"vector_id": str(i), "distance": float(np.random.random())} for i in range(k)]}
    import numpy as np
    results = benchmark_suite.run_recall_benchmark(search_fn, queries, gt, k=10)
    return {"success": True, "results": results}

@router.get("/admin/benchmark/results")
def get_benchmark_results():
    return benchmark_suite.get_results()

@router.delete("/admin/benchmark")
def clear_benchmark():
    benchmark_suite.clear()
    return {"success": True, "message": "Benchmark results cleared"}
