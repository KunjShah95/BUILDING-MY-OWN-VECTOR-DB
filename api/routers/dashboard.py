"""Dashboard router - serves UI pages and data endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from pathlib import Path

from config.database import get_db
from services.vector_service import VectorService
from services.collection_service import CollectionService

router = APIRouter(tags=["Dashboard"])

STATIC_DIR = Path(__file__).parent.parent / "static"


def mount_static(app):
    """Mount static files. Called from main.py during integration."""
    if STATIC_DIR.exists():
        app.mount("/dashboard/static", StaticFiles(directory=str(STATIC_DIR)), name="dashboard_static")


@router.get("/api/dashboard/stats")
async def dashboard_stats(db: Session = Depends(get_db)):
    """Aggregated statistics for the dashboard overview."""
    svc = VectorService(db)
    coll_svc = CollectionService(db)
    db_stats = svc.get_database_stats()
    vec_stats = db_stats.get("stats", {})
    colls = coll_svc.list_collections(limit=500)
    return {
        "success": True,
        "stats": {
            "total_vectors": vec_stats.get("total_vectors", 0),
            "total_collections": colls.get("count", 0) if colls.get("success") else 0,
            "collections": [
                {"id": c["collection_id"], "name": c["name"], "modality": c["modality"]}
                for c in (colls.get("collections") or [])
            ] if colls.get("success") else [],
        }
    }


@router.get("/api/dashboard/latency")
async def dashboard_latency(db: Session = Depends(get_db)):
    """Query latency information."""
    svc = VectorService(db)
    db_stats = svc.get_database_stats()
    stats = db_stats.get("stats", {})
    total_searches = stats.get("total_searches", 0)
    total_search_time = stats.get("total_search_time", 0)
    avg_latency = round((total_search_time / total_searches) * 1000, 2) if total_searches > 0 else 0.0
    return {
        "success": True,
        "latency": {
            "avg_ms": avg_latency,
            "total_searches": total_searches,
        }
    }


@router.get("/api/dashboard/index-info")
async def dashboard_index_info(db: Session = Depends(get_db)):
    """Index status information."""
    svc = VectorService(db)
    db_stats = svc.get_database_stats()
    stats = db_stats.get("stats", {})
    hnsw = stats.get("hnsw_index") or {}
    return {
        "success": True,
        "index_info": {
            "hnsw_loaded": stats.get("hnsw_available", True) and bool(stats.get("hnsw_index")),
            "ivf_loaded": stats.get("ivf_available", False),
            "total_nodes": hnsw.get("num_nodes", 0) or stats.get("hnsw_nodes", 0),
        }
    }


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    """Serve the main dashboard SPA page."""
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Dashboard UI not found</h1>")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))
