"""
Cross-region replication monitoring API (Phase 11).

Endpoints:
  GET  /api/replication/status    — per-region lag, last_sync_at, healthy, vector_count
  GET  /api/replication/regions   — list registered regions with health
  POST /api/replication/sync      — trigger manual delta sync
  GET  /api/replication/conflicts — list unresolved conflicts
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from services.crdt_sync import CRDTSyncService
from services.region_router import RegionRouter, get_region_router

router = APIRouter(prefix="/api/replication", tags=["Replication"])

# ---------------------------------------------------------------------------
# Module-level singletons (one per process; shared across requests)
# ---------------------------------------------------------------------------

_crdt_service: CRDTSyncService = CRDTSyncService(
    node_id="local",
    clock_store_path="crdt_clocks.json",
)

_region_router: RegionRouter = get_region_router()

# Seed a few example regions if none are configured via env
if not _region_router.list_regions():
    _region_router.register_region("us-east-1",  lat=37.79, lon=-122.39, endpoint="http://us-east-1.internal:8000")
    _region_router.register_region("eu-west-1",  lat=53.35, lon=-6.26,   endpoint="http://eu-west-1.internal:8000")
    _region_router.register_region("ap-south-1", lat=19.08, lon=72.88,   endpoint="http://ap-south-1.internal:8000")

# Simulated per-region sync state (in production this comes from a shared store)
_region_sync_state: Dict[str, Dict[str, Any]] = {}

def _get_region_sync(name: str) -> Dict[str, Any]:
    if name not in _region_sync_state:
        _region_sync_state[name] = {
            "last_sync_at": None,
            "lag_seconds": 0.0,
            "vector_count": 0,
        }
    return _region_sync_state[name]


# ---------------------------------------------------------------------------
# GET /api/replication/status
# ---------------------------------------------------------------------------

@router.get("/status")
async def replication_status() -> Dict[str, Any]:
    """Per-region replication status: lag, last_sync_at, healthy, vector_count."""
    regions = _region_router.list_regions()
    now = time.time()
    per_region = []
    for r in regions:
        sync = _get_region_sync(r["name"])
        lag = (now - sync["last_sync_at"]) if sync["last_sync_at"] else None
        per_region.append({
            "name": r["name"],
            "endpoint": r["endpoint"],
            "healthy": r["healthy"],
            "last_sync_at": sync["last_sync_at"],
            "lag_seconds": round(lag, 2) if lag is not None else None,
            "vector_count": sync["vector_count"],
        })
    return {
        "success": True,
        "regions": per_region,
        "local_vector_count": _crdt_service.vector_count(),
        "unresolved_conflicts": len(_crdt_service.get_conflicts()),
    }


# ---------------------------------------------------------------------------
# GET /api/replication/regions
# ---------------------------------------------------------------------------

@router.get("/regions")
async def list_regions() -> Dict[str, Any]:
    """List all registered regions with their health state."""
    return {
        "success": True,
        "regions": _region_router.list_regions(),
    }


# ---------------------------------------------------------------------------
# POST /api/replication/sync
# ---------------------------------------------------------------------------

@router.post("/sync")
async def trigger_sync() -> Dict[str, Any]:
    """
    Trigger a manual delta sync.
    Produces a delta from local state and records sync timestamps.
    In a real deployment this delta would be pushed to remote regions.
    """
    try:
        delta = _crdt_service.produce_delta()
        now = time.time()
        # Update simulated sync state for all regions
        for r in _region_router.list_regions():
            state = _get_region_sync(r["name"])
            state["last_sync_at"] = now
            state["lag_seconds"] = 0.0
            state["vector_count"] = _crdt_service.vector_count()
        return {
            "success": True,
            "message": "Delta sync triggered",
            "delta_summary": {
                "node_id": delta["node_id"],
                "timestamp": delta["timestamp"],
                "additions_count": len(delta.get("additions", {})),
                "deletions_count": len(delta.get("deletions", {})),
                "vector_clock_entries": len(delta.get("vector_clocks", {})),
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /api/replication/conflicts
# ---------------------------------------------------------------------------

@router.get("/conflicts")
async def list_conflicts() -> Dict[str, Any]:
    """List unresolved concurrent-write conflicts detected by vector clocks."""
    conflicts = _crdt_service.get_conflicts()
    return {
        "success": True,
        "count": len(conflicts),
        "conflicts": conflicts,
    }
