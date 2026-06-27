"""Web admin console backend endpoints (Phase 16).

These are distinct from /admin/index-tuning (admin_index_tuning.py) and
from /api/dashboard (dashboard.py).
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from config.database import get_db
from services.billing_service import BillingService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin/console", tags=["Admin Console"])

_billing = BillingService()

# In-memory ring-buffer of slow queries (populated externally or via middleware)
_slow_queries: Deque[Dict[str, Any]] = deque(maxlen=100)


def record_slow_query(path: str, duration_ms: float, tenant_id: Optional[str] = None):
    """Called by performance middleware to record slow queries."""
    if duration_ms >= 100:
        _slow_queries.appendleft({
            "path": path,
            "duration_ms": round(duration_ms, 2),
            "tenant_id": tenant_id,
            "timestamp": datetime.utcnow().isoformat(),
            "estimated_cost_usd": round(duration_ms * 0.00000001, 10),
        })


# ---- Endpoints ---------------------------------------------------------------

@router.get("/overview")
def cluster_overview(db: Session = Depends(get_db)):
    """Return high-level cluster health metrics."""
    try:
        from models.schema import Vector
        total_vectors = db.execute(
            __import__("sqlalchemy", fromlist=["text"]).text("SELECT COUNT(*) FROM vectors")
        ).scalar() or 0
    except Exception:
        total_vectors = 0

    try:
        from models.schema import Collection
        active_tenants_row = db.execute(
            __import__("sqlalchemy", fromlist=["text"]).text(
                "SELECT COUNT(DISTINCT tenant_id) FROM collections"
            )
        ).scalar()
        active_tenants = active_tenants_row or 0
    except Exception:
        active_tenants = 0

    # Estimate storage from index files
    storage_bytes = 0
    for base in ["indexes", "data", "backups"]:
        if os.path.exists(base):
            for root, _, files in os.walk(base):
                for f in files:
                    try:
                        storage_bytes += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass

    return {
        "success": True,
        "total_vectors": total_vectors,
        "active_tenants": active_tenants,
        "storage_bytes": storage_bytes,
        "storage_mb": round(storage_bytes / (1024 * 1024), 2),
        "qps_estimate": None,    # requires Prometheus integration
        "p95_latency_ms": None,  # requires Prometheus integration
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/query-analyzer")
def slow_query_analyzer():
    """Return last 100 slow queries (>100ms) with cost estimates."""
    return {
        "success": True,
        "slow_queries": list(_slow_queries),
        "threshold_ms": 100,
        "count": len(_slow_queries),
    }


@router.get("/cost-explorer")
def cost_explorer(
    tenant_id: str = Query(...),
    month: str = Query(
        default=datetime.utcnow().strftime("%Y-%m"),
        description="Month in YYYY-MM format",
    ),
):
    """Per-tenant cost breakdown for a given month."""
    try:
        year, mon = int(month[:4]), int(month[5:7])
    except (ValueError, IndexError):
        raise HTTPException(status_code=400, detail=f"Invalid month: {month!r}")

    from datetime import timedelta
    start = datetime(year, mon, 1)
    end = (
        datetime(year + 1, 1, 1) - timedelta(seconds=1)
        if mon == 12
        else datetime(year, mon + 1, 1) - timedelta(seconds=1)
    )
    bill = _billing.compute_bill(tenant_id=tenant_id, start_date=start, end_date=end)
    return {
        "success": True,
        "tenant_id": tenant_id,
        "month": month,
        "line_items": [asdict(li) for li in bill.line_items],
        "total_usd": bill.total_usd,
    }


@router.get("/collections")
def list_all_collections(db: Session = Depends(get_db)):
    """Return all collections across all tenants with sizes."""
    try:
        rows = db.execute(
            __import__("sqlalchemy", fromlist=["text"]).text(
                """
                SELECT c.collection_id, c.name, c.tenant_id, c.modality,
                       COUNT(v.id) as vector_count
                FROM collections c
                LEFT JOIN vectors v ON v.collection_id = c.collection_id
                GROUP BY c.collection_id, c.name, c.tenant_id, c.modality
                ORDER BY vector_count DESC
                """
            )
        ).fetchall()
        collections = [
            {
                "collection_id": r[0],
                "name": r[1],
                "tenant_id": r[2],
                "modality": r[3],
                "vector_count": r[4],
            }
            for r in rows
        ]
    except Exception as exc:
        logger.warning("Could not query collections: %s", exc)
        collections = []

    return {"success": True, "collections": collections, "total": len(collections)}


@router.post("/vacuum")
def trigger_vacuum():
    """Trigger compaction: scan index files and remove tombstoned entries."""
    compacted = 0
    errors: List[str] = []

    for base in ["indexes", "data"]:
        if not os.path.exists(base):
            continue
        for root, _, files in os.walk(base):
            for fname in files:
                if fname.endswith(".tombstone") or fname.endswith(".deleted"):
                    fpath = os.path.join(root, fname)
                    try:
                        os.remove(fpath)
                        compacted += 1
                    except OSError as exc:
                        errors.append(str(exc))

    return {
        "success": True,
        "compacted_files": compacted,
        "errors": errors,
        "timestamp": datetime.utcnow().isoformat(),
    }
