"""PITR (point-in-time recovery) API endpoints (Phase 16)."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from services.pitr_service import PITRService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin/backups", tags=["PITR Backups"])

_pitr = PITRService()


class SnapshotRequest(BaseModel):
    collection_id: str
    label: Optional[str] = None


class RestoreRequest(BaseModel):
    target_collection_id: Optional[str] = None


@router.post("/snapshot")
def create_snapshot(body: SnapshotRequest):
    """Create a point-in-time snapshot for a collection."""
    snap = _pitr.create_snapshot(
        collection_id=body.collection_id,
        label=body.label,
    )
    return {"success": True, "snapshot": asdict(snap)}


@router.get("/snapshots")
def list_snapshots(collection_id: Optional[str] = Query(None)):
    """List all snapshots, optionally filtered by collection."""
    snaps = _pitr.list_snapshots(collection_id=collection_id)
    return {"success": True, "snapshots": [asdict(s) for s in snaps], "count": len(snaps)}


@router.post("/restore/{snapshot_id}")
def restore_snapshot(snapshot_id: str, body: Optional[RestoreRequest] = None):
    """Restore a snapshot, optionally to a different collection id."""
    target = body.target_collection_id if body else None
    result = _pitr.restore(snapshot_id=snapshot_id, target_collection_id=target)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("message"))
    return result


@router.get("/verify/{snapshot_id}")
def verify_snapshot(snapshot_id: str):
    """Verify integrity of a snapshot via SHA-256 checksums."""
    ok = _pitr.verify_snapshot(snapshot_id=snapshot_id)
    return {
        "success": True,
        "snapshot_id": snapshot_id,
        "integrity_ok": ok,
        "message": "All checksums match" if ok else "Checksum mismatch or missing files",
    }
