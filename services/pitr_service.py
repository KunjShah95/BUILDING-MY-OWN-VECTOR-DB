"""Point-in-time recovery service (Phase 16: Managed Cloud Platform).

Builds on top of backup_service.py patterns to provide snapshot-based PITR
with file-level checksums, restore, and periodic scheduling.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SNAPSHOT_ROOT = Path("backups/snapshots")


@dataclass
class Snapshot:
    snapshot_id: str
    collection_id: str
    timestamp: str
    size_bytes: int
    label: Optional[str]
    files: List[str]
    path: str


class PITRService:
    """Point-in-time recovery via file-level snapshots."""

    def __init__(self, snapshot_root: str = "backups/snapshots"):
        self.snapshot_root = Path(snapshot_root)
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        self._schedulers: Dict[str, threading.Thread] = {}
        self._running: Dict[str, bool] = {}

    # ---- Snapshot creation ---------------------------------------------------

    def create_snapshot(
        self,
        collection_id: str,
        label: Optional[str] = None,
        source_dirs: Optional[List[str]] = None,
    ) -> Snapshot:
        """Copy index files for collection_id into a timestamped snapshot dir.

        If source_dirs is None, searches common index locations automatically.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        snapshot_id = f"{collection_id}_{timestamp}"
        dest = self.snapshot_root / snapshot_id
        dest.mkdir(parents=True, exist_ok=True)

        copied_files: List[str] = []
        total_size = 0

        search_dirs = source_dirs or [
            "indexes",
            "data",
            "wal_logs",
            "media_storage",
        ]

        snap_root_resolved = self.snapshot_root.resolve()

        for sd in search_dirs:
            src_path = Path(sd).resolve()
            if not src_path.exists():
                continue
            # Skip if this source dir is inside (or is) the snapshot root
            try:
                src_path.relative_to(snap_root_resolved)
                continue  # would cause self-copy
            except ValueError:
                pass
            # Copy files that reference this collection
            for root, dirs, files in os.walk(src_path):
                # Prune any walk branches that enter the snapshot root
                dirs[:] = [
                    d for d in dirs
                    if not (Path(root) / d).resolve().is_relative_to(snap_root_resolved)
                ]
                for fname in files:
                    if collection_id in fname or collection_id in root:
                        src_file = Path(root) / fname
                        rel = src_file.relative_to(src_path)
                        dst_file = dest / Path(sd).name / rel
                        dst_file.parent.mkdir(parents=True, exist_ok=True)
                        if src_file.resolve() == dst_file.resolve():
                            continue
                        try:
                            shutil.copy2(str(src_file), str(dst_file))
                        except (PermissionError, shutil.SameFileError):
                            try:
                                shutil.copy(str(src_file), str(dst_file))
                            except shutil.SameFileError:
                                pass
                        copied_files.append(str(dst_file.relative_to(dest)))
                        total_size += src_file.stat().st_size

        # Write metadata
        meta = {
            "snapshot_id": snapshot_id,
            "collection_id": collection_id,
            "timestamp": timestamp,
            "size_bytes": total_size,
            "label": label,
            "files": copied_files,
            "path": str(dest),
        }
        with open(dest / "pitr_meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        logger.info("PITR snapshot created: %s (%d files)", snapshot_id, len(copied_files))
        return Snapshot(**meta)

    # ---- List ----------------------------------------------------------------

    def list_snapshots(self, collection_id: Optional[str] = None) -> List[Snapshot]:
        """Return all snapshots, optionally filtered by collection_id."""
        snapshots: List[Snapshot] = []
        for entry in self.snapshot_root.iterdir():
            meta_file = entry / "pitr_meta.json"
            if not meta_file.exists():
                continue
            try:
                with open(meta_file, encoding="utf-8") as fh:
                    meta = json.load(fh)
                if collection_id and meta.get("collection_id") != collection_id:
                    continue
                snapshots.append(Snapshot(**meta))
            except Exception as exc:
                logger.warning("Failed to read snapshot meta %s: %s", entry, exc)
        snapshots.sort(key=lambda s: s.timestamp, reverse=True)
        return snapshots

    # ---- Restore -------------------------------------------------------------

    def restore(
        self,
        snapshot_id: str,
        target_collection_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Copy snapshot files back to their original locations.

        If target_collection_id is given, file names are rewritten so the
        snapshot is restored under a new collection id (clone).
        """
        snap_dir = self.snapshot_root / snapshot_id
        meta_file = snap_dir / "pitr_meta.json"
        if not meta_file.exists():
            return {"success": False, "message": f"Snapshot {snapshot_id!r} not found"}

        with open(meta_file, encoding="utf-8") as fh:
            meta = json.load(fh)

        source_collection = meta["collection_id"]
        target = target_collection_id or source_collection
        restored: List[str] = []

        for rel_file in meta.get("files", []):
            src = snap_dir / rel_file
            if not src.exists():
                continue
            # Reconstruct original path
            dst = Path(rel_file)
            if target != source_collection:
                dst = Path(str(dst).replace(source_collection, target))
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
            restored.append(str(dst))

        logger.info(
            "PITR restore complete: %s -> %s (%d files)",
            snapshot_id, target, len(restored),
        )
        return {
            "success": True,
            "snapshot_id": snapshot_id,
            "target_collection_id": target,
            "restored_files": restored,
        }

    # ---- Scheduler -----------------------------------------------------------

    def schedule_snapshot(self, collection_id: str, interval_hours: float = 6.0):
        """Start a background thread that creates periodic snapshots."""
        if self._running.get(collection_id):
            logger.info("Scheduler already running for %s", collection_id)
            return

        self._running[collection_id] = True

        def _loop():
            while self._running.get(collection_id):
                try:
                    self.create_snapshot(collection_id, label="scheduled")
                except Exception as exc:
                    logger.error("Scheduled snapshot failed for %s: %s", collection_id, exc)
                time.sleep(interval_hours * 3600)

        t = threading.Thread(target=_loop, daemon=True, name=f"pitr-{collection_id}")
        self._schedulers[collection_id] = t
        t.start()
        logger.info("Snapshot scheduler started for %s (every %sh)", collection_id, interval_hours)

    def stop_scheduler(self, collection_id: str):
        self._running[collection_id] = False

    # ---- Verify --------------------------------------------------------------

    def verify_snapshot(self, snapshot_id: str) -> bool:
        """Verify all files in a snapshot by recomputing SHA-256 checksums.

        Stores checksums in pitr_checksums.json on first call; subsequent calls
        compare against stored checksums.
        """
        snap_dir = self.snapshot_root / snapshot_id
        meta_file = snap_dir / "pitr_meta.json"
        if not meta_file.exists():
            logger.error("Snapshot %s not found", snapshot_id)
            return False

        with open(meta_file, encoding="utf-8") as fh:
            meta = json.load(fh)

        checksum_file = snap_dir / "pitr_checksums.json"
        computed: Dict[str, str] = {}

        for rel_file in meta.get("files", []):
            fpath = snap_dir / rel_file
            if not fpath.exists():
                logger.warning("Missing snapshot file: %s", rel_file)
                return False
            sha = hashlib.sha256()
            with open(fpath, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    sha.update(chunk)
            computed[rel_file] = sha.hexdigest()

        if checksum_file.exists():
            with open(checksum_file, encoding="utf-8") as fh:
                stored = json.load(fh)
            if computed != stored:
                logger.error("Checksum mismatch in snapshot %s", snapshot_id)
                return False
        else:
            with open(checksum_file, "w", encoding="utf-8") as fh:
                json.dump(computed, fh, indent=2)

        return True
