"""
Startup crash-recovery: replay Write-Ahead Logs into indexes on app boot.

On a clean restart the in-memory HNSW/IVF indexes are empty until an index
file is loaded. Loading an index already triggers WAL replay (see
``HNSWVectorDatabase.load_hnsw_index`` / ``IVFVectorDatabase.load_ivf_index``),
so this module's job is to enumerate every collection that has a persisted
index plus a non-empty WAL and load it, bringing the index back to its
pre-crash state without losing acknowledged writes.

Best-effort: failures for one collection are logged and never block boot.
"""

import logging
import os
from typing import Dict, Any, List, Optional

from utils.wal import WriteAheadLog
from utils.index_paths import get_hnsw_path, get_ivf_path

logger = logging.getLogger(__name__)


def _collections_with_pending_wal(log_dir: str = "wal_logs") -> List[str]:
    """Return collection ids whose WAL has entries past the last checkpoint."""
    if not os.path.isdir(log_dir):
        return []
    pending: List[str] = []
    for name in os.listdir(log_dir):
        if not name.endswith(".wal"):
            continue
        cid = name[: -len(".wal")]
        try:
            if WriteAheadLog(cid, log_dir=log_dir).pending_entries():
                pending.append(cid)
        except OSError:
            continue
    return pending


def recover_all(db_session, log_dir: str = "wal_logs") -> Dict[str, Any]:
    """
    Replay WALs for every collection that has pending entries and a persisted
    index on disk. Returns a per-collection summary.

    Args:
        db_session: SQLAlchemy session used to construct the DB wrappers.
        log_dir: WAL directory.
    """
    # Imported here to avoid a circular import at module load
    from database.hnsw_database import HNSWVectorDatabase
    from database.ivf_database import IVFVectorDatabase

    results: Dict[str, Any] = {}
    pending = _collections_with_pending_wal(log_dir)
    if not pending:
        logger.info("Startup recovery: no pending WALs to replay")
        return results

    hnsw_db = HNSWVectorDatabase(db_session)
    ivf_db = IVFVectorDatabase(db_session)

    for cid in pending:
        scope: Optional[str] = None if cid == "global" else cid
        summary: Dict[str, Any] = {}

        if os.path.exists(get_hnsw_path(scope)):
            try:
                res = hnsw_db.load_hnsw_index(scope)
                summary["hnsw"] = res.get("wal_recovery") if res.get("success") else res.get("message")
            except Exception as exc:  # noqa: BLE001 - boot must not fail
                logger.exception("HNSW recovery failed for %s", cid)
                summary["hnsw"] = {"error": str(exc)}

        if os.path.exists(get_ivf_path(scope)):
            try:
                res = ivf_db.load_ivf_index(scope)
                summary["ivf"] = res.get("wal_recovery") if res.get("success") else res.get("message")
            except Exception as exc:  # noqa: BLE001
                logger.exception("IVF recovery failed for %s", cid)
                summary["ivf"] = {"error": str(exc)}

        if summary:
            results[cid] = summary
            logger.info("Startup recovery for %s: %s", cid, summary)

    return results
