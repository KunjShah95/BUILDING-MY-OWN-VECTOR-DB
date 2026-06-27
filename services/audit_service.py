"""SOC2/GDPR audit log service with append-only JSON lines storage."""
import json
import os
import threading
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import logging

logger = logging.getLogger(__name__)

_PII_REDACTED = "[REDACTED]"


class AuditService:
    """
    Append-only audit log.  Each entry is a JSON line written to audit.log.

    Fields per entry:
        timestamp, tenant_id, user_id, action, resource, metadata
    """

    def __init__(self, log_path: Optional[str] = None) -> None:
        self.log_path = Path(log_path or "audit.log")
        self._lock = threading.Lock()
        # Ensure parent dir exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Write -----------------------------------------------------------

    def log_event(
        self,
        tenant_id: str,
        user_id: str,
        action: str,
        resource: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Append a timestamped audit event to the log file."""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "tenant_id": tenant_id,
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "metadata": metadata or {},
        }
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        return entry

    # ---- Read ------------------------------------------------------------

    def _read_all(self) -> List[Dict[str, Any]]:
        """Read all log entries. Returns empty list if file doesn't exist."""
        if not self.log_path.exists():
            return []
        entries: List[Dict[str, Any]] = []
        with self.log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed audit line: %r", line)
        return entries

    # ---- Exports ---------------------------------------------------------

    def export_soc2(
        self,
        tenant_id: str,
        start_date: Union[str, date, datetime, None] = None,
        end_date: Union[str, date, datetime, None] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return all audit events for tenant_id filtered by date range.
        start_date / end_date can be ISO strings or date/datetime objects.
        """
        def _to_dt(d) -> Optional[datetime]:
            if d is None:
                return None
            if isinstance(d, datetime):
                return d
            if isinstance(d, date):
                return datetime(d.year, d.month, d.day)
            # string
            s = str(d)
            for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
                try:
                    return datetime.strptime(s, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Cannot parse date: {d!r}")

        start_dt = _to_dt(start_date)
        end_dt = _to_dt(end_date)
        if end_dt and not end_dt.hour and not end_dt.minute:
            # Treat end_date as inclusive day-end
            end_dt = end_dt.replace(hour=23, minute=59, second=59, microsecond=999999)

        results: List[Dict[str, Any]] = []
        for entry in self._read_all():
            if entry.get("tenant_id") != tenant_id:
                continue
            ts_raw = entry.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_raw.rstrip("Z"))
            except ValueError:
                continue
            if start_dt and ts < start_dt:
                continue
            if end_dt and ts > end_dt:
                continue
            results.append(entry)
        return results

    def export_gdpr(self, user_id: str) -> List[Dict[str, Any]]:
        """Return all audit events for a specific user (DSAR / data subject access request)."""
        return [e for e in self._read_all() if e.get("user_id") == user_id]

    # ---- GDPR erasure ----------------------------------------------------

    def purge_user_data(self, user_id: str) -> int:
        """
        Redact PII from all audit log entries belonging to user_id.
        Rewrites the entire log file. Returns count of redacted entries.
        """
        if not self.log_path.exists():
            return 0

        entries = self._read_all()
        count = 0
        updated: List[str] = []
        for entry in entries:
            if entry.get("user_id") == user_id:
                entry["user_id"] = _PII_REDACTED
                entry["metadata"] = {
                    k: (_PII_REDACTED if isinstance(v, str) else v)
                    for k, v in entry.get("metadata", {}).items()
                }
                count += 1
            updated.append(json.dumps(entry))

        with self._lock:
            with self.log_path.open("w", encoding="utf-8") as fh:
                fh.write("\n".join(updated) + ("\n" if updated else ""))

        return count
