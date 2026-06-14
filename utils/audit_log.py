"""
Audit Logging (Phase 9: Security & Compliance).

Provides an immutable, append-only log of every mutation and access event in
the vector database, suitable for compliance audits (SOC2, HIPAA, GDPR).

Events are written to a structured JSON log file with:
  - Timestamp (nanosecond precision for ordering)
  - Actor: API key hash, tenant ID, user if available
  - Action: CREATE, READ, UPDATE, DELETE, SEARCH, EXPORT
  - Resource: collection, vector ID, index type
  - Metadata: IP address, request ID, result status
  - Payload hash: SHA-256 of the request/response for tamper detection

The log is append-only and never modified after writing. For production,
ship to a SIEM (Splunk, ELK, Datadog) via async transport.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Module-level audit logger state
_audit_file: Optional[str] = None
_audit_lock = threading.Lock()
_audit_buffer: List[Dict[str, Any]] = []
_audit_flush_interval = 5.0  # seconds
_audit_last_flush = time.time()


def _init_audit_log(log_dir: str = "audit_logs"):
    """Initialize the audit log directory."""
    global _audit_file
    os.makedirs(log_dir, exist_ok=True)
    date_str = datetime.utcnow().strftime("%Y%m%d")
    _audit_file = os.path.join(log_dir, f"audit_{date_str}.log")


def _flush_buffer():
    """Flush the in-memory buffer to disk."""
    global _audit_buffer, _audit_last_flush, _audit_file

    if not _audit_buffer:
        return

    if _audit_file is None:
        _init_audit_log()

    with _audit_lock:
        if not _audit_buffer:
            return

        try:
            with open(_audit_file, "a") as f:
                for entry in _audit_buffer:
                    f.write(json.dumps(entry) + "\n")
                f.flush()
                os.fsync(f.fileno())
            _audit_buffer = []
            _audit_last_flush = time.time()
        except Exception as exc:
            logger.error("Audit log write failed: %s", exc)


def _needs_flush() -> bool:
    return len(_audit_buffer) >= 100 or (time.time() - _audit_last_flush) >= _audit_flush_interval


def _make_payload_hash(data: Any) -> str:
    """Create a SHA-256 hash of the data for tamper detection."""
    try:
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()
    except Exception:
        return ""


def log_event(
    action: str,
    resource: str,
    actor: str = "anonymous",
    tenant_id: Optional[str] = None,
    resource_id: Optional[str] = None,
    status: str = "success",
    ip_address: Optional[str] = None,
    request_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    payload: Optional[Any] = None,
):
    """Log an audit event.

    Args:
        action: One of CREATE, READ, UPDATE, DELETE, SEARCH, EXPORT, LOGIN, CONFIG.
        resource: Type of resource (vector, collection, index, api_key, tenant).
        actor: Who performed the action (API key hash, user ID).
        tenant_id: Tenant scope for multi-tenant support.
        resource_id: Specific resource identifier.
        status: success, failure, denied.
        ip_address: Client IP address.
        request_id: Correlating request ID.
        details: Additional structured details.
        payload: Original request/response data (will be hashed).
    """
    now = datetime.utcnow()
    entry: Dict[str, Any] = {
        "timestamp": now.isoformat() + "Z",
        "timestamp_ns": time.time_ns(),
        "action": action.upper(),
        "resource": resource,
        "actor": actor,
        "tenant_id": tenant_id,
        "resource_id": resource_id,
        "status": status,
        "ip_address": ip_address,
        "request_id": request_id,
    }

    if details:
        entry["details"] = details
    if payload is not None:
        entry["payload_hash"] = _make_payload_hash(payload)

    with _audit_lock:
        _audit_buffer.append(entry)

    if _needs_flush():
        _flush_buffer()


def log_access(
    actor: str,
    resource: str,
    resource_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    granted: bool = True,
    ip_address: Optional[str] = None,
    request_id: Optional[str] = None,
):
    """Convenience for access control logging."""
    log_event(
        action="ACCESS",
        resource=resource,
        actor=actor,
        tenant_id=tenant_id,
        resource_id=resource_id,
        status="granted" if granted else "denied",
        ip_address=ip_address,
        request_id=request_id,
    )


def log_mutation(
    action: str,
    resource: str,
    actor: str,
    resource_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    payload: Optional[Any] = None,
    request_id: Optional[str] = None,
):
    """Convenience for mutation (CUD) logging."""
    log_event(
        action=action,
        resource=resource,
        actor=actor,
        tenant_id=tenant_id,
        resource_id=resource_id,
        status="success",
        payload=payload,
        request_id=request_id,
    )


def flush():
    """Force-flush the audit log buffer to disk."""
    _flush_buffer()


def query_logs(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    actions: Optional[List[str]] = None,
    actor: Optional[str] = None,
    tenant_id: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Query the audit log (simple file-based query).

    For production, use a SIEM integration instead.
    """
    results = []

    if _audit_file is None:
        return results

    try:
        with open(_audit_file) as f:
            for line in f:
                if len(results) >= limit:
                    break
                try:
                    entry = json.loads(line)
                    if start_time and entry.get("timestamp", "") < start_time:
                        continue
                    if end_time and entry.get("timestamp", "") > end_time:
                        continue
                    if actions and entry.get("action", "").upper() not in [a.upper() for a in actions]:
                        continue
                    if actor and entry.get("actor") != actor:
                        continue
                    if tenant_id and entry.get("tenant_id") != tenant_id:
                        continue
                    results.append(entry)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass

    return results


def get_stats() -> Dict[str, Any]:
    """Get audit log statistics."""
    if _audit_file and os.path.exists(_audit_file):
        size = os.path.getsize(_audit_file)
        with open(_audit_file) as f:
            count = sum(1 for _ in f)
        return {
            "file": _audit_file,
            "size_bytes": size,
            "entries": count,
            "buffer_pending": len(_audit_buffer),
        }
    return {
        "file": None,
        "size_bytes": 0,
        "entries": 0,
        "buffer_pending": len(_audit_buffer),
    }
