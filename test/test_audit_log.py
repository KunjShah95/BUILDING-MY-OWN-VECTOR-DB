"""Tests for Audit Logging (Phase 9)."""

import json
import os
import time
from unittest.mock import patch

import pytest

from utils.audit_log import (
    log_event,
    log_access,
    log_mutation,
    flush,
    query_logs,
    get_stats,
)


@pytest.fixture(autouse=True)
def _isolate_audit_log(tmp_path):
    """Isolate each test to its own audit log file at a temp path."""
    from utils import audit_log
    audit_log._audit_file = str(tmp_path / "audit.log")
    audit_log._audit_buffer.clear()
    audit_log._audit_last_flush = 0.0

class TestAuditLogEvent:
    def test_log_event_basic(self):
        log_event(
            action="CREATE",
            resource="vector",
            actor="test-user",
            resource_id="vec_001",
        )
        flush()

    def test_log_event_with_all_fields(self):
        log_event(
            action="SEARCH",
            resource="collection",
            actor="api-key-hash",
            tenant_id="acme",
            resource_id="my-docs",
            status="success",
            ip_address="192.168.1.1",
            request_id="req-123",
            details={"query": "test", "k": 10},
            payload={"query_vector": [0.1, 0.2]},
        )
        flush()

    def test_log_event_payload_hash(self):
        log_event(
            action="UPDATE",
            resource="vector",
            actor="user",
            payload={"vector_id": "v1"},
        )
        flush()
        entries = query_logs(limit=10)
        last = entries[-1]
        assert "payload_hash" in last
        assert isinstance(last["payload_hash"], str)
        assert len(last["payload_hash"]) == 64  # SHA-256 hex

    def test_log_event_timestamp_ns(self):
        log_event(action="READ", resource="vector", actor="u1")
        flush()
        entries = query_logs(limit=10)
        last = entries[-1]
        assert "timestamp_ns" in last
        assert last["timestamp_ns"] > 0


class TestAuditLogConvenience:
    def test_log_access_granted(self):
        log_access(
            actor="admin",
            resource="collection",
            resource_id="coll_1",
            granted=True,
        )
        flush()

    def test_log_access_denied(self):
        log_access(
            actor="anon",
            resource="vector",
            resource_id="vec_001",
            granted=False,
        )
        flush()

    def test_log_mutation_create(self):
        log_mutation(
            action="CREATE",
            resource="api_key",
            actor="admin",
            resource_id="key_001",
        )
        flush()

    def test_log_mutation_with_payload(self):
        log_mutation(
            action="DELETE",
            resource="vector",
            actor="user",
            resource_id="vec_001",
            payload={"vector_id": "vec_001", "collection": "docs"},
        )
        flush()


class TestAuditLogQuery:
    def test_query_logs_by_action(self):
        log_event(action="CREATE", resource="vector", actor="u1")
        log_event(action="DELETE", resource="vector", actor="u1")
        log_event(action="SEARCH", resource="vector", actor="u1")
        flush()

        creates = query_logs(actions=["CREATE"], limit=10)
        assert all(e["action"] == "CREATE" for e in creates)

    def test_query_logs_by_actor(self):
        log_event(action="READ", resource="vector", actor="specific-user")
        flush()

        entries = query_logs(actor="specific-user", limit=10)
        assert all(e["actor"] == "specific-user" for e in entries)

    def test_query_logs_by_tenant(self):
        log_event(action="SEARCH", resource="vector", actor="u1", tenant_id="tenant-a")
        log_event(action="SEARCH", resource="vector", actor="u2", tenant_id="tenant-b")
        flush()

        a_entries = query_logs(tenant_id="tenant-a", limit=10)
        assert all(e["tenant_id"] == "tenant-a" for e in a_entries)

    def test_query_logs_limit(self):
        for i in range(20):
            log_event(action="CREATE", resource="vector", actor="u1", resource_id=f"v{i}")
        flush()

        entries = query_logs(limit=5)
        assert len(entries) == 5

    def test_query_logs_empty(self):
        entries = query_logs(limit=10)
        assert isinstance(entries, list)

    def test_query_logs_file_not_found(self, monkeypatch):
        monkeypatch.setattr("utils.audit_log._audit_file", "/nonexistent/audit.log")
        entries = query_logs(limit=10)
        assert entries == []


class TestAuditLogStats:
    def test_get_stats(self):
        stats = get_stats()
        assert "entries" in stats
        assert "size_bytes" in stats
        assert "buffer_pending" in stats

    def test_get_stats_no_file(self, monkeypatch):
        monkeypatch.setattr("utils.audit_log._audit_file", None)
        stats = get_stats()
        assert stats["file"] is None
        assert stats["entries"] == 0


class TestAuditLogBuffer:
    def test_buffer_flush(self):
        for i in range(10):
            log_event(action="CREATE", resource="vector", actor="u1", resource_id=f"v{i}")
        flush()
        stats = get_stats()
        assert stats["buffer_pending"] == 0
        assert stats["entries"] >= 10

    def test_flush_idempotent(self):
        flush()
        flush()
        flush()  # no error

    def test_log_event_no_crash_on_write_failure(self, monkeypatch):
        def failing_open(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("builtins.open", failing_open)
        # Should not raise
        log_event(action="TEST", resource="vector", actor="u1")
        flush()
