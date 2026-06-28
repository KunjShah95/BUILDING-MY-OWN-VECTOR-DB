"""Tests for Phase 16: Managed Cloud Platform.

Covers billing service, PITR service, and API routers.
Run with: pytest test/test_cloud_platform.py -v
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def billing_svc(tmp_path):
    from services.billing_service import BillingService
    return BillingService(data_dir=str(tmp_path / "billing"))


@pytest.fixture()
def pitr_svc(tmp_path):
    from services.pitr_service import PITRService
    snap_root = tmp_path / "pitr_snapshots"
    snap_root.mkdir()
    return PITRService(snapshot_root=str(snap_root))


# ---------------------------------------------------------------------------
# 1. BillingService — record_usage writes to JSONL
# ---------------------------------------------------------------------------

def test_record_usage_writes_jsonl(billing_svc, tmp_path):
    billing_svc.record_usage("tenant-1", "search", 100, dimensions=128)
    billing_svc.record_usage("tenant-1", "insert", 50, dimensions=128)

    events_file = billing_svc._events_file
    assert events_file.exists(), "billing_events.jsonl should be created"

    lines = events_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["tenant_id"] == "tenant-1"
    assert first["operation"] == "search"
    assert first["count"] == 100


# ---------------------------------------------------------------------------
# 2. BillingService — compute_bill returns correct totals
# ---------------------------------------------------------------------------

def test_compute_bill(billing_svc):
    billing_svc.record_usage("t1", "search", 1_000_000, dimensions=128)
    billing_svc.record_usage("t1", "insert", 500_000, dimensions=128)
    billing_svc.record_usage("t1", "index_build", 10_000, dimensions=128)

    start = datetime.utcnow() - timedelta(hours=1)
    end = datetime.utcnow() + timedelta(hours=1)
    bill = billing_svc.compute_bill("t1", start, end)

    # search: 1_000_000 * 0.000001 = $1.00
    # insert: 500_000 * 0.000002 = $1.00
    # index_build: (10_000/1000) * 0.01 = $0.10
    assert bill.total_usd == pytest.approx(2.10, rel=1e-5)
    assert bill.tenant_id == "t1"
    assert len(bill.line_items) == 3


# ---------------------------------------------------------------------------
# 3. BillingService — export_csv produces valid CSV
# ---------------------------------------------------------------------------

def test_export_csv(billing_svc):
    billing_svc.record_usage("csv-tenant", "search", 10)
    month = datetime.utcnow().strftime("%Y-%m")
    csv_data = billing_svc.export_csv("csv-tenant", month)
    assert "tenant_id" in csv_data
    assert "csv-tenant" in csv_data
    assert "search" in csv_data


# ---------------------------------------------------------------------------
# 4. BillingService — get_usage_summary aggregates correctly
# ---------------------------------------------------------------------------

def test_get_usage_summary(billing_svc):
    billing_svc.record_usage("sum-t", "search", 5)
    billing_svc.record_usage("sum-t", "search", 3)
    billing_svc.record_usage("sum-t", "insert", 7)
    summary = billing_svc.get_usage_summary("sum-t", days=1)
    assert summary["operations"]["search"] == 8
    assert summary["operations"]["insert"] == 7
    assert summary["total_events"] == 3


# ---------------------------------------------------------------------------
# 5. PITRService — create_snapshot and list_snapshots
# ---------------------------------------------------------------------------

def test_pitr_create_and_list(pitr_svc, tmp_path):
    # Create a fake index file so snapshot has something to copy
    fake_index_dir = tmp_path / "indexes"
    fake_index_dir.mkdir()
    fake_file = fake_index_dir / "test-col_hnsw.bin"
    fake_file.write_bytes(b"fake index data")

    snap = pitr_svc.create_snapshot(
        "test-col",
        label="test",
        source_dirs=[str(fake_index_dir)],
    )
    assert snap.snapshot_id.startswith("test-col_")
    assert snap.collection_id == "test-col"
    assert snap.label == "test"

    snaps = pitr_svc.list_snapshots("test-col")
    assert len(snaps) == 1
    assert snaps[0].snapshot_id == snap.snapshot_id


# ---------------------------------------------------------------------------
# 6. PITRService — verify_snapshot passes on intact files
# ---------------------------------------------------------------------------

def test_pitr_verify_snapshot(pitr_svc, tmp_path):
    fake_dir = tmp_path / "data"
    fake_dir.mkdir()
    (fake_dir / "col-x_vectors.json").write_text('{"vectors": []}')

    snap = pitr_svc.create_snapshot(
        "col-x",
        label="verify-test",
        source_dirs=[str(fake_dir)],
    )

    # First call stores checksums
    result = pitr_svc.verify_snapshot(snap.snapshot_id)
    assert result is True

    # Second call compares against stored checksums
    result2 = pitr_svc.verify_snapshot(snap.snapshot_id)
    assert result2 is True


# ---------------------------------------------------------------------------
# 7. PITRService — restore copies files back
# ---------------------------------------------------------------------------

def test_pitr_restore(pitr_svc, tmp_path):
    fake_dir = tmp_path / "indexes"
    fake_dir.mkdir()
    src_file = fake_dir / "restore-col_index.bin"
    src_file.write_bytes(b"restore test")

    snap = pitr_svc.create_snapshot(
        "restore-col",
        source_dirs=[str(fake_dir)],
    )

    # Delete the source file to simulate data loss
    src_file.unlink()

    result = pitr_svc.restore(snap.snapshot_id)
    assert result["success"] is True
    assert len(result["restored_files"]) >= 1


# ---------------------------------------------------------------------------
# 8. BillingService — multi-tenant isolation
# ---------------------------------------------------------------------------

def test_billing_tenant_isolation(billing_svc):
    billing_svc.record_usage("tenant-A", "search", 100)
    billing_svc.record_usage("tenant-B", "search", 999)

    summary_a = billing_svc.get_usage_summary("tenant-A", days=1)
    summary_b = billing_svc.get_usage_summary("tenant-B", days=1)

    assert summary_a["operations"]["search"] == 100
    assert summary_b["operations"]["search"] == 999
