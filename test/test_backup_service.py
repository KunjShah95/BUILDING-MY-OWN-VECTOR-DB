"""Tests for Backup & Restore Service (Phase 8)."""

import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from services.backup_service import BackupService


class TestBackupServiceInit:
    def test_init_creates_directories(self, tmp_path):
        snap = str(tmp_path / "snap")
        wal = str(tmp_path / "wal")
        svc = BackupService(snapshot_dir=snap, wal_archive_dir=wal)
        assert os.path.exists(snap)
        assert os.path.exists(wal)
        assert svc.cloud_backend is None

    def test_init_with_cloud(self):
        svc = BackupService(cloud_backend="s3")
        assert svc.cloud_backend == "s3"
        assert svc.cloud_prefix == "vector-db-backups"


class TestBackupSnapshots:
    def test_create_snapshot_with_save_binary(self, tmp_path):
        svc = BackupService(snapshot_dir=str(tmp_path / "snap"), wal_archive_dir=str(tmp_path / "wal"))
        mock_index = MagicMock()
        mock_index.save_binary = MagicMock()
        mock_index.graph = {"n1": {}, "n2": {}}

        result = svc.create_snapshot(mock_index, collection_id="col_001", method="hnsw")
        assert result["success"] is True
        assert result["snapshot"]["collection_id"] == "col_001"
        assert result["snapshot"]["method"] == "hnsw"
        assert "timestamp" in result["snapshot"]
        assert os.path.exists(result["snapshot"]["path"])
        # Check meta file
        meta_path = os.path.join(result["snapshot"]["path"], "snapshot_meta.json")
        assert os.path.exists(meta_path)

    def test_create_snapshot_with_save(self, tmp_path):
        svc = BackupService(
            snapshot_dir=str(tmp_path / "snap"),
            wal_archive_dir=str(tmp_path / "wal"),
        )
        # Use a simple object (not MagicMock) so save_binary doesn't auto-exist
        class MockIndexWithSave:
            def __init__(self):
                self.graph = {"n1": {}}
                self.save_called = False
            def save(self, path):
                self.save_called = True
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write("{}")

        mock_index = MockIndexWithSave()

        result = svc.create_snapshot(mock_index, collection_id="col_002")
        assert result["success"] is True
        assert mock_index.save_called is True

    def test_create_snapshot_no_save_method(self, tmp_path):
        svc = BackupService(
            snapshot_dir=str(tmp_path / "snap"),
            wal_archive_dir=str(tmp_path / "wal"),
        )
        mock_index = object()  # No save or save_binary

        result = svc.create_snapshot(mock_index, collection_id="col_003")
        assert result["success"] is False

    def test_create_snapshot_failure_handled(self, tmp_path):
        svc = BackupService(
            snapshot_dir=str(tmp_path / "snap"),
            wal_archive_dir=str(tmp_path / "wal"),
        )
        mock_index = MagicMock()
        mock_index.save_binary.side_effect = Exception("disk error")

        result = svc.create_snapshot(mock_index)
        assert result["success"] is False
        assert "disk error" in result["message"]


class TestBackupListRestore:
    def test_list_snapshots_empty(self, tmp_path):
        svc = BackupService(
            snapshot_dir=str(tmp_path / "empty_snap"),
            wal_archive_dir=str(tmp_path / "wal"),
        )
        assert svc.list_snapshots() == []

    def test_list_snapshots(self, tmp_path):
        svc = BackupService(
            snapshot_dir=str(tmp_path / "snap"),
            wal_archive_dir=str(tmp_path / "wal"),
        )
        mock_index = MagicMock()
        mock_index.save_binary = MagicMock()
        mock_index.graph = {"v1": {}}
        svc.create_snapshot(mock_index, collection_id="col_a")
        svc.create_snapshot(mock_index, collection_id="col_b")

        snapshots = svc.list_snapshots()
        assert len(snapshots) == 2

    def test_list_snapshots_filter_by_collection(self, tmp_path):
        svc = BackupService(
            snapshot_dir=str(tmp_path / "snap_filter"),
            wal_archive_dir=str(tmp_path / "wal"),
        )
        mock_index = MagicMock()
        mock_index.save_binary = MagicMock()
        mock_index.graph = {}
        svc.create_snapshot(mock_index, collection_id="target")
        svc.create_snapshot(mock_index, collection_id="other")

        filtered = svc.list_snapshots(collection_id="target")
        assert len(filtered) == 1
        assert filtered[0]["collection_id"] == "target"

    def test_restore_snapshot_found(self, tmp_path):
        svc = BackupService(
            snapshot_dir=str(tmp_path / "snap_r"),
            wal_archive_dir=str(tmp_path / "wal"),
        )
        mock_index = MagicMock()
        mock_index.save_binary = MagicMock()
        mock_index.graph = {}
        result = svc.create_snapshot(mock_index, collection_id="col_x")
        ts = result["snapshot"]["timestamp"]

        restore_result = svc.restore_snapshot("col_x", ts)
        assert restore_result["success"] is True

    def test_restore_snapshot_not_found(self, tmp_path):
        svc = BackupService(
            snapshot_dir=str(tmp_path / "snap_nf"),
            wal_archive_dir=str(tmp_path / "wal"),
        )
        result = svc.restore_snapshot("nonexistent", "2000-01-01")
        assert result["success"] is False


class TestBackupWALArchive:
    def test_archive_wal(self, tmp_path):
        wal_dir = str(tmp_path / "wal_logs")
        os.makedirs(wal_dir, exist_ok=True)
        wal_file = os.path.join(wal_dir, "col_001.wal")
        with open(wal_file, "w") as f:
            f.write("test wal entry\n")

        svc = BackupService(
            snapshot_dir=str(tmp_path / "snap"),
            wal_archive_dir=str(tmp_path / "wal_archive"),
        )
        result = svc.archive_wal("col_001", wal_dir=wal_dir)
        assert result["success"] is True
        assert "archive_path" in result
        assert result["size_bytes"] > 0

    def test_archive_wal_not_found(self, tmp_path):
        svc = BackupService(
            snapshot_dir=str(tmp_path / "snap"),
            wal_archive_dir=str(tmp_path / "wal_archive"),
        )
        result = svc.archive_wal("nonexistent", wal_dir="/nonexistent")
        assert result["success"] is False

    def test_list_wal_archives(self, tmp_path):
        svc = BackupService(
            snapshot_dir=str(tmp_path / "snap"),
            wal_archive_dir=str(tmp_path / "wal_archive"),
        )
        wal_dir = str(tmp_path / "wal_logs")
        os.makedirs(wal_dir, exist_ok=True)
        with open(os.path.join(wal_dir, "col_001.wal"), "w") as f:
            f.write("data")
        svc.archive_wal("col_001", wal_dir=wal_dir)
        svc.archive_wal("col_001", wal_dir=wal_dir)

        archives = svc.list_wal_archives()
        assert len(archives) == 2

    def test_list_wal_archives_filtered(self, tmp_path):
        svc = BackupService(
            snapshot_dir=str(tmp_path / "snap"),
            wal_archive_dir=str(tmp_path / "wal_archive"),
        )
        wal_dir = str(tmp_path / "wal_logs")
        os.makedirs(wal_dir, exist_ok=True)
        for coll in ["a", "b"]:
            with open(os.path.join(wal_dir, f"{coll}.wal"), "w") as f:
                f.write("x")
            svc.archive_wal(coll, wal_dir=wal_dir)

        a_archives = svc.list_wal_archives(collection_id="a")
        assert len(a_archives) == 1


class TestBackupCloud:
    def test_cloud_upload_s3_not_configured(self, tmp_path):
        svc = BackupService(
            snapshot_dir=str(tmp_path / "snap"),
            wal_archive_dir=str(tmp_path / "wal"),
            cloud_backend="s3",
        )
        # Without proper AWS creds, _upload_to_cloud should return False gracefully
        mock_index = MagicMock()
        mock_index.save_binary = MagicMock()
        mock_index.graph = {}
        result = svc.create_snapshot(mock_index)
        # Snapshot itself should succeed even if cloud upload fails
        assert result["success"] is True
        # Cloud upload failure is logged but doesn't crash

    @patch("services.backup_service.shutil.copy2")
    def test_cloud_upload_logs_failure(self, mock_copy, tmp_path):
        svc = BackupService(
            snapshot_dir=str(tmp_path / "snap"),
            wal_archive_dir=str(tmp_path / "wal"),
            cloud_backend="azure",
        )
        result = svc._upload_to_cloud(str(tmp_path))
        assert result is False  # No connection string


class TestBackupScheduler:
    def test_start_stop_scheduler(self, tmp_path):
        svc = BackupService(
            snapshot_dir=str(tmp_path / "snap"),
            wal_archive_dir=str(tmp_path / "wal"),
        )
        mock_index = MagicMock()
        mock_index.graph = {}

        svc.start_scheduler(get_index_fn=lambda: mock_index, interval_hours=999)
        assert svc._running is True
        svc.stop_scheduler()
        assert svc._running is False

    def test_double_start_no_op(self, tmp_path):
        svc = BackupService(
            snapshot_dir=str(tmp_path / "snap"),
            wal_archive_dir=str(tmp_path / "wal"),
        )
        svc.start_scheduler(get_index_fn=lambda: None, interval_hours=999)
        svc.start_scheduler(get_index_fn=lambda: None, interval_hours=999)
        svc.stop_scheduler()

    def test_get_status(self, tmp_path):
        svc = BackupService(
            snapshot_dir=str(tmp_path / "snap"),
            wal_archive_dir=str(tmp_path / "wal"),
        )
        status = svc.get_status()
        assert "snapshot_dir" in status
        assert "wal_archive_dir" in status
        assert "scheduler_running" in status
        assert status["scheduler_running"] is False
