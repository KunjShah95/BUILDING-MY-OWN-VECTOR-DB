"""Tests for WAL crash recovery: checkpoint, replay, corruption tolerance."""

import numpy as np
import pytest

from utils.wal import WriteAheadLog
from utils.hnsw_index import HNSWIndex


@pytest.fixture
def wal(tmp_path):
    return WriteAheadLog("col_test", log_dir=str(tmp_path))


def _rand(dim=8):
    return list(np.random.rand(dim).astype(float))


def test_log_and_read_all(wal):
    wal.log_insert("v1", _rand(), {"n": 1})
    wal.log_delete("v2")
    wal.log_update_metadata("v3", {"k": "v"})
    entries = wal.read_all()
    assert [e["op"] for e in entries] == ["INSERT", "DELETE", "UPDATE_META"]


def test_checkpoint_truncates_and_marks(wal):
    wal.log_insert("v1", _rand())
    assert wal.last_checkpoint_ts() == 0.0
    wal.checkpoint()
    assert wal.last_checkpoint_ts() > 0.0
    assert wal.read_all() == []


def test_pending_entries_only_after_checkpoint(wal):
    wal.log_insert("old", _rand())
    wal.checkpoint()
    wal.log_insert("new", _rand())
    pending = wal.pending_entries()
    assert len(pending) == 1
    assert pending[0]["data"]["id"] == "new"


def test_replay_reapplies_inserts_and_deletes(wal):
    idx = HNSWIndex()
    for i in range(5):
        idx.insert(_rand(), f"v{i}", {"n": i})
    wal.checkpoint()  # snapshot point
    for i in range(5, 10):
        wal.log_insert(f"v{i}", _rand(), {"n": i})
    wal.log_delete("v0")

    summary = idx_recover(wal, idx)
    assert summary["by_op"]["INSERT"] == 5
    assert summary["by_op"]["DELETE"] == 1
    assert "v9" in idx.graph
    assert "v0" in idx.deleted


def test_replay_skips_corrupt_lines(wal, tmp_path):
    wal.log_insert("v1", _rand())
    # Append a corrupt line directly
    with open(wal.wal_path, "a") as f:
        f.write("{not valid json\n")
    entries = wal.read_all()
    assert len(entries) == 1  # corrupt line dropped


def idx_recover(wal, idx):
    return wal.replay(idx)


def test_replay_empty_wal_noop(wal):
    idx = HNSWIndex()
    idx.insert(_rand(), "v0")
    summary = wal.replay(idx)
    assert summary["replayed"] == 0


def test_replay_into_ivf_index_via_add_fallback(wal):
    """WAL replay must work with IVFIndex (add / delete_vector method names)."""
    from utils.ivf_index import IVFIndex

    idx = IVFIndex(n_clusters=4)
    train_data = [_rand() for _ in range(40)]
    idx.train(train_data)
    for i, v in enumerate(train_data):
        idx.add(v, f"v{i}")

    wal.checkpoint()
    for i in range(40, 45):
        wal.log_insert(f"v{i}", _rand(), None)
    wal.log_delete("v0")

    summary = wal.replay(idx)
    assert summary["by_op"]["INSERT"] == 5
    assert summary["by_op"]["DELETE"] == 1
    assert summary["skipped"] == 0
