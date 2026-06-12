"""Tests for startup WAL recovery helpers (no DB required)."""

import numpy as np

from utils.wal import WriteAheadLog
from services.startup_recovery import _collections_with_pending_wal


def _rand(dim=8):
    return list(np.random.rand(dim).astype(float))


def test_no_pending_when_dir_missing(tmp_path):
    assert _collections_with_pending_wal(str(tmp_path / "nope")) == []


def test_detects_pending_wal(tmp_path):
    log_dir = str(tmp_path)
    wal = WriteAheadLog("colA", log_dir=log_dir)
    wal.log_insert("v1", _rand())
    pending = _collections_with_pending_wal(log_dir)
    assert "colA" in pending


def test_checkpointed_wal_not_pending(tmp_path):
    log_dir = str(tmp_path)
    wal = WriteAheadLog("colB", log_dir=log_dir)
    wal.log_insert("v1", _rand())
    wal.checkpoint()  # everything persisted
    pending = _collections_with_pending_wal(log_dir)
    assert "colB" not in pending


def test_only_collections_with_new_writes(tmp_path):
    log_dir = str(tmp_path)
    a = WriteAheadLog("a", log_dir=log_dir)
    a.log_insert("v1", _rand())
    a.checkpoint()
    a.log_insert("v2", _rand())  # new write after checkpoint -> pending

    b = WriteAheadLog("b", log_dir=log_dir)
    b.log_insert("v1", _rand())
    b.checkpoint()  # no new writes -> not pending

    pending = set(_collections_with_pending_wal(log_dir))
    assert "a" in pending
    assert "b" not in pending
