"""Tests for storage backends."""
import pytest
import tempfile
from services.storage_backend import LocalStorageBackend, StorageFactory


class TestLocalStorage:
    def test_save_load_bytes(self, tmp_path):
        backend = LocalStorageBackend()
        path = str(tmp_path / "test.bin")
        assert backend.save(path, b"hello world")
        data = backend.load(path)
        assert data == b"hello world"

    def test_exists(self, tmp_path):
        backend = LocalStorageBackend()
        path = str(tmp_path / "test.bin")
        assert not backend.exists(path)
        backend.save(path, b"data")
        assert backend.exists(path)

    def test_delete(self, tmp_path):
        backend = LocalStorageBackend()
        path = str(tmp_path / "test.bin")
        backend.save(path, b"data")
        assert backend.delete(path)
        assert not backend.exists(path)

    def test_list_keys(self, tmp_path):
        backend = LocalStorageBackend()
        p1 = str(tmp_path / "a.txt")
        p2 = str(tmp_path / "b.txt")
        backend.save(p1, b"a")
        backend.save(p2, b"b")
        keys = backend.list_keys(str(tmp_path))
        assert len(keys) >= 2

    def test_save_load_fileobj(self, tmp_path):
        import io
        backend = LocalStorageBackend()
        path = str(tmp_path / "test.bin")
        buf = io.BytesIO(b"stream data")
        assert backend.save(path, buf)
        data = backend.load(path)
        assert data == b"stream data"

    def test_delete_nonexistent(self, tmp_path):
        backend = LocalStorageBackend()
        path = str(tmp_path / "nonexistent.bin")
        assert not backend.delete(path)

    def test_load_nonexistent(self):
        backend = LocalStorageBackend()
        assert backend.load("/tmp/__nonexistent_file_xyz__") is None

    def test_list_keys_empty(self, tmp_path):
        backend = LocalStorageBackend()
        empty_dir = str(tmp_path / "empty")
        keys = backend.list_keys(empty_dir)
        assert keys == []

    def test_list_keys_nonexistent(self):
        backend = LocalStorageBackend()
        keys = backend.list_keys("/tmp/__nonexistent_dir_xyz__")
        assert keys == []


class TestStorageFactory:
    def test_create_local_default(self):
        backend = StorageFactory.create("local")
        from services.storage_backend import LocalStorageBackend
        assert isinstance(backend, LocalStorageBackend)

    def test_create_local_none(self):
        backend = StorageFactory.create(None)
        from services.storage_backend import LocalStorageBackend
        assert isinstance(backend, LocalStorageBackend)

    def test_create_local_empty(self):
        backend = StorageFactory.create("")
        from services.storage_backend import LocalStorageBackend
        assert isinstance(backend, LocalStorageBackend)
