"""
Tests for Phase 13: Plugin SDK, Registry, Loader, and Marketplace API.
Run with: pytest test/test_plugin_sdk.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import pytest

# Ensure repo root is in path
_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from services.plugin_registry import (
    PluginManifest,
    PluginRegistry,
    _parse_semver,
)
from sdk.plugin_sdk.base_index import BaseIndexPlugin
from sdk.plugin_sdk.base_encoder import BaseEncoderPlugin
from sdk.plugin_sdk.base_storage import BaseStoragePlugin
from services.plugin_loader import PluginLoader, PluginResourceError


# ── helpers ────────────────────────────────────────────────────────────────

def make_manifest(**kwargs):
    defaults = dict(
        name="test_plugin",
        version="1.0.0",
        plugin_type="index",
        entry_point="test.entry",
        description="Test",
        author="Tester",
        requires=[],
    )
    defaults.update(kwargs)
    return PluginManifest(**defaults)


class DummyIndex(BaseIndexPlugin):
    def __init__(self):
        self._data = {}

    def train(self, vectors):
        pass

    def add(self, vector, id, metadata):
        self._data[id] = (vector, metadata)

    def search(self, query, k):
        return [{"id": k, "score": 0.0, "metadata": {}} for k in list(self._data)[:k]]

    def delete(self, id):
        return self._data.pop(id, None) is not None

    def get_stats(self):
        return {"vector_count": len(self._data), "dimension": 0, "index_type": "dummy"}

    def save(self, path):
        pass

    def load(self, path):
        pass


class DummyEncoder(BaseEncoderPlugin):
    def encode(self, text):
        return [0.0] * 4

    def encode_batch(self, texts):
        return [[0.0] * 4 for _ in texts]

    def dimension(self):
        return 4


class DummyStorage(BaseStoragePlugin):
    def __init__(self):
        self._store = {}

    def put(self, key, data):
        self._store[key] = data

    def get(self, key):
        if key not in self._store:
            raise KeyError(key)
        return self._store[key]

    def delete(self, key):
        return self._store.pop(key, None) is not None

    def list_keys(self, prefix=""):
        return [k for k in self._store if k.startswith(prefix)]


# ── Test 1: PluginManifest validation ─────────────────────────────────────

def test_manifest_valid():
    m = make_manifest()
    assert m.name == "test_plugin"
    assert m.semver == (1, 0, 0)


def test_manifest_invalid_type():
    with pytest.raises(ValueError, match="plugin_type"):
        make_manifest(plugin_type="unknown")


def test_manifest_invalid_semver():
    with pytest.raises(ValueError, match="semver"):
        make_manifest(version="1.0")


# ── Test 2: PluginRegistry register/get/list/unregister ──────────────────

def test_registry_register_and_get(tmp_path):
    reg = PluginRegistry(registry_file=str(tmp_path / "registry.json"))
    m = make_manifest()
    reg.register(m, DummyIndex)
    got_m, got_cls = reg.get("test_plugin")
    assert got_m.version == "1.0.0"
    assert got_cls is DummyIndex


def test_registry_latest_version(tmp_path):
    reg = PluginRegistry(registry_file=str(tmp_path / "registry.json"))
    reg.register(make_manifest(version="1.0.0"), DummyIndex)
    reg.register(make_manifest(version="2.0.0"), DummyIndex)
    reg.register(make_manifest(version="1.5.0"), DummyIndex)
    m, _ = reg.get("test_plugin")
    assert m.version == "2.0.0"


def test_registry_list_plugins(tmp_path):
    reg = PluginRegistry(registry_file=str(tmp_path / "registry.json"))
    reg.register(make_manifest(name="idx", plugin_type="index"), DummyIndex)
    reg.register(make_manifest(name="enc", plugin_type="encoder", entry_point="e"), DummyEncoder)
    all_p = reg.list_plugins()
    assert len(all_p) == 2
    index_p = reg.list_plugins(plugin_type="index")
    assert len(index_p) == 1 and index_p[0].name == "idx"


def test_registry_unregister(tmp_path):
    reg = PluginRegistry(registry_file=str(tmp_path / "registry.json"))
    reg.register(make_manifest(), DummyIndex)
    removed = reg.unregister("test_plugin", "1.0.0")
    assert removed is True
    with pytest.raises(KeyError):
        reg.get("test_plugin")


def test_registry_persistence(tmp_path):
    reg_file = str(tmp_path / "registry.json")
    reg1 = PluginRegistry(registry_file=reg_file)
    reg1.register(make_manifest(), DummyIndex)

    # Load fresh instance from same file
    reg2 = PluginRegistry(registry_file=reg_file)
    m, cls = reg2.get("test_plugin")
    assert m.name == "test_plugin"
    # class is None after load (not serialized)
    assert cls is None


# ── Test 3: Base classes are abstract ─────────────────────────────────────

def test_base_index_is_abstract():
    with pytest.raises(TypeError):
        BaseIndexPlugin()  # type: ignore[abstract]


def test_base_encoder_is_abstract():
    with pytest.raises(TypeError):
        BaseEncoderPlugin()  # type: ignore[abstract]


def test_base_storage_is_abstract():
    with pytest.raises(TypeError):
        BaseStoragePlugin()  # type: ignore[abstract]


# ── Test 4: Concrete dummy implementations work ────────────────────────────

def test_dummy_index_operations():
    idx = DummyIndex()
    idx.train([[1.0, 0.0]])
    idx.add([1.0, 0.0], "v1", {"tag": "a"})
    results = idx.search([1.0, 0.0], k=1)
    assert results[0]["id"] == "v1"
    assert idx.delete("v1") is True
    assert idx.delete("v1") is False
    stats = idx.get_stats()
    assert stats["vector_count"] == 0


def test_dummy_encoder():
    enc = DummyEncoder()
    assert enc.dimension() == 4
    vec = enc.encode("hello")
    assert len(vec) == 4
    batch = enc.encode_batch(["a", "b"])
    assert len(batch) == 2


def test_dummy_storage():
    st = DummyStorage()
    st.put("k1", b"data")
    assert st.get("k1") == b"data"
    keys = st.list_keys("")
    assert "k1" in keys
    assert st.delete("k1") is True
    assert st.delete("k1") is False
    with pytest.raises(KeyError):
        st.get("k1")


# ── Test 5: PluginLoader.load_from_file ────────────────────────────────────

_PLUGIN_SOURCE = '''
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from sdk.plugin_sdk.base_index import BaseIndexPlugin
from typing import Any, Dict, List

class FilePlugin(BaseIndexPlugin):
    def __init__(self):
        self._d = {}
    def train(self, vectors): pass
    def add(self, vector, id, metadata): self._d[id] = vector
    def search(self, query, k): return [{"id": i, "score": 0.0, "metadata": {}} for i in list(self._d)[:k]]
    def delete(self, id): return self._d.pop(id, None) is not None
    def get_stats(self): return {"vector_count": len(self._d), "dimension": 0, "index_type": "file"}
    def save(self, path): pass
    def load(self, path): pass
'''


def test_loader_load_from_file(tmp_path):
    plugin_file = tmp_path / "file_plugin.py"
    plugin_file.write_text(_PLUGIN_SOURCE)
    loader = PluginLoader()
    cls = loader.load_from_file(str(plugin_file))
    assert issubclass(cls, BaseIndexPlugin)
    instance = cls()
    instance.add([1.0], "x", {})
    assert instance.get_stats()["vector_count"] == 1


def test_loader_missing_file():
    loader = PluginLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_from_file("/nonexistent/path/plugin.py")


# ── Test 6: BruteForcePlugin (example plugin) ─────────────────────────────

def test_brute_force_plugin():
    from sdk.plugin_sdk.example_plugin import BruteForcePlugin
    p = BruteForcePlugin(dimension=3, metric="cosine")
    p.train([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p.add([1.0, 0.0, 0.0], "a", {"tag": "first"})
    p.add([0.0, 1.0, 0.0], "b", {"tag": "second"})
    results = p.search([1.0, 0.0, 0.0], k=2)
    assert results[0]["id"] == "a"
    assert results[0]["metadata"]["tag"] == "first"
    stats = p.get_stats()
    assert stats["vector_count"] == 2
    assert p.delete("a") is True
    assert p.get_stats()["vector_count"] == 1


def test_brute_force_plugin_save_load(tmp_path):
    from sdk.plugin_sdk.example_plugin import BruteForcePlugin
    p = BruteForcePlugin(dimension=2, metric="euclidean")
    p.add([1.0, 2.0], "v1", {})
    path = str(tmp_path / "bf.json")
    p.save(path)
    p2 = BruteForcePlugin(dimension=2, metric="euclidean")
    p2.load(path)
    assert p2.get_stats()["vector_count"] == 1


# ── Test 7: @plugin decorator auto-registers ──────────────────────────────

def test_plugin_decorator(tmp_path):
    from services.plugin_registry import PluginRegistry
    import services.plugin_registry as pr_module

    # Swap global registry temporarily
    original = pr_module._global_registry
    pr_module._global_registry = PluginRegistry(registry_file=str(tmp_path / "r.json"))

    try:
        from sdk.plugin_sdk.decorators import plugin

        @plugin(name="decorated", version="0.1.0", plugin_type="index")
        class DecoratedPlugin(DummyIndex):
            pass

        reg = pr_module.get_registry()
        m, cls = reg.get("decorated")
        assert m.version == "0.1.0"
        assert cls is DecoratedPlugin
    finally:
        pr_module._global_registry = original


# ── Test 8: semver comparison ─────────────────────────────────────────────

def test_semver_parsing():
    assert _parse_semver("1.2.3") == (1, 2, 3)
    assert _parse_semver("0.0.1") == (0, 0, 1)
    with pytest.raises(ValueError):
        _parse_semver("1.2")
    with pytest.raises(ValueError):
        _parse_semver("a.b.c")
