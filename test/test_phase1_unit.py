"""Unit tests for Phase 1 metadata and index path helpers."""

import os

from models.vector_model import VectorModel
from utils.index_paths import get_hnsw_path, get_ivf_path, get_index_dir
from utils.metadata_contract import build_vector_metadata


class _StubModel(VectorModel):
    """VectorModel without DB for filter helper tests."""

    def __init__(self):
        pass


def test_metadata_matches_exact_keys():
    model = _StubModel()
    meta = {"modality": "text", "collection_id": "docs", "tag": "a"}
    assert model._metadata_matches(meta, {"modality": "text"}) is True
    assert model._metadata_matches(meta, {"modality": "image"}) is False
    assert model._metadata_matches(meta, {"tag": "a", "modality": "text"}) is True
    assert model._metadata_matches(None, {"modality": "text"}) is False


def test_build_vector_metadata_merges_extra():
    collection = {
        "collection_id": "my-docs",
        "modality": "text",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
    }
    meta = build_vector_metadata(collection, extra={"text": "hello", "tags": ["a"]})
    assert meta["collection_id"] == "my-docs"
    assert meta["modality"] == "text"
    assert meta["dimension"] == 384
    assert meta["text"] == "hello"
    assert meta["tags"] == ["a"]


def test_index_paths_global_and_collection():
    assert get_hnsw_path() == "hnsw_index_data.json"
    assert get_hnsw_path("product-catalog") == os.path.join(
        "indexes", "product-catalog", "hnsw.json"
    )
    assert get_ivf_path("product-catalog") == os.path.join(
        "indexes", "product-catalog", "ivf.json"
    )
    assert get_index_dir("product-catalog") == os.path.join("indexes", "product-catalog")
    assert get_index_dir() == "."
