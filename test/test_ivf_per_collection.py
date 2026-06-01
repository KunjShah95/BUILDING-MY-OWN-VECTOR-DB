import pytest
from unittest.mock import MagicMock, patch
import numpy as np


class TestCollectionIVF:
    def test_ivf_index_scoped(self):
        from database.ivf_database import IVFVectorDatabase

        db = IVFVectorDatabase(db_session=None)
        db.vector_model = MagicMock()
        db.vector_model.get_vectors_by_collection.return_value = [
            MagicMock(vector_data=[0.1, 0.2], vector_id="a1", meta_data={})
        ]
        db.vector_model.get_vector.return_value = MagicMock(meta_data={})

        result = db.create_ivf_index(n_clusters=1, n_probes=1, collection_id="coll-a")
        assert result["success"], f"Failed: {result.get('message')}"

        idx = db.get_ivf_index("coll-a")
        assert idx is not None, "Scoped IVF index should exist"
        assert idx.is_trained

    def test_ivf_index_isolation(self):
        from database.ivf_database import IVFVectorDatabase

        db = IVFVectorDatabase(db_session=None)
        db.vector_model = MagicMock()

        def mock_get_vectors(collection_id):
            if collection_id == "coll-a":
                return [MagicMock(vector_data=[0.1, 0.2], vector_id="a1", meta_data={})]
            if collection_id == "coll-b":
                return [MagicMock(vector_data=[0.9, 0.8], vector_id="b1", meta_data={})]
            return []

        db.vector_model.get_vectors_by_collection.side_effect = mock_get_vectors
        db.vector_model.get_vector.return_value = MagicMock(meta_data={})

        r1 = db.create_ivf_index(n_clusters=1, n_probes=1, collection_id="coll-a")
        assert r1["success"]

        r2 = db.create_ivf_index(n_clusters=1, n_probes=1, collection_id="coll-b")
        assert r2["success"]

        idx_a = db.get_ivf_index("coll-a")
        idx_b = db.get_ivf_index("coll-b")
        assert idx_a is not None
        assert idx_b is not None
        assert idx_a is not idx_b

    def test_ivf_global_vs_scoped(self):
        from database.ivf_database import IVFVectorDatabase

        db = IVFVectorDatabase(db_session=None)
        db.vector_model = MagicMock()
        db.vector_model.get_all_vectors.return_value = [
            MagicMock(vector_data=[0.5, 0.5], vector_id="g1", meta_data={})
        ]
        db.vector_model.get_vectors_by_collection.return_value = [
            MagicMock(vector_data=[0.1, 0.2], vector_id="a1", meta_data={})
        ]
        db.vector_model.get_vector.return_value = MagicMock(meta_data={})

        r_global = db.create_ivf_index(n_clusters=1, n_probes=1)
        assert r_global["success"]

        r_scoped = db.create_ivf_index(n_clusters=1, n_probes=1, collection_id="coll-a")
        assert r_scoped["success"]

        assert db.ivf_index is not None
        assert db.get_ivf_index() is db.ivf_index
        assert db.get_ivf_index("coll-a") is not db.ivf_index

    def test_ivf_index_not_found(self):
        from database.ivf_database import IVFVectorDatabase

        db = IVFVectorDatabase(db_session=None)
        assert db.get_ivf_index("nonexistent") is None

    def test_get_ivf_index_stats(self):
        from database.ivf_database import IVFVectorDatabase

        db = IVFVectorDatabase(db_session=None)
        db.vector_model = MagicMock()
        db.vector_model.get_vectors_by_collection.return_value = [
            MagicMock(vector_data=[0.1, 0.2], vector_id="a1", meta_data={})
        ]
        db.vector_model.get_vector.return_value = MagicMock(meta_data={})

        db.create_ivf_index(n_clusters=1, n_probes=1, collection_id="coll-a")
        stats = db.get_index_stats(collection_id="coll-a")
        assert stats["success"]
        assert stats["stats"]["total_vectors"] == 1
        assert stats["stats"]["n_clusters"] == 1
        assert stats["stats"]["n_probes"] == 1

    def test_ivf_index_rebuild(self):
        from database.ivf_database import IVFVectorDatabase

        db = IVFVectorDatabase(db_session=None)
        db.vector_model = MagicMock()
        db.vector_model.get_vectors_by_collection.return_value = [
            MagicMock(vector_data=[0.1, 0.2], vector_id="a1", meta_data={})
        ]
        db.vector_model.get_vector.return_value = MagicMock(meta_data={})

        r1 = db.create_ivf_index(n_clusters=1, n_probes=1, collection_id="coll-a")
        assert r1["success"]

        r2 = db.create_ivf_index(n_clusters=2, n_probes=2, collection_id="coll-a", force_rebuild=True)
        assert r2["success"]
        assert r2["stats"]["n_clusters"] == 2

    def test_ivf_index_exists(self):
        from database.ivf_database import IVFVectorDatabase

        db = IVFVectorDatabase(db_session=None)
        db.vector_model = MagicMock()
        db.vector_model.get_vectors_by_collection.return_value = [
            MagicMock(vector_data=[0.1, 0.2], vector_id="a1", meta_data={})
        ]
        db.vector_model.get_vector.return_value = MagicMock(meta_data={})

        db.create_ivf_index(n_clusters=1, n_probes=1, collection_id="coll-a")
        result = db.create_ivf_index(n_clusters=2, n_probes=2, collection_id="coll-a")
        assert not result["success"]
        assert "already exists" in result["message"].lower()
