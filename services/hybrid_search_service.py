"""Hybrid dense + sparse search service."""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from services.embedding_service import embed_text
from utils.bm25_index import BM25Index


class HybridSearchService:
    """
    Combines dense vector similarity with BM25 sparse retrieval.
    Uses reciprocal rank fusion (RRF) for result merging.
    """

    def __init__(self, db_session=None):
        self.db = db_session
        self.sparse_indexes: Dict[str, BM25Index] = {}

    def get_sparse_index(self, collection_id: str) -> Optional[BM25Index]:
        if collection_id not in self.sparse_indexes:
            path = f"indexes/{collection_id}/sparse.json"
            import os
            if os.path.exists(path):
                self.sparse_indexes[collection_id] = BM25Index.load(path)
        return self.sparse_indexes.get(collection_id)

    def build_sparse_index(self, collection_id: str):
        from services.collection_service import CollectionService
        from models.vector_model import VectorModel
        coll_svc = CollectionService(self.db)
        vec_model = VectorModel(self.db)

        coll = coll_svc.get_collection(collection_id)
        if not coll.get("success"):
            return coll

        vectors = vec_model.get_vectors_by_collection(collection_id)

        idx = BM25Index()
        for v in vectors:
            meta = v.meta_data or {}
            text = meta.get("text") or meta.get("content", "")
            if text:
                idx.add_document(text, v.vector_id)

        import os
        os.makedirs(f"indexes/{collection_id}", exist_ok=True)
        idx.save(f"indexes/{collection_id}/sparse.json")
        self.sparse_indexes[collection_id] = idx

        return {"success": True, "message": f"Sparse index built with {idx.total_docs} documents"}

    def search(
        self,
        collection_id: str,
        query: str,
        k: int = 10,
        alpha: float = 0.5,
        dense_k: int = 20,
        sparse_k: int = 20,
    ) -> Dict[str, Any]:
        from services.collection_index_service import CollectionIndexService
        from services.collection_service import CollectionService

        coll_svc = CollectionService(self.db)
        coll = coll_svc.get_collection(collection_id)
        if not coll.get("success"):
            return coll

        query_vector = embed_text(query)
        index_svc = CollectionIndexService(self.db)
        dense_results = index_svc.search_collection_indexed(
            collection_id=collection_id,
            query_vector=query_vector,
            k=dense_k,
            method="brute",
            distance_metric="cosine",
        )

        sparse_idx = self.get_sparse_index(collection_id)
        if sparse_idx:
            sparse_raw = sparse_idx.search(query, k=sparse_k)
            sparse_results = [
                {"id": doc_id, "score": score, "vector_id": doc_id}
                for doc_id, score in sparse_raw
            ]
        else:
            sparse_results = []

        rrf_scores = {}
        constant = 60

        dense_list = dense_results.get("results", []) if dense_results.get("success") else []
        for rank, r in enumerate(dense_list):
            vid = r.get("vector_id") or r.get("id")
            rrf_scores[vid] = rrf_scores.get(vid, 0) + alpha / (rank + constant)

        for rank, r in enumerate(sparse_results):
            vid = r.get("vector_id") or r.get("id")
            rrf_scores[vid] = rrf_scores.get(vid, 0) + (1 - alpha) / (rank + constant)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        return {
            "success": True,
            "results": [
                {"vector_id": vid, "hybrid_score": score}
                for vid, score in ranked
            ],
            "total_results": len(ranked),
            "query": query,
            "method": "hybrid",
            "alpha": alpha,
        }
