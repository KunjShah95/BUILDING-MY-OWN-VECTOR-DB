"""
Learning-to-Rank (LTR) Service (Phase 7: Production Search Orchestration).

Replaces static Reciprocal Rank Fusion (RRF) with a trained XGBoost/LightGBM
model that learns to rank search results from user click-through feedback.

Pipeline:
  1. Collect training data from /playground/feedback (search queries, result
     features, user clicks).
  2. Train a LambdaMART model (XGBoost ranker) on labeled query-document pairs.
  3. At search time, extract the same features for candidate results and
     predict relevance scores.
  4. Return results sorted by predicted relevance.

Feature engineering:
  - Dense similarity score (cosine distance to query)
  - Sparse similarity score (BM25 score)
  - Cross-encoder score (if reranker available)
  - RRF score (from fusion)
  - Document-level features: freshness, popularity, vector norm
  - Query-level features: length, intent type
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LTRFeature:
    """A single feature for the LTR model."""

    name: str
    value: float
    group: str = "default"


@dataclass
class LTRTrainingExample:
    """A labeled query-document pair for LTR training."""

    qid: int  # query group ID
    features: List[float]
    relevance: float  # 0 = irrelevant, 1 = relevant, 2 = highly relevant, 3 = perfect
    query_text: str = ""
    document_id: str = ""
    document_text: str = ""


class LTRService:
    """Learning-to-Rank service with XGBoost LambdaMART.

    Usage::

        ltr = LTRService()
        features = ltr.extract_features(query, candidates)
        scores = ltr.predict(features)  # if model is trained
        ltr.train(examples)  # train from labeled data
    """

    def __init__(self, model_path: str = "ltr_model.pkl"):
        self.model_path = model_path
        self._model: Any = None
        self._feature_names: List[str] = []
        self._lock = threading.RLock()

        # Try to load existing model
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    self._model = pickle.load(f)
                logger.info("Loaded LTR model from %s", model_path)
            except Exception as exc:
                logger.warning("Could not load LTR model: %s", exc)

    def is_trained(self) -> bool:
        return self._model is not None

    # ---- Feature extraction -----------------------------------------------

    def extract_features(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[List[float]]:
        """Extract feature vectors for a list of candidate results.

        Args:
            query: The original search query text.
            candidates: List of candidate result dicts with scores/metadata.

        Returns:
            List of feature vectors, one per candidate.
        """
        self._feature_names = [
            "distance_score",
            "bm25_score",
            "rrf_score",
            "rerank_score",
            "length_score",
            "freshness_score",
            "popularity_score",
            "query_length",
        ]
        n_features = len(self._feature_names)
        query_len = len(query.split())

        vectors = []
        for cand in candidates:
            vec = np.zeros(n_features, dtype=np.float32)

            vec[0] = float(cand.get("distance", 0.5))
            vec[1] = float(cand.get("bm25_score", 0.0))
            vec[2] = float(cand.get("rrf_score", 0.0))
            vec[3] = float(cand.get("rerank_score", 0.5))

            meta = cand.get("metadata", {}) or {}
            text = meta.get("text", "") or meta.get("content", "")
            vec[4] = min(len(text) / 1000.0, 1.0)  # length feature

            # Freshness: how recently was this document created?
            ts = meta.get("created_at") or meta.get("timestamp")
            if ts:
                from datetime import datetime
                try:
                    if isinstance(ts, str):
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    else:
                        dt = ts
                    age_days = (datetime.utcnow() - dt).days
                    vec[5] = max(0, 1.0 - age_days / 365.0)
                except (ValueError, TypeError):
                    vec[5] = 0.5
            else:
                vec[5] = 0.5

            vec[6] = float(meta.get("popularity", 0.5))
            vec[7] = min(query_len / 20.0, 1.0)

            vectors.append(vec.tolist())

        return vectors

    # ---- Training ---------------------------------------------------------

    def train_from_feedback(
        self,
        feedback_entries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Train the LTR model from collected feedback entries.

        Args:
            feedback_entries: List of feedback dicts with query, results, and
                user clicks/ratings.

        Returns:
            Training statistics.
        """
        examples: List[LTRTrainingExample] = []
        qid_counter = 0

        for entry in feedback_entries:
            query = entry.get("query", "")
            results = entry.get("results", [])
            clicks = entry.get("clicks", {})  # vector_id -> click count/rating

            if not query or not results:
                continue

            qid_counter += 1
            for res in results:
                vid = res.get("vector_id", "")
                features = self.extract_features(query, [res])[0]

                # Determine relevance from clicks
                rating = clicks.get(vid, 0)
                if isinstance(rating, bool):
                    relevance = 2.0 if rating else 0.0
                elif isinstance(rating, (int, float)):
                    relevance = min(float(rating), 3.0)
                else:
                    relevance = 0.0

                examples.append(LTRTrainingExample(
                    qid=qid_counter,
                    features=features,
                    relevance=relevance,
                    query_text=query,
                    document_id=vid,
                ))

        if len(examples) < 10:
            return {"success": False, "message": "Need at least 10 training examples"}

        return self._train_xgboost(examples)

    def _train_xgboost(
        self, examples: List[LTRTrainingExample]
    ) -> Dict[str, Any]:
        """Train XGBoost LambdaMART ranker."""
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning(
                "XGBoost not installed. Install with: pip install xgboost"
            )
            return {"success": False, "message": "xgboost not installed"}

        n_features = len(examples[0].features)
        X = np.zeros((len(examples), n_features), dtype=np.float32)
        y = np.zeros(len(examples), dtype=np.float32)
        qid_list = []
        groups = []

        for i, ex in enumerate(examples):
            X[i] = ex.features
            y[i] = float(ex.relevance)
            qid_list.append(ex.qid)

        # Determine group sizes for ranking task
        from collections import Counter
        group_counts = Counter(qid_list)
        groups = [group_counts[qid] for qid in sorted(group_counts.keys())]

        dtrain = xgb.DMatrix(X, label=y)
        dtrain.set_group(groups)

        params = {
            "objective": "rank:ndcg",
            "eval_metric": "ndcg@5",
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
        }

        self._model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            verbose_eval=False,
        )

        # Save model
        with open(self.model_path, "wb") as f:
            pickle.dump(self._model, f)

        logger.info("LTR model trained on %d examples", len(examples))

        return {
            "success": True,
            "num_examples": len(examples),
            "num_queries": len(set(qid_list)),
            "model_path": self.model_path,
            "features": self._feature_names,
        }

    # ---- Prediction -------------------------------------------------------

    def predict(
        self,
        feature_vectors: List[List[float]],
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Score candidates with the LTR model and rerank.

        Args:
            feature_vectors: Extracted features for each candidate.
            candidates: Original candidate result dicts.

        Returns:
            Re-ranked candidates with LTR scores.
        """
        if self._model is None:
            return candidates

        try:
            import xgboost as xgb
            dmatrix = xgb.DMatrix(np.array(feature_vectors, dtype=np.float32))
            scores = self._model.predict(dmatrix)
        except Exception as exc:
            logger.warning("LTR predict failed: %s", exc)
            return candidates

        for i, (cand, score) in enumerate(zip(candidates, scores)):
            cand["ltr_score"] = float(score)
            cand["original_rank"] = i

        candidates.sort(key=lambda x: x.get("ltr_score", 0), reverse=True)
        return candidates

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_trained": self.is_trained(),
            "model_path": self.model_path,
            "num_features": len(self._feature_names),
            "features": self._feature_names,
        }
