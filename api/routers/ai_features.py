"""AI Features router — Phase 15: Vector-Native AI.

Wires LTR trainer, RLHF feedback, vector explainer, and federated embedding
under /api/ltr, /api/rlhf, /api/explain, and /api/federated.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Path, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Singleton service instances ──────────────────────────────────────────────

from services.ltr_trainer import LTRTrainer, TrainingPair
from services.rlhf_service import RLHFService
from services.vector_explainer import VectorExplainer
from services.federated_embedding import FederatedEmbeddingService

_ltr_trainer = LTRTrainer()
_rlhf_service = RLHFService()
_explainer = VectorExplainer()
_federated = FederatedEmbeddingService()

router = APIRouter(tags=["AI Features"])


# ===========================================================================
# Pydantic models
# ===========================================================================

class CollectPairRequest(BaseModel):
    query_id: str
    doc_id: str
    label: float
    features: Dict[str, float]


class TrainRequest(BaseModel):
    model_type: str = "linear"   # 'linear' | 'mlp' | 'pairwise'


class SignalRequest(BaseModel):
    query_id: str
    result_id: str
    signal_type: str             # 'click' | 'skip' | 'dwell'
    dwell_ms: Optional[float] = None


class PredictRelevanceRequest(BaseModel):
    query_vector: List[float]
    doc_vector: List[float]


class ExplainRequest(BaseModel):
    query_vector: List[float]
    result_vector: List[float]
    metadata: Optional[Dict[str, Any]] = None
    top_k_dims: int = 10
    result_id: Optional[str] = None


class ExplainBatchRequest(BaseModel):
    query_vector: List[float]
    results: List[Dict[str, Any]]
    top_k_dims: int = 10


class RegisterClientRequest(BaseModel):
    client_id: str
    model_version: str
    local_stats: Dict[str, Any] = {}


class FederatedUpdateRequest(BaseModel):
    client_id: str
    local_embeddings: List[List[float]]
    labels: List[int]


# ===========================================================================
# LTR endpoints
# ===========================================================================

@router.post("/api/ltr/collect")
async def ltr_collect(req: CollectPairRequest):
    """Add a training pair to ltr_training_data.jsonl."""
    pair = _ltr_trainer.collect_training_pair(
        query_id=req.query_id,
        doc_id=req.doc_id,
        label=req.label,
        features=req.features,
    )
    return {"success": True, "pair": asdict(pair)}


@router.post("/api/ltr/train")
async def ltr_train(req: TrainRequest):
    """Trigger a training run, return model metrics."""
    try:
        if req.model_type == "pairwise":
            model = _ltr_trainer.train_pairwise()
        else:
            model = _ltr_trainer.train_pointwise(req.model_type)
        # Quick self-evaluation on training data
        pairs = _ltr_trainer.load_training_data()
        metrics = _ltr_trainer.evaluate(pairs) if pairs else {}
        return {
            "success": True,
            "model_type": model.model_type,
            "feature_names": model.feature_names,
            "metrics": metrics,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/ltr/model/stats")
async def ltr_model_stats():
    """Current model weights + training data size."""
    return {"success": True, **_ltr_trainer.stats()}


# ===========================================================================
# RLHF endpoints
# ===========================================================================

@router.post("/api/rlhf/signal")
async def rlhf_signal(req: SignalRequest):
    """Record a user feedback signal."""
    sig = _rlhf_service.record_signal(
        query_id=req.query_id,
        result_id=req.result_id,
        signal_type=req.signal_type,
        dwell_ms=req.dwell_ms,
    )
    return {"success": True, "signal": asdict(sig)}


@router.post("/api/rlhf/train")
async def rlhf_train():
    """Retrain the reward model on collected signals."""
    try:
        model = _rlhf_service.train_reward_model()
        return {
            "success": True,
            "accuracy": model.accuracy,
            "n_pairs": model.n_pairs,
            "weights": model.weights,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/rlhf/predict")
async def rlhf_predict(req: PredictRelevanceRequest):
    """Predict relevance for a (query_vector, doc_vector) pair."""
    score = _rlhf_service.predict_relevance(req.query_vector, req.doc_vector)
    return {"success": True, "relevance": score}


@router.get("/api/rlhf/stats")
async def rlhf_stats():
    """Signal counts and model accuracy."""
    return {"success": True, **_rlhf_service.stats()}


# ===========================================================================
# Explain endpoints
# ===========================================================================

@router.post("/api/explain/{collection_id}")
async def explain_result(
    collection_id: str = Path(...),
    body: ExplainRequest = Body(...),
):
    """Explain why a single result matched the query."""
    exp = _explainer.explain(
        query_vector=body.query_vector,
        result_vector=body.result_vector,
        metadata=body.metadata,
        top_k_dims=body.top_k_dims,
        result_id=body.result_id,
    )
    return {"success": True, "collection_id": collection_id, "explanation": asdict(exp)}


@router.post("/api/explain/batch")
async def explain_batch(body: ExplainBatchRequest):
    """Explain multiple results for a query vector."""
    explanations = _explainer.explain_batch(
        query_vector=body.query_vector,
        results=body.results,
        top_k_dims=body.top_k_dims,
    )
    return {
        "success": True,
        "explanations": [asdict(e) for e in explanations],
        "count": len(explanations),
    }


# ===========================================================================
# Federated embedding endpoints
# ===========================================================================

@router.post("/api/federated/register")
async def federated_register(req: RegisterClientRequest):
    """Register a federated client."""
    result = _federated.register_client(
        client_id=req.client_id,
        model_version=req.model_version,
        local_stats=req.local_stats,
    )
    return result


@router.post("/api/federated/update")
async def federated_update(req: FederatedUpdateRequest):
    """Submit a local embedding update; receive global delta."""
    result = _federated.compute_gradient_update(
        client_id=req.client_id,
        local_embeddings=req.local_embeddings,
        labels=req.labels,
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message", "error"))
    return result


@router.post("/api/federated/aggregate")
async def federated_aggregate():
    """Aggregate all pending client updates (FedAvg)."""
    result = _federated.aggregate_updates()
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message", "error"))
    return result


@router.get("/api/federated/status")
async def federated_status():
    """Connected clients, model version, rounds completed."""
    return {"success": True, **_federated.status()}
