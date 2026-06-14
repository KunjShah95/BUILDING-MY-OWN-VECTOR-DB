"""Ecosystem integration endpoints."""
import logging
from typing import Optional, List
from fastapi import APIRouter
from pydantic import BaseModel

from services.metadata_enrichment import metadata_enricher
from services.embedding_model_lifecycle import model_registry

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Integrations"])

# --- Metadata Enrichment ---

class EnrichmentRequest(BaseModel):
    text: str
    existing_metadata: Optional[dict] = None

@router.post("/enrich/metadata")
def enrich_metadata(body: EnrichmentRequest):
    result = metadata_enricher.enrich(body.text, body.existing_metadata)
    return {"success": True, "enriched_metadata": result}

# --- Embedding Model Lifecycle ---

class ModelRegisterRequest(BaseModel):
    model_id: str
    provider: str
    dimension: int
    version: str = "1"
    metadata: Optional[dict] = None

class ABTestRequest(BaseModel):
    model_a: str
    model_b: str
    traffic_split: float = 0.5

@router.get("/admin/models")
def list_models():
    return {"success": True, "models": model_registry.list_models()}

@router.post("/admin/models")
def register_model(body: ModelRegisterRequest):
    model = model_registry.register_model(**body.model_dump())
    return {"success": True, "model": model}

@router.post("/admin/models/{model_id}/activate")
def activate_model(model_id: str):
    model_registry.activate_model(model_id)
    return {"success": True, "message": f"{model_id} activated"}

@router.post("/admin/models/{model_id}/deactivate")
def deactivate_model(model_id: str):
    model_registry.deactivate_model(model_id)
    return {"success": True, "message": f"{model_id} deactivated"}

@router.post("/admin/models/ab-test")
def set_ab_test(body: ABTestRequest):
    model_registry.set_ab_test(**body.model_dump())
    return {"success": True, "message": f"A/B test configured: {body.model_a} vs {body.model_b}"}
