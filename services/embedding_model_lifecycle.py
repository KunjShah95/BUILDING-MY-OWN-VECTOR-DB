"""Embedding model lifecycle — registry, versioning, A/B testing."""
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class EmbeddingModelRegistry:
    """Registry for embedding models with versioning and A/B testing."""

    def __init__(self):
        self._models: Dict[str, Dict[str, Any]] = {}
        self._register_defaults()

    def _register_defaults(self):
        self._models["text-embedding-ada-002"] = {
            "model_id": "text-embedding-ada-002",
            "provider": "openai",
            "dimension": 1536,
            "version": "2",
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._models["BAAI/bge-small-en-v1.5"] = {
            "model_id": "BAAI/bge-small-en-v1.5",
            "provider": "sentence-transformers",
            "dimension": 384,
            "version": "1.5",
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def register_model(self, model_id: str, provider: str, dimension: int,
                       version: str = "1", metadata: Optional[Dict] = None) -> Dict[str, Any]:
        model = {
            "model_id": model_id,
            "provider": provider,
            "dimension": dimension,
            "version": version,
            "status": "inactive",
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._models[model_id] = model
        return model

    def activate_model(self, model_id: str):
        if model_id in self._models:
            self._models[model_id]["status"] = "active"

    def deactivate_model(self, model_id: str):
        if model_id in self._models:
            self._models[model_id]["status"] = "inactive"

    def list_models(self) -> List[Dict[str, Any]]:
        return list(self._models.values())

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self._models.get(model_id)

    def get_active_model(self) -> Optional[Dict[str, Any]]:
        for m in self._models.values():
            if m["status"] == "active":
                return m
        return None

    def set_ab_test(self, model_a: str, model_b: str, traffic_split: float = 0.5):
        """Configure A/B test between two models."""
        self._models[model_a]["ab_test"] = {"group": "A", "traffic": traffic_split}
        self._models[model_b]["ab_test"] = {"group": "B", "traffic": 1 - traffic_split}
        logger.info("A/B test configured: %s (%.0f%%) vs %s (%.0f%%)",
                    model_a, traffic_split * 100, model_b, (1 - traffic_split) * 100)

    def select_model_for_query(self) -> str:
        """Select which model to use based on A/B test configuration."""
        candidates = [m for m in self._models.values() if m.get("ab_test")]
        if not candidates:
            active = self.get_active_model()
            return active["model_id"] if active else "text-embedding-ada-002"
        import random
        r = random.random()
        for m in candidates:
            ab = m["ab_test"]
            if r < ab["traffic"]:
                return m["model_id"]
        return candidates[-1]["model_id"]

model_registry = EmbeddingModelRegistry()
