import re
import uuid
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from config.settings import get_settings
from database.schema import Collection, Vector
from services.embedding_service import default_model_for_modality, expected_dimension
from utils.index_paths import get_index_dir

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class CollectionService:
    """CRUD and validation for multimodal collections.

    Supports multi-tenancy: when ``tenant_id`` is provided collections are
    scoped to that tenant.
    """

    def __init__(self, db_session: Session):
        self.db = db_session
        self.settings = get_settings()

    def _normalize_slug(self, slug: str) -> str:
        return slug.strip().lower()

    def _slug_from_name(self, name: str) -> str:
        base = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
        if not base:
            base = "collection"
        return f"{base}-{uuid.uuid4().hex[:8]}"

    def _tenant_filter(self, query, tenant_id: Optional[str] = None):
        """Apply tenant scoping to a query if a tenant_id is given."""
        if tenant_id:
            return query.filter(Collection.tenant_id == tenant_id)
        return query

    def create_collection(
        self,
        name: str,
        collection_id: Optional[str] = None,
        description: Optional[str] = None,
        modality: str = "text",
        embedding_model: Optional[str] = None,
        dimension: Optional[int] = None,
        distance_metric: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            slug = self._normalize_slug(collection_id) if collection_id else self._slug_from_name(name)
            if not _SLUG_RE.match(slug):
                return {
                    "success": False,
                    "message": "collection_id must be lowercase alphanumeric with optional hyphens",
                }

            existing = self.get_collection(slug)
            if existing.get("success"):
                return {"success": False, "message": f"Collection '{slug}' already exists"}

            model_name = embedding_model or default_model_for_modality(modality)
            dim = dimension if dimension is not None else expected_dimension(modality, model_name)
            metric = distance_metric or self.settings.DEFAULT_DISTANCE_METRIC

            if modality == "multimodal" and dim != self.settings.DEFAULT_IMAGE_DIMENSION:
                return {
                    "success": False,
                    "message": (
                        f"Multimodal collections should use dimension "
                        f"{self.settings.DEFAULT_IMAGE_DIMENSION} (shared CLIP space for text/image)"
                    ),
                }

            record = Collection(
                collection_id=slug,
                tenant_id=tenant_id,
                name=name.strip(),
                description=description,
                modality=modality,
                embedding_model=model_name,
                dimension=dim,
                distance_metric=metric,
            )
            self.db.add(record)
            self.db.commit()
            self.db.refresh(record)

            return {
                "success": True,
                "message": "Collection created successfully",
                "collection": record.to_dict(),
            }
        except Exception as exc:
            self.db.rollback()
            return {"success": False, "message": f"Error creating collection: {exc}"}

    def get_collection(
        self, collection_id: str, tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            slug = self._normalize_slug(collection_id)
            query = self.db.query(Collection).filter(Collection.collection_id == slug)
            query = self._tenant_filter(query, tenant_id)
            record = query.first()
            if not record:
                return {"success": False, "message": "Collection not found"}
            return {"success": True, "collection": record.to_dict()}
        except Exception as exc:
            return {"success": False, "message": f"Error getting collection: {exc}"}

    def list_collections(
        self, limit: int = 100, offset: int = 0, tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            query = self.db.query(Collection).order_by(Collection.created_at.desc())
            query = self._tenant_filter(query, tenant_id)
            rows = query.offset(offset).limit(limit).all()
            return {
                "success": True,
                "collections": [row.to_dict() for row in rows],
                "count": len(rows),
                "limit": limit,
                "offset": offset,
            }
        except Exception as exc:
            return {"success": False, "message": f"Error listing collections: {exc}"}

    def delete_collection(
        self, collection_id: str, tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            slug = self._normalize_slug(collection_id)
            query = self.db.query(Collection).filter(Collection.collection_id == slug)
            query = self._tenant_filter(query, tenant_id)
            record = query.first()
            if not record:
                return {"success": False, "message": "Collection not found"}

            deleted_vectors = (
                self.db.query(Vector).filter(Vector.collection_id == slug).delete()
            )
            self.db.delete(record)

            index_dir = Path(get_index_dir(slug))
            if index_dir.exists() and index_dir.is_dir():
                shutil.rmtree(index_dir)

            self.db.commit()

            return {
                "success": True,
                "message": f"Collection '{slug}' deleted",
                "deleted_vectors": deleted_vectors,
            }
        except Exception as exc:
            self.db.rollback()
            return {"success": False, "message": f"Error deleting collection: {exc}"}

    def validate_vector_dimension(
        self, collection_id: str, vector_data: List[float],
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = self.get_collection(collection_id, tenant_id=tenant_id)
        if not result.get("success"):
            return result

        expected = result["collection"]["dimension"]
        actual = len(vector_data)
        if actual != expected:
            return {
                "success": False,
                "message": (
                    f"Vector dimension {actual} does not match collection "
                    f"dimension {expected}"
                ),
            }
        return {"success": True, "collection": result["collection"]}
