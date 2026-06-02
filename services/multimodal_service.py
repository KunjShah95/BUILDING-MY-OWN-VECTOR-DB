from typing import Any, Dict, List, Optional, Union

from sqlalchemy.orm import Session

from config.settings import get_settings
from services.collection_service import CollectionService
from services.collection_index_service import CollectionIndexService
from services.embedding_service import (
    embed_audio,
    embed_clip_text,
    embed_image,
    embed_text,
    default_model_for_modality,
    is_clip_model,
)
from services.media_store import read_media_bytes, save_media
from services.vector_service import VectorService
from utils.image_metadata import extract_image_metadata
from utils.metadata_contract import build_vector_metadata

_TEXT_SNIPPET_MAX = 500

PathOrBytes = Union[str, bytes]


class MultimodalService:
    """
    Text, image, and audio ingest/search scoped to collections.
    """

    def __init__(self, db_session: Session):
        self.db = db_session
        self.collection_service = CollectionService(db_session)
        self.vector_service = VectorService(db_session)
        self.collection_index_service = CollectionIndexService(db_session)
        self.settings = get_settings()

    def _modality_allowed(self, collection: Dict[str, Any], content_modality: str) -> Optional[Dict[str, Any]]:
        coll_mod = collection["modality"]
        model = collection.get("embedding_model") or ""
        allowed = {
            "text": {"text"},
            "image": {"image"},
            "audio": {"audio"},
            "multimodal": {"text", "image"},
        }
        if coll_mod == "image" and content_modality == "text" and is_clip_model(model):
            return None  # captions / text-to-image queries in CLIP space
        if content_modality not in allowed.get(coll_mod, set()):
            return {
                "success": False,
                "message": (
                    f"Collection modality '{coll_mod}' does not allow "
                    f"'{content_modality}' content"
                ),
            }
        return None

    def _embed_text_for_collection(
        self, text: str, collection: Dict[str, Any]
    ) -> List[float]:
        model_name = self._model_for_content(collection, "text")
        if collection["modality"] in ("image", "multimodal") and is_clip_model(model_name):
            return embed_clip_text(text, model_name=model_name)
        return embed_text(text, model_name=model_name)

    def _model_for_content(self, collection: Dict[str, Any], content_modality: str) -> str:
        if collection["modality"] == "multimodal" and content_modality in ("text", "image"):
            return collection["embedding_model"] or self.settings.DEFAULT_IMAGE_MODEL
        if content_modality == "audio":
            return collection["embedding_model"] or self.settings.DEFAULT_AUDIO_MODEL
        return collection["embedding_model"] or default_model_for_modality(collection["modality"])

    def ingest_text(
        self,
        collection_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        vector_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            coll = self.collection_service.get_collection(collection_id, tenant_id=tenant_id)
            if not coll.get("success"):
                return coll

            collection = coll["collection"]
            denied = self._modality_allowed(collection, "text")
            if denied:
                return denied

            embedding = self._embed_text_for_collection(text, collection)
            dim_check = self.collection_service.validate_vector_dimension(
                collection_id, embedding
            )
            if not dim_check.get("success"):
                return dim_check

            snippet = text[:_TEXT_SNIPPET_MAX]
            merged_meta = build_vector_metadata(
                collection,
                extra={
                    **(metadata or {}),
                    "modality": "text",
                    "text": snippet,
                    "content_type": "text",
                },
            )

            return self.vector_service.create_vector(
                vector_data=embedding,
                metadata=merged_meta,
                vector_id=vector_id,
                collection_id=collection["collection_id"],
            )
        except RuntimeError as exc:
            return {"success": False, "message": str(exc)}
        except Exception as exc:
            return {"success": False, "message": f"Error ingesting text: {exc}"}

    def ingest_image(
        self,
        collection_id: str,
        file: PathOrBytes,
        filename: str = "upload.jpg",
        metadata: Optional[Dict[str, Any]] = None,
        vector_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            coll = self.collection_service.get_collection(collection_id, tenant_id=tenant_id)
            if not coll.get("success"):
                return coll

            collection = coll["collection"]
            denied = self._modality_allowed(collection, "image")
            if denied:
                return denied

            raw = read_media_bytes(file)
            content_uri = save_media(collection["collection_id"], filename, raw)

            model_name = self._model_for_content(collection, "image")
            embedding = embed_image(raw, model_name=model_name)
            dim_check = self.collection_service.validate_vector_dimension(
                collection_id, embedding
            )
            if not dim_check.get("success"):
                return dim_check

            img_meta = extract_image_metadata(raw, filename)
            merged_meta = build_vector_metadata(
                collection,
                extra={
                    **(metadata or {}),
                    "modality": "image",
                    "content_type": "image",
                    "filename": filename,
                    **img_meta,
                },
                content_uri=content_uri,
            )

            return self.vector_service.create_vector(
                vector_data=embedding,
                metadata=merged_meta,
                vector_id=vector_id,
                collection_id=collection["collection_id"],
            )
        except RuntimeError as exc:
            return {"success": False, "message": str(exc)}
        except Exception as exc:
            return {"success": False, "message": f"Error ingesting image: {exc}"}

    def ingest_audio(
        self,
        collection_id: str,
        file: PathOrBytes,
        filename: str = "upload.wav",
        metadata: Optional[Dict[str, Any]] = None,
        vector_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        chunk_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        try:
            coll = self.collection_service.get_collection(collection_id, tenant_id=tenant_id)
            if not coll.get("success"):
                return coll

            collection = coll["collection"]
            denied = self._modality_allowed(collection, "audio")
            if denied:
                return denied

            raw = read_media_bytes(file)
            content_uri = save_media(collection["collection_id"], filename, raw)

            model_name = self._model_for_content(collection, "audio")
            embedding = embed_audio(raw, model_name=model_name, chunk_seconds=chunk_seconds)

            # Handle chunked audio (returns list of vectors)
            if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
                segment_vectors = embedding  # list of segment vectors
                created_ids = []
                for idx, seg_vec in enumerate(segment_vectors):
                    seg_meta = build_vector_metadata(
                        collection,
                        extra={
                            **(metadata or {}),
                            "modality": "audio",
                            "content_type": "audio",
                            "filename": filename,
                            "segment_index": idx,
                            "segment_count": len(segment_vectors),
                            "chunk_seconds": chunk_seconds,
                            "audio_duration_sec": len(segment_vectors) * (chunk_seconds or 5.0),
                        },
                        content_uri=content_uri,
                    )
                    seg_id = f"{vector_id}_seg{idx}" if vector_id else None
                    result = self.vector_service.create_vector(
                        vector_data=seg_vec,
                        metadata=seg_meta,
                        vector_id=seg_id,
                        collection_id=collection["collection_id"],
                    )
                    if result.get("success"):
                        created_ids.append(result.get("vector_id"))
                return {
                    "success": True,
                    "message": f"Ingested {len(created_ids)}/{len(segment_vectors)} audio segments",
                    "vector_ids": created_ids,
                    "segment_count": len(segment_vectors),
                }

            # Single vector mode
            dim_check = self.collection_service.validate_vector_dimension(
                collection_id, embedding
            )
            if not dim_check.get("success"):
                return dim_check

            merged_meta = build_vector_metadata(
                collection,
                extra={
                    **(metadata or {}),
                    "modality": "audio",
                    "content_type": "audio",
                    "filename": filename,
                },
                content_uri=content_uri,
            )

            return self.vector_service.create_vector(
                vector_data=embedding,
                metadata=merged_meta,
                vector_id=vector_id,
                collection_id=collection["collection_id"],
            )
        except RuntimeError as exc:
            return {"success": False, "message": str(exc)}
        except Exception as exc:
            return {"success": False, "message": f"Error ingesting audio: {exc}"}

    def search_text(
        self,
        collection_id: str,
        query: str,
        k: int = 5,
        method: str = "brute",
        ef_search: Optional[int] = None,
        n_probes: Optional[int] = None,
        use_rerank: Optional[bool] = True,
        filters: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            coll = self.collection_service.get_collection(collection_id, tenant_id=tenant_id)
            if not coll.get("success"):
                return coll

            collection = coll["collection"]
            denied = self._modality_allowed(collection, "text")
            if denied:
                return denied

            query_vector = self._embed_text_for_collection(query, collection)

            result = self.collection_index_service.search_collection_indexed(
                collection_id=collection["collection_id"],
                query_vector=query_vector,
                k=k,
                method=method,
                ef_search=ef_search,
                n_probes=n_probes,
                use_rerank=use_rerank,
                filters=filters,
                distance_metric=collection["distance_metric"],
                tenant_id=tenant_id,
            )

            if result.get("success"):
                result["query_text"] = query
                result["collection_id"] = collection["collection_id"]

            return result
        except RuntimeError as exc:
            return {"success": False, "message": str(exc)}
        except Exception as exc:
            return {"success": False, "message": f"Error searching text: {exc}"}

    def search_image(
        self,
        collection_id: str,
        file_or_path: PathOrBytes,
        k: int = 5,
        method: str = "brute",
        ef_search: Optional[int] = None,
        n_probes: Optional[int] = None,
        use_rerank: Optional[bool] = True,
        filters: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            coll = self.collection_service.get_collection(collection_id, tenant_id=tenant_id)
            if not coll.get("success"):
                return coll

            collection = coll["collection"]
            denied = self._modality_allowed(collection, "image")
            if denied:
                return denied

            model_name = self._model_for_content(collection, "image")
            query_vector = embed_image(file_or_path, model_name=model_name)

            result = self.collection_index_service.search_collection_indexed(
                collection_id=collection["collection_id"],
                query_vector=query_vector,
                k=k,
                method=method,
                ef_search=ef_search,
                n_probes=n_probes,
                use_rerank=use_rerank,
                filters=filters,
                distance_metric=collection["distance_metric"],
                tenant_id=tenant_id,
            )

            if result.get("success"):
                result["query_modality"] = "image"
                result["collection_id"] = collection["collection_id"]

            return result
        except RuntimeError as exc:
            return {"success": False, "message": str(exc)}
        except Exception as exc:
            return {"success": False, "message": f"Error searching image: {exc}"}

    def search_audio(
        self,
        collection_id: str,
        file_or_path: PathOrBytes,
        k: int = 5,
        method: str = "brute",
        ef_search: Optional[int] = None,
        n_probes: Optional[int] = None,
        use_rerank: Optional[bool] = True,
        filters: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            coll = self.collection_service.get_collection(collection_id, tenant_id=tenant_id)
            if not coll.get("success"):
                return coll

            collection = coll["collection"]
            denied = self._modality_allowed(collection, "audio")
            if denied:
                return denied

            model_name = self._model_for_content(collection, "audio")
            query_vector = embed_audio(file_or_path, model_name=model_name)

            result = self.collection_index_service.search_collection_indexed(
                collection_id=collection["collection_id"],
                query_vector=query_vector,
                k=k,
                method=method,
                ef_search=ef_search,
                n_probes=n_probes,
                use_rerank=use_rerank,
                filters=filters,
                distance_metric=collection["distance_metric"],
                tenant_id=tenant_id,
            )

            if result.get("success"):
                result["query_modality"] = "audio"
                result["collection_id"] = collection["collection_id"]

            return result
        except RuntimeError as exc:
            return {"success": False, "message": str(exc)}
        except Exception as exc:
            return {"success": False, "message": f"Error searching audio: {exc}"}
