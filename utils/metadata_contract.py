from typing import Any, Dict, Optional


def build_vector_metadata(
    collection: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
    *,
    content_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge collection policy fields into vector metadata."""
    base: Dict[str, Any] = {
        "collection_id": collection["collection_id"],
        "modality": collection["modality"],
        "embedding_model": collection["embedding_model"],
        "dimension": collection["dimension"],
    }
    if content_uri:
        base["content_uri"] = content_uri
    if extra:
        base.update(extra)
    return base
