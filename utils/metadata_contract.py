from typing import Any, Dict, Optional


def build_vector_metadata(
    collection: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
    *,
    content_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge collection policy fields into vector metadata."""
    base: Dict[str, Any] = dict(extra or {})
    base.update({
        "collection_id": collection["collection_id"],
        "modality": collection["modality"],
        "embedding_model": collection["embedding_model"],
        "dimension": collection["dimension"],
    })
    if content_uri is not None:
        base["content_uri"] = content_uri
    else:
        base.pop("content_uri", None)
    return base
