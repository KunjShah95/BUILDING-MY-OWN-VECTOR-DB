import uuid
from pathlib import Path
from typing import Union, Optional

from config.settings import get_settings
from services.storage_backend import get_storage_backend


def save_media(
    collection_id: str,
    filename: str,
    data: bytes,
    storage_provider: Optional[str] = None,
) -> str:
    """
    Persist uploaded bytes using the configured storage backend.

    Supports local filesystem, S3, and Azure Blob Storage via StorageBackend.
    Returns a content_uri (posix-style) stored in vector metadata.
    """
    if not data:
        raise ValueError("Cannot save empty media payload")

    safe_name = Path(filename or "upload").name
    suffix = Path(safe_name).suffix or ".bin"
    unique_name = f"{uuid.uuid4().hex}{suffix}"

    # Build storage path: {MEDIA_STORAGE_PATH}/{collection_id}/{unique_name}
    settings = get_settings()
    prefix = settings.MEDIA_STORAGE_PATH.strip("/")
    collection_slug = collection_id.strip().lower()
    storage_path = f"{prefix}/{collection_slug}/{unique_name}"

    # Use the configured storage backend
    if storage_provider:
        from services.storage_backend import StorageFactory
        backend = StorageFactory.create(storage_provider)
    else:
        backend = get_storage_backend()

    backend.save(storage_path, data)

    # Return posix-style relative URI
    return storage_path.replace("\\", "/")


def resolve_media_path(content_uri: str) -> Path:
    """Resolve a stored content_uri to a filesystem path (local backend only).

    For S3/Azure backends, use ``read_media_bytes()`` instead which works with
    all backends.
    """
    if not content_uri:
        raise ValueError("content_uri is required")

    settings = get_settings()
    root = Path(settings.MEDIA_STORAGE_PATH).resolve()

    uri_path = Path(content_uri.replace("\\", "/"))
    if uri_path.is_absolute():
        resolved_path = uri_path.resolve()
    else:
        if uri_path.parts and uri_path.parts[0] == root.name:
            uri_path = Path(*uri_path.parts[1:])
        resolved_path = (root / uri_path).resolve()

    if not resolved_path.is_relative_to(root):
        raise ValueError("Access denied: Path traversal detected")
    return resolved_path


def read_media_bytes(source: Union[str, Path, bytes]) -> bytes:
    """Read media bytes from any storage backend or raw bytes input."""
    if isinstance(source, bytes):
        return source

    # Try storage backend first (for S3/Azure)
    try:
        backend = get_storage_backend()
        path_str = str(source) if not isinstance(source, Path) else str(source)
        result = backend.load(path_str)
        if result is not None:
            return result
    except Exception:
        pass

    # Fall back to local filesystem
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Media file not found: {path}")
    return path.read_bytes()
