import uuid
from pathlib import Path
from typing import Union

from config.settings import get_settings


def _storage_root() -> Path:
    settings = get_settings()
    root = Path(settings.MEDIA_STORAGE_PATH)
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_media(
    collection_id: str,
    filename: str,
    data: bytes,
) -> str:
    """
    Persist uploaded bytes under MEDIA_STORAGE_PATH/{collection_id}/.

    Returns a relative content_uri (posix-style) stored in vector metadata.
    """
    if not data:
        raise ValueError("Cannot save empty media payload")

    safe_name = Path(filename or "upload").name
    suffix = Path(safe_name).suffix or ".bin"
    unique_name = f"{uuid.uuid4().hex}{suffix}"

    dest_dir = _storage_root() / collection_id.strip().lower()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / unique_name
    dest_path.write_bytes(data)

    settings = get_settings()
    root_name = Path(settings.MEDIA_STORAGE_PATH).name
    return f"{root_name}/{collection_id.strip().lower()}/{unique_name}"


def resolve_media_path(content_uri: str) -> Path:
    """Resolve a stored content_uri to an absolute filesystem path."""
    settings = get_settings()
    root = Path(settings.MEDIA_STORAGE_PATH).resolve()
    if not content_uri:
        raise ValueError("content_uri is required")

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
    if isinstance(source, bytes):
        return source
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Media file not found: {path}")
    return path.read_bytes()
