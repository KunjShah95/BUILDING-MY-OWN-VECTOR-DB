import io
import mimetypes
from typing import Any, Dict, Optional


def extract_image_metadata(raw: bytes, filename: Optional[str] = None) -> Dict[str, Any]:
    """Read width, height, format, and mime type from image bytes."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for image metadata") from exc

    with Image.open(io.BytesIO(raw)) as img:
        width, height = img.size
        fmt = (img.format or "").lower()
        mime = Image.MIME.get(img.format) if img.format else None

    if not mime and filename:
        mime, _ = mimetypes.guess_type(filename)

    return {
        "width": width,
        "height": height,
        "format": fmt or None,
        "mime_type": mime,
    }
