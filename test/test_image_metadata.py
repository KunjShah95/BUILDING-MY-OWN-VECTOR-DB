"""Unit tests for image metadata and CLIP routing helpers."""

import struct
import zlib

from services.embedding_service import is_clip_model
from utils.image_metadata import extract_image_metadata


def _minimal_png() -> bytes:
    """1x1 RGBA PNG without external fixtures."""
    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    raw = b"\x00" + b"\x00\x00\x00\x01" + b"\x00\x00\x00\x01" + b"\x00\x00\xff\xff\xff\xff"
    idat = zlib.compress(raw)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", idat)
        + chunk(b"IEND", b"")
    )


def test_extract_image_metadata_from_png():
    raw = _minimal_png()
    meta = extract_image_metadata(raw, "dot.png")
    assert meta["width"] == 1
    assert meta["height"] == 1
    assert meta["format"] == "png"
    assert meta["mime_type"] in ("image/png", "image/x-png")


def test_is_clip_model():
    assert is_clip_model("clip-ViT-B-32") is True
    assert is_clip_model("sentence-transformers/all-MiniLM-L6-v2") is False
    assert is_clip_model("openai/clip-vit-base-patch32") is True
