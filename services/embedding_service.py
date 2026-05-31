"""
Embedding backends for text, image, and audio.

Text: sentence-transformers (e.g. all-MiniLM-L6-v2, 384-dim).
Image: CLIP ViT-B/32 via sentence-transformers (512-dim), lazy-loaded.
Audio: librosa MFCC mean-pooled to 128-dim (CPU-friendly, no GPU model).
      Optional heavy path is wav2vec2; we avoid it by default for CI/CPU cost.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from config.settings import get_settings

logger = logging.getLogger(__name__)

PathOrBytes = Union[str, Path, bytes, io.BytesIO]

_text_model = None
_text_model_name: Optional[str] = None
_image_model = None
_image_model_name: Optional[str] = None

_EMBEDDING_IMPORT_ERROR = (
    "sentence-transformers is not installed. "
    "Install it with: pip install 'sentence-transformers>=2.2.0,<3.0.0'"
)
_IMAGE_IMPORT_ERROR = (
    "Image embedding requires sentence-transformers and Pillow. "
    "pip install 'sentence-transformers>=2.2.0,<3.0.0' pillow"
)
_AUDIO_IMPORT_ERROR = (
    "Audio embedding requires librosa and soundfile. "
    "pip install 'librosa>=0.10.0,<0.11.0' soundfile>=0.12.0"
)


def _load_text_model(model_name: Optional[str] = None):
    global _text_model, _text_model_name
    settings = get_settings()
    target = model_name or settings.DEFAULT_EMBEDDING_MODEL

    if _text_model is not None and _text_model_name == target:
        return _text_model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(_EMBEDDING_IMPORT_ERROR) from exc

    logger.info("Loading text embedding model %s on %s", target, settings.EMBEDDING_DEVICE)
    _text_model = SentenceTransformer(target, device=settings.EMBEDDING_DEVICE)
    _text_model_name = target
    return _text_model


def _load_image_model(model_name: Optional[str] = None):
    global _image_model, _image_model_name
    settings = get_settings()
    target = model_name or settings.DEFAULT_IMAGE_MODEL

    if _image_model is not None and _image_model_name == target:
        return _image_model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(_IMAGE_IMPORT_ERROR) from exc

    logger.info("Loading image embedding model %s on %s", target, settings.EMBEDDING_DEVICE)
    _image_model = SentenceTransformer(target, device=settings.EMBEDDING_DEVICE)
    _image_model_name = target
    return _image_model


def _read_bytes(source: PathOrBytes) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, io.BytesIO):
        return source.getvalue()
    path = Path(source)
    return path.read_bytes()


def _pil_image(source: PathOrBytes):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(_IMAGE_IMPORT_ERROR) from exc

    if isinstance(source, (str, Path)):
        return Image.open(Path(source)).convert("RGB")
    data = _read_bytes(source)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > 0:
        return vec / norm
    return vec


def embed_text(text: str, model_name: Optional[str] = None) -> List[float]:
    """Embed a single text string into a dense vector."""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    model = _load_text_model(model_name)
    vector = model.encode(text, convert_to_numpy=True)
    return vector.tolist()


def embed_texts(texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
    """Embed multiple text strings."""
    if not texts:
        raise ValueError("Texts list cannot be empty")
    cleaned = [t.strip() if t else "" for t in texts]
    if any(not t for t in cleaned):
        raise ValueError("All texts must be non-empty")
    model = _load_text_model(model_name)
    vectors = model.encode(cleaned, convert_to_numpy=True)
    return [row.tolist() for row in vectors]


def embed_image(path_or_bytes: PathOrBytes, model_name: Optional[str] = None) -> List[float]:
    """Embed an image file or bytes with CLIP (default clip-ViT-B-32, 512-dim)."""
    image = _pil_image(path_or_bytes)
    model = _load_image_model(model_name)
    vector = model.encode(image, convert_to_numpy=True)
    return vector.tolist()


def embed_audio(path_or_bytes: PathOrBytes, model_name: Optional[str] = None) -> List[float]:
    """
    Embed audio using librosa MFCC statistics (128-dim by default).

    Loads mono audio at AUDIO_SAMPLE_RATE, caps length at AUDIO_MAX_DURATION_SEC,
    computes n_mfcc=128 MFCCs, mean-pools over time, L2-normalizes.
    The model_name argument is accepted for collection metadata parity only.
    """
    del model_name  # librosa path is fixed; collection stores DEFAULT_AUDIO_MODEL id
    settings = get_settings()
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError(_AUDIO_IMPORT_ERROR) from exc

    if isinstance(path_or_bytes, (str, Path)):
        y, _sr = librosa.load(
            str(path_or_bytes),
            sr=settings.AUDIO_SAMPLE_RATE,
            mono=True,
            duration=settings.AUDIO_MAX_DURATION_SEC,
        )
    else:
        data = _read_bytes(path_or_bytes)
        y, _sr = librosa.load(
            io.BytesIO(data),
            sr=settings.AUDIO_SAMPLE_RATE,
            mono=True,
            duration=settings.AUDIO_MAX_DURATION_SEC,
        )

    if y.size == 0:
        raise ValueError("Audio could not be decoded or is empty")

    n_mfcc = settings.DEFAULT_AUDIO_DIMENSION
    mfcc = librosa.feature.mfcc(y=y, sr=settings.AUDIO_SAMPLE_RATE, n_mfcc=n_mfcc)
    vec = mfcc.mean(axis=1)
    vec = _l2_normalize(vec.astype(np.float64))
    return vec.tolist()


def expected_dimension(modality: str, model_name: Optional[str] = None) -> int:
    """Return default vector dimension for a modality (used at collection create)."""
    settings = get_settings()
    if modality == "image":
        return settings.DEFAULT_IMAGE_DIMENSION
    if modality == "audio":
        return settings.DEFAULT_AUDIO_DIMENSION
    if modality == "multimodal":
        return settings.DEFAULT_IMAGE_DIMENSION
    return settings.DEFAULT_TEXT_DIMENSION


def default_model_for_modality(modality: str) -> str:
    settings = get_settings()
    if modality == "image":
        return settings.DEFAULT_IMAGE_MODEL
    if modality == "audio":
        return settings.DEFAULT_AUDIO_MODEL
    if modality == "multimodal":
        return settings.DEFAULT_IMAGE_MODEL
    return settings.DEFAULT_EMBEDDING_MODEL


def is_clip_model(model_name: Optional[str] = None) -> bool:
    """True if the model id refers to a CLIP / shared text-image space."""
    settings = get_settings()
    name = (model_name or settings.DEFAULT_IMAGE_MODEL).lower()
    return "clip" in name or name.startswith("openai/")


def embed_clip_text(text: str, model_name: Optional[str] = None) -> List[float]:
    """Embed text in the same CLIP space as embed_image (text-to-image search)."""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    model = _load_image_model(model_name)
    vector = model.encode(text.strip(), convert_to_numpy=True)
    return vector.tolist()
