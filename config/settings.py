import os
from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    """

    # database settings
    DATABASE_URL: str = "postgresql://vector_user:vector_password@localhost:5432/vector_db"

    # Application settings
    APP_NAME: str = "Vector Database API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Index settings - Optimized for 10K vector dataset (best configuration)
    DEFAULT_M: int = 32
    DEFAULT_M0: int = 64
    DEFAULT_EF_CONSTRUCTION: int = 300
    DEFAULT_EF_SEARCH: int = 50
    DEFAULT_DISTANCE_METRIC: str = "cosine"
    DEFAULT_N_CLUSTERS: int = 100
    DEFAULT_N_PROBES: int = 10

    # pgvector toggle
    USE_PGVECTOR: bool = False

    # Text embedding
    DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_TEXT_DIMENSION: int = 384
    EMBEDDING_DEVICE: str = "cpu"

    # Image embedding (CLIP ViT-B/32, 512-dim, lazy-loaded via sentence-transformers)
    DEFAULT_IMAGE_MODEL: str = "clip-ViT-B-32"
    DEFAULT_IMAGE_DIMENSION: int = 512

    # Audio embedding (CPU-friendly librosa MFCC mean-pool; see embedding_service docstring)
    DEFAULT_AUDIO_MODEL: str = "librosa-mfcc-128"
    DEFAULT_AUDIO_DIMENSION: int = 128
    AUDIO_SAMPLE_RATE: int = 22050
    AUDIO_MAX_DURATION_SEC: float = 30.0

    MEDIA_STORAGE_PATH: str = "media_storage"
    STORAGE_PROVIDER: str = "local"

    # Security settings
    API_KEY: str = "your-api-key-here"
    ALLOWED_HOSTS: List[str] = ["*"]
    CORS_ORIGINS: str = "*"

    # Rate limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_TIME: int = 60
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_BACKEND: str = "memory"

    # Request limits
    MAX_REQUEST_SIZE_MB: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get application settings with caching."""
    return Settings()

