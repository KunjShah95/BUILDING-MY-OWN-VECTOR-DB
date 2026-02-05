import os 
from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    """

    #database settings
    DATABASE_URL:str="postgresql://vector_user:vector_password@localhost:5432/vector_db"

    #Application settings

    APP_NAME: str = "Vector Database API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Index settings - Optimized for 10K vector dataset (best configuration)
    DEFAULT_M: int = 32  # Higher connectivity gives better search efficiency
    DEFAULT_M0: int = 64  # (2*m)
    DEFAULT_EF_CONSTRUCTION: int = 300  # Quality vs speed balance
    DEFAULT_EF_SEARCH: int = 50  # Search parameter for reasonable latency
    DEFAULT_DISTANCE_METRIC: str = "cosine"
    DEFAULT_N_CLUSTERS: int = 100
    DEFAULT_N_PROBES: int = 10
    
    # Security settings
    API_KEY: str = "your-api-key-here"
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Rate limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_TIME: int = 60  # seconds
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings()-> Settings:
    """
    Get application settings with caching
    """
    return Settings()