from pydantic import BaseModel, Field, validator, model_validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class SearchMethod(str, Enum):
    """Search method enumeration"""
    HNSW = "hnsw"
    IVF = "ivf"
    BRUTE = "brute"

class DistanceMetric(str, Enum):
    """Distance metric enumeration"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"

class VectorCreate(BaseModel):
    """Model for creating a vector"""
    vector: List[float] = Field(..., description="Vector data as list of floats", example=[0.1, 0.2, 0.3])
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata", example={"text": "sample"})
    vector_id: Optional[str] = Field(None, description="Optional custom vector ID")

    @validator('vector')
    def validate_vector(cls, v):
        if not v:
            raise ValueError('Vector cannot be empty')
        if len(v) == 0:
            raise ValueError('Vector must have at least one dimension')
        return v

class VectorUpdate(BaseModel):
    """Model for updating a vector"""
    vector: Optional[List[float]] = Field(None, description="Updated vector data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")

class VectorResponse(BaseModel):
    """Model for vector response"""
    success: bool
    vector_id: str
    vector: Dict[str, Any]
    message: Optional[str] = None

class VectorInDB(VectorCreate):
    """Model for vector in database"""
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class BatchInsert(BaseModel):
    """Model for batch insert"""
    vectors: List[Dict[str, Any]] = Field(..., description="List of vectors with 'vector', 'vector_id', and 'metadata'")
    batch_name: Optional[str] = Field(None, description="Optional batch name")
    description: Optional[str] = Field(None, description="Optional batch description")

    @validator('vectors')
    def validate_vectors(cls, v):
        if not v:
            raise ValueError('Vectors list cannot be empty')
        for i, item in enumerate(v):
            if 'vector' not in item:
                raise ValueError(f'Vector at index {i} missing "vector" field')
        return v

class SearchRequest(BaseModel):
    """Model for search request"""
    query_vector: List[float] = Field(..., description="Query vector", example=[0.1, 0.2, 0.3])
    k: int = Field(5, description="Number of results", ge=1, le=100)
    method: SearchMethod = Field(SearchMethod.HNSW, description="Search method")
    ef_search: Optional[int] = Field(None, description="HNSW search parameter", ge=1)
    n_probes: Optional[int] = Field(None, description="IVF probes to search", ge=1)
    use_rerank: Optional[bool] = Field(True, description="IVF: rerank results for accuracy")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters")

class SearchResult(BaseModel):
    """Model for search result"""
    vector_id: str
    distance: float
    metadata: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    """Model for search response"""
    success: bool
    results: List[SearchResult]
    total_results: int
    search_time: float
    method: str
    query_vector: List[float]

class IndexCreate(BaseModel):
    """Model for creating an index"""
    method: SearchMethod = Field(SearchMethod.HNSW, description="Indexing method")
    collection_id: Optional[str] = Field(
        None, description="Scope index to a collection (recommended for multimodal)"
    )
    m: int = Field(16, description="HNSW: Number of neighbors", ge=1, le=256)
    m0: Optional[int] = Field(None, description="HNSW: Neighbors in layer 0", ge=1)
    ef_construction: int = Field(200, description="HNSW: Construction parameter", ge=1)
    n_clusters: int = Field(100, description="IVF: Number of clusters", ge=1, le=10000)
    n_probes: int = Field(10, description="IVF: Number of probes", ge=1)

class IndexStats(BaseModel):
    """Model for index statistics"""
    total_nodes: int
    total_edges: int
    avg_connections: float
    max_level: int
    level_distribution: Dict[str, int]

class IndexResponse(BaseModel):
    """Model for index response"""
    success: bool
    message: str
    stats: Optional[Dict[str, Any]] = None

class BatchResponse(BaseModel):
    """Model for batch response"""
    success: bool
    message: str
    batch_id: Optional[int] = None
    vector_count: Optional[int] = None

class StatsResponse(BaseModel):
    """Model for statistics response"""
    success: bool
    stats: Dict[str, Any]

class ErrorResponse(BaseModel):
    """Model for error response"""
    success: bool = False
    error: str
    detail: Optional[str] = None
    status_code: int

class HealthResponse(BaseModel):
    """Model for health check response"""
    status: str
    database: str
    index_available: bool
    total_vectors: int


class ApiTemplateCreate(BaseModel):
    """Model for API playground templates"""
    name: str = Field(..., description="Template name")
    description: Optional[str] = Field(None, description="Optional description")
    method: str = Field(..., description="HTTP method", example="POST")
    path: str = Field(..., description="API path", example="/search")
    payload: Dict[str, Any] = Field(..., description="JSON payload")


class ApiTemplateResponse(BaseModel):
    """Template response model"""
    success: bool
    template: Dict[str, Any]


class ApiTemplateListResponse(BaseModel):
    """Template list response model"""
    success: bool
    templates: List[Dict[str, Any]]


class FeedbackCreate(BaseModel):
    """Model for feedback submission"""
    name: Optional[str] = Field(None, description="User name")
    email: Optional[str] = Field(None, description="User email")
    rating: Optional[int] = Field(None, description="Rating 1-5", ge=1, le=5)
    message: str = Field(..., description="Feedback message")


class FeedbackResponse(BaseModel):
    """Feedback response model"""
    success: bool
    feedback: Dict[str, Any]


class Modality(str, Enum):
    """Supported collection modalities."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


class CollectionCreate(BaseModel):
    """Model for creating a collection."""
    name: str = Field(..., description="Human-readable collection name")
    collection_id: Optional[str] = Field(
        None, description="Unique slug (auto-generated from name if omitted)"
    )
    description: Optional[str] = Field(None, description="Optional description")
    modality: Modality = Field(Modality.TEXT, description="Primary modality")
    embedding_model: Optional[str] = Field(
        None, description="Embedding model id (defaults from settings)"
    )
    dimension: Optional[int] = Field(
        None, description="Vector dimension (auto from modality if omitted)", ge=1
    )
    distance_metric: DistanceMetric = Field(
        DistanceMetric.COSINE, description="Distance metric for search"
    )

    @model_validator(mode="after")
    def apply_modality_defaults(self):
        from services.embedding_service import expected_dimension, default_model_for_modality

        modality_val = self.modality.value if self.modality else "text"
        if self.dimension is None:
            object.__setattr__(
                self,
                "dimension",
                expected_dimension(modality_val, self.embedding_model),
            )
        if self.embedding_model is None:
            object.__setattr__(
                self,
                "embedding_model",
                default_model_for_modality(modality_val),
            )
        return self


class CollectionResponse(BaseModel):
    """Collection API response."""
    success: bool
    collection: Optional[Dict[str, Any]] = None
    collections: Optional[List[Dict[str, Any]]] = None
    message: Optional[str] = None
    count: Optional[int] = None


class TextIngestRequest(BaseModel):
    """Ingest raw text into a collection (auto-embedded)."""
    text: str = Field(..., description="Text to embed and store", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Extra metadata")
    vector_id: Optional[str] = Field(None, description="Optional custom vector id")


class TextSearchRequest(BaseModel):
    """Natural-language search within a collection."""
    query: str = Field(..., description="Search query text", min_length=1)
    k: int = Field(5, description="Number of results", ge=1, le=100)
    method: SearchMethod = Field(
        SearchMethod.BRUTE,
        description="Search method (brute recommended for collection scope)",
    )
    ef_search: Optional[int] = Field(None, description="HNSW search parameter", ge=1)
    n_probes: Optional[int] = Field(None, description="IVF probes", ge=1)
    use_rerank: Optional[bool] = Field(True, description="IVF rerank flag")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")


class MediaSearchParams(BaseModel):
    """Shared search parameters for multipart image/audio query uploads."""
    k: int = Field(5, ge=1, le=100)
    method: SearchMethod = Field(SearchMethod.BRUTE)
    ef_search: Optional[int] = Field(None, ge=1)
    n_probes: Optional[int] = Field(None, ge=1)
    use_rerank: Optional[bool] = True
    filters: Optional[Dict[str, Any]] = None


class MediaIngestResponse(BaseModel):
    """Response shape for media ingest endpoints."""
    success: bool
    vector_id: Optional[str] = None
    message: Optional[str] = None
    vector: Optional[Dict[str, Any]] = None
