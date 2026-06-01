from fastapi import FastAPI, Depends, HTTPException, Query, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import json
import logging
import time

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


def _rate_limit_key(request) -> str:
    """Use tenant identity for rate limiting when available, fall back to IP."""
    tenant_id = getattr(request.state, "tenant_id", None)
    if tenant_id:
        return f"tenant:{tenant_id}"
    return get_remote_address(request)

# Import configurations
from config.database import Base, engine, get_db
from config.settings import get_settings
from models.pydantic_models import (
    VectorCreate, VectorUpdate, VectorResponse,
    BatchInsert, BatchResponse, SearchRequest, SearchResponse,
    IndexCreate, IndexResponse, StatsResponse, HealthResponse,
    ErrorResponse, SearchMethod,
    ApiTemplateCreate, ApiTemplateResponse, ApiTemplateListResponse,
    FeedbackCreate, FeedbackResponse,
    CollectionCreate, CollectionResponse,
    TextIngestRequest, TextSearchRequest,
)
from database.schema import ApiTemplate, FeedbackEntry
from services.vector_service import VectorService
from services.collection_service import CollectionService
from services.collection_index_service import CollectionIndexService
from services.multimodal_service import MultimodalService
from services.media_store import resolve_media_path

# Import VectorIndexer API
try:
    from examples.vector_indexer_api import router as indexer_router
    INDEXER_AVAILABLE = True
except ImportError:
    INDEXER_AVAILABLE = False

# Prometheus metrics
from prometheus_client import make_asgi_app, Counter, Histogram

VECTOR_SEARCH_REQUESTS = Counter("vector_search_requests_total", "Total vector search requests")
VECTOR_INGEST_REQUESTS = Counter("vector_ingest_requests_total", "Total vector ingest requests")
QUERY_LATENCY = Histogram("vector_query_latency_seconds", "Query latency in seconds")

# Initialize FastAPI app
settings = get_settings()
app = FastAPI(
    title=settings.APP_NAME,
    description="A production-ready Vector Database API with HNSW, IVF, and Hybrid indexing support. Includes unified VectorIndexer with batch processing and auto-optimization.",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Vectors", "description": "Vector CRUD operations"},
        {"name": "Collections", "description": "Multimodal collection namespaces"},
        {"name": "Ingest", "description": "Text, image, and audio ingest with auto-embedding"},
        {"name": "Media", "description": "Stored media file access"},
        {"name": "Search", "description": "Vector similarity search"},
        {"name": "Index", "description": "Index management"},
        {"name": "Stats", "description": "Statistics and monitoring"},
        {"name": "Health", "description": "Health checks"},
        {"name": "Vector Indexer", "description": "Unified HNSW/IVF Vector Indexer API"},
        {"name": "Playground", "description": "Frontend playground support"},
    ]
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Auth middleware
from api.middleware.auth_middleware import auth_middleware
app.middleware("http")(auth_middleware)

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Dependency
def get_vector_service(db: Session = Depends(get_db)) -> VectorService:
    """
    Dependency to get vector service
    """
    return VectorService(db)


def get_collection_service(db: Session = Depends(get_db)) -> CollectionService:
    return CollectionService(db)


def get_multimodal_service(db: Session = Depends(get_db)) -> MultimodalService:
    return MultimodalService(db)


def get_collection_index_service(db: Session = Depends(get_db)) -> CollectionIndexService:
    return CollectionIndexService(db)


def _parse_metadata_form(metadata: Optional[str]) -> Optional[Dict[str, Any]]:
    if not metadata:
        return None
    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail={"success": False, "message": f"Invalid metadata JSON: {exc}"},
        ) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=400,
            detail={"success": False, "message": "metadata must be a JSON object"},
        )
    return parsed


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(
            status_code=400,
            detail={"success": False, "message": "Uploaded file is empty"},
        )
    return data


def _http_error_from_result(result: Dict[str, Any], not_found_codes: bool = True) -> int:
    if not result.get("success"):
        msg = result.get("message", "").lower()
        if not_found_codes and "not found" in msg:
            return 404
        return 400
    return 200

# ==================== Rate Limiting ====================

if settings.RATE_LIMIT_ENABLED:
    limiter = Limiter(
        key_func=_rate_limit_key,
        default_limits=[f"{settings.RATE_LIMIT_REQUESTS}/{settings.RATE_LIMIT_TIME}seconds"]
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ==================== Error Handlers ====================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler
    """
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else None
        }
    )

# ==================== Middleware ====================

@app.middleware("http")
async def add_process_time_header(request, call_next):
    """
    Add processing time header
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# ==================== Health Check ====================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint
    """
    service = VectorService(db)
    health = service.get_health_status()
    
    status_code = 200 if health["status"] == "healthy" else 503
    
    return JSONResponse(
        status_code=status_code,
        content=health
    )

@app.get("/ready", tags=["Health"])
async def readiness_check():
    """
    Readiness check endpoint
    """
    return {"status": "ready"}

# ==================== Collection Operations ====================

@app.post(
    "/collections",
    response_model=Dict[str, Any],
    tags=["Collections"],
    status_code=status.HTTP_201_CREATED,
)
async def create_collection(
    body: CollectionCreate,
    service: CollectionService = Depends(get_collection_service),
):
    """
    Create a collection namespace with fixed embedding model and dimension.

    Vectors ingested into this collection must match the declared dimension.
    """
    result = service.create_collection(
        name=body.name,
        collection_id=body.collection_id,
        description=body.description,
        modality=body.modality.value,
        embedding_model=body.embedding_model,
        dimension=body.dimension,
        distance_metric=body.distance_metric.value,
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result)
    return result


@app.get("/collections", tags=["Collections"])
async def list_collections(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    service: CollectionService = Depends(get_collection_service),
):
    result = service.list_collections(limit=limit, offset=offset)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result)
    return result


@app.get("/collections/{collection_id}", tags=["Collections"])
async def get_collection(
    collection_id: str,
    service: CollectionService = Depends(get_collection_service),
):
    result = service.get_collection(collection_id)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result)
    return result


@app.delete("/collections/{collection_id}", tags=["Collections"])
async def delete_collection(
    collection_id: str,
    service: CollectionService = Depends(get_collection_service),
):
    result = service.delete_collection(collection_id)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result)
    return result


@app.post("/collections/{collection_id}/index", tags=["Index"])
async def build_collection_index(
    collection_id: str,
    index_data: IndexCreate,
    service: CollectionIndexService = Depends(get_collection_index_service),
):
    """
    Build and persist an HNSW index for vectors in this collection only.
    """
    result = service.build_collection_index(
        collection_id=collection_id,
        method=index_data.method.value if index_data.method else "hnsw",
        m=index_data.m,
        m0=index_data.m0,
        ef_construction=index_data.ef_construction,
        n_clusters=index_data.n_clusters,
        n_probes=index_data.n_probes,
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result)
    return result


@app.get("/collections/{collection_id}/index/stats", tags=["Index"])
async def collection_index_stats(
    collection_id: str,
    service: CollectionIndexService = Depends(get_collection_index_service),
):
    """Per-collection HNSW index status (on disk, loaded, graph stats)."""
    result = service.get_collection_index_stats(collection_id)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result)
    return result


@app.post(
    "/collections/{collection_id}/ingest/text",
    response_model=Dict[str, Any],
    tags=["Ingest"],
    status_code=status.HTTP_201_CREATED,
)
async def ingest_text(
    collection_id: str,
    body: TextIngestRequest,
    service: MultimodalService = Depends(get_multimodal_service),
):
    """
    Embed text server-side and store as a vector in the collection.
    """
    result = service.ingest_text(
        collection_id=collection_id,
        text=body.text,
        metadata=body.metadata,
        vector_id=body.vector_id,
    )
    if not result["success"]:
        status_code = 404 if "not found" in result.get("message", "").lower() else 400
        raise HTTPException(status_code=status_code, detail=result)
    return result


@app.post("/collections/{collection_id}/search/text", tags=["Search"])
async def search_collection_text(
    collection_id: str,
    body: TextSearchRequest,
    service: MultimodalService = Depends(get_multimodal_service),
):
    """
    Natural-language search: embeds query text, then searches within the collection.
    """
    result = service.search_text(
        collection_id=collection_id,
        query=body.query,
        k=body.k,
        method=body.method.value if body.method else "brute",
        ef_search=body.ef_search,
        n_probes=body.n_probes,
        use_rerank=body.use_rerank,
        filters=body.filters,
    )
    if not result["success"]:
        raise HTTPException(status_code=_http_error_from_result(result), detail=result)
    return result


@app.post("/collections/{collection_id}/search", tags=["Search"], include_in_schema=False)
async def search_collection_text_legacy(
    collection_id: str,
    body: TextSearchRequest,
    service: MultimodalService = Depends(get_multimodal_service),
):
    """Deprecated alias for /search/text."""
    return await search_collection_text(collection_id, body, service)


@app.post(
    "/collections/{collection_id}/ingest/image",
    response_model=Dict[str, Any],
    tags=["Ingest"],
    status_code=status.HTTP_201_CREATED,
)
async def ingest_image(
    collection_id: str,
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    vector_id: Optional[str] = Form(None),
    service: MultimodalService = Depends(get_multimodal_service),
):
    """Upload an image; embed with CLIP and store in the collection."""
    raw = await _read_upload(file)
    result = service.ingest_image(
        collection_id=collection_id,
        file=raw,
        filename=file.filename or "upload.jpg",
        metadata=_parse_metadata_form(metadata),
        vector_id=vector_id,
    )
    if not result["success"]:
        raise HTTPException(status_code=_http_error_from_result(result), detail=result)
    return result


@app.post(
    "/collections/{collection_id}/ingest/audio",
    response_model=Dict[str, Any],
    tags=["Ingest"],
    status_code=status.HTTP_201_CREATED,
)
async def ingest_audio(
    collection_id: str,
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    vector_id: Optional[str] = Form(None),
    service: MultimodalService = Depends(get_multimodal_service),
):
    """Upload audio; embed with librosa MFCC features and store in the collection."""
    raw = await _read_upload(file)
    result = service.ingest_audio(
        collection_id=collection_id,
        file=raw,
        filename=file.filename or "upload.wav",
        metadata=_parse_metadata_form(metadata),
        vector_id=vector_id,
    )
    if not result["success"]:
        raise HTTPException(status_code=_http_error_from_result(result), detail=result)
    return result


@app.post("/collections/{collection_id}/search/image", tags=["Search"])
async def search_collection_image(
    collection_id: str,
    file: UploadFile = File(...),
    k: int = Form(5),
    method: SearchMethod = Form(SearchMethod.BRUTE),
    ef_search: Optional[int] = Form(None),
    n_probes: Optional[int] = Form(None),
    use_rerank: Optional[bool] = Form(True),
    metadata_filters: Optional[str] = Form(None, alias="filters"),
    service: MultimodalService = Depends(get_multimodal_service),
):
    """Search by image similarity (query file embedded server-side)."""
    raw = await _read_upload(file)
    filters = _parse_metadata_form(metadata_filters)
    result = service.search_image(
        collection_id=collection_id,
        file_or_path=raw,
        k=k,
        method=method.value if method else "brute",
        ef_search=ef_search,
        n_probes=n_probes,
        use_rerank=use_rerank,
        filters=filters,
    )
    if not result["success"]:
        raise HTTPException(status_code=_http_error_from_result(result), detail=result)
    return result


@app.post("/collections/{collection_id}/search/audio", tags=["Search"])
async def search_collection_audio(
    collection_id: str,
    file: UploadFile = File(...),
    k: int = Form(5),
    method: SearchMethod = Form(SearchMethod.BRUTE),
    ef_search: Optional[int] = Form(None),
    n_probes: Optional[int] = Form(None),
    use_rerank: Optional[bool] = Form(True),
    metadata_filters: Optional[str] = Form(None, alias="filters"),
    service: MultimodalService = Depends(get_multimodal_service),
):
    """Search by audio similarity (query file embedded server-side)."""
    raw = await _read_upload(file)
    filters = _parse_metadata_form(metadata_filters)
    result = service.search_audio(
        collection_id=collection_id,
        file_or_path=raw,
        k=k,
        method=method.value if method else "brute",
        ef_search=ef_search,
        n_probes=n_probes,
        use_rerank=use_rerank,
        filters=filters,
    )
    if not result["success"]:
        raise HTTPException(status_code=_http_error_from_result(result), detail=result)
    return result


@app.get("/media", tags=["Media"])
async def get_stored_media(content_uri: str = Query(..., description="content_uri from vector metadata")):
    """
    Serve a file previously stored during image/audio ingest.

    Pass the `content_uri` value from vector metadata (e.g. `media_storage/my-catalog/abc.jpg`).
    """
    try:
        path = resolve_media_path(content_uri)
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"success": False, "message": str(exc)}) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail={"success": False, "message": "File not found"})
    return FileResponse(path)

# ==================== Vector Operations ====================

@app.post("/vectors", response_model=Dict[str, Any], tags=["Vectors"],
          status_code=status.HTTP_201_CREATED)
async def create_vector(
    vector_data: VectorCreate,
    service: VectorService = Depends(get_vector_service)
):
    """
    Create a new vector
    
    - **vector**: Vector data as list of floats
    - **metadata**: Optional metadata dictionary
    - **vector_id**: Optional custom vector ID
    """
    result = service.create_vector(
        vector_data=vector_data.vector,
        metadata=vector_data.metadata,
        vector_id=vector_data.vector_id
    )
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result)
    
    return result

@app.post("/vectors/batch", response_model=BatchResponse, tags=["Vectors"],
          status_code=status.HTTP_201_CREATED)
async def create_vector_batch(
    batch_data: BatchInsert,
    service: VectorService = Depends(get_vector_service)
):
    """
    Create multiple vectors in a batch
    
    - **vectors**: List of vector dictionaries with 'vector', 'vector_id', and 'metadata'
    - **batch_name**: Optional batch name
    - **description**: Optional batch description
    """
    result = service.create_vector_batch(
        vectors=batch_data.vectors,
        batch_name=batch_data.batch_name,
        description=batch_data.description
    )
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result)
    
    return result

@app.get("/vectors", tags=["Vectors"])
async def get_all_vectors(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of vectors"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    service: VectorService = Depends(get_vector_service)
):
    """
    Get all vectors with pagination
    
    - **limit**: Maximum number of vectors to return (1-1000)
    - **offset**: Offset for pagination
    """
    result = service.get_all_vectors(limit=limit, offset=offset)
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result)
    
    return result

@app.get("/vectors/{vector_id}", tags=["Vectors"])
async def get_vector(
    vector_id: str,
    service: VectorService = Depends(get_vector_service)
):
    """
    Get a vector by ID
    
    - **vector_id**: Vector ID
    """
    result = service.get_vector(vector_id)
    
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result)
    
    return result

@app.put("/vectors/{vector_id}", tags=["Vectors"])
async def update_vector(
    vector_id: str,
    update_data: VectorUpdate,
    service: VectorService = Depends(get_vector_service)
):
    """
    Update a vector
    
    - **vector_id**: Vector ID
    - **vector**: Optional new vector data
    - **metadata**: Optional new metadata
    """
    result = service.update_vector(
        vector_id=vector_id,
        vector_data=update_data.vector,
        metadata=update_data.metadata
    )
    
    if not result["success"]:
        raise HTTPException(status_code=404 if "not found" in result.get("message", "") else 400, 
                          detail=result)
    
    return result

@app.delete("/vectors/{vector_id}", tags=["Vectors"])
async def delete_vector(
    vector_id: str,
    service: VectorService = Depends(get_vector_service)
):
    """
    Delete a vector
    
    - **vector_id**: Vector ID
    """
    result = service.delete_vector(vector_id)
    
    if not result["success"]:
        raise HTTPException(status_code=404 if "not found" in result.get("message", "") else 400, 
                          detail=result)
    
    return result

# ==================== Search Operations ====================@app.post("/search", response_model=Dict[str, Any], tags=["Search"])

async def search_vectors(
    search_data: SearchRequest,
    service: VectorService = Depends(get_vector_service)
):
    """
    Search for similar vectors
    
    - **query_vector**: Query vector
    - **k**: Number of results (1-100)
    - **method**: Search method (hnsw, ivf, brute, pq, hybrid)
    - **ef_search**: HNSW search parameter
    - **n_probes**: IVF probes to search
    - **use_rerank**: IVF rerank for accuracy
    - **cross_encoder_rerank**: Enable cross-encoder re-ranking (requires query_text)
    - **rerank_top_k**: Candidates to send to cross-encoder (default: k*3)
    - **filters**: Optional metadata filters
    """
    result = service.search_vectors(
        query_vector=search_data.query_vector,
        k=search_data.k,
        method=search_data.method.value if search_data.method else 'hnsw',
        ef_search=search_data.ef_search,
        n_probes=search_data.n_probes,
        use_rerank=search_data.use_rerank,
        filters=search_data.filters,
        cross_encoder_rerank=search_data.cross_encoder_rerank,
        rerank_top_k=search_data.rerank_top_k,
        query_text=search_data.query_text,
    )
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result)
    
    return result

@app.get("/search/compare", tags=["Search"])
async def compare_search_methods(
    query_vector: str = Query(..., description="Query vector as comma-separated values"),
    k: int = Query(5, ge=1, le=100, description="Number of results"),
    service: VectorService = Depends(get_vector_service)
):
    """
    Compare search methods
    
    - **query_vector**: Query vector as comma-separated values
    - **k**: Number of results
    """
    try:
        query_list = [float(x) for x in query_vector.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid query vector format")
    
    result = service.compare_search_methods(query_list, k)
    return result

@app.post("/search/batch", tags=["Search"])
async def batch_search(
    queries: List[SearchRequest],
    service: VectorService = Depends(get_vector_service)
):
    """
    Perform multiple searches in batch
    
    - **queries**: List of search requests
    """
    results = []
    for i, query in enumerate(queries):
        result = service.search_vectors(
            query_vector=query.query_vector,
            k=query.k,
            method=query.method.value if query.method else 'hnsw',
            ef_search=query.ef_search,
            n_probes=query.n_probes,
            use_rerank=query.use_rerank,
            filters=query.filters,
            cross_encoder_rerank=query.cross_encoder_rerank,
            rerank_top_k=query.rerank_top_k,
            query_text=query.query_text,
        )
        results.append({
            "query_index": i,
            "result": result
        })
    
    return {
        "success": True,
        "results": results,
        "total_queries": len(queries)
    }

# ==================== Index Operations ====================

@app.post("/index", response_model=Dict[str, Any], tags=["Index"])
async def create_index(
    index_data: IndexCreate,
    service: VectorService = Depends(get_vector_service)
):
    """
    Create an index
    
    - **method**: Indexing method (hnsw, ivf)
    - **m**: HNSW: Number of neighbors
    - **m0**: HNSW: Neighbors in layer 0
    - **ef_construction**: HNSW: Construction parameter
    - **n_clusters**: IVF: Number of clusters
    - **n_probes**: IVF: Number of probes
    """
    result = service.create_index(
        method=index_data.method.value if index_data.method else 'hnsw',
        m=index_data.m,
        m0=index_data.m0,
        ef_construction=index_data.ef_construction,
        n_clusters=index_data.n_clusters,
        n_probes=index_data.n_probes,
        collection_id=index_data.collection_id,
    )
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result)
    
    return result

@app.post("/index/save", tags=["Index"])
async def save_index(
    method: SearchMethod = Query(SearchMethod.HNSW, description="Indexing method"),
    collection_id: Optional[str] = Query(None, description="Collection scope"),
    service: VectorService = Depends(get_vector_service)
):
    """
    Save index to disk
    
    - **method**: Indexing method
    """
    result = service.save_index(method.value if method else 'hnsw', collection_id=collection_id)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result)
    
    return result

@app.post("/index/load", tags=["Index"])
async def load_index(
    method: SearchMethod = Query(SearchMethod.HNSW, description="Indexing method"),
    collection_id: Optional[str] = Query(None, description="Collection scope"),
    service: VectorService = Depends(get_vector_service)
):
    """
    Load index from disk
    
    - **method**: Indexing method
    """
    result = service.load_index(method.value if method else 'hnsw', collection_id=collection_id)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result)
    
    return result

@app.get("/index", tags=["Index"])
async def get_index_info(
    method: SearchMethod = Query(SearchMethod.HNSW, description="Indexing method"),
    service: VectorService = Depends(get_vector_service)
):
    """
    Get index information
    
    - **method**: Indexing method
    """
    result = service.get_index_info(method.value if method else 'hnsw')
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result)
    
    return result

# ==================== Statistics ====================

@app.get("/stats", response_model=StatsResponse, tags=["Stats"])
async def get_statistics(
    service: VectorService = Depends(get_vector_service)
):
    """
    Get database statistics
    """
    result = service.get_database_stats()
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result)
    
    return result

# ==================== Playground Support ====================

@app.get("/playground/templates", response_model=ApiTemplateListResponse, tags=["Playground"])
async def list_api_templates(db: Session = Depends(get_db)):
    templates = db.query(ApiTemplate).order_by(ApiTemplate.created_at.desc()).all()
    return {
        "success": True,
        "templates": [template.to_dict() for template in templates]
    }


@app.post(
    "/playground/templates",
    response_model=ApiTemplateResponse,
    tags=["Playground"],
    status_code=status.HTTP_201_CREATED
)
async def create_api_template(template: ApiTemplateCreate, db: Session = Depends(get_db)):
    normalized_method = template.method.strip().upper()
    normalized_path = template.path.strip() or "/"

    record = ApiTemplate(
        name=template.name.strip(),
        description=template.description,
        method=normalized_method,
        path=normalized_path,
        payload=template.payload
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "success": True,
        "template": record.to_dict()
    }


@app.post(
    "/playground/feedback",
    response_model=FeedbackResponse,
    tags=["Playground"],
    status_code=status.HTTP_201_CREATED
)
async def submit_feedback(entry: FeedbackCreate, db: Session = Depends(get_db)):
    record = FeedbackEntry(
        name=entry.name,
        email=entry.email,
        rating=entry.rating,
        message=entry.message.strip()
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "success": True,
        "feedback": record.to_dict()
    }

# ==================== Root ====================

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint
    """
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }

# ==================== VectorIndexer Routes ====================

# Include VectorIndexer routes if available
if INDEXER_AVAILABLE:
    app.include_router(indexer_router)
    logger.info("✅ VectorIndexer API routes successfully integrated")
    logger.info("   Endpoints available at: /api/indexer/*")
    logger.info("   See /docs for complete API documentation")
else:
    logger.warning("⚠️ VectorIndexer API not available - using standard Vector API only")

# ==================== RAG Routes ====================

from api.routers.rag import router as rag_router
app.include_router(rag_router, tags=["RAG"])
logger.info("RAG API routes integrated")

# ==================== Streaming RAG Routes ====================

from api.routers.streaming_rag import router as streaming_rag_router
app.include_router(streaming_rag_router)
logger.info("Streaming RAG API routes integrated")

# Auth API routes
from api.routers.auth_api import router as auth_router
app.include_router(auth_router, tags=["Auth"])
logger.info("Auth API routes integrated")

# Tenant API routes
from api.routers.tenants import router as tenant_router
app.include_router(tenant_router, tags=["Tenants"])
logger.info("Tenant API routes integrated")

# WebSocket search routes
from api.routers.ws_search import router as ws_router
app.include_router(ws_router, tags=["WebSocket"])
logger.info("WebSocket search routes integrated")

# ==================== Dashboard Routes ====================

from api.routers.dashboard import router as dashboard_router
from api.routers.dashboard import mount_static
app.include_router(dashboard_router, tags=["Dashboard"])
mount_static(app)
logger.info("Dashboard routes integrated")

# ==================== Enhanced Search Routes ====================

from api.routers.search_enhanced import router as search_enhanced_router
app.include_router(search_enhanced_router)
logger.info("Enhanced search API routes integrated")

# ==================== OpenAI-Compatible API Routes ====================

from api.routers.openai_compat import router as openai_compat_router
app.include_router(openai_compat_router, tags=["OpenAI-Compatible"])
logger.info("OpenAI-compatible API routes integrated")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info"
    )
