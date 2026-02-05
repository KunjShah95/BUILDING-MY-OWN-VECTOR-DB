from fastapi import FastAPI, Depends, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging
import time

# Import configurations
from config.database import Base, engine, get_db
from config.settings import get_settings
from models.pydantic_models import (
    VectorCreate, VectorUpdate, VectorResponse,
    BatchInsert, BatchResponse, SearchRequest, SearchResponse,
    IndexCreate, IndexResponse, StatsResponse, HealthResponse,
    ErrorResponse, SearchMethod
)
from services.vector_service import VectorService

# Import VectorIndexer API
try:
    from examples.vector_indexer_api import router as indexer_router
    INDEXER_AVAILABLE = True
except ImportError:
    INDEXER_AVAILABLE = False

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
        {"name": "Search", "description": "Vector similarity search"},
        {"name": "Index", "description": "Index management"},
        {"name": "Stats", "description": "Statistics and monitoring"},
        {"name": "Health", "description": "Health checks"},
        {"name": "Vector Indexer", "description": "Unified HNSW/IVF Vector Indexer API"}
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

# Dependency
def get_vector_service(db: Session = Depends(get_db)) -> VectorService:
    """
    Dependency to get vector service
    """
    return VectorService(db)

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

# ==================== Search Operations ====================

@app.post("/search", response_model=Dict[str, Any], tags=["Search"])
async def search_vectors(
    search_data: SearchRequest,
    service: VectorService = Depends(get_vector_service)
):
    """
    Search for similar vectors
    
    - **query_vector**: Query vector
    - **k**: Number of results (1-100)
    - **method**: Search method (hnsw, ivf, brute)
    - **ef_search**: HNSW search parameter
    - **n_probes**: IVF probes to search
    - **use_rerank**: IVF rerank for accuracy
    - **filters**: Optional metadata filters
    """
    result = service.search_vectors(
        query_vector=search_data.query_vector,
        k=search_data.k,
        method=search_data.method.value if search_data.method else 'hnsw',
        ef_search=search_data.ef_search,
        n_probes=search_data.n_probes,
        use_rerank=search_data.use_rerank
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
            use_rerank=query.use_rerank
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
        n_probes=index_data.n_probes
    )
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result)
    
    return result

@app.post("/index/save", tags=["Index"])
async def save_index(
    method: SearchMethod = Query(SearchMethod.HNSW, description="Indexing method"),
    service: VectorService = Depends(get_vector_service)
):
    """
    Save index to disk
    
    - **method**: Indexing method
    """
    result = service.save_index(method.value if method else 'hnsw')
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result)
    
    return result

@app.post("/index/load", tags=["Index"])
async def load_index(
    method: SearchMethod = Query(SearchMethod.HNSW, description="Indexing method"),
    service: VectorService = Depends(get_vector_service)
):
    """
    Load index from disk
    
    - **method**: Indexing method
    """
    result = service.load_index(method.value if method else 'hnsw')
    
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info"
    )
