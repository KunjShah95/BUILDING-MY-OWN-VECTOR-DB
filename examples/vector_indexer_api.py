"""
Vector Indexer API Integration
FastAPI endpoints for VectorIndexer HNSW/IVF indexing
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import numpy as np

from services.vector_indexer import VectorIndexer, VectorIndexerConfig, IndexMethod
from utils.optimization import QueryOptimizer

# ============================================================================
# Pydantic Models
# ============================================================================

class IndexerConfigModel(BaseModel):
    """Indexer configuration model"""
    method: str = Field(default="hnsw", description="Index method: hnsw, ivf, hybrid")
    num_vectors: int = Field(default=10000, description="Expected number of vectors")
    vector_dim: int = Field(default=128, description="Vector dimensionality")
    recall_target: float = Field(default=0.95, description="Target recall (0-1)")
    speed_priority: bool = Field(default=False, description="Optimize for speed")


class VectorModel(BaseModel):
    """Vector model"""
    id: Optional[str] = None
    vector: List[float] = Field(description="Vector data")
    metadata: Optional[Dict[str, Any]] = None


class VectorBatchModel(BaseModel):
    """Batch of vectors"""
    vectors: List[VectorModel]
    method: Optional[str] = None


class SearchQueryModel(BaseModel):
    """Search query model"""
    query_vector: List[float] = Field(description="Query vector")
    k: int = Field(default=10, ge=1, le=1000, description="Number of results")
    method: Optional[str] = None
    ef_search: Optional[int] = None


class BatchSearchModel(BaseModel):
    """Batch search model"""
    queries: List[List[float]] = Field(description="List of query vectors")
    k: int = Field(default=10, ge=1)


class ParameterSuggestionModel(BaseModel):
    """Parameter suggestion model"""
    database_size: int = Field(ge=100, description="Number of vectors")
    recall_target: float = Field(default=0.95, ge=0.5, le=1.0)


# ============================================================================
# Global Indexer Instance
# ============================================================================

# Store indexer instance (in production, use proper state management)
_indexer: Optional[VectorIndexer] = None
_indexer_config: Optional[VectorIndexerConfig] = None


def get_indexer() -> VectorIndexer:
    """Get or create indexer instance"""
    global _indexer
    if _indexer is None:
        _indexer = VectorIndexer(VectorIndexerConfig())
    return _indexer


def set_indexer_config(config: VectorIndexerConfig):
    """Set indexer configuration"""
    global _indexer, _indexer_config
    _indexer_config = config
    _indexer = VectorIndexer(config)


# ============================================================================
# Routes
# ============================================================================

router = APIRouter(prefix="/api/indexer", tags=["Vector Indexer"])


@router.post("/config", summary="Configure indexer")
async def configure_indexer(config: IndexerConfigModel) -> Dict[str, Any]:
    """
    Configure the vector indexer
    
    Parameters:
    - method: Indexing method (hnsw, ivf, hybrid)
    - num_vectors: Expected dataset size
    - vector_dim: Vector dimensionality
    - recall_target: Target recall (0-1)
    - speed_priority: Optimize for throughput
    """
    try:
        indexer_config = VectorIndexerConfig(
            method=config.method,
            num_vectors=config.num_vectors,
            vector_dim=config.vector_dim,
            recall_target=config.recall_target,
            speed_priority=config.speed_priority
        )
        set_indexer_config(indexer_config)
        
        return {
            "success": True,
            "config": indexer_config.to_dict(),
            "message": "Indexer configured successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/create", summary="Create index")
async def create_index(vectors: List[List[float]] = Body(
    description="List of vectors to index"
)) -> Dict[str, Any]:
    """
    Create index from vectors
    
    Body: List of vectors (each vector is a list of floats)
    """
    try:
        indexer = get_indexer()
        result = indexer.create_index(vectors)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Creation failed"))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", summary="Search for nearest neighbors")
async def search(query: SearchQueryModel) -> Dict[str, Any]:
    """
    Search for nearest neighbors
    
    Parameters:
    - query_vector: Query vector
    - k: Number of results
    - method: Override search method (hnsw, ivf, hybrid)
    - ef_search: HNSW search parameter
    """
    try:
        indexer = get_indexer()
        
        result = indexer.search(
            query_vector=query.query_vector,
            k=query.k,
            method=query.method,
            ef_search=query.ef_search
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Search failed"))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search-batch", summary="Batch search")
async def batch_search(batch: BatchSearchModel) -> Dict[str, Any]:
    """
    Batch search multiple queries at once
    
    Parameters:
    - queries: List of query vectors
    - k: Number of results per query
    
    Returns:
    - Batch search results with performance metrics
    """
    try:
        indexer = get_indexer()
        
        result = indexer.batch_search(
            query_vectors=batch.queries,
            k=batch.k
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Batch search failed"))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", summary="Get indexer statistics")
async def get_stats() -> Dict[str, Any]:
    """Get comprehensive statistics about the indexer"""
    try:
        indexer = get_indexer()
        stats = indexer.get_stats()
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save", summary="Save index to disk")
async def save_index(path: Optional[str] = Query(None, description="Save path")) -> Dict[str, Any]:
    """
    Save index to disk
    
    Parameters:
    - path: Optional custom path (auto-generated if not provided)
    """
    try:
        indexer = get_indexer()
        result = indexer.save_index(path=path)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Save failed"))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load", summary="Load index from disk")
async def load_index(path: str = Query(description="Index path")) -> Dict[str, Any]:
    """
    Load index from disk
    
    Parameters:
    - path: Path to index directory
    """
    try:
        indexer = VectorIndexer()
        result = indexer.load_index(path)
        
        if result["success"]:
            # Replace global indexer
            global _indexer
            _indexer = indexer
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Load failed"))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/parameters/suggest", summary="Suggest optimal parameters")
async def suggest_parameters(suggestion: ParameterSuggestionModel) -> Dict[str, Any]:
    """
    Get suggested parameters based on dataset size
    
    Parameters:
    - database_size: Number of vectors in database
    - recall_target: Target recall (0-1)
    
    Returns:
    - Suggested HNSW and IVF parameters
    """
    try:
        suggestions = QueryOptimizer.suggest_index_parameters(
            database_size=suggestion.database_size,
            recall_target=suggestion.recall_target
        )
        
        return {
            "success": True,
            "database_size": suggestion.database_size,
            "recall_target": suggestion.recall_target,
            "suggestions": suggestions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/methods", summary="Get supported index methods")
async def get_methods() -> Dict[str, Any]:
    """Get list of supported indexing methods"""
    return {
        "methods": [
            {
                "name": "hnsw",
                "description": "Hierarchical Navigable Small World - Fast, balanced",
                "use_cases": ["General purpose", "Balanced performance"]
            },
            {
                "name": "ivf",
                "description": "Inverted File - Fast building, simple parameters",
                "use_cases": ["Large datasets", "Speed critical"]
            },
            {
                "name": "hybrid",
                "description": "Combined HNSW+IVF - Best recall",
                "use_cases": ["High precision", "Recall > 0.99"]
            }
        ]
    }


@router.get("/health", summary="Health check")
async def health() -> Dict[str, Any]:
    """Check indexer health"""
    try:
        indexer = get_indexer()
        stats = indexer.get_stats()
        
        return {
            "status": "healthy",
            "indexer_ready": stats["vector_count"] > 0,
            "vectors_indexed": stats["vector_count"],
            "hnsw_available": stats["hnsw_available"],
            "ivf_available": stats["ivf_available"]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# ============================================================================
# Example Integration with FastAPI App
# ============================================================================

def add_indexer_routes(app):
    """
    Add Vector Indexer routes to FastAPI application
    
    Usage in main.py:
    
    from fastapi import FastAPI
    from services.vector_indexer_api import add_indexer_routes
    
    app = FastAPI()
    add_indexer_routes(app)
    
    # Now API has endpoints:
    # POST /api/indexer/config - Configure indexer
    # POST /api/indexer/create - Create index
    # POST /api/indexer/search - Search
    # POST /api/indexer/search-batch - Batch search
    # POST /api/indexer/parameters/suggest - Suggest parameters
    # GET /api/indexer/stats - Get stats
    # GET /api/indexer/health - Health check
    # POST /api/indexer/save - Save index
    # POST /api/indexer/load - Load index
    """
    app.include_router(router)


# ============================================================================
# Alternative: Standalone ASGI Application
# ============================================================================

if __name__ == "__main__":
    from fastapi import FastAPI
    import uvicorn
    
    app = FastAPI(
        title="Vector Database Indexer API",
        description="Unified HNSW/IVF vector indexer API",
        version="1.0.0"
    )
    
    # Add routes
    add_indexer_routes(app)
    
    # Run
    uvicorn.run(app, host="0.0.0.0", port=8001)
