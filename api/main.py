from fastapi import FastAPI, Depends, HTTPException
from database.ivf_database import IVFVectorDatabase
from config.database import SessionLocal
from typing import List, Dict, Any, Optional
import numpy as np

app = FastAPI(title="IVF Vector Database API", version="1.0.0")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def read_root():
    return {"message": "Welcome to IVF Vector Database API"}

@app.post("/vectors")
def insert_vector(vector_data: List[float], metadata: Dict[str, Any] = None, 
                 vector_id: str = None, db=Depends(get_db)):
    """
    Insert a vector into the database
    """
    vector_db = IVFVectorDatabase(db)
    result = vector_db.insert_vector(vector_data, metadata, vector_id)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.post("/vectors/batch")
def insert_vector_batch(vectors: List[Dict[str, Any]], batch_name: str = None,
                       description: str = None, db=Depends(get_db)):
    """
    Insert a batch of vectors
    """
    vector_db = IVFVectorDatabase(db)
    result = vector_db.insert_vector_batch(vectors, batch_name, description)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.post("/index/ivf")
def create_ivf_index(n_clusters: int = 100, n_probes: int = 10, 
                    force_rebuild: bool = False, db=Depends(get_db)):
    """
    Create an IVF index
    
    Args:
        n_clusters: Number of clusters
        n_probes: Number of clusters to probe during search
        force_rebuild: Force rebuild even if index exists
    """
    vector_db = IVFVectorDatabase(db)
    result = vector_db.create_ivf_index(n_clusters, n_probes, force_rebuild)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.post("/index/ivf/load")
def load_ivf_index(db=Depends(get_db)):
    """
    Load IVF index from disk
    """
    vector_db = IVFVectorDatabase(db)
    result = vector_db.load_ivf_index()
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.post("/index/ivf/save")
def save_ivf_index(db=Depends(get_db)):
    """
    Save IVF index to disk
    """
    vector_db = IVFVectorDatabase(db)
    result = vector_db.save_ivf_index()
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.post("/index/ivf/rebuild")
def rebuild_ivf_index(n_clusters: int = 100, n_probes: int = 10, db=Depends(get_db)):
    """
    Rebuild the IVF index
    
    Args:
        n_clusters: Number of clusters
        n_probes: Number of clusters to probe
    """
    vector_db = IVFVectorDatabase(db)
    result = vector_db.rebuild_ivf_index(n_clusters, n_probes)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.post("/search")
def search_vectors(query_vector: List[float], k: int = 5, 
                  use_ivf: bool = True, use_rerank: bool = True,
                  n_probes: int = 10, db=Depends(get_db)):
    """
    Search for similar vectors
    
    Args:
        query_vector: Query vector
        k: Number of results to return
        use_ivf: Whether to use IVF index
        use_rerank: Whether to use reranking
        n_probes: Number of clusters to probe
    """
    vector_db = IVFVectorDatabase(db)
    result = vector_db.search(query_vector, k, use_ivf, use_rerank, n_probes)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.post("/search/compare")
def compare_search_methods(query_vector: List[float], k: int = 5, db=Depends(get_db)):
    """
    Compare different search methods
    """
    vector_db = IVFVectorDatabase(db)
    result = vector_db.compare_search_methods(query_vector, k)
    return result

@app.get("/index/ivf/stats")
def get_index_stats(db=Depends(get_db)):
    """
    Get IVF index statistics
    """
    vector_db = IVFVectorDatabase(db)
    result = vector_db.get_index_stats()
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.get("/stats")
def get_stats(db=Depends(get_db)):
    """
    Get database statistics
    """
    vector_db = IVFVectorDatabase(db)
    result = vector_db.get_database_stats()
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
