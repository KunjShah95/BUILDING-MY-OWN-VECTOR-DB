from fastapi import FastAPI, Depends, HTTPException
from database.vector_database import VectorDatabase
from config.database import SessionLocal
from typing import List, Dict, Any, Optional
import numpy as np

app = FastAPI(title="Vector Database with Indexing API", version="1.0.0")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def read_root():
    return {"message": "Welcome to Vector Database with Indexing API"}

@app.post("/vectors")
def insert_vector(vector_data: List[float], metadata: Dict[str, Any] = None, 
                 vector_id: str = None, db=Depends(get_db)):
    """
    Insert a vector into the database
    """
    vector_db = VectorDatabase(db)
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
    vector_db = VectorDatabase(db)
    result = vector_db.insert_vector_batch(vectors, batch_name, description)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.post("/index")
def create_index(k: int = 100, force_rebuild: bool = False, db=Depends(get_db)):
    """
    Create an index using K-Means clustering
    
    Args:
        k: Number of clusters
        force_rebuild: Force rebuild even if index exists
    """
    vector_db = VectorDatabase(db)
    result = vector_db.create_index(k=k, force_rebuild=force_rebuild)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.post("/search")
def search_vectors(query_vector: List[float], k: int = 5, 
                  distance_metric: str = 'cosine', 
                  filters: Dict[str, Any] = None,
                  use_index: bool = True,
                  n_probes: int = 10, db=Depends(get_db)):
    """
    Search for similar vectors
    
    Args:
        query_vector: Query vector
        k: Number of results to return
        distance_metric: Distance metric to use
        filters: Metadata filters (not implemented in this version)
        use_index: Whether to use the index (if available)
        n_probes: Number of clusters to probe for approximate search
    """
    vector_db = VectorDatabase(db)
    result = vector_db.search(query_vector, k, distance_metric, filters, use_index, n_probes)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.get("/index/info")
def get_index_info(db=Depends(get_db)):
    """
    Get information about the current index
    """
    vector_db = VectorDatabase(db)
    result = vector_db.get_index_info()
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.get("/index/clusters/{cluster_id}")
def get_cluster_vectors(cluster_id: int, db=Depends(get_db)):
    """
    Get all vectors in a specific cluster
    """
    vector_db = VectorDatabase(db)
    result = vector_db.get_cluster_vectors(cluster_id)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.get("/index/clusters")
def get_all_clusters(db=Depends(get_db)):
    """
    Get all clusters with their vectors
    """
    vector_db = VectorDatabase(db)
    result = vector_db.get_all_clusters()
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.post("/index/rebuild")
def rebuild_index(k: int = 100, db=Depends(get_db)):
    """
    Rebuild the index with new parameters
    
    Args:
        k: Number of clusters
    """
    vector_db = VectorDatabase(db)
    result = vector_db.rebuild_index(k=k)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
