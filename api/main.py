from fastapi import FastAPI, Depends, HTTPException
from database.hnsw_database import HNSWVectorDatabase
from config.database import SessionLocal
from typing import List, Dict, Any, Optional
import numpy as np

app = FastAPI(title="HNSW Vector Database API", version="1.0.0")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def read_root():
    return {"message": "Welcome to HNSW Vector Database API"}

@app.post("/vectors")
def insert_vector(vector_data: List[float], metadata: Dict[str, Any] = None, 
                 vector_id: str = None, db=Depends(get_db)):
    """
    Insert a vector into the database
    """
    vector_db = HNSWVectorDatabase(db)
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
    vector_db = HNSWVectorDatabase(db)
    result = vector_db.insert_vector_batch(vectors, batch_name, description)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.post("/index/hnsw")
def create_hnsw_index(m: int = 16, m0: int = None, 
                     ef_construction: int = 200, db=Depends(get_db)):
    """
    Create an HNSW index
    
    Args:
        m: Number of neighbors per node
        m0: Number of neighbors in layer 0
        ef_construction: Construction parameter
    """
    vector_db = HNSWVectorDatabase(db)
    result = vector_db.create_hnsw_index(m, m0, ef_construction)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.post("/index/hnsw/load")
def load_hnsw_index(db=Depends(get_db)):
    """
    Load HNSW index from disk
    """
    vector_db = HNSWVectorDatabase(db)
    result = vector_db.load_hnsw_index()
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.post("/index/hnsw/save")
def save_hnsw_index(db=Depends(get_db)):
    """
    Save HNSW index to disk
    """
    vector_db = HNSWVectorDatabase(db)
    result = vector_db.save_hnsw_index()
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.post("/index/hnsw/rebuild")
def rebuild_hnsw_index(m: int = 16, m0: int = None, 
                      ef_construction: int = 200, db=Depends(get_db)):
    """
    Rebuild HNSW index
    
    Args:
        m: Number of neighbors per node
        m0: Number of neighbors in layer 0
        ef_construction: Construction parameter
    """
    vector_db = HNSWVectorDatabase(db)
    result = vector_db.rebuild_hnsw_index(m, m0, ef_construction)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.post("/search/hnsw")
def search_hnsw(query_vector: List[float], k: int = 5, 
               ef_search: int = None, db=Depends(get_db)):
    """
    Search using HNSW index
    
    Args:
        query_vector: Query vector
        k: Number of results to return
        ef_search: Search parameter (higher = better recall)
    """
    vector_db = HNSWVectorDatabase(db)
    result = vector_db.search_hnsw(query_vector, k, ef_search)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.post("/search")
def search_vectors(query_vector: List[float], k: int = 5, 
                  method: str = 'hnsw', ef_search: int = None, db=Depends(get_db)):
    """
    Search using specified method
    
    Args:
        query_vector: Query vector
        k: Number of results to return
        method: Search method ('hnsw', 'brute')
        ef_search: HNSW search parameter
    """
    vector_db = HNSWVectorDatabase(db)
    result = vector_db.search(query_vector, k, method, ef_search)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.post("/search/compare")
def compare_search_methods(query_vector: List[float], k: int = 5, db=Depends(get_db)):
    """
    Compare different search methods
    """
    vector_db = HNSWVectorDatabase(db)
    result = vector_db.compare_search_methods(query_vector, k)
    return result

@app.get("/index/hnsw/info")
def get_hnsw_index_info(db=Depends(get_db)):
    """
    Get HNSW index information
    """
    vector_db = HNSWVectorDatabase(db)
    result = vector_db.get_hnsw_index_info()
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.delete("/vectors/{vector_id}")
def delete_vector(vector_id: str, db=Depends(get_db)):
    """
    Delete a vector by ID
    """
    vector_db = HNSWVectorDatabase(db)
    result = vector_db.delete_vector(vector_id)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.get("/stats")
def get_stats(db=Depends(get_db)):
    """
    Get database statistics
    """
    vector_db = HNSWVectorDatabase(db)
    result = vector_db.get_database_stats()
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
