from fastapi import FastAPI, Depends, HTTPException
from database.vector_database import VectorDatabase
from config.database import SessionLocal
from typing import List, Dict, Any, Optional
import numpy as np

app = FastAPI(title="Enhanced Vector Database API", version="1.0.0")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def read_root():
    return {"message": "Welcome to Enhanced Vector Database API"}

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

@app.post("/search")
def search_vectors(query_vector: List[float], k: int = 5, 
                  distance_metric: str = 'cosine', 
                  filters: Dict[str, Any] = None, db=Depends(get_db)):
    """
    Search for similar vectors
    """
    vector_db = VectorDatabase(db)
    result = vector_db.search(query_vector, k, distance_metric, filters)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.get("/vectors/{vector_id}")
def get_vector(vector_id: str, db=Depends(get_db)):
    """
    Get a specific vector by ID
    """
    vector_db = VectorDatabase(db)
    result = vector_db.get_vector(vector_id)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    return result

@app.put("/vectors/{vector_id}")
def update_vector(vector_id: str, metadata: Dict[str, Any] = None,
                 vector_data: List[float] = None, db=Depends(get_db)):
    """
    Update an existing vector
    """
    vector_db = VectorDatabase(db)
    result = vector_db.update_vector(vector_id, metadata, vector_data)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.delete("/vectors/{vector_id}")
def delete_vector(vector_id: str, db=Depends(get_db)):
    """
    Delete a vector by ID
    """
    vector_db = VectorDatabase(db)
    result = vector_db.delete_vector(vector_id)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.get("/stats")
def get_stats(db=Depends(get_db)):
    """
    Get database statistics
    """
    vector_db = VectorDatabase(db)
    result = vector_db.get_database_stats()
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.get("/vectors")
def get_all_vectors(limit: int = 1000, offset: int = 0, db=Depends(get_db)):
    """
    Get all vectors with pagination
    """
    vector_db = VectorDatabase(db)
    result = vector_db.get_all_vectors(limit, offset)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.get("/metadata/fields")
def get_metadata_fields(db=Depends(get_db)):
    """
    Get all unique metadata fields
    """
    vector_db = VectorDatabase(db)
    result = vector_db.get_vector_metadata_fields()
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.get("/batches")
def get_all_batches(db=Depends(get_db)):
    """
    Get all batches
    """
    vector_db = VectorDatabase(db)
    result = vector_db.get_all_batches()
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
