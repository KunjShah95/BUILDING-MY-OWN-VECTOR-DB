from fastapi import FastAPI, Depends
from database.vector_database import VectorDatabase
from config.database import SessionLocal
from typing import List, Dict, Any
import numpy as np

app = FastAPI(title="Vector Database API", version="1.0.0")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def read_root():
    return {"message": "Welcome to Vector Database API"}

@app.post("/vectors")
def insert_vector(vector_data: List[float], metadata: Dict[str, Any] = None, 
                 vector_id: str = None, db=Depends(get_db)):
    """
    Insert a vector into the database
    """
    vector_db = VectorDatabase(db)
    result = vector_db.insert_vector(vector_data, metadata, vector_id)
    return result

@app.post("/search")
def search_vectors(query_vector: List[float], k: int = 5, 
                  distance_metric: str = 'cosine', db=Depends(get_db)):
    """
    Search for similar vectors
    """
    vector_db = VectorDatabase(db)
    result = vector_db.search(query_vector, k, distance_metric)
    return result

@app.get("/vectors/{vector_id}")
def get_vector(vector_id: str, db=Depends(get_db)):
    """
    Get a specific vector by ID
    """
    vector_db = VectorDatabase(db)
    result = vector_db.get_vector(vector_id)
    return result

@app.get("/stats")
def get_stats(db=Depends(get_db)):
    """
    Get database statistics
    """
    vector_db = VectorDatabase(db)
    result = vector_db.get_database_stats()
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
