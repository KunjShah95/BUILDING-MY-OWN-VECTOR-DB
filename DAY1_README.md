# Vector Database - Day 1 Summary

## Project Overview

Building a custom vector database from scratch using PostgreSQL and Python. This is a learning project to understand vector storage, similarity search, and database optimization techniques.

---

## ✅ What's Been Completed (Day 1)

### 1. **Project Structure**

```
BUILDING MY OWN VECTOR DB/
├── main.py                      # Entry point and demo usage
├── requirements.txt             # Dependencies
├── .env                         # Environment variables (DB connection)
├── api/
│   └── main.py                 # FastAPI endpoints (placeholder)
├── config/
│   └── database.py             # SQLAlchemy engine and session setup
├── database/
│   ├── schema.py               # Vector ORM model
│   └── vector_database.py      # VectorDatabase wrapper class
├── models/
│   └── vector_model.py         # VectorModel with CRUD & search logic
├── utils/
│   └── distance.py             # Distance metric calculations
└── test/
    └── test_vector_db.py       # Unit tests with pytest
```

### 2. **Database Schema**

Created a PostgreSQL table `vectors` with the following columns:

- `id` (Integer, Primary Key)
- `vector_data` (Float Array) - Stores the vector embeddings
- `meta_data` (Text) - JSON metadata associated with each vector
- `vector_id` (String, Unique Index) - Custom identifier for vectors
- `created_at` (DateTime, Auto-generated)
- `updated_at` (DateTime, Auto-generated)

### 3. **Core Features Implemented**

#### **Vector Operations**

- ✅ `insert_vector()` - Add vectors with metadata to the database
- ✅ `search()` - Find k-most similar vectors (using brute force)
- ✅ `get_vector()` - Retrieve a specific vector by ID
- ✅ `get_all_vectors()` - Fetch all vectors in the database
- ✅ `delete_vector()` - Remove a vector by ID
- ✅ `get_database_stats()` - Get statistics (total vectors, dimensions, size)

#### **Distance Metrics**

Located in `utils/distance.py`:

- ✅ **Euclidean Distance** - L2 norm distance
- ✅ **Cosine Distance** - 1 - cosine similarity (best for embeddings)
- Returns normalized distance values for comparison

### 4. **Technology Stack**

- **Database**: PostgreSQL with psycopg2 driver
- **ORM**: SQLAlchemy 2.0.46
- **API Framework**: FastAPI 0.104.1 (prepared but not yet used)
- **Server**: Uvicorn 0.24.0
- **Numerical Computing**: NumPy 2.4.1
- **Testing**: pytest 7.0+
- **Environment**: Python with virtual environment

### 5. **Testing Infrastructure**

Created `test/test_vector_db.py` with test cases:

- `test_insert_vector()` - Validates vector insertion
- `test_search_vector()` - Validates search functionality
- `test_database_stats()` - Validates statistics retrieval

---

## 🔍 Brute Force Search Implementation

### **Where is Brute Force Applied?**

**File**: `models/vector_model.py` (Lines 57-86)  
**Method**: `search_vectors()`

### **How It Works**

```python
def search_vectors(self, query_vector: List[float], k: int = 5, 
                  distance_metric: str = 'cosine') -> List[Dict[str, Any]]:
    """
    Search for similar vectors using brute force
    """
    vectors = self.get_all_vectors()              # Step 1: Load ALL vectors
    results = []
    
    for vector in vectors:                         # Step 2: Loop through EVERY vector
        distance = self._calculate_distance(query_vector, vector.vector_data, distance_metric)
        results.append({...})                      # Step 3: Calculate distance for each
    
    results.sort(key=lambda x: x["distance"])     # Step 4: Sort by distance
    return results[:k]                             # Step 5: Return top k
```

### **Brute Force Algorithm Breakdown**

| Step | Operation | Complexity | Details |
|------|-----------|-----------|---------|
| 1 | Load all vectors from DB | O(n) | Executes SQL query: `SELECT * FROM vectors` |
| 2 | Iterate through each vector | O(n) | Loop through n vectors |
| 3 | Calculate distance | O(d) | d = vector dimensions (currently 128) |
| 4 | Sort results | O(n log n) | Sorting by distance values |
| 5 | Return top k | O(k) | Slice top k results |
| **TOTAL** | **Search Operation** | **O(n×d + n log n)** | For 128D vectors |

### **Why Brute Force?**

✅ **Pros:**

- Simple to implement and understand
- **100% accurate** - no approximations
- Works well for small datasets (< 100K vectors)
- Easy to debug and verify
- No preprocessing required

❌ **Cons:**

- **O(n×d)** complexity - doesn't scale for large datasets
- Must load all vectors into memory
- Query time increases linearly with dataset size
- Each search touches every record in the database

### **Performance Characteristics**

For a dataset with:

- **100 vectors** → ~1-5ms per search
- **1K vectors** → ~10-50ms per search
- **10K vectors** → ~100-500ms per search
- **100K+ vectors** → SLOW (seconds per query)

---

## 🚀 How to Use

### **1. Setup Database**

```bash
# Ensure PostgreSQL is running
# Grant table creation privileges to vector_user:
psql -U postgres -d vector_db -c "GRANT CREATE ON SCHEMA public TO vector_user;"
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run the Demo**

```bash
python main.py
```

### **4. Run Tests**

```bash
pytest test/test_vector_db.py -v
```

---

## 📊 Example Usage

```python
from database.vector_database import VectorDatabase
from config.database import SessionLocal
import numpy as np

# Initialize database
db = SessionLocal()
vector_db = VectorDatabase(db)

# Insert vectors
vector_data = np.random.rand(128).tolist()
result = vector_db.insert_vector(
    vector_data, 
    metadata={"text": "sample embedding", "id": 1},
    vector_id="vec_1"
)

# Search for similar vectors
query_vector = np.random.rand(128).tolist()
results = vector_db.search(query_vector, k=5, distance_metric='cosine')

# Get statistics
stats = vector_db.get_database_stats()
print(f"Total vectors: {stats['total_vectors']}")

db.close()
```

---

## 🔄 Next Steps (Day 2+)

### **Optimization Techniques to Implement:**

1. **Indexing** - Add database indexes for faster retrieval
2. **Approximate Nearest Neighbors (ANN)** - FAISS or Annoy library
3. **Vector Quantization** - Reduce memory footprint
4. **Batch Processing** - Handle multiple searches efficiently
5. **Caching** - Cache frequent search results
6. **Dimensionality Reduction** - PCA or similar techniques
7. **API Endpoints** - Complete FastAPI integration
8. **Performance Benchmarks** - Speed testing at scale

---

## 📝 Key Learnings

- **Brute Force is baseline**: Essential for small datasets and verification
- **Database design matters**: Proper indexing and schema design is crucial
- **Distance metrics**: Different metrics suited for different use cases
- **Scalability trade-offs**: Accuracy vs Speed vs Memory tradeoffs
- **Testing importance**: Unit tests validate correctness before optimization

---

## 📦 Dependencies

```
psycopg2-binary>=2.9.9      # PostgreSQL driver
numpy>=1.26.0                # Numerical operations
fastapi==0.104.1             # Web framework
uvicorn==0.24.0              # ASGI server
sqlalchemy==2.0.46           # ORM
python-dotenv==1.0.0         # Environment variables
pytest>=7.0.0                # Testing framework
```

---

**Created**: January 29, 2026  
**Status**: Day 1 Completion ✅  
**Next Review**: Day 2 Optimization Sprint
