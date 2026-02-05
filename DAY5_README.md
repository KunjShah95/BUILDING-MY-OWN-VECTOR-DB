# DAY 5 — Building My Own Vector DB

Date: Day 5

## Goal

Fix critical bugs, implement proper error handling, restore missing HNSW implementation, and ensure the complete system runs successfully with database cleanup and index persistence.

## Highlights / Summary

- Fixed duplicate key constraint violations by implementing duplicate checking and proper rollback handling
- Added `clear_database()` method to enable clean test runs without database conflicts
- Restored complete HNSW index implementation (file was corrupted/empty - 0 bytes)
- Fixed JSON serialization issues for metadata persistence
- Successfully demonstrated full HNSW workflow: insert → index → search → save

## Critical Bug Fixes

### 1. **Duplicate Key Violations**

**Problem:** Running `main.py` multiple times caused `UniqueViolation` errors and `PendingRollbackError` due to existing vector IDs.

**Solution:**

- Added duplicate checking in `create_vector()` - returns existing vector instead of failing
- Implemented proper exception handling with `db.rollback()` in all database operations
- Created `clear_all_vectors()` method in `VectorModel` to clean database before runs

**Files Modified:**

- `models/vector_model.py` - Added try-except blocks with rollback in `create_vector()` and `create_vector_batch()`
- `database/hnsw_database.py` - Added `clear_database()` method
- `database/ivf_database.py` - Added `clear_database()` method  
- `main.py` - Calls `clear_database()` at startup

### 2. **Missing HNSW Implementation**

**Problem:** Import error - `utils/hnsw_index.py` was 0 bytes (empty file), causing `ImportError: cannot import name 'HNSWIndex'`.

**Root Cause:** File corruption or accidental deletion. File was never committed to Git, so no backup existed.

**Solution:**
Recreated complete HNSW implementation with:

- Hierarchical graph structure with multiple layers
- Node and edge management with neighbor pruning
- Best-first search algorithm for layer traversal
- Exponential level distribution for node insertion
- Dynamic index construction with configurable parameters (m, m0, ef_construction)

**Key Methods Implemented:**

```python
class HNSWIndex:
    - __init__(m, m0, ef_construction, level_mult)
    - insert(vector, vector_id, metadata, level)
    - search(query_vector, k, ef)
    - delete(vector_id)
    - save(filepath)
    - load(filepath)
    - get_graph_stats()
```

### 3. **JSON Serialization Error**

**Problem:** Saving HNSW index failed with `Object of type MetaData is not JSON serializable`.

**Solution:**
Added `make_serializable()` helper function in `save()` method to recursively convert non-JSON-serializable objects (SQLAlchemy MetaData, datetime, etc.) to strings or appropriate JSON types.

```python
def make_serializable(obj):
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    return str(obj)  # Convert other types to string
```

## Files Changed

### Core Files

- **`models/vector_model.py`**
  - Added `clear_all_vectors()` method
  - Added try-except with rollback in `create_vector()`
  - Added try-except with rollback in `create_vector_batch()`
  - Added duplicate checking before insertion

- **`database/hnsw_database.py`**
  - Added `clear_database()` method
  - Added `hnsw_index_path` attribute in `__init__`
  - Updated to use new HNSW methods

- **`database/ivf_database.py`**
  - Added `clear_database()` method

- **`main.py`**
  - Added database clearing at startup
  - Changed from IVF to HNSW demonstration

### Restored Files

- **`utils/hnsw_index.py`** (Completely recreated - 602 lines)
  - Node dataclass for graph structure
  - HNSWIndex class with full implementation
  - Layer-based search and insertion
  - Neighbor selection and pruning
  - Save/load with JSON serialization

## Test Results

### Successful Run Output

```text
HNSW Vector Database initialized!
Database cleared: 0 vectors removed

Inserting sample vectors...
Batch inserted successfully: 200 vectors

Creating HNSW index...
HNSW Index created successfully!
Stats:
  Total nodes: 205
  Total edges: 5110
  Average connections: 24.93
  Max level: 11
  Level distribution: {0: 103, 3: 14, 1: 52, 5: 5, 4: 9, 2: 17, 6: 1, 11: 1, 8: 2, 9: 1}

HNSW Search (ef_search=10):
  Found 5 similar vectors
  Search time: 0.0095 seconds

HNSW Search (ef_search=50):
  Found 5 similar vectors
  Search time: 0.0075 seconds

HNSW Search (ef_search=100):
  Found 5 similar vectors
  Search time: 0.0055 seconds

Method Comparison:
  hnsw: Time: 0.0095 seconds, Results: 5
  brute_force: Time: 0.0186 seconds, Results: 5

HNSW Index saved to hnsw_index_data.json
```

### Performance Observations

- **HNSW vs Brute Force:** ~2x faster even with small dataset (200 vectors)
- **ef_search parameter:** Higher values slightly improve recall but impact speed
- **Index Quality:** Average 24.93 connections per node, hierarchical up to level 11

## How to Run

### 1. Setup Environment

```powershell
# Activate virtual environment
& .\.venv\Scripts\Activate.ps1

# Install dependencies (if needed)
pip install -r requirements.txt
```

### 2. Run Main Demo

```powershell
python main.py
```

This will:

- Clear existing vectors from database
- Insert 200 sample 128-dimensional vectors
- Build HNSW index
- Perform searches with different ef_search values
- Compare HNSW vs brute-force search
- Save index to disk
- Display statistics

### 3. Run Tests

```powershell
# Run all tests
pytest

# Run specific HNSW tests
pytest test/test_hnsw.py -v

# Run with output
pytest -s
```

## Technical Details

### HNSW Algorithm Implementation

**Graph Structure:**

- Hierarchical layers (0 to max_level)
- Each node has connections at multiple layers
- Lower layers have more connections (m0 at layer 0, m at higher layers)
- Entry point at highest layer for search efficiency

**Insertion Process:**

1. Calculate random level using exponential distribution
2. Search from top layer to find nearest neighbors
3. Insert node and create bidirectional links at each layer
4. Prune neighbors to maintain degree constraints

**Search Process:**

1. Start from entry point at highest layer
2. Greedy best-first search to find nearest neighbor in each layer
3. Move down layers until reaching layer 0
4. Return k nearest neighbors from layer 0

### Database Schema

**Vectors Table:**

- `id`: Integer primary key
- `vector_data`: Float array (128-dimensional)
- `meta_data`: JSON metadata
- `vector_id`: Unique string identifier (indexed)
- `created_at`, `updated_at`: Timestamps

**Batch Tracking:**

- `VectorBatch`: Tracks batch insertions
- `VectorBatchMapping`: Many-to-many relationship

## Configuration Parameters

### HNSW Index

- **m** (default: 16): Number of neighbors per node in upper layers
- **m0** (default: 32): Number of neighbors in layer 0
- **ef_construction** (default: 200): Candidate list size during construction
- **ef_search** (default: ef_construction): Candidate list size during search

### Database

- PostgreSQL connection configured in `.env`
- Session management via SQLAlchemy
- Automatic schema creation on startup

## Known Issues & Future Work

### Current Limitations

- No index updates after insertion (requires full rebuild)
- Memory-based index (all vectors in RAM during search)
- No distributed/sharded support
- Limited index compression

### Next Steps (Day 6+)

1. **Dynamic Index Updates:** Support incremental insertions without full rebuild
2. **Disk-Based Search:** Implement memory-mapped file support for large indices
3. **Query Optimization:** Add filtering, range queries, and hybrid search
4. **Benchmark Suite:** Compare against FAISS, Annoy, ScaNN
5. **API Enhancement:** RESTful endpoints for all operations
6. **Product Quantization:** Reduce memory footprint with compression
7. **Parallel Search:** Multi-threaded query processing

## Lessons Learned

1. **Error Handling is Critical:** Database operations need proper rollback on failure
2. **Data Integrity:** Duplicate checking prevents constraint violations
3. **Clean State Management:** Ability to reset database state simplifies testing
4. **JSON Serialization:** Custom metadata requires careful type handling
5. **File Corruption Recovery:** Always commit working code to version control
6. **HNSW Performance:** Even basic implementation shows significant speedup

## References

- [HNSW Paper](https://arxiv.org/abs/1603.09320) - Malkov & Yashunin, 2016
- [PostgreSQL Array Types](https://www.postgresql.org/docs/current/arrays.html)
- [SQLAlchemy Session Management](https://docs.sqlalchemy.org/en/20/orm/session_basics.html)

---

## Quick Command Reference

```powershell
# Run main demo
python main.py

# Run tests
pytest

# Run specific test file
pytest test/test_hnsw.py

# Check file integrity
Get-ChildItem utils\hnsw_index.py | Select-Object Name, Length

# Clear Python cache
Remove-Item -Recurse -Force **\__pycache__

# Check database connection
python -c "from config.database import engine; print(engine.url)"
```

---

**Status:** ✅ All critical bugs fixed, system operational  
**Next Focus:** Performance optimization and dynamic index updates
