# ANN Search Engine → Vector DB Merge Design

**Date:** 2026-06-21  
**Status:** Approved

## Goal

Merge unique components from `C:\ann search engine` into `C:\BUILDING MY OWN VECTOR DB`. Skip anything the vector DB already has a better version of.

## What Gets Merged

### 1. `utils/base_index.py` — BaseIndex ABC
Abstract base class with abstract methods: `add`, `search`, `delete`, `clear`, `save`, `load`, `get_stats`, plus helpers `_validate_vector` and `_default_vector_ids`. Vector DB has many index implementations but no shared interface. This unifies them.

### 2. `utils/brute_force_index.py` — BruteForceIndex
Exact nearest-neighbor search via vectorized numpy. Supports cosine, euclidean, and dot-product distance. O(N) search — not scalable but gives 100% recall. Useful for small collections, ground-truth benchmarking, and testing recall of approximate indexes. Extends `BaseIndex`.

### 3. `services/ann_index_service.py` — AnnIndexService
Named index lifecycle management: create (build + populate from DB vectors), save to disk, load from disk, get stats. Imports vector DB's own `HNSWIndex` and `IVFIndex` from `utils/`, plus the new `BruteForceIndex`. The ANN engine's original HNSW/IVF implementations are skipped.

### 4. `models/pydantic_models.py` — Additions
- `IndexType` enum: `HNSW`, `IVF`, `BRUTE` (BRUTE is new — vector DB only has HNSW/IVF/etc in `SearchMethod`)
- `IndexCreateRequest`: parameters for building any index type
- `IndexStatsResponse`: structured response for index info
- `DistanceMetric.DOT`: dot-product support (missing from existing enum)

### 5. `api/routers/ann_indexes.py` — New Router
Prefix: `/api/v1/ann/`

| Method | Path | Action |
|--------|------|--------|
| POST | `/index` | Build + populate index from stored vectors |
| GET | `/index` | Info and stats for all/specific index |
| POST | `/index/save` | Persist index to disk |
| POST | `/index/load` | Load index from disk |
| GET | `/search/compare` | Run query across HNSW+IVF+Brute, return timing comparison |

Router registered in `api/main.py`.

## What's Skipped

| ANN Component | Reason |
|---|---|
| `indexes/hnsw_index.py` | Wrapper over brute force — not real HNSW; vector DB has full graph implementation |
| `indexes/ivf_index.py` | Vector DB IVF has product quantization, more mature |
| `services/search_service.py` | Vector DB has richer multi-stage pipeline |
| `services/vector_service.py` | Vector DB has its own |
| `api/routes.py` (vector CRUD, health, search) | Already covered by vector DB's existing API |
| ANN database schema | Incompatible with vector DB schema |

## File Map

```
utils/base_index.py          ← new (from ANN indexes/base_index.py)
utils/brute_force_index.py   ← new (from ANN indexes/brute_force.py, adapted)
services/ann_index_service.py ← new (from ANN services/index_service.py, adapted)
models/pydantic_models.py    ← additions only
api/routers/ann_indexes.py   ← new
api/main.py                  ← register new router
```

## Key Adaptations

- `BruteForceIndex` import path changed: `from utils.base_index import BaseIndex`
- `AnnIndexService` uses `from utils.hnsw_index import HNSWIndex` and `from utils.ivf_index import IVFIndex` (vector DB's implementations)
- New router uses `from config.database import get_db` (vector DB's DB config)
- New Pydantic classes added to existing `models/pydantic_models.py` (no new file)
