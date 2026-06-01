# Time-Series Vector Support Design

## Overview
Enable the vector database to store and query time-series vectors — sequences of vectors
indexed by timestamp, grouped into logical series. This unlocks use-cases such as
real-time embedding streams, IoT sensor embedding logs, and financial tick embeddings.

## Data Model

### New Columns on `Vector` Model
| Column       | Type      | Description                                 |
|-------------|-----------|---------------------------------------------|
| `timestamp`  | datetime  | When this vector was recorded (UTC)         |
| `series_id`  | str       | Logical grouping key for a time-series      |

Existing columns (`vector_id`, `vector`, `metadata`, `collection_id`) remain unchanged.

### Example Row
```
vector_id: "sensor-a-20260101T0000"
vector:    [0.12, -0.34, 0.56, ...]
timestamp: 2026-01-01T00:00:00Z
series_id: "sensor-a"
metadata:  {"unit": "celsius", "floor": 3}
```

## API Extensions

### Insert with Timestamp
`POST /collections/{collection_id}/vectors/timeseries`
```json
{
  "vector": [0.1, 0.2, 0.3],
  "series_id": "sensor-a",
  "timestamp": "2026-01-01T00:00:00Z",
  "metadata": {}
}
```

### Time-Range Search
`GET /collections/{collection_id}/search/timeseries?from=...&to=...&series_id=...`

### Most Recent per Series
`GET /collections/{collection_id}/series/{series_id}/latest`

### Aggregate over Time Window
`GET /collections/{collection_id}/series/{series_id}/aggregate?window=1h`

## Index Strategy

### Hybrid Index: Time-Range Filter + ANN
- **Time filter first**: Use a B-tree index on `(collection_id, series_id, timestamp)` to
  quickly narrow to the relevant time window.
- **ANN on filtered set**: Run HNSW / IVF / brute-force search only over vectors
  within the time range.
- The `VectorService.search_vectors()` method already accepts a `filters` dict — add
  `timestamp__gte`, `timestamp__lte`, and `series_id` as built-in filter keys.

### Partitioning
For large time-series collections (millions+ of vectors):
- Partition the `Vector` table by month (`timestamp`)
- Store per-partition HNSW indexes under `indexes/{collection_id}/{partition}/`
- Query planner prunes partitions outside the requested time range

## Implementation Plan

### Phase 1 — Schema + Basic Queries
1. Add `timestamp` and `series_id` columns to the `Vector` ORM model.
2. Add a composite index: `idx_vector_ts (collection_id, series_id, timestamp)`.
3. Extend `VectorModel.create_vector()` to accept `timestamp` and `series_id`.
4. Add `TimeSeriesService` with `insert_timeseries()`, `search_timeseries()`,
   `get_latest_per_series()`, and `aggregate_series()` methods.
5. Expose via a new `api/routers/timeseries.py` router.

### Phase 2 — Time-Windowed ANN
1. Modify `VectorService.search_vectors()` to interpret `timestamp__gte`/`timestamp__lte`
   filters as SQL time-range predicates before passing the filtered ID set to ANN.
2. Benchmark time-range + ANN vs. brute-force over time range.

### Phase 3 — Partitioning (Scale)
1. Implement table partitioning by month on `timestamp`.
2. Implement partitioned HNSW index storage.
3. Add partition-aware query routing.

## Open Questions
- Should `series_id` be at the collection level or the vector level? (Chosen: vector-level
  for flexibility — one collection can hold many series.)
- Retention policy: automatic TTL-based eviction of old vector data? (Defer to later phase.)
- Write-optimized ingestion: the `BulkIngestionQueue` in `services/ingestion_service.py`
  should be extended to accept timestamped batches.
