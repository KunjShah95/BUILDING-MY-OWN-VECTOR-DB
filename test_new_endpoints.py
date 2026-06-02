"""
Comprehensive test script for new endpoints:
1. GraphQL API
2. Time-series vectors
3. Async ingestion queue
4. Per-collection IVF save/load/rebuild
"""
import sys
from datetime import datetime, timedelta

import httpx

BASE = "http://localhost:8000"
API_KEY = "your-api-key-here"
client = httpx.Client(base_url=BASE, timeout=30, headers={"X-API-Key": API_KEY}, follow_redirects=True)

passed = 0
failed = 0


def check(name, response, expect_success=True, status_code=None):
    global passed, failed
    try:
        if status_code:
            assert response.status_code == status_code, \
                f"Expected {status_code}, got {response.status_code}: {response.text[:200]}"
        if expect_success:
            data = response.json()
            assert data.get("success", False), \
                f"Expected success=True, got: {response.text[:300]}"
        print(f"  [PASS] {name}")
        passed += 1
    except AssertionError as e:
        print(f"  [FAIL] {name}: {e}")
        failed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        failed += 1


# Clean up any existing collections (gracefully handle failures)
print("\n=== CLEANUP ===")
try:
    r = client.delete("/collections/new-features-test")
    print(f"  Cleanup collection: HTTP {r.status_code}")
except Exception as e:
    print(f"  Cleanup collection skipped: {e}")
try:
    r = client.delete("/collections/gql-test")
    print(f"  Cleanup gql-test: HTTP {r.status_code}")
except Exception as e:
    print(f"  Cleanup gql-test skipped: {e}")

# ==================== Setup: Create a collection ====================
print("\n=== SETUP: Creating test collection ===")
r = client.post("/collections", json={
    "name": "New Features Test",
    "collection_id": "new-features-test",
    "modality": "text",
    "dimension": 128,
})
check("Create collection", r, status_code=201)

print("\n=== SETUP: Inserting test vectors ===")
for i in range(10):
    vec = [float(i) / 10.0] * 128
    r = client.post("/vectors", json={
        "vector": vec,
        "metadata": {"text": f"Test vector {i}", "index": i},
        "vector_id": f"test_vec_{i:03d}",
    })
    check(f"Insert vector {i}", r)

# GraphQL helper
import json as _json

def gql_check(name, response, expect_data=True):
    """Check GraphQL responses (use 'data' key instead of 'success')."""
    global passed, failed
    try:
        data = response.json()
        assert "errors" not in data or not data["errors"], \
            f"GraphQL errors: {_json.dumps(data.get('errors'), indent=2)[:300]}"
        if expect_data:
            assert "data" in data, f"Expected 'data' key, got: {response.text[:300]}"
            assert data["data"] is not None, f"data is null: {_json.dumps(data, indent=2)[:300]}"
        print(f"  [PASS] {name}")
        passed += 1
    except AssertionError as e:
        print(f"  [FAIL] {name}: {e}")
        failed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        failed += 1

# ==================== 1. GraphQL API ====================
print("\n=== 1. GRAPHQL API ===")

r = client.post("/graphql", json={
    "query": "{ health { status database indexAvailable totalVectors } }"
})
gql_check("GraphQL: health query", r)
data = r.json()
print(f"     Health: {data['data']['health']}")

r = client.post("/graphql", json={
    "query": "{ collections { collectionId name modality } }"
})
gql_check("GraphQL: collections query", r)
data = r.json()
print(f"     Collections: {len(data['data']['collections'])} found")

r = client.post("/graphql", json={
    "query": 'mutation { createCollection(name: "GQL Test", collectionId: "gql-test", modality: "text", dimension: 128) { success message collectionId } }'
})
gql_check("GraphQL: createCollection", r)
data = r.json()
print(f"     Create: {data['data']['createCollection']}")

r = client.post("/graphql", json={
    "query": 'mutation { createVector(vector: [0.1, 0.2, 0.3], collectionId: "gql-test") { success message vectorId } }'
})
gql_check("GraphQL: createVector", r)
data = r.json()
print(f"     Vector: {data['data']['createVector']}")

r = client.post("/graphql", json={
    "query": '{ searchVectors(queryVector: [0.1, 0.2, 0.3], k: 3, method: "brute") { success totalResults results { vectorId distance } } }'
})
gql_check("GraphQL: searchVectors", r)
data = r.json()
print(f"     Search results: {data['data']['searchVectors']['totalResults']}")

r = client.post("/graphql", json={
    "query": 'mutation { buildIndex(method: "hnsw", collectionId: "new-features-test", m: 8, efConstruction: 100) { success message } }'
})
gql_check("GraphQL: buildIndex", r)
data = r.json()
print(f"     Index: {data['data']['buildIndex']}")

# Cleanup gql-test
client.delete("/collections/gql-test")

# ==================== 2. TIME-SERIES ENDPOINTS ====================
print("\n=== 2. TIME-SERIES VECTORS ===")

base_ts = datetime.utcnow()
for i in range(5):
    ts = (base_ts + timedelta(hours=i)).isoformat()
    vec = [float(i) * 0.1] * 128
    r = client.post(
        f"/collections/new-features-test/vectors/timeseries",
        json={"vector": vec, "series_id": "sensor-a", "timestamp": ts, "metadata": {"reading": i}},
    )
    check(f"TS insert sensor-a {i}", r, status_code=201)

for i in range(3):
    ts = (base_ts + timedelta(hours=i * 2)).isoformat()
    vec = [0.5 + float(i) * 0.05] * 128
    r = client.post(
        f"/collections/new-features-test/vectors/timeseries",
        json={"vector": vec, "series_id": "sensor-b", "timestamp": ts, "metadata": {"reading": i + 10}},
    )
    check(f"TS insert sensor-b {i}", r, status_code=201)

r = client.get(f"/collections/new-features-test/series")
check("List series", r)
data = r.json()
print(f"     Series: {data.get('series')}")

queries = ",".join(["0.1"] * 128)
r = client.get(
    f"/collections/new-features-test/search/timeseries",
    params={"query_vector": queries, "series_id": "sensor-a", "k": 3},
)
check("TS search", r)
data = r.json()
print(f"     Results: {data.get('total_results')}")

r = client.get(f"/collections/new-features-test/series/sensor-a/latest", params={"k": 2})
check("Latest per series", r)
data = r.json()
print(f"     Latest: {data.get('count')}")

r = client.get(f"/collections/new-features-test/series/sensor-a/aggregate", params={"window": "1h"})
check("Aggregate series", r)
data = r.json()
print(f"     Windows: {data.get('count')}")

# ==================== 3. INGESTION QUEUE ====================
print("\n=== 3. INGESTION QUEUE ===")

r = client.get("/ingest/status")
check("Queue status", r)
data = r.json()
print(f"     Size: {data.get('queue_size')}, batch: {data.get('batch_size')}")

r = client.post("/ingest/enqueue", json={
    "vector": [0.42] * 128,
    "collection_id": "new-features-test",
    "metadata": {"source": "queue-test"},
})
check("Enqueue single", r, status_code=202)

vectors = [[float(i) * 0.2] * 128 for i in range(5)]
r = client.post("/ingest/enqueue-many", json={
    "vectors": vectors,
    "collection_id": "new-features-test",
    "metadata_list": [{"idx": i} for i in range(5)],
})
check("Enqueue many", r, status_code=202)
print(f"     {r.json().get('message')}")

r = client.post("/ingest/flush")
check("Flush queue", r)
print(f"     {r.json().get('message')}")

r = client.get("/ingest/status")
check("Queue status after flush", r)
print(f"     Size after: {r.json().get('queue_size')}")

# ==================== 4. IVF SAVE/LOAD/REBUILD ====================
print("\n=== 4. PER-COLLECTION IVF ===")

r = client.post(
    f"/collections/new-features-test/index",
    json={"method": "ivf", "n_clusters": 3, "n_probes": 2},
)
check("Build IVF", r)

r = client.get(f"/collections/new-features-test/index/stats")
check("Index stats", r)
data = r.json()
print(f"     method={data['stats'].get('method')}, indexed={data['stats'].get('is_indexed')}")

r = client.post(f"/collections/new-features-test/index/save", params={"method": "ivf"})
check("Save IVF", r)

r = client.post(f"/collections/new-features-test/index/load", params={"method": "ivf"})
check("Load IVF", r)

r = client.post(f"/collections/new-features-test/index/ivf/rebuild", params={"n_clusters": 5, "n_probes": 3})
check("Rebuild IVF", r)

# ==================== RESULTS ====================
print("\n" + "=" * 50)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
print("=" * 50)

client.close()
sys.exit(0 if failed == 0 else 1)
