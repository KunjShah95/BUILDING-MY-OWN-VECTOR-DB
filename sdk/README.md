# Vector DB Python Client

Sync HTTP client for the multimodal Vector Database API.

## Install

From the repository root:

```bash
pip install -e sdk
```

Or add `sdk` to `PYTHONPATH`:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/BUILDING MY OWN VECTOR DB/sdk"
```

## Quickstart

```python
from vector_db_client import VectorDBClient

client = VectorDBClient("http://localhost:8000")

# Image collection (CLIP 512-dim by default on server)
col = client.collections.create(
    name="Photos",
    collection_id="photos",
    modality="image",
    dimension=512,
)

client.multimodal.ingest_image("photos", path="cat.jpg")
results = client.multimodal.search_image("photos", path="query.jpg", k=5)
for hit in results.results:
    print(hit.vector_id, hit.distance)

# Text collection
docs = client.collections.create(
    name="Docs",
    collection_id="docs",
    modality="text",
)
client.multimodal.ingest_text("docs", "Returns accepted within 30 days.")
hits = client.multimodal.search_text("docs", "return policy")

client.close()
```

## API surface

| Resource | Methods |
|----------|---------|
| `client.collections` | `create`, `list`, `get`, `delete` |
| `client.vectors` | `create`, `get`, `delete`, `search` |
| `client.multimodal` | `ingest_text`, `search_text`, `ingest_image`, `ingest_audio`, `search_image`, `search_audio` |

Errors raise `VectorDBHTTPError` with `status_code` and `detail`.
