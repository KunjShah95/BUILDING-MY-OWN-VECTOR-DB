"""
End-to-end multimodal demo using the Vector DB Python SDK.

Requires the API running locally (see scripts/run_local.ps1 or docs/LOCAL_SETUP.md).
"""

from __future__ import annotations

import sys
import uuid

from vector_db_client import VectorDBClient


def _print_search(label: str, result) -> None:
    print(f"\n--- {label} ---")
    print(f"method: {getattr(result, 'method', '?')}  hits: {len(result.results)}")
    for i, hit in enumerate(result.results, 1):
        meta = hit.metadata or {}
        text = meta.get("text", meta.get("filename", hit.vector_id))
        print(f"  {i}. {hit.vector_id}  dist={hit.distance:.4f}  {text}")


def main() -> int:
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    coll_id = f"demo-text-{uuid.uuid4().hex[:8]}"

    print(f"API: {base_url}")
    print(f"Collection: {coll_id}")

    with VectorDBClient(base_url=base_url) as client:
        col = client.collections.create(
            name="Demo docs",
            collection_id=coll_id,
            modality="text",
            dimension=384,
        )
        print(f"Created collection: {col.collection_id} (dim={col.dimension})")

        client.multimodal.ingest_text(
            coll_id,
            "Python is great for machine learning and data pipelines.",
            metadata={"topic": "python"},
        )
        client.multimodal.ingest_text(
            coll_id,
            "JavaScript powers interactive web applications in the browser.",
            metadata={"topic": "js"},
        )

        brute = client.multimodal.search_text(coll_id, "machine learning with Python", k=2)
        _print_search("Search (brute)", brute)

        print("\nBuilding per-collection HNSW index...")
        index_result = client.collections.build_index(coll_id, method="hnsw", m=8, ef_construction=100)
        print(index_result.get("message", index_result))

        stats = client.collections.index_stats(coll_id)
        print("Index stats:", stats.get("stats", stats))

        hnsw = client.multimodal.search_text(
            coll_id, "machine learning with Python", k=2, method="hnsw"
        )
        _print_search("Search (hnsw)", hnsw)

        try:
            from PIL import Image
            import io

            img = Image.new("RGB", (64, 64), color=(120, 80, 200))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            img_coll = f"demo-img-{uuid.uuid4().hex[:8]}"
            client.collections.create(
                name="Demo images",
                collection_id=img_coll,
                modality="image",
                dimension=512,
            )
            client.multimodal.ingest_image(img_coll, data=buf.getvalue(), filename="purple.jpg")
            print(f"\nImage collection '{img_coll}': ingest OK (optional CLIP path)")
        except Exception as exc:
            print(f"\nSkipping image demo (optional deps): {exc}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
