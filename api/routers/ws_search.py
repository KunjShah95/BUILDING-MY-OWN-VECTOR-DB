from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import asyncio
import time
import logging

from services.collection_index_service import CollectionIndexService
from services.embedding_service import embed_text

logger = logging.getLogger(__name__)
router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/search/{collection_id}")
async def websocket_search(websocket: WebSocket, collection_id: str):
    await websocket.accept()

    try:
        data = await websocket.receive_json()
        query_text = data.get("query", "")
        k = int(data.get("k", 10))
        method = data.get("method", "hnsw")

        if not query_text:
            await websocket.send_json({"type": "error", "message": "Query is required"})
            await websocket.close()
            return

        start = time.time()

        query_vector = embed_text(query_text)

        from config.database import get_db
        db = next(get_db())
        try:
            index_svc = CollectionIndexService(db)
            result = index_svc.search_collection_indexed(
                collection_id=collection_id,
                query_vector=query_vector,
                k=k,
                method=method,
                distance_metric="cosine",
            )

            if not result.get("success"):
                await websocket.send_json({"type": "error", "message": result.get("message", "Search failed")})
                await websocket.close()
                return

            results = result.get("results", [])
            elapsed = time.time() - start

            for rank, r in enumerate(results, 1):
                await websocket.send_json({
                    "type": "result",
                    "vector_id": r.get("vector_id") or r.get("id"),
                    "distance": r.get("distance", 0),
                    "rank": rank,
                    "metadata": r.get("metadata"),
                })
                await asyncio.sleep(0)

            await websocket.send_json({
                "type": "done",
                "total": len(results),
                "time_ms": round(elapsed * 1000, 2),
                "method": method,
            })
        finally:
            db.close()

        await websocket.close()

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {collection_id}")
    except Exception as e:
        logger.exception("WebSocket search error")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close()
        except Exception:
            pass


@router.websocket("/ws/health")
async def websocket_health(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"status": "healthy", "service": "vector-db-ws"})
    await websocket.close()
