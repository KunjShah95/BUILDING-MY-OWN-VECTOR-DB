"""gRPC server that wraps the existing VectorService."""
import json
import logging
from concurrent import futures

import grpc
import numpy as np
from sqlalchemy.orm import Session

from api.grpc import vector_service_pb2, vector_service_pb2_grpc
from config.database import SessionLocal
from services.vector_service import VectorService

logger = logging.getLogger(__name__)


class VectorDBServicer(vector_service_pb2_grpc.VectorDBServicer):
    """gRPC servicer delegating to the existing VectorService."""

    def _get_service(self) -> VectorService:
        db: Session = SessionLocal()
        return VectorService(db_session=db)

    def Search(self, request, context):
        svc = self._get_service()
        query = list(request.query_vector)
        filters = dict(request.filters) if request.filters else None
        result = svc.search_vectors(
            query_vector=query,
            k=request.k,
            method=request.method or "hnsw",
            ef_search=request.ef_search if request.ef_search else None,
            n_probes=request.n_probes if request.n_probes else None,
            collection_id=request.collection_id or None,
            filters=filters,
            distance_metric=request.distance_metric or "cosine",
        )
        pb_results = [
            vector_service_pb2.SearchResult(
                id=r["vector_id"],
                score=r["distance"],
                metadata=json.dumps(r.get("metadata", {})),
            )
            for r in result.get("results", [])
        ]
        return vector_service_pb2.SearchResponse(
            success=result.get("success", False),
            results=pb_results,
            total_results=result.get("total_results", 0),
            search_time_ms=result.get("search_time", 0.0) * 1000,
            method=result.get("method", request.method),
            message=result.get("message", ""),
        )

    def Insert(self, request, context):
        svc = self._get_service()
        metadata = json.loads(request.metadata) if request.metadata else None
        result = svc.create_vector(
            vector_data=list(request.vector),
            metadata=metadata,
            vector_id=request.vector_id or None,
            collection_id=request.collection_id or None,
        )
        return vector_service_pb2.InsertResponse(
            success=result.get("success", False),
            vector_id=result.get("vector_id", ""),
            message=result.get("message", ""),
        )

    def BatchInsert(self, request, context):
        svc = self._get_service()
        vectors = []
        for entry in request.vectors:
            vec_dict = {
                "vector": list(entry.vector),
                "vector_id": entry.vector_id or None,
            }
            if entry.metadata:
                vec_dict["metadata"] = json.loads(entry.metadata)
            vectors.append(vec_dict)
        result = svc.create_vector_batch(
            vectors=vectors,
            batch_name=request.batch_name or None,
        )
        return vector_service_pb2.BatchInsertResponse(
            success=result.get("success", False),
            vector_count=result.get("vector_count", 0),
            message=result.get("message", ""),
        )

    def Delete(self, request, context):
        svc = self._get_service()
        vid = request.vector_id
        result = svc.delete_vector(vector_id=vid)
        return vector_service_pb2.DeleteResponse(
            success=result.get("success", False),
            message=result.get("message", ""),
        )

    def GetVector(self, request, context):
        svc = self._get_service()
        result = svc.get_vector(vector_id=request.vector_id)
        if result.get("success"):
            vec = result.get("vector", {})
            meta_str = json.dumps(vec.get("metadata", {})) if vec.get("metadata") else ""
            return vector_service_pb2.GetVectorResponse(
                success=True,
                vector_id=vec.get("vector_id", ""),
                vector=vec.get("vector", []),
                metadata=meta_str,
            )
        return vector_service_pb2.GetVectorResponse(
            success=False, message=result.get("message", ""),
        )

    def ListCollections(self, request, context):
        from services.collection_service import CollectionService
        db = SessionLocal()
        svc = CollectionService(db_session=db)
        result = svc.list_collections(limit=request.limit or 100, offset=request.offset or 0)
        colls = []
        if result.get("success"):
            for c in result.get("collections", []):
                colls.append(vector_service_pb2.CollectionInfo(
                    collection_id=c.get("collection_id", ""),
                    name=c.get("name", ""),
                    modality=c.get("modality", ""),
                    dimension=c.get("dimension", 0),
                    embedding_model=c.get("embedding_model", ""),
                    distance_metric=c.get("distance_metric", "cosine"),
                ))
        return vector_service_pb2.ListCollectionsResponse(
            success=result.get("success", False),
            collections=colls,
        )

    def Health(self, request, context):
        svc = self._get_service()
        health = svc.get_health_status()
        return vector_service_pb2.HealthResponse(
            status=health.get("status", "unhealthy"),
            database_connected=health.get("database", "") == "connected",
            index_available=health.get("index_available", False),
            total_vectors=health.get("total_vectors", 0),
        )


def serve(host: str = "0.0.0.0", port: int = 50051, max_workers: int = 10):
    """Start the gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    vector_service_pb2_grpc.add_VectorDBServicer_to_server(VectorDBServicer(), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logger.info("gRPC server started on %s:%s", host, port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
