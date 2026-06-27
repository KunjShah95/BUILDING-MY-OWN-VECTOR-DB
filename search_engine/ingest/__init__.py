"""Web ingest: crawled page -> dense vector + BM25 sparse index."""

from .web_ingest import WebIngestor, doc_id_for

__all__ = ["WebIngestor", "doc_id_for"]
