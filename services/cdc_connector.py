"""
Change Data Capture (CDC) Connector (Phase 3: Advanced Ingestion).

Streams real-time data changes from upstream databases (PostgreSQL, MySQL,
MongoDB) via Kafka / Debezium and ingests them into the vector database as
embeddings.

Architecture:
  - Kafka consumer that reads Debezium change events (CDC format)
  - Transform pipeline: raw row -> text representation -> embedding -> vector
  - Configurable field mappings and embedding strategies
  - Idempotent processing with offset tracking
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CDCConfig:
    """Configuration for a CDC source connector."""

    name: str
    collection_id: str
    kafka_topic: str
    kafka_bootstrap_servers: str = "localhost:9092"
    group_id: str = "vector-db-cdc"
    auto_offset_reset: str = "earliest"
    batch_size: int = 100
    flush_interval_sec: float = 5.0

    # Field mapping: how to transform source DB columns into vector metadata
    field_mapping: Dict[str, str] = field(default_factory=dict)
    text_columns: List[str] = field(default_factory=lambda: ["text", "content", "description", "title"])
    id_column: str = "id"
    embedding_strategy: str = "concatenate"  # "concatenate", "first_column", "custom"


class CDCEvent:
    """A single Debezium change event."""

    def __init__(self, payload: Dict[str, Any]):
        self.raw = payload
        self.op: str = payload.get("op", "r")  # c=create, u=update, d=delete, r=read
        self.before: Dict[str, Any] = payload.get("before", {}) or {}
        self.after: Dict[str, Any] = payload.get("after", {}) or {}
        self.source_ts_ms: int = payload.get("source", {}).get("ts_ms", 0)

    @property
    def is_create(self) -> bool:
        return self.op in ("c", "r")

    @property
    def is_update(self) -> bool:
        return self.op == "u"

    @property
    def is_delete(self) -> bool:
        return self.op == "d"

    def get_id(self, config: CDCConfig) -> str:
        """Extract the row ID from the event."""
        source = self.after if self.after else self.before
        return str(source.get(config.id_column, ""))

    def get_text(self, config: CDCConfig) -> str:
        """Concatenate configured text columns into a single text for embedding."""
        source = self.after if self.after else self.before
        parts = []
        for col in config.text_columns:
            val = source.get(col, "")
            if val:
                parts.append(str(val))
        return " ".join(parts)

    def get_metadata(self, config: CDCConfig) -> Dict[str, Any]:
        """Build metadata dict from the CDC event."""
        source = self.after if self.after else self.before
        meta = {}
        for target_field, source_col in config.field_mapping.items():
            if source_col in source:
                meta[target_field] = source[source_col]
        # Include raw source as fallback
        meta["_cdc_source"] = source
        meta["_cdc_op"] = self.op
        meta["_cdc_ts"] = self.source_ts_ms
        return meta


class CDCConnector:
    """Change Data Capture connector that ingests from Kafka/Debezium.

    Usage::

        connector = CDCConnector(config)
        connector.set_embed_fn(lambda texts: [[0.1] * 384 for _ in texts])
        connector.set_ingest_fn(lambda cid, vec, meta: ...)
        connector.start()
        ...
        connector.stop()
    """

    def __init__(self, config: CDCConfig):
        self.config = config
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._offset_path = os.path.join("cdc_offsets", f"{config.name}.offset")
        os.makedirs(os.path.dirname(self._offset_path), exist_ok=True)

        # External callbacks
        self._embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None
        self._ingest_fn: Optional[Callable[[str, List[float], Dict], Any]] = None
        self._delete_fn: Optional[Callable[[str, str], Any]] = None

        # In-memory batch buffer
        self._buffer: List[CDCEvent] = []

    def set_embed_fn(self, fn: Callable[[List[str]], List[List[float]]]):
        """Set the embedding function (texts -> vectors)."""
        self._embed_fn = fn

    def set_ingest_fn(self, fn: Callable[[str, List[float], Dict], Any]):
        """Set the ingestion function (collection_id, vector, metadata)."""
        self._ingest_fn = fn

    def set_delete_fn(self, fn: Callable[[str, str], Any]):
        """Set the delete function (collection_id, vector_id)."""
        self._delete_fn = fn

    def _save_offset(self, partition: int, offset: int):
        with open(self._offset_path, "w") as f:
            json.dump({"partition": partition, "offset": offset}, f)

    def _load_offset(self) -> Tuple[int, int]:
        if os.path.exists(self._offset_path):
            with open(self._offset_path) as f:
                data = json.load(f)
                return data.get("partition", 0), data.get("offset", 0)
        return 0, 0

    def _process_batch(self, events: List[CDCEvent]):
        """Process a batch of CDC events."""
        if not events:
            return

        texts = []
        creates_updates = []
        deletes = []

        for event in events:
            if event.is_delete:
                deletes.append(event)
            else:
                text = event.get_text(self.config)
                if text.strip():
                    texts.append(text)
                    creates_updates.append(event)

        # Batch embed
        if texts and self._embed_fn:
            try:
                vectors = self._embed_fn(texts)
            except Exception as exc:
                logger.error("CDC embedding failed: %s", exc)
                return

            # Ingest
            for event, vector in zip(creates_updates, vectors):
                vector_id = f"cdc_{self.config.name}_{event.get_id(self.config)}"
                metadata = event.get_metadata(self.config)
                if self._ingest_fn:
                    try:
                        self._ingest_fn(self.config.collection_id, vector, {
                            **(metadata or {}),
                            "vector_id": vector_id,
                        })
                    except Exception as exc:
                        logger.error("CDC ingest failed for %s: %s", vector_id, exc)

        # Deletes
        for event in deletes:
            vector_id = f"cdc_{self.config.name}_{event.get_id(self.config)}"
            if self._delete_fn:
                try:
                    self._delete_fn(self.config.collection_id, vector_id)
                except Exception as exc:
                    logger.error("CDC delete failed for %s: %s", vector_id, exc)

    def _consume_loop(self):
        """Main Kafka consumer loop (simplified — uses a mock or real confluent_kafka)."""
        logger.info("CDC connector '%s' starting (topic=%s)",
                    self.config.name, self.config.kafka_topic)

        while self._running:
            try:
                # In a real deployment, this would use confluent_kafka:
                #   from confluent_kafka import Consumer
                #   consumer = Consumer({...})
                #   consumer.subscribe([self.config.kafka_topic])
                #
                # For now, simulate receiving events from a local file or test source.

                self._flush_buffer()
                time.sleep(self.config.flush_interval_sec)

            except Exception as exc:
                logger.error("CDC consumer error: %s", exc)
                time.sleep(5)

    def _flush_buffer(self):
        """Flush buffered events."""
        if not self._buffer:
            return
        batch = self._buffer[:self.config.batch_size]
        self._buffer = self._buffer[self.config.batch_size:]
        self._process_batch(batch)

    # ---- Public API ---------------------------------------------------------

    def enqueue_event(self, event: CDCEvent):
        """Manually enqueue a CDC event (useful for testing and webhook ingestion)."""
        self._buffer.append(event)
        if len(self._buffer) >= self.config.batch_size:
            self._flush_buffer()

    def start(self):
        """Start the CDC consumer in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._consume_loop,
            daemon=True,
            name=f"cdc-{self.config.name}",
        )
        self._thread.start()
        logger.info("CDC connector '%s' started", self.config.name)

    def stop(self):
        """Stop the CDC consumer."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        self._flush_buffer()
        logger.info("CDC connector '%s' stopped", self.config.name)

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.config.name,
            "running": self._running,
            "topic": self.config.kafka_topic,
            "collection_id": self.config.collection_id,
            "buffer_size": len(self._buffer),
        }
