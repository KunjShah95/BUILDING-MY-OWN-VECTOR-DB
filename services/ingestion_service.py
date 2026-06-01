"""Bulk ingestion queue for batched vector inserts."""
import asyncio
import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BulkIngestionQueue:
    """Async bulk ingestion queue that batches vector inserts.

    Accumulates items in an in-memory deque and flushes them to the
    database when the batch size is reached or the flush interval elapses.
    """

    def __init__(self, batch_size: int = 100, flush_interval: float = 5.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._queue: deque[tuple[str, np.ndarray, Optional[Dict[str, Any]]]] = deque()
        self._task: Optional[asyncio.Task] = None
        self._last_flush: float = time.time()

    async def enqueue(
        self,
        collection_id: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a single item to the ingestion queue."""
        self._queue.append((collection_id, vector, metadata))
        if len(self._queue) >= self.batch_size:
            await self._flush()

    async def enqueue_many(
        self,
        collection_id: str,
        vectors: List[np.ndarray],
        metadata_list: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> None:
        """Add multiple items at once."""
        if metadata_list is None:
            metadata_list = [None] * len(vectors)
        for vec, meta in zip(vectors, metadata_list):
            self._queue.append((collection_id, vec, meta))
        if len(self._queue) >= self.batch_size:
            await self._flush()

    async def _flush(self) -> None:
        """Flush up to batch_size items from the queue."""
        batch: list[tuple[str, np.ndarray, Optional[Dict[str, Any]]]] = []
        while self._queue and len(batch) < self.batch_size:
            batch.append(self._queue.popleft())
        if not batch:
            return
        await self._process_batch(batch)
        self._last_flush = time.time()

    async def _process_batch(
        self,
        batch: list[tuple[str, np.ndarray, Optional[Dict[str, Any]]]],
    ) -> None:
        """Insert a batch of vectors into the database."""
        from services.vector_service import VectorService
        from config.database import SessionLocal

        db = SessionLocal()
        try:
            svc = VectorService(db_session=db)
            for collection_id, vector, metadata in batch:
                vec_list = vector.tolist() if isinstance(vector, np.ndarray) else vector
                svc.create_vector(
                    vector_data=vec_list,
                    metadata=metadata,
                    collection_id=collection_id,
                )
            db.commit()
            logger.info("Ingested batch of %d items", len(batch))
        except Exception:
            db.rollback()
            logger.exception("Batch ingestion failed, rolled back")
            raise
        finally:
            db.close()

    async def flush_all(self) -> None:
        """Force-flush all remaining items."""
        while self._queue:
            await self._flush()

    async def start_periodic_flush(self) -> None:
        """Start a background task that flushes on a timer."""
        if self._task is not None and not self._task.done():
            return

        async def _periodic():
            while True:
                await asyncio.sleep(self.flush_interval)
                if self._queue and (time.time() - self._last_flush) >= self.flush_interval:
                    await self._flush()

        self._task = asyncio.create_task(_periodic())

    async def stop_periodic_flush(self) -> None:
        """Stop the periodic flush background task."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    @property
    def size(self) -> int:
        return len(self._queue)

    @property
    def is_flush_needed(self) -> bool:
        return (
            len(self._queue) > 0
            and (time.time() - self._last_flush) >= self.flush_interval
        )
