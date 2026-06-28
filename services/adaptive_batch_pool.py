"""
Adaptive Batch Pool — asyncio-based worker pool for 10K+ QPS vector operations.

Design:
  - Requests enqueue into an asyncio.Queue.
  - A coordinator drains the queue in adaptive batches: small batches when
    the queue is shallow (low latency), large batches when deep (high QPS).
  - `max_workers` coroutines process batches concurrently.
  - Backpressure: `put()` raises `QueueFullError` when the queue exceeds
    `max_queue_size`, preventing unbounded memory growth.
  - Each request returns an `asyncio.Future` resolved when its batch completes.

Usage:

    async def my_handler(batch):
        # batch is List[Any] — the items enqueued as a group
        return [process(item) for item in batch]

    pool = AdaptiveBatchPool(handler=my_handler, max_workers=8)
    await pool.start()

    future = await pool.put(my_item)
    result = await future          # waits for this item's batch to finish

    await pool.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class QueueFullError(RuntimeError):
    """Raised when the pool queue is at capacity (backpressure)."""


@dataclass
class _Request:
    item: Any
    future: asyncio.Future
    enqueued_at: float = field(default_factory=time.monotonic)


class AdaptiveBatchPool:
    """Asyncio worker pool with adaptive batch sizing.

    Parameters
    ----------
    handler : async callable
        ``async def handler(batch: List[Any]) -> List[Any]``
        Receives a list of items; must return a list of results in the same
        order.
    max_workers : int
        Concurrent coroutines processing batches. Default 8.
    min_batch : int
        Smallest batch size dispatched. Default 1.
    max_batch : int
        Largest batch size. Default 256.
    batch_timeout : float
        Seconds to wait for min_batch items before flushing a smaller batch.
        Default 0.005 (5 ms).
    max_queue_size : int
        Hard cap on pending requests. Backpressure kicks in above this.
        Default 10_000.
    target_queue_depth : int
        When queue depth exceeds this, batch size scales up toward max_batch.
        Default 64.
    """

    def __init__(
        self,
        handler: Callable[[List[Any]], Awaitable[List[Any]]],
        max_workers: int = 8,
        min_batch: int = 1,
        max_batch: int = 256,
        batch_timeout: float = 0.005,
        max_queue_size: int = 10_000,
        target_queue_depth: int = 64,
    ):
        self._handler = handler
        self.max_workers = max_workers
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.batch_timeout = batch_timeout
        self.max_queue_size = max_queue_size
        self.target_queue_depth = target_queue_depth

        self._queue: asyncio.Queue[_Request] = asyncio.Queue()
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._coordinator: Optional[asyncio.Task] = None
        self._running = False

        # Metrics
        self._stats: Dict[str, float] = {
            "total_requests": 0,
            "total_batches": 0,
            "total_errors": 0,
            "avg_batch_size": 0.0,
            "avg_latency_ms": 0.0,
        }
        self._latencies: List[float] = []

    # ----------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        if self._running:
            return
        self._semaphore = asyncio.Semaphore(self.max_workers)
        self._running = True
        self._coordinator = asyncio.create_task(self._coordinate(), name="batch-coordinator")
        logger.info(
            "AdaptiveBatchPool started: workers=%d min_batch=%d max_batch=%d",
            self.max_workers, self.min_batch, self.max_batch,
        )

    async def stop(self, drain: bool = True) -> None:
        self._running = False
        if drain:
            # Wait for queue to empty
            try:
                await asyncio.wait_for(self._queue.join(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("BatchPool drain timed out")
        if self._coordinator:
            self._coordinator.cancel()
            try:
                await self._coordinator
            except asyncio.CancelledError:
                pass

    # ----------------------------------------------------------------- public API

    async def put(self, item: Any) -> asyncio.Future:
        """Enqueue an item. Returns a Future resolved with the item's result.

        Raises QueueFullError if queue is at capacity.
        """
        if self._queue.qsize() >= self.max_queue_size:
            raise QueueFullError(
                f"BatchPool queue full ({self.max_queue_size} items). "
                "Reduce request rate or increase max_queue_size."
            )
        loop = asyncio.get_event_loop()
        req = _Request(item=item, future=loop.create_future())
        await self._queue.put(req)
        return req.future

    async def submit(self, item: Any) -> Any:
        """Enqueue and await result in one call."""
        future = await self.put(item)
        return await future

    def qsize(self) -> int:
        return self._queue.qsize()

    def stats(self) -> Dict[str, Any]:
        s = dict(self._stats)
        s["queue_depth"] = self._queue.qsize()
        s["running"] = self._running
        if self._latencies:
            arr = sorted(self._latencies[-1000:])
            n = len(arr)
            s["p50_latency_ms"] = arr[int(n * 0.5)] * 1000
            s["p95_latency_ms"] = arr[int(n * 0.95)] * 1000
            s["p99_latency_ms"] = arr[int(n * 0.99)] * 1000
        return s

    # ----------------------------------------------------------------- internals

    async def _coordinate(self) -> None:
        """Pull items from queue in adaptive batches and dispatch to workers."""
        while self._running:
            batch = await self._collect_batch()
            if batch:
                asyncio.create_task(self._dispatch(batch))
            else:
                await asyncio.sleep(0.001)

    async def _collect_batch(self) -> List[_Request]:
        """Collect a batch using adaptive sizing based on queue depth."""
        queue_depth = self._queue.qsize()

        # Adaptive batch size: scale linearly between min and max
        if queue_depth <= 0:
            target = self.min_batch
        elif queue_depth >= self.target_queue_depth:
            target = self.max_batch
        else:
            frac = queue_depth / self.target_queue_depth
            target = int(self.min_batch + frac * (self.max_batch - self.min_batch))
            target = max(self.min_batch, min(self.max_batch, target))

        batch: List[_Request] = []
        deadline = time.monotonic() + self.batch_timeout

        while len(batch) < target:
            remaining = deadline - time.monotonic()
            if remaining <= 0 and batch:
                break
            try:
                req = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=max(remaining, 0.0001),
                )
                batch.append(req)
            except asyncio.TimeoutError:
                if batch:
                    break
                # No items yet; yield control
                return []

        return batch

    async def _dispatch(self, batch: List[_Request]) -> None:
        """Process a batch under the worker semaphore."""
        assert self._semaphore is not None
        start = time.monotonic()
        async with self._semaphore:
            items = [r.item for r in batch]
            try:
                results = await self._handler(items)
                if len(results) != len(batch):
                    raise ValueError(
                        f"Handler returned {len(results)} results for {len(batch)} items"
                    )
                for req, result in zip(batch, results):
                    if not req.future.done():
                        req.future.set_result(result)
            except Exception as exc:
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(exc)
                self._stats["total_errors"] += len(batch)
            finally:
                for _ in batch:
                    self._queue.task_done()

        elapsed = time.monotonic() - start
        self._latencies.append(elapsed)
        if len(self._latencies) > 10_000:
            self._latencies = self._latencies[-5_000:]

        n = len(batch)
        total_b = self._stats["total_batches"] + 1
        self._stats["total_requests"] += n
        self._stats["total_batches"] = total_b
        prev_avg = self._stats["avg_batch_size"]
        self._stats["avg_batch_size"] = prev_avg + (n - prev_avg) / total_b
        prev_lat = self._stats["avg_latency_ms"]
        self._stats["avg_latency_ms"] = prev_lat + (elapsed * 1000 - prev_lat) / total_b

        logger.debug("Batch dispatched: size=%d latency=%.1fms", n, elapsed * 1000)
