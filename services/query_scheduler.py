"""Budget-aware query scheduler with token-bucket rate limiting per tenant."""
import threading
import time
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TenantCreditPool:
    tenant_id: str
    credits_per_second: float
    max_burst: float
    current_credits: float = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self):
        self.current_credits = self.max_burst

    def consume(self, cost: float) -> bool:
        with self._lock:
            if self.current_credits >= cost:
                self.current_credits -= cost
                return True
            return False

    def refill(self) -> None:
        with self._lock:
            self.current_credits = min(self.max_burst, self.current_credits + self.credits_per_second * 0.1)


@dataclass(order=True)
class _PrioritizedTask:
    priority: float  # lower = higher priority (negative credits_per_second)
    tenant_id: str = field(compare=False)
    query_fn: Callable = field(compare=False)
    future: Future = field(compare=False)


class QueryScheduler:
    """Priority queue scheduler with per-tenant token-bucket budgets."""

    def __init__(self, max_workers: int = 4):
        self._lock = threading.Lock()
        self._tenants: Dict[str, TenantCreditPool] = {}
        self._queues: Dict[str, list] = {}  # tenant_id -> pending tasks
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._running = True
        self._refill_thread = threading.Thread(target=self._refill_loop, daemon=True)
        self._refill_thread.start()

    def register_tenant(
        self, tenant_id: str, credits_per_second: float = 10.0, max_burst: float = 50.0
    ) -> None:
        with self._lock:
            if tenant_id not in self._tenants:
                self._tenants[tenant_id] = TenantCreditPool(
                    tenant_id=tenant_id,
                    credits_per_second=credits_per_second,
                    max_burst=max_burst,
                )
                self._queues[tenant_id] = []
                logger.info("Registered tenant %s (cps=%.1f, burst=%.1f)", tenant_id, credits_per_second, max_burst)

    def schedule(self, tenant_id: str, query_fn: Callable, cost_estimate: float = 1.0) -> Future:
        future: Future = Future()

        with self._lock:
            if tenant_id not in self._tenants:
                self.register_tenant(tenant_id)
            pool = self._tenants[tenant_id]

        if not pool.consume(cost_estimate):
            future.set_exception(
                RuntimeError(f"Tenant {tenant_id} insufficient credits (need {cost_estimate:.2f})")
            )
            return future

        # Submit directly — budget already consumed
        self._executor.submit(self._run_task, query_fn, future)
        return future

    def get_queue_depth(self, tenant_id: str) -> int:
        with self._lock:
            return len(self._queues.get(tenant_id, []))

    def shutdown(self) -> None:
        self._running = False
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_task(self, query_fn: Callable, future: Future) -> None:
        try:
            result = query_fn()
            future.set_result(result)
        except Exception as exc:
            future.set_exception(exc)

    def _refill_loop(self) -> None:
        while self._running:
            time.sleep(0.1)
            with self._lock:
                pools = list(self._tenants.values())
            for pool in pools:
                pool.refill()


query_scheduler = QueryScheduler()
