"""
Auto-Reindex Service with Dynamic Quantization (Phase 3 & Phase 8).

Monitors index health, memory pressure, and pending changes to automatically:
  1. Rebuild indexes when enough changes accumulate (original behavior)
  2. Auto-apply dynamic quantization under memory pressure (Phase 3)
  3. Trigger index health checks and auto-tuning (Phase 8)

The QuantizationMonitor thread runs in the background and periodically checks
system memory; when free memory drops below configured thresholds, it
recommends a precision tier and can swap live indexes to Int8/PQ/Binary.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class IndexStatus:
    collection_id: str
    vector_count: int
    last_rebuild: float
    pending_changes: int
    current_precision: str = "fp32"
    last_quantization_check: float = 0.0


@dataclass
class QuantizationMonitorConfig:
    """Configuration for the background quantization monitor."""

    check_interval_seconds: float = 30.0
    int8_threshold: float = 0.30  # free memory fraction → downgrade to Int8
    pq_threshold: float = 0.15
    binary_threshold: float = 0.07
    memory_budget_mb: Optional[float] = None
    cooldown_seconds: float = 300.0  # minimum time between quantization changes


class QuantizationMonitor:
    """Background monitor that auto-applies dynamic quantization under memory pressure.

    Runs as a daemon thread, periodically checking system memory and swapping
    live indexes to a lower precision tier when free memory is tight.
    """

    def __init__(
        self,
        config: QuantizationMonitorConfig,
        get_indexes_fn: Callable[[], Dict[str, Any]],
        apply_quantization_fn: Optional[Callable[[str, str], Any]] = None,
    ):
        self.config = config
        self.get_indexes_fn = get_indexes_fn
        self.apply_quantization_fn = apply_quantization_fn
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_swap: Dict[str, float] = {}

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="quantization-monitor",
        )
        self._thread.start()
        logger.info(
            "QuantizationMonitor started (check_interval=%ss)",
            self.config.check_interval_seconds,
        )

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def _get_free_memory_fraction(self) -> Optional[float]:
        """Get fraction of system memory that is free."""
        try:
            import psutil
            vm = psutil.virtual_memory()
            return vm.available / vm.total
        except ImportError:
            return None
        except Exception:
            return None

    def _should_downgrade(self, free_fraction: float) -> Optional[str]:
        """Check if we should downgrade precision based on free memory.

        Returns the target precision (int8, pq, binary) or None.
        """
        cfg = self.config
        if free_fraction < cfg.binary_threshold:
            return "binary"
        if free_fraction < cfg.pq_threshold:
            return "pq"
        if free_fraction < cfg.int8_threshold:
            return "int8"
        return None

    def _monitor_loop(self):
        while self._running:
            try:
                free = self._get_free_memory_fraction()
                if free is not None:
                    target = self._should_downgrade(free)
                    if target:
                        indexes = self.get_indexes_fn()
                        for cid, index_info in indexes.items():
                            cooldown_ok = (
                                self._last_swap.get(cid, 0.0)
                                + self.config.cooldown_seconds
                                < time.time()
                            )
                            if not cooldown_ok:
                                continue

                            current = index_info.get("current_precision", "fp32")
                            priority = {"fp32": 0, "int8": 1, "pq": 2, "binary": 3}
                            if priority.get(target, 0) > priority.get(current, 0):
                                logger.info(
                                    "QuantizationMonitor: downgrading %s from %s to %s "
                                    "(free mem %.1f%%)",
                                    cid, current, target, free * 100,
                                )
                                if self.apply_quantization_fn:
                                    try:
                                        self.apply_quantization_fn(cid, target)
                                        self._last_swap[cid] = time.time()
                                    except Exception as exc:
                                        logger.error(
                                            "Quantization apply failed for %s: %s",
                                            cid, exc,
                                        )
                time.sleep(self.config.check_interval_seconds)
            except Exception as exc:
                logger.warning("QuantizationMonitor error: %s", exc)
                time.sleep(self.config.check_interval_seconds)


class AutoReindexService:
    """Auto-reindex service with integrated quantization monitoring (Phase 3 + Phase 8)."""

    def __init__(
        self,
        threshold: int = 1000,
        cooldown: int = 300,
        enable_quantization_monitor: bool = False,
    ):
        self.threshold = threshold
        self.cooldown = cooldown
        self._statuses: Dict[str, IndexStatus] = {}
        self._quantization_monitor: Optional[QuantizationMonitor] = None
        self._enable_quantization = enable_quantization_monitor

    # ---- Index management --------------------------------------------------

    def register_collection(
        self,
        collection_id: str,
        vector_count: int = 0,
        precision: str = "fp32",
    ):
        if collection_id not in self._statuses:
            self._statuses[collection_id] = IndexStatus(
                collection_id=collection_id,
                vector_count=vector_count,
                last_rebuild=0.0,
                pending_changes=0,
                current_precision=precision,
            )

    def record_change(self, collection_id: str, delta: int = 1):
        if collection_id not in self._statuses:
            self.register_collection(collection_id)
        self._statuses[collection_id].pending_changes += delta
        self._statuses[collection_id].vector_count += delta
        self._maybe_schedule_rebuild(collection_id)

    def _maybe_schedule_rebuild(self, collection_id: str):
        status = self._statuses[collection_id]
        if (status.pending_changes >= self.threshold and
                time.time() - status.last_rebuild >= self.cooldown):
            self._schedule_rebuild(collection_id)

    def _schedule_rebuild(self, collection_id: str):
        logger.info("Scheduling index rebuild for %s", collection_id)
        status = self._statuses[collection_id]
        status.pending_changes = 0
        status.last_rebuild = time.time()

    def record_precision_change(self, collection_id: str, new_precision: str):
        """Record that an index was switched to a different precision tier."""
        status = self._statuses.get(collection_id)
        if status:
            status.current_precision = new_precision
            status.last_quantization_check = time.time()

    # ---- Quantization monitor ----------------------------------------------

    def start_quantization_monitor(
        self,
        config: Optional[QuantizationMonitorConfig] = None,
        apply_quantization_fn: Optional[Callable[[str, str], Any]] = None,
    ):
        """Start the background quantization pressure monitor.

        Args:
            config: Quantization monitor settings.
            apply_quantization_fn: Callable (collection_id, target_precision) that
                applies the precision switch to the live index.
        """
        if not self._enable_quantization:
            return

        cfg = config or QuantizationMonitorConfig()

        def get_indexes():
            return {
                cid: {
                    "vector_count": s.vector_count,
                    "current_precision": s.current_precision,
                }
                for cid, s in self._statuses.items()
            }

        self._quantization_monitor = QuantizationMonitor(
            config=cfg,
            get_indexes_fn=get_indexes,
            apply_quantization_fn=apply_quantization_fn,
        )
        self._quantization_monitor.start()

    def stop_quantization_monitor(self):
        if self._quantization_monitor:
            self._quantization_monitor.stop()

    # ---- Health & status ---------------------------------------------------

    def check_index_health(self, index) -> Dict[str, Any]:
        """Check index health and recommend actions."""
        from utils.index_health import IndexHealthChecker
        health = IndexHealthChecker.check_health(index)
        recommendations = []

        if health.get("status") == "degraded":
            if health.get("distribution", {}).get("under_connected_percent", 0) > 10:
                recommendations.append("Rebuild with higher M parameter")
            if health.get("connectivity", {}).get("unreachable_nodes", 0) > 0:
                recommendations.append("Compact index to remove orphaned nodes")

        health["recommendations"] = recommendations
        return health

    def get_status(self, collection_id: str) -> Dict:
        status = self._statuses.get(collection_id)
        if not status:
            return {}
        return {
            "collection_id": status.collection_id,
            "vector_count": status.vector_count,
            "pending_changes": status.pending_changes,
            "needs_rebuild": status.pending_changes >= self.threshold,
            "last_rebuild": status.last_rebuild,
            "current_precision": status.current_precision,
        }

    def get_all_statuses(self) -> Dict[str, Dict]:
        return {
            cid: self.get_status(cid)
            for cid in self._statuses
        }

    def get_quantization_status(self) -> Dict[str, Any]:
        if not self._quantization_monitor:
            return {"enabled": False}
        return {
            "enabled": True,
            "running": self._quantization_monitor._running if self._quantization_monitor else False,
        }
