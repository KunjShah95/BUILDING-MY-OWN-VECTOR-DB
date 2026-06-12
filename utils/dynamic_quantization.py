"""
Dynamic quantization policy (ROADMAP Phase 3).

Chooses a vector precision tier based on dataset size and system memory
pressure, auto-downgrading fp32 → Int8 → PQ → Binary as memory tightens. The
policy is advisory by default (``recommend``) and can materialize the chosen
representation (``apply``) for an in-memory vector set.

Tiers and their per-vector cost (D = dimension):
    fp32   : 4 * D bytes   (full precision, best recall)
    int8   : 1 * D bytes   (4x smaller, ~minimal recall loss)
    pq     : M bytes       (M sub-quantizers; 10-48x smaller)
    binary : D / 8 bytes   (32x smaller, coarse recall)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class Precision(str, Enum):
    FP32 = "fp32"
    INT8 = "int8"
    PQ = "pq"
    BINARY = "binary"


def _bytes_per_vector(precision: Precision, dim: int, pq_m: int = 16) -> int:
    if precision == Precision.FP32:
        return 4 * dim
    if precision == Precision.INT8:
        return dim
    if precision == Precision.PQ:
        return pq_m
    if precision == Precision.BINARY:
        return max(1, dim // 8)
    raise ValueError(precision)


def _available_memory_fraction() -> Optional[float]:
    """Return fraction of system memory still free, or None if psutil absent."""
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return vm.available / vm.total
    except Exception:
        return None


@dataclass
class QuantizationPolicy:
    """
    Decide a precision tier from dataset shape + memory pressure.

    Args:
        memory_budget_mb: soft cap on how much RAM the vector set may use.
            When set, the policy picks the highest-precision tier that fits.
        pressure_thresholds: free-memory fractions that trigger downgrades.
            Defaults: <0.30 free -> int8, <0.15 -> pq, <0.07 -> binary.
        pq_m: number of PQ sub-quantizers (controls PQ byte cost).
    """
    memory_budget_mb: Optional[float] = None
    pressure_thresholds: Dict[str, float] = None
    pq_m: int = 16

    def __post_init__(self):
        if self.pressure_thresholds is None:
            self.pressure_thresholds = {"int8": 0.30, "pq": 0.15, "binary": 0.07}

    def _fits_budget(self, precision: Precision, n: int, dim: int) -> bool:
        if self.memory_budget_mb is None:
            return True
        mb = _bytes_per_vector(precision, dim, self.pq_m) * n / (1024 * 1024)
        return mb <= self.memory_budget_mb

    def recommend(
        self,
        n_vectors: int,
        dim: int,
        free_memory_fraction: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Recommend a precision tier.

        Args:
            n_vectors: number of vectors in the set.
            dim: vector dimension.
            free_memory_fraction: 0..1 free RAM; auto-detected via psutil if None.
        """
        free = free_memory_fraction
        if free is None:
            free = _available_memory_fraction()

        reasons: List[str] = []

        # 1. Budget-driven selection (pick highest precision that fits).
        chosen = Precision.FP32
        if self.memory_budget_mb is not None:
            for tier in (Precision.FP32, Precision.INT8, Precision.PQ, Precision.BINARY):
                if self._fits_budget(tier, n_vectors, dim):
                    chosen = tier
                    break
            else:
                chosen = Precision.BINARY
            reasons.append(f"budget {self.memory_budget_mb}MB -> {chosen.value}")

        # 2. Memory-pressure-driven downgrade (only ever lowers precision).
        if free is not None:
            t = self.pressure_thresholds
            pressure_tier = Precision.FP32
            if free < t["binary"]:
                pressure_tier = Precision.BINARY
            elif free < t["pq"]:
                pressure_tier = Precision.PQ
            elif free < t["int8"]:
                pressure_tier = Precision.INT8
            if _order(pressure_tier) > _order(chosen):
                chosen = pressure_tier
                reasons.append(f"free RAM {free:.2f} -> downgrade to {chosen.value}")

        if not reasons:
            reasons.append("no pressure / no budget -> keep fp32")

        est_mb = _bytes_per_vector(chosen, dim, self.pq_m) * n_vectors / (1024 * 1024)
        fp32_mb = _bytes_per_vector(Precision.FP32, dim, self.pq_m) * n_vectors / (1024 * 1024)
        return {
            "precision": chosen.value,
            "estimated_mb": round(est_mb, 3),
            "fp32_mb": round(fp32_mb, 3),
            "compression_ratio": round(fp32_mb / est_mb, 2) if est_mb else None,
            "free_memory_fraction": free,
            "reasons": reasons,
        }

    def apply(
        self,
        vectors: List[List[float]],
        precision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Materialize a precision tier for a vector set. Returns the encoded
        representation plus the metadata needed to decode/search it.

        For FP32 the vectors pass through; INT8 does per-dimension min/max
        quantization; BINARY does sign-bit packing; PQ is deferred to the
        existing PQIndex (returns a marker so callers route to it).
        """
        prec = Precision(precision) if precision else Precision.FP32
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("vectors must be a 2D list")

        if prec == Precision.FP32:
            return {"precision": "fp32", "data": arr}

        if prec == Precision.INT8:
            vmin = arr.min(axis=0)
            vmax = arr.max(axis=0)
            scale = np.where(vmax > vmin, (vmax - vmin), 1.0)
            codes = np.round((arr - vmin) / scale * 255).astype(np.uint8)
            return {"precision": "int8", "data": codes, "min": vmin, "scale": scale}

        if prec == Precision.BINARY:
            bits = (arr > 0).astype(np.uint8)
            packed = np.packbits(bits, axis=1)
            return {"precision": "binary", "data": packed, "dim": arr.shape[1]}

        # PQ: signal the caller to use PQIndex (heavy codebook training).
        return {"precision": "pq", "route_to": "PQIndex", "pq_m": self.pq_m}


def _order(p: Precision) -> int:
    """Lower precision -> higher order number (more aggressive compression)."""
    return {Precision.FP32: 0, Precision.INT8: 1, Precision.PQ: 2, Precision.BINARY: 3}[p]
