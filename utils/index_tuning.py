"""
Index tuning — AI-powered parameter recommendations based on dataset analysis.

Provides data-driven HNSW parameter suggestions (M, ef_construction, ef_search)
by analyzing the statistical properties of a vector dataset (dimensionality,
variance, intrinsic dimensionality, cluster structure, norm distribution).

This is a Phase 10 feature (AI-Native & Intelligent Search).
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DatasetStatistics:
    """Analysed statistics for a vector dataset."""

    num_vectors: int
    dimension: int

    # Per-dimension statistics
    dim_variances: np.ndarray          # shape (dim,)
    dim_means: np.ndarray              # shape (dim,)
    mean_variance: float
    variance_coefficient: float        # CV of per-dim variances (low → uniform)

    # Norm statistics (for cosine distance)
    norm_mean: float
    norm_std: float
    norm_min: float
    norm_max: float

    # Intrinsic dimensionality estimates
    intrinsic_dim_twonn: float         # Two-NN estimator
    intrinsic_dim_pca: float           # Participation-ratio estimator

    # Cluster structure
    avg_nn_distance: float             # Average distance to nearest neighbour
    avg_far_distance: float            # Average distance to farthest point in sample
    cluster_separation_ratio: float    # far / nn — high → well-separated clusters
    estimated_cluster_count: int       # Rough heuristic

    # Density
    density_estimate: float            # Points per unit volume (heuristic)

    @property
    def is_normalized(self) -> bool:
        """Whether vectors appear L2-normalised (for cosine distance)."""
        return abs(self.norm_mean - 1.0) < 0.05 and self.norm_std < 0.05

    @property
    def is_high_dimensional(self) -> bool:
        """Heuristic: dim > 50 or intrinsic dim > 30."""
        return self.dimension > 50 or self.intrinsic_dim_pca > 30

    @property
    def summary(self) -> Dict[str, object]:
        return {
            "num_vectors": self.num_vectors,
            "dimension": self.dimension,
            "mean_variance": round(float(self.mean_variance), 4),
            "variance_coefficient": round(float(self.variance_coefficient), 4),
            "norm_mean": round(float(self.norm_mean), 4),
            "is_normalized": self.is_normalized,
            "intrinsic_dim_twonn": round(float(self.intrinsic_dim_twonn), 2),
            "intrinsic_dim_pca": round(float(self.intrinsic_dim_pca), 2),
            "cluster_separation_ratio": round(float(self.cluster_separation_ratio), 2),
            "estimated_cluster_count": self.estimated_cluster_count,
            "is_high_dimensional": self.is_high_dimensional,
            "density_estimate": round(float(self.density_estimate), 6),
        }


@dataclass
class HNSWParameterRecommendation:
    """Tuning recommendation for HNSW index parameters."""

    m: int
    m0: int
    ef_construction: int
    ef_search: int           # recommended default; increase for higher recall
    expected_recall: float   # estimated recall@10 at the recommended ef_search
    confidence: str          # "high" | "medium" | "low"
    reasoning: List[str]     # human-readable bullet points

    def to_dict(self) -> Dict[str, object]:
        return {
            "hnsw": {
                "m": self.m,
                "m0": self.m0,
                "ef_construction": self.ef_construction,
                "ef_search": self.ef_search,
            },
            "expected_recall": round(self.expected_recall, 3),
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


# ---------------------------------------------------------------------------
# DatasetAnalyser
# ---------------------------------------------------------------------------

class DatasetAnalyzer:
    """
    Analyse a set of vectors and produce DatasetStatistics.

    Uses random sub-sampling for expensive statistics so it scales to
    large datasets without O(N²) cost.
    """

    def __init__(
        self,
        sample_size: int = 1000,
        random_seed: int = 42,
    ):
        """
        Args:
            sample_size: Number of random vectors to use for pairwise
                distance statistics.  Pass -1 to use the full dataset.
            random_seed: Seed for reproducible sub-sampling.
        """
        self.sample_size = sample_size
        self.random_seed = random_seed

    def analyze(self, vectors: np.ndarray) -> DatasetStatistics:
        """
        Produce DatasetStatistics from a vector array.

        Args:
            vectors: Shape (N, D) float32/float64 array.

        Returns:
            DatasetStatistics with all computed fields.
        """
        rng = np.random.default_rng(self.random_seed)
        n, d = vectors.shape

        # --- 1. Basic per-dimension statistics ---
        dim_means = np.mean(vectors, axis=0)
        dim_variances = np.var(vectors, axis=0)
        mean_variance = float(np.mean(dim_variances))
        # Coefficient of variation of per-dim variances
        var_std = float(np.std(dim_variances))
        variance_coefficient = var_std / mean_variance if mean_variance > 1e-12 else 0.0

        # --- 2. Norm statistics ---
        norms = np.linalg.norm(vectors, axis=1)
        norm_mean = float(np.mean(norms))
        norm_std = float(np.std(norms))
        norm_min = float(np.min(norms))
        norm_max = float(np.max(norms))

        # --- 3. Intrinsic dimensionality via Two-NN ---
        sample = self._subsample(vectors, rng)
        intrinsic_twonn = self._estimate_intrinsic_dim_twonn(sample, rng)

        # --- 4. Intrinsic dimensionality via PCA participation ratio ---
        intrinsic_pca = self._estimate_intrinsic_dim_pca(sample)

        # --- 5. Cluster structure ---
        nn_dist, far_dist = self._estimate_cluster_structure(sample, rng)
        cluster_sep = far_dist / nn_dist if nn_dist > 1e-12 else 1.0
        # Rough cluster count heuristic
        est_clusters = max(1, min(int(cluster_sep * 2), n // 10))

        # --- 6. Density ---
        # Average nearest-neighbour distance as a proxy: higher = sparser
        volume = (nn_dist ** d) if d < 100 else (nn_dist ** 100)  # avoid overflow
        density = (n / (volume + 1e-12)) if volume < 1e100 else 0.0

        return DatasetStatistics(
            num_vectors=n,
            dimension=d,
            dim_variances=dim_variances,
            dim_means=dim_means,
            mean_variance=mean_variance,
            variance_coefficient=variance_coefficient,
            norm_mean=norm_mean,
            norm_std=norm_std,
            norm_min=norm_min,
            norm_max=norm_max,
            intrinsic_dim_twonn=intrinsic_twonn,
            intrinsic_dim_pca=intrinsic_pca,
            avg_nn_distance=float(nn_dist),
            avg_far_distance=float(far_dist),
            cluster_separation_ratio=float(cluster_sep),
            estimated_cluster_count=est_clusters,
            density_estimate=float(density),
        )

    # -- internal helpers ------------------------------------------------

    def _subsample(
        self, vectors: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Return a random subset (deterministic seed) for pairwise stats."""
        n = len(vectors)
        if self.sample_size <= 0 or self.sample_size >= n:
            return vectors
        indices = rng.choice(n, size=self.sample_size, replace=False)
        return vectors[indices]

    @staticmethod
    def _estimate_intrinsic_dim_twonn(
        sample: np.ndarray, rng: np.random.Generator
    ) -> float:
        """
        Two-Nearest-Neighbour estimator of intrinsic dimensionality
        (Facco et al., 2017).

        Uses a random subset of the sample to keep O(k·n) cost reasonable.
        """
        n = len(sample)
        if n < 10:
            return float(sample.shape[1])  # fallback to ambient dim

        # Pick a random reference subset (up to 500 points)
        ref_n = min(n, 500)
        ref_indices = rng.choice(n, size=ref_n, replace=False)
        queries = sample[ref_indices]
        rest = np.delete(sample, ref_indices, axis=0)

        ratios: List[float] = []
        for q in queries:
            dists = np.linalg.norm(rest - q, axis=1)
            if len(dists) < 2:
                continue
            dists.sort()
            r1, r2 = dists[0], dists[1]
            if r1 < 1e-12:
                continue
            ratios.append(float(r2 / r1))

        if len(ratios) < 2:
            return float(sample.shape[1])

        # Intrinsic dim = -mean(log(1/ratio))⁻¹
        log_ratios = np.log(ratios)
        mu = float(np.mean(log_ratios))
        if mu < 1e-12:
            return float(sample.shape[1])
        return float(-1.0 / mu)

    @staticmethod
    def _estimate_intrinsic_dim_pca(sample: np.ndarray) -> float:
        """
        Participation-ratio estimator: (Σ λᵢ)² / Σ λᵢ²   where λ are
        the eigenvalues of the covariance matrix.

        Gives an effective number of dimensions that capture most variance.
        """
        if len(sample) < 2:
            return float(sample.shape[1])

        centered = sample - np.mean(sample, axis=0)
        _, eigenvalues, _ = np.linalg.svd(centered, full_matrices=False)
        ev = eigenvalues ** 2  # singular values → eigenvalues
        total = np.sum(ev)
        if total < 1e-12:
            return 1.0
        return float(total ** 2 / np.sum(ev ** 2))

    @staticmethod
    def _estimate_cluster_structure(
        sample: np.ndarray, rng: np.random.Generator
    ) -> Tuple[float, float]:
        """
        Estimate average nearest-neighbour distance and average distance to
        a *far* point (90th percentile of pairwise distances for each query).

        Returns (avg_nn_dist, avg_far_dist).
        """
        n = len(sample)
        if n < 4:
            return 0.5, 1.0

        # Use a random subset of queries for pairwise stats
        nq = min(n, 300)
        q_idx = rng.choice(n, size=nq, replace=False)
        queries = sample[q_idx]
        rest = np.delete(sample, q_idx, axis=0)

        nn_dists: List[float] = []
        far_dists: List[float] = []

        for q in queries:
            dists = np.linalg.norm(rest - q, axis=1)
            if len(dists) < 2:
                continue
            dists.sort()
            nn_dists.append(float(dists[0]))
            # Far = 90th percentile distance
            far_dists.append(float(dists[max(1, int(len(dists) * 0.9))]))

        avg_nn = float(np.mean(nn_dists)) if nn_dists else 0.5
        avg_far = float(np.mean(far_dists)) if far_dists else 1.0
        return max(avg_nn, 1e-12), max(avg_far, avg_nn + 1e-12)


# ---------------------------------------------------------------------------
# HNSWParameterRecommender
# ---------------------------------------------------------------------------

class HNSWParameterRecommender:
    """
    Recommend HNSW index parameters based on dataset statistics.

    Uses heuristic decision rules derived from:
      - The original HNSW paper (Malkov & Yashunin, 2016)
      - Empirical benchmarks on the project's own dataset
      - Common practice in Milvus, FAISS, and Qdrant
    """

    def __init__(self, recall_target: float = 0.95):
        """
        Args:
            recall_target: Desired recall@10 (0.0 – 1.0).  Higher targets
                increase ef_construction and ef_search recommendations.
        """
        self.recall_target = min(max(recall_target, 0.7), 0.999)

    def recommend(
        self, stats: DatasetStatistics
    ) -> HNSWParameterRecommendation:
        """
        Produce a parameter recommendation from dataset statistics.

        Args:
            stats: DatasetStatistics from DatasetAnalyzer.

        Returns:
            HNSWParameterRecommendation with tuned parameters and reasoning.
        """
        reasoning: List[str] = []
        n = stats.num_vectors
        d = stats.dimension
        idim = stats.intrinsic_dim_pca  # use PCA estimate as primary

        # --- Step 1: Recommend M ----------------------------------------
        # Base M on intrinsic dimensionality + dataset size
        if idim < 10:
            base_m = 8
        elif idim < 25:
            base_m = 12
        elif idim < 50:
            base_m = 16
        elif idim < 80:
            base_m = 24
        else:
            base_m = 32

        # Scale up with dataset size for connectivity
        if n > 1_000_000:
            size_boost = 2
        elif n > 100_000:
            size_boost = 1
        else:
            size_boost = 0

        m = min(base_m + size_boost * 4, 64)

        # Lower M when data is very uniform (low variance CV) — fewer
        # distinct directions to connect
        if stats.variance_coefficient < 0.15 and m > 8:
            m = max(m - 4, 8)
            reasoning.append(
                f"Low variance coefficient ({stats.variance_coefficient:.2f}) "
                f"suggests uniform data; reduced M to {m}."
            )

        m0 = 2 * m

        reasoning.append(
            f"Intrinsic dimensionality ≈ {idim:.0f} (ambient {d}) "
            f"with {n:,} vectors → base M={m}."
        )

        # --- Step 2: Recommend ef_construction --------------------------
        # Higher for: larger datasets, higher intrinsic dim, higher recall target,
        # poorly separated clusters (need more beam width to cross clusters)

        ef_construction = 100

        # Scale by dataset size
        if n > 500_000:
            ef_construction += 200
        elif n > 50_000:
            ef_construction += 100
        elif n > 5_000:
            ef_construction += 50

        # Scale by intrinsic dimensionality
        if idim > 80:
            ef_construction += 100
        elif idim > 50:
            ef_construction += 50

        # Scale by recall target
        if self.recall_target >= 0.99:
            ef_construction += 100
        elif self.recall_target >= 0.97:
            ef_construction += 50

        # Boost when clusters are poorly separated (high inter-cluster gap
        # means the search beam needs to be wider to jump between modes)
        if stats.cluster_separation_ratio > 5.0:
            ef_construction += 50
            reasoning.append(
                f"High cluster separation ({stats.cluster_separation_ratio:.1f}×) "
                f"— increased ef_construction to {ef_construction} for inter-cluster reach."
            )

        # Boost when vectors are NOT pre-normalised but using cosine —
        # the effective distance structure is harder to navigate
        if not stats.is_normalized and stats.norm_std > 0.1:
            ef_construction += 50
            reasoning.append(
                f"Vectors are not L2-normalised (norm σ={stats.norm_std:.2f}) "
                f"— increased ef_construction to {ef_construction}."
            )

        ef_construction = min(ef_construction, 800)

        reasoning.append(
            f"Dataset size {n:,}, {self.recall_target:.0%} recall target "
            f"→ ef_construction={ef_construction}."
        )

        # --- Step 3: Recommend ef_search --------------------------------
        if self.recall_target >= 0.99:
            ef_search = 200
        elif self.recall_target >= 0.97:
            ef_search = 100
        elif self.recall_target >= 0.95:
            ef_search = 50
        elif self.recall_target >= 0.90:
            ef_search = 30
        else:
            ef_search = 20

        reasoning.append(
            f"Target recall {self.recall_target:.0%} → ef_search={ef_search}."
        )

        # --- Step 4: Expected recall & confidence -----------------------
        # Confidence is based on sample size
        if n < 100:
            confidence = "low"
        elif n < 1000 or (
            stats.intrinsic_dim_twonn > 100 and stats.dimension > 200
        ):
            confidence = "medium"
        else:
            confidence = "high"

        # Expected recall — a rough estimate based on M and ef
        # Higher M + ef → higher expected recall at given ef_search
        recall_estimate = min(
            self.recall_target,
            0.5
            + 0.3 * min(1.0, m / 24)
            + 0.2 * min(1.0, ef_construction / 300),
        )

        return HNSWParameterRecommendation(
            m=m,
            m0=m0,
            ef_construction=ef_construction,
            ef_search=ef_search,
            expected_recall=recall_estimate,
            confidence=confidence,
            reasoning=reasoning,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def tune_hnsw(
    vectors: np.ndarray,
    recall_target: float = 0.95,
    sample_size: int = 1000,
) -> HNSWParameterRecommendation:
    """
    One-shot convenience: analyse vectors and get HNSW parameter
    recommendations.

    Args:
        vectors: Shape (N, D) array of vectors.
        recall_target: Desired recall@10.
        sample_size: Sub-sample size for pairwise distance stats.

    Returns:
        HNSWParameterRecommendation.
    """
    analyzer = DatasetAnalyzer(sample_size=sample_size)
    recommender = HNSWParameterRecommender(recall_target=recall_target)
    stats = analyzer.analyze(vectors)
    logger.info("Dataset analysis complete:\n%s", _format_stats(stats))
    return recommender.recommend(stats)


def _format_stats(stats: DatasetStatistics) -> str:
    """Format statistics as a human-readable block."""
    lines = [
        f"  Vectors:         {stats.num_vectors:,} × {stats.dimension}D",
        f"  Mean variance:   {stats.mean_variance:.4f}  (CV={stats.variance_coefficient:.2f})",
        f"  Norm:            μ={stats.norm_mean:.3f}  σ={stats.norm_std:.3f}  "
        f"{'(L2-normalised)' if stats.is_normalized else ''}",
        f"  Intrinsic dim:   twonn={stats.intrinsic_dim_twonn:.1f}  "
        f"pca={stats.intrinsic_dim_pca:.1f}",
        f"  Cluster sep:     {stats.cluster_separation_ratio:.2f}×  "
        f"(~{stats.estimated_cluster_count} clusters)",
        f"  Density:         {stats.density_estimate:.6f}",
    ]
    return "\n".join(lines)
