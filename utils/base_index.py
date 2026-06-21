"""
Abstract base class for ANN index implementations.
Merged from C:\\ann search engine on 2026-06-21.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class BaseIndex(ABC):
    """Common interface for all vector index implementations."""

    def __init__(self, dimension: int, metric: str = "cosine", **kwargs):
        self.dimension = dimension
        self.metric = metric
        self._is_trained = False
        self._vector_count = 0

    @property
    @abstractmethod
    def index_type(self) -> str:
        pass

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        pass

    @property
    @abstractmethod
    def vector_count(self) -> int:
        pass

    @abstractmethod
    def add(self, vectors: List[List[float]], vector_ids: Optional[List[str]] = None) -> None:
        pass

    @abstractmethod
    def search(self, query: List[float], k: int = 10) -> List[Tuple[str, float]]:
        pass

    @abstractmethod
    def delete(self, vector_id: str) -> bool:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass

    def _validate_vector(self, vector: List[float]) -> np.ndarray:
        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension {len(vector)} != index dimension {self.dimension}"
            )
        return np.array(vector, dtype=np.float32)

    def _default_vector_ids(self, count: int, start: int = 0) -> List[str]:
        return [f"vec_{i}" for i in range(start, start + count)]
