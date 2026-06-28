"""Base class for index plugins."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseIndexPlugin(ABC):
    """Abstract base for all index plugins."""

    @abstractmethod
    def train(self, vectors: List[List[float]]) -> None:
        """Train the index on a set of vectors (may be a no-op for exact methods)."""

    @abstractmethod
    def add(self, vector: List[float], id: str, metadata: Dict[str, Any]) -> None:
        """Add a single vector with its ID and metadata."""

    @abstractmethod
    def search(self, query: List[float], k: int) -> List[Dict[str, Any]]:
        """
        Search for k nearest neighbours.
        Returns list of dicts with at least keys: id, score, metadata.
        """

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Remove a vector by ID. Returns True if it existed."""

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return a dict with at least: vector_count, dimension, index_type."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the index to the given file path."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the index from the given file path."""
