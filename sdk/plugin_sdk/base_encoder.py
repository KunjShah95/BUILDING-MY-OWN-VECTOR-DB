"""Base class for encoder plugins."""
from abc import ABC, abstractmethod
from typing import List


class BaseEncoderPlugin(ABC):
    """Abstract base for text-to-vector encoder plugins."""

    @abstractmethod
    def encode(self, text: str) -> List[float]:
        """Encode a single text into a vector."""

    @abstractmethod
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts into vectors."""

    @abstractmethod
    def dimension(self) -> int:
        """Return the output vector dimension."""
