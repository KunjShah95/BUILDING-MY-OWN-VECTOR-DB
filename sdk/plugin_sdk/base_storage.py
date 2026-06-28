"""Base class for storage backend plugins."""
from abc import ABC, abstractmethod
from typing import List


class BaseStoragePlugin(ABC):
    """Abstract base for storage backend plugins."""

    @abstractmethod
    def put(self, key: str, data: bytes) -> None:
        """Store bytes under the given key."""

    @abstractmethod
    def get(self, key: str) -> bytes:
        """Retrieve bytes by key. Raises KeyError if not found."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if it existed."""

    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """Return all keys matching the given prefix."""
