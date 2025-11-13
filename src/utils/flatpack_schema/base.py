from abc import ABC, abstractmethod
from typing import Any


class FlatpackSchema(ABC):
    """Base interface for Flatpack schemas."""

    schema_name: str

    @abstractmethod
    def can_handle(self, value: Any) -> bool:
        """Return True when the schema can serialize the provided value."""

    @abstractmethod
    def serialize(self, value: Any) -> bytes:
        """Serialize the value into bytes using the schema format."""
