"""Small lifecycle contract for operations docked to asynchronous sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar

from src.main_operations.definitions.base.base_class import OperationInstance


AsyncResult = TypeVar("AsyncResult")


class AsyncDockedOperation(OperationInstance, ABC, Generic[AsyncResult]):
    """Operation whose backend work is fed independently from graph execution."""

    dock_source_action: str
    dock_source_port: str
    dock_target_port: str

    @abstractmethod
    def bind(
        self,
        source: OperationInstance,
        should_remain_active: Callable[[], bool],
    ) -> None:
        """Bind the operation to its direct source before runtimes start."""

    @abstractmethod
    def activate(self) -> None:
        """Resume asynchronous input feeding."""

    @abstractmethod
    def wait_for_next(self) -> AsyncResult | None:
        """Wait for and return the newest completed result."""

    @abstractmethod
    def deactivate(self) -> None:
        """Pause input feeding and wake graph waiters."""

    @property
    @abstractmethod
    def terminal_error(self) -> BaseException | None:
        """Return a non-recoverable backend failure, if one occurred."""

    @abstractmethod
    def close(self) -> None:
        """Permanently close this binding and wake all waiters."""
