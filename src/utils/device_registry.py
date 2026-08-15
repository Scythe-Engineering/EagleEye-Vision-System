"""Immutable inventory of inference devices discovered at application startup."""

from __future__ import annotations

import glob
import importlib
import os
import platform
from dataclasses import dataclass
from typing import Any, Iterable


class DeviceRegistryError(RuntimeError):
    """Base error raised by the device registry."""


class DeviceNotFoundError(DeviceRegistryError, KeyError):
    """Raised when a canonical device ID is not in the inventory."""


@dataclass(frozen=True, slots=True)
class DeviceDescriptor:
    """A stable description of one physical inference device."""

    device_id: str
    display_name: str
    device_type: str
    physical_index: int | None


def _get_cpu_display_name() -> str:
    """Return a best-effort CPU hardware label without affecting inventory."""
    processor_name = platform.processor().strip()
    return processor_name or "CPU"


class DeviceRegistry:
    """A fixed device inventory; discovery is performed exactly once."""

    def __init__(self, devices: Iterable[DeviceDescriptor]) -> None:
        """Initialize the immutable inventory.

        Args:
            devices: Device descriptors to index by canonical ID.

        Raises:
            DeviceRegistryError: If canonical device IDs are duplicated.
        """
        entries = tuple(devices)
        by_id = {entry.device_id: entry for entry in entries}
        if len(by_id) != len(entries):
            raise DeviceRegistryError("duplicate canonical device ID")
        self._entries = entries
        self._by_id = by_id

    @classmethod
    def discover(
        cls,
        logger: Any = None,
        *,
        cuda_devices: Iterable[str] | None = None,
        mx3_paths: Iterable[str] | None = None,
    ) -> "DeviceRegistry":
        """Discover CPU, CUDA, and Linux ``/dev/memxN`` devices.

        Optional device values are injection points for deterministic tests.  Torch is
        imported lazily here and is never imported while this module is loaded.
        """
        devices = [
            DeviceDescriptor(
                device_id="cpu",
                display_name=_get_cpu_display_name(),
                device_type="cpu",
                physical_index=None,
            )
        ]

        if cuda_devices is None:
            cuda_names: list[str] = []
            try:
                torch = importlib.import_module("torch")
                if torch.cuda.is_available():
                    cuda_names = [
                        torch.cuda.get_device_name(index)
                        for index in range(torch.cuda.device_count())
                    ]
            except (ImportError, RuntimeError, OSError) as error:
                if logger is not None:
                    log = getattr(logger, "log", None)
                    if callable(log):
                        log(f"CUDA discovery failed: {error}")
        else:
            cuda_names = list(cuda_devices)

        devices.extend(
            DeviceDescriptor(f"cuda:{index}", str(name), "cuda", index)
            for index, name in enumerate(cuda_names)
        )

        if mx3_paths is not None:
            paths = list(mx3_paths)
        elif os.name == "posix":
            paths = glob.glob("/dev/memx[0-9]*")
        else:
            paths = []

        indexed_paths: list[tuple[int, str]] = []
        for path in paths:
            suffix = os.path.basename(path).removeprefix("memx")
            if suffix.isdigit():
                indexed_paths.append((int(suffix), path))

        for index, path in sorted(indexed_paths):
            devices.append(
                DeviceDescriptor(
                    device_id=f"mx3:{index}",
                    display_name=f"MemryX MX3 ({path})",
                    device_type="mx3",
                    physical_index=index,
                )
            )
        return cls(devices)

    def descriptors(self) -> tuple[DeviceDescriptor, ...]:
        """Return the immutable startup inventory in deterministic order."""
        return self._entries

    def get(self, device_id: str) -> DeviceDescriptor:
        """Return a descriptor by exact canonical ID (aliases are not accepted).

        Args:
            device_id: Canonical device ID such as ``cpu``, ``cuda:0``, or ``mx3:0``.

        Returns:
            The matching startup descriptor.

        Raises:
            DeviceNotFoundError: If the ID is not part of the startup inventory.
        """
        try:
            return self._by_id[device_id]
        except KeyError as error:
            raise DeviceNotFoundError(f"unknown device ID: {device_id!r}") from error
