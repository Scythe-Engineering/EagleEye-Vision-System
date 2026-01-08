from typing import Any
from src.secondary_operations.base_class import SecondaryOperation


class DeviceInput(SecondaryOperation):
    """Device input operation that provides the initial device frame."""

    def __init__(self) -> None:
        """Initialize the device input operation."""

    def run(self, frame: Any) -> Any:
        """Return the input frame unchanged. Should not be used, but if it is do not error.

        Args:
            frame: Input camera frame.

        Returns:
            The input frame.
        """
        print(
            "DeviceInput.run() should not be called during normal operation, frame should be injected into next operations instead."
        )
        return frame
