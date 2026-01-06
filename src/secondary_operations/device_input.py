from typing import Any


class DeviceInput:
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
        print("DeviceInput.run() should not be called during normal operation, frame should be injected into next operations instead.")
        return frame

    def visualize(self, frame: Any) -> Any:
        """Return the frame for visualization.

        Args:
            frame: Input frame.

        Returns:
            The input frame unchanged.
        """
        return frame

    def update_config(self, _: dict) -> None:
        """Update the configuration (no parameters to update).

        Args:
            json_config: Configuration (ignored).
        """
        pass