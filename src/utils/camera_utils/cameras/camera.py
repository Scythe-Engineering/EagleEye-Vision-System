import abc
from typing import Callable, Optional

import numpy as np

from src.utils.colors import Colors


class Camera(abc.ABC):
    """Abstract base class defining a common camera interface for frame transport only.

    This class handles only frame acquisition - all calibration (intrinsics/extrinsics)
    and rotation concerns are owned by operations, not camera classes.
    """

    def __init__(
        self,
        camera_name: str,
        log: Callable[[str], None] = print,
    ) -> None:
        """Initialize camera with identity and prepare for frame transport.

        Args:
            camera_name: Name of the camera for identification.
            log: Logging function, e.g. `print` or logger. Defaults to `print`.
        """
        self.name: str = camera_name
        self.log = log
        self.camera_ready: bool = False
        self.cap = None

        self.log(
            f"{Colors.CYAN}Camera: {self.name} initialized (calibration-agnostic){Colors.RESET}"
        )

        self._start_camera()

    @abc.abstractmethod
    def _start_camera(self) -> None:
        """Open or start whatever backend is needed for this camera."""
        pass

    @abc.abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """Retrieve a raw frame without any rotation or transformation.

        Returns:
            The latest frame, or None on failure/end-of-stream.
        """
        pass

    @abc.abstractmethod
    def get_achieved_fps(self) -> int:
        """Get the FPS that the camera is operating at.

        Returns:
            Achieved FPS value.
        """
        pass

    def get_name(self) -> str:
        """Returns the human-readable name of this camera."""
        return self.name
