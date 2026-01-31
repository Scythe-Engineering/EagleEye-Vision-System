from __future__ import annotations

import threading
import time
import traceback
from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple, Union

import numpy as np

from src.utils.camera_utils.add_system_cameras import add_system_cameras
from src.utils.camera_utils.add_video_file_cameras import add_video_file_cameras
from src.utils.camera_utils.cameras.physical_camera import PhysicalCamera
from src.utils.camera_utils.cameras.video_file_camera import VideoFileCamera
from src.utils.colors import Colors
from src.utils.logging.logger import Logger

if TYPE_CHECKING:
    from src.webui.web_server import EagleEyeInterface


class FailureTracker:
    """Tracks consecutive failures for a camera feed."""

    def __init__(self, max_failures: int = 10) -> None:
        self.count = 0
        self.max_failures = max_failures

    def record_failure(self) -> bool:
        """Record a failure and return True if max failures exceeded."""
        self.count += 1
        return self.count >= self.max_failures

    def reset(self) -> None:
        """Reset the failure count."""
        self.count = 0

    def should_log(self) -> bool:
        """Return True if we should log this failure (first 3 only)."""
        return self.count <= 3


class CameraWorker:
    """Encapsulates a single camera's state and thread with thread-safe operations."""

    def __init__(
        self, camera_name: str, camera: Union[PhysicalCamera, VideoFileCamera]
    ) -> None:
        self.camera_name = camera_name
        self.camera = camera
        self.running = True
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._current_frame: Optional[np.ndarray] = None
        self._current_timestamp: float = 0.0
        self._last_cached_frame: Optional[np.ndarray] = None

    def set_current_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """Thread-safe frame update."""
        with self._lock:
            self._current_frame = frame.copy()
            self._current_timestamp = timestamp

    def get_current_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """Thread-safe frame retrieval."""
        with self._lock:
            if self._current_frame is None:
                return None
            return (self._current_frame.copy(), self._current_timestamp)

    def set_cached_frame(self, frame: np.ndarray) -> None:
        """Thread-safe cached frame update."""
        with self._lock:
            self._last_cached_frame = frame.copy()

    def get_cached_frame(self) -> Optional[np.ndarray]:
        """Thread-safe cached frame retrieval."""
        with self._lock:
            return self._last_cached_frame.copy() if self._last_cached_frame is not None else None

    def start(self, worker_fn) -> None:
        """Start the camera worker thread."""
        self.thread = threading.Thread(
            target=worker_fn,
            args=(self,),
            daemon=True,
            name=f"CameraThread-{self.camera_name}",
        )
        self.thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the camera worker thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=timeout)


class CameraThreadManager:
    """Manages camera threads and serves them to the web interface."""

    def __init__(self, web_interface: EagleEyeInterface, logger: Logger) -> None:
        """
        Initialize the camera thread manager.

        Args:
            web_interface: The web interface to serve camera feeds to.
            logger: Logger instance for logging.
        """
        self.web_interface = web_interface
        self.logger = logger
        self.cameras: Dict[str, CameraWorker] = {}
        self.start_time_ms = time.time() * 1000.0
        self.known_cameras: Set[str] = set()
        self._initialize_cameras()

    def _initialize_cameras(self) -> None:
        """
        Register system and video file cameras with the manager.
        """
        self.logger.log(f"{Colors.CYAN}Registering system cameras...{Colors.RESET}")
        self.known_cameras = add_system_cameras(
            self.web_interface,
            self,
            self.known_cameras,
            logger=self.logger,
        )

        self.known_cameras = add_video_file_cameras(
            self.web_interface,
            self,
            self.known_cameras,
            logger=self.logger,
        )

    def camera_feed_worker(self, worker: CameraWorker) -> None:
        """
        Worker function that continuously captures frames and updates the web interface.

        Args:
            worker: The CameraWorker instance containing camera and state.
        """
        camera_name = worker.camera_name
        camera = worker.camera

        self.logger.log(
            f"{Colors.CYAN}Starting camera feed worker for {camera_name}{Colors.RESET}"
        )

        failure_tracker = FailureTracker(max_failures=10)

        camera_fps = camera.get_achieved_fps()
        target_frame_time = (
            1.0 / (camera_fps + 5) if camera_fps > 0 else 0.033
        )

        self.logger.log(
            f"{Colors.CYAN}Camera thread for {camera_name} is running at {camera_fps} target fps{Colors.RESET}"
        )

        while worker.running:
            try:
                start_time = time.time()
                frame = camera.get_frame()

                if frame is not None:
                    failure_tracker.reset()
                    worker.set_cached_frame(frame)
                    current_time_ms = time.time() * 1000.0
                    timestamp_from_start = current_time_ms - self.start_time_ms
                    worker.set_current_frame(frame, timestamp_from_start)
                    self.web_interface.update_camera_frame(camera_name, frame)
                else:
                    if failure_tracker.record_failure():
                        self.logger.log(
                            f"{Colors.RED}Too many consecutive failures from {camera_name}, stopping worker{Colors.RESET}"
                        )
                        break

                    cached_frame = worker.get_cached_frame()
                    if cached_frame is not None:
                        current_time_ms = time.time() * 1000.0
                        timestamp_from_start = current_time_ms - self.start_time_ms
                        worker.set_current_frame(cached_frame, timestamp_from_start)

                    if failure_tracker.should_log():
                        self.logger.log(
                            f"{Colors.YELLOW}Failed to get frame from {camera_name} ({failure_tracker.count}){Colors.RESET}"
                        )
                    time.sleep(0.001)

                time_to_sleep = target_frame_time - (time.time() - start_time)
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)

            except Exception as camera_error:
                self.logger.log(
                    f"{Colors.RED}Error in camera feed worker for {camera_name}: {camera_error}{Colors.RESET}"
                )
                if failure_tracker.record_failure():
                    self.logger.log(
                        f"{Colors.RED}Too many errors from {camera_name}, stopping worker{Colors.RESET}"
                    )
                    break
                time.sleep(0.01)

        self.logger.log(
            f"{Colors.CYAN}Camera feed worker for {camera_name} stopped{Colors.RESET}"
        )

    def start_camera_thread(
        self,
        camera_name: str,
        camera_calibration_folder: str | None,
        video_file_path: Optional[str] = None,
        camera_index: Optional[int] = None,
    ) -> bool:
        """
        Start a thread for a specific camera.

        Args:
            camera_name: The name of the camera.
            camera_calibration_folder: The path to the camera calibration folder.
            video_file_path: The path to the video file.
            camera_index: The index of the physical camera.

        Returns:
            True if the thread was started successfully, False otherwise.
        """
        try:
            if video_file_path:
                camera = VideoFileCamera(
                    camera_name,
                    camera_calibration_folder,
                    video_file_path,
                    self.logger.log,
                )
            else:
                if camera_index is None:
                    raise ValueError("Camera index is required for physical cameras")
                camera = PhysicalCamera(
                    camera_name,
                    camera_index,
                    camera_calibration_folder,
                    self.logger.log,
                )

            worker = CameraWorker(camera_name, camera)
            self.cameras[camera_name] = worker
            worker.start(self.camera_feed_worker)

            self.logger.log(
                f"{Colors.GREEN}Successfully started camera thread for {camera_name} (index: {camera_index}){Colors.RESET}"
            )
            return True

        except Exception as start_error:
            self.logger.log(
                f"{Colors.RED}Failed to start camera thread for {camera_name}: {start_error}{Colors.RESET}"
            )
            self.logger.log(
                f"{Colors.RED}Full traceback: {traceback.format_exc()}{Colors.RESET}"
            )
            return False

    def stop_camera_thread(self, camera_name: str) -> None:
        """
        Stop a specific camera thread.

        Args:
            camera_name: The name of the camera to stop.
        """
        if worker := self.cameras.pop(camera_name, None):
            self.logger.log(
                f"{Colors.CYAN}Stopping camera thread for {camera_name}{Colors.RESET}"
            )
            worker.stop()

    def stop_all_cameras(self) -> None:
        """Stop all camera threads."""
        self.logger.log(f"{Colors.CYAN}Stopping all camera threads...{Colors.RESET}")
        camera_names = list(self.cameras.keys())
        for camera_name in camera_names:
            self.stop_camera_thread(camera_name)
        self.logger.log(f"{Colors.CYAN}All camera threads stopped{Colors.RESET}")

    def get_current_frame(self, camera_name: str) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get the most current frame and timestamp for a specific camera.

        Args:
            camera_name: The name of the camera.

        Returns:
            Tuple of (frame, timestamp_ms_from_start) if available, None otherwise.
        """
        worker = self.cameras.get(camera_name)
        return worker.get_current_frame() if worker else None

    def get_all_current_frames(self) -> Dict[str, Tuple[np.ndarray, float]]:
        """
        Get the most current frames and timestamps for all cameras.

        Returns:
            Dictionary mapping camera names to (frame, timestamp_ms_from_start) tuples.
        """
        result = {}
        for name, worker in self.cameras.items():
            if frame_data := worker.get_current_frame():
                result[name] = frame_data
        return result

    def get_start_time_ms(self) -> float:
        """
        Get the start time in milliseconds when the manager was initialized.

        Returns:
            Start time in milliseconds since epoch.
        """
        return self.start_time_ms

    def get_all_camera_names(self) -> list[str]:
        """
        Get the names of all cameras.

        Returns:
            List of camera names.
        """
        return list(self.cameras.keys())

    def get_camera_ready(self, camera_name: str) -> bool:
        """
        Get the ready state of a specific camera.

        Args:
            camera_name: The name of the camera.
        """
        worker = self.cameras.get(camera_name)
        return worker.camera.camera_ready if worker else False

    def get_video_camera_index(self) -> int:
        """
        Get the index of the video camera.
        """
        for worker in self.cameras.values():
            if isinstance(worker.camera, VideoFileCamera):
                return worker.camera.get_frame_index()
        return -1
