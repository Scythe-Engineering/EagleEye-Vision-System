from __future__ import annotations

import threading
import time
import traceback
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import ntcore
import numpy as np

from src.utils.camera_utils.add_system_cameras import add_system_cameras
from src.utils.camera_utils.add_video_file_cameras import add_video_file_cameras
from src.utils.camera_utils.cameras.physical_camera import PhysicalCamera
from src.utils.camera_utils.cameras.video_file_camera import VideoFileCamera
from src.utils.colors import Colors
from src.utils.logging.logger import Logger
from src.utils.timing import FramePacket, TimedValue, TimingMetadata

if TYPE_CHECKING:
    from src.webui.web_server import EagleEyeInterface


class FailureTracker:
    """Tracks consecutive failures for a camera feed."""

    def __init__(self, max_failures: int = 10) -> None:
        """Initialize the object.
        
        Args:
            max_failures (int): Max failures."""
        self.count = 0
        self.max_failures = max_failures

    def record_failure(self) -> bool:
        """Record failure.
        
        Returns:
            bool: Result of record failure."""
        self.count += 1
        return self.count >= self.max_failures

    def reset(self) -> None:
        """Reset."""
        self.count = 0

    def should_log(self) -> bool:
        """Should log.
        
        Returns:
            bool: Result of should log."""
        return self.count <= 3


class CameraWorker:
    """Encapsulates a single camera's state and thread with thread-safe operations."""

    def __init__(
        self, camera_name: str, camera: Union[PhysicalCamera, VideoFileCamera]
    ) -> None:
        """Initialize the object.
        
        Args:
            camera_name (str): Camera name.
            camera (Union[PhysicalCamera, VideoFileCamera]): Camera."""
        self.camera_name = camera_name
        self.camera = camera
        self.running = True
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._current_packet: FramePacket | None = None
        self._last_cached_packet: FramePacket | None = None
    def set_current_packet(self, packet: FramePacket) -> None:
        """Set current packet.
        
        Args:
            packet (FramePacket): Packet."""
        with self._lock:
            self._current_packet = TimedValue(packet.value.copy(), packet.timing)

    def get_current_packet(self) -> FramePacket | None:
        """Get current packet.
        
        Returns:
            FramePacket | None: Result of get current packet."""
        with self._lock:
            if self._current_packet is None:
                return None
            return TimedValue(self._current_packet.value.copy(), self._current_packet.timing)

    def set_current_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """Set current frame.
        
        Args:
            frame (np.ndarray): Frame.
            timestamp (float): Timestamp."""
        timing = TimingMetadata(capture_nt_us=int(timestamp * 1000))
        self.set_current_packet(TimedValue(frame, timing))

    def get_current_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """Get current frame.
        
        Returns:
            Optional[Tuple[np.ndarray, float]]: Result of get current frame."""
        packet = self.get_current_packet()
        if packet is None:
            return None
        return (packet.value.copy(), packet.timing.capture_nt_us / 1000.0)

    def set_cached_packet(self, packet: FramePacket) -> None:
        """Set cached packet.
        
        Args:
            packet (FramePacket): Packet."""
        with self._lock:
            self._last_cached_packet = TimedValue(packet.value.copy(), packet.timing)

    def get_cached_packet(self) -> FramePacket | None:
        """Get cached packet.
        
        Returns:
            FramePacket | None: Result of get cached packet."""
        with self._lock:
            if self._last_cached_packet is None:
                return None
            return TimedValue(
                self._last_cached_packet.value.copy(),
                self._last_cached_packet.timing,
            )

    def set_cached_frame(self, frame: np.ndarray) -> None:
        """Set cached frame.
        
        Args:
            frame (np.ndarray): Frame."""
        timing = TimingMetadata(capture_nt_us=ntcore._now())
        self.set_cached_packet(TimedValue(frame, timing))

    def get_cached_frame(self) -> Optional[np.ndarray]:
        """Get cached frame.
        
        Returns:
            Optional[np.ndarray]: Result of get cached frame."""
        packet = self.get_cached_packet()
        return packet.value.copy() if packet is not None else None

    def start(self, worker_fn) -> None:
        """Start.
        
        Args:
            worker_fn: Worker fn."""
        self.thread = threading.Thread(
            target=worker_fn,
            args=(self,),
            daemon=True,
            name=f"CameraThread-{self.camera_name}",
        )
        self.thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop.
        
        Args:
            timeout (float): Timeout."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=timeout)


class CameraThreadManager:
    """Manages camera threads and serves them to the web interface."""

    def __init__(self, web_interface: EagleEyeInterface, logger: Logger) -> None:
        """Initialize the object.
        
        Args:
            web_interface (EagleEyeInterface): Web interface.
            logger (Logger): Logger."""
        self.web_interface = web_interface
        self.logger = logger
        self.cameras: Dict[str, CameraWorker] = {}
        self.start_time_ms = time.time() * 1000.0
        self.known_cameras: list[dict[str, str | int]] = []
        self.bus_id_to_name: Dict[str, str] = {}
        self._initialize_cameras()

    def _initialize_cameras(self) -> None:
        """Initialize cameras."""
        self.logger.log(f"{Colors.CYAN}Registering system cameras...{Colors.RESET}")
        system_cameras = add_system_cameras(
            self.web_interface,
            self,
            logger=self.logger,
        )

        video_file_cameras = add_video_file_cameras(
            self.web_interface,
            self,
            logger=self.logger,
        )
        self.known_cameras = system_cameras + video_file_cameras

    def camera_feed_worker(self, worker: CameraWorker) -> None:
        """Camera feed worker.
        
        Args:
            worker (CameraWorker): Worker."""
        camera_name = worker.camera_name
        camera = worker.camera

        self.logger.log(
            f"{Colors.CYAN}Starting camera feed worker for {camera_name}{Colors.RESET}"
        )

        failure_tracker = FailureTracker(max_failures=10)

        camera_fps = camera.get_achieved_fps()
        target_frame_time = 1.0 / (camera_fps + 5) if camera_fps > 0 else 0.033

        self.logger.log(
            f"{Colors.CYAN}Camera thread for {camera_name} is running at {camera_fps} target fps{Colors.RESET}"
        )

        while worker.running:
            try:
                start_time = time.time()
                frame = camera.get_frame()

                if frame is not None:
                    failure_tracker.reset()
                    packet = TimedValue(
                        frame,
                        TimingMetadata(capture_nt_us=ntcore._now()),
                    )
                    worker.set_cached_packet(packet)
                    worker.set_current_packet(packet)
                    self.web_interface.update_camera_frame(camera_name, frame)
                else:
                    if failure_tracker.record_failure():
                        self.logger.log(
                            f"{Colors.RED}Too many consecutive failures from {camera_name}, stopping worker{Colors.RESET}"
                        )
                        break

                    cached_packet = worker.get_cached_packet()
                    if cached_packet is not None:
                        worker.set_current_packet(cached_packet)

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
        video_file_path: Optional[str] = None,
        camera_index: Optional[int] = None,
        frame_width: int = 1280,
        frame_height: int = 720,
    ) -> bool:
        """Start camera thread.
        
        Args:
            camera_name (str): Camera name.
            video_file_path (Optional[str]): Video file path.
            camera_index (Optional[int]): Camera index.
            frame_width (int): Frame width.
            frame_height (int): Frame height.
        
        Returns:
            bool: Result of start camera thread."""
        try:
            if video_file_path:
                camera = VideoFileCamera(
                    camera_name,
                    video_file_path,
                    self.logger.log,
                )
            else:
                if camera_index is None:
                    raise ValueError("Camera index is required for physical cameras")
                camera = PhysicalCamera(
                    camera_name,
                    camera_index,
                    frame_width,
                    frame_height,
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
        """Stop camera thread.
        
        Args:
            camera_name (str): Camera name."""
        if worker := self.cameras.pop(camera_name, None):
            self.logger.log(
                f"{Colors.CYAN}Stopping camera thread for {camera_name}{Colors.RESET}"
            )
            worker.stop()

    def stop_all_cameras(self) -> None:
        """Stop all cameras."""
        self.logger.log(f"{Colors.CYAN}Stopping all camera threads...{Colors.RESET}")
        camera_names = list(self.cameras.keys())
        for camera_name in camera_names:
            self.stop_camera_thread(camera_name)
        self.logger.log(f"{Colors.CYAN}All camera threads stopped{Colors.RESET}")

    def get_current_packet(self, camera_name: str) -> FramePacket | None:
        """Get current packet.
        
        Args:
            camera_name (str): Camera name.
        
        Returns:
            FramePacket | None: Result of get current packet."""
        worker = self.cameras.get(camera_name)
        return worker.get_current_packet() if worker else None

    def get_current_frame(self, camera_name: str) -> Optional[Tuple[np.ndarray, float]]:
        """Get current frame.
        
        Args:
            camera_name (str): Camera name.
        
        Returns:
            Optional[Tuple[np.ndarray, float]]: Result of get current frame."""
        worker = self.cameras.get(camera_name)
        return worker.get_current_frame() if worker else None

    def get_all_current_frames(self) -> Dict[str, Tuple[np.ndarray, float]]:
        """Get all current frames.
        
        Returns:
            Dict[str, Tuple[np.ndarray, float]]: Result of get all current frames."""
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

    def wait_for_all_cameras_ready(
        self, timeout_s: float = 30.0, poll_interval_s: float = 0.1
    ) -> bool:
        """
        Wait until all cameras report ready or timeout expires.

        Args:
            timeout_s: Maximum time to wait in seconds.
            poll_interval_s: Sleep interval between readiness checks.

        Returns:
            True if all cameras are ready, False if timeout occurs.
        """
        camera_names = self.get_all_camera_names()
        if not camera_names:
            self.logger.log(
                f"{Colors.YELLOW}No cameras registered to wait for.{Colors.RESET}"
            )
            return True

        deadline = time.monotonic() + timeout_s
        pending = set(camera_names)
        while pending and time.monotonic() < deadline:
            pending = {
                camera_name
                for camera_name in pending
                if not self.get_camera_ready(camera_name)
            }
            if pending:
                time.sleep(poll_interval_s)

        if pending:
            self.logger.log(
                f"{Colors.YELLOW}Camera readiness timeout. Not ready: {sorted(pending)}{Colors.RESET}"
            )
            return False

        self.logger.log(f"{Colors.GREEN}All cameras are ready.{Colors.RESET}")
        return True

    def get_camera_ready(self, camera_name: str) -> bool:
        """Get camera ready.
        
        Args:
            camera_name (str): Camera name.
        
        Returns:
            bool: Result of get camera ready."""
        worker = self.cameras.get(camera_name)
        return worker.camera.camera_ready if worker else False

    def register_bus_id(self, bus_id: str, camera_name: str) -> None:
        """Register a mapping from bus_id to camera name.

        Args:
            bus_id: The USB bus identifier for the camera.
            camera_name: The camera name associated with this bus_id.
        """
        self.bus_id_to_name[bus_id] = camera_name

    def get_camera_name_by_bus_id(self, bus_id: str) -> Optional[str]:
        """Get the camera name associated with a bus_id.

        Args:
            bus_id: The USB bus identifier to look up.

        Returns:
            The camera name if found, None otherwise.
        """
        return self.bus_id_to_name.get(bus_id)

    def get_bus_id_for_camera_name(self, camera_name: str) -> str | None:
        """Get bus id for camera name.
        
        Args:
            camera_name (str): Camera name.
        
        Returns:
            str | None: Result of get bus id for camera name."""
        for bus_id, registered_name in self.bus_id_to_name.items():
            if registered_name == camera_name:
                return bus_id
        return None

    def get_current_packet_by_bus_id(self, bus_id: str) -> FramePacket | None:
        """Get current packet by bus id.
        
        Args:
            bus_id (str): Bus id.
        
        Returns:
            FramePacket | None: Result of get current packet by bus id."""
        camera_name = self.get_camera_name_by_bus_id(bus_id)
        if camera_name is None:
            return None
        return self.get_current_packet(camera_name)

    def get_current_frame_by_bus_id(
        self, bus_id: str
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Get the current frame for a camera identified by bus_id.

        Args:
            bus_id: The USB bus identifier for the camera.

        Returns:
            Tuple of (frame, timestamp_ms_from_start) if available, None otherwise.
        """
        camera_name = self.get_camera_name_by_bus_id(bus_id)
        if camera_name is None:
            return None
        return self.get_current_frame(camera_name)

    def get_all_bus_ids(self) -> list[str]:
        """Get all registered bus IDs.

        Returns:
            List of registered bus IDs.
        """
        return list(self.bus_id_to_name.keys())
