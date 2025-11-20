import csv
import faulthandler
import os
import subprocess
from pathlib import Path
from time import sleep
from typing import Callable, Dict, Set

faulthandler.enable()

from src.config.utils.generate_all_pipelines import generate_all_pipelines  # noqa: E402
from src.config.utils.pipeline import Pipeline  # noqa: E402
from src.rust_implementations.build import main as rust_build  # noqa: E402
from src.utils.camera_utils.camera_thread_manager import (
    CameraThreadManager,  # noqa: E402
)
from src.utils.camera_utils.check_and_add_new_cameras import (
    check_and_add_new_cameras,  # noqa: E402
)
from src.utils.colors import Colors  # noqa: E402
from src.utils.device_management_utils.compute_pool import ComputePool  # noqa: E402
from src.utils.device_management_utils.cpu import CPU  # noqa: E402
from src.utils.get_available_devices import get_available_devices  # noqa: E402
from src.utils.logging.logger import Logger  # noqa: E402
from src.webui.web_server import EagleEyeInterface  # noqa: E402
from networktables import NetworkTables  # noqa: E402
from src.utils.flatpack_schema.schema_manifest import generate_schema_manifest_bytes  # noqa: E402

# Build the Rust implementations (removed during uv sync)
logger = Logger()
logger.log(
    f"{Colors.CYAN}Building Rust implementations (long first time build)...{Colors.RESET}"
)
build_success = rust_build(logger=logger)
if not build_success:
    error_msg = (
        "Failed to build Rust implementations. Backend initialization cannot continue."
    )
    logger.log(f"{Colors.RED}{error_msg}{Colors.RESET}")
    raise RuntimeError(error_msg)
logger.log(f"{Colors.GREEN}Rust implementations built successfully.{Colors.RESET}")

available_devices = get_available_devices(logger=logger)
logger.log(
    f"{Colors.CYAN}Detected Available Devices:{Colors.RESET} {available_devices}"
)

current_dir = Path(__file__).parent
SCHEMA_MANIFEST_KEY = "schema_manifest"


class DummyNetworkTable:
    def __init__(self, video_camera_index_callback: Callable[[], str]):
        self.video_camera_index_callback = video_camera_index_callback
        self.sim_data = []

        with open(
            os.path.join(current_dir, "utils", "sim_videos", "basic_test_data.csv"),
            "r",
        ) as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header row
            self.sim_data = {}
            for row in reader:
                frame_key = row[0]
                # Create a dictionary mapping column names to values
                frame_data = {header[i]: row[i] for i in range(1, len(header))}
                self.sim_data[frame_key] = frame_data

    def get_number(self, key: str, default: float | bool = 0.0) -> float | bool:
        frame_key = str(self.video_camera_index_callback())
        if frame_key in self.sim_data and str(key) in self.sim_data[frame_key]:
            return float(self.sim_data[frame_key][str(key)])
        return default


def add_video_file_cameras(
    web_interface: EagleEyeInterface,
    camera_manager: CameraThreadManager,
    known_cameras: Set[str],
    logger: Logger,
) -> None:
    """
    Add video file cameras to the system. (Mostly for testing and development purposes)
    """
    # Add a video file cameras
    video_folder = os.path.join(current_dir, "utils", "sim_videos")
    video_files = list(Path(video_folder).glob("*.mp4"))
    for video_file in video_files:
        camera_name = video_file.stem
        web_interface.add_camera(camera_name, -1)
        camera_manager.start_camera_thread(
            camera_name,
            os.path.join(
                current_dir,
                "utils",
                "camera_utils",
                "camera_calibrations",
                "sim_camera",
            ),
            str(video_file),
        )
        known_cameras.add(camera_name)
        logger.log(
            f"{Colors.GREEN}Added video file camera: {camera_name}{Colors.RESET}"
        )


class MainBackend:
    def __init__(self, logger: Logger):
        try:
            self.logger = logger

            self.logger.log(
                f"{Colors.YELLOW}Initializing EagleEye backend...{Colors.RESET}"
            )

            NetworkTables.initialize(server="10.0.0.62")
            self.network_table = NetworkTables.getTable("EagleEye")
            schema_manifest_payload = generate_schema_manifest_bytes()
            self.network_table.putRaw(SCHEMA_MANIFEST_KEY, schema_manifest_payload)

            self.web_interface = EagleEyeInterface(
                restart_callback=self.restart,
                pipeline_objects_callback=self.get_pipelines,
                logger=self.logger,
            )
            self.camera_manager = CameraThreadManager(
                self.web_interface, logger=self.logger
            )
            self.known_cameras: Set[str] = set()

            self.compute_pool = ComputePool()

            # Add CPU device if available
            if available_devices.get("CPU"):
                cpu_device = CPU()
                self.compute_pool.add_compute_device(cpu_device)
                self.logger.log(
                    f"{Colors.GREEN}Added CPU device: {available_devices['CPU'][0]}{Colors.RESET}"
                )

            # Set logger for MX3 module if available
            if available_devices.get("MX3"):
                from src.utils.device_management_utils.mx3_accelerator import (
                    set_mx3_logger,
                )

                set_mx3_logger(self.logger)

            # Add TPU devices if available
            tpu_devices = available_devices.get("TPU", [])
            if tpu_devices:
                from src.utils.device_management_utils.mx3_accelerator import (
                    MX3Accelerator,
                )  # noqa: E402

                for tpu_device in tpu_devices:
                    if tpu_device.startswith("memx:"):
                        try:
                            # Extract device index from memx:X
                            device_parts = tpu_device.split(":", 1)
                            if len(device_parts) != 2:
                                self.logger.log(
                                    f"{Colors.YELLOW}Warning: Invalid TPU device format '{tpu_device}', expected 'memx:X'. Skipping.{Colors.RESET}"
                                )
                                continue

                            device_index = device_parts[1]
                            mx3_device = MX3Accelerator(
                                device_id=f"MX3_{device_index}", logger=self.logger
                            )
                            self.compute_pool.add_compute_device(mx3_device)

                            self.logger.log(
                                f"{Colors.GREEN}Added Memryx TPU device: {tpu_device}{Colors.RESET}"
                            )
                        except Exception as e:
                            self.logger.log(
                                f"{Colors.YELLOW}Warning: Failed to add TPU device '{tpu_device}': {e}. Skipping.{Colors.RESET}"
                            )

            # Add GPU devices if available
            gpu_devices = available_devices.get("GPU", [])
            if gpu_devices:
                from src.utils.device_management_utils.gpu import GPU  # noqa: E402

                for gpu_index, gpu_device_name in enumerate(gpu_devices):
                    try:
                        gpu_device = GPU(device_id=f"GPU_{gpu_index}")
                        self.compute_pool.add_compute_device(gpu_device)

                        self.logger.log(
                            f"{Colors.GREEN}Added GPU device {gpu_index}: {gpu_device_name}{Colors.RESET}"
                        )
                    except RuntimeError as e:
                        self.logger.log(
                            f"{Colors.YELLOW}Warning: Failed to add GPU device '{gpu_device_name}': {e}. Skipping.{Colors.RESET}"
                        )
                    except Exception as e:
                        self.logger.log(
                            f"{Colors.YELLOW}Warning: Unexpected error adding GPU device '{gpu_device_name}': {e}. Skipping.{Colors.RESET}"
                        )

            self.pipelines: Dict[str, Dict[str, Pipeline]] = generate_all_pipelines(
                self.web_interface,
                self.compute_pool,
                self.network_table,
                self.camera_manager,
                logger=self.logger,
            )

            # Initial camera detection
            self.logger.log(
                f"{Colors.CYAN}Performing initial camera detection...{Colors.RESET}"
            )
            self.known_cameras = check_and_add_new_cameras(
                self.web_interface, self.camera_manager, self.known_cameras, self.logger
            )

            add_video_file_cameras(
                self.web_interface, self.camera_manager, self.known_cameras, self.logger
            )

            available_cameras = self.camera_manager.get_all_camera_names()
            for camera_name, pipelines in self.pipelines.items():
                if camera_name in available_cameras:
                    for pipeline_name, pipeline in pipelines.items():
                        pipeline.thread_run(self.camera_manager, camera_name)
                        self.logger.log(
                            f"{Colors.GREEN}Started pipeline for camera bus id: {camera_name} and pipeline name: {pipeline_name}{Colors.RESET}"
                        )
                else:
                    self.logger.log(
                        f"{Colors.YELLOW}Pipeline bus id: {camera_name} was not found in available cameras{Colors.RESET}"
                    )

            if not self.known_cameras:
                self.logger.log(
                    f"{Colors.YELLOW}No cameras detected initially.{Colors.RESET}"
                )
            else:
                self.logger.log(
                    f"{Colors.CYAN}Initially detected {len(self.known_cameras)} cameras: {list(self.known_cameras)}{Colors.RESET}"
                )
        except KeyboardInterrupt:
            self.shutdown()

    def get_pipelines(self) -> Dict[str, Dict[str, Pipeline]]:
        """
        Get the pipelines.
        """
        return self.pipelines

    def shutdown(self, restart_service: bool = False) -> None:
        """
        Shutdown the backend.

        Args:
            restart_service: If True, trigger a systemctl restart of the service after shutdown.
        """
        if restart_service:
            self.logger.log(
                f"{Colors.CYAN}Restarting systemctl service...{Colors.RESET}"
            )
            try:
                # Get the service name from environment or use a default
                service_name = os.environ.get("SERVICE_NAME", "eagleeye")
                subprocess.run(
                    ["sudo", "systemctl", "restart", service_name],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.logger.log(
                    f"{Colors.GREEN}Successfully restarted systemctl service: {service_name}{Colors.RESET}"
                )
            except subprocess.CalledProcessError as e:
                error_msg = f"Failed to restart systemctl service: {service_name}, return code: {e.returncode}"
                if e.stdout:
                    error_msg += f", stdout: {e.stdout}"
                if e.stderr:
                    error_msg += f", stderr: {e.stderr}"
                self.logger.log(f"{Colors.RED}{error_msg}{Colors.RESET}")
                raise RuntimeError(error_msg) from e

    def restart(self) -> None:
        """
        Restart the backend by shutting down and triggering systemctl service restart.
        """
        self.shutdown(restart_service=True)


def main() -> None:
    """Main function to initialize and continuously monitor for cameras."""
    global backend
    try:
        backend = MainBackend(logger=logger)
        # Keep the main thread alive to allow restart threads to complete
        while True:
            sleep(1)
    except KeyboardInterrupt:
        if "backend" in globals():
            backend.shutdown()


if __name__ == "__main__":
    main()
