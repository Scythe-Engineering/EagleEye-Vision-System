import faulthandler
import os
import subprocess
from pathlib import Path
from time import sleep
from typing import Dict

faulthandler.enable()

from src.config.utils.generate_all_pipelines import generate_all_pipelines  # noqa: E402
from src.config.utils.pipeline import Pipeline  # noqa: E402
from src.rust_implementations.build import main as rust_build  # noqa: E402
from src.utils.camera_utils.camera_thread_manager import (  # noqa: E402
    CameraThreadManager,  # noqa: E402
)
from src.utils.colors import Colors  # noqa: E402
from src.utils.device_management_utils.compute_pool import ComputePool  # noqa: E402
from src.utils.device_management_utils.cpu import CPU  # noqa: E402
from src.utils.get_available_devices import get_available_devices  # noqa: E402
from src.utils.logging.logger import Logger  # noqa: E402
from src.webui.web_server import EagleEyeInterface  # noqa: E402
from networktables import NetworkTables  # noqa: E402  # ty:ignore[unresolved-import]
from src.utils.flatpack_schema.schema_manifest import generate_schema_manifest_bytes  # noqa: E40, E402
import json  # noqa: E402

# Bootstrap Rust modules (removed during uv sync)
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

# Discover hardware devices early for compute pool initialization
available_devices = get_available_devices(logger=logger)
logger.log(
    f"{Colors.CYAN}Detected Available Devices:{Colors.RESET} {available_devices}"
)

# Static configuration and NetworkTables schema keying
current_dir = Path(__file__).parent
SCHEMA_MANIFEST_KEY = "schema_manifest"

# Ensure general config file exists for NetworkTables initialization
if not os.path.exists("src/general_conf.json"):
    # make empty json file with 0.0.0.0 as the address
    with open("src/general_conf.json", "w") as f:
        json.dump({"network_table_address": "0.0.0.0"}, f)

general_conf = json.load(open("src/general_conf.json"))


class MainBackend:
    def __init__(self, logger: Logger):
        try:
            self.logger = logger

            self.logger.log(
                f"{Colors.YELLOW}Initializing EagleEye backend...{Colors.RESET}"
            )

            # NetworkTables wiring and schema publication
            NetworkTables.initialize(server=general_conf["network_table_address"])
            self.network_table = NetworkTables.getTable("EagleEye")
            schema_manifest_payload = generate_schema_manifest_bytes()
            self.network_table.putRaw(SCHEMA_MANIFEST_KEY, schema_manifest_payload)

            # Web interface, camera manager, and known camera cache
            self.web_interface = EagleEyeInterface(
                restart_callback=self.restart,
                pipeline_objects_callback=self.get_pipelines,
                logger=self.logger,
            )
            self.camera_manager = CameraThreadManager(
                self.web_interface, logger=self.logger
            )
            self.known_cameras = self.camera_manager.known_cameras

            all_cameras_ready = self.camera_manager.wait_for_all_cameras_ready()
            if not all_cameras_ready:
                self.logger.log(
                    f"{Colors.YELLOW}Proceeding with pipeline creation despite camera readiness timeout.{Colors.RESET}"
                )

            # Compute pool setup
            self.compute_pool = ComputePool()
            self._initialize_compute_devices()

            # Pipeline creation and start-up
            self.pipelines: Dict[str, Pipeline] = generate_all_pipelines(
                self.web_interface,
                self.compute_pool,
                self.network_table,
                self.camera_manager,
                logger=self.logger,
            )

            available_cameras = set(self.camera_manager.get_all_camera_names())
            for pipeline_name, pipeline in self.pipelines.items():
                camera_name = pipeline.camera_bus_id
                if camera_name is None:
                    self.logger.log(
                        f"{Colors.YELLOW}Pipeline {pipeline_name} has no camera configured. Skipping start.{Colors.RESET}"
                    )
                    continue
                if camera_name not in available_cameras:
                    self.logger.log(
                        f"{Colors.YELLOW}Pipeline {pipeline_name} camera '{camera_name}' not found. Skipping start.{Colors.RESET}"
                    )
                    continue
                pipeline.thread_run(self.camera_manager)
                self.logger.log(
                    f"{Colors.GREEN}Started pipeline: {pipeline_name}{Colors.RESET}"
                )

            # Initial camera inventory logging
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

    def _initialize_compute_devices(self) -> None:
        """
        Initialize and add all available compute devices to the compute pool.
        """
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

        self._initialize_tpu_devices()
        self._initialize_gpu_devices()

    def _initialize_tpu_devices(self) -> None:
        """
        Add TPU devices to the compute pool.
        """
        tpu_devices = available_devices.get("TPU", [])
        if not tpu_devices:
            return

        from src.utils.device_management_utils.mx3_accelerator import (  # noqa: E402
            MX3Accelerator,
        )

        for tpu_device in tpu_devices:
            if not tpu_device.startswith("memx:"):
                self.logger.log(
                    f"{Colors.YELLOW}Warning: Invalid TPU device format '{tpu_device}', expected 'memx:X'. Skipping.{Colors.RESET}"
                )
                continue

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

    def _initialize_gpu_devices(self) -> None:
        """
        Add GPU devices to the compute pool.
        """
        gpu_devices = available_devices.get("GPU", [])
        if not gpu_devices:
            return

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

    def get_pipelines(self) -> Dict[str, Pipeline]:
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
