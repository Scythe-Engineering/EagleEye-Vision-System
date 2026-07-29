import faulthandler
import json
import os
import subprocess
from pathlib import Path
from time import sleep
from typing import Dict

faulthandler.enable()

from src.utils.colors import Colors  # noqa: E402
from src.utils.logging.logger import Logger  # noqa: E402
from src.startup.install_check import StartupInstallChecker  # noqa: E402

logger = Logger()
StartupInstallChecker(logger=logger).ensure_startup_requirements()

from src.config.utils.generate_all_pipelines import generate_all_pipelines  # noqa: E402
from src.config.utils.pipeline import Pipeline  # noqa: E402
from src.rust_implementations.build import main as rust_build  # noqa: E402
from src.utils.camera_utils.camera_thread_manager import (  # noqa: E402
    CameraThreadManager,
)
from src.utils.camera_utils.camera_config_manager import (  # noqa: E402
    CameraConfigRegistry,
)
from src.utils.device_registry import DeviceRegistry  # noqa: E402
from src.utils.model_library import ModelLibrary  # noqa: E402
from src.webui.web_server import DEFAULT_GENERAL_CONF, EagleEyeInterface  # noqa: E402
import ntcore  # noqa: E402

# Bootstrap Rust modules (removed during uv sync)
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

current_dir = Path(__file__).parent

# Ensure config files exist before backend initialization.
general_conf_path = current_dir / "general_conf.json"
if not general_conf_path.exists():
    with general_conf_path.open("w", encoding="utf-8") as f:
        json.dump(DEFAULT_GENERAL_CONF, f)

pipeline_conf_path = current_dir / "config" / "pipeline_config.json"
pipeline_conf_path.parent.mkdir(parents=True, exist_ok=True)
if not pipeline_conf_path.exists():
    with pipeline_conf_path.open("w", encoding="utf-8") as f:
        json.dump({}, f, indent=4)

with general_conf_path.open("r", encoding="utf-8") as f:
    general_conf = {**DEFAULT_GENERAL_CONF, **json.load(f)}


class MainBackend:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.pipelines: Dict[str, Pipeline] = {}
        self.camera_manager: CameraThreadManager | None = None

        try:
            self.logger.log(
                f"{Colors.YELLOW}Initializing EagleEye backend...{Colors.RESET}"
            )

            # NetworkTables wiring
            network_tables_inst = ntcore.NetworkTableInstance.getDefault()
            network_tables_inst.startClient4("EagleEye")
            network_tables_inst.setServer(general_conf["network_table_address"])
            self.network_table = network_tables_inst.getTable("EagleEye")

            # Hardware discovery happens once during backend initialization.
            self.device_registry = DeviceRegistry.discover(logger=self.logger)
            self.model_library = ModelLibrary(
                root=current_dir.parent / "files" / "models",
                pipeline_config_path=pipeline_conf_path,
            )
            descriptor_ids = [
                descriptor.device_id
                for descriptor in self.device_registry.descriptors()
            ]
            self.logger.log(
                f"{Colors.CYAN}Detected inference devices: {descriptor_ids}{Colors.RESET}"
            )

            # Web interface, camera manager, and known camera cache
            self.web_interface = EagleEyeInterface(
                restart_callback=self.restart,
                pipeline_objects_callback=self.get_pipelines,
                logger=self.logger,
                network_table_instance=network_tables_inst,
                device_registry=self.device_registry,
                model_library=self.model_library,
            )
            self.camera_manager = CameraThreadManager(
                self.web_interface, logger=self.logger
            )
            self.known_cameras = self.camera_manager.known_cameras

            # Camera configuration registry and shared config object for
            # operation-level injection. This centralizes camera config access
            # so operations do not need every camera parameter in action_params.
            self.camera_config_registry = CameraConfigRegistry()
            self.camera_config_registry.load_all_from_directory()
            for camera_info in self.known_cameras:
                camera_bus_id = camera_info.get("bus_id")
                if camera_bus_id is None:
                    continue
                self.camera_config_registry.get_config(str(camera_bus_id))
            self.camera_configs = self.camera_config_registry.get_all_configs()
            self.web_interface.camera_config_registry = self.camera_config_registry

            all_cameras_ready = self.camera_manager.wait_for_all_cameras_ready()
            if not all_cameras_ready:
                self.logger.log(
                    f"{Colors.YELLOW}Proceeding with pipeline creation despite camera readiness timeout.{Colors.RESET}"
                )

            # Pipeline creation and start-up
            self.pipelines = generate_all_pipelines(
                self.web_interface,
                self.network_table,
                self.camera_manager,
                self.camera_config_registry,
                self.device_registry,
                self.model_library,
                logger=self.logger,
            )

            available_bus_ids = set(self.camera_manager.get_all_bus_ids())
            for pipeline_name, pipeline in self.pipelines.items():
                bus_ids = getattr(pipeline, "camera_bus_ids", [])
                if not bus_ids:
                    self.logger.log(
                        f"{Colors.YELLOW}Pipeline {pipeline_name} has no cameras configured. Skipping start.{Colors.RESET}"
                    )
                    continue
                missing_bus_ids = [
                    bus_id for bus_id in bus_ids if bus_id not in available_bus_ids
                ]
                if missing_bus_ids:
                    self.logger.log(
                        f"{Colors.YELLOW}Pipeline {pipeline_name} missing cameras with bus_ids {missing_bus_ids}. Skipping start.{Colors.RESET}"
                    )
                    continue
                pipeline.thread_run(self.camera_manager)
                self.logger.log(
                    f"{Colors.GREEN}Started pipeline: {pipeline_name}{Colors.RESET}"
                )

            # Initial camera inventory logging
            if not self.known_cameras:
                self.logger.log(
                    f"{Colors.RED}No cameras detected initially.{Colors.RESET}"
                )
            else:
                self.logger.log(
                    f"{Colors.CYAN}Detected {len(self.known_cameras)} cameras: {list(self.known_cameras)}{Colors.RESET}"
                )
        except BaseException:
            self.shutdown()
            raise

    def get_pipelines(self) -> Dict[str, Pipeline]:
        """
        Get the pipelines.
        """
        return self.pipelines

    def shutdown(self, restart_service: bool = False) -> None:
        """Stop camera workers and optionally restart the systemd service.

        Args:
            restart_service: Restart the system service after cameras stop.
        """
        if self.camera_manager is not None:
            self.camera_manager.stop_all_cameras()

        if restart_service:
            service_name = os.environ.get("SERVICE_NAME", "eagleeye")
            self.logger.log(
                f"{Colors.CYAN}Restarting systemctl service: {service_name}{Colors.RESET}"
            )
            try:
                subprocess.run(
                    ["sudo", "systemctl", "restart", service_name],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as error:
                error_message = (
                    f"Failed to restart systemctl service: {service_name}, "
                    f"return code: {error.returncode}"
                )
                if error.stdout:
                    error_message += f", stdout: {error.stdout}"
                if error.stderr:
                    error_message += f", stderr: {error.stderr}"
                self.logger.log(f"{Colors.RED}{error_message}{Colors.RESET}")
                raise RuntimeError(error_message) from error

            self.logger.log(
                f"{Colors.GREEN}Successfully restarted systemctl service: "
                f"{service_name}{Colors.RESET}"
            )

    def restart(self) -> None:
        """
        Restart the backend by shutting down and triggering systemctl service restart.
        """
        self.shutdown(restart_service=True)


def main() -> None:
    """Initialize the backend and stop it on process interruption."""
    backend: MainBackend | None = None
    try:
        backend = MainBackend(logger=logger)
        while True:
            sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if backend is not None:
            backend.shutdown()


if __name__ == "__main__":
    main()
