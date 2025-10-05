import csv
import faulthandler
import os
from pathlib import Path
from time import sleep
from typing import Callable, Dict, Set

faulthandler.enable()

from src.config.utils.generate_all_pipelines import generate_all_pipelines
from src.config.utils.pipeline import Pipeline
from src.utils.camera_utils.camera_thread_manager import CameraThreadManager
from src.utils.camera_utils.check_and_add_new_cameras import check_and_add_new_cameras
from src.utils.device_management_utils.compute_pool import ComputePool
from src.utils.device_management_utils.mx3_accelerator import MX3Accelerator
from src.webui.web_server import EagleEyeInterface

current_dir = Path(__file__).parent


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
        print(f"Added video file camera: {camera_name}")


class MainBackend:
    def __init__(self):
        try:
            print("Initializing EagleEye backend...")

            self.web_interface = EagleEyeInterface(
                restart_callback=self.restart,
                pipeline_objects_callback=self.get_pipelines,
            )
            self.camera_manager = CameraThreadManager(self.web_interface)
            self.known_cameras: Set[str] = set()

            self.compute_pool = ComputePool()
            self.compute_pool.add_compute_device(MX3Accelerator())

            self.pipelines: Dict[str, Dict[str, Pipeline]] = generate_all_pipelines(
                self.web_interface,
                self.compute_pool,
                DummyNetworkTable(self.camera_manager.get_video_camera_index),
            )

            # Initial camera detection
            print("Performing initial camera detection...")
            self.known_cameras = check_and_add_new_cameras(
                self.web_interface, self.camera_manager, self.known_cameras
            )

            add_video_file_cameras(
                self.web_interface, self.camera_manager, self.known_cameras
            )

            available_cameras = self.camera_manager.get_all_camera_names()
            for camera_name, pipelines in self.pipelines.items():
                if camera_name in available_cameras:
                    for pipeline_name, pipeline in pipelines.items():
                        pipeline.thread_run(self.camera_manager, camera_name)
                        print(
                            f"Started pipeline for camera bus id: {camera_name} and pipeline name: {pipeline_name}"
                        )
                else:
                    print(
                        f"Pipeline bus id: {camera_name} was not found in available cameras"
                    )

            if not self.known_cameras:
                print("No cameras detected initially.")
            else:
                print(
                    f"Initially detected {len(self.known_cameras)} cameras: {list(self.known_cameras)}"
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
            print("Restarting systemctl service...")
            try:
                # Get the service name from environment or use a default
                service_name = os.environ.get("SERVICE_NAME", "eagleeye")
                result = os.system(f"sudo systemctl restart {service_name}")
                if result == 0:
                    print(f"Successfully restarted systemctl service: {service_name}")
                else:
                    print(f"Failed to restart systemctl service: {service_name}")
            except Exception as e:
                print(f"Error restarting systemctl service: {e}")

    def restart(self) -> None:
        """
        Restart the backend by shutting down and triggering systemctl service restart.
        """
        self.shutdown(restart_service=True)


def main() -> None:
    """Main function to initialize and continuously monitor for cameras."""
    global backend
    try:
        backend = MainBackend()
        # Keep the main thread alive to allow restart threads to complete
        while True:
            sleep(1)
    except KeyboardInterrupt:
        if "backend" in globals():
            backend.shutdown()


if __name__ == "__main__":
    main()
