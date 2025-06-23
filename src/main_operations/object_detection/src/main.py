import os
import sys

from line_profiler import profile

print(
    f"Setting Working Dir to: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
)
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants.constants import constants
from src.custom_logging.log import Logger

logger = Logger(None)
log = logger.log

# Define constant for the web server key
RUN_WEB_SERVER_KEY = "DisplayConstants.run_web_server"

# run web server that streams video
if constants[RUN_WEB_SERVER_KEY]:
    from webui.web_server import EagleEyeInterface

    web_interface = EagleEyeInterface(settings_object=constants, log=log)
else:
    web_interface = None

import struct
from threading import Lock, Thread
from time import sleep, time

import numpy as np
from networktables import NetworkTables

from src.devices.simple_device import SimpleDevice
from src.math_conversions import (
    calculate_local_position,
    convert_to_global_position,
    pixels_to_degrees,
)
from src.utils.results_to_image import results_to_image

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

# As a client to connect to a robot
NetworkTables.initialize(server=constants["NetworkTableConstants.server_address"])
game_piece_nt = NetworkTables.getTable("GamePieces")
eagle_eye_nt = NetworkTables.getTable("EagleEye")
advantage_kit_nt = NetworkTables.getTable("AdvantageKit")


def time_ms():
    return time() * 1000


class EagleEye:
    def __init__(self):
        model_path = self._select_model_path()
        cameras = self._group_cameras_by_device()
        self.devices = self._initialize_devices_and_cameras(cameras, model_path)
        self.data = {}
        self.data_lock = Lock()
        self._start_detection_threads()
        class_names = self._aggregate_class_names()
        sleep(1)
        log("All threads running")
        game_piece_nt.putStringArray("class_names", class_names)
        self._main_detection_loop(class_names)

    def _select_model_path(self) -> str:
        model_paths = [
            f"src/models/{model}"
            for model in os.listdir("src/models")
            if not model.endswith(".md") and not model.startswith("_")
        ]
        model_path = model_paths[0]  # only load first found model
        log(f"Loading model: {model_path}")
        return model_path

    def _group_cameras_by_device(self) -> dict:
        cameras = {}
        for camera in constants["CameraConstants.camera_list"]:
            if camera["processing_device"] not in cameras:
                cameras[camera["processing_device"]] = []
            cameras[camera["processing_device"]].append(camera)
        log(f"{len(cameras)} devices found")
        log(f"Cameras: {cameras}")
        log("Starting devices...")
        return cameras

    def _initialize_devices_and_cameras(self, cameras: dict, model_path: str) -> list:
        devices = []
        for device, camera_list in cameras.items():
            device = SimpleDevice(
                "gpu", model_path, log, eagle_eye_nt, len(self.devices)
            )
            for camera in camera_list:
                device.add_camera(camera)
                web_interface.serve_camera_feed(camera["name"])
            devices.append(device)
        log(f"{len(self.devices)} devices started")
        return devices

    def _start_detection_threads(self):
        detection_threads = []
        for device in self.devices:
            t = Thread(target=self.detection_thread, args=(device,))
            detection_threads.append(t)
            t.start()

    def _aggregate_class_names(self) -> list:
        class_names = []
        for device in self.devices:
            class_names += list(device.get_class_names().values())
        return class_names

    def _main_detection_loop(self, class_names: list):
        while True:
            collected_detections, num_detections = self._collect_detections()
            if num_detections == 0:
                self._reset_network_tables(class_names)
                sleep(0.016)
                continue
            self._sort_detections_by_distance(collected_detections)
            self._filter_close_detections(collected_detections)
            self._update_network_tables(collected_detections)
            sleep(0.016)

    def _collect_detections(self) -> tuple[dict, int]:
        collected_detections = {}
        num_detections = 0

        for camera in constants["CameraConstants.camera_list"]:
            with self.data_lock:
                detections = self.data.get(camera["name"], [])
                for detection in detections:
                    if detection["class"] not in collected_detections:
                        collected_detections[detection["class"]] = []
                    collected_detections[detection["class"]].append(detection)
                    num_detections += 1

        return collected_detections, num_detections

    def _reset_network_tables(self, class_names: list):
        for class_name in class_names:
            game_piece_nt.putNumberArray(f"{class_name}_yaw_angles", [])
            game_piece_nt.putStringArray(f"{class_name}_local_positions", [])
            game_piece_nt.putStringArray(f"{class_name}_global_positions", [])
            game_piece_nt.putNumberArray(f"{class_name}_distances", [])
            game_piece_nt.putNumberArray(f"{class_name}_ratio", [])

    def _sort_detections_by_distance(self, collected_detections: dict):
        for class_name, detections in collected_detections.items():
            collected_detections[class_name] = sorted(
                detections, key=lambda x: x["distance"]
            )

    def _filter_close_detections(self, collected_detections: dict):
        for class_name, detections in collected_detections.items():
            if len(detections) > 1:
                for i in range(len(detections) - 1):
                    for j in range(i + 1, len(detections)):
                        distance = np.linalg.norm(
                            detections[i]["local_position"]
                            - detections[j]["local_position"]
                        )
                        if (
                            distance
                            < constants["ObjectDetectionConstants.combined_threshold"]
                        ):
                            detections.pop(j)
                            break

    def _update_network_tables(self, collected_detections: dict):
        for class_name, detections in collected_detections.items():
            game_piece_nt.putNumberArray(
                f"{class_name}_yaw_angles",
                [detection["yaw_angle"] for detection in detections],
            )
            game_piece_nt.putStringArray(
                f"{class_name}_local_positions",
                [
                    str(detection["local_position"].tolist())
                    .replace("]", "")
                    .replace("[", "")
                    for detection in detections
                ],
            )
            game_piece_nt.putStringArray(
                f"{class_name}_global_positions",
                [
                    str(detection["global_position"].tolist())
                    .replace("]", "")
                    .replace("[", "")
                    for detection in detections
                ],
            )
            game_piece_nt.putNumberArray(
                f"{class_name}_distances",
                [detection["distance"] for detection in detections],
            )
            game_piece_nt.putNumberArray(
                f"{class_name}_ratio",
                [detection["ratio"] for detection in detections],
            )

    @profile
    def detection_thread(self, device: SimpleDevice):
        log(f"Starting thread for {device.get_current_camera().get_name()} camera")
        estimated_fps = 0
        while True:
            start_time = time_ms()
            robot_pose = np.array(
                struct.unpack(
                    "ddd",
                    advantage_kit_nt.getValue(
                        "RealOutputs/Odometry/Robot", np.array([0, 0, 0])
                    ),
                )
            )
            results, frame_size, frame = device.detect()

            if results is None:
                log(
                    f"{RED}No frame{RESET}",
                    force_no_log=(not constants["Constants"]["detection_logging"]),
                )
                sleep(0.002)
                continue

            log(
                f"Speeds: {results.speed}",
                force_no_log=(not constants["Constants"]["detection_logging"]),
            )

            # if no detections, continue
            if not results.boxes:
                with self.data_lock:
                    self.data[device.get_current_camera().get_name()] = []
                if constants[RUN_WEB_SERVER_KEY]:
                    estimated_fps = int(1000 / (time_ms() - start_time))
                    web_interface.update_camera_frame(
                        device.get_current_camera().get_name(),
                        results_to_image(frame=frame, results=[], fps=estimated_fps),
                    )
                sleep(0.002)
                continue

            detections = []
            debug_points = []

            for box in results.boxes:
                box_class = device.get_class_names()[int(box.cls[0])]
                box_confidence = box.conf.tolist()[0]
                box_lx = box.xyxy.tolist()[0][0]
                box_bottom_center_y = box.xyxy.tolist()[0][3]

                box_rx = box.xyxy.tolist()[0][2]

                box_width = box_rx - box_lx
                box_height = box_bottom_center_y - box.xyxy.tolist()[0][1]
                box_ratio = box_width / box_height

                box_bottom_center_x = (box_lx + box_rx) / 2

                debug_points.append(
                    [int(box_bottom_center_x), int(box_bottom_center_y)]
                )

                # make pixel positions relative to the center
                box_bottom_center_x -= frame_size[0] // 2
                box_bottom_center_y -= frame_size[1] // 2
                box_bottom_center_y = -box_bottom_center_y

                yaw_angle = pixels_to_degrees(
                    box_bottom_center_x,
                    frame_size[0],
                    float(device.get_current_camera().get_fov()[0]),
                    log,
                )
                object_local_position = calculate_local_position(
                    np.array([box_bottom_center_x, box_bottom_center_y]),
                    frame_size,
                    device.get_current_camera().get_fov(),
                    device.get_current_camera().get_camera_offset_pos(),
                    log,
                )
                object_global_position = convert_to_global_position(
                    object_local_position, robot_pose
                )

                distance = np.linalg.norm(object_local_position)
                if distance > constants["ObjectDetectionConstants.max_distance"]:
                    continue

                detections.append(
                    {
                        "class": box_class,
                        "confidence": box_confidence,
                        "yaw_angle": yaw_angle,
                        "local_position": object_local_position,
                        "global_position": object_global_position,
                        "distance": distance,
                        "ratio": box_ratio,
                    }
                )

            if constants[RUN_WEB_SERVER_KEY]:
                web_interface.update_camera_frame(
                    device.get_current_camera().get_name(),
                    results_to_image(frame=frame, results=results, fps=estimated_fps),
                )

            with self.data_lock:
                self.data[device.get_current_camera().get_name()] = detections

            total_inference_time = sum(results.speed.values()) + (
                time_ms() - start_time
            )
            estimated_fps = 1000 / total_inference_time
            log(
                f"Total processing time (ms): {total_inference_time}",
                force_no_log=(not constants["Constants"]["detection_logging"]),
            )
            log(
                f"Post processing time (ms): {time_ms() - start_time}",
                force_no_log=(not constants["Constants"]["detection_logging"]),
            )
            log(
                f"Estimated fps: {estimated_fps}",
                force_no_log=(not constants["Constants"]["detection_logging"]),
            )


if __name__ == "__main__":
    EagleEye()
