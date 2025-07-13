from line_profiler import profile
from numpy import ndarray
from ultralytics import YOLO
from networktables import NetworkTable
from ultralytics.engine.results import Results

from src.constants.constants import constants
from src.devices.device import Device

ObjectDetectionConstants = constants["ObjectDetectionConstants"]


class SimpleDevice(Device):
    def __init__(
        self,
        device_type: str,
        model_path: str,
        log: callable,
        eagle_eye_nt: NetworkTable,
        device_index: int = 0,
    ):
        """
        Initializes the SimpleDevice with a specified device type.

        :param device_type: The type of device ("gpu", "cpu", or "tpu").
        :param model_path: The path to the YOLO model file.
        :param log: A callable logger function.
        :param eagle_eye_nt: The NetworkTable instance used for communication.
        :param device_index: The index of the device, in case of multiple devices.
        """
        self.device_type = device_type.lower()
        super().__init__(log, eagle_eye_nt, device_index)

        # Setup the camera key based on the device type, e.g., "gpu:0_active_camera"
        camera_key = f"{self.device_type}:{device_index}_active_camera"
        eagle_eye_nt.putNumber(camera_key, 0)
        eagle_eye_nt.addEntryListener(
            self._change_camera,
            key=camera_key,
            immediateNotify=True,
            localNotify=False,
        )

        # Load the YOLO model
        self.log(f"Loading model from {model_path} on device {self.device_type}")
        self.model = YOLO(model_path, task="detect")
        self.log(f"Model loaded from {model_path}")

    def _change_camera(self, table, key, value, _) -> None:
        """
        Handles updates to the active camera via NetworkTables.
        """
        expected_key = f"{self.device_type}:{self.device_index}_active_camera"
        if table == self.eagle_eye_nt and key == expected_key:
            self.set_camera(value)

    @profile
    def detect(self) -> tuple[None, None, None] | tuple[Results, tuple[int, int], ndarray]:
        """
        Captures a frame from the active camera and runs YOLO detection using the configured device.

        Returns:
            tuple: A tuple containing:
                - detection_result: The first element of the YOLO prediction output.
                - frame_size (tuple): A tuple (width, height) representing the frame size.
                - frame (numpy.ndarray): The captured frame.
        """
        camera = self.get_current_camera()
        frame = camera.get_frame()
        if frame is None:
            return None, None, None

        # Determine the inference device string based on the device type
        if self.device_type == "gpu":
            infer_device = f"cuda:{self.device_index}"
        elif self.device_type == "tpu":
            infer_device = f"tpu:{self.device_index}"
        elif self.device_type == "cpu":
            infer_device = "cpu"
        else:
            raise ValueError(f"Unsupported device type: {self.device_type}")

        results = self.model.predict(
            frame,
            show=False,
            device=infer_device,
            conf=ObjectDetectionConstants["confidence_threshold"],
            imgsz=ObjectDetectionConstants["input_size"],
            verbose=False,
            iou=0.5,
        )

        frame_height, frame_width = frame.shape[0], frame.shape[1]
        frame_size = (frame_width, frame_height)
        return results[0], frame_size, frame

    def get_class_names(self) -> dict[int, str]:
        """
        Returns a dictionary mapping class IDs to class names from the YOLO model.
        """
        return self.model.names
