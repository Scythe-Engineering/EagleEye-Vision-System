from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Generator

import cv2
from flask import Response

from src.webui.web_server_utils.constants import (
    PIPELINE_NOT_FOUND_MESSAGE,
    TEXT_PLAIN_MIMETYPE,
    VISUALIZATION_STREAM_FPS,
    no_image_jpeg_bytes,
)

if TYPE_CHECKING:
    from src.config.utils.pipeline import Pipeline
    from src.main_operations.definitions.base.base_class import OperationInstance


class VisualizationMixin:
    def start_visualize(
        self, pipeline_name: str, operation_uuid: str
    ) -> tuple[dict, int]:
        """
        Start visualizing the pipeline.

        Args:
            pipeline_name: Name of the pipeline to visualize.
            operation_uuid: UUID of the operation instance to visualize.

        Returns:
            A response message and HTTP status code.
        """
        try:
            pipeline = self.pipeline_objects_callback()[pipeline_name]
            operation = pipeline.get_operation_by_uuid(operation_uuid)
            if operation is None:
                return {"message": "Operation not found"}, 404
            if not self._instance_has_visualization(operation.instance):
                with pipeline.visualization_data_lock:
                    pipeline.set_visualize = False
                    pipeline.visualization_operation_uuid = None
                    pipeline.visualization_data = None
                return {"message": "Operation has no visualization"}, 400
            pipeline.start_visualize(operation.uuid)
        except KeyError:
            return {"message": PIPELINE_NOT_FOUND_MESSAGE}, 404
        return {"message": "Pipeline visualized successfully"}, 200

    def stop_visualize(self, pipeline_name: str) -> tuple[dict, int]:
        """
        Stop visualizing the pipeline.

        Args:
            pipeline_name: Name of the pipeline.
        """
        try:
            self.pipeline_objects_callback()[pipeline_name].stop_visualize()
        except KeyError:
            return {"message": PIPELINE_NOT_FOUND_MESSAGE}, 404
        return {"message": "Pipeline visualized stopped"}, 200

    def visualize(self, pipeline_name: str) -> Response:
        """
        Visualize the pipeline.

        Args:
            pipeline_name: Name of the pipeline.

        Returns the image as JPEG binary data.
        """
        try:
            pipeline = self.pipeline_objects_callback()[pipeline_name]
        except KeyError:
            return Response(
                PIPELINE_NOT_FOUND_MESSAGE, status=404, mimetype=TEXT_PLAIN_MIMETYPE
            )

        with pipeline.visualization_data_lock:
            visualization_data = pipeline.visualization_data

        if visualization_data is None:
            return Response(
                "No visualization data available",
                status=500,
                mimetype=TEXT_PLAIN_MIMETYPE,
            )

        image_array = visualization_data.get("visualization_data")

        if image_array is None:
            return Response(
                "Function has no visualization",
                status=500,
                mimetype=TEXT_PLAIN_MIMETYPE,
            )

        success, encoded_image = cv2.imencode(".jpg", image_array)
        if not success:
            return Response(
                "Failed to encode image", status=500, mimetype=TEXT_PLAIN_MIMETYPE
            )

        return Response(encoded_image.tobytes(), mimetype="image/jpeg")

    def visualize_stream(self, pipeline_name: str) -> Response:
        """Stream visualization frames as MJPEG."""
        try:
            pipeline = self.pipeline_objects_callback()[pipeline_name]
        except KeyError:
            return Response(
                PIPELINE_NOT_FOUND_MESSAGE, status=404, mimetype=TEXT_PLAIN_MIMETYPE
            )

        return Response(
            self._visualization_frame_generator(pipeline),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def _visualization_frame_generator(
        self, pipeline: "Pipeline"
    ) -> Generator[bytes, Any, Any]:
        frame_interval = 1.0 / VISUALIZATION_STREAM_FPS
        last_frame_time = 0.0
        while True:
            now = time.time()
            elapsed = now - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_frame_time = time.time()

            with pipeline.visualization_data_lock:
                visualization_data = pipeline.visualization_data

            image_array = None
            if visualization_data is not None:
                image_array = visualization_data.get("visualization_data")

            if image_array is None:
                frame_bytes = no_image_jpeg_bytes
            else:
                success, encoded_image = cv2.imencode(".jpg", image_array)
                frame_bytes = (
                    encoded_image.tobytes() if success else no_image_jpeg_bytes
                )

            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    def _instance_has_visualization(
        self, operation_instance: "OperationInstance"
    ) -> bool:
        """Check whether an operation instance has a custom visualize method.

        Determines if the operation instance has overridden the base visualize
        method from OperationInstance, indicating it provides visualization
        capabilities.

        Args:
            operation_instance (OperationInstance): The operation instance to
                check for visualization support.

        Returns:
            bool: True if the instance's class has overridden the visualize
                method, False if it uses the default OperationInstance.visualize.
        """
        from src.main_operations.definitions.base.base_class import OperationInstance

        return operation_instance.__class__.visualize is not OperationInstance.visualize
