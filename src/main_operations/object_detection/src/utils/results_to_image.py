import cv2
import numpy as np
from ultralytics.engine.results import Results


def results_to_image(frame: np.ndarray, results: list | Results, fps: float) -> bytes:
    """
    Convert results to an image with bounding boxes and labels.

    Args:
        frame (np.ndarray): The original image frame.
        results (list): List of detection results.
        fps (float): Frames per second for display.

    Returns:
        bytes: The frame with drawn bounding boxes and labels, encoded as bytes.
    """
    if not isinstance(results, list):
        frame = results.plot()

    compression_ratio = 0.5
    frame = cv2.resize(frame, (0, 0), fx=compression_ratio, fy=compression_ratio)

    # Correct FPS text placement and formatting, adjusted for compression ratio
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (int(10 * compression_ratio), int(30 * compression_ratio)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7 * compression_ratio,
        (255, 255, 255),
        int(2 * compression_ratio),
    )

    _, buffer = cv2.imencode(".jpg", frame)
    frame_bytes = buffer.tobytes()

    return frame_bytes
