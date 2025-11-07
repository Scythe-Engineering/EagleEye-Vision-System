import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

from line_profiler import profile
from src.main_operations.modules.object_detection.utils.letterbox import letterbox_image


class ColorThresholdDetectionImplementation:
    """Implementation for color-based object detection with preprocessing.

    This implementation:
    1. Downscales and letterboxes input frames to 320x320
    2. Performs color thresholding in HSV space for multiple color ranges
    3. Extracts contours and calculates bounding boxes
    4. Returns detections with bounding boxes as 0-1 percentages of resized content area

    Input: np.ndarray (BGR image)
    Output: List[Dict[str, Any]] with keys: bbox, class_id, color_name
    """

    def __init__(
        self,
        target_size: int = 320,
        color_ranges: List[Dict[str, Any]] = None,
        min_area: int = 100,
        max_area: int = 50000,
        blur_kernel_size: int = 5,
        morphology_kernel_size: int = 5,
        morphology_iterations: int = 2,
    ):
        """Initialize color threshold detection implementation.

        Args:
            target_size: Target size for square letterboxed image (default 320)
            color_ranges: List of color dictionaries with format:
                {
                    "name": "red",
                    "class_id": 0,
                    "lower_hsv": [0, 100, 100],
                    "upper_hsv": [10, 255, 255]
                }
            min_area: Minimum contour area to consider as detection
            max_area: Maximum contour area to consider as detection
            blur_kernel_size: Gaussian blur kernel size for noise reduction
            morphology_kernel_size: Kernel size for morphological operations
            morphology_iterations: Number of morphological operation iterations
        """
        self.target_size = target_size
        self.min_area = min_area
        self.max_area = max_area
        self.blur_kernel_size = blur_kernel_size
        self.morphology_kernel_size = morphology_kernel_size
        self.morphology_iterations = morphology_iterations

        if color_ranges is None:
            self.color_ranges = [
                {
                    "name": "red",
                    "class_id": 0,
                    "lower_hsv": [0, 100, 100],
                    "upper_hsv": [10, 255, 255],
                }
            ]
        else:
            self.color_ranges = color_ranges

        self.morphology_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morphology_kernel_size, self.morphology_kernel_size),
        )

    def letterbox_image(
        self, image: np.ndarray, target_size: int
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """Letterbox image to square target size maintaining aspect ratio.

        Args:
            image: Input BGR image
            target_size: Target square size

        Returns:
            Tuple of (letterboxed_image, (resized_width, resized_height), (pad_x, pad_y))
        """
        # Use the existing letterbox function
        letterboxed_image, (resized_width, resized_height), (pad_x, pad_y) = (
            letterbox_image(
                img=image,
                target_size=(target_size, target_size),
                power_two_scaling=False,  # Use standard scaling for better precision
                greyscale=False,  # Keep color channels
                return_resized_size_and_padding=True,
            )
        )

        return letterboxed_image, (resized_width, resized_height), (pad_x, pad_y)

    @profile
    def create_color_mask(
        self, hsv_image: np.ndarray, lower_hsv: List[int], upper_hsv: List[int]
    ) -> np.ndarray:
        """Create binary mask for color range using HSV thresholding.

        Args:
            hsv_image: Image in HSV color space
            lower_hsv: Lower HSV bounds [H, S, V] (0-179 for H, 0-255 for S,V)
            upper_hsv: Upper HSV bounds [H, S, V] (0-179 for H, 0-255 for S,V)

        Returns:
            Binary mask where pixels within HSV range are 255
        """
        lower_bound = np.array(lower_hsv, dtype=np.uint8)
        upper_bound = np.array(upper_hsv, dtype=np.uint8)

        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        return mask

    @profile
    def process_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up mask.

        Args:
            mask: Binary mask

        Returns:
            Cleaned binary mask
        """
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.morphology_kernel,
            iterations=self.morphology_iterations,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            self.morphology_kernel,
            iterations=self.morphology_iterations,
        )

        return mask

    @profile
    def extract_bounding_boxes(
        self, mask: np.ndarray, class_id: int, color_name: str
    ) -> List[Dict[str, Any]]:
        """Extract bounding boxes from binary mask using contours.

        Args:
            mask: Binary mask
            class_id: Class ID for this color
            color_name: Name of the color

        Returns:
            List of detection dictionaries
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(contour)

                bbox = [float(x), float(y), float(x + w), float(y + h)]

                detection = {
                    "bbox": bbox,
                    "class_id": class_id,
                    "color_name": color_name,
                    "area": area,
                }

                detections.append(detection)

        return detections

    @profile
    def run(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Process frame and detect colored objects.

        Args:
            frame: Input BGR image

        Returns:
            Tuple of:
                - List of detection dictionaries with keys:
                    - bbox: [x1, y1, x2, y2] as percentages (0-1) of resized content area dimensions
                    - class_id: Integer class identifier
                    - color_name: String name of detected color
                    - area: Contour area in letterboxed coordinates
                - Combined thresholded mask as BGR image (for visualization)
        """
        letterboxed_frame, (resized_width, resized_height), (pad_x, pad_y) = (
            self.letterbox_image(frame, self.target_size)
        )

        if self.blur_kernel_size > 0:
            letterboxed_frame = cv2.GaussianBlur(
                letterboxed_frame, (self.blur_kernel_size, self.blur_kernel_size), 0
            )

        hsv_frame = cv2.cvtColor(letterboxed_frame, cv2.COLOR_BGR2HSV)

        all_detections = []
        combined_mask = np.zeros((self.target_size, self.target_size), dtype=np.uint8)

        for color_range in self.color_ranges:
            mask = self.create_color_mask(
                hsv_frame, color_range["lower_hsv"], color_range["upper_hsv"]
            )

            if self.morphology_iterations > 0 and self.morphology_kernel_size > 0:
                mask = self.process_mask(mask)

            combined_mask = cv2.bitwise_or(combined_mask, mask)

            detections = self.extract_bounding_boxes(
                mask, color_range["class_id"], color_range["name"]
            )

            for detection in detections:
                # Convert bbox to 0-1 percentages of resized content area (like YOLO)
                # First subtract padding, then divide by resized dimensions
                x1, y1, x2, y2 = detection["bbox"]
                x1 = (x1 - pad_x) / resized_width
                y1 = (y1 - pad_y) / resized_height
                x2 = (x2 - pad_x) / resized_width
                y2 = (y2 - pad_y) / resized_height
                detection["bbox"] = [x1, y1, x2, y2]

            all_detections.extend(detections)

        thresholded_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)

        return all_detections, thresholded_bgr
