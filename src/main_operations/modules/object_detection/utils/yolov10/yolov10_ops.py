import numpy as np
from src.main_operations.modules.object_detection.utils.letterbox import letterbox_image
from typing import Tuple, Union
import torch


class YoloV10:
    """
    A helper class to run YOLOv10 pre- and post-processing operations.

    This class handles preprocessing of input images (letterboxing, normalization)
    and postprocessing of model predictions (confidence filtering, box scaling).
    """

    def __init__(
        self,
        original_image_shape: Tuple[int, int, int] | None,
        input_shape: Tuple[int, int],
        max_det: int,
        conf_threshold: float,
    ):
        """
        Initialize the YOLOv10 processor.

        Args:
            original_image_shape: Shape of the original input image as (height, width, channels).
            input_shape: Target input shape for the model as (height, width).
            max_det: Maximum number of detections to return.
            conf_threshold: Confidence threshold used for filtering detections.
        """
        self.input_shape = input_shape
        self.max_det = max_det
        self.conf_threshold = conf_threshold

        self.padding = None
        self.resized_size = None

        self.original_image_shape = original_image_shape
        self.original_img_size = (
            original_image_shape[:2] if original_image_shape is not None else None
        )

    def preprocess(self, img):
        """
        Preprocess an image for YOLOv10 model inference.

        Applies letterboxing to resize the image while maintaining aspect ratio,
        normalizes pixel values, and formats the tensor for model input.

        Args:
            img: Input image as numpy array with shape (height, width, channels).

        Returns:
            Preprocessed image tensor with shape (1, channels, height, width).
        """
        if self.original_image_shape is None:
            self.original_image_shape = img.shape
            self.original_img_size = self.original_image_shape[:2]

        img, resized_size, padding = letterbox_image(
            img,
            self.input_shape[:2],
            power_two_scaling=False,
            greyscale=False,
            return_resized_size_and_padding=True,
        )
        self.padding = padding
        self.resized_size = resized_size

        img = img.astype(np.float32)
        img /= 255.0  # Scale

        # Update input shape to what the original ONNX model expects ie B,C,H,W
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img

    def scale_boxes_to_original(
        self, boxes: np.ndarray
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Scale boxes from letterboxed coordinate space to original image coordinate space.

        This accounts for both padding and resizing transformations applied during preprocessing.
        The resized size is assumed to be the model input size.
        Boxes are expected to be in xyxy format (x1, y1, x2, y2).

        Args:
            boxes: Boxes in letterboxed coordinate space, shape (N, 4) or (..., 4).
                  Format: xyxy (x1, y1, x2, y2).

        Returns:
            Scaled boxes in original image coordinate space, same shape and type as input.
        """
        pad_x, pad_y = self.padding
        resized_width, resized_height = self.resized_size
        original_width, original_height = self.original_image_shape[:2]

        scale_x = original_width / resized_width
        scale_y = original_height / resized_height

        boxes -= np.array([pad_x, pad_y, pad_x, pad_y])
        boxes *= np.array([scale_x, scale_y, scale_x, scale_y])
        return boxes

    def postprocess(self, preds):
        """
        Postprocess YOLOv10 model predictions.

        Filters predictions by confidence threshold, scales bounding boxes back to
        original image coordinates, and returns the processed detections.

        Args:
            preds: Model predictions. Can be a dict with 'one2one' key, a list/tuple,
                  or a numpy array. Expected shape is (N, 6) where columns are
                  [x1, y1, x2, y2, confidence, class_id].

        Returns:
            Filtered and scaled predictions with bounding boxes in original image
            coordinate space. Same type as input preds.
        """
        if isinstance(preds, dict):
            preds = preds["one2one"]

        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        mask = preds[..., 4] > self.conf_threshold
        preds = preds[mask]

        if len(preds) > 0:
            boxes = preds[..., :4]
            scaled_boxes = self.scale_boxes_to_original(boxes)
            preds[..., :4] = scaled_boxes

        results = []
        for box, score, class_id in zip(preds[..., :4], preds[..., 4], preds[..., 5]):
            results.append(
                {
                    "bbox": box,
                    "score": score,
                    "class_id": class_id,
                }
            )

        return results
