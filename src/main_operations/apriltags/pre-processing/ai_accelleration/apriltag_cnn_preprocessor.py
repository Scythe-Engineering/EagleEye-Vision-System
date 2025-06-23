import cv2
import numpy as np
import torch
from torchvision import transforms

from grid_detectors.predictor import GridPredictor
from utils import (
    TARGET_WIDTH,
    TARGET_HEIGHT,
    CONF_THRESHOLD,
    LetterboxTransform,
    letterbox_image,
    calculate_crop_regions_from_grid,
    GRID_WIDTH,
    GRID_HEIGHT,
)


class ApriltagCnnPreprocessor:
    """A class to handle AprilTag CNN preprocessing and inference.

    This class loads a trained model and performs inference on video frames to detect
    potential AprilTag locations using a grid-based prediction approach.

    Attributes:
        model_path: Path to the trained model weights.
        device: The computation device (CPU/CUDA).
        model: The loaded GridPredictor model.
        transform: Image transformation pipeline.
        conf_threshold: Confidence threshold for predictions.
    """

    def __init__(
        self, model_path: str, conf_threshold: float = CONF_THRESHOLD
    ) -> None:
        """Initialize the AprilTag CNN preprocessor.

        Args:
            model_path: Path to the trained model weights file.
            conf_threshold: Confidence threshold for predictions.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transform = transforms.Compose(
            [
                LetterboxTransform((TARGET_WIDTH, TARGET_HEIGHT)),
                transforms.ToTensor(),
            ]
        )
        
        self.last_probs = None

    def _load_model(self) -> GridPredictor:
        """Load and prepare the model for inference.

        Returns:
            The loaded and prepared GridPredictor model.

        Raises:
            FileNotFoundError: If the model file is not found.
            RuntimeError: If there's an error loading the model.
        """
        model = GridPredictor().to(self.device)
        try:
            model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            model.eval()
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def _overlay_grid_cv2(
        self, frame: np.ndarray, grid: np.ndarray
    ) -> np.ndarray:
        """Overlay the prediction grid and bounding boxes on the frame.

        Args:
            frame: Input frame to overlay predictions on.
            grid: Prediction grid probabilities.

        Returns:
            Frame with overlaid predictions and bounding boxes.
        """
        if len(frame.shape) == 2:
            frame_vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_vis = frame.copy()
        h, w = frame_vis.shape[:2]
        cell_w = w // GRID_WIDTH
        cell_h = h // GRID_HEIGHT

        overlay = frame_vis.copy()
        conf_grid_mask = grid >= self.conf_threshold

        for row in range(GRID_HEIGHT):
            for column in range(GRID_WIDTH):
                left = column * cell_w
                top = row * cell_h
                right = left + cell_w
                bottom = top + cell_h

                if not conf_grid_mask[row, column]:
                    cv2.rectangle(
                        overlay, (left, top), (right, bottom), (0, 0, 0), -1
                    )

                cv2.rectangle(
                    overlay, (left, top), (right, bottom), (255, 255, 255), 1
                )

        cv2.addWeighted(overlay, 1, frame_vis, 0, 0, frame_vis)

        crop_regions = calculate_crop_regions_from_grid(
            conf_grid_mask, cell_w, cell_h
        )
        for region in crop_regions:
            left, top, right, bottom = region
            cv2.rectangle(frame_vis, (left, top), (right, bottom), (0, 0, 255), 2)

        return frame_vis

    def process_frame(
        self, frame: np.ndarray, output_size: tuple[int, int] | None = None, return_visualization: bool = False
    ) -> tuple[list[tuple[int, int, int, int]], np.ndarray | None]:
        """Process a single frame through the model.

        Args:
            frame: Input frame to process.
            output_size: Optional output size for the regions.
            return_visualization: Whether to return the visualization frame.

        Returns:
            A tuple containing:
                - List of crop regions as (left, top, right, bottom)
                - Visualization frame if return_visualization is True, else None
        """
        scaled_frame = letterbox_image(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        tensor = self.transform(scaled_frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            self.last_probs = probs

        h, w = scaled_frame.shape[:2]
        cell_w = w // GRID_WIDTH
        cell_h = h // GRID_HEIGHT
        conf_grid_mask = probs >= self.conf_threshold
        crop_regions = calculate_crop_regions_from_grid(
            conf_grid_mask, cell_w, cell_h
        )

        if return_visualization:
            vis_frame = self._overlay_grid_cv2(scaled_frame, probs)
            if output_size is not None:
                crop_regions = [
                    (
                        int(region[0] * output_size[1] / TARGET_WIDTH),
                        int(region[1] * output_size[0] / TARGET_HEIGHT),
                        int(region[2] * output_size[1] / TARGET_WIDTH),
                        int(region[3] * output_size[0] / TARGET_HEIGHT),
                    )
                    for region in crop_regions
                ]
            return crop_regions, vis_frame
            
        if output_size is not None:
            crop_regions = [
                (
                    int(region[0] * output_size[1] / TARGET_WIDTH),
                    int(region[1] * output_size[0] / TARGET_HEIGHT),
                    int(region[2] * output_size[1] / TARGET_WIDTH),
                    int(region[3] * output_size[0] / TARGET_HEIGHT),
                )
                for region in crop_regions
            ]
        return crop_regions, None
    
    def direct_visualize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Overlay the prediction grid and bounding boxes on the frame.

        Args:
            frame: Input frame to overlay predictions on.
            probs: Prediction grid probabilities.
        """
        scaled_frame = letterbox_image(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        return self._overlay_grid_cv2(scaled_frame, self.last_probs)
