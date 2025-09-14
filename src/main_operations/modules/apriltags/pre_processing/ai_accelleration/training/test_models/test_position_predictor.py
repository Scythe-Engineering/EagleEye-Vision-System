#!/usr/bin/env python3
"""
Testing script for the PositionPredictor model.

This script loads an image, preprocesses it, loads a trained model (if available),
makes predictions for AprilTag detections, and saves the annotated results using OpenCV.

The script uses hardcoded configuration variables instead of command line arguments.
Results are automatically saved to a file with OpenCV annotations.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn

# Add the project root to the path to import modules
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from src.main_operations.modules.apriltags.pre_processing.ai_accelleration.position_detectors.predictor import (
    PositionPredictor,
)
from src.main_operations.modules.apriltags.pre_processing.ai_accelleration.utils import (
    letterbox_image,
)


def load_model(model_path: str, device: str = "auto") -> Tuple[nn.Module, torch.device]:
    """
    Load the trained PositionPredictor model.

    Args:
        model_path (str): Path to the model file (.pth or .pt)
        device (str): Device to load the model on ('auto', 'cpu', 'cuda')

    Returns:
        Tuple of (loaded model, device)
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")

    model = PositionPredictor()
    model = model.to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "", 1) if k.startswith("module.") else k
            cleaned_state_dict[new_key] = v
        model.load_state_dict(cleaned_state_dict, strict=False)
        model.eval()
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Warning: Model file not found at {model_path}")
        print("Using untrained model for demonstration purposes.")
        print("To use a trained model, train it first using train_position_model.py")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using untrained model for demonstration purposes.")

    return model, device


def preprocess_image(
    image_path: str, target_size: Tuple[int, int] = (320, 320)
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Load and preprocess an image for the model.

    Args:
        image_path (str): Path to the input image
        target_size (Tuple[int, int]): Target size for letterboxing

    Returns:
        Tuple of (original_image, preprocessed_image, resized_size)
    """
    print(f"Loading image from: {image_path}")

    # Load original image
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    print(f"Original image shape: {original_img.shape}")

    # Preprocess for model (letterbox and convert to greyscale)
    preprocessed_img, resized_size = letterbox_image(
        original_img, target_size, greyscale=True, return_resized_size=True
    )

    print(f"Preprocessed image shape: {preprocessed_img.shape}")
    print(f"Resized size: {resized_size}")

    return original_img, preprocessed_img, resized_size


def predict_detections(
    model: nn.Module, preprocessed_img: np.ndarray, device: torch.device
) -> torch.Tensor:
    """
    Run inference on the preprocessed image and return raw grid outputs.

    Args:
        model: The PositionPredictor model
        preprocessed_img: Preprocessed image array
        device: Device to run inference on

    Returns:
        Model outputs tensor of shape (Gh, Gw, 4)
    """
    img_tensor = torch.from_numpy(preprocessed_img).float()
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor)  # (1, Gh, Gw, 4)

    return outputs.squeeze(0).cpu()


def decode_grid_predictions(
    outputs: torch.Tensor,
    target_size: Tuple[int, int] = (320, 320),
    confidence_threshold: float = 0.5,
    top_k: int = 12,
) -> list:
    """
    Decode grid outputs into pixel-space detections.

    Args:
        outputs: Tensor of shape (Gh, Gw, 4)
        target_size: Model input size (width, height)
        confidence_threshold: Minimum probability to keep a cell
        top_k: Max detections to keep

    Returns:
        List of detection dicts with pixel coordinates
    """
    grid_h, grid_w, _ = outputs.shape
    w, h = target_size
    cell_w = w / grid_w
    cell_h = h / grid_h

    obj_logits = outputs[..., 0]
    dx_hat = outputs[..., 1]
    dy_hat = outputs[..., 2]
    ds_hat = outputs[..., 3]

    obj_probs = torch.sigmoid(obj_logits)

    # Thresholding
    mask = obj_probs > confidence_threshold
    if not torch.any(mask):
        return []

    ys, xs = torch.nonzero(mask, as_tuple=True)
    scores = obj_probs[ys, xs]

    # Top-k selection
    if scores.numel() > top_k:
        topk_scores, topk_idx = torch.topk(scores, top_k)
        ys = ys[topk_idx]
        xs = xs[topk_idx]
        scores = topk_scores

    detections = []
    for y_idx, x_idx, score in zip(ys.tolist(), xs.tolist(), scores.tolist()):
        dx = torch.sigmoid(dx_hat[y_idx, x_idx]).item()
        dy = torch.sigmoid(dy_hat[y_idx, x_idx]).item()
        size_px = (torch.exp(ds_hat[y_idx, x_idx]) * cell_w).item()

        cx = (x_idx + dx) * cell_w
        cy = (y_idx + dy) * cell_h

        # Clip to image bounds
        cx = float(max(0.0, min(w - 1.0, cx)))
        cy = float(max(0.0, min(h - 1.0, cy)))
        size_px = float(max(1.0, min(min(w, h), size_px)))

        detections.append(
            {
                "x": cx,
                "y": cy,
                "scale": size_px,
                "confidence": float(score),
                "grid_i": int(y_idx),
                "grid_j": int(x_idx),
            }
        )

    return detections


def load_ground_truth(json_path: str) -> list:
    """
    Load ground truth data from JSON file and sort tags top-to-bottom for consistency with training.

    Args:
        json_path: Path to the JSON file containing ground truth

    Returns:
        List of ground truth tag dictionaries with corner coordinates, sorted top-to-bottom
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        tags = data.get("tags", [])

        # Sort tags top-to-bottom for consistency with training data ordering
        if tags:
            img_height = data.get("image_height", 1)

            def calculate_center_y(tag):
                """Calculate normalized y-coordinate of tag center."""
                corners = tag.get("corners", [])
                if not corners:
                    return 0.0
                ys = [corner["y"] for corner in corners]
                center_y = (min(ys) + max(ys)) / 2
                return center_y / img_height

            tags.sort(key=calculate_center_y)

        return tags
    except FileNotFoundError:
        print(f"Warning: Ground truth file not found at {json_path}")
        return []
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return []


def visualize_detections(
    original_img: np.ndarray,
    detections: list,
    ground_truth: list = None,
    save_path: Optional[str] = None,
    show_image: bool = True,
):
    """
    Visualize detections and ground truth on the original image using OpenCV.

    Args:
        original_img: Original image array
        detections: List of detection dictionaries
        ground_truth: List of ground truth tag dictionaries
        save_path: Path to save the visualization
        show_image: Whether to display the image using cv2.imshow
    """
    img_copy = original_img.copy()

    print(f"\nFound {len(detections)} detections:")

    # Draw detections only (in green)
    for i, detection in enumerate(detections):
        x, y, scale, confidence = (
            detection["x"],
            detection["y"],
            detection["scale"],
            detection["confidence"],
        )

        print(
            f"Detection {i + 1}: Center=({x:.1f}, {y:.1f}), "
            f"Scale={scale:.1f}, Confidence={confidence:.3f}"
        )

        # Draw detection
        center = (int(x), int(y))
        half_size = int(scale / 2)

        # Draw bounding box
        cv2.rectangle(
            img_copy,
            (center[0] - half_size, center[1] - half_size),
            (center[0] + half_size, center[1] + half_size),
            (0, 255, 0),  # Green for predictions
            2,
        )

        # Draw very small center point (nearly invisible)
        cv2.circle(img_copy, center, 1, (0, 0, 255), -1)

        # Add confidence label only
        label = f"{confidence:.2f}"
        cv2.putText(
            img_copy,
            label,
            (center[0] - half_size, center[1] - half_size - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Save the annotated image
    if save_path:
        cv2.imwrite(save_path, img_copy)
        print(f"Visualization saved to: {save_path}")
    else:
        print("No save path provided, skipping file save")

    # Show the image using cv2.imshow
    if show_image:
        cv2.imshow("AprilTag Detection Results", img_copy)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main function to run the testing script."""
    # Configuration variables
    image_path = None
    model_path = "/home/eagle/EagleEye-Object-Detection/src/main_operations/modules/apriltags/pre_processing/ai_accelleration/training/position_model.pth"
    confidence_threshold = 0.7
    device = "auto"
    save_path = "/home/eagle/EagleEye-Object-Detection/src/main_operations/modules/apriltags/pre_processing/ai_accelleration/training/detection_results.png"

    # Set default image path if not provided
    if image_path is None:
        default_image = "/home/eagle/EagleEye-Object-Detection/src/main_operations/modules/apriltags/pre_processing/ai_accelleration/training/frame_0636.png"
        if os.path.exists(default_image):
            image_path = default_image
        else:
            print("No image path provided and default image not found.")
            print("Please provide a path to an image file.")
            return

    try:
        original_img, preprocessed_img, resized_size = preprocess_image(
            image_path, target_size=(320, 320)
        )

        model, device = load_model(model_path, device=device)

        print(
            f"preprocessed_img shape: {preprocessed_img.shape}, dtype: {preprocessed_img.dtype}, min: {preprocessed_img.min()}, max: {preprocessed_img.max()}"
        )
        print(
            f"preprocessed_img array:\n{np.array2string(preprocessed_img, threshold=100, max_line_width=120)}"
        )

        outputs = predict_detections(model, preprocessed_img, device)
        print(f"outputs: {outputs}")

        # Decode grid outputs in model input space (320x320)
        detections_input = decode_grid_predictions(
            outputs,
            target_size=(320, 320),
            confidence_threshold=confidence_threshold,
            top_k=12,
        )

        # Map detections back to original image coordinates accounting for letterbox
        w_in, h_in = 320, 320
        w_rs, h_rs = resized_size
        offset_x = (w_in - w_rs) // 2
        offset_y = (h_in - h_rs) // 2

        detections = []
        for det in detections_input:
            x_resized = det["x"] - offset_x
            y_resized = det["y"] - offset_y
            scale_resized = det["scale"]

            scale_factor = max(resized_size) / max(
                original_img.shape[1], original_img.shape[0]
            )
            x_original = x_resized / scale_factor
            y_original = y_resized / scale_factor
            scale_original = scale_resized / scale_factor

            detections.append(
                {
                    "x": x_original,
                    "y": y_original,
                    "scale": scale_original,
                    "confidence": det["confidence"],
                }
            )

        json_path = image_path.replace(".png", ".json")
        ground_truth = load_ground_truth(json_path)

        visualize_detections(
            original_img,
            detections,
            ground_truth=ground_truth,
            save_path=save_path,
            show_image=True,
        )

    except Exception as e:
        print(f"Error during testing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
