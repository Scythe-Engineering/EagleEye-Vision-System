#!/usr/bin/env python3
"""
Testing script for the PositionPredictor model on video files.

This script loads a video, processes each frame through the model,
makes predictions for AprilTag detections, and saves the annotated video using OpenCV.

The script uses hardcoded configuration variables instead of command line arguments.
Results are automatically saved to a video file with OpenCV annotations.
"""

import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

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


def preprocess_frame(
    frame: np.ndarray, target_size: Tuple[int, int] = (640, 640)
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Preprocess a video frame for the model.

    Args:
        frame (np.ndarray): Input video frame
        target_size (Tuple[int, int]): Target size for letterboxing

    Returns:
        Tuple of (original_frame, preprocessed_frame, resized_size)
    """
    original_frame = frame.copy()

    # Preprocess for model (letterbox and convert to greyscale)
    preprocessed_frame, resized_size = letterbox_image(
        original_frame, target_size, greyscale=True, return_resized_size=True
    )

    return original_frame, preprocessed_frame, resized_size


def predict_detections(
    model: nn.Module, preprocessed_frame: np.ndarray, device: torch.device
) -> torch.Tensor:
    """
    Run inference on the preprocessed frame and return raw grid outputs.

    Args:
        model: The PositionPredictor model
        preprocessed_frame: Preprocessed frame array
        device: Device to run inference on

    Returns:
        Model outputs tensor of shape (Gh, Gw, 4)
    """
    # Convert grayscale letterboxed frame to 3-channel RGB NHWC uint8
    frame_rgb = cv2.cvtColor(preprocessed_frame, cv2.COLOR_GRAY2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).to(torch.uint8)
    frame_tensor = frame_tensor.unsqueeze(0).to(device)  # (1, H, W, 3)

    with torch.no_grad():
        outputs = model(frame_tensor)  # (1, 4, Gh, Gw)

    return outputs.squeeze(0).permute(1, 2, 0).cpu()


def decode_grid_predictions(
    model: nn.Module,
    outputs: torch.Tensor,
    target_size: Tuple[int, int] = (640, 640),
    confidence_threshold: float = 0.5,
) -> list:
    """
    Decode grid outputs into pixel-space detections using model.decode.

    Args:
        model: The PositionPredictor model
        outputs: Tensor of shape (Gh, Gw, 4)
        target_size: Model input size (width, height)
        confidence_threshold: Minimum probability to keep a cell

    Returns:
        List of detection dicts with pixel coordinates
    """
    grid_h, grid_w, _ = outputs.shape
    w, h = target_size
    cell_w = w / grid_w
    cell_h = h / grid_h

    # Convert back to (4, Gh, Gw) for model.decode
    logits_chw = outputs.permute(2, 0, 1)
    decoded = model.decode(logits_chw, conf_threshold=confidence_threshold)

    detections = []
    for i, j, dx, dy, ds, score in decoded:
        size_px = float(torch.exp(torch.tensor(ds)) * cell_w)
        cx = (float(j) + dx) * cell_w
        cy = (float(i) + dy) * cell_h

        cx = float(max(0.0, min(w - 1.0, cx)))
        cy = float(max(0.0, min(h - 1.0, cy)))
        size_px = float(max(1.0, min(min(w, h), size_px)))

        detections.append(
            {
                "x": cx,
                "y": cy,
                "scale": size_px,
                "confidence": float(score),
                "grid_i": int(i),
                "grid_j": int(j),
            }
        )

    return detections


def annotate_frame(
    frame: np.ndarray, detections: list, frame_number: int
) -> np.ndarray:
    """
    Annotate a frame with detection results.

    Args:
        frame: Original frame array
        detections: List of detection dictionaries
        frame_number: Current frame number for display

    Returns:
        Annotated frame
    """
    annotated_frame = frame.copy()

    # Draw detections only (in green)
    for detection in detections:
        x, y, scale, confidence = (
            detection["x"],
            detection["y"],
            detection["scale"],
            detection["confidence"],
        )

        # Draw detection
        center = (int(x), int(y))
        half_size = int(scale / 2)

        # Draw bounding box
        cv2.rectangle(
            annotated_frame,
            (center[0] - half_size, center[1] - half_size),
            (center[0] + half_size, center[1] + half_size),
            (0, 255, 0),  # Green for predictions
            2,
        )

        # Draw very small center point (nearly invisible)
        cv2.circle(annotated_frame, center, 1, (0, 0, 255), -1)

        # Add confidence label only
        label = f"{confidence:.2f}"
        cv2.putText(
            annotated_frame,
            label,
            (center[0] - half_size, center[1] - half_size - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Add frame number
    cv2.putText(
        annotated_frame,
        f"Frame: {frame_number}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    return annotated_frame


def process_video(
    video_path: str,
    model: nn.Module,
    device: torch.device,
    output_path: str,
    confidence_threshold: float = 0.7,
    target_size: Tuple[int, int] = (640, 640),
):
    """
    Process a video file frame by frame and save annotated video.

    Args:
        video_path: Path to input video
        model: The PositionPredictor model
        device: Device to run inference on
        output_path: Path to save output video
        confidence_threshold: Minimum confidence for detections
        target_size: Model input size
    """
    print(f"Opening video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(
        f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames"
    )

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    frame_number = 0
    total_detections = 0

    print("Processing video frames...")

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1

            try:
                # Preprocess frame
                original_frame, preprocessed_frame, resized_size = preprocess_frame(
                    frame, target_size=target_size
                )

                # Run inference
                outputs = predict_detections(model, preprocessed_frame, device)

                # Decode grid outputs in model input space (640x640)
                detections_input = decode_grid_predictions(
                    model,
                    outputs,
                    target_size=target_size,
                    confidence_threshold=confidence_threshold,
                )

                # Map detections back to original frame coordinates accounting for letterbox
                w_in, h_in = target_size
                w_rs, h_rs = resized_size
                offset_x = (w_in - w_rs) // 2
                offset_y = (h_in - h_rs) // 2

                detections = []
                for det in detections_input:
                    x_resized = det["x"] - offset_x
                    y_resized = det["y"] - offset_y
                    scale_resized = det["scale"]

                    scale_factor = max(resized_size) / max(
                        original_frame.shape[1], original_frame.shape[0]
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

                total_detections += len(detections)

                # Annotate and write frame
                annotated_frame = annotate_frame(
                    original_frame, detections, frame_number
                )
                out.write(annotated_frame)

                pbar.set_postfix(detections=total_detections)

            except Exception as e:
                print(f"Error processing frame {frame_number}: {e}")
                # Write original frame if processing fails
                out.write(frame)

            pbar.update(1)

    # Cleanup
    cap.release()
    out.release()

    print("Video processing complete!")
    print(f"Total frames processed: {frame_number}")
    print(f"Total detections: {total_detections}")
    print(f"Output saved to: {output_path}")


def main():
    """Main function to run the video testing script."""
    # Configuration variables
    video_path = r"E:/Ceph-Mirror/Python-Files/Projects/FIRST-Note-Detection/src/main_operations/modules/apriltags/pre_processing/ai_accelleration/training/test_models/basic_test.mp4"
    model_path = "E:/Ceph-Mirror/Python-Files/Projects/FIRST-Note-Detection/src/main_operations/modules/apriltags/pre_processing/ai_accelleration/training/position_model.pth"
    confidence_threshold = 0.4
    device = "auto"
    output_path = f"E:/Ceph-Mirror/Python-Files/Projects/FIRST-Note-Detection/src/main_operations/modules/apriltags/pre_processing/ai_accelleration/training/test_models/{video_path.split('/')[-1].split('.')[0]}_detection_results.mp4"

    try:
        # Load model
        model, device = load_model(model_path, device=device)

        # Process video
        process_video(
            video_path=video_path,
            model=model,
            device=device,
            output_path=output_path,
            confidence_threshold=confidence_threshold,
        )

    except Exception as e:
        print(f"Error during video processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
