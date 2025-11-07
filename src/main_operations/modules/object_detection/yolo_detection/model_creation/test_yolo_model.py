"""Test YOLO Model on Video with Detection Visualization."""

import os
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from ultralytics import YOLO

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def validate_file_path(file_path: str, file_type: str) -> bool:
    """Validate that a file path exists and is accessible.

    Args:
        file_path: Path to the file to validate.
        file_type: Type of file for error messaging (e.g., 'model', 'video').

    Returns:
        True if file exists, False otherwise.
    """
    if not file_path:
        print(f"Error: {file_type} path cannot be empty!")
        return False

    if not os.path.exists(file_path):
        print(f"Error: {file_type} file '{file_path}' does not exist!")
        return False

    return True


def generate_output_video_path(input_video_path: str) -> str:
    """Generate output video path in test_videos folder.

    Args:
        input_video_path: Path to the input video file.

    Returns:
        Output video path with '_detections' suffix as .mp4 in test_videos folder.
    """
    script_dir = Path(__file__).parent
    test_videos_dir = script_dir / "test_videos"
    test_videos_dir.mkdir(exist_ok=True)

    video_path_obj = Path(input_video_path)
    output_filename = f"{video_path_obj.stem}_detections.mp4"
    output_path = test_videos_dir / output_filename
    return str(output_path)


def main() -> None:
    """Run YOLO model inference on video and save output with detections."""
    print("=== EagleEye YOLO Model Testing on Video ===")

    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            print(f"✅ GPU available: {gpu_count} device(s)")
            print(f"   Using: {device_name}")
        else:
            print("❌ GPU not available - inference will use CPU (slower)")
    else:
        print("⚠️  PyTorch not available - cannot check GPU status")
        print("   Inference device will be auto-detected by Ultralytics")

    model_path = input("Enter YOLO model path (.pt file): ").strip()
    if not validate_file_path(model_path, "Model"):
        return

    video_path = input("Enter input video path: ").strip()
    if not validate_file_path(video_path, "Video"):
        return

    output_path_input = input(
        "Enter output video path (press Enter for auto-generated name): "
    ).strip()

    if output_path_input:
        output_video_path = output_path_input
        output_directory = os.path.dirname(output_video_path)
        if output_directory and not os.path.exists(output_directory):
            print(f"Error: Output directory '{output_directory}' does not exist!")
            return
    else:
        output_video_path = generate_output_video_path(video_path)

    confidence_input = input(
        "Enter confidence threshold (0.0-1.0, press Enter for default 0.25): "
    ).strip()
    confidence_threshold: Optional[float] = None
    if confidence_input:
        try:
            confidence_threshold = float(confidence_input)
            if not 0.0 <= confidence_threshold <= 1.0:
                print("Error: Confidence threshold must be between 0.0 and 1.0!")
                return
        except ValueError:
            print("Error: Confidence threshold must be a number!")
            return

    print("\nStarting video inference with:")
    print(f"  Model: {os.path.basename(model_path)}")
    print(f"  Input video: {os.path.basename(video_path)}")
    print(f"  Output video: {os.path.basename(output_video_path)}")
    if confidence_threshold is not None:
        print(f"  Confidence threshold: {confidence_threshold}")

    try:
        import cv2
    except ImportError:
        print("Error: OpenCV (cv2) is required for video processing!")
        return

    print("\nLoading model...")
    model = YOLO(model_path)

    print("Processing video...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open input video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Cannot create output video: {output_video_path}")
        cap.release()
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"   Processing {total_frames} frames...")

    with tqdm(
        total=total_frames, desc="Processing frames", unit="frame"
    ) as progress_bar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            predict_kwargs = (
                {"conf": confidence_threshold}
                if confidence_threshold is not None
                else {}
            )

            results = model.predict(frame, verbose=False, **predict_kwargs)

            annotated_frame = results[0].plot()

            out.write(annotated_frame)

            progress_bar.update(1)

    cap.release()
    out.release()

    print("\n✅ Video processing completed!")
    print(f"   Output saved to: {output_video_path}")
    print(f"   Processed {total_frames} frames")


if __name__ == "__main__":
    main()
