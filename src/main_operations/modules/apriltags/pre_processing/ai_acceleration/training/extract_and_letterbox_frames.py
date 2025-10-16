import os

import cv2
from tqdm import tqdm

from src.main_operations.modules.apriltags.pre_processing.ai_acceleration.utils import (
    letterbox_image,
)

target_width = 320
target_height = 320


def extract_frames_from_video(video_path: str, output_dir: str) -> None:
    """Extract all frames from a video and letterbox them.

    Args:
        video_path: Path to the video file.
        output_dir: Directory to save the letterboxed frames.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0

    with tqdm(
        total=total_frames, unit="frame", desc="Extracting frames"
    ) as progress_bar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Letterbox the frame
            letterboxed_frame = letterbox_image(
                frame, (target_width, target_height), greyscale=False
            )

            # Save the frame
            frame_filename = f"frame_{frame_index:04d}.png"
            output_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(output_path, letterboxed_frame)

            frame_index += 1
            progress_bar.update(1)

    cap.release()


def main() -> None:
    video_path = "E:/Ceph-Mirror/Python-Files/Projects/FIRST-Note-Detection/src/main_operations/modules/apriltags/pre_processing/ai_acceleration/training/0001-0750.mp4"  # Update this path as needed
    output_dir = "E:/Ceph-Mirror/Python-Files/Projects/FIRST-Note-Detection/src/main_operations/modules/apriltags/pre_processing/ai_acceleration/training/training_data"
    extract_frames_from_video(video_path, output_dir)


if __name__ == "__main__":
    main()
