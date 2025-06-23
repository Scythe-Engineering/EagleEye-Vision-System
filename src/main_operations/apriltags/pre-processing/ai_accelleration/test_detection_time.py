import time
import cv2
import numpy as np
from pupil_apriltags import Detector
from apriltag_cnn_preprocessor import ApriltagCnnPreprocessor
from utils import VIDEO_PATH, MODEL_PATH

def draw_apriltag_boxes(frame: np.ndarray, tags: list) -> np.ndarray:
    """Draw green boxes around detected AprilTags on the frame."""
    for tag in tags:
        pts = tag.corners.astype(int)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    return frame

def draw_cnn_regions(frame: np.ndarray, regions: list) -> np.ndarray:
    """Draw red rectangles for CNN-predicted regions on the frame."""
    for left, top, right, bottom in regions:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    return frame

def run_full_frame_detection(video_path: str) -> tuple[float, float]:
    """Run AprilTag detection on every frame and return (average detection time in ms, percent frames with detection). Displays detections."""
    cap = cv2.VideoCapture(video_path)
    detector = Detector(
        families="tag36h11",
        nthreads=8,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
    )
    frame_count = 0
    total_time = 0.0
    detection_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)
        elapsed = time.time() - start
        if frame_count > 0:
            total_time += elapsed
            if len(tags) > 0:
                detection_frames += 1
        vis_frame = frame.copy()
        vis_frame = draw_apriltag_boxes(vis_frame, tags)
        cv2.imshow("Full-frame AprilTag Detection", vis_frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()
    if frame_count > 1:
        avg_ms = (total_time / (frame_count - 1)) * 1000
        percent_detect = (detection_frames / (frame_count - 1)) * 100
        return avg_ms, percent_detect
    return 0.0, 0.0

def detect_tags_in_crops(
    frame: np.ndarray, crop_regions: list, detector: Detector
) -> list:
    """Detect AprilTags in the given crop regions of the frame.

    Args:
        frame: The video frame.
        crop_regions: List of (left, top, right, bottom) tuples.
        detector: AprilTag detector instance.

    Returns:
        List of detected tags with adjusted coordinates.
    """
    tags = []
    for left, top, right, bottom in crop_regions:
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            continue
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        detected = detector.detect(gray_crop)
        for tag in detected:
            tag_corners = tag.corners.copy()
            tag_corners[:, 0] += left
            tag_corners[:, 1] += top
            tag.corners = tag_corners
            tags.append(tag)
    return tags

def run_selective_detection(video_path: str, model_path: str) -> tuple[float, float]:
    """Run AprilTag detection only on regions predicted by the CNN and return (average detection time in ms, percent frames with detection). Displays detections and regions."""
    cap = cv2.VideoCapture(video_path)
    detector = Detector(
        families="tag36h11",
        nthreads=8,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
    )
    preprocessor = ApriltagCnnPreprocessor(model_path)
    frame_count = 0
    total_time = 0.0
    detection_frames = 0
    crop_regions = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cap_size = frame.shape[:2]
        start = time.time()
        if frame_count % 5 == 0:
            crop_regions, _ = preprocessor.process_frame(
                frame, output_size=cap_size, return_visualization=False
            )
        tags = detect_tags_in_crops(frame, crop_regions, detector)
        elapsed = time.time() - start
        if frame_count > 0:
            total_time += elapsed
            if len(tags) > 0:
                detection_frames += 1
        vis_frame = frame.copy()
        vis_frame = draw_cnn_regions(vis_frame, crop_regions)
        vis_frame = draw_apriltag_boxes(vis_frame, tags)
        cv2.imshow("Selective (CNN region) AprilTag Detection", vis_frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()
    if frame_count > 1:
        avg_ms = (total_time / (frame_count - 1)) * 1000
        percent_detect = (detection_frames / (frame_count - 1)) * 100
        return avg_ms, percent_detect
    return 0.0, 0.0

def main() -> None:
    print("Running full-frame AprilTag detection...")
    avg_full, percent_full = run_full_frame_detection(VIDEO_PATH)
    print(f"Average full-frame detection time: {avg_full:.2f} ms per frame")
    print(f"Percent frames with at least one detection: {percent_full:.2f}%")

    print("\nRunning selective (CNN region) AprilTag detection...")
    avg_selective, percent_selective = run_selective_detection(VIDEO_PATH, MODEL_PATH)
    print(f"Average selective detection time: {avg_selective:.2f} ms per frame")
    print(f"Percent frames with at least one detection: {percent_selective:.2f}%")
    speedup = avg_full / avg_selective if avg_selective > 0 else 0.0
    print(f"\nCNN region detection is {speedup:.2f}x faster than full-frame detection")

if __name__ == "__main__":
    main()
