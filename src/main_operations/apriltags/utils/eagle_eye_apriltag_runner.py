import json
import os
import time

import cv2
import numpy as np
from pupil_apriltags import Detector

from src.main_operations.apriltags.utils.fmap_parser import load_fmap_file
from src.webui.web_server import EagleEyeInterface

VIDEO_PATH = "E:/Ceph-Mirror/Python-Files/Projects/FIRST-Note-Detection/src/apriltags/pre-processing/ai_accelleration/test.mp4"  # Set your video file path here


def load_camera_parameters(json_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load camera matrix and distortion coefficients from a JSON file.

    Args:
        json_path (str): Path to the camera intrinsics JSON file.

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera matrix and distortion coefficients.
    """
    with open(json_path, "r") as file:
        data = json.load(file)
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    distortion_coefficients = np.array(
        data["distortion_coefficients"], dtype=np.float64
    )
    return camera_matrix, distortion_coefficients


def main() -> None:
    """Run EagleEyeInterface and full-frame Apriltag detection on a video, updating sphere position from detections."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    camera_intrinsics_path = "E:/Ceph-Mirror/Python-Files/Projects/FIRST-Note-Detection/src/utils/camera_calibration/camera_intrinsics.json"
    fmap_path = os.path.join(script_dir, "frc2025r2.json")
    video_path = (
        os.path.join(script_dir, VIDEO_PATH)
        if not os.path.isabs(VIDEO_PATH)
        else VIDEO_PATH
    )

    camera_matrix, distortion_coefficients = load_camera_parameters(
        camera_intrinsics_path
    )
    apriltag_map = load_fmap_file(fmap_path)

    eagle_eye_interface = EagleEyeInterface(dev_mode=False)

    detector = Detector(families="tag36h11")

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    frame_interval = 1.0 / fps

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            start_time = time.time()

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_tags = detector.detect(gray_frame)

            display_frame = frame.copy()

            # Collect all image and object points from detected tags
            image_points_list = []
            object_points_list = []
            valid_tags_found = False

            for tag in detected_tags:
                tag_center = np.mean(tag.corners, axis=0).astype(int)
                cv2.circle(display_frame, tuple(tag_center), 10, (0, 255, 0), -1)
                cv2.putText(
                    display_frame,
                    f"ID: {tag.tag_id}",
                    (tag_center[0] + 15, tag_center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                tag_id = tag.tag_id
                if tag_id not in apriltag_map:
                    continue

                image_points_list.extend(tag.corners.astype(np.float32))

                apriltag_obj = apriltag_map[tag_id]
                object_points_list.extend(
                    apriltag_obj.global_corners.astype(np.float32)
                )
                valid_tags_found = True

            camera_transform = None
            if valid_tags_found and len(image_points_list) >= 4:
                image_points = np.array(image_points_list, dtype=np.float32).reshape(
                    -1, 2
                )
                object_points = np.array(object_points_list, dtype=np.float32).reshape(
                    -1, 3
                )

                success, rotation_vector, translation_vector = cv2.solvePnP(
                    object_points, image_points, camera_matrix, distortion_coefficients
                )

                if success:
                    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

                    camera_space_transform = np.eye(4)
                    camera_space_transform[:3, :3] = rotation_matrix
                    camera_space_transform[:3, 3] = translation_vector.flatten()

                    global_camera_transform = np.linalg.inv(camera_space_transform)
                    camera_transform = global_camera_transform.copy()

            display_frame = cv2.resize(display_frame, (0, 0), fx=0.5, fy=0.5)

            cv2.imshow("AprilTag Detection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if camera_transform is not None:
                eagle_eye_interface.update_robot_position(camera_transform)

            elapsed = time.time() - start_time
            sleep_time = max(0.0, frame_interval - elapsed)
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("Exiting EagleEye Apriltag runner.")
    finally:
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
