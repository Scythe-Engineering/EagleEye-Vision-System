from time import sleep
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.main_operations.modules.apriltags.utils.apriltag import Apriltag


class TemporalAcceleration:
    """Predicts regions of interest based on last estimated camera pose.

    This class receives back-propagated camera poses (4x4 camera-to-world transforms)
    and projects known AprilTag corners into the image using the last measured pose
    to generate crop regions that accelerate subsequent detection.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion_coefficients: np.ndarray,
        apriltag_map: Dict[int, Apriltag],
        padding_factor: float = 0.35,
        max_regions: int = 20,
        min_region_size_px: int = 16,
        max_region_size_px: Optional[int] = None,
    ) -> None:
        """Initialize the temporal acceleration preprocessor.

        Args:
            camera_matrix: Camera intrinsic matrix.
            distortion_coefficients: Distortion coefficients for the camera.
            apriltag_map: Mapping of AprilTag id to Apriltag metadata.
            padding_factor: Fractional padding applied to ROI size.
            max_regions: Maximum number of ROIs to return.
            min_region_size_px: Minimum side length for ROI squares.
            max_region_size_px: Optional maximum side length for ROI squares.
        """
        self.camera_matrix = camera_matrix.astype(np.float32, copy=False)
        dist = distortion_coefficients.astype(np.float32, copy=False)
        if dist.ndim == 1 or (dist.ndim == 2 and dist.shape[0] == 1):
            dist = dist.reshape((-1, 1))
        self.distortion_coefficients = dist
        self.apriltag_map = apriltag_map

        self.padding_factor = float(padding_factor)
        self.max_regions = int(max_regions)
        self.min_region_size_px = int(min_region_size_px)
        self.max_region_size_px = (
            int(max_region_size_px) if max_region_size_px is not None else None
        )

        self._last_pose_world_from_camera: Optional[np.ndarray] = None

    def back_propagate_input(self, input_transform: np.ndarray) -> None:
        """Receive the latest camera-to-world transform at the end of a run.

        Args:
            input_transform: 4x4 transform mapping camera to world coordinates.
        """
        if (
            isinstance(input_transform, np.ndarray)
            and input_transform.shape == (4, 4)
            and np.isfinite(input_transform).all()
        ):
            self._last_pose_world_from_camera = input_transform.astype(
                np.float32, copy=False
            )

    def _predict_pose_world_from_camera(self) -> Optional[np.ndarray]:
        """Return the last measured camera-to-world pose.

        Returns:
            Last measured 4x4 camera-to-world transform, or None if no measurement available.
        """
        if self._last_pose_world_from_camera is None:
            return None
        return self._last_pose_world_from_camera.copy()

    def _invert_se3(self, transform: np.ndarray) -> np.ndarray:
        """Compute inverse of a 4x4 SE3 transform.

        Args:
            transform: 4x4 transform matrix.

        Returns:
            4x4 inverse transform matrix.
        """
        R = transform[:3, :3]
        t = transform[:3, 3]
        R_inv = R.T
        t_inv = -R_inv @ t
        T_inv = np.eye(4, dtype=transform.dtype)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv
        return T_inv

    def _frustum_cull(
        self,
        world_to_camera: np.ndarray,
        corners_world: np.ndarray,
        width: int,
        height: int,
    ) -> bool:
        """Check if tag corners are within camera frustum.

        Args:
            world_to_camera: 4x4 transform mapping world to camera coordinates.
            corners_world: Array of shape (4, 3) with tag corner positions.
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            True if tag should be kept (at least one corner in frustum), False otherwise.
        """
        R_wc = world_to_camera[:3, :3]
        t_wc = world_to_camera[:3, 3]
        corners_camera = (R_wc @ corners_world.T).T + t_wc

        z_values = corners_camera[:, 2]
        min_depth = 0.01
        if np.all(z_values < min_depth):
            return False

        fx = float(self.camera_matrix[0, 0])
        fy = float(self.camera_matrix[1, 1])

        margin_factor = 0.5
        fov_x_half = np.arctan(((width * 0.5) * (1.0 + margin_factor)) / fx)
        fov_y_half = np.arctan(((height * 0.5) * (1.0 + margin_factor)) / fy)

        valid_corners = corners_camera[z_values > min_depth]
        if len(valid_corners) == 0:
            return False

        angles_x = np.arctan2(np.abs(valid_corners[:, 0]), valid_corners[:, 2])
        angles_y = np.arctan2(np.abs(valid_corners[:, 1]), valid_corners[:, 2])

        in_fov_x = angles_x < fov_x_half
        in_fov_y = angles_y < fov_y_half

        return bool(np.any(in_fov_x & in_fov_y))

    def _project_tag_corners(
        self, world_to_camera: np.ndarray, corners_world: np.ndarray
    ) -> Optional[np.ndarray]:
        """Project 4x3 world-space corners into image space.

        Args:
            world_to_camera: 4x4 transform mapping world to camera coordinates.
            corners_world: Array of shape (4, 3) with tag corner positions.

        Returns:
            Array of shape (4, 2) with image pixel coordinates, or None if invalid.
        """
        R_wc = world_to_camera[:3, :3].astype(np.float32, copy=False)
        t_wc = world_to_camera[:3, 3].reshape(3, 1).astype(np.float32, copy=False)
        rvec, _ = cv2.Rodrigues(R_wc)
        img_pts, _ = cv2.projectPoints(
            corners_world.astype(np.float32, copy=False),
            rvec,
            t_wc,
            self.camera_matrix,
            self.distortion_coefficients,
        )
        img_pts = img_pts.reshape(-1, 2)
        return img_pts

    def _bbox_from_points(
        self, points: np.ndarray, width: int, height: int
    ) -> Tuple[int, int, int, int]:
        """Compute padded square ROI bounding box from 2D points.

        Args:
            points: Array of shape (N, 2) with pixel coordinates.
            width: Image width.
            height: Image height.

        Returns:
            Tuple describing (left, top, right, bottom) within image bounds.
        """
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        cx = float((min_xy[0] + max_xy[0]) * 0.5)
        cy = float((min_xy[1] + max_xy[1]) * 0.5)
        size = float(max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]))
        size *= 1.0 + self.padding_factor
        size = max(size, float(self.min_region_size_px))
        if self.max_region_size_px is not None:
            size = min(size, float(self.max_region_size_px))
        half = size * 0.5
        left = max(0, int(cx - half))
        top = max(0, int(cy - half))
        right = min(width, int(cx + half))
        bottom = min(height, int(cy + half))
        return left, top, right, bottom

    def _generate_crops_from_boxes(
        self, frame: np.ndarray, boxes: List[Tuple[int, int, int, int]]
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[int, int, int, int]]]:
        """Crop image regions and attach offsets.

        Args:
            frame: Input frame BGR.
            boxes: List of ROI boxes (l, t, r, b).

        Returns:
            Tuple of (cropped_images_with_offsets, crop_regions).
        """
        cropped_images: List[Tuple[np.ndarray, np.ndarray]] = []
        crop_regions: List[Tuple[int, int, int, int]] = []
        for l, t, r, b in boxes:
            if r <= l or b <= t:
                continue
            cropped_images.append((frame[t:b, l:r], (l, t)))
            crop_regions.append((l, t, r, b))
        return cropped_images, crop_regions

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[int, int, int, int]]]:
        """Generate ROIs and cropped images based on last pose projection only.

        Args:
            frame: Current frame for which to generate ROIs.

        Returns:
            Tuple of (cropped_images_with_offsets, crop_regions). If no pose
            measurement is available, returns the entire frame as a single crop.
        """
        height, width = frame.shape[:2]

        T_pred_world_from_camera = self._predict_pose_world_from_camera()
        if T_pred_world_from_camera is None:
            full_region = (0, 0, width, height)
            return [(frame, (0, 0))], [full_region]

        T_world_to_camera = self._invert_se3(T_pred_world_from_camera)

        boxes: List[Tuple[int, int, int, int]] = []
        R_wc = T_world_to_camera[:3, :3]
        t_wc = T_world_to_camera[:3, 3]
        for apriltag in self.apriltag_map.values():
            corners_world = apriltag.global_corners

            # Compute tag facing using world-space corner winding, then rotate to camera space
            edge_one_world = corners_world[1] - corners_world[0]
            edge_two_world = corners_world[2] - corners_world[0]
            normal_world = np.cross(edge_one_world, edge_two_world)
            if not np.isfinite(normal_world).all():
                continue
            normal_camera = R_wc @ normal_world
            if float(normal_camera[2]) >= 0.0:
                continue

            # Depth and frustum sanity checks using tag center and corners
            center_world = apriltag.global_center
            if not np.isfinite(center_world).all():
                continue
            center_camera = R_wc @ center_world + t_wc
            if float(center_camera[2]) <= 0.01:
                continue
            if not self._frustum_cull(T_world_to_camera, corners_world, width, height):
                continue

            img_pts = self._project_tag_corners(T_world_to_camera, corners_world)
            if img_pts is None:
                continue
            if not np.isfinite(img_pts).all():
                continue

            box = self._bbox_from_points(img_pts, width, height)
            boxes.append(box)

        boxes = boxes[: self.max_regions]

        if not boxes:
            full_region = (0, 0, width, height)
            return [(frame, (0, 0))], [full_region]

        cropped_images, crop_regions = self._generate_crops_from_boxes(frame, boxes)
        return cropped_images, crop_regions
