import json
from typing import Dict, List, Tuple

import cv2
import numpy as np


def rotate_image_and_annotations(
    image: np.ndarray, annotations: Dict, angle: int
) -> Tuple[np.ndarray, Dict]:
    """Rotate image and corresponding AprilTag annotations by specified angle.

    Args:
        image: Input image as numpy array
        annotations: Dictionary containing frame data and tags
        angle: Rotation angle in degrees (90, 180, 270)

    Returns:
        Tuple of (rotated_image, updated_annotations)
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    updated_annotations = annotations.copy()
    updated_annotations["rotation_applied"] = angle

    filtered_tags: List[Dict] = []

    for tag in updated_annotations["tags"]:
        corners = tag["corners"]
        rotated_corners = []

        for corner in corners:
            point = np.array([[corner["x"], corner["y"]]], dtype=np.float32)
            rotated_point = cv2.transform(point.reshape(1, -1, 2), rotation_matrix)[0][
                0
            ]
            rotated_corners.append(
                {"x": int(round(rotated_point[0])), "y": int(round(rotated_point[1]))}
            )

        all_inside = all(
            0 <= c["x"] < width and 0 <= c["y"] < height for c in rotated_corners
        )
        if all_inside:
            kept_tag = tag.copy()
            kept_tag["corners"] = rotated_corners
            filtered_tags.append(kept_tag)

    updated_annotations["tags"] = filtered_tags

    return rotated_image, updated_annotations


def apply_rotation_augmentations(
    image: np.ndarray, annotations: Dict, output_base_name: str, output_dir: str
) -> List[Tuple[str, str]]:
    """Apply all rotation augmentations (90, 180, 270 degrees).

    Args:
        image: Input image
        annotations: Original annotations
        output_base_name: Base name for output files (without extension)
        output_dir: Directory to save augmented data

    Returns:
        List of tuples (image_path, json_path) for augmented files
    """
    augmented_files = []

    for angle in [90, 180, 270]:
        rotated_image, rotated_annotations = rotate_image_and_annotations(
            image, annotations, angle
        )

        image_filename = f"{output_base_name}_rot{angle}.png"
        json_filename = f"{output_base_name}_rot{angle}.json"

        image_path = f"{output_dir}/{image_filename}"
        json_path = f"{output_dir}/{json_filename}"

        cv2.imwrite(image_path, rotated_image)

        with open(json_path, "w") as f:
            json.dump(rotated_annotations, f, indent=2)

        augmented_files.append((image_path, json_path))

    return augmented_files
