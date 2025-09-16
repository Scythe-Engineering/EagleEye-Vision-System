import json
from typing import Dict, List, Tuple

import cv2
import numpy as np


def scale_image_and_annotations(
    image: np.ndarray, annotations: Dict, scale_factor: float
) -> Tuple[np.ndarray, Dict]:
    """Scale image and adjust AprilTag annotations accordingly.

    Args:
        image: Input image as numpy array
        annotations: Dictionary containing frame data and tags
        scale_factor: Scaling factor (0.8 = zoom in/crop, 1.2 = zoom out/pad with black)

    Returns:
        Tuple of (scaled_image, updated_annotations)
    """
    height, width = image.shape[:2]

    if scale_factor < 1.0:
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        x1 = (width - new_width) // 2
        y1 = (height - new_height) // 2
        x2 = x1 + new_width
        y2 = y1 + new_height

        scaled_image = image[y1:y2, x1:x2]

        scaled_image = cv2.resize(
            scaled_image, (width, height), interpolation=cv2.INTER_LINEAR
        )

        updated_annotations = annotations.copy()
        updated_annotations["scale_factor"] = scale_factor
        updated_annotations["scale_operation"] = "crop"

        scale_x = width / new_width
        scale_y = height / new_height
        translate_x = -x1
        translate_y = -y1

        filtered_tags: List[Dict] = []
        for tag in updated_annotations["tags"]:
            corners = tag["corners"]
            scaled_corners = []

            for corner in corners:
                new_x = (corner["x"] + translate_x) * scale_x
                new_y = (corner["y"] + translate_y) * scale_y
                scaled_corners.append({"x": int(round(new_x)), "y": int(round(new_y))})

            all_inside = all(
                0 <= c["x"] < width and 0 <= c["y"] < height for c in scaled_corners
            )
            if all_inside:
                kept_tag = tag.copy()
                kept_tag["corners"] = scaled_corners
                filtered_tags.append(kept_tag)

        updated_annotations["tags"] = filtered_tags

    else:
        down_width = max(1, int(round(width / scale_factor)))
        down_height = max(1, int(round(height / scale_factor)))

        resized_image = cv2.resize(
            image, (down_width, down_height), interpolation=cv2.INTER_LINEAR
        )

        scaled_image = np.zeros((height, width, 3), dtype=np.uint8)

        x_offset = (width - down_width) // 2
        y_offset = (height - down_height) // 2
        scaled_image[
            y_offset : y_offset + down_height, x_offset : x_offset + down_width
        ] = resized_image

        updated_annotations = annotations.copy()
        updated_annotations["scale_factor"] = scale_factor
        updated_annotations["scale_operation"] = "pad"

        inv_scale = 1.0 / scale_factor

        filtered_tags: List[Dict] = []
        for tag in updated_annotations["tags"]:
            corners = tag["corners"]
            scaled_corners = []

            for corner in corners:
                new_x = corner["x"] * inv_scale + x_offset
                new_y = corner["y"] * inv_scale + y_offset
                scaled_corners.append({"x": int(round(new_x)), "y": int(round(new_y))})

            all_inside = all(
                0 <= c["x"] < width and 0 <= c["y"] < height for c in scaled_corners
            )
            if all_inside:
                kept_tag = tag.copy()
                kept_tag["corners"] = scaled_corners
                filtered_tags.append(kept_tag)

        updated_annotations["tags"] = filtered_tags

    return scaled_image, updated_annotations


def apply_scale_augmentations(
    image: np.ndarray,
    annotations: Dict,
    output_base_name: str,
    output_dir: str,
    factors: List[float] = None,
) -> List[Tuple[str, str]]:
    """Apply scale augmentations (zoom in and zoom out).

    Args:
        image: Input image
        annotations: Original annotations
        output_base_name: Base name for output files (without extension)
        output_dir: Directory to save augmented data
        factors: List of scale factors to apply. If None, uses defaults [0.5, 0.7, 1.3, 1.5]

    Returns:
        List of tuples (image_path, json_path) for augmented files
    """
    augmented_files = []
    if factors is None:
        factors = [0.5, 0.7, 1.3, 1.5]

    for factor in factors:
        scaled_image, scaled_annotations = scale_image_and_annotations(
            image, annotations, factor
        )

        factor_str = str(factor).replace(".", "_")
        image_filename = f"{output_base_name}_scale{factor_str}.png"
        json_filename = f"{output_base_name}_scale{factor_str}.json"

        image_path = f"{output_dir}/{image_filename}"
        json_path = f"{output_dir}/{json_filename}"

        cv2.imwrite(image_path, scaled_image)

        with open(json_path, "w") as f:
            json.dump(scaled_annotations, f, indent=2)

        augmented_files.append((image_path, json_path))

    return augmented_files
