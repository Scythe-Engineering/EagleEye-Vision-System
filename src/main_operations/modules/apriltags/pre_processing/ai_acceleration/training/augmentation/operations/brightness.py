import copy
import json
from typing import Dict, List, Tuple

import cv2
import numpy as np


def adjust_brightness(
    image: np.ndarray, annotations: Dict, brightness_factor: float
) -> Tuple[np.ndarray, Dict]:
    """Adjust image brightness and record adjustment factor in annotations.

    Adjusts the brightness of the input image by multiplying pixel values by the brightness_factor.
    Tag data in annotations remains unchanged, but a 'brightness_factor' field is added to track
    the brightness adjustment applied.

    Args:
        image: Input image as numpy array
        annotations: Dictionary containing frame data and tags
        brightness_factor: Brightness adjustment factor (0.5 = darker, 1.0 = original, 1.5 = brighter)

    Returns:
        Tuple of (adjusted_image, updated_annotations)
    """
    # Record original dtype to preserve precision
    original_dtype = image.dtype

    # Convert to float32 for safe brightness adjustment
    adjusted_image = image.astype(np.float32) * brightness_factor

    # Determine appropriate clipping range based on original dtype
    if np.issubdtype(original_dtype, np.integer):
        # For integer types, use the full range of the dtype
        min_val, max_val = np.iinfo(original_dtype).min, np.iinfo(original_dtype).max
    else:
        # For float types, use [0.0, 1.0] as logical range
        min_val, max_val = 0.0, 1.0

    # Clip values to dtype-appropriate range
    adjusted_image = np.clip(adjusted_image, min_val, max_val)

    # Cast back to original dtype to preserve precision
    adjusted_image = adjusted_image.astype(original_dtype)

    # Update annotations with brightness info
    updated_annotations = copy.deepcopy(annotations)
    updated_annotations["brightness_factor"] = brightness_factor

    return adjusted_image, updated_annotations


def apply_brightness_augmentations(
    image: np.ndarray,
    annotations: Dict,
    output_base_name: str,
    output_dir: str,
    factors: List[float] = None,
) -> List[Tuple[str, str]]:
    """Apply brightness augmentations with different intensity levels.

    Args:
        image: Input image
        annotations: Original annotations
        output_base_name: Base name for output files (without extension)
        output_dir: Directory to save augmented data
        factors: List of brightness factors to apply. If None, uses defaults [0.5, 0.7, 1.3, 1.5]

    Returns:
        List of tuples (image_path, json_path) for augmented files
    """
    augmented_files = []
    if factors is None:
        factors = [0.5, 0.7, 1.3, 1.5]

    for factor in factors:
        # Apply brightness adjustment
        adjusted_image, adjusted_annotations = adjust_brightness(
            image, annotations, factor
        )

        # Create output filenames
        factor_str = str(factor).replace(".", "_")
        image_filename = f"{output_base_name}_bright{factor_str}.png"
        json_filename = f"{output_base_name}_bright{factor_str}.json"

        image_path = f"{output_dir}/{image_filename}"
        json_path = f"{output_dir}/{json_filename}"

        # Save adjusted image
        image_write_success = cv2.imwrite(image_path, adjusted_image)
        if not image_write_success:
            raise IOError(f"Failed to write image to {image_path}")

        # Save adjusted annotations
        with open(json_path, "w") as f:
            json.dump(adjusted_annotations, f, indent=2)

        augmented_files.append((image_path, json_path))

    return augmented_files
