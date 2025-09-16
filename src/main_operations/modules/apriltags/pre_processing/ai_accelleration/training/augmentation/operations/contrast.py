import json
from typing import Dict, List, Tuple

import cv2
import numpy as np


def adjust_contrast(
    image: np.ndarray, annotations: Dict, contrast_factor: float
) -> Tuple[np.ndarray, Dict]:
    """Adjust image contrast and keep annotations unchanged.

    Args:
        image: Input image as numpy array
        annotations: Dictionary containing frame data and tags
        contrast_factor: Contrast adjustment factor (0.5 = lower contrast, 1.0 = original, 1.5 = higher contrast)

    Returns:
        Tuple of (adjusted_image, updated_annotations)
    """
    # Convert to float for contrast adjustment
    adjusted_image = image.astype(np.float32)

    # Calculate mean brightness
    mean_brightness = np.mean(adjusted_image)

    # Apply contrast adjustment
    adjusted_image = mean_brightness + contrast_factor * (
        adjusted_image - mean_brightness
    )

    # Clip values to valid range
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

    # Update annotations with contrast info
    updated_annotations = annotations.copy()
    updated_annotations["contrast_factor"] = contrast_factor

    return adjusted_image, updated_annotations


def apply_contrast_augmentations(
    image: np.ndarray,
    annotations: Dict,
    output_base_name: str,
    output_dir: str,
    factors: List[float] = None,
) -> List[Tuple[str, str]]:
    """Apply contrast augmentations with different intensity levels.

    Args:
        image: Input image
        annotations: Original annotations
        output_base_name: Base name for output files (without extension)
        output_dir: Directory to save augmented data
        factors: List of contrast factors to apply. If None, uses defaults [0.5, 0.7, 1.3, 1.5]

    Returns:
        List of tuples (image_path, json_path) for augmented files
    """
    augmented_files = []
    if factors is None:
        factors = [0.5, 0.7, 1.3, 1.5]

    for factor in factors:
        # Apply contrast adjustment
        adjusted_image, adjusted_annotations = adjust_contrast(
            image, annotations, factor
        )

        # Create output filenames
        factor_str = str(factor).replace(".", "_")
        image_filename = f"{output_base_name}_contrast{factor_str}.png"
        json_filename = f"{output_base_name}_contrast{factor_str}.json"

        image_path = f"{output_dir}/{image_filename}"
        json_path = f"{output_dir}/{json_filename}"

        # Save adjusted image
        cv2.imwrite(image_path, adjusted_image)

        # Save adjusted annotations
        with open(json_path, "w") as f:
            json.dump(adjusted_annotations, f, indent=2)

        augmented_files.append((image_path, json_path))

    return augmented_files
