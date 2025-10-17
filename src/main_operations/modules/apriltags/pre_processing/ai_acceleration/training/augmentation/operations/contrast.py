import json
from copy import deepcopy
from pathlib import Path
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
    # Store original dtype for final conversion
    original_dtype = image.dtype

    # Convert to float32 for contrast adjustment
    adjusted_image = image.astype(np.float32)

    # Calculate per-channel means (average across height and width dimensions)
    # For grayscale: shape is (H, W), mean over all pixels
    # For RGB: shape is (H, W, 3), mean over (H, W) for each channel
    if adjusted_image.ndim == 3:
        # Color image: compute mean per channel
        channel_means = np.mean(adjusted_image, axis=(0, 1), keepdims=True)
        # Apply contrast adjustment per channel
        adjusted_image = channel_means + contrast_factor * (
            adjusted_image - channel_means
        )
    else:
        # Grayscale image: compute single global mean
        global_mean = np.mean(adjusted_image)
        adjusted_image = global_mean + contrast_factor * (adjusted_image - global_mean)

    # Clip values to valid range and convert back to original dtype
    if original_dtype == np.uint8:
        adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
    else:
        # For other dtypes, clip to the appropriate range
        adjusted_image = np.clip(
            adjusted_image, np.iinfo(original_dtype).min, np.iinfo(original_dtype).max
        ).astype(original_dtype)

    # Update annotations with contrast info
    updated_annotations = deepcopy(annotations)
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

        image_path = str(Path(output_dir) / image_filename)
        json_path = str(Path(output_dir) / json_filename)

        # Save adjusted image
        cv2.imwrite(image_path, adjusted_image)

        # Save adjusted annotations
        with open(json_path, "w") as f:
            json.dump(adjusted_annotations, f, indent=2)

        augmented_files.append((image_path, json_path))

    return augmented_files
