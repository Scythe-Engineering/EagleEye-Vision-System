from typing import Tuple, Union
import numpy as np
import cv2
import math


def letterbox_image(
    img: np.ndarray,
    target_size: Tuple[int, int],
    power_two_scaling: bool = True,
    greyscale: bool = True,
    return_resized_size_and_padding: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]]:
    """
    Resize the image to the target size. If power_two_scaling is True, resize by a single power-of-two factor to the largest size that fits within the target, then pad to exact target.

    Args:
        img (np.ndarray): Input OpenCV image; greyscale (H, W) or BGR color (H, W, 3).
        target_size (Tuple[int, int]): Target size as (width, height).
        power_two_scaling (bool): Whether to scale the image to a power of two. (is faster to scale to a power of two)
        greyscale (bool): Whether to convert output to greyscale. Conversion happens after resizing for performance.
        return_resized_size (bool): If True, also return the resized inner (width, height).

    Returns:
        Union[np.ndarray, tuple[np.ndarray, tuple[int, int]]]: Letterboxed image and optionally the inner resized size.
    """
    input_height, input_width = img.shape[:2]
    target_width, target_height = target_size

    # Early return for images that already match target size
    if target_width == input_width and target_height == input_height:
        base_img = img
        if greyscale and (img.ndim == 3 and img.shape[2] == 3):
            base_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if return_resized_size_and_padding:
            return base_img, (input_width, input_height), (0, 0)
        return base_img

    # Calculate the scaling ratio to fit the image within target dimensions
    ratio_to_fit = min(target_width / input_width, target_height / input_height)
    ratio_to_fit = max(ratio_to_fit, 1e-9)

    # Apply power-of-two scaling if requested, otherwise use standard scaling
    if power_two_scaling:
        exponent = int(math.floor(math.log2(ratio_to_fit)))

        if exponent >= 0:
            new_width = input_width << exponent
            new_height = input_height << exponent
        else:
            shift = -exponent
            new_width = max(1, input_width >> shift)
            new_height = max(1, input_height >> shift)
    else:
        new_width = max(1, int(input_width * ratio_to_fit))
        new_height = max(1, int(input_height * ratio_to_fit))

    if input_height <= 0 or input_width <= 0:
        raise ValueError("Input image must have positive dimensions")

    try:
        resized_img = cv2.resize(
            img, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
    except cv2.error:
        resized_img = None

    if resized_img is None:
        row_indices = np.linspace(0, input_height - 1, new_height).astype(int)
        col_indices = np.linspace(0, input_width - 1, new_width).astype(int)
        resized_img = img[row_indices][:, col_indices]

    # Convert to greyscale if requested (after resizing for performance)
    if greyscale and resized_img.ndim == 3 and resized_img.shape[2] == 3:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    resized_height, resized_width = resized_img.shape[:2]

    # Calculate padding needed to center the resized image
    pad_x = (target_width - resized_width) // 2
    pad_y = (target_height - resized_height) // 2

    # Create padded output image and place resized image in center
    if greyscale or resized_img.ndim == 2:
        output_img = np.zeros((target_height, target_width), dtype=np.uint8)
        output_img[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = (
            resized_img
        )
    else:
        output_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        output_img[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width, :] = (
            resized_img
        )

    # Return the letterboxed image with optional metadata
    if return_resized_size_and_padding:
        return output_img, (resized_width, resized_height), (pad_x, pad_y)
    return output_img
