from typing import Tuple, Union, overload

import cv2
import numpy as np
import math


@overload
def letterbox_image(
    img: np.ndarray,
    target_size: Tuple[int, int],
    greyscale: bool = True,
    return_resized_size: bool = False,
) -> np.ndarray: ...


@overload
def letterbox_image(
    img: np.ndarray,
    target_size: Tuple[int, int],
    greyscale: bool = True,
    return_resized_size: bool = True,
) -> Tuple[np.ndarray, Tuple[int, int]]: ...


def letterbox_image(
    img: np.ndarray,
    target_size: Tuple[int, int],
    greyscale: bool = True,
    return_resized_size: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Resize by a single power-of-two factor to the largest size that fits within the target, then pad to exact target.

    Args:
        img (np.ndarray): Input OpenCV image; greyscale (H, W) or BGR color (H, W, 3).
        target_size (Tuple[int, int]): Target size as (width, height).
        greyscale (bool): Whether to convert output to greyscale. Conversion happens after resizing for performance.
        return_resized_size (bool): If True, also return the resized inner (width, height).

    Returns:
        Union[np.ndarray, tuple[np.ndarray, tuple[int, int]]]: Letterboxed image and optionally the inner resized size.
    """
    input_height, input_width = img.shape[:2]
    target_width, target_height = target_size

    if target_width == input_width and target_height == input_height:
        base_img = img
        if greyscale and (img.ndim == 3 and img.shape[2] == 3):
            base_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if return_resized_size:
            return base_img, (input_width, input_height)
        return base_img

    ratio_to_fit = min(target_width / input_width, target_height / input_height)
    ratio_to_fit = max(ratio_to_fit, 1e-9)

    exponent = int(math.floor(math.log2(ratio_to_fit)))

    if exponent >= 0:
        new_width = input_width << exponent
        new_height = input_height << exponent
    else:
        shift = -exponent
        new_width = max(1, input_width >> shift)
        new_height = max(1, input_height >> shift)

    resized_img = cv2.resize(
        img, (new_width, new_height), interpolation=cv2.INTER_NEAREST
    )

    if greyscale and resized_img.ndim == 3 and resized_img.shape[2] == 3:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    resized_height, resized_width = resized_img.shape[:2]

    pad_x = (target_width - resized_width) // 2
    pad_y = (target_height - resized_height) // 2

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

    if return_resized_size:
        return output_img, (resized_width, resized_height)
    return output_img


class LetterboxTransform:
    """Torchvision transform for letterboxing PIL images to a target size."""

    def __init__(self, target_size: Tuple[int, int]):
        self.target_size = target_size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return letterbox_image(img, self.target_size, return_resized_size=False)


def calculate_crop_regions_from_grid(
    conf_grid_mask: np.ndarray, cell_w: int, cell_h: int, min_group_size: int = 2
) -> list:
    """
    Calculate the crop regions based on the connected components in the grid mask.

    Args:
        conf_grid_mask (np.ndarray): The binary mask of the grid.
        cell_w (int): Width of each cell in pixels.
        cell_h (int): Height of each cell in pixels.
        min_group_size (int): Minimum size of a group of cells to be considered a crop region.

    Returns:
        list: A list of tuples representing the crop regions (x0, y0, x1, y1).
    """
    # this works, it has been tested, ignore the type checker
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(  # type: ignore
        conf_grid_mask.astype(np.uint8),
        8,
        cv2.CV_32S,  # type: ignore
    )

    regions = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_group_size:
            x_min_cell = stats[i, cv2.CC_STAT_LEFT]
            y_min_cell = stats[i, cv2.CC_STAT_TOP]
            width_cells = stats[i, cv2.CC_STAT_WIDTH]
            height_cells = stats[i, cv2.CC_STAT_HEIGHT]

            x0_pixel = x_min_cell * cell_w
            y0_pixel = y_min_cell * cell_h
            x1_pixel = (x_min_cell + width_cells) * cell_w
            y1_pixel = (y_min_cell + height_cells) * cell_h

            regions.append((x0_pixel, y0_pixel, x1_pixel, y1_pixel))
    return regions
