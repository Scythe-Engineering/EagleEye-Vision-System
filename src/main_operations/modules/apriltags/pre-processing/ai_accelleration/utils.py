import os
from typing import Tuple

import cv2
import numpy as np

# Shared constants
TARGET_WIDTH = 320
TARGET_HEIGHT = 320
CONF_THRESHOLD = 0.15
MIN_GROUP_SIZE = 2
SRC_PATH = os.path.dirname(__file__)
VIDEO_PATH = os.path.join(SRC_PATH, "test.mp4")
MODEL_PATH = os.path.join(SRC_PATH, "model.pth")
AI_EVAL_FRAME_INTERVAL = 5
GRID_WIDTH = 20
GRID_HEIGHT = 20


def letterbox_image(
    img: np.ndarray, target_size: Tuple[int, int], greyscale: bool = True, return_resized_size: bool = False
) -> np.ndarray:
    """
    Resize and letterbox a color image to a target size while maintaining aspect ratio. Input can be color or greyscale, output is always greyscale.

    Args:
        img: A numpy array (OpenCV image), color (3 channels) or greyscale (1 channel)
        target_size: Target (width, height) for the output image
        greyscale: Whether to convert the image to greyscale (default: True)
        return_resized_size: Whether to return the resized size (default: False)
    Returns:
        Resized, letterboxed and greyscale image
    """        
    ih, iw = img.shape[:2]
    w, h = target_size
    scale = min(w/iw, h/ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    if greyscale:
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            img_grey = resized
    
        new_img = np.zeros((h, w), dtype=np.uint8)
        new_img[(h-nh)//2:(h-nh)//2 + nh, (w-nw)//2:(w-nw)//2 + nw] = img_grey
    else:
        new_img = np.zeros((h, w, 3), dtype=np.uint8)
        new_img[(h-nh)//2:(h-nh)//2 + nh, (w-nw)//2:(w-nw)//2 + nw, :] = resized
        
    if return_resized_size:
        return new_img, (nw, nh)
    else:
        return new_img


class LetterboxTransform:
    """Torchvision transform for letterboxing PIL images to a target size."""

    def __init__(self, target_size: Tuple[int, int]):
        self.target_size = target_size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return letterbox_image(img, self.target_size)


def calculate_crop_regions_from_grid(
    conf_grid_mask: np.ndarray, cell_w: int, cell_h: int
) -> list:
    """
    Calculate the crop regions based on the connected components in the grid mask.

    Args:
        conf_grid_mask (np.ndarray): The binary mask of the grid.
        cell_w (int): Width of each cell in pixels.
        cell_h (int): Height of each cell in pixels.

    Returns:
        list: A list of tuples representing the crop regions (x0, y0, x1, y1).
    """
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        conf_grid_mask.astype(np.uint8), 8, cv2.CV_32S
    )

    regions = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= MIN_GROUP_SIZE:
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
