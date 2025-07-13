import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw
from pupil_apriltags import Detector
from tqdm import tqdm

from utils import TARGET_WIDTH, TARGET_HEIGHT, letterbox_image, GRID_WIDTH, GRID_HEIGHT


def confidence_to_color(conf, min_conf, max_conf):
    """Convert confidence to a color (green for high, red for low)."""
    normalized = (conf - min_conf) / (max_conf - min_conf + 1e-6)
    red = int((1 - normalized) * 255)
    green = int(normalized * 255)
    return (red, green, 0, 128)  # RGBA with transparency


def visualize_grid_on_image(img, grid):
    """
    Overlay a 10×10 grid on the image, display confidence values, and adjust hue based on confidence.
    """
    draw = ImageDraw.Draw(img)
    width, height = img.size
    cell_width = width // GRID_WIDTH
    cell_height = height // GRID_HEIGHT

    # Normalize confidence values for hue adjustment
    min_conf = min(min(row) for row in grid)
    max_conf = max(max(row) for row in grid)

    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            conf = grid[i][j]
            x0, y0 = j * cell_width, i * cell_height
            x1, y1 = x0 + cell_width, y0 + cell_height

            # Adjust hue of the image inside the grid cell
            cell = img.crop((x0, y0, x1, y1)).convert("RGBA")
            overlay = Image.new(
                "RGBA", cell.size, confidence_to_color(conf, min_conf, max_conf)
            )
            blended = Image.alpha_composite(cell, overlay)
            img.paste(blended, (x0, y0, x1, y1))

            # Draw grid lines and confidence value
            draw.rectangle([x0, y0, x1, y1], outline="white", width=1)
            draw.text((x0 + 5, y0 + 5), f"{conf:.2f}", fill="white")

    return img


def visualize_grid_on_image_cv2(frame, grid, tag_boxes):
    """
    Overlay a 10×10 grid on the image using OpenCV, display detected tag boxes,
    and slightly darken grid tiles without tags.
    """
    h, w, _ = frame.shape
    cell_width = w // GRID_WIDTH
    cell_height = h // GRID_HEIGHT

    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            x0, y0 = j * cell_width, i * cell_height
            x1, y1 = x0 + cell_width, y0 + cell_height

            # Check if the grid cell contains a tag
            if grid[i][j] == 0:
                # Slightly darken the tile by blending with a black overlay
                overlay = frame[y0:y1, x0:x1].copy()
                black_overlay = np.zeros_like(
                    overlay
                )  # Create a black overlay of the same size
                darkened = cv2.addWeighted(overlay, 0.7, black_overlay, 0.3, 0)
                frame[y0:y1, x0:x1] = darkened

            # Draw grid lines
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), thickness=1)

    # Draw detected tag boxes
    for box in tag_boxes:
        pts = box.astype(int)
        cv2.polylines(
            frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2
        )  # Green box

    return frame


def process_single_image(img_path: str, output_dir: str, frame_index: int, detector: Detector) -> bool:
    """Process a single image: detect tags, save outputs, and visualize.

    Args:
        img_path: Path to the image file.
        output_dir: Directory to save outputs.
        frame_index: Index of the frame for naming outputs.
        detector: AprilTag detector instance.

    Returns:
        True to continue processing, False to break early (if 'q' is pressed).
    """
    frame = cv2.imread(img_path)
    if frame is None:
        raise RuntimeError(f"Unable to read image: {img_path}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray)

    resized_frame, (nw, nh) = letterbox_image(frame, (TARGET_WIDTH, TARGET_HEIGHT), greyscale=False, return_resized_size=True)
    h, w = resized_frame.shape[:2]
    rows, cols = GRID_HEIGHT, GRID_WIDTH
    ch, cw = h // rows, w // cols
    tag_cells = set()
    tag_boxes = []

    buffer_percentage = 1
    scale_x = nw / frame.shape[1]
    scale_y = nh / frame.shape[0]
    pad_x = (w - nw) // 2
    pad_y = (h - nh) // 2

    for tag in tags:
        pts = tag.corners
        pts_resized = pts.copy()
        pts_resized[:, 0] = pts[:, 0] * scale_x + pad_x
        pts_resized[:, 1] = pts[:, 1] * scale_y + pad_y
        tag_boxes.append(pts_resized)

        min_x, min_y = pts_resized.min(axis=0)
        max_x, max_y = pts_resized.max(axis=0)
        tag_width = max_x - min_x
        tag_height = max_y - min_y
        min_x = max(0, min_x - buffer_percentage * tag_width)
        min_y = max(0, min_y - buffer_percentage * tag_height)
        max_x = min(w - 1, max_x + buffer_percentage * tag_width)
        max_y = min(h - 1, max_y + buffer_percentage * tag_height)
        sr, sc = int(min_y // ch), int(min_x // cw)
        er, ec = int(max_y // ch), int(max_x // cw)
        for r in range(sr, er + 1):
            for c in range(sc, ec + 1):
                tag_cells.add((r, c))

    fname = f"{frame_index:06d}"
    output_img_path = os.path.join(output_dir, fname + ".png")
    output_json_path = os.path.join(output_dir, fname + ".json")
    cv2.imwrite(output_img_path, resized_frame)
    grid = [
        [1 if (r, c) in tag_cells else 0 for c in range(cols)] for r in range(rows)
    ]
    with open(output_json_path, "w") as jf:
        json.dump({"grid": grid}, jf, indent=2)
    vis_frame = visualize_grid_on_image_cv2(resized_frame.copy(), grid, tag_boxes)
    cv2.imshow("Visualization", vis_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        return False
    return True


def detect_apriltags_in_images(
    input_dir: str, output_dir: str = "apriltags/ai_accelleration/training_data"
) -> None:
    """Process all images in a folder, detect AprilTags, and save raw frames + grid data."""
    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted(
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not image_files:
        raise RuntimeError(f"No image files found in directory: {input_dir}")
    detector = Detector(
        families="tag36h11",
        nthreads=32,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
    )
    pbar = tqdm(image_files, unit="image", desc="Processing images")
    for frame_index, image_file in enumerate(pbar):
        img_path = os.path.join(input_dir, image_file)
        if not process_single_image(img_path, output_dir, frame_index, detector):
            break
    pbar.close()
    cv2.destroyAllWindows()


def main() -> None:
    input_dir = "apriltags/ai_accelleration/input_data"
    detect_apriltags_in_images(input_dir)


if __name__ == "__main__":
    main()
