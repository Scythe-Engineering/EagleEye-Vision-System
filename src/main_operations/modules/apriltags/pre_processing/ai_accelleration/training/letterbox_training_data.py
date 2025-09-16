import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Add the project root to the Python path so we can import modules
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../..")
)
sys.path.insert(0, project_root)

from src.main_operations.modules.apriltags.pre_processing.ai_accelleration.utils import (
    letterbox_image,
)

target_width = 320
target_height = 320
target_size = (target_width, target_height)


class ThreadSafeCounter:
    """Thread-safe counter for tracking progress across multiple threads."""

    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def increment(self):
        with self.lock:
            self.value += 1

    def get_value(self):
        with self.lock:
            return self.value


def is_image_greyscale(image: np.ndarray) -> bool:
    """Determine if an image is greyscale.

    Args:
        image: Image array loaded by OpenCV.

    Returns:
        bool: True if the image is greyscale, False otherwise.
    """
    if image is None:
        return False
    if len(image.shape) == 2:
        return True
    if image.shape[2] == 1:
        return True
    blue_channel, green_channel, red_channel = cv2.split(image)
    return np.array_equal(blue_channel, green_channel) and np.array_equal(
        green_channel, red_channel
    )


def verify_and_letterbox_image(image_path: str) -> Tuple[bool, bool]:
    """Verify image size and greyscale, letterbox and/or convert as needed.

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple[bool, bool]: (was_letterboxed, was_greyscaled)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Unable to read image file: {image_path}")

    height, width = img.shape[:2]
    image_is_greyscale = is_image_greyscale(img)

    was_letterboxed = False
    was_greyscaled = False

    if width != target_width or height != target_height:
        letterboxed_img = letterbox_image(img, target_size, greyscale=True)
        was_letterboxed = True
        if not image_is_greyscale:
            was_greyscaled = True
        cv2.imwrite(image_path, letterboxed_img)
        return was_letterboxed, was_greyscaled

    if not image_is_greyscale:
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(image_path, grey_img)
        was_greyscaled = True

    return was_letterboxed, was_greyscaled


def process_single_image(
    image_path: str, counter: ThreadSafeCounter, results: Dict[str, List]
) -> None:
    """Process a single image file and update shared counters.

    Args:
        image_path: Path to the image file to process.
        counter: Thread-safe counter for progress tracking.
        results: Dictionary to store results (letterboxed and greyscaled counts).
    """
    try:
        was_letterboxed, was_greyscaled = verify_and_letterbox_image(image_path)
        if was_letterboxed:
            results["letterboxed"].append(True)
        if was_greyscaled:
            results["greyscaled"].append(True)
    except Exception as e:
        print(f"Error processing {os.path.basename(image_path)}: {e}")
    finally:
        counter.increment()


def update_progress_bar(
    progress_bar: tqdm,
    counter: ThreadSafeCounter,
    total: int,
    stop_event: threading.Event,
) -> None:
    """Monitor the counter and update the progress bar.

    Args:
        progress_bar: tqdm progress bar to update.
        counter: Thread-safe counter to monitor.
        total: Total number of items to process.
        stop_event: Event to signal when to stop monitoring.
    """
    last_count = 0
    while not stop_event.is_set():
        current_count = counter.get_value()
        if current_count > last_count:
            progress_bar.update(current_count - last_count)
            last_count = current_count
        if current_count >= total:
            break
        time.sleep(0.1)  # Small delay to avoid busy waiting

    # Ensure progress bar reaches 100%
    final_count = counter.get_value()
    if final_count > last_count:
        progress_bar.update(final_count - last_count)


def process_training_data_images(training_data_dir: str) -> Tuple[int, int, int]:
    """Process images to ensure correct size and greyscale output.

    Args:
        training_data_dir: Path to the training data directory containing images.

    Returns:
        Tuple[int, int, int]: (total_images, images_letterboxed, images_greyscaled)
    """
    if not os.path.exists(training_data_dir):
        raise RuntimeError(
            f"Training data directory does not exist: {training_data_dir}"
        )

    image_files = [f for f in os.listdir(training_data_dir) if f.endswith(".png")]
    total_images = len(image_files)

    if total_images == 0:
        return 0, 0, 0

    print(f"Processing {total_images} images in {training_data_dir}")

    # Shared data structures for thread communication
    counter = ThreadSafeCounter()
    results = {"letterboxed": [], "greyscaled": []}
    stop_event = threading.Event()

    # Start progress bar monitoring thread
    progress_bar = tqdm(
        total=total_images, unit="image", desc="Verifying size and greyscale"
    )
    progress_thread = threading.Thread(
        target=update_progress_bar,
        args=(progress_bar, counter, total_images, stop_event),
    )
    progress_thread.start()

    # Process images in parallel
    max_workers = min(8, total_images)  # Limit to 8 threads or number of images
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_single_image,
                os.path.join(training_data_dir, image_file),
                counter,
                results,
            )
            for image_file in image_files
        ]

        # Wait for all futures to complete
        for future in as_completed(futures):
            future.result()  # This will raise any exceptions that occurred

    # Stop the progress monitoring thread
    stop_event.set()
    progress_thread.join()
    progress_bar.close()

    images_letterboxed = len(results["letterboxed"])
    images_greyscaled = len(results["greyscaled"])

    print("\nImage processing complete!")
    print(f"Successfully letterboxed: {images_letterboxed} images")
    print(f"Converted to greyscale: {images_greyscaled} images")

    return total_images, images_letterboxed, images_greyscaled


def letterbox_coordinates(
    corners: List[Dict[str, float]],
    original_width: int,
    original_height: int,
    target_width: int,
    target_height: int,
) -> List[Dict[str, float]]:
    """Apply letterboxing transformation to AprilTag corner coordinates.

    Args:
        corners: List of corner dictionaries with 'x' and 'y' keys.
        original_width: Original image width.
        original_height: Original image height.
        target_width: Target letterboxed width.
        target_height: Target letterboxed height.

    Returns:
        List of transformed corner coordinates.
    """
    # Calculate scale factor (same as in letterbox_image function)
    scale = min(target_width / original_width, target_height / original_height)

    # Calculate new dimensions after scaling
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Calculate offsets to center the scaled image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Transform each corner coordinate
    transformed_corners = []
    for corner in corners:
        transformed_corner = {
            "x": int(corner["x"] * scale) + x_offset,
            "y": int(corner["y"] * scale) + y_offset,
        }
        transformed_corners.append(transformed_corner)

    return transformed_corners


def process_json_file(json_path: str, target_width: int, target_height: int) -> bool:
    """Process a single JSON file to letterbox its annotation coordinates and remove camera attribute.

    Args:
        json_path: Path to the JSON file to process.
        target_width: Target letterboxed width.
        target_height: Target letterboxed height.

    Returns:
        True if the file was processed, False if it was already letterboxed and skipped.
    """
    # Verify file exists before processing
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Read the original data
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {json_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error reading file {json_path}: {e}")

    # Store original dimensions for verification
    original_width = data["image_width"]
    original_height = data["image_height"]

    # Check if file is already letterboxed (dimensions match target)
    if original_width == target_width and original_height == target_height:
        return False  # Skip processing, file is already letterboxed

    # Update image dimensions in the JSON to match letterboxed size
    data["image_width"] = target_width
    data["image_height"] = target_height

    # Remove camera attribute if it exists
    if "camera" in data:
        del data["camera"]

    # Transform coordinates for each tag
    if len(data["tags"]) > 1:  # Only show progress bar if there are multiple tags
        for tag in tqdm(
            data["tags"],
            desc=f"Tags in {os.path.basename(json_path)}",
            unit="tag",
            leave=False,
        ):
            tag["corners"] = letterbox_coordinates(
                tag["corners"],
                original_width,
                original_height,
                target_width,
                target_height,
            )
    else:
        # Process single tag without progress bar
        for tag in data["tags"]:
            tag["corners"] = letterbox_coordinates(
                tag["corners"],
                original_width,
                original_height,
                target_width,
                target_height,
            )

    # Write the modified data back to the same file (overwrites original)
    try:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Error writing to file {json_path}: {e}")

    return True  # Successfully processed the file


def process_single_json(
    json_path: str,
    target_width: int,
    target_height: int,
    counter: ThreadSafeCounter,
    results: Dict[str, List],
    failed_files: List,
) -> None:
    """Process a single JSON file and update shared counters.

    Args:
        json_path: Path to the JSON file to process.
        target_width: Target letterboxed width.
        target_height: Target letterboxed height.
        counter: Thread-safe counter for progress tracking.
        results: Dictionary to store results (processed and skipped counts).
        failed_files: List to store failed file information.
    """
    json_file = os.path.basename(json_path)
    try:
        was_processed = process_json_file(json_path, target_width, target_height)
        if was_processed:
            results["processed"].append(True)
        else:
            results["skipped"].append(True)
    except Exception as e:
        failed_files.append((json_file, str(e)))
    finally:
        counter.increment()


def process_training_data_json(
    training_data_dir: str, target_width: int, target_height: int
) -> Tuple[int, int, int]:
    """Process all JSON files in the training data directory to letterbox coordinates and remove camera attributes.

    Args:
        training_data_dir: Path to the directory containing JSON annotation files.
        target_width: Target letterboxed width.
        target_height: Target letterboxed height.

    Returns:
        Tuple[int, int, int]: (total_files, processed_files, skipped_files)
    """
    if not os.path.exists(training_data_dir):
        raise FileNotFoundError(
            f"Training data directory not found: {training_data_dir}"
        )

    # Find all JSON files in the directory
    json_files = [f for f in os.listdir(training_data_dir) if f.endswith(".json")]

    if not json_files:
        print(f"No JSON files found in {training_data_dir}")
        return 0, 0, 0

    total_files = len(json_files)
    print(f"Found {total_files} JSON files to process")
    print(
        f"Letterboxing coordinates to {target_width}x{target_height} and removing camera attributes"
    )

    # Shared data structures for thread communication
    counter = ThreadSafeCounter()
    results = {"processed": [], "skipped": []}
    failed_files = []
    stop_event = threading.Event()

    # Start progress bar monitoring thread
    progress_bar = tqdm(total=total_files, desc="Processing JSON files", unit="file")
    progress_thread = threading.Thread(
        target=update_progress_bar,
        args=(progress_bar, counter, total_files, stop_event),
    )
    progress_thread.start()

    # Process JSON files in parallel
    max_workers = min(8, total_files)  # Limit to 8 threads or number of files
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_single_json,
                os.path.join(training_data_dir, json_file),
                target_width,
                target_height,
                counter,
                results,
                failed_files,
            )
            for json_file in json_files
        ]

        # Wait for all futures to complete
        for future in as_completed(futures):
            future.result()  # This will raise any exceptions that occurred

    # Stop the progress monitoring thread
    stop_event.set()
    progress_thread.join()
    progress_bar.close()

    processed_files = len(results["processed"])
    skipped_files = len(results["skipped"])

    # Summary
    print("\nJSON processing complete!")
    print(f"Successfully processed: {processed_files} files")
    if skipped_files > 0:
        print(f"Skipped (already letterboxed): {skipped_files} files")

    if failed_files:
        print(f"Failed to process: {len(failed_files)} files")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")

    return total_files, processed_files, skipped_files


def main() -> None:
    """Main function to process both images and JSON annotations."""
    training_data_dir = "E:/Ceph-Mirror/Python-Files/Projects/FIRST-Note-Detection/src/main_operations/modules/apriltags/pre_processing/ai_accelleration/training/training_data"

    print("Starting combined letterboxing process...")
    print("=" * 50)

    # Process images first
    print("\n1. Processing images...")
    try:
        total_images, letterboxed_images, greyscaled_images = (
            process_training_data_images(training_data_dir)
        )
        print("   Images processed successfully!")
    except Exception as e:
        print(f"   Error processing images: {e}")
        total_images = letterboxed_images = greyscaled_images = 0

    # Process JSON files
    print("\n2. Processing JSON annotations...")
    try:
        total_json, processed_json, skipped_json = process_training_data_json(
            training_data_dir, target_width, target_height
        )
        print("   JSON files processed successfully!")
    except Exception as e:
        print(f"   Error processing JSON files: {e}")
        total_json = processed_json = skipped_json = 0

    # Final summary
    print("\n" + "=" * 50)
    print("COMBINED PROCESSING COMPLETE!")
    print("=" * 50)
    print(
        f"Images: {total_images} total, {letterboxed_images} letterboxed, {greyscaled_images} greyscaled"
    )
    print(
        f"JSON files: {total_json} total, {processed_json} processed, {skipped_json} already letterboxed"
    )
    print("All images outputted as greyscale format")


if __name__ == "__main__":
    main()
