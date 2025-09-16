import glob
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

# Import augmentation operations
try:
    # Try relative imports first (when run as part of package)
    from .operations.brightness import apply_brightness_augmentations
    from .operations.contrast import apply_contrast_augmentations
    from .operations.rotate import apply_rotation_augmentations
    from .operations.scale import apply_scale_augmentations
except ImportError:
    # Fall back to absolute imports (when run as standalone script)
    import sys
    from pathlib import Path

    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))

    from operations.brightness import apply_brightness_augmentations
    from operations.contrast import apply_contrast_augmentations
    from operations.rotate import apply_rotation_augmentations
    from operations.scale import apply_scale_augmentations

# File extensions
JSON_EXTENSION = ".json"
PNG_EXTENSION = ".png"

# Thread-safe counter and lock for frame numbering
frame_counter_lock = Lock()
global_frame_counter = {"value": 0}


def get_next_frame_number() -> int:
    """Thread-safe function to get the next frame number.

    Returns:
        Next frame number to use
    """
    with frame_counter_lock:
        global_frame_counter["value"] += 1
        return global_frame_counter["value"]


def load_training_sample(
    json_path: Union[str, Path],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a training sample (image and annotations) from JSON file.

    Args:
        json_path: Path to the JSON annotation file

    Returns:
        Tuple of (image_array, annotations_dict)
    """
    # Load JSON annotations
    with open(json_path, "r") as f:
        annotations = json.load(f)

    # Get corresponding image path
    image_path = str(json_path).replace(JSON_EXTENSION, PNG_EXTENSION)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    return image, annotations


def copy_original_data(json_files: List[str], output_dir: str) -> List[Tuple[str, str]]:
    """Copy original training data to output directory.

    Args:
        json_files: List of JSON file paths
        output_dir: Output directory path

    Returns:
        List of tuples (image_path, json_path) for copied files
    """
    copied_files = []

    for json_file in tqdm(json_files, desc="Copying original data", unit="file"):
        # Load original data
        _, annotations = load_training_sample(json_file)

        # Get next frame number thread-safely
        frame_number = get_next_frame_number()

        # Create new filenames with sequential numbering
        image_filename = f"frame_{frame_number:04d}{PNG_EXTENSION}"
        json_filename = f"frame_{frame_number:04d}{JSON_EXTENSION}"

        image_path = os.path.join(output_dir, image_filename)
        json_path = os.path.join(output_dir, json_filename)

        # Copy image
        shutil.copy2(json_file.replace(JSON_EXTENSION, PNG_EXTENSION), image_path)

        # Copy and update annotations
        updated_annotations = annotations.copy()
        updated_annotations["original_frame"] = annotations["frame"]
        updated_annotations["frame"] = frame_number
        updated_annotations["augmentation_type"] = "original"

        with open(json_path, "w") as f:
            json.dump(updated_annotations, f, indent=2)

        copied_files.append((image_path, json_path))

    return copied_files


def apply_augmentation_to_sample(
    image: np.ndarray,
    annotations: Dict,
    augmentation_func: Callable,
    output_dir: str,
) -> List[Tuple[str, str]]:
    """Apply a single augmentation to a training sample.

    Args:
        image: Input image array
        annotations: Annotations dictionary
        augmentation_func: Function to apply augmentation
        output_dir: Output directory

    Returns:
        List of tuples (image_path, json_path) for augmented files
    """
    # Get next frame number thread-safely
    frame_number = get_next_frame_number()
    base_name = f"frame_{frame_number:04d}"
    return augmentation_func(image, annotations, base_name, output_dir)


def process_sample_worker(
    json_file: str,
    augmentations_to_apply: Dict[str, Callable],
    output_dir: str,
    progress_bar_position: int = 0,
    brightness_factors: List[float] = None,
    scale_factors: List[float] = None,
    contrast_factors: List[float] = None,
) -> Tuple[str, int]:
    """Worker function to process a single sample with all augmentations.

    Args:
        json_file: Path to JSON file to process
        augmentations_to_apply: Dictionary of augmentation functions
        output_dir: Output directory for augmented files
        progress_bar_position: Position for this thread's progress bar
        brightness_factors: List of brightness adjustment factors
        scale_factors: List of scale adjustment factors
        contrast_factors: List of contrast adjustment factors

    Returns:
        Tuple of (json_file_basename, number_of_augmented_files_created)
    """
    try:
        # Load sample
        image, annotations = load_training_sample(json_file)

        augmented_count = 0

        # Apply each enabled augmentation
        for aug_name, aug_func in augmentations_to_apply.items():
            try:
                base_name = f"frame_{get_next_frame_number():04d}"
                if aug_name == "brightness":
                    augmented_files = aug_func(
                        image, annotations, base_name, output_dir, brightness_factors
                    )
                elif aug_name == "scale":
                    augmented_files = aug_func(
                        image, annotations, base_name, output_dir, scale_factors
                    )
                elif aug_name == "contrast":
                    augmented_files = aug_func(
                        image, annotations, base_name, output_dir, contrast_factors
                    )
                else:  # rotation doesn't need factors
                    augmented_files = aug_func(
                        image, annotations, base_name, output_dir
                    )
                augmented_count += len(augmented_files)
            except Exception as e:
                print(
                    f"Error applying {aug_name} to {os.path.basename(json_file)}: {e}"
                )
                continue

        return os.path.basename(json_file), augmented_count

    except Exception as e:
        print(f"Error processing {os.path.basename(json_file)}: {e}")
        return os.path.basename(json_file), 0


def count_orphaned_images(input_dir: str) -> int:
    """Count images that don't have corresponding JSON files.

    Args:
        input_dir: Directory to check for orphaned images

    Returns:
        Number of orphaned images found
    """
    orphaned_count = 0

    # Find all PNG files in input directory
    png_pattern = os.path.join(input_dir, f"*{PNG_EXTENSION}")
    png_files = glob.glob(png_pattern)

    for png_file in png_files:
        # Check if corresponding JSON file exists
        json_file = png_file.replace(PNG_EXTENSION, JSON_EXTENSION)
        if not os.path.exists(json_file):
            orphaned_count += 1

    return orphaned_count


def augment_training_data(
    input_dir: str,
    output_dir: str,
    enabled_augmentations: List[str] = None,
    max_workers: int = None,
    brightness_factors: List[float] = None,
    scale_factors: List[float] = None,
    contrast_factors: List[float] = None,
) -> None:
    """Augment training data with various transformations using multithreading.

    Args:
        input_dir: Directory containing training data (JSON and PNG files)
        output_dir: Directory to save augmented data
        enabled_augmentations: List of augmentations to apply. If None, applies all.
        max_workers: Maximum number of worker threads. If None, uses CPU count.
        brightness_factors: List of brightness adjustment factors. If None, uses defaults [0.5, 0.7, 1.3, 1.5].
        scale_factors: List of scale adjustment factors. If None, uses defaults [0.5, 0.7, 1.3, 1.5].
        contrast_factors: List of contrast adjustment factors. If None, uses defaults [0.5, 0.7, 1.3, 1.5].
    """
    # Default augmentations
    all_augmentations = {
        "rotation": apply_rotation_augmentations,
        "brightness": apply_brightness_augmentations,
        "scale": apply_scale_augmentations,
        "contrast": apply_contrast_augmentations,
    }

    # Filter enabled augmentations
    if enabled_augmentations is None:
        enabled_augmentations = list(all_augmentations.keys())

    augmentations_to_apply = {
        name: func
        for name, func in all_augmentations.items()
        if name in enabled_augmentations
    }

    # Set default max_workers to CPU count if not specified
    if max_workers is None:
        import multiprocessing

        max_workers = multiprocessing.cpu_count()

    # Set default factor values if not specified
    if brightness_factors is None:
        brightness_factors = [0.5, 0.7, 1.3, 1.5]
    if scale_factors is None:
        scale_factors = [0.5, 0.7, 1.3, 1.5]
    if contrast_factors is None:
        contrast_factors = [0.5, 0.7, 1.3, 1.5]

    # Check for orphaned images (but don't delete them)
    print("Checking for orphaned images...")
    orphaned_count = count_orphaned_images(input_dir)
    if orphaned_count > 0:
        print(
            f"Found {orphaned_count} orphaned image(s) - these will be skipped during augmentation"
        )
    else:
        print("No orphaned images found")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all JSON files in input directory
    json_pattern = os.path.join(input_dir, f"*{JSON_EXTENSION}")
    json_files = sorted(glob.glob(json_pattern))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} training samples")
    print(f"Using {max_workers} worker threads")

    # Initialize global frame counter to 0 (will be incremented as needed)
    global_frame_counter["value"] = 0

    # Copy original data first
    print("Copying original data...")
    original_files = copy_original_data(json_files, output_dir)
    print(f"Copied {len(original_files)} original samples")

    # Apply augmentations using multithreading
    print("Applying augmentations with multithreading...")

    total_augmented = 0

    # Use ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(
                process_sample_worker,
                json_file,
                augmentations_to_apply,
                output_dir,
                i % max_workers,  # Position for progress bar (if needed)
                brightness_factors,
                scale_factors,
                contrast_factors,
            ): json_file
            for i, json_file in enumerate(json_files)
        }

        # Process completed tasks with a single progress bar
        with tqdm(
            total=len(json_files),
            desc="Processing samples",
            unit="sample",
            position=0,
            leave=True,
        ) as pbar:
            for future in as_completed(future_to_file):
                json_file = future_to_file[future]
                try:
                    _, augmented_count = future.result()
                    total_augmented += augmented_count
                except Exception as e:
                    print(f"Error processing {os.path.basename(json_file)}: {e}")
                finally:
                    pbar.update(1)

    print("Augmentation complete!")
    print(f"Original samples: {len(original_files)}")
    print(f"Augmented samples: {total_augmented}")
    print(f"Total samples: {len(original_files) + total_augmented}")


def main():
    """Main function to run data augmentation."""

    # Configuration variables - modify these as needed
    input_dir = "training_data"  # Set to None for auto-detection, or specify path
    output_dir = (
        "augmented_training_data"  # Set to None for auto-detection, or specify path
    )
    max_workers = 12  # Set to None for CPU count, or specify number of threads
    enabled_augmentations = (
        None  # Set to None for all, or list like ["rotation", "brightness"]
    )
    brightness_factors = [0.5, 0.7, 1.3, 1.5]  # Brightness adjustment factors
    scale_factors = [0.5, 0.7, 1.3, 1.5]  # Scale adjustment factors
    contrast_factors = [0.5, 0.7, 1.3, 1.5]  # Contrast adjustment factors

    # Check if we're running from the augmentation directory
    current_dir = Path.cwd()
    if current_dir.name == "augmentation":
        input_dir = "../training_data"
        output_dir = "../augmented_training_data"
    elif "augmentation" in str(current_dir):
        # Try to find relative paths
        input_dir = "../../training_data"
        output_dir = "../../augmented_training_data"

    # Create full paths
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    print(f"Input directory: {input_path.absolute()}")
    print(f"Output directory: {output_path.absolute()}")

    # Default augmentations if not specified
    if enabled_augmentations is None:
        enabled_augmentations = ["rotation", "brightness", "scale", "contrast"]

    print(f"Enabled augmentations: {enabled_augmentations}")

    # Set default max_workers to CPU count if not specified
    if max_workers is None:
        import multiprocessing

        max_workers = multiprocessing.cpu_count()

    print(f"Using {max_workers} worker threads")

    augment_training_data(
        str(input_path),
        str(output_path),
        enabled_augmentations,
        max_workers,
        brightness_factors,
        scale_factors,
        contrast_factors,
    )


if __name__ == "__main__":
    main()
