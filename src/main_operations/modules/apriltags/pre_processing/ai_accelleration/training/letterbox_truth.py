import json
import os
from typing import Dict, List

from tqdm import tqdm


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


def process_json_file(json_path: str, target_width: int, target_height: int) -> None:
    """Process a single JSON file to letterbox its annotation coordinates and remove camera attribute.

    Args:
        json_path: Path to the JSON file to process.
        target_width: Target letterboxed width.
        target_height: Target letterboxed height.
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


def process_training_data(
    training_data_dir: str, target_width: int, target_height: int
) -> None:
    """Process all JSON files in the training data directory to letterbox coordinates and remove camera attributes.

    Args:
        training_data_dir: Path to the directory containing JSON annotation files.
        target_width: Target letterboxed width.
        target_height: Target letterboxed height.
    """
    if not os.path.exists(training_data_dir):
        raise FileNotFoundError(
            f"Training data directory not found: {training_data_dir}"
        )

    # Find all JSON files in the directory
    json_files = [f for f in os.listdir(training_data_dir) if f.endswith(".json")]

    if not json_files:
        print(f"No JSON files found in {training_data_dir}")
        return

    print(f"Found {len(json_files)} JSON files to process")
    print(
        f"Letterboxing coordinates to {target_width}x{target_height} and removing camera attributes"
    )

    # Process each JSON file with progress bar
    successful_files = 0
    failed_files = []

    for json_file in tqdm(json_files, desc="Processing files", unit="file"):
        json_path = os.path.join(training_data_dir, json_file)
        try:
            process_json_file(json_path, target_width, target_height)
            successful_files += 1
        except Exception as e:
            tqdm.write(f"ERROR processing {json_file}: {e}")
            failed_files.append((json_file, str(e)))

    # Summary
    print("\nLetterboxing complete!")
    print(f"Successfully processed: {successful_files} files")

    if failed_files:
        print(f"Failed to process: {len(failed_files)} files")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")
    else:
        print("All files processed successfully!")


def main() -> None:
    """Main function to run the letterboxing process and remove camera attributes."""
    # User-configurable variables
    training_data_directory = "training_data"
    target_width = 320
    target_height = 320

    process_training_data(training_data_directory, target_width, target_height)


if __name__ == "__main__":
    main()
