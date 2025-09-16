import json
import os

import cv2
import numpy as np


def visualize_apriltag_corners(image_path: str) -> None:
    """Visualize AprilTag corners from JSON detection data on an image.

    Args:
        image_path: Path to the image file (.png). The corresponding JSON file
                   will be found by replacing .png with .json.

    Returns:
        None. Displays the image with detected corners overlaid.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Construct JSON path by replacing .png with .json
    json_path = image_path.replace(".png", ".json")

    # Check if JSON file exists
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return

    # Load and parse JSON data
    try:
        with open(json_path, "r") as file:
            detection_data = json.load(file)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    # Extract tags from detection data
    detected_tags = detection_data.get("tags", [])

    # Draw each tag's corners
    for tag in detected_tags:
        tag_name = tag.get("tag_name", "unknown")
        corners = tag.get("corners", [])

        if len(corners) != 4:
            print(f"Warning: Tag {tag_name} does not have 4 corners")
            continue

        # Convert corners to list of points for drawing
        corner_points = []
        for corner in corners:
            x = int(corner["x"])
            y = int(corner["y"])
            corner_points.append((x, y))

        # Draw the quadrilateral connecting the corners
        points = np.array(corner_points, np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(image, [points], True, (0, 255, 0), 2)

        # Draw corner points as circles
        for i, (x, y) in enumerate(corner_points):
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

        # Add tag ID label near the first corner
        cv2.putText(
            image,
            f"ID:{tag_name}",
            (corner_points[0][0] + 5, corner_points[0][1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    # Display the result
    window_name = f"AprilTag Detection - {os.path.basename(image_path)}"
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage - you can modify this path as needed
    example_image_path = r"E:\Ceph-Mirror\Python-Files\Projects\FIRST-Note-Detection\src\main_operations\modules\apriltags\pre_processing\ai_accelleration\training\augmented_training_data\frame_2744_scale1_1.png"
    visualize_apriltag_corners(example_image_path)
