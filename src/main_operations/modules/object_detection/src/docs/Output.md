# Network Table Outputs Documentation (main.py)

This document describes all outputs sent to NetworkTables by the object detection system, focusing on the `GamePieces` table as implemented in `main.py`.

## Table: GamePieces

### class_names
- **Type:** String Array
- **Description:** List of all detected class names (object types) that the system can identify. Populated at startup and used as a reference for all other outputs.

For each class name (e.g., `note`, `cube`, `cone`), the following outputs are published. Replace `<class_name>` with the actual class name (e.g., `note_yaw_angles`).

---

### `<class_name>_yaw_angles`
- **Type:** Number Array
- **Description:** Yaw angles (in degrees) from the camera to each detected object of this class. Each entry corresponds to a detected instance in the current frame.

### `<class_name>_local_positions`
- **Type:** String Array
- **Description:** Local (robot-relative) positions of each detected object of this class, formatted as comma-separated values (e.g., `x, y`). Each entry is a string representing the position vector for a detected instance.

### `<class_name>_global_positions`
- **Type:** String Array
- **Description:** Global (field-relative) positions of each detected object of this class, formatted as comma-separated values (e.g., `x, y`). Each entry is a string representing the position vector for a detected instance.

### `<class_name>_distances`
- **Type:** Number Array
- **Description:** Euclidean distances (in the same units as the position vectors) from the robot to each detected object of this class.

### `<class_name>_ratio`
- **Type:** Number Array
- **Description:** Aspect ratio (width/height) of the bounding box for each detected object of this class. Useful for filtering or further classification.

---

## Output Behavior
- If no detections are present for a class in a given frame, the corresponding arrays for that class are published as empty arrays.
- All outputs are updated at a regular interval (approximately every 16 ms, or ~60 Hz).

## Example
If the system detects two objects of class `note`, the following keys will be present in the `GamePieces` table:
- `note_yaw_angles`: [12.5, -7.3]
- `note_local_positions`: ["1.2, 0.5", "-0.8, 0.3"]
- `note_global_positions`: ["5.1, 2.0", "3.9, 1.7"]
- `note_distances`: [1.3, 0.85]
- `note_ratio`: [1.8, 1.7]

## Other Tables
- `EagleEye` and `AdvantageKit` tables are initialized but not written to directly in this file.
