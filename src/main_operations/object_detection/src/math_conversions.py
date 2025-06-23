import numpy as np


def rotate2d(point: np.array, angle: float) -> np.array:
    """
    Rotates a point around the origin.

    Args:
        point (np.array): The point to rotate as [x, y].
        angle (float): The angle to rotate the point by, in radians.

    Returns:
        np.array: The rotated point as [x, y].
    """
    px, py = point

    sin = np.sin(angle)
    cos = np.cos(angle)

    qx = cos * px - sin * py
    qy = sin * px + cos * py
    return qx, qy


def pixels_to_degrees(
    pixel_position: float, total_pixels: int, fov: float, log: callable
) -> float:
    """
    Converts a pixel position to a degree position.

    Args:
        pixel_position (float): The position in pixels (from the center).
        total_pixels (int): The total number of pixels across the axis.
        fov (float): The field of view of the camera in degrees.
        log (callable): Logging function to handle errors.

    Returns:
        float: The position in degrees (from the center).
    """
    pixel_percent = pixel_position / (total_pixels / 2)  # Normalize from -1 to 1

    if np.any(np.abs(pixel_percent) > 1):
        log("ERROR: Pixel position is outside of expected range.")

    return pixel_percent * (fov / 2)


def calculate_local_position(
    pixel_position: np.array,
    total_pixels: np.array,
    camera_fov: np.array,
    camera_offset_pos: np.array,
    log: callable,
) -> np.array:
    """
    Calculates the local position of the note.

    Args:
        pixel_position (np.array): The position of the note in pixels (from the center) as [x, y].
        total_pixels (np.array): The total number of pixels as [width, height].
        camera_fov (np.array): The field of view of the camera in degrees as [fov_x, fov_y].
        camera_offset_pos (np.array): The offset position of the camera in meters as [x, y, z].
        log (callable): The logger to use for error handling.

    Returns:
        np.array: The local position of the note as [x, y].
    """
    screen_angle_x = (
        pixels_to_degrees(pixel_position[0], total_pixels[0], camera_fov[0], log=log)
        * 1.3312675733
    )
    screen_angle_y = (
        pixels_to_degrees(pixel_position[1], total_pixels[1], camera_fov[1], log=log)
        * 1.3312675733
    )
    flat_distance = camera_offset_pos[2] * np.tan(np.radians(90 + screen_angle_y))
    return rotate2d([flat_distance, 0], np.radians(-screen_angle_x)) + np.array(
        camera_offset_pos[:2]
    )


def convert_to_global_position(
    local_position: np.array, robot_pose: np.array
) -> np.array:
    """
    Converts the local position to the global position on the field.

    Args:
        local_position (np.array): The local position of the note as [x, y].
        robot_pose (np.array): The pose of the robot in meters and radians as [x, y, theta].

    Returns:
        np.array: The global position of the note as [x, y].
    """
    return rotate2d(local_position, robot_pose[2]) + robot_pose[:2]
