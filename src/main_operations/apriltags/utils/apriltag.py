import numpy as np


class Apriltag:
    """Represents an Apriltag with all necessary data from the fmap file."""

    def __init__(
        self,
        tag_id: int,
        family: str,
        size: float,
        transform: list[float],
        unique: bool,
        field_length: float,
        field_width: float,
    ) -> None:
        """Initialize an Apriltag object.

        Args:
            tag_id (int): The ID of the Apriltag.
            family (str): The family of the Apriltag.
            size (float): The size of the Apriltag.
            transform (list[float]): The transformation matrix as a flat list.
            unique (bool): Whether the tag is unique.
            field_length (float): The length of the field.
            field_width (float): The width of the field.
        """
        self.tag_id = tag_id
        self.family = family
        self.size = size / 1000
        self.transform = transform
        self.unique = unique
        self.field_length = field_length
        self.field_width = field_width

        transform_matrix = np.array(self.transform, dtype=np.float64).reshape((4, 4))
        rotation_y_90 = np.array(
            [
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        transform_matrix = transform_matrix @ rotation_y_90

        transform_matrix[0, 3] += self.field_length / 2
        transform_matrix[1, 3] += self.field_width / 2

        self.transform = transform_matrix.flatten().tolist()

        self.global_corners = self._get_global_corner_positions()

    def _get_global_transform_matrix(self) -> np.ndarray:
        """Get the 4x4 global transform matrix for this Apriltag."""
        return np.array(self.transform, dtype=np.float64).reshape((4, 4))

    def _get_global_corner_positions(self) -> np.ndarray:
        """Compute the global positions of the four corners of the Apriltag.

        Returns:
            np.ndarray: An array of shape (4, 3) containing the global positions of the tag's corners in field space.
        """
        half_size = self.size / 2
        local_corners = np.array(
            [
                [-half_size, -half_size, 0, 1],
                [half_size, -half_size, 0, 1],
                [half_size, half_size, 0, 1],
                [-half_size, half_size, 0, 1],
            ]
        )

        global_transform_matrix = self._get_global_transform_matrix()
        global_corners_homogeneous = (global_transform_matrix @ local_corners.T).T
        global_corners = global_corners_homogeneous[:, :3]
        global_corners = global_corners[[0, 3, 2, 1]]
        return global_corners
