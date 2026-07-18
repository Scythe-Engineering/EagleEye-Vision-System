import numpy as np

from src.main_operations.modules.apriltags.utils.apriltag import Apriltag


def test_global_corners_preserve_pupil_apriltags_corner_order() -> None:
    """Transform tag corners without changing pupil-apriltags correspondence order."""
    tag = Apriltag(
        tag_id=1,
        family="apriltag3_36h11_classic",
        size=200.0,
        transform=[
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            2.0,
            0.0,
            0.0,
            1.0,
            3.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        unique=True,
        field_length=10.0,
        field_width=5.0,
    )

    local_corners_homogeneous = np.column_stack(
        (tag.single_solve_local_corners, np.ones(4))
    )
    expected_corners = (
        tag.tag_to_global_transform_matrix @ local_corners_homogeneous.T
    ).T[:, :3]

    np.testing.assert_allclose(tag.global_corners, expected_corners)
