"""Current-frame single-tag ambiguity regression and capture-clock safeguards."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.config.utils.operation import Operation
from src.main_operations.definitions.pnp_camera_localization import (
    PnpCameraLocalizationDefinition,
)
from src.main_operations.modules.apriltags.pnp_localization import PnpLocalization
from src.main_operations.modules.apriltags.utils.fmap_parser import load_fmap_file
from src.utils.timing import TimedValue, TimingMetadata, get_timing, unwrap_timed

# North-lane replay frame 639: both IPPE solutions have subpixel image error,
# but the image-only winner puts the camera almost eight meters away.
CORNERS = np.array(
    [
        [267.22445559413995, 326.37355491258336],
        [250.54890965049898, 326.15967427091465],
        [250.20466603295745, 343.78973478998216],
        [266.7490824815759, 343.44239448349384],
    ]
)
PREVIOUS = np.array(
    [
        [
            -0.004654246889793157,
            -0.0046913777067750795,
            -0.9999781642421504,
            12.472524916380547,
        ],
        [
            0.9999888290929729,
            -0.0008462590504686451,
            -0.0046503263201895795,
            7.392208519463698,
        ],
        [
            -0.0008244241345330416,
            -0.9999886373458609,
            0.00469526399838438,
            0.45760050927201323,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
TRUTH_POSITION = np.array([12.474364280700684, 7.349999904632568, 0.5])


def estimator() -> PnpLocalization:
    """Create the calibrated field solver used by continuity regressions."""
    root = Path(__file__).resolve().parents[1]
    return PnpLocalization(
        np.array(
            [
                [762.7222992602944, 0.0, 640.0],
                [0.0, 762.7222992602944, 400.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        np.array([-0.06, 0.012, 0.0005, -0.0005, 0.0]),
        load_fmap_file(
            root
            / "src/webui/assets/fields/2026/apriltag_maps/FE-2026-_REBUILTTM_Playing_Field.fmap"
        ),
    )


def detection(corners: np.ndarray) -> list:
    """Wrap four image corners in the minimal AprilTag detection shape."""
    return [SimpleNamespace(tag_id=19, corners=corners)]


def timed(corners: np.ndarray, timestamp: int) -> TimedValue:
    """Attach a capture timestamp to one test detection."""
    return TimedValue(detection(corners), TimingMetadata(timestamp // 1000, timestamp))


def primed_operation() -> PnpCameraLocalizationDefinition:
    """Return an operation with the known good pose stored as history."""
    operation = object.__new__(PnpCameraLocalizationDefinition)
    operation.pose_estimator = estimator()
    operation.use_pose_continuity = True
    operation.uses_timed_inputs = True
    operation._reset_pose_history()
    camera_from_field = PnpLocalization.fast_se3_inverse(PREVIOUS)
    corners = cv2.projectPoints(
        operation.pose_estimator.apriltag_map[19].global_corners,
        cv2.Rodrigues(camera_from_field[:3, :3])[0],
        camera_from_field[:3, 3],
        operation.pose_estimator.camera_matrix,
        operation.pose_estimator.distortion_coefficients,
    )[0].reshape(4, 2)
    output = operation.run(timed(corners, 1_000_000_000))
    np.testing.assert_allclose(output["camera_pose"], PREVIOUS, atol=1e-5)
    return operation


def test_single_tag_tie_uses_current_image_without_blending() -> None:
    """Choose the nearby current-image IPPE branch without blending the prior."""
    solver = estimator()
    image_only = solver.estimate_pose_from_detections(detection(CORNERS))[0]
    selected = solver.estimate_pose_from_detections(detection(CORNERS), PREVIOUS)[0]
    assert np.linalg.norm(image_only[:3, 3] - TRUTH_POSITION) > 7.0
    assert np.linalg.norm(selected[:3, 3] - TRUTH_POSITION) < 0.5
    # Selection returns the solved candidate, not the prior or a blended output.
    assert np.linalg.norm(selected[:3, 3] - PREVIOUS[:3, 3]) > 0.1
    np.testing.assert_allclose(
        primed_operation().run(timed(CORNERS, 1_008_333_333))["camera_pose"],
        selected,
        atol=1e-5,
    )
    far_prior = PREVIOUS.copy()
    far_prior[:3, 3] += 10.0
    np.testing.assert_allclose(
        solver.estimate_pose_from_detections(detection(CORNERS), far_prior)[0],
        image_only,
    )


@pytest.mark.parametrize(
    "mode", ["stale", "duplicate", "backward", "untimed", "disabled"]
)
def test_invalid_history_keeps_stateless_output(mode: str) -> None:
    """Clear invalid history and retain the ordinary image-only solution."""
    operation = primed_operation()
    expected = operation.pose_estimator.estimate_pose_from_detections(
        detection(CORNERS)
    )[0]
    timestamp = {
        "stale": 1_300_000_000,
        "duplicate": 1_000_000_000,
        "backward": 999_000_000,
    }.get(mode, 1_008_333_333)
    if mode == "disabled":
        operation.update_config({"use_pose_continuity": False})
    value = detection(CORNERS) if mode == "untimed" else timed(CORNERS, timestamp)
    np.testing.assert_allclose(operation.run(value)["camera_pose"], expected)


def projected(solver: PnpLocalization, pose: np.ndarray, tag_id: int) -> np.ndarray:
    """Project one mapped tag into the image for a known field camera pose."""
    camera = solver.fast_se3_inverse(pose)
    return cv2.projectPoints(
        solver.apriltag_map[tag_id].global_corners,
        cv2.Rodrigues(camera[:3, :3])[0],
        camera[:3, 3],
        solver.camera_matrix,
        solver.distortion_coefficients,
    )[0].reshape(4, 2)


def test_jump_is_unavailable_not_held_and_reacquires_after_expiry() -> None:
    """Reject a jump without holding output, then allow expired-history recovery."""
    operation = primed_operation()
    moved = PREVIOUS.copy()
    moved[0, 3] += 3.0
    corners = projected(operation.pose_estimator, moved, 19)
    for capture_ns in (1_008_333_333, 1_200_000_000):
        assert operation.run(timed(corners, capture_ns)) == {
            "camera_pose": None,
            "pose_meta": None,
        }
    # Rejections must not refresh the old anchor and prevent recovery forever.
    recovered = operation.run(timed(corners, 1_300_000_000))
    np.testing.assert_allclose(recovered["camera_pose"], moved, atol=3e-5)


def test_multitag_pose_is_image_only_and_preserves_capture_timing() -> None:
    """Keep multi-tag solving image-only and retain capture timing on both outputs."""
    operation = primed_operation()
    solver = operation.pose_estimator
    moved = PREVIOUS.copy()
    moved[0, 3] += 3.0
    detections = [
        SimpleNamespace(tag_id=tag, corners=projected(solver, moved, tag))
        for tag in (19, 20)
    ]
    expected = solver.estimate_pose_from_detections(detections)[0]
    clock = TimingMetadata(1_008_333, 1_008_333_333)
    wrapper = Operation(
        operation,
        "pnp-timed-test",
        "pnp_camera_localization",
        input_ports=("detections",),
        output_ports=("camera_pose", "pose_meta"),
    )
    output = wrapper.run(TimedValue(detections, clock))
    np.testing.assert_allclose(unwrap_timed(output["camera_pose"]), expected)
    assert get_timing(output["camera_pose"]) == clock
    assert get_timing(output["pose_meta"]) == clock
    # The multi-tag result becomes the new anchor rather than being jump-rejected.
    single = operation.run(timed(projected(solver, moved, 19), 1_016_666_666))
    assert single["camera_pose"] is not None
    np.testing.assert_allclose(single["camera_pose"], moved, atol=3e-5)
