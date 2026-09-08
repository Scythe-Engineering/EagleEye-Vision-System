"""Flush completed vision branches together, never a skipped/unfinished cycle."""

from collections import deque
from threading import Lock
from types import SimpleNamespace

import pytest

from src.config.utils.pipeline import Pipeline


@pytest.mark.parametrize(
    "completed,publisher,expected",
    [
        (True, True, ["pose", "meta", "flush"]),
        (False, True, ["pose", "meta"]),
        (True, False, ["pose", "meta"]),
    ],
)
def test_flush_follows_completed_publisher_branches(
    completed: bool, publisher: bool, expected: list[str]
) -> None:
    """Check that only completed publishing flows flush after their branches.

    Args:
        completed: Whether the flow reports completion.
        publisher: Whether the flow contains an NT publisher.
        expected: Expected branch and flush event order.
    """
    events = []
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.limit_frames_to_camera_capture_speed = False
    pipeline.operations = {
        "output": SimpleNamespace(
            name="publish_to_networktables" if publisher else "robot_pose_output"
        )
    }
    pipeline.network_table = SimpleNamespace(
        getInstance=lambda: SimpleNamespace(flush=lambda: events.append("flush"))
    )
    pipeline.total_time_history = deque(maxlen=5)
    pipeline.total_time_history_lock = Lock()

    def run_flow() -> bool:
        """Record both branch events and return the configured completion state."""
        events.extend(["pose", "meta"])
        return completed

    pipeline.flow_manager = SimpleNamespace(
        run_flow=run_flow,
        operation_outputs={},
        set_latest_profile_cycle_time=lambda _: None,
    )
    pipeline.run()
    assert events == expected
