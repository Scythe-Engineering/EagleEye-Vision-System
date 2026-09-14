"""Focused contracts for finite synthetic-video replay."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from benchmarks.replay import (
    DecodedFrame,
    ReplayCameraManager,
    ReplayError,
    SequentialVideoDecoder,
    build_pipeline,
    rational_timestamp_ns,
    run_accuracy,
    stream_annotations,
)
from src.main_operations.modules.apriltags.utils.apriltag import Apriltag


def test_rational_120fps_does_not_accumulate_rounded_period() -> None:
    """Frame timestamps should derive from the index, not repeated addition."""
    assert rational_timestamp_ns(1, 120) == 8_333_333
    assert rational_timestamp_ns(2, 120) == 16_666_667
    assert rational_timestamp_ns(120, 120) == 1_000_000_000


def test_packet_latch_is_same_object_and_same_instant() -> None:
    """One cycle must retain one immutable packet identity."""
    manager = ReplayCameraManager(epoch_ns=2_000_000_000)
    packet = manager.publish(np.zeros((1, 1, 3), dtype=np.uint8), 4, 12_000)
    assert manager.begin_cycle() is packet
    assert manager.get_current_packet_by_bus_id(manager.bus_id) is packet
    assert packet.timing.capture_monotonic_ns == 2_000_012_000
    manager.end_cycle()


def test_sequential_decoder_stops_at_finite_eof(tmp_path: Path) -> None:
    """The benchmark decoder must neither preload nor loop at EOF."""
    video = tmp_path / "tiny.mkv"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"FFV1"),  # type: ignore[attr-defined]
        120.0,
        (16, 16),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV FFV1 writer unavailable")
    for value in (0, 80, 160):
        writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
    writer.release()

    decoder = SequentialVideoDecoder(video, max_lookahead=2)
    try:
        frames = list(decoder)
        assert [frame.index for frame in frames] == [0, 1, 2]
        assert decoder.read() is None
        assert len(decoder._queue) <= 2
    finally:
        decoder.close()


def test_fmap_corner_order_matches_detector_contract() -> None:
    """Map corners should use pupil-apriltags' canonical decoded order."""
    tag = Apriltag(
        tag_id=1,
        family="apriltag3_36h11_classic",
        size=1000.0,
        transform=np.eye(4).reshape(-1).tolist(),
        unique=True,
        field_length=0.0,
        field_width=0.0,
    )
    np.testing.assert_allclose(
        tag.global_corners,
        [
            [0.0, 0.5, 0.5],
            [0.0, -0.5, 0.5],
            [0.0, -0.5, -0.5],
            [0.0, 0.5, -0.5],
        ],
    )


def test_gzip_annotations_require_exact_timestamps(tmp_path: Path) -> None:
    """Truth rows with a drifted timestamp must fail before scoring."""
    truth = tmp_path / "truth.jsonl.gz"
    with gzip.open(truth, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps({"frame_index": 0, "timestamp_ns": 1}) + "\n")

    with pytest.raises(ReplayError, match="annotation alignment"):
        list(stream_annotations(truth, 120))


class _Decoder:
    """Small finite decoder used by deterministic pacing tests."""

    max_lookahead = 8

    def __init__(self, count: int) -> None:
        self.items = [DecodedFrame(i, i) for i in range(count)]

    def read(self) -> DecodedFrame | None:
        """Return the next numbered frame."""
        return self.items.pop(0) if self.items else None

    def __iter__(self):
        """Yield every remaining numbered frame."""
        while (item := self.read()) is not None:
            yield item

    def close(self) -> None:
        """Release no resources."""


@dataclass
class _Clock:
    """Thread-safe-enough monotonic clock for one producer and one consumer."""

    value: int = 1_000_000_000

    def now_ns(self) -> int:
        """Return the current synthetic time."""
        return self.value

    def sleep_until_ns(self, deadline_ns: int) -> None:
        """Advance immediately to the requested deadline."""
        self.value = max(self.value, deadline_ns)


def test_accuracy_does_not_reuse_a_stale_profile() -> None:
    """A skipped cycle must not inherit the preceding profile snapshot."""

    class Pipeline:
        def __init__(self) -> None:
            self.profile = {"frame_seq": 1}

        def run(self) -> None:
            """Leave the profile unchanged to represent a skipped cycle."""

        def get_latest_profile_snapshot(self) -> dict[str, int]:
            """Return one stale profile."""
            return self.profile.copy()

        def get_operation_errors(self) -> list[object]:
            """Return no operation failures."""
            return []

    manager = ReplayCameraManager(epoch_ns=1)
    records = run_accuracy(
        _Decoder(1),
        manager,
        Pipeline(),
        120,
        aligned_truth=iter([{"frame_index": 0}]),
        collect=lambda pipeline, index: {
            "frame_index": index,
            "profile": pipeline.get_latest_profile_snapshot(),
        },
    )
    assert records[0]["skipped"] is True
    assert records[0]["output"]["profile"] is None


@pytest.mark.parametrize("temporal", [False, True])
def test_real_benchmark_graph_constructs_and_runs_blank_frame(temporal: bool) -> None:
    """Both graphs must initialize their real production operation classes."""
    calibration = {
        "camera_matrix": [[762.7, 0.0, 640.0], [0.0, 762.7, 400.0], [0.0, 0.0, 1.0]],
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    manager = ReplayCameraManager(epoch_ns=1_000_000)
    with build_pipeline(temporal, calibration, manager=manager) as pipeline:
        manager.publish(np.zeros((800, 1280, 3), dtype=np.uint8), 0, 0)
        manager.begin_cycle()
        try:
            pipeline.run()
        finally:
            manager.end_cycle()
        assert pipeline.get_operation_errors() == []
        assert pipeline.get_operation_by_uuid("bench-detect") is not None
