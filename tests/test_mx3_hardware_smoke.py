from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from src.utils.model_library import ResolvedArtifact
from src.utils.mx3_runtime import Mx3ResultPacket, Mx3RuntimeCoordinator
from src.utils.timing import TimedValue, TimingMetadata


class _OneFrameSource:
    """Emit one typed test frame, then poll until the stream is stopped."""

    def __init__(self) -> None:
        self.sent = False

    def wait_for_next_packet(
        self,
        after_frame_seq: int,
        should_continue: Callable[[], bool],
    ) -> TimedValue[np.ndarray] | None:
        """Return the one frame or wait cooperatively for stream shutdown."""
        if not self.sent:
            self.sent = True
            return TimedValue(
                np.zeros((320, 320, 3), dtype=np.uint8),
                TimingMetadata(1, time.monotonic_ns(), frame_seq=1),
            )
        while should_continue():
            time.sleep(0.01)
        return None


@pytest.mark.hardware_skip
def test_real_mx3_single_stream_smoke() -> None:
    """Run one frame through a configured physical MX3 and cropped post-model."""
    dfp_path = os.environ.get("MX3_DFP_PATH")
    post_path = os.environ.get("MX3_POSTPROCESSOR_PATH")
    if not dfp_path or not post_path:
        pytest.skip("Set MX3_DFP_PATH and MX3_POSTPROCESSOR_PATH for hardware smoke")

    profile = {
        "input_width": 320,
        "input_height": 320,
        "color_order": "rgb",
        "layout": "hwzc",
        "normalization": "zero_to_one",
        "use_model_shape": [False, True],
        "decoder": "yolo_nms_xyxy",
        "adjustable_controls": {"confidence": True, "max_detections": True},
        "max_inflight": 2,
    }
    artifact = ResolvedArtifact(
        model_id="hardware-smoke",
        device_id="mx3:0",
        slot="mx3_dfp",
        path=Path(dfp_path),
        postprocessor_path=Path(post_path),
        mx3_profile=profile,
    )
    coordinator = Mx3RuntimeCoordinator()
    binding = coordinator.register_stream(
        0,
        artifact,
        _OneFrameSource(),
        None,
        0.25,
        100,
        lambda: True,
    )
    try:
        coordinator.start()
        binding.activate()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(binding.wait_for_next)
            try:
                result: Mx3ResultPacket | None = future.result(timeout=30)
            except FutureTimeoutError as exc:
                # Wake the worker before its executor context waits for completion.
                binding.deactivate()
                raise AssertionError(
                    "MX3 did not complete one frame within 30 seconds"
                ) from exc
        assert result is not None
        assert result.frame.timing.frame_seq == 1
        assert isinstance(result.detections.value, list)
    finally:
        coordinator.stop()
