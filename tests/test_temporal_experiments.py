"""Summaries of temporal benchmark runs."""

from benchmarks.temporal_experiments import summarize


def test_summary_operation_ms_stays_completed_only() -> None:
    """Skipped rows count toward the all-frames op timings, not operation_ms."""
    def row(index: int, skipped: bool, ops: dict[str, float]) -> dict[str, object]:
        return {
            "clip": "clip",
            "frame_index": index,
            "pipeline_ms": sum(ops.values()),
            "decode_ms": 0.0,
            "operation_ms": ops,
            "skipped": skipped,
            "provisional_truth_tags": 0,
            "provisional_matched": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "failure": None,
            "pose_available": not skipped,
            "robot_errors": None,
        }

    rows = [
        row(0, False, {"detect": 1.0, "pnp": 0.5}),
        row(1, True, {"detect": 9.0}),
    ]
    group = summarize(rows, include_reacquisition=False)["groups"]["all"]
    assert group["skipped_count"] == 1
    assert group["operation_ms"]["detect"]["mean"] == 1.0
    assert group["operation_ms"]["detect"]["denominator"] == 1
    assert group["operation_ms_all_frames"]["detect"]["mean"] == 5.0
    assert group["operation_ms_all_frames"]["pnp"]["denominator"] == 1
    assert group["pose_available"]["count"] == 1
