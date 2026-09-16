"""Pure checks for temporal experiment compaction and summaries."""

from benchmarks.temporal_experiments import _reacquisition, compact_record, summarize


def _row(index: int, truth: int, matched: int, pose: bool = False) -> dict:
    """Build one compact frame row.

    Args:
        index: Frame index in the synthetic clip.
        truth: Count of provisional truth tags.
        matched: Count of matched provisional tags.
        pose: Whether the row has an available pose.

    Returns:
        A compact frame record accepted by ``summarize`` and ``_reacquisition``.
    """
    return {
        "clip": "clip",
        "frame_index": index,
        "pipeline_ms": 1.0,
        "decode_ms": 0.5,
        "operation_ms": {"detect": 0.25},
        "provisional_truth_tags": truth,
        "provisional_matched": matched,
        "total_matched": matched,
        "detected_ids": [],
        "pose_available": pose,
        "robot_errors": None,
        "failure": None,
        "tp": matched,
        "fp": 0,
        "fn": truth - matched,
    }


def test_summary_uses_provisional_or_eligible_truth_not_raw_tag_list() -> None:
    """Only provisional or eligible labels contribute to visibility groups."""
    record = {
        "frame_index": 0,
        "pipeline_duration_ns": 1_000_000,
        "decode_duration_ns": 500_000,
        "failure": None,
        "output": {"detections": [{"tag_id": 4}], "profile": None},
        "metrics": {
            "pose_available": False,
            "robot_pose": None,
            "detection": {
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "matches": [{}],
                "by_tag": [
                    {
                        "tag_id": 4,
                        "eligible": False,
                        "category": "always-listed",
                        "detected": True,
                    },
                    {
                        "tag_id": 5,
                        "eligible": True,
                        "category": "eligible",
                        "detected": True,
                    },
                ],
            },
        },
    }
    row = compact_record(record, "clip")
    assert row["provisional_truth_tags"] == 1
    summary = summarize([row])
    assert summary["groups"]["tag_present"]["frames_denominator"] == 1
    assert (
        summary["groups"]["all"]["provisional_match_rate_proxy_not_recall"]["rate"] == 1
    )


def test_summary_includes_pose_availability_and_failure_count() -> None:
    """Summaries retain pose availability and failures as explicit counts."""
    failed = _row(1, 0, 0, pose=True)
    failed["failure"] = "pipeline failure"
    group = summarize([_row(0, 1, 1, pose=True), failed])["groups"]["all"]
    assert group["pose_available"] == {"count": 2, "rate": 1.0}
    assert group["failure_count"] == 1


def test_reacquisition_keeps_unrecovered_delay_null() -> None:
    """An unrecovered visible interval retains a null delay."""
    rows = [_row(0, 0, 0), _row(1, 1, 0), _row(2, 1, 0)]
    assert _reacquisition(rows, "provisional_matched") == [
        {
            "after_no_provisional_start": 0,
            "visibility_frame": 1,
            "recovered_frame": None,
            "delay_frames": None,
        }
    ]


def test_reacquisition_does_not_cross_a_later_tag_free_gap() -> None:
    """Recovery must not borrow a detection from a later visible interval."""
    rows = [
        _row(0, 0, 0),
        _row(1, 1, 0),
        _row(2, 0, 0),
        _row(3, 1, 1),
    ]
    assert _reacquisition(rows, "provisional_matched") == [
        {
            "after_no_provisional_start": 0,
            "visibility_frame": 1,
            "recovered_frame": None,
            "delay_frames": None,
        },
        {
            "after_no_provisional_start": 2,
            "visibility_frame": 3,
            "recovered_frame": 3,
            "delay_frames": 0,
        },
    ]
