# EagleEye In-Depth Operation Tests

This test suite initializes and runs each operation with dummy data without
executing the full pipeline end-to-end.

## What it does

- Auto-discovers main and secondary operations under
  [`src/main_operations/definitions`](../src/main_operations/definitions) and
  [`src/secondary_operations`](../src/secondary_operations).
- Excludes YOLO-based object detection operations by name.
- Initializes each operation with config defaults and dummy dependencies.
- Calls `run()` with operation-specific dummy inputs.
- Initializes pipelines without starting any threads or camera access.
- Loads benchmark replay metadata from `src/utils/sim_videos/benchmark_manifest.json`
  and replays available MP4 assets through the configured full pipeline.

## Running tests

```bash
pytest -q tests
```

## Environment overrides

| Variable | Purpose | Default |
| --- | --- | --- |
| `EAGLEEYE_DUMMY_FRAME_WIDTH` | Dummy frame width | `640` |
| `EAGLEEYE_DUMMY_FRAME_HEIGHT` | Dummy frame height | `480` |
| `EAGLEEYE_TEST_PROJECT_ROOT` | Project root override | auto-detected |
| `EAGLEEYE_TEST_CAMERA_PARAMETERS_PATH` | Camera intrinsics file | `files/camera_parameters_path/intrinsics.json` |
| `EAGLEEYE_TEST_APRILTAG_MAP_PATH` | AprilTag map file | `files/apriltag_map_path/frc2025r2.json` |
| `EAGLEEYE_TEST_CAMERA_NAME` | Device input camera name | `test_camera` |
| `EAGLEEYE_TEST_NETWORK_TABLE_KEY` | NetworkTables key | `test_key` |

## Benchmark replay assets

Full-pipeline replay tests consume versioned assets from `src/utils/sim_videos/`.
Each benchmark entry declares the MP4 video, ground-truth CSV, pipeline name,
camera bus ID, and pose/detection/timing thresholds. When a large benchmark MP4
is not checked out in the local environment, pytest reports a `hardware_skip`
instead of failing unrelated smoke tests.

## Notes

- Rust-backed operations (temporal acceleration and pose outlier filter)
  are skipped when the extensions are unavailable.
- Hardware-specific operations rely on fakes and should not access devices.
