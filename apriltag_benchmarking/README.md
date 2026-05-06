# AprilTag Benchmarking

CLI tools for benchmarking AprilTag detector accuracy and speed on the synthetic Blender dataset in:

```text
/Users/darkeden/EagleEye-Vision-System/apriltag_benchmark_data
```

## Run from repo root

```bash
cd /Users/darkeden/EagleEye-Vision-System
```

Use `uv run`, not direct `python`, so the project environment/dependencies are used.

## Run all detectors with defaults

With no arguments, the CLI runs every detector implementation using tuned defaults and prints a comparison table.

```bash
uv run python -m apriltag_benchmarking.entry
```

Default detector settings are `--quad-decimate 1`, `--quad-sigma .8`, `--nthreads 4`, and `--max-frames 200`. Use `--max-frames 0` to process every frame.

This writes:

```text
apriltag_benchmarking/results.json                 # combined report
apriltag_benchmarking/results_pupil.json           # baseline report
apriltag_benchmarking/results_temporal_pupil.json  # temporal ROI report
```

## Baseline detector only

Runs `pupil-apriltags` on the full image.

```bash
uv run python -m apriltag_benchmarking.entry \
  --detector pupil \
  --output apriltag_benchmarking/results_pupil.json
```

## Temporal ROI detector

Uses the Rust `temporal_acceleration` module to predict regions where tags should be, then runs `pupil-apriltags` only on those crops.

```bash
uv run python -m apriltag_benchmarking.entry \
  --detector temporal-pupil \
  --quad-decimate 1 \
  --quad-sigma .8 \
  --nthreads 4 \
  --temporal-padding-factor 2 \
  --output apriltag_benchmarking/results_temporal.json
```

For quick smoke tests:

```bash
uv run python -m apriltag_benchmarking.entry \
  --detector temporal-pupil \
  --max-frames 60 \
  --quad-decimate 1 \
  --quad-sigma .8 \
  --nthreads 4 \
  --output /tmp/apriltag_temporal_bench.json
```

## Useful options

```text
--data-root PATH                  Dataset root. Defaults to apriltag_benchmark_data.
--output PATH                     JSON report path.
--detector pupil|temporal-pupil   Detector implementation.
--max-frames N                    Limit frames. Default 200; use 0 for all frames.
--verbose                         Print per-frame stats.

--families tag36h11               AprilTag family.
--nthreads N                      Detector threads.
--quad-decimate FLOAT             Pupil detector decimation.
--quad-sigma FLOAT                Pupil detector blur sigma.
--refine-edges 0|1                Pupil edge refinement.
--decode-sharpening FLOAT         Pupil decode sharpening.

--temporal-padding-factor FLOAT   ROI padding expansion for temporal detector.
--temporal-max-regions N          Max predicted ROIs per frame.
--temporal-min-region-size-px N   Minimum ROI size.
--no-temporal-merge               Disable merging overlapping predicted ROIs.
```

## Metrics

The JSON report contains:

- frames processed
- total runtime
- FPS and average ms/frame
- detections, true positives, false positives, false negatives
- precision and recall
- 3D pose translation error in meters
- 2D center error in pixels
- 2D corner error in pixels

Ground truth positives are metadata tags with `visible=true`. Detections are matched by `tag_family + tag_id`.

## Notes

- Pose estimates use crop-adjusted intrinsics for the temporal detector.
- The synthetic generator's `tag_size_m` is the full rendered texture plane. The pupil pose solver uses a scaled tag size internally to match the detectable AprilTag square.
- On macOS, full temporal crop runs can expose native instability in `pupil_apriltags` after many crop calls. The CLI defaults to `--max-frames 200`, which keeps the temporal speedup stable on the current dataset. Use `--max-frames 0` for full-dataset runs if your platform is stable.
