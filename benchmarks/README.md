# Synthetic video benchmarks

The [benchmark documentation](https://scythe-engineering.github.io/EagleEye-Docs/docs/codebase/synthetic-video-benchmarks) covers the dataset contract and Blender workflows. Run these commands from the repository root; each output directory must be new (or use `--overwrite`).

## Historical baseline (unchanged defaults)

```sh
uv run --no-sync python -m benchmarks run \
  --dataset /path/to/dataset/manifest.json \
  --cache-dir /path/to/dataset/cache \
  --pipeline both --output /tmp/eagleeye-baseline
```

The default remains normal PnP, minimum two detected tags, both full-frame and temporal pipelines, with their original configuration labels and detector settings.

## Matched normal versus 2D PnP

```sh
# Rapid original-clock prefix of one verified real clip; explicitly partial.
uv run --no-sync python -m benchmarks run \
  --dataset /path/to/dataset/manifest.json \
  --cache-dir /path/to/dataset/cache \
  --clip CLIP_ID --max-frames 120 \
  --pipeline both --solver both --minimum-tags both \
  --gyro-mode ideal-oracle --output /tmp/eagleeye-2d-oracle-smoke

# Full selected dataset, deterministic artificial noise/bias and delayed delivery.
uv run --no-sync python -m benchmarks run \
  --dataset /path/to/dataset/manifest.json \
  --cache-dir /path/to/dataset/cache \
  --pipeline both --solver both --minimum-tags both \
  --gyro-seed 2026 --gyro-noise-std-rad 0.005 --gyro-bias-rad 0.01 \
  --gyro-delivery-delay-ms 15 --gyro-processing-offset-ms 0 \
  --output /tmp/eagleeye-2d-delayed
```

Replace the manifest/cache paths with your dataset paths. Repeat `--clip ID` to select multiple clips. Omit `--max-frames` for complete clips; it never renumbers frames or shifts their original capture clock. `--timeout SECONDS` also stops cleanly and marks a partial result. `--solver 2d` runs only the constrained pipeline; `--minimum-tags 1` or `2` selects one gate threshold.

## What is supplied and measured

- **Artificial gyro, not recorded physical gyro:** heading is `atan2(R[1,0], R[0,0])` of the annotated `T_field_from_robot`, NWU CCW radians. Position truth is never supplied to either solver. `ideal-oracle` is exact, unperturbed capture heading and requires zero noise/bias/delay/processing offset. Zero perturbation and delay in synthetic mode is prominently labeled oracle-equivalent too.
- Measurements are generated once per annotated capture, perturbed with seeded NumPy noise and constant bias, then wrapped to `[-pi, pi)`. Each native double is published as `publisher.set(yaw, measurement_nt_us)`, retaining its **original measurement** timestamp, not delivery time. Delivery is at capture plus the configured delay; processing is at capture plus the configured offset. Only already-generated, delivered measurements are published: there is no future truth interpolation, heading extrapolation, or timestamp shifting. A positive processing offset does not synthesize future captures.
- The real production timestamped NT reader (`timestamped=true`, `history_size=256`) and 2D PnP operation do source consumption and capture alignment: nearest allowance 20 ms, interpolation gap limit 100 ms, refinement iterations 10. Excess delay can legitimately yield missing/stale-gyro rejection. Settings, seed, resolved graphs, calibration/map/media/truth hashes, actual measurement/delivery/processing timestamps and available sample inputs are saved.
- `--solver both` independently runs both production pipelines for each matched minimum-count threshold, keeping calibration, mounting, resolution, detector settings and temporal ROI feedback wiring matched. There is no below-count detector fallback or expected-ID filter. Independent temporal runs may subsequently produce different detections because solver feedback differs.
- Each independent run also invokes the **other production solver on the identical actual detector objects**, with its own history, same count gate, and same delivered samples. This shadow never drives temporal feedback. Thus each detector stream has a paired comparison, separately from independent-pipeline comparisons.
- Solver invocation latency is measured separately from pipeline latency. Pipeline latency includes the production scheduler/reader/detector/primary solver/conversion, not decoding, synthetic sample publishing, shadow solving, scoring or reporting. Shadow solver latency measures only its solver invocation; it is sequential, outside the scheduler, so use primary solver latency for production timing comparisons.

## Output

Open `OUTPUT/index.html` offline. `run.json` contains reproducibility metadata; `frames.jsonl.gz` contains actual detections, outputs, gyro inputs and rejection diagnostics; `summary.json` and `summary.csv` contain aggregate errors, p95/p99/max tails, explicit XY and 3D errors over 1 m, availability/missing intervals, rejection counts and separate solver/pipeline latency.

Each pipeline/solver/minimum-count variant stays separate in plots and rows. `identical_detector_outputs_paired` reports common-frame accuracy and error deltas (including signed means), lost/gained/neither counts and frame IDs, plus errors on the lost/gained populations. Independent comparisons also report attempted/failed populations, unmatched attempts and matched-frame pose availability. `independent_pipeline_comparisons` reports the corresponding comparison for separately replayed pipelines. Common-frame accuracy alone is not an availability claim. Acceptance is lower XY error/outliers without material availability loss; numerical gates have not been set. `partial=true` explicitly identifies prefix or timeout results; do not compare a prefix with a complete historical run as if their populations matched.

```sh
uv run --no-sync pytest tests/test_benchmark_gyro.py tests/test_benchmark_replay.py \
  tests/test_benchmark_metrics.py tests/test_benchmark_report.py -q
```

## Native resource limits

Use the existing one-thread detector presets for these comparisons. Native multithreading has an unresolved upstream reliability issue. Large tag families can require substantial decoding-table memory. The cleanup wrapper does not fix either limitation.

## Production wiring and calibration

The benchmark builds sanctioned copies of the existing presets, not a second detector graph: `device_input -> [temporal preprocessor] -> detect_apriltags -> minimum_apriltag_count -> pnp_camera_localization_2d -> camera_to_robot_pose`. The timestamped `get_networktables_value` source's `data` port connects to the solver's `gyro_samples` port; temporal feedback still comes from the primary solver's `camera_pose`. The original strict preset-port/feedback validator remains active before and after the allowed solver/source transformation.

A production gyro source node uses:

```json
{
  "action_name": "get_networktables_value.py",
  "action_params": {
    "network_table_key": "gyro",
    "timestamped": true,
    "history_size": 256
  }
}
```

`timestamped=false` remains the original latest-double reader. Timestamped mode uses a native generic subscription and `readQueue()`: `.value()` is the native value and `.time()` is the measurement timestamp translated by ntcore into the **receiver-local NT clock**. The reader returns a bounded timestamp-ordered history of plain `{"timestamp_us": int, "value": native_value}` dictionaries; there is no JSON envelope on the wire. **This solver requires finite yaw radians, NWU CCW about field +Z, with zero pointing along field +X**. The heading must already be aligned to the AprilTag map's field axes. Do not send an arbitrary gyro startup zero or an alliance-mirrored heading. Publish `publisher.set(yaw, measurement_nt_us)` with the actual measurement time in the publisher's local NT clock, just as AprilTag `publish_to_networktables._publish` calls `.set(value, capture_nt_us)`, in reverse. ntcore handles clock translation automatically. Client-mode sources fail closed until NT clock synchronization is available; server and isolated `startLocal()` modes require no server offset. Delayed publication must retain the original measurement timestamp, never substitute arrival or send time.

Configure the publisher with `keepDuplicates=true`, `sendAll=true`, and a short publication period such as 10 ms. Flush the publishing instance as the vision pipeline does; server-local publishers can otherwise encounter ntcore's separate 100 ms input pump. The timestamped reader requests 10 ms network updates rather than the default 100 ms. Neither setting guarantees delivery within the 20 ms nearest-sample tolerance: stale samples still reject, and a stationary heading must continue producing timestamped measurements.

The 2D operation accepts `{"detections": TimedValue[list[Detection]], "gyro_samples": list[dict]}`. Detections must retain the actual image capture timestamp. It emits `camera_pose`, `pose_meta`, and `diagnostics` (including alignment/rejection `reason`); missing/stale/invalid heading rejects the current pose rather than substituting a held pose. The model assumes a level robot and solves field XY with gyro-constrained heading, not arbitrary 6-DoF robot pose.

Use the same `camera_bus_id` for device input, solver and camera-to-robot conversion, and inject the production `CameraConfigRegistry`. Intrinsics must match the replay image resolution/distortion model; mounting pitch/yaw/roll are degrees and offsets are meters. The current mounting transform is part of the constrained model, so correct camera height/orientation/translation is essential. Benchmarks load those exact manifest calibration and mounting values into an isolated production registry, while both solvers consume the same field map. Real deployment needs an actual synchronized gyro publisher and calibrated mounting; synthetic oracle improvements alone do not validate physical sensor accuracy.
