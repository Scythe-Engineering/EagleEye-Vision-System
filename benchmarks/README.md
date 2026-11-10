# Benchmark documentation

The benchmark commands, dataset contract, Blender workflows, and report interpretation now live in the [EagleEye documentation site](https://scythe-engineering.github.io/EagleEye-Docs/docs/codebase/synthetic-video-benchmarks).

The implementation remains in this directory. Start with `uv run python -m benchmarks --help` or use the documentation for the full workflow.
## Comparing detector builds

`temporal_experiments` replays the cached clips through a real pipeline graph. It writes per-frame rows to `frames.jsonl.gz` and totals to `summary.json`.

```sh
uv run python -m benchmarks.temporal_experiments \
  --config benchmarks/configs/temporal_rust.json --output /tmp/rust-after
```

- `configs/temporal.json` uses pupil-apriltags. `configs/temporal_rust.json` uses the Rust detector with its production defaults.
- A config can live anywhere, but its graph has to match the `apriltag_localization` template in `src/webui/js/pipeline/pipelineTemplates.json`. Run from a full checkout.
- `--output` must be a directory that doesn't exist yet.
- To compare two builds, run each one and diff the `detections` and `camera_pose` fields of the frame rows, then compare `pipeline_ms` and `native_detector_ms` in the summaries.
