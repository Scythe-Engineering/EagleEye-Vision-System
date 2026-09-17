# Local accuracy benchmark

This package replays finite, hash-pinned local video clips through EagleEye's production full-frame and temporal localization graphs. It scores every frame for AprilTag detection and pose accuracy, and writes a static offline report with diagnostic images. Synthetic replay does not measure physical camera transport or application-loop performance.

## Dataset

Published datasets use two archives: `EagleEye-current-benchmark-videos.zip` contains the videos, while `EagleEye-current-benchmark-metadata.zip` contains `manifest.json` plus every calibration, ground-truth, and events file referenced by it. `run` downloads both archives into the repository root and reuses them for later runs. Missing selected assets are extracted into the ignored `benchmarks/cache` directory and verified against the manifest. Before writing, the downloader checks that the target filesystem has room for the archive and extracted assets. Override the videos source with `--archive-url URL`; it must end in `-videos.zip` so the metadata URL can be derived.

## Verify and run

```bash
uv run python -m benchmarks verify path/to/manifest.json --subset pilot
uv run python -m benchmarks run --subset pilot \
  --pipeline both --output benchmark-results/pilot
```

When `--dataset` is omitted, `run` reads `manifest.json` from the metadata ZIP and retains a copy in the repository root. Pass `--dataset path/to/manifest.json` to use a local manifest instead. Delete the downloaded ZIP and manifest copy to fetch them again.

Use `--manifest-sha256` to pin the manifest itself and `--overwrite` to replace an existing output directory. `--pipeline` also accepts `full-frame` or `temporal`. For a quick partial evaluation, pass `--timeout SECONDS`; the run stops between frames and writes a valid partial report. Without it, the full selected dataset runs.

During processing, the CLI shows frame progress, elapsed time, and estimated time remaining. When it finishes, it prints the total processing time and stores it as `processing_seconds` in `summary.json`. Timing starts after dataset loading and setup.

For performance tuning, `uv run python -m benchmarks.temporal_experiments --frames-per-clip 600 --timeout 90 --output benchmark-results/temporal-screen` replays equal-length contiguous prefixes and writes compact per-frame timings, visibility-stratified summaries, and reacquisition measurements. Use `--config path/to/temporal.json` for a configuration variant and omit `--frames-per-clip` for full validation. Do not compare unequal timeout-limited frame sets. See [temporal experiment results](temporal-performance-results.md) for the tested approaches and measurement limits, and [pose accuracy results](pose-accuracy-results.md) for tiny-ROI detection, single-tag continuity and repeated edge validation.

A standard `run` writes `run.json`, gzip JSON Lines frame records, `summary.json`, `summary.csv`, a dependency-free `index.html`, and a bounded diagnostic image set. Provenance includes manifest and graph hashes, calibration/map identities, repository state, dependencies, and platform details. Exit code `0` means reporting completed without pipeline frame failures; invalid input, missing or corrupt assets, and pipeline failures return `2`.
