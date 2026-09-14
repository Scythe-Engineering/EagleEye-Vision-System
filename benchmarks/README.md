# Local accuracy benchmark

This package replays finite, hash-pinned local video clips through EagleEye's production full-frame and temporal localization graphs. It scores every frame for AprilTag detection and pose accuracy, and writes a static offline report with diagnostic images. Synthetic replay does not measure physical camera transport or application-loop performance.

## Dataset

Provide a `DatasetManifest` whose `Asset` entries contain safe relative paths, byte sizes, and SHA-256 hashes. Place calibration, truth, and event assets at their content-addressed cache paths under `~/.cache/eagleeye/benchmarks` (or a directory supplied with `--cache-dir`).

`run` downloads missing selected video assets from the configured ZIP archive, shows download and extraction progress bars, verifies each video against the manifest, then deletes the temporary ZIP. Override the source with `--archive-url URL`.

## Verify and run

```bash
uv run python -m benchmarks verify path/to/manifest.json --subset pilot
uv run python -m benchmarks run --subset pilot \
  --pipeline both --output benchmark-results/pilot
```

When `--dataset` is omitted, `run` derives `*_manifest.json` from the `--archive-url` ZIP URL and caches it under the benchmark cache directory. Pass `--dataset path/to/manifest.json` to use a local manifest instead. Delete the cached manifest to fetch it again.

Use `--manifest-sha256` to pin the manifest itself and `--overwrite` to replace an existing output directory. `--pipeline` also accepts `full-frame` or `temporal`. For a quick partial evaluation, pass `--timeout SECONDS`; the run stops between frames and writes a valid partial report. Without it, the full selected dataset runs.

A run writes `run.json`, gzip JSON Lines frame records, `summary.json`, `summary.csv`, a dependency-free `index.html`, and a bounded diagnostic image set. Provenance includes manifest and graph hashes, calibration/map identities, repository state, dependencies, and platform details. Exit code `0` means reporting completed without pipeline frame failures; invalid input, missing or corrupt assets, and pipeline failures return `2`.
