# Blender match generator

Rendered media belongs in `/tmp` or ignored `benchmark-results/`; do not track
rendered assets. The retained production collection is six CAD-backed,
30-second routes (180 seconds / 21,600 frames total), each rendered only as
`combined-realistic`. That variant composes lighting, exposure, sensor noise,
defocus, motion blur, and lens distortion. `clean` remains solely for the
three-frame local smoke recipe.

The generator validates route motion, robot/blocker clearance against the CAD
field, field dimensions, and AprilTag mounting faces before rendering. It
records CAD/map hashes, pose and corner truth, and validation provenance. The
actual field mesh is required for match routes.

## Three-frame clean smoke

This requires Blender 4.4 or later and FFmpeg.

```bash
blender --background --python-exit-code 1 --python benchmarks/blender/generate.py -- \
  --recipe benchmarks/blender/smoke-recipe.json --output /tmp/eagleeye-blender-smoke \
  --start-x 15 --start-y 4.0345 --yaw-degrees 180 --speed 0
uv run python benchmarks/blender/package.py \
  --frames /tmp/eagleeye-blender-smoke/frames \
  --output /tmp/eagleeye-blender-smoke/clean.mkv --fps 120
```

Frames are 16-bit PNG intermediates. Packaging writes lossless FFV1/bgr0,
decodes every frame, and requires exact pixel hashes. Keep frames until the
manifest is verified; `--cleanup-frames` is the only removal path.

## Match collection

Inspect assignments, validate all CAD/motion routes, then render sparse
full-quality preflight frames without marking a full job complete:

```bash
uv run python benchmarks/blender/render_match.py --output benchmark-results/frc2026-media --list
uv run python benchmarks/blender/render_match.py --output benchmark-results/frc2026-media --validate-only
uv run python benchmarks/blender/render_match.py --output benchmark-results/frc2026-media \
  --device OPTIX --samples 64 --denoise --persistent-data --frame-indices 0,960,1920,2880
```

Start a resumable, losslessly packaged job by omitting `--validate-only` and
`--frame-indices`:

```bash
uv run python benchmarks/blender/render_match.py --output benchmark-results/frc2026-media \
  --device OPTIX --samples 64 --denoise --persistent-data \
  --jobs red-end-range-ladder:combined-realistic
```

Per-job `runner.log` and atomic `status.json` record progress. Resume skips
only jobs whose input signature, generated truth/provenance/settings digests,
FFV1 manifest, video size, and video hash all match. Use a new output directory
after changing render inputs.

Run the sole opt-in rendered end-to-end smoke check with:

```bash
EAGLEEYE_RUN_RENDERED_TESTS=1 uv run pytest -q tests/test_rendered_benchmark.py
```

Set `EAGLEEYE_BLENDER` and `EAGLEEYE_FFMPEG` to override tools.
