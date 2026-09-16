# Temporal pipeline experiments

## Scope and measurement

Baseline source revision: `10f5e2deb11789d133d1712f4261aecaab12e810`.
Manifest SHA-256: `57fc05dccabd81f82450781cf169a07a3ee5050c78cb6d2d5ea980cafbefe533`.

Dataset: `eagleeye-current-benchmark`, release `seven-clips-2026-09-15`. All seven clips contain 22,080 frames at 1280 × 800. The pilot has 4,080 frames. Videos and annotations were hash-verified; Pi runs used the original assets, not transcoded copies.

The dataset has provisional geometric visibility labels, not verified eligible tags. Official eligible recall remains null. Supplemental provisional matching is a proxy, not measured recall. Timing groups distinguish frames with provisional tags from frames without them. The annotations list all 32 field tags even when none is visible, so a nonempty `tags` array is not a visibility test.

Measurements use `pipeline_duration_ns`, excluding video decoding, scoring and report generation. These are sequential, single-pipeline replay inference timings, not camera-to-NetworkTables latency or sustainable application FPS. Concurrent cameras, frame dropping and other CPU workloads were not benchmarked. Operation profiles omit some incomplete graph cycles; their smaller denominators are reported and their timings must not be summed to estimate overall latency.

Short screens used identical contiguous prefixes, at most 600 frames per clip, rather than comparing how many frames different configurations processed before a timeout. Final comparisons use identical complete frame identities. Reacquisition is measured only within the next contiguous interval with provisional tags, never across a later disappearance.

## Changes

- Add opt-in `full_frame_nthreads`. Zero preserves existing behavior. A value of two uses a separate detector for full-frame searches while retaining single-threaded temporal crop detection. Unlike setting `nthreads=2`, it does not impose thread-pool overhead on every small crop.
- Map ROI detection corners directly instead of copying and normalizing center/homography metadata that ROI outputs discard. Full-frame metadata remains unchanged.
- Reuse camera-space corners during Rust culling/projection and compare frustum slopes without `atan`.
- Make the shared Rust builder use `maturin develop --release` for default builds and reinstalls. Invalidate old debug-build cache entries so normal startup does not silently retain unoptimized extensions.
- Add `python -m benchmarks.temporal_experiments` for bounded, fixed-frame experiments, compact records, visibility-stratified summaries and per-clip recovery checks. Existing scoring is reused unchanged.

The Pi deployment was not changed. Tests used a separate `~/temporal-performance-20260915` directory and its own native library builds. The installed Python environment supplied dependencies without being modified.

## Complete Raspberry Pi 5 validation

Both runs completed all seven clips, 22,080 identical frame identities, without failures or timeouts. Baseline and production had identical per-frame detected IDs, provisional matches, pose availability, scored pose errors and reacquisition results. Each produced 38,566 provisional matches and 12,358 poses. These counts differ slightly from the local machine, so comparisons stay within each platform.

| Frame group | Frames | Baseline mean, ms | Production mean, ms | Baseline p95, ms | Production p95, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 22,080 | 14.529 | 10.611 | 42.744 | 27.317 |
| With provisional tags | 20,418 | 13.141 | 9.799 | 42.587 | 27.252 |
| Without provisional tags | 1,662 | 31.577 | 20.592 | 44.592 | 27.665 |

Mean inference latency fell 27.0% overall and 25.4% on tag-present frames. Overall p95 fell 36.1%. The larger 34.8% mean gain on tag-free frames is reported separately, not used to imply faster tracking than measured.

The Pi used OpenCV 4.11.0, four OpenCV threads, the existing `ondemand` governor and separate release native libraries. No other benchmark ran concurrently. Temperature was 41.7°C before the baseline, 46.6°C afterward and 47.7°C after production; `get_throttled` remained `0x0`. Source hashes from the Pi production run match the retained Python implementation.

The mean remains above the 8.33 ms frame period of this 120 Hz dataset. These results do not establish live 120 FPS performance.

## ROI-only compiler attribution follow-up

A subsequent short Pi rerun measured actual ROI-only processing, rather than inferring it from visible tags or successful poses. Experiment-only instrumentation counted calls to the detector and required projected 3×3 crop mappings with no full-frame detection call. Whole-frame sentinel crops were explicitly excluded. Synthetic-tag checks verified ROI success, full-frame recovery, sentinel crops and direct full-frame inputs.

Each trial replayed 300 consecutive frames from each of the seven clips, 2,100 frames total. The first 30 frames per clip were excluded from the analysis. Exactly 905 frames qualified as ROI-only in every trial: 270 each from blue-end slalom, red-end range ladder and south-center curves, 76 from the custom sprint, and 19 from north-lane approach. The other two clips had no qualifying frames in these short prefixes. This is a short screen, not full-clip ROI-only validation.

Four configurations were repeated three times in different orders. All reported mean values below pool the same 905 frames across three repeats. OpenCV remained at four threads. The original detector's `run_detection` method was restored for the first three conditions; the last used the retained production implementation. Native library hashes confirmed three distinct debug/baseline-release/optimized-release binaries. The optimized Rust binary was identical in the final two conditions.

| Configuration | ROI-only mean, ms | Individual repeat means, ms |
| --- | ---: | --- |
| Baseline detector and Rust, debug build | 3.724 | 3.659, 3.737, 3.776 |
| Same baseline, release build only | 3.615 | 3.594, 3.680, 3.571 |
| Optimized Rust release, baseline detector path | 3.616 | 3.674, 3.540, 3.635 |
| Final production implementation, release | 3.484 | 3.467, 3.492, 3.494 |

The measured ROI-only mean reduction was **6.4% from the old debug-build default**, or **3.6% when comparing release builds on both sides**. Compiler settings alone saved about 0.109 ms; the complete change saved about 0.240 ms. That attributes about 45% of the saved time to release compilation using pooled means. Using median repeat means instead gives about 58%, so the defensible attribution is **roughly half**, not a precise split or a clear majority. The earlier 27% full-pipeline result already compared release builds on both sides and was not a compiler-setting gain.

The Rust math changes alone were not distinguishable from timing noise at the ROI pipeline level. On the 855 common ROI frames with complete operation profiles, the detector stage averaged 1.032 ms for baseline release and 0.889 ms for production. That supports removal of unused ROI metadata normalization as the main remaining code-level benefit. No full-frame detector call occurred on the selected frames, so they do not include the large direct benefit of full-frame threading.

Tail latency was not consistently improved in this short screen. Pooled ROI-only p95 was 6.074 ms for baseline release and 6.706 ms for production, with substantial variation between repeats. The full-pipeline p95 improvement must not be presented as an ROI-only p95 improvement.

ROI classification and all recorded quality fields were identical across every trial, including frames excluded from the timing analysis. The highest observed temperature was 49.4°C, and every throttling check returned `0x0`. The live deployment was untouched.

Reproduction scripts are `roi_ablation.py`, `run_roi_ablation.sh` and `analyze_roi_ablation.py` under `benchmark-results/temporal-performance-session/`. Records are in `pi/roi-ablation/`; `pi/roi-ablation-comparison.json` contains the analysis and `pi/roi-ablation-verification.json` records library hashes and path checks. The scripts do not modify production detector files.

## Raspberry Pi 5 screens

OpenCV 4.11.0, CPU/NEON, four OpenCV threads unless specified. OpenCL was unavailable. Each row covers the same 1,080 pilot-prefix frames using the same baseline release Rust library. These are screening results, not the final full-dataset comparison.

| Experiment | Mean inference, ms | Provisional matches | Poses |
| --- | ---: | ---: | ---: |
| Baseline | 12.017 | 1,804 | 732 |
| Two detector threads everywhere | 13.413 | 1,804 | 732 |
| Two threads only for full-frame searches | 9.788 | 1,804 | 732 |
| Convert to grayscale before warping | 12.151 | 1,805 | 732 |
| One inverse transform per perspective warp | 12.029 | 1,804 | 732 |
| Five LM iterations instead of ten | 11.993 | 1,804 | 732 |
| One OpenCV thread | 12.025 | 1,804 | 732 |
| VVS refinement instead of LM | 11.952 | 1,804 | 732 |
| Normalize only ROI corners | 11.936 | 1,804 | 732 |
| Route unwarped full-frame requests directly to base detector | 12.023 | 1,805 | 732 |

The full-frame-only threading experiment also passed a complete 4,080-frame Pi pilot comparison: mean 9.128 → 8.034 ms, p95 26.645 → 20.041 ms. Both runs produced 6,858 provisional matches and 2,671 poses. Detected IDs and scored pose errors were identical frame by frame. The complete dataset comparison above confirms a gain in both visibility groups.

## Complete local validation

Both release-library runs processed the same 22,080 frames. Detected IDs, provisional match counts, pose availability and all scored pose errors were identical frame by frame: 38,554 provisional matches and 12,361 poses in each run.

| Frame group | Frames | Baseline mean, ms | Production mean, ms | Baseline p95, ms | Production p95, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 22,080 | 5.029 | 3.752 | 14.835 | 10.174 |
| With provisional tags | 20,418 | 4.589 | 3.483 | 14.862 | 10.270 |
| Without provisional tags | 1,662 | 10.431 | 7.060 | 14.534 | 9.420 |

Local OpenCV was 4.13.0 with ten threads. Local timing drift was substantial: a later control with full-frame threading disabled measured 5.920 ms mean, while retaining exactly the same outputs. Treat local timings as supporting evidence, not precise Pi predictions. The tag-present improvement shows the gain is not solely from tag-free frames.

The temporal benchmark and new AprilTag-localization preset use `full_frame_nthreads=2`. Existing custom graphs keep the backward-compatible default of zero unless explicitly updated; mixed object-detection presets were not retuned.

## Rejected approaches

- Smaller padding, 0.25 instead of 0.35, and axis-aligned crops lost matches and pose availability in local fixed-prefix screens.
- Four threads everywhere improved one local screen but changed matches and poses; two threads everywhere were slower on the Pi.
- Larger ROI decimation, grayscale-first warps, inverse-map warps and reduced OpenCV threading did not show a convincing Pi inference gain. Warp variants can also alter interpolation rounding.
- Five LM iterations lost 54 provisional matches across the complete local dataset, with 36 poses lost and 33 gained. Some shared-frame translation errors worsened by up to 0.145 m. Three iterations also changed outputs in screening. Keep ten.
- VVS refinement did not improve Pi latency enough to justify changing pose solutions.
- Bypassing the initial decimate-three full-frame region search changed detections and poses without a repeatable full-run gain. Keep the current search order.
- GPU work stopped at capability inspection; this Pi OpenCV build had no OpenCL backend. No GPU dependencies or alternate runtime were added.

## Native projection probe

A seeded 2,000-pose probe produced identical complete region/quad hashes before and after the Rust change. Seven repeated batches measured median time per back-propagation plus projection call:

| Hardware | Baseline release, µs | Optimized release, µs |
| --- | ---: | ---: |
| Local Apple Silicon | 0.915 | 0.787 |
| Raspberry Pi 5 | 3.715 | 2.646 |

This is a microbenchmark, not a pipeline speedup. The originally installed local extension measured 8.908 µs; rebuilding the unchanged baseline in release mode accounted for most of that difference. Final comparisons use separate baseline and optimized release libraries rather than crediting build-mode differences to the code change.

## Checks

- 45 relevant Python tests passed; the opt-in Blender render test was skipped.
- Seven Rust tests and the pipeline-template UUID Node test passed.
- Mypy passed for the four changed Python production/benchmark modules.
- New experiment/test files passed Ruff checks and formatting. Existing detector files retain their pre-existing Ruff style findings; no unrelated cleanup was applied.
- The final standard local benchmark completed all 22,080 frames without failures. A raw-output comparison found no changes in detections, camera/robot poses, corner errors or tag matches. Its offline report is `benchmark-results/temporal-performance-session/final-official-local/index.html`.

## Reproduce

```bash
# Rebuild the native extension in release mode.
uv run python src/rust_implementations/build.py temporal_acceleration

# Complete run. Use a fresh output directory each time.
uv run python -m benchmarks.temporal_experiments \
  --config benchmarks/configs/temporal.json \
  --opencv-threads 4 \
  --output benchmark-results/temporal-recheck

# Fast screen across every clip, preserving state within each prefix.
uv run python -m benchmarks.temporal_experiments \
  --frames-per-clip 600 --timeout 90 \
  --output benchmark-results/temporal-screen
```

A timeout makes a result partial. Do not compare unequal frame sets. `metadata.json` records graph content/hashes, dataset identity, dependency/platform information, OpenCV thread count, production source hashes and loaded native-library hashes. `frames.jsonl.gz` contains compact per-frame measurements; `summary.json` contains group statistics and per-clip recovery events.

Local experiment artifacts, configuration variants, exploratory monkeypatches, comparison scripts, native probes and raw results are under `benchmark-results/temporal-performance-session/`. Experimental monkeypatches are not production implementations. Earlier screens predate native-library hash recording; their explicit library paths and separate release builds are retained alongside the results.
