# 2D PnP validation

Branch: `feat/2d-pnp`, based on main `2fb666c424e205c309062416f05d8896bb39e31e`. Validation ran against the uncommitted implementation recorded by the source fingerprints below. No deployment was made.

## Implementation

- Separate `pnp_camera_localization_2d` operation: finite field-aligned NWU yaw, level robot, robot-origin field Z=0; solve XY using existing intrinsics and six mounting extrinsics. Live mounting edits apply. Missing/stale/invalid gyro or failed geometry yields no pose and a diagnostic, never a normal-PnP fallback.
- Timestamped generic NT input uses native payloads and `Value.time()` in the receiving machine's NT clock. Publishers supply the original measurement timestamp through `.set(value, timestamp_us)`, reversing the existing AprilTag contract. No JSON timestamp envelope or manual clock-offset arithmetic. Unsynchronized clients discard history. Legacy scalar mode remains unchanged.
- Timestamped subscriptions request 10 ms updates. Unflushed native relay controls measured median batch gaps of 100.14 ms before versus 9.68 ms after; median measurement ages were 61.41 versus 14.21 ms. This is not a 20 ms delivery guarantee. See [wiring and publishing requirements](README.md#production-wiring-and-calibration).
- Benchmark comparisons include both pipelines, minimum tag counts 1/2, identical-detection shadow solves, independent temporal feedback, causal heading noise/bias/delay, rejection/availability and latency metrics.
- User-approved scheduler cleanup stops/joins owned workers before freeing operations; timeout preserves collaborators for retry. Native Pupil cleanup destroys the detector before its families. Backend shutdown preserves shared cameras/MX3 runtimes if a pipeline cannot drain.

Normal PnP solving code and the historical benchmark defaults were not changed.

## Verification

- Local full suite: **465 passed, 8 expected skips**.
- Isolated Pi suite: **448 passed, 8 expected skips**, excluding 17 unrelated installer tests.
- Edited Python lint: **zero new findings**; 66 pre-existing findings remain (67 in the base versions). Relevant formatting passed. Focused mypy passed; untyped external Pupil imports excluded from that check.
- Fresh main WebUI build passed. Browser smoke was unavailable because `agent-browser` was not installed; this is not a browser-verified release.
- All **19 source fingerprints** in the final local/Pi run metadata match the current code. All **31 staged candidate files** matched before/after Pi testing.
- The isolated Pi borrowed dependencies read-only. Production git status and all **1,202 shared native-library fingerprints** remained unchanged. `eagleeye.service` was restored and independently rechecked active.

## Final replay scope

All variants use the existing one-thread detector presets. Prefix runs are partial datasets, not whole-clip accuracy results. Counts below distinguish processed cycles from valid robot poses.

| Run | Variant-frames attempted | Completed cycles | Valid poses | Gated cycles | Failed cycles |
| --- | ---: | ---: | ---: | ---: | ---: |
| Local, two clips × first 120 frames × eight variants | 1,920 | 1,847 | 1,847 | 73 | 0 |
| Pi, seven clips × first 120 frames × eight variants | 6,720 | 5,195 | 5,195 | 1,525 | 0 |
| Pi, complete 480-frame sprint, ideal heading | 3,840 | 2,356 | 2,352 | 1,484 | 0 |
| Pi, complete sprint, perturbed heading | 3,840 | 2,354 | 2,342 | 1,486 | 0 |

Perturbation: seed 2026, Gaussian heading noise 0.002 rad, bias 0.0035 rad, delivery delay 10 ms. Only truth rotation is supplied as gyro input, never truth position.

## Whole-sprint Pi results

XY RMSE and median latency below are independent full-pipeline measurements. Availability denominators are all 480 frames, not only successful/common frames.

| Heading | Pipeline | Min tags | Poses normal → 2D | XY RMSE mm normal → 2D | Pipeline median ms normal → 2D | Solver median ms normal → 2D |
| --- | --- | ---: | --- | --- | --- | --- |
| Ideal | Full-frame | 1 | 442 → 442 | 37.68 → 14.86 | 25.98 → 27.02 | 1.06 → 1.43 |
| Ideal | Full-frame | 2 | 158 → 158 | 34.56 → 6.83 | 25.81 → 26.10 | 1.44 → 1.43 |
| Ideal | Temporal | 1 | 444 → 444 | 44.72 → 16.56 | 10.49 → 11.18 | 1.06 → 1.39 |
| Ideal | Temporal | 2 | 132 → 132 | 33.14 → 6.38 | 19.18 → 18.80 | 1.45 → 1.37 |
| Perturbed | Full-frame | 1 | 442 → 440 | 37.68 → 32.87 | 26.03 → 27.00 | 1.07 → 1.52 |
| Perturbed | Full-frame | 2 | 158 → 156 | 34.56 → 36.58 | 25.86 → 26.03 | 1.44 → 1.44 |
| Perturbed | Temporal | 1 | 444 → 442 | 44.72 → 31.81 | 9.91 → 11.62 | 1.07 → 1.48 |
| Perturbed | Temporal | 2 | 132 → 128 | 33.14 → 33.64 | 19.01 → 18.92 | 1.45 → 1.44 |

Ideal full-frame one-tag p95 improved 83.85 → 36.43 mm; maximum improved 181.43 → 99.21 mm. Two-tag p95 improved 64.56 → 13.85 mm. No final available XY error exceeded 1 m; this does not mean there were no smaller outliers.

Equal availability does not mean identical frames: ideal temporal/min2 had 115 common poses, 17 lost and 17 gained. Perturbed temporal/min2 had 110 common, 22 lost and 18 gained. Identical-detection comparisons lost zero ideal poses; the perturbed shadow comparisons lost two initial poses from missing gyro. Full individual/common-frame tails and rejection reasons remain in the detailed artifacts.

## Limits and measured exceptions

- This is **not an unconditional accuracy or speed improvement**. Perturbed two-tag RMSE worsened. Even ideal heading worsened north-lane full-frame/min1 prefix RMSE 12.58 → 13.36 mm, although p95 improved 22.95 → 13.61 mm. Several 2D pipeline/solver medians are slower.
- Synthetic replay does not validate a physical gyro, camera transport, tilted/elevated robots, the outer threaded application loop, or preview performance. Deployment needs calibrated mounting and synchronized, field-aligned heading.
- Raw Pupil 1.0.4.post11 also reproduced the native two-thread detection fault after cleanup/recreation. Owned cleanup does not fix that worker-pool risk. User chose unchanged one-thread validation and documentation, not a dependency upgrade. Multithreaded reliability is not established.
- Stock `tagCircle49h12` construction consumed about 6.2 GiB and OOM-killed the initial broad cleanup test on the Pi. The ownership regression now covers small/target/multi-family cases; production family support and constructor/detection semantics are unchanged.
- First multi-clip validation OOM was a separate scheduler leak: sixteen short local pipelines grew roughly 294 MB → 2.34 GB with retained workers/detectors. After cleanup, repeated pipelines stayed near 190 MiB with no retained owned workers/detectors. Final Pi sampled benchmark process-tree peak was **457.9 MiB**; the 100 ms sampler may miss short peaks or double-count shared pages. No thermal throttling was observed. Existing swap usage was high from earlier attempts.

## Evidence and reproduction

Complete detailed report: `/tmp/eagleeye-pi-complete-validation-local/FINAL-VALIDATION-REPORT.md`.

- Final local results: `/tmp/eagleeye-2d-pnp-work/local-final-complete/`.
- Copied Pi results: `/tmp/eagleeye-pi-complete-validation-local/eagleeye-2d-pnp-complete-results/`.
- Remote isolated results: `/tmp/eagleeye-2d-pnp-complete-results/`.
- Each completed run retains `run.json`, `summary.json`, `frames.jsonl.gz`, HTML and diagnostic images. Wrapper commands, calibration/asset manifest, telemetry, native fingerprints and restore proofs are included. Earlier failed attempts remain separate; their partial counts were not merged into final results.

Use the commands in [benchmarks/README.md](README.md) with the original manifest/cache. The exact isolated Pi commands and environment are saved in the copied `eagleeye-pi-complete-validation-wrapper.sh`. Large artifacts stay outside the repository.
