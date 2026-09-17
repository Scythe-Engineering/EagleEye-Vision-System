# Pose accuracy and availability experiments

## Scope

Baseline source: `d58bfd1`. Both versions replay the same seven clips, all 22,080 frames at 1280 × 800. Manifest SHA-256: `57fc05dccabd81f82450781cf169a07a3ee5050c78cb6d2d5ea980cafbefe533`.

The primary comparison allows one-tag poses, matching the published EagleEye/PhotonVision page. The checked-in benchmark and localization preset still require two detections; their gate was not lowered to manufacture an availability gain. That configuration was also checked locally.

Final hardware is a Raspberry Pi Compute Module 5 Rev 1.0, with the existing `ondemand` governor, OpenCV 4.11.0 and four OpenCV threads, NumPy 1.26.4, and pupil-apriltags 1.0.4.post11. ROI detection uses one thread; direct full-frame recovery uses two. The Rust projector was rebuilt from current source in release mode in a separate test directory. Its binary SHA-256 is `6a7dfc755e9c44523921c0b602aa0ce39c7dd5cfc1f01b06865d51049474f231`.

The published comparison used a different compiled projector. Its 7.48 cm mean 3D error and 71.14% provisional matching rate are not the baseline for claiming these changes' benefit. Fresh current-source baseline runs measured about 6.16 cm and 75.40%, respectively.

Timing measures sequential pipeline inference, excluding decoding, scoring, file output, physical camera transport, the outer application loop, UI and NetworkTables delivery. These results are not sustainable live FPS measurements.

## Where the baseline falls behind

The public page's graphs use rolling medians. This investigation used unsmoothed frame records instead.

- Distant single-tag estimates account for the largest failures. The current edge baseline's single-tag mean/p99/max 3D errors were 10.34 cm / 1.505 m / 8.033 m. Two-tag maximum error was 0.999 m; three-or-more-tag maximum was 0.229 m.
- Some seven-to-eight-meter jumps have only about 0.06–0.14 pixels reprojection error. Small reprojection residuals alone cannot resolve planar ambiguity.
- North-lane approach/retreat and south-lane sprint/turn expose these failures. Detection and pose changes must be tested through the full temporal feedback loop: changing corners or candidate selection also changes later ROIs.
- Counts of provisionally visible tags are not counts of tags contributing to PnP. A frame with several geometric visibility labels can still be a single-tag solve.

## Retained behavior

1. **Preserve detail in tiny ROIs.** A region whose shorter side is below 32 pixels uses a reused decimation-1 detector. Ordinary crops and direct full-frame detection retain their existing settings. `small_roi_max_px=0` disables this behavior.
2. **Resolve nearby single-tag ambiguity.** Initial IPPE hypotheses within 0.25 pixels RMS of the best image fit can be ranked against a recent field-space camera pose. Selection requires a candidate within 1 m and 0.35 radians of that pose, both before and after LM. Otherwise ordinary image-only solving is used. Multi-tag solving remains image-only.
3. **Reject large discontinuities without emitting an old pose.** If a single-tag result is over 2 m or 0.75 radians from a valid recent pose, both output ports are `None`. The previous estimate is never substituted or blended into the output. Rejections do not renew history, which expires after 250 ms of capture time. Missing, repeated and backward timestamps clear history. `use_pose_continuity=false` disables both history-based behaviors.

The jump gate was explicitly approved during the experiment, conditional on a net availability gain. A stricter 1 m / 0.35 rad output gate failed that condition and was discarded.

## Final edge validation

Two original-source baseline runs and two final-production runs completed without timeouts or failed frames. Per-frame detections, pose availability, errors and false-detection counts reproduced exactly within each pair. Source/config/native hashes were checked. There were no thermal-throttling flags; recorded temperature stayed below 69°C.

Quality counts below are per 22,080-frame run. Timing combines the two runs of each version.

| Metric | Baseline | Retained changes |
| --- | ---: | ---: |
| Pose frames | 19,810 (89.72%) | 19,853 (89.91%) |
| Provisional matched tags | 38,571 | 40,939 (+6.14%) |
| Scored false positives / wrong IDs | 3 / 0 | 2 / 0 |
| Mean 3D error | 0.06156 m | 0.04698 m (−23.7%) |
| p95 3D error | 0.20300 m | 0.20022 m |
| p99 3D error | 0.86383 m | 0.33038 m (−61.8%) |
| Maximum 3D error | 8.03348 m | 2.33653 m |
| Mean / p95 XY error | 0.04139 / 0.08354 m | 0.03112 / 0.07007 m |
| p99 rotation error | 0.13898 rad | 0.08949 rad |
| Mean processing time | 5.889 ms | 5.908 ms (+0.31%) |
| p95 processing time | 22.039 ms | 21.913 ms (−0.57%) |
| p99 processing time | 27.407 ms | 27.352 ms |
| Maximum recorded processing time | 102.995 ms | 219.343 ms |

Even comparing the slower candidate run against the faster baseline run, mean and p95 remain within the agreed 10% cap. The maximum is a regression, not covered by that cap: both candidate repeats have a 213–219 ms frame at south-lane frame 2774, mostly charged to PnP by the instrumentation. Re-solving the identical two-tag corners 50 times separately reproduced the pose but not the pause: median 1.17 ms, maximum 2.26 ms. The whole-replay pause remains unexplained; these measurements do not establish a worst-case latency bound.

The net availability gain is **90 gained pose frames minus 47 lost frames = 43**. This is not merely lower error from suppressing output: on 19,763 shared pose frames, mean/p99 error fell from 0.05982 / 0.82707 m to 0.04467 / 0.26933 m. Across all 22,080 inputs, frames with an available pose below 10 cm error increased from 17,823 to 18,037; those below 25 cm increased from 19,407 to 19,595.

On 19,685 common adjacent-frame pairs, mean/p99 translation-error step fell from 0.03654 / 0.52819 m to 0.01986 / 0.22333 m. No averaging filter or repeated output pose was used.

For frames whose nearest provisionally visible tag is beyond 6 m, pose count increased from 782 to 842. Mean/p99 3D error fell from 0.60099 / 7.22239 m to 0.29411 / 1.95829 m. Every clip's aggregate p95, p99 and maximum 3D error improved or stayed equal on the edge, but individual frames did not uniformly improve.

**Remaining handoff regression:** the under-2-m geometric-label cohort lost 30 pose frames, and its p99/maximum error increased from 0.03411 / 0.30994 m to 0.03544 / 2.08471 m. At south-lane frame 1040, baseline detected nearby tag 7 and had 4.8 mm error; the candidate still used distant tag 17 and had 2.085 m error. Its estimated contributing-tag distance was 8.41 m, despite a near tag being available. Better tiny-tag detection can keep the ROI path active instead of triggering full-frame recovery. This remains a weakness, not a near-range improvement claim.

`final-verification.json` contains the checked aggregates and all four timing distributions. `edge-final-comparison.json` includes per-clip, distance, shared/gained/lost and fixed-denominator results.

## Local regression checks

The retained one-tag configuration produced 19,853 poses versus 19,826, and 40,901 provisional tag matches versus 38,561. Mean/p99/max 3D error changed from 0.05958 / 0.83940 / 7.93538 m to 0.04660 / 0.29098 / 2.33565 m. Both runs scored three false positives and zero wrong IDs. A repeat reproduced every frame's detected IDs, pose availability, pose errors and false-positive count.

With the existing two-detection gate, poses increased from 12,361 to 12,760 and provisional matches from 38,554 to 40,461. Mean 3D error fell from 0.03619 to 0.03493 m and maximum error from 0.99936 to 0.50231 m. Overall p99 increased by 0.33 mm as coverage expanded; shared-frame p99 improved from 0.21613 to 0.21391 m. False positives fell from seven to five, with zero wrong IDs. Final gate changes reproduced the earlier two-tag candidate's per-frame quality results exactly.

Disabling both new settings reproduced the original local baseline's per-frame quality results exactly.

## Rejected experiments

| Experiment | Reason not retained |
| --- | --- |
| Original-image cornerSubPix | Worse error tails and lower availability through the feedback loop. |
| Simple original-image gradient/line refinement | Worse far-field errors, lower availability, and excessive overhead. This does not rule out a better appearance-model refinement. |
| Native re-detection of small tags in original-image crops | Additional cost and new large outliers. |
| Decimation 1 for all small/medium ROIs | More detections, but worse near/mid-range tails. Limit the override to tiny crops instead. |
| Decimation 1.5 | New large outliers; changing full-frame decimation also exceeded the local latency budget. |
| Refine only the image-best IPPE initializer | No pose improvement in offline screens. |
| Per-tag residual outlier rejection | No accuracy or availability benefit in the tested offline screen. |
| Unbounded previous-pose tie-breaking | Bad previous poses could propagate failures. |
| Continuity without output rejection | Improved aggregate tails but introduced a 6.5 m loss/reacquisition outlier. |
| Strict 1 m / 0.35 rad output rejection | Better accuracy but a net availability loss, contrary to the agreed acceptance rule. |

## Limits

These are synthetic, benchmark-tuned replay results, not an independent physical-camera validation. Visibility labels are provisional geometric labels without verified occlusion masks, so matching rate is a proxy, not certified recall. Scored false positives and wrong IDs do not establish real-world precision.

Pose errors are conditional on availability. Comparisons also retain common-frame errors, gained/lost pose sets, fixed-denominator accuracy-threshold counts, and consecutive translation-error changes on common adjacent frames. The last metric subtracts ground-truth motion; it is an error-step diagnostic, not a physical stationary-jitter measurement.

The previous pose can still be wrong, and an abrupt real camera movement can trigger rejection. The history timeout bounds rejection lockout, but an accepted wrong branch can keep refreshing history. It does not guarantee correction of a bad initial branch. A live-camera and robot-motion trial remains necessary before deployment.

## Artifacts and reproduction

The raw experiment scripts, graph variants, per-frame records, summaries, and paired comparisons used during investigation are intentionally ignored under `benchmark-results/pose-accuracy-session/`. The isolated edge candidate directory was `/home/eagleeye/pose-accuracy-session`; the untouched-source baseline directory was `/home/eagleeye/pose-accuracy-baseline`. They shared existing video assets and the same release native library, not the live deployment's source files.

For an independent clean-checkout one-tag replay, use the tracked one-tag graph. It writes standard temporal-experiment output rather than the unpublished diagnostic records from this investigation:

```bash
uv run python -m benchmarks.temporal_experiments \
  --config benchmarks/configs/temporal-one-tag.json \
  --opencv-threads 4 --output benchmark-results/pose-accuracy-recheck
```

Validation: 47 relevant tests passed, covering the actual large-ambiguity fixture, untimed/stale/repeated/backward capture times, disabled behavior, rejection/reacquisition, unchanged multi-tag solving, timing propagation, tiny/full-frame detector routing and live disabling. Focused Ruff checks, four-file mypy checking and `git diff --check` passed.

The live edge deployment was not updated; its service was confirmed active after the isolated runs. Physical-camera trials and a fresh matched PhotonVision comparison remain outstanding.
