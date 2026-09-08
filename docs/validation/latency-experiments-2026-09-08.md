# Further latency experiments — 2026-09-08

The previous 10 ms publisher change and first report were committed locally as
**577bfca** (cherry-picked as **bbadcf1**). PR #163 includes that change, the
candidate pipeline flush, optional robot example, flush tests, and diagnostic harnesses.
Raw CSV/JSON evidence and device backups remain on the author’s machines only.

The strongest small follow-up is **one NetworkTables flush after a completed
pipeline cycle plus a 5 ms robot odometry/vision callback**. On the actual camera
it measured **24.86 ms mean / 29.16 ms p95** capture-to-Java ingestion. The remaining
delay after the pose and metadata were both ready was **9.37 ms mean / 12.55 ms p95**.

## Changes included in this PR and local deployment

1. `src/config/utils/pipeline.py`: if a completed pipeline contains a NetworkTables
   publisher, call `network_table.getInstance().flush()` after `run_flow()` finishes.
   This batches all completed branches together, rather than flushing metadata
   before the pose branch finishes. The committed 10 ms publisher option remains.
2. `library/examples/localization-sim/src/main/java/frc/robot/Robot.java`: move
   odometry updates and SDK vision consumption into an optional 5 ms `addPeriodic` callback
   selected with `LOW_LATENCY_VISION`; the default remains the simple 20 ms path;
   keep the display on the 20 ms robot loop. All estimator access stays on one thread.
   The synthetic camera samples analytic ground truth at its own capture time,
   avoiding a timestamp mismatch with the preceding high-rate odometry callback.
3. The existing `/Users/dark/Custom-Apps/EagleEye-Java-Sim` bench has the equivalent
   5 ms callback and keeps its physical camera diagnostic-only.
4. Three pipeline flush ordering/completion tests, plus the optional harnesses in
   `tools/latency_experiments/`, are included in this PR.

The Pi is running the candidate pipeline flush with the regular backend entrypoint.
The temporary systemd override, UDP mirror, audit-topic publisher, CSV recorders,
and experiment worker threads have been stopped. Camera/pipeline/calibration files
are unchanged. The original 2 px production gate still rejects the physical camera's
approximately 2.7 px reprojection error; this experiment does not fix pose quality.

## Same-live-frame receiver comparison

All consumers ran simultaneously on this Mac's WPILib simulation server against
identical physical camera frames. The table uses the actual candidate pipeline
flush (scenario 7), 30 seconds, 3,610 frame IDs common to all six consumers.
There were 3,618–3,619 NT observations and 3,612 UDP observations in the receive
window; boundary differences are reported rather than asserted to be lossless.

| Receiver | Capture → ingestion mean / p95, ms | Result ready → ingestion mean / p95, ms |
| --- | ---: | ---: |
| Normal 20 ms robot loop | 32.31 / 42.30 | 16.82 / 25.40 |
| Synchronous `addPeriodic`, 5 ms | **24.86 / 29.16** | **9.37 / 12.55** |
| Separate `Notifier`, 5 ms | 24.88 / 29.05 | 9.39 / 12.30 |
| Separate `Notifier`, 1 ms | 22.73 / 25.82 | 7.23 / 9.06 |
| NT arrival callback | 22.24 / 25.29 | 6.75 / 8.53 |
| Diagnostic UDP receiver | 21.68 / 24.86 | 6.18 / 8.04 |

The Pi's mean capture-to-result-ready duration was 15.49 ms for those same frames.
Here "ready" means both the original Pose3d and its metadata have been published
locally. The subsequent full pipeline completion/flush can be a little later, so
this is a conservative measurement of the post-result segment rather than a claim
to timestamp every last profiling/UI instruction in the pipeline.

The 5 ms callback and separate 5 ms thread are effectively equivalent in these
measurements. A 1 ms thread buys another approximately 2 ms but wakes five times
as often. Arrival callbacks are fastest within NT4, but should enqueue observations
for one estimator owner rather than mutate an estimator concurrently with odometry.
The synchronous callback gives most of the benefit without that concurrency change.

[WPILib 2026.2.1 TimedRobot source](https://github.com/wpilibsuite/allwpilib/blob/v2026.2.1/wpilibj/src/main/java/edu/wpi/first/wpilibj/TimedRobot.java)
that `addPeriodic` runs synchronously with TimedRobot functions, whereas
[Notifier callbacks](https://github.com/wpilibsuite/allwpilib/blob/v2026.2.1/wpilibj/src/main/java/edu/wpi/first/wpilibj/Notifier.java)
run on a separate thread. Thread safety was considered explicitly in the experiments.

## Send cadence, flush, and protocol comparison

Link timing varied during the run, so matched-frame differences are more reliable
than subtracting averages from different scenarios. Each row below has more than
3,600 frames common to the compared receivers in its 30-second window.

| Sender mode | NT event ingestion minus UDP ingestion, mean ms |
| --- | ---: |
| 10 ms periodic, no explicit flush | 4.94 |
| Requested 5 ms periodic, no flush | 4.87 |
| Requested 1 ms periodic, no flush | 2.63 |
| Flush after both live branches ready | 0.37 |
| Repeat explicit flush | 0.52 |
| Repeat 10 ms periodic, no flush | 4.89 |
| Actual candidate: flush after pipeline completes | **0.56** |

The paired difference removes the shared capture/processing/clock offset from each
comparison. It does not establish identical network treatment of TCP and UDP.
Switching protocols offered only about half a millisecond of average improvement
over flushed NT4 in the clean candidate run. That alone does not justify replacing
NT4's schemas, clock sync, topic discovery, and existing SDK integration.

The source for installed WPILib 2026.2.1 explains the periodic behavior:
[ClientImpl.cpp](https://github.com/wpilibsuite/allwpilib/blob/v2026.2.1/ntcore/src/main/native/cpp/net/ClientImpl.cpp)
rounds publisher intervals to multiples of 10 ms;
[ClientImpl.h](https://github.com/wpilibsuite/allwpilib/blob/v2026.2.1/ntcore/src/main/native/cpp/net/ClientImpl.h)
sets the minimum to 5 ms. Thus a 5 ms request rounds to 10 ms, and a 1 ms request
rounds to zero then clamps to 5 ms. A 1 ms request is not an actual 1 kHz send rate.
The tested frame-completion flush avoids waiting for the periodic tick and remains
subject to ntcore's built-in rate limiting.

## Separate-thread odometry and vision fusion

A second test used the actual EagleEye Java SDK with a common 120 Hz synthetic
capture source, one 50 Hz main-loop estimator, and one independently owned 200 Hz
Notifier estimator. Both use WPILib's DifferentialDrivePoseEstimator. The physical
camera is not coupled to this synthetic motion and was never fused into it.

| Synthetic capture → SDK consumption | 50 Hz main loop | 200 Hz thread |
| --- | ---: | ---: |
| Mean, ms | 9.85 | 2.57 |
| p95, ms | 18.65 | 5.05 |
| p99, ms | 19.86 | 5.87 |
| Common frames in 30-second comparison | 3,601 | 3,601 |
| Observations accepted in complete 120-second run | 14,400 | 14,400 |
| Odometry updates over run | 6,002 | 24,001 |

Both final position errors were approximately 1.81e-6 m in this deterministic,
no-noise test. This checks implementation/clock consistency, not physical accuracy.
The desktop is not hard real-time: the 200 Hz worker's worst observed update gap
was 16.10 ms; the main loop's was 26.65 ms. The test stopped itself at two minutes,
so its final dashboard count snapshot is intentionally stationary.

Each estimator/subscriber had one thread owner and published an immutable snapshot.
The legacy bench briefly changes a static SDK threshold for its diagnostic poll;
all such polls shared a lock during the thread experiment, preventing that relaxed
threshold from leaking into the strict estimator path. The final synchronous
callback avoids adding this shared-state complication to the production example.

## Tail latency and unresolved behavior

Do not interpret the clean-window means as a worst-case guarantee.

- Scenario 4 had correlated large tails on both transports (roughly 77–80 ms p95
  capture-to-arrival), despite a small paired NT-vs-UDP difference.
- Scenario 6 recorded a severe NT backlog: raw NT event samples reached **29.15 s**
  capture age while fresh UDP observations continued. A common-frame intersection
  excluded many of those delayed frames, so its paired summary must not be used
  to claim low loss or bounded latency. The full CSV and raw-lane summary are kept.
- The precise cause of that backlog was not isolated. Device TCP statistics showed
  retransmissions, but this does not prove whether the stall originated in the
  network, the NT client/server, or the desktop runtime. It needs an independently
  controlled fault/reconnect test before claiming a transport reliability advantage.
- The UDP probe counts sequence order but does not implement a production receiver
  with independent clock sync, authentication, replay/freshness policy, or fusion.
  None of its packets were allowed to move robot state.

All one-way durations use NT4's estimated clock offset. Clock asymmetry/error was
not independently calibrated; no remote wall-clock subtraction or photon-to-actuator
claim is made. Camera resolution/120 FPS, current auto exposure, detector settings,
10 PnP iterations, map, and calibration were held fixed. Changing them would require
separate accuracy/exposure testing. The Pi reported no throttling.

## Validation, reproduction, and rollback

- **36 Python tests passed on the actual Pi**, including publisher timestamps,
  frame limiting, capture timing, and the new flush tests.
- Instrumented bench, final bench, and repository Java example tests/build passed.
- The final bench uses the 5 ms callback without diagnostic transport/CSV helpers.
- Raw data and summaries are at
  `/Users/dark/Custom-Apps/EagleEye-Java-Sim/evidence/latency-experiments-2026-09-08/`.
  Important files: `scenario-{1,2,3,4,5,6,7,8}.{csv,json}`,
  `paired-differences.json`, `fusion-summary.json`, `fusion-snapshot.json`, and build logs.
  Full `lanes-*.csv` files preserve samples outside selected windows.
- Harness reproduction notes: `tools/latency_experiments/README.md`.
- Pi pre-experiment pipeline backup:
  `/home/eagleeye/eagleeye-latency-2026-09-08/pipeline.py.before-flush`.
  Restoring it to `src/config/utils/pipeline.py` and restarting `eagleeye.service`
  reverts the new flush while retaining the committed 10 ms publisher change.
- Bench pre-experiment Robot source: `evidence/latency-experiments-2026-09-08/Robot.java.before`.

No new commits were made after 577bfca. Existing unrelated WebUI edits were preserved.

Reproducing the recorded results or using `pipeline.py.before-flush` requires access
to the author’s Mac and Pi; those artifacts are not downloadable from this repository.
Other operators can run the supplied harnesses to collect new evidence. For source-only
rollback of the flush, restore `src/config/utils/pipeline.py` from base commit
`9d7c5debcdaf02389ccca0bcfa8a8a88d9ac0eac` in your deployment and restart the backend.

PR review subsequently tightened harness shutdown, reduced the diagnostic lock scope,
and made evidence paths configurable. The measurements above describe the original
September 8 run; they were not remeasured with these harness maintenance changes.
