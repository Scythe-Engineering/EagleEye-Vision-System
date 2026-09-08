# Uncommitted live latency experiments

These are diagnostic harnesses, not a replacement robot communications protocol.
Results and limitations: `docs/validation/latency-experiments-2026-09-08.md`.
Raw measurements are in the existing sibling `EagleEye-Java-Sim` workspace under
`evidence/latency-experiments-2026-09-08/`.

## Components

- `pi_launcher.py`: starts the existing backend with temporary publisher wrappers.
  It preserves the live pose/meta payloads, stamps the instant both are ready, and
  mirrors the same diagnostic packet into an NT4 topic and a UDP datagram.
  UDP destination is this test Mac, `100.75.14.59:5810`. It borrows NT4 clock sync.
  Run as `python -m tools.latency_experiments.pi_launcher` from the Pi repository.
- `LatencyExperiment.java`: simultaneous 20 ms main-loop, 5 ms addPeriodic,
  5 ms Notifier, 1 ms Notifier, NT event, and UDP consumers. It checks the live
  pose/meta pair against the diagnostic packet by capture timestamp and payload.
  Its CSV recording stops after 15 minutes. UDP data never feed an estimator.
- `HighRateVisionExperiment.java`: two independent estimators receive the same
  120 Hz synthetic captures; a 200 Hz Notifier owns one and the main loop owns
  the other. CSV recording and worker threads stop after two minutes. The legacy
  bench's temporary static diagnostic gate requires the shared SDK lock shown in
  the instrumented Robot. This is bench-specific and not a general SDK API.
- `Robot.instrumented.java`: copy as the bench's `src/main/java/frc/robot/Robot.java`
  together with the two experiment classes to reproduce the tests. Save any current
  user edits first. Run WPILib Sim GUI. The repository Java SDK must be on the classpath.
- `Robot.low-latency.java`: final bench variant, with a synchronous 5 ms
  odometry/vision callback and the existing 20 ms display loop. No UDP or CSV recorder.
- `summarize.py`: snapshots the latest complete 30/40 second receive-time window for
  a scenario. It reports common-frame comparisons and counts per lane. Do not use
  the common-frame subset to assess delivery reliability: inspect raw lane tails,
  sequence gaps, and unmatched counts, particularly for scenario 6.
- `fusion_snapshot.py`: reads the two-minute estimator test's status topics.

The Pi launcher reads `/tmp/eagleeye-latency-experiment.json` every 200 ms:

```json
{"periodic": 0.01, "flush": false, "scenario": 1, "udp": true}
```

Supported requested periods are `.01`, `.005`, `.001`. `flush: true` flushes when
both live outputs are ready. If the candidate pipeline-level flush is installed,
`flush: false` in this config does **not** disable that candidate; restore the
pre-flush pipeline to reproduce unflushed baselines. Exclude connection warmup,
scenario transitions, and experiment restarts from steady-state comparisons.

For this run a temporary `/run/systemd/system/eagleeye.service.d/90-latency-experiment.conf`
launched the wrapper. **That override has been removed** and the regular service
entrypoint restored. The final running setup has no UDP mirror or experiment topics.
Do not install a permanent service override merely to reproduce a benchmark.

The UDP experiment has no independent clock synchronization, authentication protocol,
retransmission, production freshness policy, or robot fusion integration. Its limited
source-address/framing checks are diagnostic checks, not production hardening.
