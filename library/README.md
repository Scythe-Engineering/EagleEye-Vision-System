# EagleEye robot library

Robot-side code for consuming EagleEye localization on a roboRIO. Copy
`java/frc/robot/vision/EagleEyeCamera.java` into your robot project and adjust the package
declaration if your vision code lives elsewhere.

There is no vendordep. One file with no dependencies beyond WPILib is less to maintain than a
maven repository, and copying it lets a team read and tune the thing that decides where their
robot thinks it is.

## The NetworkTables contract

EagleEye runs as a NetworkTables **client**; the roboRIO is the server. Each localization source
publishes two topics under its own subtable:

| Topic | Type | Contents |
| --- | --- | --- |
| `EagleEye/localization/<source>/pose` | `Pose3d` struct | Robot pose in field coordinates |
| `EagleEye/localization/<source>/meta` | `double[3]` | `[tagCount, meanTagDistanceMeters, reprojectionErrorPixels]` |

Both come from one solver output on one frame, so both carry the identical capture timestamp.
`EagleEyeCamera` joins them on exact timestamp equality rather than searching for a near match.

`<source>` is yours to name — `front`, `back`, `left`. Use one per camera.

## Wiring the coprocessor

`localization_pipeline_preset.json` is a working single-camera pipeline. Paste its contents into
the pipeline JSON editor in the WebUI, or merge the `"localization"` key into
`src/config/pipeline_config.json`, then set `camera_bus_id` on the three operations that take one
and point `apriltag_map_path` at your field map.

The chain is:

```
device_input → detect_apriltags → minimum_apriltag_count → pnp_camera_localization
                                                              ├─ camera_pose → camera_to_robot_pose → robot_pose_output → publish "localization/front/pose"
                                                              └─ pose_meta ─────────────────────────────────────────────→ publish "localization/front/meta"
```

Operations may sit on either branch, but only single-input ones. A single-input operation passes
its capture timestamp through untouched, which is why the pose branch can run through
`camera_to_robot_pose` and `robot_pose_output` and still match the metrics exactly. A multi-input
operation averages the timestamps of everything feeding it, producing a time that no longer
matches the other branch, and the robot silently drops every sample. `pose_fusion` is the one to
watch for here.

For a second camera, duplicate the whole chain with a different `camera_bus_id` and a different
source name. Insert `temporal_acceleration_preprocessor_rust` between `device_input` and
`detect_apriltags` if you want its region-of-interest cropping; it does not affect the contract.

## Robot code

```java
private final EagleEyeCamera[] cameras = {
  new EagleEyeCamera("front"), new EagleEyeCamera("back"),
};

@Override
public void periodic() {
  poseEstimator.update(gyro.getRotation2d(), modulePositions);
  EagleEyeCamera.update(poseEstimator::addVisionMeasurement, cameras);
}
```

Call it from a subsystem's `periodic()`, not from a NetworkTables listener thread —
`SwerveDrivePoseEstimator` is not thread safe.

Use `camera.poll()` directly if you want the raw observations for logging or for your own
filtering; it returns them in capture order with the quality metrics attached.

## Time synchronization

You do not need to compensate for latency. Do not subtract one, and do not pass
`Timer.getFPGATimestamp()` as the measurement time.

The timestamp on a sample traces back through:

1. The V4L2 driver's `CLOCK_MONOTONIC` buffer timestamp — the moment the frame was *exposed*, not
   the moment EagleEye finished with it.
2. `src/utils/timing.py`, which converts that to the NetworkTables clock using an offset measured
   once at startup (median of eleven paired readings; both clocks tick at the same rate, so there
   is nothing to re-measure).
3. `publish_to_networktables`, which sets the value with that timestamp instead of "now".
4. ntcore on the roboRIO, which translates the client's timestamp into server time using its
   negotiated clock offset.
5. WPILib, where NetworkTables server time on a roboRIO *is* FPGA time — the same clock
   `Timer.getFPGATimestamp()` reads.

So `sample.timestamp / 1e6` is already in the domain `addVisionMeasurement` wants, and it refers
to the exposure instant rather than to any point in the processing or transport path.

Two consequences worth knowing:

- For the first second or so after an EagleEye connects, ntcore has not converged on the clock
  offset and timestamps can be nonsense, including timestamps in the future. `poll()` drops
  anything that is negative-age or older than `maximumSampleAgeSeconds`.
- If you ever reconfigure EagleEye to be the NetworkTables *server*, this breaks. Keep the
  roboRIO as the server.

If you measure the offset jitter and find it unacceptable, the upgrade path is a dedicated UDP
time-sync channel of the kind PhotonVision uses. On a wired connection it is very unlikely to be
worth it.

## Tuning

Every knob is a public static field on `EagleEyeCamera`, set once in `RobotContainer`:

| Field | Default | Meaning |
| --- | --- | --- |
| `translationStdDevBase` | `0.02` | Metres of trust at one metre with one tag; scaled by distance squared over tag count |
| `rotationStdDev` | `Double.MAX_VALUE` | Leave enormous unless vision should move your heading |
| `minimumTagCount` | `2` | One tag is ambiguous enough to be worth discarding |
| `maximumReprojectionErrorPixels` | `2.0` | Rejects solutions that do not explain the corners they were solved from |
| `maximumTagDistanceMeters` | `6.0` | Rejects poses that are mostly noise |
| `maximumSampleAgeSeconds` | `0.5` | Staleness and unconverged-clock guard |

The distance-squared-over-count standard deviation model is a starting point, not a fitted one.
If the estimator misweights close or distant tags, fit a curve against measured error and replace
`standardDeviations`.

## Why per-camera rather than fused

Publish each camera as its own source and let the pose estimator combine them. `pose_fusion` is
still useful for the WebUI's single consensus pose, but for localization it averages poses
captured at different instants and stamps the result with the mean of their capture times — a
pose that was never true at any moment. `addVisionMeasurement` instead replays odometry to each
measurement's own timestamp, which is strictly better information.

## Troubleshooting

**Poses visible in AdvantageScope but `poll()` returns nothing.** Almost always a missing `meta`
topic. Check that `EagleEye/localization/<source>/meta` exists in OutlineViewer and that the
source name in your Java matches the `target_key` in the pipeline. A pose without metrics cannot
be weighted, so it is dropped rather than injected at a guessed confidence.

**Measurements arrive but the estimate is jumpy.** Raise `translationStdDevBase`, or tighten
`maximumReprojectionErrorPixels`. Log the raw `poll()` observations and look at the reprojection
error before changing anything else.

**Everything arrives in bursts, and the estimate lags.** Confirm the pipeline is actually keeping
up on the coprocessor. `sendAll` means the robot sees every frame EagleEye produces, so a burst is
a coprocessor-side stall rather than a subscription problem.
