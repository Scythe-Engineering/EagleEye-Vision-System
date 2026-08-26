# EagleEye robot library

Robot-side code for consuming EagleEye localization on a roboRIO. Copy
`java/frc/robot/vision/EagleEyeCamera.java` into your robot project and adjust the package
declaration if your vision code lives elsewhere. Copy `EagleEyeCameraSim.java` too if you develop
in WPILib simulation.

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

Nothing about those names is baked into the library. Robot code supplies both keys, relative to
the `EagleEye` table, and they must match the `target_key` on the matching
`publish_to_networktables` operation in the WebUI character for character. `EagleEye` itself is
the one fixed part: the coprocessor hands every publish operation that same root table.

```java
// Follows the shipped preset: localization/front/pose and localization/front/meta.
EagleEyeCamera.forSource("localization/front");

// Any other layout: pass both keys.
new EagleEyeCamera("vision/left_cam/robot_pose", "vision/left_cam/quality");
```

A key nothing publishes raises a Driver Station warning naming the key and pointing at the WebUI,
repeated every `warningIntervalSeconds`, rather than failing silently.

## Wiring the coprocessor

Create a pipeline in the WebUI using the bundled "Basic localization" template. Then set
`camera_bus_id` on the three operations that take one and point `apriltag_map_path` at your field
map.

The chain is:

```
device_input → detect_apriltags → minimum_apriltag_count → pnp_camera_localization
                                                              ├─ camera_pose → camera_to_robot_pose → publish "localization/front/pose"
                                                              │                                      └→ robot_pose_output
                                                              └─ pose_meta ─────────────────────────→ publish "localization/front/meta"
```

Operations may sit on either publishing branch, but only single-input ones. A single-input
operation passes its capture timestamp through untouched, which is why the pose branch can run
through `camera_to_robot_pose` and still match the metrics exactly. A multi-input
operation averages the timestamps of everything feeding it, producing a time that no longer
matches the other branch, and the robot silently drops every sample. `pose_fusion` is the one to
watch for here.

For a second camera, duplicate the whole chain with a different `camera_bus_id` and a different
source name. Insert `temporal_acceleration_preprocessor_rust` between `device_input` and
`detect_apriltags` if you want its region-of-interest cropping; it does not affect the contract.

## Robot code

Keys live at the call site, in whichever subsystem owns the pose estimator:

```java
public class Drive extends SubsystemBase {
  private final EagleEyeCamera[] cameras = {
    EagleEyeCamera.forSource("localization/front"),
    EagleEyeCamera.forSource("localization/back"),
  };

  @Override
  public void periodic() {
    poseEstimator.update(gyro.getRotation2d(), modulePositions);
    EagleEyeCamera.update(poseEstimator::addVisionMeasurement, cameras);
  }
}
```

Call it from a subsystem's `periodic()`, not from a NetworkTables listener thread —
`SwerveDrivePoseEstimator` is not thread safe.

Use `camera.poll()` directly if you want the raw observations for logging or for your own
filtering; it returns them in capture order with the quality metrics attached.

## Simulation

`EagleEyeCameraSim` publishes the same two topics with the same shared timestamp from a simulated
robot pose, so `EagleEyeCamera` — and everything downstream of it — runs unchanged in WPILib
simulation with no camera and no coprocessor.

```java
// simulationInit, one per source your real robot has:
EagleEyeCameraSim frontSim =
    new EagleEyeCameraSim(
        "localization/front",
        new Transform3d(),   // camera mounting; only yaw and translation matter
        AprilTagFieldLayout.loadField(AprilTagFields.kDefaultField));

// simulationPeriodic, with ground truth from your drive sim:
frontSim.update(driveSim.getPose());
```

Given a tag layout, a tag counts as visible when it is inside `cameraFovDegrees` and within
`maximumTagRangeMeters`; too few visible tags and nothing publishes, which reproduces vision dead
zones. Noise is Gaussian, scaled by the same distance-squared-over-tag-count model the consumer
uses for standard deviations — set `translationNoiseBase` to zero for deterministic tests. The
single-argument constructor skips the layout and always publishes fixed metrics, for testing the
NetworkTables plumbing alone.

What it does not simulate: pipeline latency (samples are stamped "now" rather than an exposure
instant in the past) and solver failure modes like reprojection outliers. It validates your robot
code's consumption, filtering, and dead-zone handling — not the solver.

## Time synchronization

You do not need to compensate for latency. Do not subtract one, and do not pass
`Timer.getFPGATimestamp()` as the measurement time.

The timestamp on a sample traces back through:

1. V4L2 provides the exposure time when the buffer has both `TIMESTAMP_MONOTONIC` and
   `TSTAMP_SRC_SOE`. Other V4L2 drivers, OpenCV, and video-file sources use delivery time instead.
2. `src/utils/timing.py` converts that timestamp to the NetworkTables clock using an offset cached
   on first use. It takes the median of eleven paired readings; both clocks tick at the same rate,
   so there is nothing to re-measure.
3. `publish_to_networktables` sets the value with that timestamp instead of "now".
4. ntcore on the roboRIO translates the client's timestamp into server time using its negotiated
   clock offset.
5. WPILib uses NetworkTables server time as FPGA time on a roboRIO, the same clock
   `Timer.getFPGATimestamp()` reads.

So `sample.timestamp / 1e6` is already in the domain `addVisionMeasurement` wants. With supported
V4L2 drivers it identifies exposure time; with fallback sources it identifies delivery time.

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
| `warningIntervalSeconds` | `5.0` | How often to repeat the warning about a key nothing publishes |

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

**A Driver Station warning naming one of your keys.** Nothing publishes it. Open OutlineViewer,
find the key under the `EagleEye` table, and compare it to the string you passed to
`EagleEyeCamera` — a leading slash, a stray plural, or a renamed source are the usual causes. If
only the `meta` key is missing, the pipeline is short a publisher on the solver's `pose_meta`
port; every pose is dropped, because a pose without metrics cannot be weighted.

**Measurements arrive but the estimate is jumpy.** Raise `translationStdDevBase`, or tighten
`maximumReprojectionErrorPixels`. Log the raw `poll()` observations and look at the reprojection
error before changing anything else.

**Everything arrives in bursts, and the estimate lags.** Confirm the pipeline is actually keeping
up on the coprocessor. `sendAll` means the robot sees every frame EagleEye produces, so a burst is
a coprocessor-side stall rather than a subscription problem.
