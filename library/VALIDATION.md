# Localization contract validation

The September 2026 desktop integration audit exercised the actual Python publisher on a
remote coprocessor, NetworkTables 4, the Java SDK in WPILib simulation, and AdvantageScope.
The coprocessor remained the client and the desktop simulator was the server.

- 640 live pose samples joined all 640 metadata samples by exact capture timestamp.
- 762 live Java diagnostic Pose2d echoes preserved the incoming Pose3d X/Y/yaw, with
  maximum component error 4.44e-16 (floating-point rounding).
- Five remote fixtures (yaw 0°, +90°, −90°, 180°, 37°) passed through the real mounting
  transform, publisher, network connection, and Java consumer. A nonzero compound mount
  included translation and all three Euler angles. Invalid negative metadata was rejected.
- Live reprojection error of 2.685–3.017 px was correctly rejected by the default 2 px gate.
  A separate 4 px diagnostic consumer was used only to inspect coordinates. It did not
  feed the unrelated simulated drivetrain and is not a recommended production threshold.
- The frontend reported both live topics present; AdvantageScope displayed the Java output.

## Reproduce the software checks

From the repository root, after installing the project's dependencies:

```sh
uv run pytest tests/test_position_contract.py tests/test_camera_to_robot_pose.py tests/test_publish_to_networktables.py tests/test_v4l2_capture.py tests/test_timing.py
node --test tests/js/position_contract.test.mjs
npm run build
cd library/examples/localization-sim
./gradlew test build
```

The focused Python run passed 44 tests (one Linux ABI test skipped on macOS); the
frontend suite passed 15 and Java passed 10. CI runs the frontend and Java contract suites.
The Java example compiles the SDK directly so tests track the shipped source.

## Scope

These checks establish transport, coordinate conversion, joining, and filtering behavior.
They do not establish physical absolute localization accuracy on a measured robot or a
roboRIO deployment. The room test used a custom two-tag map, not an official competition
layout. The WebUI's field-center constants also need checking for other field dimensions.

Camera capture showed intermittent restart failures outside EagleEye as well as in it.
That investigation is separate from these passing current-stream tests; no camera-reliability
fix is claimed here. Capability discovery now precedes opening the capture handle, avoiding
a second open during configuration. Production quality thresholds were not relaxed.

## Follow-up: model axes and physical room geometry

The follow-up corrected the robot asset transform: the bundled glTF model is Y-up and
+Z-forward, while the converted view pose is Y-up and +X-forward. Apply the asset's
Y-axis alignment once. The prior X-axis quarter-turn incorrectly tipped the model.
Three additional frontend tests exercise the final model transform, including up/forward
vectors at zero, 90° yaw, and compound pitch/roll (18 frontend tests total).

AdvantageScope's 3D view displayed the live Pose3d beside the Java diagnostic Pose2d.
X/Y/yaw agreed; flattening Z/roll/pitch is intentional for the Java 2D estimator and must
not be mistaken for full 3D equality. The running backend was not restarted.

A read-only, 25-frame diagnostic used the existing downscaled MJPEG feed, resized to the
calibration resolution. These are lossy-image diagnostics, not raw-camera accuracy tests:

| Solve | Tag 0 range | Camera upward tilt | Reprojection error |
|---|---|---|---|
| Tag 0 alone | 2.790 m | 17.8° mean | 0.076 px mean |
| Saved two-tag map | about 2.64 m | about 1–2° | about 2.7 px |
| Hypothetical adjusted map, fitted to these images | 2.793 m | 17.8° | 0.097 px mean |

The user estimated 9 ft 3 in (2.819 m) and 20–30° upward. Independent single-tag solves
implied tag 1 was about 35.8 cm sideways and 46.1 cm higher than tag 0; the saved map says
35 cm and 40 cm. The fitted-map result is a sensitivity check on the same images, not
independent validation. Physical tag size, center spacing, camera mount and distance
still need measurement. No guessed map or mount values were saved.

The actual temporal preprocessor produced crops enclosing both tags. Comparing full-frame
and cropped detections on the same 25 images gave a maximum corner difference of 0.334 px.
Position standard deviations were about [2.1, 4.9, 3.5] cm for full-frame detection and
[2.5, 5.8, 4.0] cm for cropped detection. This does not establish performance during fast
motion or rule out every temporal issue. The feedback edge consumes the original PnP camera
pose, before mounting compensation. With the saved zero-angle mount and 0.25 m Z offset,
the previous and corrected backend mounting matrices are exactly equal.

## WPILib 2027 alpha-7 migration (October 13, 2026)

The version-specific Java SDK and examples were built using real GradleRIO dependencies
and desktop HAL on macOS. WPILib 2026 with JDK 17 passed 13 contract tests;
WPILib 2027.0.0-alpha-7 with Gradle 9.4.1 and JDK 25 passed 15 tests.
Both include the original ten contract cases, 20 ms acceptance and ±600 ms rejection,
default robot-clock agreement within 1 ms, and five-second warning throttling through
actual Driver Station reporting. The alpha suite also checks detection freshness and
expiration of poses awaiting metadata; its simulator additionally skips a tag with no
planar bearing. The SDK parity check allows only explicit WPILib API and time-unit changes.

From the repository root:

```sh
python3 library/tools/check_java_sdk_parity.py
# Set JAVA_HOME to JDK 17 for 2026, then:
(cd library/examples/localization-sim && ./gradlew test build)
# Set JAVA_HOME to JDK 25 for alpha-7, then:
(cd library/examples/localization-sim-2027 && ./gradlew test build)
```

The Java CI matrix runs both projects on Ubuntu 24.04 with their matching JDKs.
These are desktop contract and build checks. Physical Systemcore/Driver Station alpha-7
deployment, camera exposure accuracy, and measured robot localization remain unverified.
The alpha clock check assumes the default `RobotController` time source; custom
`setTimeSource` integrations must translate timestamps or preserve that time base.

An additional 2026.2.1 detection-queue probe on macOS hit a native ntcore JNI crash in
`readQueueStringArray`. The upstream 2026.2.1 native constructor signature uses
`Object[]` where `TimestampedStringArray` expects `String[]`. This pre-existing WPILib
issue is outside the passing 2026 pose/meta contract suite; alpha-7's real detection-queue
regression test passes. No 2026 detection-queue or hardware validation is claimed here.

### Fresh-install check (October 14, 2026)

The October 13 results above come from the original implementation and covered software
checks only (unit tests, contract tests, and probes against the alpha-7 jars), with no GUI,
reconnection, or load testing. A second pass on October 14 added a fresh official install, a
native editor and simulator run, a web-interface check, and loopback runtime probes. It
installed the official WPILib alpha-7 macOS (arm64) package on macOS 26.6.2, with its
JDK 25 and a new Gradle home. It found that the example's `settings.gradle` still looked for
the plugin under `~/wpilib/2027/maven`, which the alpha-7 installer does not create. The
original project failed an offline build, unable to resolve `org.wpilib.GradleRIO:2027.0.0-alpha-7`,
though it built when online. The example now uses the tagged alpha-7 settings template, which searches the
installer's own folder first, and `projectYear` in `.wpilib/wpilib_preferences.json` is
`2027_alpha7`. With those changes, a clean offline build with a new Gradle home passed all 15
tests. The Gradle wrapper had been downloaded online first, so this does not show a build on a
blank machine with no internet.

In the native WPILib VS Code window, the original `projectYear` of `2027` made the extension offer
to import the example; after the change and a window reload, it did not, and Java support
loaded. **WPILib: Simulate Robot Code** ran with Sim GUI selected and started a JDK 25
`frc.robot.Main` process serving NetworkTables on port 5810. The process loaded the Sim GUI
extension and logged "Simulator GUI Initialized!", but we did not inspect the Java simulator
window itself (the native window inventory did not list it), so this shows the GUI extension
initialized, not what it displayed. A Glass window connected to the simulator and showed
`Telemetry/EagleEye/Accepted` rising (3,667 to 4,301) and `GroundTruth` and `Robot` as
`Pose2d[]` structs. We did not check whether Glass drew the field widget.

The EagleEye web interface showed Settings as Disconnected before the simulator started and
Connected after, first on the default 2026 backend and then on a backend using the alpha-7
Python packages. On the alpha-7 backend, Settings then showed Disconnected after the simulator
process was terminated and Connected again after `./gradlew run` restarted the simulator; the
simulator log recorded Glass and EagleEye connecting. The terminal-side restart used the
command line, not the editor. No pipelines or cameras were configured, so no camera ran through
a pipeline.

While trying to stop the simulator from the editor, a WPILib deploy for team 0 started by
mistake. No robot was connected, discovery timed out, and nothing was deployed.

Targeted Python tests passed (41) and the frontend production build passed. Real Python
publishers sent to the alpha-7 Java SDK over NetworkTables 4 on one computer, using both the
2026 and alpha-7 Python stacks. Each run sent three blocks of 180 frames and counted accepted
frames cumulatively: 180 before the pipeline was disabled, still 180 during the disable, 360
after re-enable, and 540 after a server restart. Polling every 20 ms against a 90 Hz publisher
also accepted 540 of 540. A 60-second run with no catch-up bursts accepted 4,420 of 4,420 frames, but the
publisher averaged 73.67 frames per second rather than the 90 requested, so it does not show
sustained 90 Hz delivery.

A five-minute (311.8 s) run at a 90 Hz target, with a publisher that caught up after delays and
an SDK polled every 3 ms, **failed** its exact-count check: 26,974 of 27,000 frames were
accepted, all 26 misses in the first third of the run. Every frame reached separate test
subscribers, and they logged bursts of 22 and 26 frames. The SDK keeps 20 samples per topic
between polls, so queue overflow is consistent with the misses, but we did not trace each missing
frame to it. The same limit exists in the 2026 SDK, so this is not a regression from the
migration, and no queue change was made. Robot code should poll every loop; see
[Poll every robot loop](README.md#poll-every-robot-loop). The
[alpha-7 report](../reports/wpilib-2027-alpha7/README.md#fresh-install-and-live-runtime-validation-october-14-2026)
has the full results.

The probes ran the Python publisher, the Java SDK, and NetworkTables on one computer, using a
50 ms synthetic capture offset. They do not measure camera exposure delay. No camera was
authorized on macOS, and no physical Systemcore, roboRIO, Driver Station, Raspberry Pi image, or
field network was used. Creating a new project from the VS Code extension was tried but not
verified, and AdvantageScope and Elastic were not tested.
