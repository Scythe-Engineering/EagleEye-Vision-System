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
