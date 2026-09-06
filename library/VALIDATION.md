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
