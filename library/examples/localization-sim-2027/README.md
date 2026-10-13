# Localization desktop example

Requires WPILib 2027.0.0-alpha-7, Gradle 9.4.1, and JDK 25.
The build follows the tagged WPILib alpha-7 Java template with desktop support enabled. Open this folder in WPILib VS Code
and run **WPILib: Simulate Robot Code** with **Sim GUI** enabled.

This example is a desktop simulation with synthetic odometry and camera readings and no drivetrain
code, so do not deploy it as a robot program. Its `.wpilib/wpilib_preferences.json` uses team
number `0`, which suits simulation. For a robot, copy the SDK into your own project, which needs
your real team number. Physical controller deployment has not been validated. For controller
setup and the coprocessor address, see [Use EagleEye with Systemcore](https://scythe-engineering.github.io/EagleEye-Docs/docs/user-guide/systemcore).

```sh
./gradlew test build
./gradlew simulateJava
```

Set `JAVA_HOME` to your WPILib JDK if your terminal uses another Java installation.
The project compiles `../../java-2027` directly; the SDK files are not duplicated.

In AdvantageScope select **File → Connect to Simulator → NetworkTables 4**. Add
`Telemetry/EagleEye` as a field, showing the robot estimate and `GroundTruth`.
`Telemetry/EagleEye/Accepted` should increase. Synthetic vision publishes to
`EagleEye/example/front/pose` and `/meta`; keep physical publishers on separate keys.

`Robot.java` demonstrates odometry first, then timestamped vision measurements with
quality-based standard deviations. Its sensor values are synthetic: replace them with
measured gyro/wheel or swerve-module positions when integrating into a real drivetrain.
For the shortest real-robot integration use the [library guide](../../README.md).

The JUnit contract tests exercise NWU coordinates, exact timestamp joins, split arrival,
missing metadata, stale/future samples, quality gates, invalid values, and uncertainty.

Sample timestamps are nanoseconds converted to seconds in the default
`Timer.getTimestamp()` time base. A custom `RobotController.setTimeSource` must preserve
that base or your integration must translate observations before estimator fusion.
The tests also check a 20 ms-old sample, ±600 ms rejection, HAL clock agreement,
Driver Station warning throttling, and an empty planar tag bearing in simulation.

## Optional low-latency path

The example defaults to the simple 20 ms robot loop. Set `LOW_LATENCY_VISION = true`
in `Robot.java` to schedule the same odometry/vision method every 5 ms instead.
The two paths are mutually exclusive; dashboard and synthetic camera publishing remain
on the normal 20 ms loop. This example demonstrates scheduling, not a camera-latency
benchmark. Both modes retain identical timestamp and quality handling.

For the Drive.java and Robot.java edits needed on a real robot, see the
[optional enhancement guide](../../README.md#optional-enhancement-lower-latency-ingestion).
