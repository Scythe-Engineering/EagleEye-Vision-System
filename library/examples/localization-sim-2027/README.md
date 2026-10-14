# Localization desktop example

Requires WPILib 2027.0.0-alpha-7, Gradle 9.4.1, and JDK 25.
The build follows the tagged WPILib alpha-7 Java template with desktop support enabled. Open this folder in WPILib VS Code
and run **WPILib: Simulate Robot Code** with **Sim GUI** enabled.

Install WPILib alpha-7 with the official installer first. The build looks for the GradleRIO
plugin in the installer's local Maven folder before it tries the internet, so a machine
without that install cannot find the plugin offline. The installer puts everything in a
`2027_alpha7` folder:

| System | Install folder |
| --- | --- |
| macOS | `~/.wpilib/2027_alpha7` (`~/wpilib/2027_alpha7` is kept as a link to it) |
| Linux | `$XDG_DATA_HOME/wpilib/2027_alpha7` when `XDG_DATA_HOME` is an absolute path, otherwise `~/.local/share/wpilib/2027_alpha7` |
| Windows | `C:\Users\Public\wpilib\2027_alpha7` (under the `PUBLIC` folder) |

Only the macOS install was run. The Linux and Windows folders come from the same WPILib template
that the example's `settings.gradle` copies.

This example is a desktop simulation with synthetic odometry and camera readings and no drivetrain
code, so do not deploy it as a robot program. Its `.wpilib/wpilib_preferences.json` uses team
number `0`, which suits simulation. For a robot, copy the SDK into your own project, which needs
your real team number. Physical controller deployment has not been validated. For controller
setup and the coprocessor address, see [Use EagleEye with Systemcore](https://scythe-engineering.github.io/EagleEye-Docs/docs/user-guide/systemcore).

```sh
./gradlew test build
```

Start the simulation from VS Code with **WPILib: Simulate Robot Code**. In alpha-7 the
simulation task is `simulateExternalJava`; Gradle also accepts `./gradlew simulateJava` as an
abbreviation of it. That task only builds the project and writes `build/sim/java.json`, the
launch description the extension reads; it does not start the robot program. From a terminal,
`./gradlew run` starts the robot program. The command-line validation set `HALSIM_EXTENSIONS`
to `build/jni/release/libhalsim_gui.dylib` to load Sim GUI on macOS; the editor sets this for you.

Set `JAVA_HOME` to your WPILib JDK if your terminal uses another Java installation.
On macOS:

```sh
export JAVA_HOME="$HOME/.wpilib/2027_alpha7/jdk"
```

The first `./gradlew` run downloads Gradle 9.4.1 unless it is already cached, so it needs an
internet connection once. After that, with the WPILib install in place, `./gradlew --offline test build`
does not need the network.

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
