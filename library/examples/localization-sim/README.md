# Localization desktop example

Requires WPILib 2026 (GradleRIO 2026.2.1) and its JDK. Open this folder in WPILib VS Code
and run **WPILib: Simulate Robot Code** with **Sim GUI** enabled. There is no robot deployment target.

```sh
./gradlew test build
./gradlew simulateJava
```

Set `JAVA_HOME` to your WPILib JDK if your terminal uses another Java installation.
The project compiles `../../java` directly; the SDK files are not duplicated.

In AdvantageScope select **File → Connect to Simulator → NetworkTables 4**. Add
`SmartDashboard/EagleEye` as a field, showing the robot estimate and `GroundTruth`.
`SmartDashboard/EagleEye/Accepted` should increase. Synthetic vision publishes to
`EagleEye/example/front/pose` and `/meta`; keep physical publishers on separate keys.

`Robot.java` demonstrates odometry first, then timestamped vision measurements with
quality-based standard deviations. Its sensor values are synthetic: replace them with
measured gyro/wheel or swerve-module positions when integrating into a real drivetrain.
For the shortest real-robot integration use the [library guide](../../README.md).

The JUnit contract tests exercise NWU coordinates, exact timestamp joins, split arrival,
missing metadata, stale/future samples, quality gates, invalid values, and uncertainty.
