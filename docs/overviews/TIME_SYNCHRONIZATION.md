# Capture timestamp contract

EagleEye wraps a captured frame in `TimedValue` with `TimingMetadata`. `capture_monotonic_ns` records the local monotonic capture time. `capture_nt_us` is the same instant converted to the NetworkTables clock in microseconds. Pipeline operations preserve that metadata, and `PublishToNetworktables` passes `capture_nt_us` to the publisher as the value timestamp.

On Linux, V4L2 uses the kernel buffer timestamp when it is monotonic. If the driver does not provide one, camera capture falls back to the monotonic time when the frame is delivered and logs the fallback. OpenCV and video-file inputs also use delivery time. A cached frame keeps its original timestamp and frame sequence; it is not restamped.

## Robot usage

Subscribe once and keep queued samples so value and timestamp remain paired:

```java
private final StructSubscriber<Pose2d> poseSub =
    NetworkTableInstance.getDefault()
        .getTable("EagleEye")
        .getStructTopic("robot_pose", Pose2d.struct)
        .subscribe(Pose2d.kZero, PubSubOption.keepDuplicates(true),
                   PubSubOption.pollStorage(20));
```

Drain `readQueue()` after the normal odometry update. NetworkTables reports the timestamp in the robot's local FPGA microsecond clock.

```java
poseEstimator.update(gyro.getRotation2d(), modulePositions());

for (var sample : poseSub.readQueue()) {
  double captureSeconds = sample.timestamp / 1_000_000.0;
  double ageSeconds = Timer.getFPGATimestamp() - captureSeconds;
  if (sample.timestamp == 0 || ageSeconds < 0 || ageSeconds > 0.5) continue;
  poseEstimator.addVisionMeasurement(sample.value, captureSeconds);
}
```

Choose the stale threshold for the robot loop and estimator history. The example rejects samples older than 0.5 seconds.

Pass `captureSeconds` directly to `addVisionMeasurement`. Do not subtract measured camera, pipeline, or network latency. The published timestamp already identifies capture time, so subtracting latency again moves the measurement too far into the past.
