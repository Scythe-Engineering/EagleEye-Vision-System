package frc.robot;

import org.wpilib.math.estimator.DifferentialDrivePoseEstimator;
import org.wpilib.math.geometry.Pose2d;
import org.wpilib.math.geometry.Rotation2d;
import org.wpilib.framework.TimedRobot;
import org.wpilib.system.Timer;
import org.wpilib.smartdashboard.Field2d;
import org.wpilib.telemetry.Telemetry;
import frc.robot.vision.EagleEyeCamera;
import frc.robot.vision.EagleEyeCameraSim;

/** Desktop example: synthetic sensors and vision, with no actuator or deployment code. */
public class Robot extends TimedRobot {
  // Optional enhancement: set true to ingest vision and update odometry every 5 ms.
  private static final boolean LOW_LATENCY_VISION = false;
  private final EagleEyeCamera camera = new EagleEyeCamera(
      "example/front/pose", "example/front/meta");
  private final DifferentialDrivePoseEstimator estimator = new DifferentialDrivePoseEstimator(
      new Rotation2d(), 0, 0,
      new Pose2d(4, 2, new Rotation2d()));
  private final Field2d field = new Field2d();
  private EagleEyeCameraSim simulatedCamera;
  private final double start = Timer.getTimestamp();
  private Pose2d truth = new Pose2d(4, 2, new Rotation2d());
  private int accepted;

  public Robot() {
    // Runs synchronously with robotPeriodic, so the estimator has one thread owner.
    if (LOW_LATENCY_VISION) {
      addPeriodic(this::updateOdometryAndVision, 0.005, 0.002);
    }
  }

  private void updateOdometryAndVision() {
    double elapsed = Timer.getTimestamp() - start;
    double distance = 0.4 * elapsed;
    truth = simulatedTruth(elapsed);
    // On a real drivetrain, use measured gyro and left/right wheel positions here.
    estimator.update(truth.getRotation(), distance, distance);
    for (var observation : camera.poll()) {
      estimator.addVisionMeasurement(observation.pose(), observation.timestampSeconds(),
          EagleEyeCamera.standardDeviations(observation));
      accepted++;
    }
  }

  @Override
  public void robotPeriodic() {
    if (!LOW_LATENCY_VISION) {
      updateOdometryAndVision();
    }
    field.setRobotPose(estimator.getEstimatedPosition());
    field.getObject("GroundTruth").setPose(truth);
    Telemetry.log("EagleEye/Accepted", accepted);
    Telemetry.log("EagleEye", field);
  }

  @Override
  public void simulationInit() {
    EagleEyeCameraSim.translationNoiseBase = 0;
    simulatedCamera = new EagleEyeCameraSim("example/front");
  }

  private static Pose2d simulatedTruth(double elapsed) {
    double heading = 0.2 * elapsed;
    return new Pose2d(4 + 2 * Math.sin(heading), 2 + 2 * (1 - Math.cos(heading)),
        new Rotation2d(heading));
  }

  @Override
  public void simulationPeriodic() {
    simulatedCamera.update(simulatedTruth(Timer.getTimestamp() - start));
  }
}
