package frc.robot;

import edu.wpi.first.math.estimator.DifferentialDrivePoseEstimator;
import edu.wpi.first.math.geometry.Pose2d;
import edu.wpi.first.math.geometry.Rotation2d;
import edu.wpi.first.math.kinematics.DifferentialDriveKinematics;
import edu.wpi.first.wpilibj.TimedRobot;
import edu.wpi.first.wpilibj.Timer;
import edu.wpi.first.wpilibj.smartdashboard.Field2d;
import edu.wpi.first.wpilibj.smartdashboard.SmartDashboard;
import frc.robot.vision.EagleEyeCamera;
import frc.robot.vision.EagleEyeCameraSim;

/** Desktop example: synthetic sensors and vision, with no actuator or deployment code. */
public class Robot extends TimedRobot {
  // Optional enhancement: set true to ingest vision and update odometry every 5 ms.
  private static final boolean LOW_LATENCY_VISION = false;
  private final EagleEyeCamera camera = new EagleEyeCamera(
      "example/front/pose", "example/front/meta");
  private final DifferentialDrivePoseEstimator estimator = new DifferentialDrivePoseEstimator(
      new DifferentialDriveKinematics(0.6), new Rotation2d(), 0, 0,
      new Pose2d(4, 2, new Rotation2d()));
  private final Field2d field = new Field2d();
  private EagleEyeCameraSim simulatedCamera;
  private final double start = Timer.getFPGATimestamp();
  private Pose2d truth = new Pose2d(4, 2, new Rotation2d());
  private int accepted;

  public Robot() {
    SmartDashboard.putData("EagleEye", field);
    // Runs synchronously with robotPeriodic, so the estimator has one thread owner.
    if (LOW_LATENCY_VISION) {
      addPeriodic(this::updateOdometryAndVision, 0.005, 0.002);
    }
  }

  private void updateOdometryAndVision() {
    double elapsed = Timer.getFPGATimestamp() - start;
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
    SmartDashboard.putNumber("EagleEye/Accepted", accepted);
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
    simulatedCamera.update(simulatedTruth(Timer.getFPGATimestamp() - start));
  }
}
