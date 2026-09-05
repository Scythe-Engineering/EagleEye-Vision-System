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
  }

  @Override
  public void robotPeriodic() {
    double elapsed = Timer.getFPGATimestamp() - start;
    double heading = 0.2 * elapsed;
    double distance = 0.4 * elapsed;
    truth = new Pose2d(4 + 2 * Math.sin(heading), 2 + 2 * (1 - Math.cos(heading)),
        new Rotation2d(heading));
    // On a real drivetrain, use measured gyro and left/right wheel positions here.
    estimator.update(truth.getRotation(), distance, distance);
    for (var observation : camera.poll()) {
      estimator.addVisionMeasurement(observation.pose(), observation.timestampSeconds(),
          EagleEyeCamera.standardDeviations(observation));
      accepted++;
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

  @Override
  public void simulationPeriodic() {
    simulatedCamera.update(truth);
  }
}
