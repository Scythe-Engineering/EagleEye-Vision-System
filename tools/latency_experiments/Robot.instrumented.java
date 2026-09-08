package frc.robot;

import edu.wpi.first.math.VecBuilder;
import edu.wpi.first.math.estimator.DifferentialDrivePoseEstimator;
import edu.wpi.first.math.geometry.*;
import edu.wpi.first.math.kinematics.DifferentialDriveKinematics;
import edu.wpi.first.networktables.*;
import edu.wpi.first.wpilibj.*;
import edu.wpi.first.wpilibj.smartdashboard.*;
import frc.robot.vision.*;

/** Desktop integration bench. No motors, actuators, or robot deployment required. */
public class Robot extends TimedRobot {
  private LatencyExperiment latencyExperiment;
  private HighRateVisionExperiment highRateExperiment;
  private final NetworkTableInstance nt = NetworkTableInstance.getDefault();
  private final EagleEyeCamera live = new EagleEyeCamera(
      "localization/arducam-ov9281-usb-camera/pose", "localization/arducam-ov9281-usb-camera/meta");
  private final EagleEyeCamera diagnostic = new EagleEyeCamera(
      "localization/arducam-ov9281-usb-camera/pose", "localization/arducam-ov9281-usb-camera/meta");
  private final StructPublisher<Pose2d> diagnosticPublisher = nt.getStructTopic("/Audit/LiveDiagnosticOnly", Pose2d.struct).publish();
  private long diagnosticCount;
  private final EagleEyeCamera synthetic = new EagleEyeCamera("audit/local/pose", "audit/local/meta");
  private final EagleEyeCamera remote = new EagleEyeCamera("audit/remote/pose", "audit/remote/meta");
  private final Field2d field = new Field2d();
  private final DifferentialDrivePoseEstimator estimator = new DifferentialDrivePoseEstimator(
      new DifferentialDriveKinematics(0.6), new Rotation2d(), 0, 0, new Pose2d(4, 2, new Rotation2d()),
      VecBuilder.fill(0.1, 0.1, 0.1), VecBuilder.fill(0.1, 0.1, 1e6));
  private final StructPublisher<Pose2d> accepted = nt.getStructTopic("/Audit/LiveAccepted", Pose2d.struct).publish();
  private final StructPublisher<Pose2d> remoteAccepted = nt.getStructTopic("/Audit/RemoteAccepted", Pose2d.struct).publish();
  private final StructPublisher<Pose2d> truthPublisher = nt.getStructTopic("/Audit/GroundTruth", Pose2d.struct).publish();
  private final StructPublisher<Pose2d> estimatePublisher = nt.getStructTopic("/Audit/Estimate", Pose2d.struct).publish();
  private EagleEyeCameraSim simulator;
  private long liveCount, localCount, remoteCount;
  private double lastLive = -1, lastRemote = -1;
  private double start, distance;
  private Pose2d truth = new Pose2d();

  public Robot() {
    DataLogManager.start();
    SmartDashboard.putData("EagleEye field", field);
    SmartDashboard.putString("Contract", "Blue-origin NWU; meters; yaw CCW radians; pose/meta exact capture timestamps");
    start = Timer.getFPGATimestamp();
    latencyExperiment = new LatencyExperiment(this);
  }

  @Override public void robotPeriodic() {
    latencyExperiment.robotPeriodic();
    if (highRateExperiment != null) highRateExperiment.robotPeriodic();
    synchronized (EagleEyeCamera.class) {
    double now = Timer.getFPGATimestamp();
    double t = now - start;
    // A 2 m radius circle: wheel travel and gyro are consistent with the ground truth.
    double angle = 0.2 * t;
    distance = 0.4 * t;
    truth = new Pose2d(4 + 2 * Math.sin(angle), 2 + 2 * (1 - Math.cos(angle)), new Rotation2d(angle));
    estimator.updateWithTime(now, truth.getRotation(), distance, distance);
    for (var obs : synthetic.poll()) {
      localCount++;
      estimator.addVisionMeasurement(obs.pose(), obs.timestampSeconds(), EagleEyeCamera.standardDeviations(obs));
      field.getObject("LocalAccepted").setPose(obs.pose());
    }
    for (var obs : live.poll()) {
      liveCount++;
      lastLive = obs.timestampSeconds();
      accepted.set(obs.pose());
      field.getObject("LiveCamera").setPose(obs.pose());
      SmartDashboard.putNumber("LiveTagCount", obs.tagCount());
      SmartDashboard.putNumber("LiveReprojectionPx", obs.reprojectionErrorPixels());
    }
    // Observe the current camera without treating relaxed quality as trusted vision.
    // Keep the shipped 2 px gate for live and all estimator inputs.
    double strictLimit = EagleEyeCamera.maximumReprojectionErrorPixels;
    try {
      EagleEyeCamera.maximumReprojectionErrorPixels = 4.0;
      for (var obs : diagnostic.poll()) {
        diagnosticCount++;
        diagnosticPublisher.set(obs.pose());
        field.getObject("DiagnosticOnly4px").setPose(obs.pose());
      }
    } finally { EagleEyeCamera.maximumReprojectionErrorPixels = strictLimit; }
    SmartDashboard.putNumber("DiagnosticOnly4pxCount", diagnosticCount);
    SmartDashboard.putNumber("NTNowSeconds", NetworkTablesJNI.now() / 1e6);
    SmartDashboard.putNumber("FPGANowSeconds", now);
    for (var obs : remote.poll()) {
      remoteCount++;
      lastRemote = obs.timestampSeconds();
      remoteAccepted.set(obs.pose());
      field.getObject("RemoteContract").setPose(obs.pose());
    }
    truthPublisher.set(truth);
    estimatePublisher.set(estimator.getEstimatedPosition());
    field.setRobotPose(estimator.getEstimatedPosition());
    field.getObject("GroundTruth").setPose(truth);
    SmartDashboard.putNumber("LocalAcceptedCount", localCount);
    SmartDashboard.putNumber("LiveAcceptedCount", liveCount);
    SmartDashboard.putNumber("RemoteAcceptedCount", remoteCount);
    SmartDashboard.putNumber("LiveAgeSeconds", lastLive < 0 ? -1 : now - lastLive);
    SmartDashboard.putNumber("RemoteAgeSeconds", lastRemote < 0 ? -1 : now - lastRemote);
    SmartDashboard.putBoolean("LiveFresh", lastLive >= 0 && now - lastLive <= EagleEyeCamera.maximumSampleAgeSeconds);
    SmartDashboard.putNumber("EstimatorErrorMeters", estimator.getEstimatedPosition().getTranslation().getDistance(truth.getTranslation()));
  }

    }

  @Override public void simulationInit() {
    EagleEyeCameraSim.translationNoiseBase = 0;
    simulator = new EagleEyeCameraSim("audit/local");
    highRateExperiment = new HighRateVisionExperiment();
  }
  @Override public void simulationPeriodic() { simulator.update(truth); }
}
