package frc.robot.vision;

import edu.wpi.first.apriltag.AprilTagFieldLayout;
import edu.wpi.first.math.geometry.Pose2d;
import edu.wpi.first.math.geometry.Pose3d;
import edu.wpi.first.math.geometry.Rotation2d;
import edu.wpi.first.math.geometry.Transform3d;
import edu.wpi.first.math.geometry.Translation2d;
import edu.wpi.first.networktables.DoubleArrayPublisher;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.networktables.NetworkTablesJNI;
import edu.wpi.first.networktables.PubSubOption;
import edu.wpi.first.networktables.StructPublisher;
import java.util.Random;

/**
 * Publishes what a real EagleEye coprocessor would, from a simulated robot pose, so robot code can
 * be developed and tested with no camera and no coprocessor.
 *
 * <p>Because it publishes the same two topics with the same shared timestamp, the consuming {@link
 * EagleEyeCamera} is byte-for-byte identical between simulation and a real field: nothing in robot
 * code changes between the two.
 *
 * <p>Typical use, alongside the real subscriber in the drive subsystem:
 *
 * <pre>
 * // In simulationInit / constructor:
 * EagleEyeCameraSim frontSim =
 *     new EagleEyeCameraSim(
 *         "localization/front",
 *         new Transform3d(),                       // camera at robot center, facing forward
 *         AprilTagFieldLayout.loadField(AprilTagFields.kDefaultField));
 *
 * // In simulationPeriodic, with the ground-truth pose from your drive sim:
 * frontSim.update(driveSim.getPose());
 * </pre>
 *
 * <p>Visibility is a range-and-field-of-view check against the tag layout, not a full render: a
 * tag counts as seen when it is within {@link #maximumTagRangeMeters} and inside the camera's
 * horizontal field of view. When fewer than {@link EagleEyeCamera#minimumTagCount} tags are
 * visible nothing is published, which reproduces driving through vision dead zones. Constructed
 * without a layout, the source is always visible with fixed metrics — useful when the test is the
 * NetworkTables plumbing rather than coverage.
 *
 * <p>Noise is Gaussian on x and y, scaled by the same distance-squared-over-tag-count model {@link
 * EagleEyeCamera#standardDeviations} uses to weight real measurements.
 */
public class EagleEyeCameraSim {
  /** Translational noise at one meter with one tag, in meters. Zero for deterministic tests. */
  public static double translationNoiseBase = 0.02;

  /** Tags farther than this are not seen, matching the consumer's distance gate. */
  public static double maximumTagRangeMeters = 6.0;

  /** Horizontal field of view used for the visibility check, in degrees. */
  public static double cameraFovDegrees = 70.0;

  /** Reprojection error reported in the metrics, in pixels. */
  public static double reportedReprojectionErrorPixels = 0.5;

  /** Metrics reported when no field layout was supplied: [tagCount, meanDistance]. */
  public static int fixedTagCount = 2;

  /** Mean tag distance reported when no field layout was supplied, in meters. */
  public static double fixedMeanTagDistanceMeters = 3.0;

  private final StructPublisher<Pose3d> posePublisher;
  private final DoubleArrayPublisher metaPublisher;
  private final Transform3d robotToCamera;
  private final AprilTagFieldLayout layout;
  private final Random random = new Random();

  /**
   * A source that is always visible, with fixed metrics.
   *
   * @param source key prefix shared by both topics, matching {@link EagleEyeCamera#forSource}.
   */
  public EagleEyeCameraSim(String source) {
    this(source, new Transform3d(), null);
  }

  /**
   * A source whose visibility and metrics come from a camera mounting and a tag layout.
   *
   * @param source key prefix shared by both topics, matching {@link EagleEyeCamera#forSource}.
   * @param robotToCamera where the camera sits on the robot; only yaw and translation affect the
   *     visibility check.
   * @param layout the season's tag layout, or null to skip the visibility check.
   */
  public EagleEyeCameraSim(String source, Transform3d robotToCamera, AprilTagFieldLayout layout) {
    this.robotToCamera = robotToCamera;
    this.layout = layout;
    var table = NetworkTableInstance.getDefault().getTable("EagleEye");
    var options =
        new PubSubOption[] {PubSubOption.sendAll(true), PubSubOption.keepDuplicates(true)};
    posePublisher =
        table.getStructTopic(source + "/pose", Pose3d.struct).publish(options);
    metaPublisher = table.getDoubleArrayTopic(source + "/meta").publish(options);
  }

  /**
   * Publish one measurement of the supplied ground-truth pose, if any tags are visible from it.
   *
   * <p>Call from {@code simulationPeriodic()} with the pose from the drive simulation. Both topics
   * are stamped with one shared timestamp, so {@link EagleEyeCamera}'s exact-timestamp join works
   * the same way it does against a real coprocessor.
   *
   * @param groundTruth the robot's true field pose in the simulation.
   */
  public void update(Pose2d groundTruth) {
    double[] meta = visibleTagMetrics(groundTruth);
    if (meta == null) {
      return;
    }
    double noise =
        translationNoiseBase * meta[1] * meta[1] / Math.max(1, (int) meta[0]);
    var noisy =
        new Pose2d(
            groundTruth.getX() + random.nextGaussian() * noise,
            groundTruth.getY() + random.nextGaussian() * noise,
            groundTruth.getRotation());
    long now = NetworkTablesJNI.now();
    posePublisher.set(new Pose3d(noisy), now);
    metaPublisher.set(
        new double[] {meta[0], meta[1], reportedReprojectionErrorPixels}, now);
  }

  /**
   * Count the tags this camera would see and their mean distance.
   *
   * @return {@code [tagCount, meanDistanceMeters]}, or null when too few tags are visible.
   */
  private double[] visibleTagMetrics(Pose2d groundTruth) {
    if (layout == null) {
      return new double[] {fixedTagCount, fixedMeanTagDistanceMeters};
    }
    var cameraPose =
        new Pose3d(groundTruth).transformBy(robotToCamera);
    var cameraPosition = new Translation2d(cameraPose.getX(), cameraPose.getY());
    var cameraHeading = cameraPose.getRotation().toRotation2d();
    double halfFov = Math.toRadians(cameraFovDegrees / 2.0);

    int count = 0;
    double distanceSum = 0.0;
    for (var tag : layout.getTags()) {
      var offset = new Translation2d(tag.pose.getX(), tag.pose.getY()).minus(cameraPosition);
      double distance = cameraPose.getTranslation().getDistance(tag.pose.getTranslation());
      if (distance > maximumTagRangeMeters || distance < 1e-6) {
        continue;
      }
      Rotation2d bearing = offset.getAngle().minus(cameraHeading);
      if (Math.abs(bearing.getRadians()) > halfFov) {
        continue;
      }
      count++;
      distanceSum += distance;
    }
    if (count < EagleEyeCamera.minimumTagCount) {
      return null;
    }
    return new double[] {count, distanceSum / count};
  }
}
