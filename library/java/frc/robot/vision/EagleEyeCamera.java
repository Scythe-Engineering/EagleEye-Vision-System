package frc.robot.vision;

import edu.wpi.first.math.Matrix;
import edu.wpi.first.math.VecBuilder;
import edu.wpi.first.math.geometry.Pose2d;
import edu.wpi.first.math.geometry.Pose3d;
import edu.wpi.first.math.numbers.N1;
import edu.wpi.first.math.numbers.N3;
import edu.wpi.first.networktables.DoubleArraySubscriber;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.networktables.NetworkTablesJNI;
import edu.wpi.first.networktables.PubSubOption;
import edu.wpi.first.networktables.StructSubscriber;
import edu.wpi.first.networktables.TimestampedDoubleArray;
import edu.wpi.first.networktables.TimestampedObject;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

/**
 * One EagleEye localization source, read from its own {@code EagleEye/localization/} subtable.
 *
 * <p>EagleEye stamps every published pose with the kernel's V4L2 exposure timestamp, and ntcore
 * translates that into roboRIO FPGA time on arrival. The timestamp on a received sample is
 * therefore already in the domain {@code addVisionMeasurement} expects: do not subtract a latency,
 * and do not use {@code Timer.getFPGATimestamp()} as the measurement time.
 *
 * <p>Typical use, once per robot loop:
 *
 * <pre>
 * private final EagleEyeCamera[] cameras = {new EagleEyeCamera("front"), new EagleEyeCamera("back")};
 *
 * public void periodic() {
 *   EagleEyeCamera.update(poseEstimator::addVisionMeasurement, cameras);
 * }
 * </pre>
 */
public class EagleEyeCamera {
  /** A single field-pose measurement and everything needed to weight it. */
  public record Observation(
      Pose2d pose,
      double timestampSeconds,
      int tagCount,
      double meanTagDistanceMeters,
      double reprojectionErrorPixels) {}

  /** Receives a weighted measurement; pass {@code poseEstimator::addVisionMeasurement}. */
  @FunctionalInterface
  public interface VisionConsumer {
    void accept(Pose2d pose, double timestampSeconds, Matrix<N3, N1> standardDeviations);
  }

  /**
   * Base translational standard deviation in meters, at one meter with one tag.
   *
   * <p>Scaled by distance squared and divided by tag count. Tune against measured error: raise it
   * if the estimator chases vision, lower it if vision barely moves the pose.
   */
  public static double translationStdDevBase = 0.02;

  /** Heading standard deviation. Keep this huge on a robot with a trustworthy gyro. */
  public static double rotationStdDev = Double.MAX_VALUE;

  /** Minimum contributing tags. A single tag is ambiguous enough to be worth discarding. */
  public static int minimumTagCount = 2;

  /** Reject solutions whose corners reprojected this far from where they were seen. */
  public static double maximumReprojectionErrorPixels = 2.0;

  /** Reject tags far enough away that the pose is mostly noise. */
  public static double maximumTagDistanceMeters = 6.0;

  /**
   * Discard samples older than this, in seconds.
   *
   * <p>This also covers the first second or so after an EagleEye connects, while ntcore is still
   * converging on the client-to-server clock offset and timestamps are not yet meaningful.
   */
  public static double maximumSampleAgeSeconds = 0.5;

  private static final int POLL_STORAGE = 20;

  private final StructSubscriber<Pose3d> poseSubscriber;
  private final DoubleArraySubscriber metaSubscriber;

  // A pose and its metrics come from two separate NetworkTables publishers, so a network flush can
  // land between them and split one frame across two poll cycles. Carrying the leftovers for a
  // cycle removes that race; anything genuinely orphaned ages out below.
  private final List<TimestampedObject<Pose3d>> carriedPoses = new ArrayList<>();
  private final List<TimestampedDoubleArray> carriedMetas = new ArrayList<>();

  /**
   * Subscribe to one EagleEye localization source.
   *
   * @param name source name, matching the {@code target_key} prefix set in the EagleEye pipeline.
   */
  public EagleEyeCamera(String name) {
    var table = NetworkTableInstance.getDefault().getTable("EagleEye/localization/" + name);
    // sendAll and keepDuplicates stop the server from coalescing frames the estimator wants: at
    // 90 fps against a 50 Hz loop, a plain subscription throws away most of the measurements.
    var options =
        new PubSubOption[] {
          PubSubOption.sendAll(true),
          PubSubOption.keepDuplicates(true),
          PubSubOption.pollStorage(POLL_STORAGE),
        };
    poseSubscriber = table.getStructTopic("pose", Pose3d.struct).subscribe(new Pose3d(), options);
    metaSubscriber = table.getDoubleArrayTopic("meta").subscribe(new double[0], options);
  }

  /**
   * Drain every measurement received since the last call, oldest first.
   *
   * <p>Samples that are stale, unconverged, or fail the quality gates are dropped. An empty result
   * while poses are visible in AdvantageScope usually means {@code meta} is not being published;
   * see the troubleshooting notes in {@code library/README.md}.
   *
   * @return accepted observations in capture order.
   */
  public List<Observation> poll() {
    carriedPoses.addAll(Arrays.asList(poseSubscriber.readQueue()));
    carriedMetas.addAll(Arrays.asList(metaSubscriber.readQueue()));

    long nowMicros = NetworkTablesJNI.now();
    var observations = new ArrayList<Observation>(carriedPoses.size());
    var unmatched = new ArrayList<TimestampedObject<Pose3d>>();

    for (var sample : carriedPoses) {
      double ageSeconds = (nowMicros - sample.timestamp) / 1e6;
      if (ageSeconds < 0.0 || ageSeconds > maximumSampleAgeSeconds) {
        continue;
      }
      double[] meta = takeMeta(sample.timestamp);
      if (meta == null) {
        unmatched.add(sample);
        continue;
      }
      var observation =
          new Observation(
              sample.value.toPose2d(), sample.timestamp / 1e6, (int) meta[0], meta[1], meta[2]);
      if (isTrustworthy(observation)) {
        observations.add(observation);
      }
    }

    carriedPoses.clear();
    carriedPoses.addAll(unmatched);
    carriedMetas.removeIf(meta -> (nowMicros - meta.timestamp) / 1e6 > maximumSampleAgeSeconds);
    return observations;
  }

  /**
   * Feed every camera's measurements to a pose estimator in capture order.
   *
   * <p>The sort is not cosmetic. {@code addVisionMeasurement} replays the odometry buffer forward
   * from each sample's timestamp, so an out-of-order sample costs a second replay.
   *
   * @param consumer usually {@code poseEstimator::addVisionMeasurement}.
   * @param cameras every localization source on the robot.
   */
  public static void update(VisionConsumer consumer, EagleEyeCamera... cameras) {
    var all = new ArrayList<Observation>();
    for (var camera : cameras) {
      all.addAll(camera.poll());
    }
    all.sort(Comparator.comparingDouble(Observation::timestampSeconds));
    for (var observation : all) {
      consumer.accept(
          observation.pose(), observation.timestampSeconds(), standardDeviations(observation));
    }
  }

  /**
   * Weight a measurement by how far away its tags were and how many there were.
   *
   * <p>Distance squared over tag count is the standard starting point, not a fitted model. Replace
   * it with a curve fitted to measured error if the estimator misweights close or distant tags.
   *
   * @param observation measurement to weight.
   * @return x, y and heading standard deviations.
   */
  public static Matrix<N3, N1> standardDeviations(Observation observation) {
    double distance = observation.meanTagDistanceMeters();
    double translation =
        translationStdDevBase * distance * distance / Math.max(1, observation.tagCount());
    return VecBuilder.fill(translation, translation, rotationStdDev);
  }

  private static boolean isTrustworthy(Observation observation) {
    return observation.tagCount() >= minimumTagCount
        && observation.meanTagDistanceMeters() <= maximumTagDistanceMeters
        && observation.reprojectionErrorPixels() <= maximumReprojectionErrorPixels;
  }

  /**
   * Consume the metrics captured at the same instant as a pose.
   *
   * <p>Both topics are stamped from one {@code TimingMetadata}, so exact equality is the join key
   * rather than a nearest-match search.
   */
  private double[] takeMeta(long timestamp) {
    for (Iterator<TimestampedDoubleArray> iterator = carriedMetas.iterator();
        iterator.hasNext(); ) {
      TimestampedDoubleArray meta = iterator.next();
      if (meta.timestamp == timestamp) {
        iterator.remove();
        return meta.value.length >= 3 ? meta.value : null;
      }
    }
    return null;
  }
}
