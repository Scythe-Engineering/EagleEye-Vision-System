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
import edu.wpi.first.networktables.StringArraySubscriber;
import edu.wpi.first.networktables.StructSubscriber;
import edu.wpi.first.networktables.TimestampedDoubleArray;
import edu.wpi.first.networktables.TimestampedObject;
import edu.wpi.first.networktables.TimestampedStringArray;
import edu.wpi.first.wpilibj.DriverStation;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Optional;

/**
 * One EagleEye localization source: pose, quality metrics, and field-space detections.
 *
 * <p>Keys are supplied by robot code and are relative to the {@code EagleEye} table, so they
 * must match the {@code target_key} on the matching {@code publish_to_networktables} operation in
 * the WebUI character for character. A key that nothing publishes produces a Driver Station
 * warning rather than silence.
 *
 * <p>EagleEye stamps every published pose with the kernel's V4L2 exposure timestamp, and ntcore
 * translates that into roboRIO FPGA time on arrival. The timestamp on a received sample is
 * therefore already in the domain {@code addVisionMeasurement} expects: do not subtract a latency,
 * and do not use {@code Timer.getFPGATimestamp()} as the measurement time.
 *
 * <p>Typical use, in the subsystem that owns the pose estimator:
 *
 * <pre>
 * public class Drive extends SubsystemBase {
 *   private final EagleEyeCamera[] cameras = {
 *     EagleEyeCamera.forSource("localization/front"),
 *     EagleEyeCamera.forSource("localization/back"),
 *   };
 *
 *   public void periodic() {
 *     poseEstimator.update(gyro.getRotation2d(), modulePositions);
 *     EagleEyeCamera.update(poseEstimator::addVisionMeasurement, cameras);
 *   }
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

  /** One detected game piece at a field-space position. */
  public record GamePiece(String className, double xMeters, double yMeters, double zMeters) {}

  private record CapturedPose(long timestampMicros, Pose2d pose) {}

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

  /** How often to repeat the warning about a key nothing publishes, in seconds. */
  public static double warningIntervalSeconds = 5.0;

  /** Root table EagleEye publishes into. Fixed on the coprocessor side too. */
  private static final String TABLE = "EagleEye";

  private static final int POLL_STORAGE = 20;

  private final String poseKey;
  private final String metaKey;
  private final String detectionsKey;
  private final StructSubscriber<Pose3d> poseSubscriber;
  private final DoubleArraySubscriber metaSubscriber;
  private final StringArraySubscriber detectionsSubscriber;
  private long lastWarningMicros;
  private long latestDetectionsTimestamp;
  private Pose2d latestDetectionPose = new Pose2d();
  private List<GamePiece> latestDetections = List.of();

  // A pose and its metrics come from two separate NetworkTables publishers, so a network flush can
  // land between them and split one frame across two poll cycles. Carrying the leftovers for a
  // cycle removes that race; anything genuinely orphaned ages out below.
  private final List<TimestampedObject<Pose3d>> carriedPoses = new ArrayList<>();
  private final List<TimestampedDoubleArray> carriedMetas = new ArrayList<>();
  private final List<TimestampedStringArray> carriedDetections = new ArrayList<>();
  private final List<CapturedPose> recentPoses = new ArrayList<>();

  /**
   * Subscribe to a source that follows the shipped preset's naming.
   *
   * <p>Subscribes to {@code /pose}, {@code /meta}, and {@code /detections} below the source. Use
   * the three-key constructor when the pipeline publishes somewhere else.
   *
   * @param source key prefix shared by the {@code /pose}, {@code /meta}, and {@code /detections}
   *     topics, such as {@code "localization/front"}.
   * @return a camera reading all three source topics.
   */
  public static EagleEyeCamera forSource(String source) {
    return new EagleEyeCamera(
        source + "/pose", source + "/meta", source + "/detections");
  }

  /**
   * Subscribe to one EagleEye localization source by its exact keys.
   *
   * <p>Both keys are relative to the {@code EagleEye} table and must match the {@code target_key}
   * set on the corresponding publish operation in the WebUI.
   *
   * @param poseKey key of the {@code Pose3d} struct topic, for example {@code
   *     "localization/front/pose"}.
   * @param metaKey key of the {@code double[3]} metrics topic, for example {@code
   *     "localization/front/meta"}.
   */
  public EagleEyeCamera(String poseKey, String metaKey) {
    this(poseKey, metaKey, null);
  }

  /**
   * Subscribe to localization and field-space detections by their exact keys.
   *
   * @param poseKey key of the {@code Pose3d} struct topic.
   * @param metaKey key of the {@code double[3]} metrics topic.
   * @param detectionsKey key of the flattened {@code String[]} detections topic.
   */
  public EagleEyeCamera(String poseKey, String metaKey, String detectionsKey) {
    this.poseKey = poseKey;
    this.metaKey = metaKey;
    this.detectionsKey = detectionsKey;
    var table = NetworkTableInstance.getDefault().getTable(TABLE);
    // sendAll and keepDuplicates stop the server from coalescing frames the estimator wants: at
    // 90 fps against a 50 Hz loop, a plain subscription throws away most of the measurements.
    var options =
        new PubSubOption[] {
          PubSubOption.sendAll(true),
          PubSubOption.keepDuplicates(true),
          PubSubOption.pollStorage(POLL_STORAGE),
        };
    poseSubscriber = table.getStructTopic(poseKey, Pose3d.struct).subscribe(new Pose3d(), options);
    metaSubscriber = table.getDoubleArrayTopic(metaKey).subscribe(new double[0], options);
    detectionsSubscriber =
        detectionsKey == null
            ? null
            : table.getStringArrayTopic(detectionsKey).subscribe(new String[0], options);
  }

  /**
   * Drain every measurement received since the last call, oldest first.
   *
   * <p>Samples that are stale, unconverged, or fail the quality gates are dropped. A key that
   * nothing publishes raises a Driver Station warning naming the key, repeated every {@link
   * #warningIntervalSeconds}.
   *
   * @return accepted observations in capture order.
   */
  public List<Observation> poll() {
    carriedPoses.addAll(Arrays.asList(poseSubscriber.readQueue()));
    carriedMetas.addAll(Arrays.asList(metaSubscriber.readQueue()));
    if (detectionsSubscriber != null) {
      carriedDetections.addAll(Arrays.asList(detectionsSubscriber.readQueue()));
    }

    long nowMicros = NetworkTablesJNI.now();
    warnIfKeysUnresolved(nowMicros);
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
        recentPoses.add(new CapturedPose(sample.timestamp, observation.pose()));
      }
    }

    carriedPoses.clear();
    carriedPoses.addAll(unmatched);
    // Samples dated more than the age limit in either direction can never join a pose the poller
    // would accept, so drop them; without the future bound a skewed publisher grows these lists
    // forever.
    carriedMetas.removeIf(
        meta -> Math.abs(nowMicros - meta.timestamp) / 1e6 > maximumSampleAgeSeconds);
    joinDetections(nowMicros);
    recentPoses.removeIf(
        pose -> (nowMicros - pose.timestampMicros()) / 1e6 > maximumSampleAgeSeconds);
    return observations;
  }

  /**
   * Return the nearest game piece from the newest detection frame joined to an accepted pose.
   *
   * <p>Call {@link #poll()} directly or {@link #update(VisionConsumer, EagleEyeCamera...)} first in
   * the current robot loop so queued NetworkTables samples are drained.
   *
   * @return the nearest field-space game piece, or empty before a joined detection frame arrives.
   */
  public Optional<GamePiece> nearestGamePiece() {
    return nearestGamePiece(null);
  }

  /**
   * Return the nearest game piece of one class from the newest joined detection frame.
   *
   * @param className exact model class name, or {@code null} to accept every class.
   * @return the nearest matching field-space game piece, or empty when none is fresh.
   */
  public Optional<GamePiece> nearestGamePiece(String className) {
    double ageSeconds = (NetworkTablesJNI.now() - latestDetectionsTimestamp) / 1e6;
    if (latestDetectionsTimestamp == 0L
        || ageSeconds < 0.0
        || ageSeconds > maximumSampleAgeSeconds) {
      return Optional.empty();
    }
    return latestDetections.stream()
        .filter(piece -> className == null || piece.className().equals(className))
        .min(
            Comparator.comparingDouble(
                piece ->
                    Math.hypot(
                        piece.xMeters() - latestDetectionPose.getX(),
                        piece.yMeters() - latestDetectionPose.getY())));
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
   * Warn when a configured key has no publisher, since the alternative is silence.
   *
   * <p>Rate limited to one report per {@link #warningIntervalSeconds}, and reset once both topics
   * appear so a coprocessor reboot warns again.
   */
  private void warnIfKeysUnresolved(long nowMicros) {
    boolean poseMissing = !poseSubscriber.getTopic().exists();
    boolean metaMissing = !metaSubscriber.getTopic().exists();
    boolean detectionsMissing =
        detectionsSubscriber != null && !detectionsSubscriber.getTopic().exists();
    if (!poseMissing && !metaMissing && !detectionsMissing) {
      lastWarningMicros = 0L;
      return;
    }
    if (lastWarningMicros != 0L
        && (nowMicros - lastWarningMicros) / 1e6 < warningIntervalSeconds) {
      return;
    }
    lastWarningMicros = nowMicros;
    DriverStation.reportWarning(
        unresolvedKeyMessage(poseMissing, metaMissing, detectionsMissing), false);
  }

  /** Say which key is missing and what to compare it against. */
  private String unresolvedKeyMessage(
      boolean poseMissing, boolean metaMissing, boolean detectionsMissing) {
    String posePath = TABLE + "/" + poseKey;
    String metaPath = TABLE + "/" + metaKey;
    if (!NetworkTableInstance.getDefault().isConnected()) {
      return "EagleEye: nothing is connected to NetworkTables, so "
          + posePath
          + " cannot arrive. Check the coprocessor is powered and pointed at this roboRIO.";
    }
    if (poseMissing && metaMissing) {
      return "EagleEye: nothing publishes "
          + posePath
          + " or "
          + metaPath
          + ". The keys passed to EagleEyeCamera must match the publish_to_networktables"
          + " target_key values in the WebUI exactly.";
    }
    if (detectionsMissing) {
      return "EagleEye: localization publishes but "
          + TABLE
          + "/"
          + detectionsKey
          + " is missing. Check the detection publisher's target_key and detections schema.";
    }
    if (metaMissing) {
      return "EagleEye: "
          + posePath
          + " is publishing but "
          + metaPath
          + " is not, so every pose is dropped. Check that key against the target_key of the"
          + " publisher on the solver's pose_meta port in the WebUI.";
    }
    return "EagleEye: "
        + metaPath
        + " is publishing but "
        + posePath
        + " is not. Check that key against the target_key of the pose publisher in the WebUI.";
  }

  /**
   * Consume the metrics captured at the same instant as a pose.
   *
   * <p>Both topics are stamped from one {@code TimingMetadata}, so exact equality is the join key
   * rather than a nearest-match search.
   */
  /** Join detection frames to trustworthy poses using their exact capture timestamps. */
  private void joinDetections(long nowMicros) {
    for (Iterator<TimestampedStringArray> iterator = carriedDetections.iterator();
        iterator.hasNext(); ) {
      TimestampedStringArray sample = iterator.next();
      Pose2d pose = poseAt(sample.timestamp);
      if (pose != null) {
        iterator.remove();
        if (sample.timestamp >= latestDetectionsTimestamp) {
          latestDetectionsTimestamp = sample.timestamp;
          latestDetectionPose = pose;
          latestDetections = parseDetections(sample.value);
        }
      }
    }
    carriedDetections.removeIf(
        sample -> Math.abs(nowMicros - sample.timestamp) / 1e6 > maximumSampleAgeSeconds);
  }

  /** Find the accepted pose captured at an exact NetworkTables timestamp. */
  private Pose2d poseAt(long timestamp) {
    for (CapturedPose pose : recentPoses) {
      if (pose.timestampMicros() == timestamp) {
        return pose.pose();
      }
    }
    return null;
  }

  /** Parse repeating class/x/y/z groups from the detections StringArray topic. */
  private static List<GamePiece> parseDetections(String[] values) {
    var detections = new ArrayList<GamePiece>(values.length / 4);
    for (int index = 0; index + 3 < values.length; index += 4) {
      try {
        double x = Double.parseDouble(values[index + 1]);
        double y = Double.parseDouble(values[index + 2]);
        double z = Double.parseDouble(values[index + 3]);
        if (!Double.isFinite(x) || !Double.isFinite(y) || !Double.isFinite(z)) {
          // "NaN" and "Infinity" parse fine but are unusable field coordinates.
          continue;
        }
        detections.add(new GamePiece(values[index], x, y, z));
      } catch (NumberFormatException ignored) {
        // Malformed detections are dropped without weakening the rest of the frame.
      }
    }
    return List.copyOf(detections);
  }

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
