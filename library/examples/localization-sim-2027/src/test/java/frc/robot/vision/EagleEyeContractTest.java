package frc.robot.vision;

import static org.junit.jupiter.api.Assertions.*;
import org.wpilib.hardware.hal.HAL;
import org.wpilib.math.geometry.*;
import org.wpilib.networktables.*;
import java.util.*;
import org.junit.jupiter.api.*;

class EagleEyeContractTest {
  private EagleEyeCamera camera;
  private StructPublisher<Pose3d> pose;
  private DoubleArrayPublisher meta;
  private long now;
  @BeforeAll static void hal() { assertTrue(HAL.initialize()); }
  @BeforeEach void setup() {
    var table = NetworkTableInstance.getDefault().getTable("EagleEye");
    String key = "test/" + UUID.randomUUID();
    camera = new EagleEyeCamera(key + "/pose", key + "/meta");
    pose = table.getStructTopic(key + "/pose", Pose3d.struct).publish(PubSubOption.KEEP_DUPLICATES);
    meta = table.getDoubleArrayTopic(key + "/meta").publish(PubSubOption.KEEP_DUPLICATES);
    now = NetworkTablesJNI.now() - 1_000_000;
  }
  void publish(Pose3d p, double[] m, long time) { pose.set(p,time); meta.set(m,time); }
  Pose3d p(double yaw) { return new Pose3d(2,3,0,new Rotation3d(0,0,yaw)); }
  @Test void nwuMetersAndCounterclockwiseYaw() {
    publish(p(Math.PI/2), new double[]{2,3,0.5}, now);
    var result = camera.poll(); assertEquals(1,result.size());
    assertEquals(2,result.get(0).pose().getX(),1e-9);
    assertEquals(3,result.get(0).pose().getY(),1e-9);
    assertEquals(Math.PI/2,result.get(0).pose().getRotation().getRadians(),1e-9);
    assertEquals(now/1e9,result.get(0).timestampSeconds(),1e-9);
  }
  @Test void joinsAcrossPollCycles() {
    pose.set(p(0),now); assertTrue(camera.poll().isEmpty());
    meta.set(new double[]{2,3,0.5},now); assertEquals(1,camera.poll().size());
    assertTrue(camera.poll().isEmpty());
  }
  @Test void mismatchedTimestampsDoNotJoin() {
    pose.set(p(0),now); meta.set(new double[]{2,3,0.5},now+1);
    assertTrue(camera.poll().isEmpty());
  }
  @Test void staleAndFutureRejected() {
    publish(p(0),new double[]{2,3,0.5},now-1_000_000_000);
    publish(p(0),new double[]{2,3,0.5},now+1_000_000_000);
    assertTrue(camera.poll().isEmpty());
  }
  @Test void qualityGates() {
    double[][] invalid={{1,3,0.5},{2,7,0.5},{2,3,3},{2,Double.NaN,0.5},{2,3,Double.NaN}};
    for(int i=0;i<invalid.length;i++) publish(p(0),invalid[i],now+i);
    assertTrue(camera.poll().isEmpty());
  }
  @Test void negativeMetricsRejected() {
    publish(p(0),new double[]{2,-1,-0.5},now);
    assertTrue(camera.poll().isEmpty());
  }
  @Test void nonfinitePoseRejected() {
    publish(new Pose3d(Double.NaN,3,0,new Rotation3d()),new double[]{2,3,0.5},now);
    assertTrue(camera.poll().isEmpty());
  }
  @Test void fractionalAndInfiniteTagCountRejected() {
    publish(p(0),new double[]{2.5,3,0.5},now);
    publish(p(0),new double[]{Double.POSITIVE_INFINITY,3,0.5},now+1);
    assertTrue(camera.poll().isEmpty());
  }
  @Test void pollReturnsCaptureOrder() {
    publish(p(0),new double[]{2,3,0.5},now);
    publish(p(1),new double[]{2,3,0.5},now+100);
    var result=camera.poll(); assertEquals(2,result.size());
    assertTrue(result.get(0).timestampSeconds() < result.get(1).timestampSeconds());
  }
  @Test void stdDevMatchesContract() {
    var obs=new EagleEyeCamera.Observation(new Pose2d(),1,2,3,0.5);
    assertEquals(0.09,EagleEyeCamera.standardDeviations(obs).get(0,0),1e-12);
  }

  @Test void freshTwentyMillisecondsAcceptedAndSixHundredMillisecondsRejected() {
    long current = NetworkTablesJNI.now();
    publish(p(0), new double[]{2,3,0.5}, current - 20 * 1_000_000L);
    publish(p(0), new double[]{2,3,0.5}, current - 600 * 1_000_000L);
    publish(p(0), new double[]{2,3,0.5}, current + 600 * 1_000_000L);
    var result = camera.poll();
    assertEquals(1, result.size());
    assertEquals((current - 20 * 1_000_000L) / 1e9, result.get(0).timestampSeconds(), 1e-9);
  }
  @Test void timestampMatchesDefaultRobotClock() {
    double robotBefore = org.wpilib.system.Timer.getTimestamp();
    long captured = NetworkTablesJNI.now();
    double robotAfter = org.wpilib.system.Timer.getTimestamp();
    publish(p(0), new double[]{2,3,0.5}, captured);
    var result = camera.poll();
    assertEquals(1, result.size());
    assertTrue(result.get(0).timestampSeconds() >= robotBefore - 0.001);
    assertTrue(result.get(0).timestampSeconds() <= robotAfter + 0.001);
  }
  @Test void unresolvedWarningUsesDriverStationAndFiveSecondThrottle() {
    var warnings = new ArrayList<String>();
    var missing = new EagleEyeCamera("missing/" + UUID.randomUUID(), "missing/" + UUID.randomUUID()) {
      @Override void reportWarning(String message) {
        warnings.add(message);
        super.reportWarning(message); // Exercise the actual Driver Station reporting path too.
      }
    };
    long initial = NetworkTablesJNI.now();
    missing.warnIfKeysUnresolved(initial);
    missing.warnIfKeysUnresolved(initial + 1 * 1_000_000L);
    missing.warnIfKeysUnresolved(initial + 4999 * 1_000_000L);
    assertEquals(1, warnings.size());
    missing.warnIfKeysUnresolved(initial + 5000 * 1_000_000L);
    assertEquals(2, warnings.size());
    assertTrue(warnings.get(0).contains("EagleEye"));
  }

  @Test void simulatorSkipsTagWithNoPlanarBearing() {
    var tagPose = new Pose3d(2,3,1,new Rotation3d());
    var field = new org.wpilib.fields.Field("test", "2027", "test", null, 10, 10, "FRC",
        List.of(new org.wpilib.fields.FieldTag(1, tagPose)));
    var sim = new EagleEyeCameraSim("sim/" + UUID.randomUUID(), new Transform3d(), field);
    // Different height leaves a nonzero 3D distance but an empty planar angle.
    assertDoesNotThrow(() -> sim.update(new Pose2d(2,3,new Rotation2d())));
  }

  @Test void joinedDetectionsExpireAndLateMetadataCannotRevivePose() {
    String source = "detections/" + UUID.randomUUID();
    var table = NetworkTableInstance.getDefault().getTable("EagleEye");
    var consumer = EagleEyeCamera.forSource(source);
    try (var poses = table.getStructTopic(source + "/pose", Pose3d.struct).publish(PubSubOption.KEEP_DUPLICATES);
        var metrics = table.getDoubleArrayTopic(source + "/meta").publish(PubSubOption.KEEP_DUPLICATES);
        var detections = table.getStringArrayTopic(source + "/detections").publish(PubSubOption.KEEP_DUPLICATES)) {
      org.wpilib.simulation.SimHooks.pauseTiming();
      try {
        long captured = NetworkTablesJNI.now() - 1_000_000L;
        poses.set(p(0), captured);
        metrics.set(new double[]{2,3,0.5}, captured);
        detections.set(new String[]{"coral", "3", "4", "0"}, captured);
        assertEquals(1, consumer.poll().size());
        assertEquals("coral", consumer.nearestGamePiece().orElseThrow().className());
        long orphan = captured + 1;
        poses.set(p(0), orphan);
        assertTrue(consumer.poll().isEmpty());
        org.wpilib.simulation.SimHooks.stepTiming(0.6);
        metrics.set(new double[]{2,3,0.5}, orphan);
        assertTrue(consumer.poll().isEmpty());
        assertTrue(consumer.nearestGamePiece().isEmpty());
      } finally {
        org.wpilib.simulation.SimHooks.resumeTiming();
      }
    }
  }
}
