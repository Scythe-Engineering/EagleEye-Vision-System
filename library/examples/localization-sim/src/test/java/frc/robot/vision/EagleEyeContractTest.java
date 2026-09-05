package frc.robot.vision;

import static org.junit.jupiter.api.Assertions.*;
import edu.wpi.first.hal.HAL;
import edu.wpi.first.math.geometry.*;
import edu.wpi.first.networktables.*;
import java.util.*;
import org.junit.jupiter.api.*;

class EagleEyeContractTest {
  private EagleEyeCamera camera;
  private StructPublisher<Pose3d> pose;
  private DoubleArrayPublisher meta;
  private long now;
  @BeforeAll static void hal() { assertTrue(HAL.initialize(500, 0)); }
  @BeforeEach void setup() {
    var table = NetworkTableInstance.getDefault().getTable("EagleEye");
    String key = "test/" + UUID.randomUUID();
    camera = new EagleEyeCamera(key + "/pose", key + "/meta");
    pose = table.getStructTopic(key + "/pose", Pose3d.struct).publish(PubSubOption.keepDuplicates(true));
    meta = table.getDoubleArrayTopic(key + "/meta").publish(PubSubOption.keepDuplicates(true));
    now = NetworkTablesJNI.now() - 1000;
  }
  void publish(Pose3d p, double[] m, long time) { pose.set(p,time); meta.set(m,time); }
  Pose3d p(double yaw) { return new Pose3d(2,3,0,new Rotation3d(0,0,yaw)); }
  @Test void nwuMetersAndCounterclockwiseYaw() {
    publish(p(Math.PI/2), new double[]{2,3,0.5}, now);
    var result = camera.poll(); assertEquals(1,result.size());
    assertEquals(2,result.get(0).pose().getX(),1e-9);
    assertEquals(3,result.get(0).pose().getY(),1e-9);
    assertEquals(Math.PI/2,result.get(0).pose().getRotation().getRadians(),1e-9);
    assertEquals(now/1e6,result.get(0).timestampSeconds(),1e-9);
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
    publish(p(0),new double[]{2,3,0.5},now-1_000_000);
    publish(p(0),new double[]{2,3,0.5},now+1_000_000);
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
}
