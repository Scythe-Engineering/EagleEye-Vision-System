package frc.robot;

import edu.wpi.first.math.estimator.DifferentialDrivePoseEstimator;
import edu.wpi.first.math.geometry.*;
import edu.wpi.first.math.kinematics.DifferentialDriveKinematics;
import edu.wpi.first.networktables.NetworkTablesJNI;
import edu.wpi.first.wpilibj.Notifier;
import edu.wpi.first.wpilibj.Timer;
import edu.wpi.first.wpilibj.smartdashboard.SmartDashboard;
import frc.robot.vision.*;
import java.io.*;
import java.nio.file.*;
import java.util.concurrent.ConcurrentLinkedQueue;

/** Experiment: 120 Hz synthetic capture, separate 200 Hz odometry+SDK+fusion owner.
 * The main loop owns an independent 50 Hz estimator for the same observations.
 * The real camera remains outside the unrelated simulated drivetrain estimator.
 */
public final class HighRateVisionExperiment implements AutoCloseable {
  private final double start = Timer.getFPGATimestamp();
  private final EagleEyeCameraSim simulated = new EagleEyeCameraSim("audit/highrate");
  private final Worker baseline = new Worker("main20");
  private final Worker fast = new Worker("thread5");
  private final Notifier capture = new Notifier(() -> simulated.update(truth(Timer.getFPGATimestamp()-start)));
  private final Notifier odometry = new Notifier(fast::update);
  private final ConcurrentLinkedQueue<String> rows = new ConcurrentLinkedQueue<>();
  private final PrintWriter out;
  private volatile boolean active = true;
  private long lastFlush;
  private record Snapshot(long updates, long accepted, double error, double maxGapMs) {}

  public HighRateVisionExperiment() {
    try {
      Path root = Path.of("evidence/latency-experiments-2026-09-08");
      Files.createDirectories(root);
      out = new PrintWriter(Files.newBufferedWriter(root.resolve("fusion-"+System.currentTimeMillis()+".csv")));
      out.println("lane,capture_us,consume_us,epoch_ms");
    } catch (IOException e) { throw new UncheckedIOException(e); }
    capture.setName("SyntheticVision120Hz"); capture.startPeriodic(1.0/120.0);
    odometry.setName("OdometryAndVision200Hz"); odometry.startPeriodic(.005);
  }

  public void robotPeriodic() {
    if (!active) return;
    baseline.update();
    String row; while ((row=rows.poll()) != null) out.println(row);
    for (Worker worker : new Worker[]{baseline, fast}) {
      Snapshot s = worker.snapshot;
      SmartDashboard.putNumber("HighRate/"+worker.name+"/Updates", s.updates());
      SmartDashboard.putNumber("HighRate/"+worker.name+"/Accepted", s.accepted());
      SmartDashboard.putNumber("HighRate/"+worker.name+"/ErrorMeters", s.error());
      SmartDashboard.putNumber("HighRate/"+worker.name+"/MaxGapMs", s.maxGapMs());
    }
    if (System.nanoTime()-lastFlush>1_000_000_000L) { out.flush(); lastFlush=System.nanoTime(); }
    if (Timer.getFPGATimestamp()-start > 120) close();
  }

  private static Pose2d truth(double t) {
    double a=.2*t;
    return new Pose2d(4+2*Math.sin(a), 2+2*(1-Math.cos(a)), new Rotation2d(a));
  }

  private final class Worker {
    private final String name;
    private final EagleEyeCamera camera = new EagleEyeCamera("audit/highrate/pose", "audit/highrate/meta");
    private final DifferentialDrivePoseEstimator estimator = new DifferentialDrivePoseEstimator(
        new DifferentialDriveKinematics(.6), new Rotation2d(), 0, 0, truth(0));
    private volatile Snapshot snapshot = new Snapshot(0,0,0,0);
    private long updates, accepted;
    private double previous, maxGap;
    Worker(String name) { this.name=name; }
    void update() {
      if (!active) return;
      // The legacy bench briefly changes the SDK's static diagnostic quality limit.
      // Its whole periodic body shares this lock, so the strict poll never sees 4 px.
      // Each estimator and subscriber has exactly one owning thread.
      synchronized (EagleEyeCamera.class) {
        double now=Timer.getFPGATimestamp(), t=now-start;
        Pose2d actual=truth(t);
        estimator.updateWithTime(now,actual.getRotation(),.4*t,.4*t);
        for (var observation : camera.poll()) {
          long consume=NetworkTablesJNI.now();
          estimator.addVisionMeasurement(observation.pose(),observation.timestampSeconds(),
              EagleEyeCamera.standardDeviations(observation));
          accepted++;
          rows.offer(name+","+Math.round(observation.timestampSeconds()*1e6)+","+consume+","+System.currentTimeMillis());
        }
        updates++;
        if (previous>0) maxGap=Math.max(maxGap,(now-previous)*1000);
        previous=now;
        snapshot=new Snapshot(updates,accepted,estimator.getEstimatedPosition().getTranslation()
            .getDistance(actual.getTranslation()),maxGap);
      }
    }
  }

  @Override public void close() {
    if (!active) return;
    // Called on the main loop outside the shared SDK lock, never while holding it.
    active=false; capture.stop(); odometry.stop();
    String row; while ((row=rows.poll()) != null) out.println(row);
    out.close(); capture.close(); odometry.close();
  }
}
