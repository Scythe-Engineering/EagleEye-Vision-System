package frc.robot;

import edu.wpi.first.math.geometry.Pose3d;
import edu.wpi.first.networktables.*;
import edu.wpi.first.wpilibj.Notifier;
import edu.wpi.first.wpilibj.TimedRobot;
import java.io.*;
import java.net.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;

/** Bounded passive experiment. UDP data are diagnostic and never fused into robot state. */
public final class LatencyExperiment implements AutoCloseable {
  private static final String ROOT = "/EagleEye/localization/arducam-ov9281-usb-camera/";
  private static final String AUDIT = "/EagleEye/audit/latency/frame";
  private final NetworkTableInstance nt = NetworkTableInstance.getDefault();
  private final ConcurrentLinkedQueue<String> rows = new ConcurrentLinkedQueue<>();
  private final PrintWriter out;
  private final long deadline = System.nanoTime() + 900_000_000_000L;
  private volatile boolean active = true;
  private final Lane normal = new Lane("robot20");
  private final Lane scheduled = new Lane("scheduled5");
  private final Lane thread5 = new Lane("thread5");
  private final Lane thread1 = new Lane("thread1");
  private final Lane eventLane = new Lane("event");
  private final Notifier five = new Notifier(thread5::poll);
  private final Notifier one = new Notifier(thread1::poll);
  private final int listener;
  private final DatagramSocket socket;
  private long lastFlush;

  public LatencyExperiment(TimedRobot robot) {
    try {
      Path root = Path.of("evidence/latency-experiments-2026-09-08");
      Files.createDirectories(root);
      out = new PrintWriter(Files.newBufferedWriter(root.resolve("lanes-" + System.currentTimeMillis() + ".csv")));
      out.println("lane,kind,seq,capture_us,ready_us,consume_us,epoch_ms,scenario,tags,error");
      socket = new DatagramSocket(new InetSocketAddress("100.75.14.59", 5810));
    } catch (IOException e) { throw new UncheckedIOException(e); }
    robot.addPeriodic(scheduled::poll, .005, .002);
    five.setName("VisionLatency5ms"); five.startPeriodic(.005);
    one.setName("VisionLatency1ms"); one.startPeriodic(.001);
    listener = nt.addListener(new String[]{ROOT, AUDIT}, EnumSet.of(NetworkTableEvent.Kind.kValueRemote), e -> eventLane.poll());
    Thread receiver = new Thread(this::udpLoop, "VisionLatencyUDP");
    receiver.setDaemon(true); receiver.start();
  }

  public void robotPeriodic() {
    if (!active) return;
    normal.poll();
    String row;
    while ((row = rows.poll()) != null) out.println(row);
    long now = System.nanoTime();
    if (now - lastFlush > 1_000_000_000L) { out.flush(); lastFlush = now; }
    if (now >= deadline) close();
  }

  private boolean valid(double[] v) {
    if (v.length != 15) return false;
    for (double d : v) if (!Double.isFinite(d)) return false;
    // Validate transport framing/metrics; keep the physical camera diagnostic-only.
    return v[0] > 0 && v[0] == Math.rint(v[0]) && v[2] >= v[1]
        && v[10] >= 0 && v[10] == Math.rint(v[10]) && v[11] >= 0 && v[12] >= 0;
  }

  private void record(String lane, String kind, double[] v, long now) {
    if (!active || !valid(v)) return;
    rows.offer(lane + "," + kind + "," + (long)v[0] + "," + (long)v[1] + "," + (long)v[2]
        + "," + now + "," + System.currentTimeMillis() + "," + (int)v[13] + "," + v[10] + "," + v[12]);
  }

  private void udpLoop() {
    byte[] data = new byte[120];
    DatagramPacket packet = new DatagramPacket(data, data.length);
    long lastSeq = 0;
    try {
      while (active) {
        packet.setLength(data.length);
        socket.receive(packet);
        long now = NetworkTablesJNI.now();
        if (packet.getLength() != 120 || !packet.getAddress().getHostAddress().equals("100.84.225.35")) continue;
        double[] v = new double[15];
        ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer().get(v);
        if (!valid(v)) continue;
        record("udp", v[0] <= lastSeq ? "reordered" : "frame", v, now);
        lastSeq = Math.max(lastSeq, (long)v[0]);
      }
    } catch (IOException e) { if (active) e.printStackTrace(); }
  }

  private final class Lane implements AutoCloseable {
    private final String name;
    private final StructSubscriber<Pose3d> poses;
    private final DoubleArraySubscriber metas;
    private final DoubleArraySubscriber frames;
    private final Map<Long, Pose3d> pendingPoses = new HashMap<>();
    private final Map<Long, double[]> pendingMetas = new HashMap<>();
    private final Map<Long, double[]> pendingFrames = new HashMap<>();
    Lane(String name) {
      this.name = name;
      PubSubOption[] opts = {PubSubOption.sendAll(true), PubSubOption.keepDuplicates(true), PubSubOption.pollStorage(4096)};
      poses = nt.getStructTopic(ROOT + "pose", Pose3d.struct).subscribe(new Pose3d(), opts);
      metas = nt.getDoubleArrayTopic(ROOT + "meta").subscribe(new double[0], opts);
      frames = nt.getDoubleArrayTopic(AUDIT).subscribe(new double[0], opts);
    }
    synchronized void poll() {
      if (!active) return;
      for (var p : poses.readQueue()) pendingPoses.put(p.timestamp, p.value);
      for (var m : metas.readQueue()) pendingMetas.put(m.timestamp, m.value);
      for (var f : frames.readQueue()) {
        record(name, "frame", f.value, NetworkTablesJNI.now());
        if (valid(f.value)) pendingFrames.put(f.timestamp, f.value);
      }
      var it = pendingFrames.entrySet().iterator();
      long now = NetworkTablesJNI.now();
      while (it.hasNext()) {
        var entry = it.next(); long capture = entry.getKey(); double[] v = entry.getValue();
        Pose3d pose = pendingPoses.get(capture); double[] meta = pendingMetas.get(capture);
        if (pose != null && meta != null && meta.length == 3) {
          boolean match = Math.abs(pose.getX()-v[3]) < 1e-9 && Math.abs(pose.getY()-v[4]) < 1e-9
              && Math.abs(pose.getZ()-v[5]) < 1e-9 && Math.abs(meta[0]-v[10]) < 1e-9
              && Math.abs(meta[1]-v[11]) < 1e-9 && Math.abs(meta[2]-v[12]) < 1e-9;
          record(name, match ? "joined" : "mismatch", v, NetworkTablesJNI.now());
          pendingPoses.remove(capture); pendingMetas.remove(capture); it.remove();
        } else if (now - capture > 1_000_000) it.remove();
      }
      pendingPoses.keySet().removeIf(t -> now-t > 1_000_000);
      pendingMetas.keySet().removeIf(t -> now-t > 1_000_000);
    }
    public synchronized void close() { poses.close(); metas.close(); frames.close(); }
  }

  @Override public void close() {
    if (!active) return;
    active = false;
    nt.removeListener(listener); five.stop(); one.stop(); socket.close();
    for (Lane lane : List.of(normal, scheduled, thread5, thread1, eventLane)) lane.close();
    String row; while ((row = rows.poll()) != null) out.println(row);
    out.close(); five.close(); one.close();
  }
}
