# Time Synchronization and Capture Latency

How EagleEye stamps a frame with the moment it was captured, carries that stamp
through the pipeline, and hands it to robot code over NetworkTables so a WPILib
pose estimator can fuse vision at the right point in its history.

## 1. How it works

### 1.1 The timing payload

`src/utils/timing.py` defines two frozen dataclasses:

```python
@dataclass(frozen=True)
class TimingMetadata:
    capture_nt_us: int            # NT-clock capture time, microseconds
    capture_monotonic_ns: int     # CLOCK_MONOTONIC capture time, nanoseconds
    frame_seq: int | None         # per-camera monotonic frame counter
    camera_name: str | None
    bus_id: str | None

@dataclass(frozen=True)
class TimedValue(Generic[T]):
    value: T
    timing: TimingMetadata
```

`TimedValue` is the envelope that travels the graph. Everything else in the
module is plumbing for it: `unwrap_timed`, `unwrap_timed_deep` (recursive,
container-shape preserving), `collect_timings`, `average_timings`, `retime`,
and `attach_output_timing`.

The module also owns the two clocks. `now_nt_us()` wraps the single private
ntcore entry point the project depends on, and `monotonic_ns_to_nt_us()`
converts a `CLOCK_MONOTONIC` reading into the NT clock using an offset measured
once (median of eleven paired samples) on first use. Both clocks tick at the
same rate, so the offset is constant and the conversion preserves intervals
exactly.

### 1.2 Where the stamp is created

The timestamp originates at the driver, not in Python. `Camera.get_frame()`
returns a `CapturedFrame` — the image plus `capture_monotonic_ns` — and each
backend supplies the best capture time it can:

| Backend | Platform | Timestamp |
| --- | --- | --- |
| `V4l2Capture` | Linux | Kernel `v4l2_buffer.timestamp`, start-of-exposure where the driver reports it |
| `OpenCvCapture` | non-Linux dev hosts | Delivery time, stamped when `read()` returns |
| `VideoFileCamera` | replay | Delivery time; replayed frames have no capture instant |

`CameraThreadManager.camera_feed_worker` converts and wraps it:

```python
packet = TimedValue(
    captured.image,
    TimingMetadata(
        capture_nt_us=monotonic_ns_to_nt_us(captured.capture_monotonic_ns),
        capture_monotonic_ns=captured.capture_monotonic_ns,
        frame_seq=worker.next_frame_seq(),
        camera_name=camera_name,
        bus_id=self.get_bus_id_for_camera_name(camera_name),
    ),
)
```

The packet is stored twice under the worker's lock: as the *cached* packet (last
known good frame) and as the *current* packet. Storing the current packet
notifies a `threading.Condition`, but only when `frame_seq` actually changed —
that is the wakeup edge consumers block on.

On capture failure the worker re-publishes the **cached** packet unchanged, so a
reused frame keeps its original timestamp and its original `frame_seq` rather
than being restamped as fresh.

#### The V4L2 backend

`src/utils/camera_utils/cameras/v4l2_capture.py` streams MJPEG through
memory-mapped buffers, with the ioctl structures defined in `ctypes`. Three
properties matter:

1. **Four buffers stay queued.** Depth costs nothing because every buffer
   carries its own capture time. The earlier `CAP_PROP_BUFFERSIZE = 1`
   experiment halved the frame rate precisely because OpenCV decoded MJPEG
   *before* requeueing, leaving the driver with nowhere to write for the whole
   decode.
2. **`read()` drains to the newest frame.** Every pending buffer is dequeued and
   requeued, but only the newest payload is decoded. A backlog is discarded
   rather than delivered late, and skipped frames cost no decode time.
3. **Decode happens after requeue.** The compressed bytes are copied out
   (~100–200 KB, tens of microseconds), the buffer goes straight back to the
   driver, and `cv2.imdecode` runs outside the window where the queue could
   starve.

`uvcvideo` derives its timestamp from the UVC payload header clock and the USB
SOF counter, so it reflects the sensor rather than when this process was
scheduled. When a driver reports a non-monotonic timestamp the backend logs once
and falls back to delivery time, which is no worse than the old behaviour.

### 1.3 Entering the pipeline

`DeviceInput.run` pulls the current packet by bus ID, applies rotation to the
image, and re-wraps with the *same* `TimingMetadata`. Rotation does not restamp.
The async path, `wait_for_next_packet`, blocks on the camera condition variable
until `frame_seq > after_frame_seq`, then returns the same way.

### 1.4 Propagation through the graph

`Operation.run` (`src/config/utils/operation.py:72`) is the single choke point:

- By default an operation is handed **unwrapped** values (`unwrap_timed_deep`),
  so ordinary image/dict processors never have to know timing exists.
- An operation opts in by setting `uses_timed_inputs = True` on the instance;
  it then receives the `TimedValue` itself. Only `PublishToNetworktables` does
  this today.
- On the way out, `attach_output_timing(output, input_data)` re-wraps the raw
  result with timing recovered from the inputs. For multi-output operations each
  declared port is wrapped independently, and a branch the operation already
  returned as a `TimedValue` is left alone.

`attach_output_timing` calls `collect_timings` to walk the input structure. One
timing found → that timing is reused verbatim (identity preserved). Several →
`average_timings` produces the arithmetic mean of `capture_nt_us` and
`capture_monotonic_ns`. That is the multi-camera fusion path: `PoseFusion` is a
plain untimed operation, and the fused pose ends up stamped with the mean
capture time of the cameras that contributed. See section 2.3.

A crucial detail lives in `FlowManager._gather_operation_inputs`: **default
(previous-frame) connections are `unwrap_timed_deep`'d** before being handed
over. Feedback edges therefore contribute no timing, so a stale loop-carried
value can never drag an output's timestamp backwards.

### 1.5 Leaving over NetworkTables

`PublishToNetworktables` is timing-aware. `_select_value` walks `data_path` on
the unwrapped value and `retime`s the selection, then `_publish` does:

```python
if timing is not None:
    self._publisher.set(wpi_value, timing.capture_nt_us)
else:
    self._publisher.set(wpi_value)
```

Because `capture_nt_us` is in the NT clock, ntcore treats it as a local
timestamp and converts it to server time using the NT4 RTT-derived offset when
the value goes on the wire. The subscriber on the robot reads it back in *its*
local base — which on a roboRIO is the FPGA timestamp base that
`Timer.getFPGATimestamp()` and the WPILib pose estimators use. No latency topic,
no subtraction, no clock agreement work on the user's part.

The NT client is started in `MainBackend` as `startClient4("EagleEye")` against
the configured server address.

### 1.6 Frame identity and freshness

Timing doubles as the pipeline's frame-identity mechanism. `Pipeline` builds a
`FrameToken = (camera_name, bus_id, frame_seq)` from the packet.
`_all_device_inputs_are_fresh` refuses to run a cycle until every `DeviceInput`
reports a token different from the one the last run consumed, which is what
`limit_frames_to_camera_capture_speed` enforces. Including `camera_name`/
`bus_id` in the token means a camera worker being replaced (its `frame_seq`
restarting at 1) cannot alias into a stale match.

### 1.7 Async inference

The MX3 runtime (`src/utils/mx3_runtime.py`) queues in-flight frames and, when
the accelerator returns detections, stamps the result with
`inflight.packet.timing` — the timing of the frame that was actually submitted,
not of whatever the camera holds now. This is the piece that makes deep async
inference safe: a 3-frame-deep queue does not smear the timestamp.

### 1.8 End-to-end summary

```
driver:        v4l2_buffer.timestamp (CLOCK_MONOTONIC, sensor-derived)
camera thread: drain to newest -> requeue -> decode -> CapturedFrame
             -> monotonic_ns_to_nt_us() -> TimedValue(frame, timing)
             -> cached + current packet (condition notify on frame_seq change)
DeviceInput  -> rotate, same timing
Operation.run-> unwrap -> instance.run -> attach_output_timing (reuse or average)
             (previous-frame edges deliberately contribute no timing)
Publish      -> publisher.set(value, timing.capture_nt_us)
ntcore       -> local -> server time conversion
robot        -> sample.timestamp (local/FPGA us) -> addVisionMeasurement
```

### 1.9 The latency metric

`Pipeline._capture_latency_ms` runs after every cycle. It takes the oldest
`capture_monotonic_ns` among the Device Input packets that cycle consumed and
subtracts it from `time.monotonic_ns()`, giving the age of the frame the
pipeline just finished computing on. The oldest is used because a cycle is only
as fresh as its stalest camera.

The value rides the existing profiling snapshot as `capture_latency_ms`, is
pushed over SSE as part of `profiling_update`, and appears in the pipeline
creator as `Latency: N ms` beside Flow and FPS, with a per-sample mean in the
profiling details popup when cumulative-average mode is on.

Everything here is measured on the monotonic clock, so the metric is unaffected
by NetworkTables clock synchronisation. It is the number to watch when tuning:
FPS tells you throughput, latency tells you how stale the pose is.

## 2. What could still be improved

### 2.1 Residual capture latency

The V4L2 backend removes USB transfer, decode, and scheduler jitter from the
stamp. What remains is whatever the driver's own timestamp does not account for:
on `uvcvideo` that is small, but a device reporting end-of-frame rather than
start-of-exposure still lags true mid-exposure by roughly the exposure time.

If residual bias shows up (see section 3.6), add a per-camera constant
subtracted at stamp time. A systematic timestamp bias becomes a pose bias
proportional to robot velocity: 1 m/s × 10 ms = 1 cm. The backend logs its
timestamp source on the first frame, so check whether it reports
`start-of-exposure` or `end-of-frame` before assuming a correction is needed.

### 2.2 Latency is measured but not published to NetworkTables

`capture_monotonic_ns` drives the WebUI latency metric (section 1.9), but
nothing puts it on NetworkTables. Publishing it as
`/EagleEye/<pipeline>/latency_ms`, plus a p95, would make inference-cost
regressions visible on a driver-station dashboard instead of showing up as
unexplained pose noise.

### 2.3 Averaging is the wrong reducer for multi-camera fusion

`average_timings` takes the arithmetic mean. If camera A was captured 40 ms ago
and camera B 5 ms ago, the fused pose is stamped 22.5 ms ago — a time at which
*neither* observation existed. The pose estimator then replays odometry to a
point that half the evidence post-dates. Taking the **oldest** contributing
timestamp is the conservative choice: the measurement is then never treated as
newer than its stalest ingredient. Better still: reject fusion when the spread
between contributing timestamps exceeds a threshold (say one frame period),
since fusing observations that far apart is unsound in the first place
regardless of how it is stamped.

Note also that `average_timings` drops `frame_seq`, `camera_name`, and `bus_id`,
so anything downstream of a fusion node loses frame identity. That is fine today
(only `DeviceInput` outputs feed the freshness tokens) but it is a sharp edge.

### 2.4 Derived-from-history values understate their own age

An output computed partly from a previous-frame (default) edge is stamped with
the current frame's capture time, because feedback edges are unwrapped. That is
the right call for the common case, but for genuinely temporal operations —
temporal acceleration ROI prediction, outlier filters with history — the emitted
value is partly a function of older data, and nothing records that. If any such
operation ever publishes directly, it will claim to be fresher than it is. Worth
either documenting as a known limitation or giving those operations
`uses_timed_inputs` so they can stamp deliberately.

### 2.5 No staleness guard, and no visibility into time-sync health

Two related gaps:

- If a camera stops delivering, the worker republishes the cached packet.
  `frame_seq` doesn't advance, so a frame-limited pipeline correctly stalls — but
  an unlimited pipeline will happily keep recomputing and republishing an old
  observation with an old timestamp. WPILib's estimator will mostly do the right
  thing with it, but publishing a `max_age_ms` guard in
  `PublishToNetworktables` (skip if `now - capture_nt_us > threshold`) is
  cheaper than relying on every consumer to filter.
- Before the NT4 client has completed RTT sync, ntcore's local/server offset is
  a best guess. There is no published indicator of sync state, so robot code
  cannot tell "coprocessor just booted, timestamps are provisional" from steady
  state. Publishing connection state plus the observed offset would let robot
  code gate `addVisionMeasurement` during the first second after connect.

## 3. Using this from robot code

### 3.1 The contract

For every timed value, EagleEye publishes with an explicit timestamp equal to
the capture time of the frame the value was derived from, expressed in the NT
clock. On the robot, `TimestampedX.timestamp` from a subscriber is that same
instant expressed in the robot's local base — the FPGA microsecond clock. So:

```
timestampSeconds = sample.timestamp / 1_000_000.0
```

is directly comparable to `Timer.getFPGATimestamp()` and is exactly what
`SwerveDrivePoseEstimator.addVisionMeasurement` wants. **Do not** subtract a
latency, and do not use `Timer.getFPGATimestamp()` at the moment you read the
value — both double-count.

### 3.2 Subscribe

Set up once, in the constructor. `getAtomic()`/`readQueue()` return value and
timestamp together, which is the only way to avoid tearing them apart.

```java
private final StructSubscriber<Pose2d> poseSub =
    NetworkTableInstance.getDefault()
        .getTable("EagleEye")
        .getStructTopic("robot_pose", Pose2d.struct)
        .subscribe(Pose2d.kZero, PubSubOption.keepDuplicates(true),
                   PubSubOption.pollStorage(20));
```

`keepDuplicates(true)` matters: a stationary robot produces near-identical poses,
and without it NT may coalesce samples you wanted to fuse. `pollStorage` sizes
the queue so a 20 ms robot loop can drain a 60 fps camera without dropping.

### 3.3 Consume in the periodic loop

Drain the queue every loop and feed each sample in capture order. This is the
whole integration:

```java
@Override
public void periodic() {
  // Odometry first, so the estimator's history covers the vision timestamps.
  poseEstimator.update(gyro.getRotation2d(), modulePositions());

  for (var sample : poseSub.readQueue()) {
    double captureTime = sample.timestamp / 1_000_000.0;   // FPGA seconds

    // Reject stale or implausible samples before they enter the estimator.
    if (Timer.getFPGATimestamp() - captureTime > 0.5) continue;
    if (!isOnField(sample.value)) continue;

    poseEstimator.addVisionMeasurement(sample.value, captureTime, stdDevsFor(sample.value));
  }
}
```

Three things make this correct:

1. **`update()` before `addVisionMeasurement()`.** The estimator replays odometry
   backwards to `captureTime`, applies the correction there, and re-integrates
   forward. It can only do that if its buffer already spans that instant. Call
   `update()` at a steady rate (every loop) regardless of whether vision arrived.
2. **The buffer must be long enough.** `SwerveDrivePoseEstimator` keeps 1.5 s of
   history by default. A vision sample older than the buffer is silently
   discarded. Capture plus inference plus network is typically 30–120 ms, so the
   default is ample — but if you add a long filter chain, check the latency
   metric from section 1.9 against it.
3. **Out-of-order arrival is fine.** Multiple cameras publishing independently
   will interleave. The estimator handles retroactive insertion; that is the
   entire point of a timestamped API. Do not sort or drop them.

### 3.4 Standard deviations

The timestamp fixes *when*; it says nothing about *how much* to trust. Scale
`addVisionMeasurement`'s std-devs with observation quality — tag distance, tag
count, ambiguity:

```java
private Matrix<N3, N1> stdDevsFor(Pose2d pose) {
  double d = distanceToNearestTag(pose);
  if (tagCount() >= 2) return VecBuilder.fill(0.5 * d, 0.5 * d, 1.0);
  return VecBuilder.fill(2.0 * d, 2.0 * d, Double.MAX_VALUE);  // ignore single-tag heading
}
```

`Double.MAX_VALUE` on theta is the idiom for "trust translation, ignore rotation"
— usually right when the gyro is far better than a single-tag yaw estimate.

### 3.5 Multi-camera pipelines

Prefer publishing each camera's pose to its own topic and letting the estimator
fuse them, over fusing on the coprocessor with `PoseFusion`. Each per-camera
sample then carries its own exact capture time and its own std-devs based on its
own tag geometry, instead of the averaged timestamp described in section 2.3.
`addVisionMeasurement` is designed for exactly this and does it optimally.

### 3.6 Sanity checks when it looks wrong

- **Pose lags during motion, correct at rest** → the timestamp is too new; you
  are hitting section 2.1. Drive a straight line at a known constant velocity
  and look at the residual between the vision pose and the odometry-only pose:
  a residual of `v × dt` with consistent sign is uncorrected latency, and
  dividing it by velocity gives `dt` directly. Apply that as a per-camera
  capture-latency constant.
- **Pose jumps ahead of reality** → too much latency is being subtracted;
  check that robot code is not also subtracting a latency topic.
- **Vision ignored entirely** → capture time is outside the estimator's buffer,
  or `update()` is not being called every loop. Log
  `Timer.getFPGATimestamp() - captureTime`; it should sit in the tens of
  milliseconds and should roughly match the WebUI latency metric plus network
  time.
- **`sample.timestamp == 0`** → no value has been received yet. Guard before use.
