"""Shared asynchronous MemryX MX3 runtime and stream bindings."""

from __future__ import annotations

import importlib
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import cv2
import numpy as np

from src.main_operations.modules.object_detection.utils.letterbox import letterbox_image
from src.utils.model_library import ResolvedArtifact
from src.utils.timing import FramePacket, TimedValue

Detection = dict[str, Any]


class Mx3RuntimeError(RuntimeError):
    """Actionable MX3 configuration or runtime failure."""


class TransformedFrameSource(Protocol):
    """Dedicated timestamped frame source used by accelerator callbacks."""

    def wait_for_next_packet(
        self,
        after_frame_seq: int,
        should_continue: Callable[[], bool],
    ) -> FramePacket | None:
        """Return the newest transformed packet newer than ``after_frame_seq``."""

    def latest_frame_seq(self) -> int | None:
        """Return the newest captured frame sequence without consuming it."""


@dataclass(frozen=True, slots=True)
class Mx3Profile:
    """Validated preprocessing and decoder contract for one DFP artifact."""

    input_width: int
    input_height: int
    color_order: str
    layout: str
    normalization: str
    use_model_shape: tuple[bool, bool]
    decoder: str
    adjustable_confidence: bool
    adjustable_max_detections: bool
    max_inflight: int

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any] | None) -> "Mx3Profile":
        """Validate the initially supported MX3 YOLO profile."""
        if metadata is None:
            raise Mx3RuntimeError("MX3 model is missing profile metadata")

        def positive_int(name: str, default: int | None = None) -> int:
            """Read one positive integer profile field."""
            value = metadata.get(name, default)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise Mx3RuntimeError(f"MX3 profile {name} must be a positive integer")
            return value

        color_order = str(metadata.get("color_order", "")).lower()
        layout = str(metadata.get("layout", "")).lower()
        normalization = str(metadata.get("normalization", "")).lower()
        decoder = str(metadata.get("decoder", ""))
        if color_order not in {"rgb", "bgr"}:
            raise Mx3RuntimeError("MX3 profile color_order must be 'rgb' or 'bgr'")
        if layout not in {"nchw", "nhwc", "hwzc"}:
            raise Mx3RuntimeError(
                "MX3 profile layout must be 'nchw', 'nhwc', or 'hwzc'"
            )
        if normalization != "zero_to_one":
            raise Mx3RuntimeError(
                "Only MX3 normalization 'zero_to_one' is currently supported"
            )
        if decoder != "yolo_nms_xyxy":
            raise Mx3RuntimeError(
                "Only MX3 decoder 'yolo_nms_xyxy' is currently supported"
            )

        raw_model_shape = metadata.get("use_model_shape", [True, True])
        if isinstance(raw_model_shape, bool):
            use_model_shape = (raw_model_shape, raw_model_shape)
        elif (
            isinstance(raw_model_shape, Sequence)
            and not isinstance(raw_model_shape, (str, bytes))
            and len(raw_model_shape) == 2
            and all(isinstance(value, bool) for value in raw_model_shape)
        ):
            use_model_shape = (raw_model_shape[0], raw_model_shape[1])
        else:
            raise Mx3RuntimeError(
                "MX3 profile use_model_shape must be a bool or two booleans"
            )

        adjustable = metadata.get("adjustable_controls", {})
        if not isinstance(adjustable, Mapping):
            raise Mx3RuntimeError("MX3 profile adjustable_controls must be an object")

        return cls(
            input_width=positive_int("input_width"),
            input_height=positive_int("input_height"),
            color_order=color_order,
            layout=layout,
            normalization=normalization,
            use_model_shape=use_model_shape,
            decoder=decoder,
            adjustable_confidence=adjustable.get("confidence", False) is True,
            adjustable_max_detections=(adjustable.get("max_detections", False) is True),
            max_inflight=positive_int("max_inflight", 8),
        )

    def to_metadata(self) -> dict[str, Any]:
        """Serialize this validated profile in the manifest/runtime format."""
        return {
            "input_width": self.input_width,
            "input_height": self.input_height,
            "color_order": self.color_order,
            "layout": self.layout,
            "normalization": self.normalization,
            "use_model_shape": list(self.use_model_shape),
            "decoder": self.decoder,
            "adjustable_controls": {
                "confidence": self.adjustable_confidence,
                "max_detections": self.adjustable_max_detections,
            },
            "max_inflight": self.max_inflight,
        }


@dataclass(frozen=True, slots=True)
class _InflightPacket:
    """Source packet retained until its corresponding output arrives."""

    packet: FramePacket
    resized_size: tuple[int, int]
    padding: tuple[int, int]
    activation: int


@dataclass(frozen=True, slots=True)
class Mx3ResultPacket:
    """Inference result correlated to the exact transformed source frame."""

    frame: FramePacket
    detections: TimedValue[list[Detection]]


class Mx3StreamBinding:
    """One camera stream connected to a shared ``MxAccl`` runtime."""

    def __init__(
        self,
        stream_id: int,
        source: TransformedFrameSource,
        profile: Mx3Profile,
        class_names: tuple[str, ...] | None,
        confidence_threshold: float,
        max_detections: int,
        should_remain_active: Callable[[], bool],
    ) -> None:
        """Initialize a paused stream binding."""
        self.stream_id = stream_id
        self.source = source
        self.profile = profile
        self.class_names = class_names
        self.confidence_threshold = float(confidence_threshold)
        self.max_detections = int(max_detections)
        self.should_remain_active = should_remain_active

        self._condition = threading.Condition(threading.RLock())
        self._active = False
        self._closed = False
        self._activation = 0
        self._last_frame_seq = -1
        self._inflight: deque[_InflightPacket] = deque()
        self._pending_discards = 0
        self._completed: Mx3ResultPacket | None = None
        self._completed_generation = 0
        self._consumed_generation = 0
        self._error: Mx3RuntimeError | None = None

    @property
    def terminal_error(self) -> Mx3RuntimeError | None:
        """Return the stream's persistent runtime failure, if present."""
        with self._condition:
            return self._error

    def activate(self) -> None:
        """Resume callbacks using only frames captured after activation."""
        boundary = self._source_frame_seq()
        with self._condition:
            if self._closed:
                raise Mx3RuntimeError("MX3 stream is closed")
            if self._error is not None:
                raise self._error
            # Frames captured while paused are stale, so resume from the source's
            # current sequence instead of the one consumed before the pause.
            if boundary is not None and boundary > self._last_frame_seq:
                self._last_frame_seq = boundary
            self._active = True
            self._completed = None
            self._consumed_generation = self._completed_generation
            self._condition.notify_all()

    def deactivate(self) -> None:
        """Pause callbacks and invalidate work submitted before a later resume."""
        with self._condition:
            if self._active:
                self._active = False
                self._activation += 1
            # The accelerator still owns these submissions and delivers their
            # outputs in order, so count them as discards instead of dropping
            # the correlation entirely and mispairing a later frame.
            self._pending_discards += len(self._inflight)
            self._inflight.clear()
            self._completed = None
            self._consumed_generation = self._completed_generation
            self._condition.notify_all()

    def _source_frame_seq(self) -> int | None:
        """Return the source's newest frame sequence when it exposes one."""
        latest_frame_seq = getattr(self.source, "latest_frame_seq", None)
        if not callable(latest_frame_seq):
            return None
        try:
            frame_seq = latest_frame_seq()
        except Exception:
            return None
        return frame_seq if isinstance(frame_seq, int) else None

    def close(self) -> None:
        """Permanently stop this stream and wake every blocked callback/waiter."""
        with self._condition:
            self._closed = True
            self._active = False
            self._condition.notify_all()

    def fail(self, error: BaseException | str) -> None:
        """Persist a terminal stream error and wake blocked callers."""
        message = str(error).strip() or type(error).__name__
        with self._condition:
            if self._error is None:
                self._error = Mx3RuntimeError(
                    f"MX3 stream {self.stream_id} failed: {message}"
                )
            self._active = False
            self._condition.notify_all()

    def update_live_settings(
        self,
        confidence_threshold: float | None = None,
        max_detections: int | None = None,
    ) -> None:
        """Apply profile-supported decoder controls atomically.

        Every supplied value is validated before any is applied so a rejected
        update never leaves the decoder running a partially changed setting.
        """
        if confidence_threshold is not None:
            if not self.profile.adjustable_confidence:
                raise Mx3RuntimeError(
                    "This MX3 profile does not support confidence updates"
                )
            if not 0.0 <= confidence_threshold <= 1.0:
                raise ValueError("confidence_threshold must be between 0 and 1")
        if max_detections is not None:
            if not self.profile.adjustable_max_detections:
                raise Mx3RuntimeError(
                    "This MX3 profile does not support max_detections updates"
                )
            if max_detections < 1:
                raise ValueError("max_detections must be positive")
        with self._condition:
            if confidence_threshold is not None:
                self.confidence_threshold = float(confidence_threshold)
            if max_detections is not None:
                self.max_detections = int(max_detections)

    def _can_feed(self) -> bool:
        """Return whether the SDK input callback should keep waiting."""
        with self._condition:
            return (
                self._active
                and not self._closed
                and self._error is None
                and self.should_remain_active()
            )

    def input_callback(self, _stream_id: int) -> list[np.ndarray] | None:
        """Wait for a unique transformed camera frame and preprocess it."""
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._closed or self._error is not None or self._active
                    )
                    if self._closed or self._error is not None:
                        return None
                    if not self.should_remain_active():
                        self.deactivate()
                        continue
                    activation = self._activation
                    while (
                        len(self._inflight) + self._pending_discards
                        >= self.profile.max_inflight
                        and self._active
                        and not self._closed
                        and self._error is None
                    ):
                        self._condition.wait(timeout=0.05)
                    if self._closed or self._error is not None:
                        return None
                    if not self._active:
                        continue
                    after_frame_seq = self._last_frame_seq

                packet = self.source.wait_for_next_packet(
                    after_frame_seq, self._can_feed
                )
                if packet is None:
                    continue
                frame_seq = packet.timing.frame_seq
                if frame_seq is None:
                    raise Mx3RuntimeError(
                        "MX3 frame source returned a packet without frame_seq"
                    )
                input_array, resized_size, padding = self._preprocess(packet.value)

                with self._condition:
                    if not self._active or activation != self._activation:
                        continue
                    self._last_frame_seq = frame_seq
                    self._inflight.append(
                        _InflightPacket(packet, resized_size, padding, activation)
                    )
                return [input_array]
        except Exception as error:
            self.fail(error)
            return None
        except BaseException as error:
            self.fail(error)
            raise

    def output_callback(self, outputs: list[np.ndarray], _stream_id: int) -> None:
        """Match the oldest in-flight packet and publish the newest completion."""
        try:
            with self._condition:
                if self._closed:
                    return
                if self._pending_discards > 0:
                    # Output belongs to a packet submitted before the last pause.
                    self._pending_discards -= 1
                    self._condition.notify_all()
                    return
                if not self._inflight:
                    if not self._active:
                        return
                    raise Mx3RuntimeError(
                        "MX3 produced output without a matching source frame"
                    )
                inflight = self._inflight.popleft()
                confidence_threshold = self.confidence_threshold
                max_detections = self.max_detections
                self._condition.notify_all()

            detections = self._decode(
                outputs,
                inflight.resized_size,
                inflight.padding,
                confidence_threshold,
                max_detections,
            )
            with self._condition:
                if (
                    self._active
                    and inflight.activation == self._activation
                    and not self._closed
                ):
                    timing = inflight.packet.timing
                    self._completed = Mx3ResultPacket(
                        frame=inflight.packet,
                        detections=TimedValue(detections, timing),
                    )
                    self._completed_generation += 1
                    self._condition.notify_all()
        except Exception as error:
            self.fail(error)
        except BaseException as error:
            self.fail(error)
            raise

    def wait_for_next(self) -> Mx3ResultPacket | None:
        """Wait for a new completion, replacing skipped intermediate results."""
        with self._condition:
            while True:
                if self._error is not None:
                    raise self._error
                if self._closed or not self._active:
                    return None
                if not self.should_remain_active():
                    self.deactivate()
                    return None
                if (
                    self._completed is not None
                    and self._completed_generation > self._consumed_generation
                ):
                    self._consumed_generation = self._completed_generation
                    return self._completed
                self._condition.wait(timeout=0.05)

    def _preprocess(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Apply the profile's YOLO letterbox and tensor formatting."""
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise Mx3RuntimeError("MX3 input frame must be a color numpy array")
        letterboxed, resized_size, padding = letterbox_image(
            frame,
            (self.profile.input_width, self.profile.input_height),
            power_two_scaling=False,
            greyscale=False,
            return_resized_size_and_padding=True,
        )
        if self.profile.color_order == "rgb":
            letterboxed = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        tensor = letterboxed.astype(np.float32) / 255.0
        if self.profile.layout == "nchw":
            tensor = np.expand_dims(np.transpose(tensor, (2, 0, 1)), axis=0)
        elif self.profile.layout == "nhwc":
            tensor = np.expand_dims(tensor, axis=0)
        else:
            # MxAccl's native feature-map shape keeps OpenCV HWC ordering and
            # inserts a singleton Z dimension: HWC -> HWZC.
            tensor = np.expand_dims(tensor, axis=2)
        return tensor, resized_size, padding

    def _decode(
        self,
        outputs: list[np.ndarray],
        resized_size: tuple[int, int],
        padding: tuple[int, int],
        confidence_threshold: float,
        max_detections: int,
    ) -> list[Detection]:
        """Decode post-NMS YOLO rows and reverse per-frame letterboxing."""
        if not outputs:
            raise Mx3RuntimeError("MX3 decoder received no output feature maps")
        predictions = np.asarray(outputs[0])
        predictions = np.squeeze(predictions)
        if predictions.ndim == 1 and predictions.size == 6:
            predictions = predictions.reshape(1, 6)
        if predictions.ndim != 2 or predictions.shape[1] < 6:
            raise Mx3RuntimeError(
                "yolo_nms_xyxy expects output shaped [detections, >=6]"
            )

        pad_x, pad_y = padding
        resized_width, resized_height = resized_size
        detections: list[Detection] = []
        for row in predictions:
            if len(detections) >= max_detections:
                break
            values = np.asarray(row[:6], dtype=np.float64)
            if not np.isfinite(values).all():
                continue
            confidence = float(values[4])
            if confidence < confidence_threshold:
                continue
            x1 = float(np.clip((values[0] - pad_x) / resized_width, 0.0, 1.0))
            y1 = float(np.clip((values[1] - pad_y) / resized_height, 0.0, 1.0))
            x2 = float(np.clip((values[2] - pad_x) / resized_width, 0.0, 1.0))
            y2 = float(np.clip((values[3] - pad_y) / resized_height, 0.0, 1.0))
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(values[5])
            detection: Detection = {
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "class_id": class_id,
            }
            if self.class_names is not None and 0 <= class_id < len(self.class_names):
                detection["class_name"] = self.class_names[class_id]
            detections.append(detection)
        return detections


class _Mx3Runtime:
    """One physical MX3 loaded with one DFP and multiple explicit streams."""

    def __init__(
        self,
        physical_index: int,
        artifact: ResolvedArtifact,
        profile: Mx3Profile,
        logger: Any,
        accelerator_factory: Callable[..., Any] | None,
    ) -> None:
        self.physical_index = physical_index
        self.artifact = artifact
        self.profile = profile
        self.logger = logger
        self.accelerator_factory = accelerator_factory
        self.bindings: list[Mx3StreamBinding] = []
        self.accelerator: Any = None
        self.started = False
        self.stopping = False
        self._monitor_thread: threading.Thread | None = None
        self._lock = threading.RLock()

    def add_binding(
        self,
        source: TransformedFrameSource,
        class_names: tuple[str, ...] | None,
        confidence_threshold: float,
        max_detections: int,
        should_remain_active: Callable[[], bool],
    ) -> Mx3StreamBinding:
        """Reserve the next explicit stream before the runtime starts."""
        with self._lock:
            if self.started:
                raise Mx3RuntimeError("Cannot add an MX3 stream after runtime start")
            binding = Mx3StreamBinding(
                len(self.bindings),
                source,
                self.profile,
                class_names,
                confidence_threshold,
                max_detections,
                should_remain_active,
            )
            self.bindings.append(binding)
            return binding

    def remove_binding(self, binding: Mx3StreamBinding) -> bool:
        """Drop one reserved stream before start and renumber the remainder."""
        with self._lock:
            if self.started or binding not in self.bindings:
                return False
            self.bindings.remove(binding)
            for stream_id, remaining in enumerate(self.bindings):
                remaining.stream_id = stream_id
            return True

    def _log(self, message: str) -> None:
        """Write a runtime lifecycle message when a backend logger is available."""
        if self.logger is not None:
            self.logger.log(message)

    def _factory(self) -> Callable[..., Any]:
        """Load the tested Python-wrapped MxAccl API lazily."""
        if self.accelerator_factory is not None:
            return self.accelerator_factory
        try:
            mxapi = importlib.import_module("memryx.mxapi")
        except ImportError as error:
            raise Mx3RuntimeError(
                "MX3 requires memryx==2.2.5 from https://developer.memryx.com/pip"
            ) from error
        return mxapi.MxAccl

    def start(self) -> None:
        """Construct, connect, and start the shared MxAccl exactly once."""
        with self._lock:
            if self.started or not self.bindings:
                return
            try:
                self.accelerator = self._factory()(
                    str(self.artifact.path),
                    device_ids_to_use=[self.physical_index],
                    use_model_shape=list(self.profile.use_model_shape),
                    local_mode=True,
                )
                if self.artifact.postprocessor_path is not None:
                    self.accelerator.connect_post_model(
                        str(self.artifact.postprocessor_path), model_id=0
                    )
                for binding in self.bindings:
                    self.accelerator.connect_stream(
                        binding.input_callback,
                        binding.output_callback,
                        stream_id=binding.stream_id,
                        model_id=0,
                    )
                self.accelerator.start()
                self.started = True
                self._log(
                    f"Started mx3:{self.physical_index} with {len(self.bindings)} stream(s)"
                )
                self._monitor_thread = threading.Thread(
                    target=self._monitor,
                    daemon=True,
                    name=f"Mx3Runtime-{self.physical_index}",
                )
                self._monitor_thread.start()
            except BaseException as error:
                self._fail_all(error)
                raise Mx3RuntimeError(
                    f"Failed to start mx3:{self.physical_index}: {error}"
                ) from error

    def _monitor(self) -> None:
        """Translate unexpected SDK termination into stream errors."""
        try:
            self.accelerator.wait()
            if not self.stopping:
                self._fail_all("MemryX runtime stopped unexpectedly")
        except BaseException as error:
            if not self.stopping:
                self._fail_all(error)

    def _fail_all(self, error: BaseException | str) -> None:
        """Fail every stream sharing this runtime."""
        self._log(f"mx3:{self.physical_index} runtime failed: {error}")
        for binding in tuple(self.bindings):
            binding.fail(error)

    def stop(self) -> None:
        """Wake streams and stop the vendor runtime idempotently."""
        with self._lock:
            if self.stopping:
                return
            self.stopping = True
            for binding in self.bindings:
                binding.close()
            accelerator = self.accelerator
            monitor_thread = self._monitor_thread
        if accelerator is not None:
            accelerator.stop()
        if monitor_thread is not None:
            monitor_thread.join(timeout=5.0)
            if monitor_thread.is_alive():
                self._log(
                    f"mx3:{self.physical_index} monitor did not stop within 5 seconds"
                )


class Mx3RuntimeCoordinator:
    """Backend owner of shared per-device MX3 runtimes and stream bindings."""

    def __init__(
        self,
        logger: Any = None,
        accelerator_factory: Callable[..., Any] | None = None,
    ) -> None:
        """Initialize an inert coordinator; MemryX is imported only on start."""
        self.logger = logger
        self.accelerator_factory = accelerator_factory
        self._runtimes: dict[int, _Mx3Runtime] = {}
        self._started = False
        self._start_error: Mx3RuntimeError | None = None
        self._lock = threading.RLock()

    def register_stream(
        self,
        physical_index: int,
        artifact: ResolvedArtifact,
        source: TransformedFrameSource,
        class_names: tuple[str, ...] | None,
        confidence_threshold: float,
        max_detections: int,
        should_remain_active: Callable[[], bool],
    ) -> Mx3StreamBinding:
        """Register one stream, sharing only an identical DFP on one MX3."""
        profile = Mx3Profile.from_metadata(artifact.mx3_profile)
        with self._lock:
            if self._started:
                raise Mx3RuntimeError("Cannot register MX3 streams after startup")
            runtime = self._runtimes.get(physical_index)
            if runtime is None:
                runtime = _Mx3Runtime(
                    physical_index,
                    artifact,
                    profile,
                    self.logger,
                    self.accelerator_factory,
                )
                self._runtimes[physical_index] = runtime
            elif runtime.artifact.path.resolve() != artifact.path.resolve():
                raise Mx3RuntimeError(
                    f"mx3:{physical_index} cannot load different DFP models simultaneously"
                )
            elif runtime.profile != profile:
                raise Mx3RuntimeError(
                    f"mx3:{physical_index} received conflicting profiles for one DFP"
                )
            return runtime.add_binding(
                source,
                class_names,
                confidence_threshold,
                max_detections,
                should_remain_active,
            )

    def unregister_stream(self, binding: Mx3StreamBinding) -> None:
        """Release a stream reserved by a pipeline that failed to initialize.

        Args:
            binding: Stream previously returned by :meth:`register_stream`.

        Raises:
            Mx3RuntimeError: If the shared runtimes have already started.
        """
        with self._lock:
            if self._started:
                raise Mx3RuntimeError("Cannot unregister MX3 streams after startup")
            for physical_index, runtime in list(self._runtimes.items()):
                if runtime.remove_binding(binding):
                    if not runtime.bindings:
                        del self._runtimes[physical_index]
                    break
        binding.close()

    def start(self) -> None:
        """Start all runtimes after every pipeline has reserved its stream."""
        with self._lock:
            if self._start_error is not None:
                raise self._start_error
            if self._started:
                return
            self._started = True
            runtimes = tuple(self._runtimes.values())
        started: list[_Mx3Runtime] = []
        try:
            for runtime in runtimes:
                runtime.start()
                started.append(runtime)
        except BaseException as error:
            # Startup is all-or-nothing: leave no runtime processing frames and
            # make every later start() report the original failure instead of
            # silently returning as if the coordinator were healthy.
            with self._lock:
                self._start_error = (
                    error
                    if isinstance(error, Mx3RuntimeError)
                    else Mx3RuntimeError(f"Failed to start MX3 runtimes: {error}")
                )
            for runtime in started:
                runtime.stop()
            raise

    def close_waiters(self) -> None:
        """Wake all callback and pipeline waiters before vendor shutdown."""
        with self._lock:
            bindings = tuple(
                binding
                for runtime in self._runtimes.values()
                for binding in runtime.bindings
            )
        for binding in bindings:
            binding.close()

    def stop(self) -> None:
        """Stop all shared runtimes without automatic recreation."""
        self.close_waiters()
        with self._lock:
            runtimes = tuple(self._runtimes.values())
        for runtime in runtimes:
            runtime.stop()
