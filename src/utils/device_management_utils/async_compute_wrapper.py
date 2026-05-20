from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import perf_counter
from uuid import uuid4

import numpy as np
import torch

from src.utils.device_management_utils.compute_device import ComputeDevice


@dataclass(frozen=True)
class AsyncComputeResult:
    """Result payload emitted by asynchronous compute wrappers."""

    request_id: str
    model_path: str
    stream_idx: int
    latency_s: float
    output_data: np.ndarray | None = None
    exception: BaseException | None = None
    callback_exceptions: tuple[BaseException, ...] = ()


@dataclass(frozen=True)
class _AsyncComputeRequest:
    """Queued compute request."""

    request_id: str
    model_path: str
    input_data: np.ndarray | torch.Tensor
    input_data_shape: tuple[int, int]
    stream_idx: int


class _AsyncComputeFuture:
    """Thread-safe result holder for a queued compute request."""

    def __init__(self) -> None:
        """Initialize the future."""
        self._completed = Event()
        self._result: AsyncComputeResult | None = None

    def set_result(self, result: AsyncComputeResult) -> None:
        """Store the completed result.

        Args:
            result: Completed asynchronous compute result.
        """
        self._result = result
        self._completed.set()

    def result(self, timeout_s: float | None) -> AsyncComputeResult:
        """Wait for and return the completed result.

        Args:
            timeout_s: Maximum number of seconds to wait.

        Returns:
            Completed asynchronous compute result.

        Raises:
            TimeoutError: If the result is not available before the timeout.
        """
        if not self._completed.wait(timeout_s):
            raise TimeoutError("Timed out waiting for async compute result")
        if self._result is None:
            raise RuntimeError("Async compute result completed without payload")
        return self._result


class AsyncComputeWrapper(ComputeDevice):
    """Event-driven, thread-safe wrapper around any compute device."""

    def __init__(
        self,
        delegate: ComputeDevice,
        max_pending_requests: int = 0,
        result_timeout_s: float | None = None,
    ) -> None:
        """Initialize the async wrapper.

        Args:
            delegate: Compute device that performs the actual inference.
            max_pending_requests: Maximum queued requests. Zero means unbounded.
            result_timeout_s: Optional timeout used by the synchronous run bridge.
        """
        super().__init__(
            device_id=delegate.device_id,
            device_type=delegate.device_type,
        )
        self.delegate = delegate
        self.result_timeout_s = result_timeout_s
        self._request_queue: Queue[_AsyncComputeRequest | None] = Queue(
            maxsize=max_pending_requests
        )
        self._callbacks: list[Callable[[AsyncComputeResult], None]] = []
        self._callbacks_lock = Lock()
        self._futures: dict[str, _AsyncComputeFuture] = {}
        self._futures_lock = Lock()
        self._stopped = Event()
        self._worker = Thread(
            target=self._worker_loop,
            name=f"async-compute-{self.device_id}",
            daemon=True,
        )
        self._worker.start()

    def load_model(
        self,
        model_path: str,
        input_data_shape: tuple[int, int],
        post_processing_model_path: str | None = None,
        is_grayscale: bool = False,
    ) -> None:
        """Load a model on the wrapped device.

        Args:
            model_path: Path to the model.
            input_data_shape: Shape of the input data.
            post_processing_model_path: Optional post-processing model path.
            is_grayscale: Whether the model expects grayscale input.
        """
        self.delegate.load_model(
            model_path,
            input_data_shape,
            post_processing_model_path,
            is_grayscale,
        )

    def on_frame(
        self,
        model_path: str,
        input_data: np.ndarray | torch.Tensor,
        input_data_shape: tuple[int, int],
        stream_idx: int,
        request_id: str | None = None,
    ) -> str:
        """Queue a frame for device inference without blocking on execution.

        Args:
            model_path: Path or loaded model key for the target model.
            input_data: Preprocessed model input.
            input_data_shape: Input shape expected by the device.
            stream_idx: Stream index for devices that support streams.
            request_id: Optional caller-provided request id.

        Returns:
            Request id that will be included in the result callback.

        Raises:
            RuntimeError: If the wrapper has stopped or its queue is full.
        """
        if self._stopped.is_set():
            raise RuntimeError(f"Async compute device {self.device_id} is stopped")

        request_id = request_id or uuid4().hex
        request = _AsyncComputeRequest(
            request_id=request_id,
            model_path=model_path,
            input_data=input_data,
            input_data_shape=input_data_shape,
            stream_idx=stream_idx,
        )
        future = _AsyncComputeFuture()
        with self._futures_lock:
            self._futures[request_id] = future

        try:
            self._request_queue.put_nowait(request)
        except Full as exc:
            with self._futures_lock:
                self._futures.pop(request_id, None)
            raise RuntimeError(
                f"Async compute device {self.device_id} queue is full"
            ) from exc

        return request_id

    def on_result(
        self, callback: Callable[[AsyncComputeResult], None]
    ) -> Callable[[], None]:
        """Register a callback for asynchronous inference results.

        Args:
            callback: Function invoked on the worker thread for every result.

        Returns:
            Function that unregisters the callback.
        """
        with self._callbacks_lock:
            self._callbacks.append(callback)

        def unsubscribe() -> None:
            """Remove the registered callback."""
            with self._callbacks_lock:
                if callback in self._callbacks:
                    self._callbacks.remove(callback)

        return unsubscribe

    def wait_for_result(
        self, request_id: str, timeout_s: float | None = None
    ) -> np.ndarray:
        """Wait for a queued request to complete.

        Args:
            request_id: Request id returned by on_frame.
            timeout_s: Maximum number of seconds to wait.

        Returns:
            Inference output.

        Raises:
            BaseException: Original device or callback exception.
            TimeoutError: If the request does not complete in time.
        """
        with self._futures_lock:
            future = self._futures.get(request_id)
        if future is None:
            raise KeyError(f"Unknown async compute request id: {request_id}")

        result = future.result(timeout_s)
        with self._futures_lock:
            self._futures.pop(request_id, None)

        if result.exception is not None:
            raise result.exception
        if result.callback_exceptions:
            raise RuntimeError("Async compute result callback failed") from (
                result.callback_exceptions[0]
            )
        if result.output_data is None:
            raise RuntimeError("Async compute result did not include output data")
        return result.output_data

    def run(
        self,
        model_path: str,
        input_data: np.ndarray | torch.Tensor,
        input_data_shape: tuple[int, int],
        stream_idx: int,
    ) -> np.ndarray:
        """Run inference through the async event contract and wait for output.

        Args:
            model_path: Path or loaded model key for the target model.
            input_data: Preprocessed model input.
            input_data_shape: Input shape expected by the device.
            stream_idx: Stream index for devices that support streams.

        Returns:
            Inference output.
        """
        request_id = self.on_frame(
            model_path,
            input_data,
            input_data_shape,
            stream_idx,
        )
        return self.wait_for_result(request_id, self.result_timeout_s)

    def connect_streams(self, num_streams: int) -> None:
        """Connect streams on the wrapped device.

        Args:
            num_streams: Number of streams to connect.
        """
        self.delegate.connect_streams(num_streams)

    def register_thread_access(self) -> int:
        """Register stream access on devices that expose stream indexing.

        Returns:
            Stream index returned by the wrapped device.

        Raises:
            AttributeError: If the wrapped device does not support stream access.
        """
        register_thread_access = getattr(self.delegate, "register_thread_access")
        return int(register_thread_access())

    def stop(self) -> None:
        """Stop the async worker and the wrapped device."""
        self._stopped.set()
        while True:
            try:
                self._request_queue.put_nowait(None)
                break
            except Full:
                try:
                    self._request_queue.get_nowait()
                    self._request_queue.task_done()
                except Empty:
                    continue
        self._worker.join()
        self._fail_pending_requests(
            RuntimeError(f"Async compute device {self.device_id} stopped")
        )
        self.delegate.stop()

    def _worker_loop(self) -> None:
        """Process queued compute requests."""
        while True:
            try:
                request = self._request_queue.get(timeout=0.1)
            except Empty:
                if self._stopped.is_set():
                    break
                continue

            if request is None:
                self._request_queue.task_done()
                break

            result = self._execute_request(request)
            self._complete_request(result)
            self._request_queue.task_done()

    def _execute_request(self, request: _AsyncComputeRequest) -> AsyncComputeResult:
        """Execute one queued request.

        Args:
            request: Queued request payload.

        Returns:
            Async compute result.
        """
        start_s = perf_counter()
        try:
            output_data = self.delegate.run(
                request.model_path,
                request.input_data,
                request.input_data_shape,
                request.stream_idx,
            )
            return AsyncComputeResult(
                request_id=request.request_id,
                model_path=request.model_path,
                stream_idx=request.stream_idx,
                latency_s=perf_counter() - start_s,
                output_data=output_data,
            )
        except BaseException as exc:
            return AsyncComputeResult(
                request_id=request.request_id,
                model_path=request.model_path,
                stream_idx=request.stream_idx,
                latency_s=perf_counter() - start_s,
                exception=exc,
            )

    def _complete_request(self, result: AsyncComputeResult) -> None:
        """Dispatch callbacks and complete the matching future.

        Args:
            result: Result payload to dispatch.
        """
        callback_exceptions = self._dispatch_result_callbacks(result)
        if callback_exceptions:
            result = AsyncComputeResult(
                request_id=result.request_id,
                model_path=result.model_path,
                stream_idx=result.stream_idx,
                latency_s=result.latency_s,
                output_data=result.output_data,
                exception=result.exception,
                callback_exceptions=tuple(callback_exceptions),
            )

        with self._futures_lock:
            future = self._futures.get(result.request_id)
        if future is not None:
            future.set_result(result)

    def _dispatch_result_callbacks(
        self, result: AsyncComputeResult
    ) -> list[BaseException]:
        """Invoke registered result callbacks.

        Args:
            result: Result payload to emit.

        Returns:
            Exceptions raised by callbacks.
        """
        with self._callbacks_lock:
            callbacks = tuple(self._callbacks)

        callback_exceptions: list[BaseException] = []
        for callback in callbacks:
            try:
                callback(result)
            except BaseException as exc:
                callback_exceptions.append(exc)
        return callback_exceptions

    def _fail_pending_requests(self, exception: BaseException) -> None:
        """Complete pending futures with an exception.

        Args:
            exception: Exception used for unfinished requests.
        """
        with self._futures_lock:
            futures = list(self._futures.values())
            self._futures.clear()

        for future in futures:
            future.set_result(
                AsyncComputeResult(
                    request_id="",
                    model_path="",
                    stream_idx=0,
                    latency_s=0.0,
                    exception=exception,
                )
            )
