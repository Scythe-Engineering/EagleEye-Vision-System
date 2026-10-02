"""Owned scheduler workers must drain before operation resources are closed."""

import threading
import weakref

import pytest

from src.config.utils.flow_manager import FlowManager
from src.config.utils.operation import Operation
from src.config.utils.thread_object import ThreadObject
from src.main_operations.definitions.base.base_class import OperationInstance


class _Logger:
    def log(self, _message):
        pass


class _BlockingSource(OperationInstance):
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def run(self, _value):
        self.started.set()
        assert self.release.wait(2)
        self.finished.set()
        return 42


def test_idle_worker_wakes_and_releases_obligations():
    worker = ThreadObject(1)
    operation = Operation(_BlockingSource(), "source", "source", is_data_source=True)
    operation.execution_timestep = operation.finish_timestep = 0
    worker.occupy(operation)
    reference = weakref.ref(operation)
    del operation
    worker.close()
    worker.close()
    assert not worker.processing_thread_object.is_alive()
    assert reference() is None
    assert worker.state == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        worker.set_needs_processing(None, 0)


def test_inflight_worker_timeout_preserves_resources_then_can_retry():
    source = _BlockingSource()
    operation = Operation(source, "source", "source", is_data_source=True)
    operation.execution_timestep = operation.finish_timestep = 0
    worker = ThreadObject(1)
    worker.occupy(operation)
    worker.set_needs_processing(None, 0)
    assert source.started.wait(1)
    try:
        with pytest.raises(TimeoutError, match="retry close"):
            worker.close(0.01)
        assert worker.processing_thread_object.is_alive()
        assert worker.operation_obligations == [operation]
        worker.reset_state()  # Cannot falsely mark inflight work idle.
        assert worker.state == "processing"
    finally:
        source.release.set()
        worker.close()
    assert source.finished.is_set()
    assert not worker.processing_thread_object.is_alive()
    assert worker.input_data is worker.output_data is None


@pytest.mark.parametrize("source_count", [1, 2])
def test_flow_close_drains_direct_and_threaded_cycles(source_count):
    sources = [_BlockingSource() for _ in range(source_count)]
    operations = {
        str(index): Operation(source, str(index), "source", is_data_source=True)
        for index, source in enumerate(sources)
    }
    flow = FlowManager(operations, _Logger())
    runner = threading.Thread(target=flow.run_flow)
    runner.start()
    assert all(source.started.wait(1) for source in sources)
    try:
        with pytest.raises(TimeoutError, match="retry close"):
            flow.close(0.01)
        assert all(
            worker.processing_thread_object.is_alive() for worker in flow.thread_objects
        )
        assert flow.operations == operations
    finally:
        for source in sources:
            source.release.set()
        runner.join(2)
        flow.close()
    assert not runner.is_alive()
    assert all(source.finished.is_set() for source in sources)
    assert all(
        not worker.processing_thread_object.is_alive() for worker in flow.thread_objects
    )
    assert flow.operation_outputs == flow.previous_operation_outputs == {}
    flow.close()
    with pytest.raises(RuntimeError, match="closed"):
        flow.run_flow()


@pytest.mark.parametrize("threaded", [False, True])
def test_pipeline_close_finishes_work_before_closing_collaborators(threaded):
    """Exercise the public owner, including its continuously running mode."""
    import numpy as np

    from benchmarks.replay import ReplayCameraManager, build_pipeline

    manager = ReplayCameraManager()
    calibration = {
        "camera_matrix": [[100, 0, 32], [0, 100, 32], [0, 0, 1]],
        "distortion_coefficients": [0] * 5,
    }
    lifecycle = build_pipeline(False, calibration, manager=manager)
    pipeline = lifecycle.pipeline
    source = pipeline.get_operation_by_uuid("bench-input").instance
    original_run = source.run
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    resource_closed = threading.Event()

    def blocking_run(value):
        started.set()
        assert release.wait(2)
        result = original_run(value)
        finished.set()
        return result

    def close_source():
        assert finished.is_set()
        resource_closed.set()

    source.run = blocking_run
    source.close = close_source
    manager.publish(np.zeros((64, 64, 3), np.uint8), 0, 0)
    runner = None
    if threaded:
        pipeline.thread_run(manager)
    else:
        runner = threading.Thread(target=pipeline.run)
        runner.start()
    assert started.wait(1)
    errors = []

    def close_pipeline():
        try:
            pipeline.close()
        except Exception as error:  # noqa: BLE001 - surface thread failure in test
            errors.append(error)

    closer = threading.Thread(target=close_pipeline)
    closer.start()
    try:
        assert not resource_closed.wait(0.02)
    finally:
        release.set()
        closer.join(3)
        if runner is not None:
            runner.join(3)
        lifecycle.close()
    assert not closer.is_alive()
    assert not errors
    assert resource_closed.is_set()
    assert all(
        not worker.processing_thread_object.is_alive()
        for worker in pipeline.flow_manager.thread_objects
    )


@pytest.mark.parametrize("temporal", [False, True])
def test_repeated_pipeline_close_releases_workers_and_native_detectors(
    temporal, tmp_path
):
    """Release real production graphs without depending on the 2D benchmark tools."""
    import gc
    import json

    import numpy as np

    from benchmarks.replay import CONFIG_DIR, ReplayCameraManager, build_pipeline

    calibration = {
        "camera_matrix": [[100, 0, 32], [0, 100, 32], [0, 0, 1]],
        "distortion_coefficients": [0] * 5,
    }
    config_name = "temporal.json" if temporal else "full_frame.json"
    graph = json.loads((CONFIG_DIR / config_name).read_text(encoding="utf-8"))
    for node in graph:
        if node["action_name"] == "detect_apriltags.py":
            # Repeated native multithreaded detection has a separate upstream fault.
            node["action_params"]["full_frame_nthreads"] = 1
    config_path = tmp_path / config_name
    config_path.write_text(json.dumps(graph), encoding="utf-8")
    baseline = set(threading.enumerate())
    for _ in range(3):
        manager = ReplayCameraManager()
        lifecycle = build_pipeline(config_path, calibration, manager=manager)
        with lifecycle as pipeline:
            detector = pipeline.get_operation_by_uuid("bench-detect").instance.detector
            references = [
                weakref.ref(item) for item in (pipeline, detector, detector.detector)
            ]
            workers = [
                worker.processing_thread_object
                for worker in pipeline.flow_manager.thread_objects
            ]
            manager.publish(np.zeros((64, 64, 3), np.uint8), 0, 0)
            pipeline.run()
            assert pipeline.get_operation_errors() == []
        lifecycle.close()
        assert all(not worker.is_alive() for worker in workers)
        with pytest.raises(RuntimeError, match="closed"):
            pipeline.run()
        del pipeline, detector
        gc.collect()
        assert all(reference() is None for reference in references)
        assert set(threading.enumerate()) == baseline


def test_owned_native_cleanup_order_and_partial_construction():
    """The detector owns decoding userdata, while the wrapper owns families."""
    import ctypes
    from types import SimpleNamespace

    from src.main_operations.modules.apriltags.native_detector import Detector

    calls = []

    class Destroy:
        def __init__(self, name):
            self.name = name

        def __call__(self, pointer):
            assert self.restype is None
            assert self.argtypes == [type(pointer)]
            calls.append(self.name)

    detector = Detector.__new__(Detector)
    detector._lifecycle_lock = threading.Lock()
    detector.libc = SimpleNamespace(
        apriltag_detector_destroy=Destroy("detector"),
        tag36h11_destroy=Destroy("family"),
    )
    detector.tag_detector_ptr = ctypes.pointer(ctypes.c_int(1))
    detector.tag_families = {"tag36h11": ctypes.pointer(ctypes.c_int(2))}
    detector.close()
    detector.close()
    assert calls == ["detector", "family"]
    assert detector.tag_detector_ptr is None
    assert detector.tag_families == {}
    partial = Detector.__new__(Detector)
    partial._lifecycle_lock = threading.Lock()
    partial.close()


def test_real_owned_native_detection_close_and_gc_subprocess():
    """Check critical native lifetimes in a child, not giant-family table stress."""
    import os
    import subprocess
    import sys

    code = """
import gc, weakref
import numpy as np
from src.main_operations.modules.apriltags.native_detector import Detector
from src.main_operations.modules.apriltags.apriltag_detector import AprilTagDetector
# Giant stock families allocate multi-GB lookup tables, unrelated to ownership.
families = ('tag16h5', 'tag36h11', 'tag16h5 tag36h11')
for family in families:
    print('native family', family, flush=True)
    # Test ownership at the 1-thread target preset, not upstream multithread stress.
    detector = Detector(families=family, quad_decimate=1, nthreads=1)
    assert detector.detect(np.zeros((32, 32), np.uint8)) == []
    detector.close()
    detector.close()
    assert detector.tag_detector_ptr is None and not detector.tag_families
    try:
        detector.detect(np.zeros((32, 32), np.uint8))
    except RuntimeError:
        pass
    else:
        raise AssertionError('closed native detector accepted detection')
    del detector
for family in families:
    detector = Detector(families=family, nthreads=1)
    detector.detect(np.zeros((32, 32), np.uint8))
    reference = weakref.ref(detector)
    del detector
    gc.collect()
    assert reference() is None
for kwargs in ({'families': 'unknown'}, {'searchpath': ()}, {'nthreads': object()}):
    try:
        Detector(**{'nthreads': 1, **kwargs})
    except Exception:
        pass
    else:
        raise AssertionError('expected constructor failure')
    gc.collect()
wrapper = AprilTagDetector(full_frame_nthreads=1)
assert wrapper.detect(np.zeros((64, 64), np.uint8)) == []
wrapper.update_parameters(nthreads=1)
wrapper.close()
wrapper.close()
assert wrapper.detect(np.zeros((64, 64), np.uint8)) is None
try:
    wrapper.update_parameters(nthreads=1)
except RuntimeError:
    pass
else:
    raise AssertionError('closed wrapper reopened')
print('native cleanup passed', flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        env={**os.environ, "PYTHONFAULTHANDLER": "1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Exception ignored" not in result.stderr, result.stderr
    assert "native cleanup passed" in result.stdout


@pytest.mark.parametrize("source_count", [1, 2])
def test_flow_close_waits_for_runner_callbacks_after_operation_finishes(source_count):
    """Idle workers do not imply the caller has finished reading the graph."""
    from types import SimpleNamespace

    callback_started = threading.Event()
    callback_release = threading.Event()

    def on_success(_operation):
        callback_started.set()
        assert callback_release.wait(2)

    operations = {
        str(index): Operation(
            SimpleNamespace(run=lambda _value: 42),
            str(index),
            "source",
            is_data_source=True,
        )
        for index in range(source_count)
    }
    flow = FlowManager(operations, _Logger(), on_operation_success=on_success)
    runner = threading.Thread(target=flow.run_flow)
    runner.start()
    assert callback_started.wait(1)
    try:
        with pytest.raises(TimeoutError, match="retry close"):
            flow.close(0.01)
        assert flow.operations is operations
        assert flow.on_operation_success is on_success
        assert all(
            worker.processing_thread_object.is_alive() for worker in flow.thread_objects
        )
    finally:
        callback_release.set()
        runner.join(2)
        flow.close()
    assert not runner.is_alive()
    assert flow.on_operation_success is None
