"""Smoke tests for operation run methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from tests.utils.config_defaults import resolve_operation_defaults
from tests.utils.dummy_data import dummy_frame
from tests.utils.dummy_dependencies import (
    DummyComputePool,
    FakeCameraThreadManager,
    FakeEagleEyeInterface,
    FakeNetworkTable,
)
from tests.utils.operation_discovery import (
    build_exclusion_list,
    discover_operations,
    filter_init_params,
    is_rust_operation,
    import_operation_class,
)
from tests.utils.operation_inputs import (
    get_fallback_input,
    get_operation_input_builders,
)
from src.utils.logging.logger import Logger


def _build_dummy_dependencies() -> dict[str, Any]:
    return {
        "web_interface": FakeEagleEyeInterface(),
        "compute_pool": DummyComputePool(),
        "network_table": FakeNetworkTable(),
        "camera_manager": FakeCameraThreadManager(default_frame=dummy_frame()),
        "logger": Logger(log_directory="logs/test"),
    }


@pytest.mark.parametrize(
    "spec", discover_operations(Path(__file__).resolve().parents[1])
)
def test_operation_run(spec) -> None:
    exclusion_list = build_exclusion_list()
    if spec.action_name in exclusion_list:
        pytest.skip("yolo_excluded")

    operation_class, import_error = import_operation_class(spec)
    if operation_class is None:
        if import_error:
            if is_rust_operation(spec.action_name):
                pytest.skip(f"rust_optional: {import_error}")
            if "flask_cors" in import_error:
                pytest.fail(f"Failed to import {spec.action_name}: {import_error}")
            pytest.fail(f"Failed to import {spec.action_name}: {import_error}")
        pytest.fail(f"Missing class for {spec.action_name}")

    defaults = resolve_operation_defaults(spec)
    init_params = defaults.action_params.copy()
    dependencies = _build_dummy_dependencies()

    init_signature = getattr(operation_class.__init__, "__code__", None)
    if init_signature is not None:
        for name, value in dependencies.items():
            if name in init_signature.co_varnames:
                init_params[name] = value

    if spec.action_name == "device_input":
        dependencies["camera_manager"].add_camera(init_params.get("camera_name", "test_camera"))

    operation_class_callable = operation_class
    if operation_class_callable is None:
        pytest.fail(f"Missing class for {spec.action_name}")
    operation_class_callable = cast(type, operation_class_callable)
    init_params = filter_init_params(operation_class_callable, init_params)

    try:
        instance = operation_class_callable(**init_params)
    except TypeError as exc:
        if "takes no arguments" in str(exc):
            instance = operation_class_callable()
        else:
            raise
    except ImportError as exc:
        if is_rust_operation(spec.action_name):
            pytest.skip(f"rust_optional: {exc}")
        raise

    input_builders = get_operation_input_builders()
    input_builder = input_builders.get(spec.action_name)
    input_data = input_builder() if input_builder else get_fallback_input()

    try:
        instance.run(input_data)
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, AttributeError) and "shape" in str(exc):
            pytest.fail(
                "Run failed with missing frame shape. "
                f"Operation: {spec.action_name}. "
                f"Input type: {type(input_data)}. Error: {exc}"
            )
        if spec.action_name == "color_threshold_detection":
            pytest.fail(
                "Run failed for color_threshold_detection. "
                "Ensure camera intrinsics and cv2 are available. "
                f"Input type: {type(input_data)}. Error: {exc}"
            )
        pytest.fail(
            f"Run failed for {spec.action_name} with input {type(input_data)}: {exc}"
        )
