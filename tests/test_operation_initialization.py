"""Smoke tests for operation initialization."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, cast

import pytest

from tests.utils.config_defaults import resolve_operation_defaults
from tests.utils.dummy_data import dummy_frame
from tests.utils.dummy_dependencies import (
    DummyDeviceRegistry,
    DummyModelLibrary,
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
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry
from src.utils.logging.logger import Logger


def _build_dummy_dependencies() -> dict[str, Any]:
    return {
        "web_interface": FakeEagleEyeInterface(),
        "device_registry": DummyDeviceRegistry(),
        "model_library": DummyModelLibrary(),
        "network_table": FakeNetworkTable(),
        "camera_manager": FakeCameraThreadManager(default_frame=dummy_frame()),
        "camera_config_registry": CameraConfigRegistry(),
        "logger": Logger(log_directory="logs/test"),
    }


@pytest.mark.parametrize(
    "spec", discover_operations(Path(__file__).resolve().parents[1])
)
def test_operation_initialization(spec) -> None:
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

    init_parameters = inspect.signature(operation_class.__init__).parameters
    for name, value in dependencies.items():
        if name in init_parameters:
            init_params[name] = value

    if spec.action_name == "device_input":
        dependencies["camera_manager"].add_camera(init_params.get("camera_name", "test_camera"))

    if operation_class is None:
        pytest.fail(f"Missing class for {spec.action_name}")

    operation_class_callable = operation_class
    if operation_class_callable is None:
        pytest.fail(f"Missing class for {spec.action_name}")
    operation_class_callable = cast(type, operation_class_callable)

    init_params = filter_init_params(operation_class_callable, init_params)

    try:
        operation_class_callable(**init_params)
    except TypeError as exc:
        if "takes no arguments" in str(exc):
            operation_class_callable()
            return
        raise
    except ImportError as exc:
        if is_rust_operation(spec.action_name):
            pytest.skip(f"rust_optional: {exc}")
        raise
