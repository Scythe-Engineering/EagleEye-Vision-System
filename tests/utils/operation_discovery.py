"""Operation discovery helpers for tests."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
from pathlib import Path
from typing import Any, Iterable, Mapping


MAIN_DEFINITION_SUFFIX = "Definition"


@dataclass(frozen=True)
class OperationSpec:
    """Descriptor for a discovered operation."""

    action_name: str
    module_path: str
    class_name: str
    category: str
    is_secondary: bool


def _snake_to_camel(snake_str: str) -> str:
    return "".join(word.capitalize() for word in snake_str.split("_"))


def _iter_operation_files(base_dir: Path) -> Iterable[str]:
    """Yield importable top-level operation modules, excluding package markers."""
    for path in base_dir.glob("*.py"):
        if path.name != "__init__.py":
            yield path.stem


def discover_operations(project_root: Path) -> list[OperationSpec]:
    """Discover main and secondary operations from the source tree."""

    operations: list[OperationSpec] = []
    main_dir = project_root / "src" / "main_operations" / "definitions"
    secondary_dir = project_root / "src" / "secondary_operations"

    for action_name in _iter_operation_files(main_dir):
        if action_name in {"base"}:
            continue
        class_name = f"{_snake_to_camel(action_name)}{MAIN_DEFINITION_SUFFIX}"
        module_path = f"src.main_operations.definitions.{action_name}"
        operations.append(
            OperationSpec(
                action_name=action_name,
                module_path=module_path,
                class_name=class_name,
                category="main",
                is_secondary=False,
            )
        )

    for action_name in _iter_operation_files(secondary_dir):
        if action_name in {"config_data"}:
            continue
        class_name = _snake_to_camel(action_name)
        module_path = f"src.secondary_operations.{action_name}"
        operations.append(
            OperationSpec(
                action_name=action_name,
                module_path=module_path,
                class_name=class_name,
                category="secondary",
                is_secondary=True,
            )
        )

    return sorted(operations, key=lambda spec: spec.action_name)


def import_operation_class(spec: OperationSpec) -> tuple[type | None, str | None]:
    """Import the operation class for the given spec."""

    try:
        module = importlib.import_module(spec.module_path)
        operation_class = getattr(module, spec.class_name)
        return operation_class, None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def filter_init_params(
    operation_class: type, init_params: Mapping[str, Any]
) -> dict[str, Any]:
    """Filter init params to match constructor signature."""

    try:
        signature = inspect.signature(operation_class.__init__)
    except (TypeError, ValueError):
        return init_params

    allowed = set(signature.parameters.keys())
    allowed.discard("self")
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        return init_params
    return {key: value for key, value in init_params.items() if key in allowed}


def build_exclusion_list() -> dict[str, str]:
    """Map smoke-test exclusions to their specific, user-visible reasons.

    Existing consumers use membership checks, so the mapping remains compatible
    while allowing future callers to report the reason instead of labelling every
    exclusion as YOLO-related.  Only discovered operations are included.
    """
    return {
        "object_detection": "yolo_excluded: requires YOLO model runtime",
        "mx3_async_object_detection": (
            "hardware_skip: requires an MX3 device and compiled DFP artifact"
        ),
    }


def is_rust_operation(action_name: str) -> bool:
    """Return True when an operation depends on Rust extensions."""

    return (
        "temporal_acceleration" in action_name or "pose_outlier_filter" in action_name
    )
