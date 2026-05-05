from __future__ import annotations

import importlib
import json
import os
import threading
import time
import traceback
from collections import deque
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
from line_profiler import profile
import ntcore

from src.config.utils.cyclical_loop_detection import detect_connection_cycles
from src.config.utils.flow_manager import FlowManager
from src.config.utils.operation import Connection, Operation
from src.main_operations.definitions.base.base_class import OperationInstance
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry
from src.utils.colors import Colors
from src.utils.device_management_utils.compute_pool import ComputePool
from src.utils.logging.logger import Logger

if TYPE_CHECKING:
    from src.utils.camera_utils.camera_thread_manager import CameraThreadManager
    from src.webui.web_server import EagleEyeInterface

debug_mode = False
NT_COMMAND_PREFIX = "commands"
NT_ACTIVE_COMMAND = "active"


class Pipeline:
    """Pipeline for processing data through a sequence of operations."""

    def __init__(
        self,
        pipeline_config: list[dict[str, Any]],
        web_interface: EagleEyeInterface,
        compute_pool: ComputePool,
        network_table: ntcore.NetworkTable,
        logger: Logger,
        camera_manager: CameraThreadManager | None = None,
        camera_config_registry: CameraConfigRegistry | None = None,
        camera_bus_ids: list[str] | None = None,
        pipeline_name: str | None = None,
    ) -> None:
        """Initialize the pipeline with configuration.

        Args:
            pipeline_config: List containing pipeline configuration.
            web_interface: The web interface to use for the pipelines.
            compute_pool: The compute pool to use for the pipelines.
            network_table: The network table to use for the pipelines.
            logger: Logger instance for logging.
            camera_manager: The camera manager to use for the pipelines.
            camera_config_registry: Shared camera config registry for
                camera intrinsics/extrinsics access.
            camera_bus_ids: USB bus IDs referenced by device_input operations.
        """
        self.pipeline_config = pipeline_config
        self.web_interface = web_interface
        self.compute_pool = compute_pool
        self.network_table = network_table
        self.camera_manager = camera_manager
        self.camera_config_registry = camera_config_registry
        self.logger = logger
        self.camera_bus_ids = list(camera_bus_ids) if camera_bus_ids else []
        self.pipeline_name = pipeline_name or "unknown"

        self.thread_running = False
        self.thread_active = False
        self.thread = None
        self.thread_state_lock = threading.Lock()
        self.operation_errors: dict[str, dict[str, Any]] = {}
        self.operation_errors_lock = threading.Lock()
        self.operations: dict[str, Operation] = self._initialize_operations()

        if not self.operations:
            raise ValueError("No operations configured in pipeline")

        self.flow_manager = FlowManager(
            self.operations,
            self.logger,
            on_operation_error=self.record_operation_error,
            on_operation_success=self.clear_operation_error,
            pipeline_name=self.pipeline_name,
        )

        self.total_time_history: deque[float] = deque(maxlen=50)
        self.total_time_history_lock = threading.Lock()

        self.set_visualize = False
        self.visualization_data = None
        self.visualization_data_lock = threading.Lock()
        self.visualization_operation_uuid = None
        self._last_nt_active_state = True

    def _snake_to_camel(self, snake_str: str) -> str:
        """Convert snake_case string to CamelCase.

        Args:
            snake_str: String in snake_case format.

        Returns:
            String in CamelCase format.
        """
        components = snake_str.split("_")
        return "".join(word.capitalize() for word in components)

    def _load_config_def(self, action_name: str) -> dict[str, Any] | None:
        """Load the config_def.json file for an operation.

        Args:
            action_name: Name of the action (without .py extension).

        Returns:
            Config definition dictionary, or None if not found.
        """
        # Try secondary operations first
        config_path = os.path.join(
            "src",
            "secondary_operations",
            "config_data",
            f"{action_name}_config_def.json",
        )
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)

        # Try main operations
        config_path = os.path.join(
            "src",
            "main_operations",
            "definitions",
            "config_data",
            f"{action_name}_config_def.json",
        )
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)

        return None

    def _initialize_operations(self) -> dict[str, Operation]:
        """Initialize operation instances based on configuration.

        Returns:
            Dict of initialized operation instances with uuids as keys.
        """
        operations: dict[str, Operation] = {}
        all_connections_unprocessed: list[dict[str, str]] = []

        # Gather all connections and operations from config
        for operation_config in self.pipeline_config:
            action_name = operation_config["action_name"].removesuffix(".py")
            action_params = operation_config.get("action_params", {})
            action_uuid = operation_config.get("uuid", None)

            for connection in operation_config.get("connections", []):
                all_connections_unprocessed.append(connection)

            if not action_uuid:
                raise ValueError(
                    Colors.YELLOW
                    + f"Error: Operation {action_name} is missing UUID in configuration. Cannot create pipeline"
                    + Colors.RESET
                )

            try:
                operation_instance = self._create_operation_instance(
                    action_name, action_params
                )
            except Exception:
                self.record_operation_init_error(
                    action_uuid,
                    action_name,
                    traceback.format_exc(),
                )
                raise

            # Load config_def to check if operation is a data source
            config_def = self._load_config_def(action_name)
            is_data_source = (
                config_def.get("is_data_source", False) if config_def else False
            )

            operations[action_uuid]: Operation = Operation(
                instance=operation_instance,
                uuid=action_uuid,
                name=action_name,
                is_data_source=is_data_source,
            )

        # link all connections once all operations are created
        for connection in all_connections_unprocessed:
            try:
                from_operation = operations[connection["from_uuid"]]
            except KeyError:
                raise ValueError(
                    f"Connection issue: operation {connection['from_uuid']} not found"
                )
            try:
                to_operation = operations[connection["to_uuid"]]
            except KeyError:
                raise ValueError(
                    f"Connection issue: operation {connection['to_uuid']} not found"
                )

            try:
                # connection will register itself with operations when it is created
                Connection(
                    from_operation=from_operation,
                    from_port=connection["from_port"],
                    to_operation=to_operation,
                    to_port=connection["to_port"],
                    data_type=connection["data_type"],
                    is_default=bool(connection.get("is_default", False)),
                )
            except KeyError as e:
                raise ValueError(
                    f"Malformed connection data: missing key {e} in connection {connection}"
                )

        self._remove_unreachable_operation_islands(operations)

        detect_connection_cycles(operations)

        return operations

    def _remove_unreachable_operation_islands(
        self, operations: dict[str, Operation]
    ) -> None:
        """Remove runtime-only operation islands disconnected from data sources.

        Args:
            operations: Mutable operation graph keyed by operation UUID.
        """
        reachable_uuids = self._find_data_source_reachable_operation_uuids(operations)
        unreachable_uuids = set(operations.keys()) - reachable_uuids

        if not unreachable_uuids:
            return

        island_groups = self._group_unreachable_operation_islands(
            operations, unreachable_uuids
        )
        formatted_groups = [
            [
                f"{operations[uuid].name}:{uuid}"
                for uuid in sorted(group, key=lambda value: operations[value].name)
            ]
            for group in island_groups
        ]
        self.logger.log(
            Colors.YELLOW
            + f"WARNING: Pipeline {self.pipeline_name} has operation islands disconnected "
            + f"from data sources. These operations will not run: {formatted_groups}"
            + Colors.RESET
        )

        for operation in operations.values():
            operation.input_connections = [
                connection
                for connection in operation.input_connections
                if connection.from_operation.uuid not in unreachable_uuids
                and connection.to_operation.uuid not in unreachable_uuids
            ]
            operation.output_connections = [
                connection
                for connection in operation.output_connections
                if connection.from_operation.uuid not in unreachable_uuids
                and connection.to_operation.uuid not in unreachable_uuids
            ]
            operation.has_input_connections = len(operation.input_connections) > 0
            operation.has_output_connections = len(operation.output_connections) > 0

        for uuid in unreachable_uuids:
            operations.pop(uuid, None)

    def _find_data_source_reachable_operation_uuids(
        self, operations: dict[str, Operation]
    ) -> set[str]:
        """Find operations reachable from any data source operation.

        Args:
            operations: Operation graph keyed by operation UUID.

        Returns:
            Set of operation UUIDs reachable through output connections.
        """
        roots = [
            operation for operation in operations.values() if operation.is_data_source
        ]
        reachable_uuids: set[str] = set()
        queue: deque[Operation] = deque(roots)

        while queue:
            operation = queue.popleft()
            if operation.uuid in reachable_uuids:
                continue

            reachable_uuids.add(operation.uuid)
            for connection in operation.output_connections:
                if connection.to_operation.uuid not in reachable_uuids:
                    queue.append(connection.to_operation)

        return reachable_uuids

    def _group_unreachable_operation_islands(
        self, operations: dict[str, Operation], unreachable_uuids: set[str]
    ) -> list[list[str]]:
        """Group unreachable operations into undirected connected components.

        Args:
            operations: Operation graph keyed by operation UUID.
            unreachable_uuids: UUIDs not reachable from any data source.

        Returns:
            List of island groups, each containing operation UUIDs.
        """
        adjacency: dict[str, set[str]] = {uuid: set() for uuid in unreachable_uuids}
        for uuid in unreachable_uuids:
            operation = operations[uuid]
            for connection in operation.input_connections:
                neighbor_uuid = connection.from_operation.uuid
                if neighbor_uuid in unreachable_uuids:
                    adjacency[uuid].add(neighbor_uuid)
                    adjacency[neighbor_uuid].add(uuid)
            for connection in operation.output_connections:
                neighbor_uuid = connection.to_operation.uuid
                if neighbor_uuid in unreachable_uuids:
                    adjacency[uuid].add(neighbor_uuid)
                    adjacency[neighbor_uuid].add(uuid)

        groups: list[list[str]] = []
        visited: set[str] = set()
        for uuid in sorted(unreachable_uuids):
            if uuid in visited:
                continue
            group: list[str] = []
            queue: deque[str] = deque([uuid])
            visited.add(uuid)
            while queue:
                current_uuid = queue.popleft()
                group.append(current_uuid)
                for next_uuid in sorted(adjacency[current_uuid]):
                    if next_uuid in visited:
                        continue
                    visited.add(next_uuid)
                    queue.append(next_uuid)
            groups.append(group)

        return groups

    def record_operation_init_error(
        self, operation_uuid: str, operation_name: str, message: str
    ) -> None:
        """Record an operation error entry during initialization.

        Args:
            operation_uuid: UUID of the operation that failed.
            operation_name: Name of the operation that failed.
            message: The error message or traceback string.
        """
        trimmed_message = message.strip() if message else ""
        if not operation_uuid or not trimmed_message:
            return
        with self.operation_errors_lock:
            record = self.operation_errors.get(operation_uuid)
            if record is None:
                self.operation_errors[operation_uuid] = {
                    "uuid": operation_uuid,
                    "name": operation_name,
                    "message": trimmed_message,
                    "last_seen_ts": time.time(),
                    "count": 1,
                }
            else:
                record["message"] = trimmed_message
                record["last_seen_ts"] = time.time()
                record["count"] = int(record.get("count", 0)) + 1

        self._publish_operation_error_snapshot()

    def _create_operation_instance(
        self, action_name: str, action_params: Dict[str, Any]
    ) -> OperationInstance:
        """Create an operation instance based on action name and parameters.

        Args:
            action_name: Name of the action to create.
            action_params: Parameters for the action.

        Returns:
            Initialized operation instance.

        Raises:
            ValueError: If action_name is not recognized or module cannot be imported.
        """
        try:
            action_name = action_name.removesuffix(".py")
            class_name = self._snake_to_camel(action_name)

            # Try to import from main_operations/definitions first
            try:
                module_path = f"src.main_operations.definitions.{action_name}"
                module = importlib.import_module(module_path, package=__name__)
                # For main operations, add "Definition" suffix
                full_class_name = f"{class_name}Definition"
                operation_class = getattr(module, full_class_name)
            except (ImportError, AttributeError):
                # Try to import from secondary_operations
                try:
                    module_path = f"src.secondary_operations.{action_name}"
                    module = importlib.import_module(module_path, package=__name__)
                    # For secondary operations, use class name as-is
                    operation_class = getattr(module, class_name)
                except (ImportError, AttributeError):
                    raise ValueError(
                        f"Could not find class for action: {class_name} at {action_name} with path {module_path}"
                    )

            # Create a shallow copy to avoid mutating the original action_params
            init_params = action_params.copy()

            if (
                hasattr(operation_class.__init__, "__code__")
                and "web_interface" in operation_class.__init__.__code__.co_varnames
            ):
                init_params["web_interface"] = self.web_interface

            if (
                hasattr(operation_class.__init__, "__code__")
                and "compute_pool" in operation_class.__init__.__code__.co_varnames
            ):
                init_params["compute_pool"] = self.compute_pool

            if (
                hasattr(operation_class.__init__, "__code__")
                and "pipeline" in operation_class.__init__.__code__.co_varnames
            ):
                init_params["pipeline"] = self

            if (
                hasattr(operation_class.__init__, "__code__")
                and "network_table" in operation_class.__init__.__code__.co_varnames
            ):
                init_params["network_table"] = self.network_table

            if (
                hasattr(operation_class.__init__, "__code__")
                and "camera_manager" in operation_class.__init__.__code__.co_varnames
            ):
                init_params["camera_manager"] = self.camera_manager

            if (
                hasattr(operation_class.__init__, "__code__")
                and "camera_config_registry"
                in operation_class.__init__.__code__.co_varnames
            ):
                init_params["camera_config_registry"] = self.camera_config_registry

            if (
                hasattr(operation_class.__init__, "__code__")
                and "camera_configs" in operation_class.__init__.__code__.co_varnames
            ):
                init_params["camera_configs"] = (
                    self.camera_config_registry.get_all_configs()
                    if self.camera_config_registry is not None
                    else {}
                )

            if (
                hasattr(operation_class.__init__, "__code__")
                and "logger" in operation_class.__init__.__code__.co_varnames
            ):
                init_params["logger"] = self.logger

            # check if operation class is a subclass of OperationInstance
            if not issubclass(operation_class, OperationInstance):
                raise ValueError(
                    f"Operation {action_name} is not a subclass of OperationInstance"
                )

            return operation_class(**init_params)

        except TypeError as e:
            raise ValueError(f"Invalid parameters for {action_name}: {str(e)}")

    @profile
    def run(
        self,
        visualize: bool = False,
        visualization_operation_uuid: str | None = None,
    ) -> np.ndarray | None:
        """Run the pipeline using FlowManager.

        Args:
            visualize: Whether to visualize the pipeline.
            visualization_operation_uuid: UUID of operation to visualize up to.

        Returns:
            If visualize is True, returns the visualized frame.
            Otherwise, returns None.

        Raises:
            ValueError: If visualization_operation_uuid is required but not provided.
        """
        start_time = time.time()

        self.flow_manager.run_flow()

        elapsed = time.time() - start_time
        with self.total_time_history_lock:
            self.total_time_history.append(elapsed)

        if visualize:
            if visualization_operation_uuid is None:
                raise ValueError(
                    "Visualization operation UUID is required when visualize is True"
                )
            # Get the frame from device_input for visualization
            start_frame = self._get_device_input_frame(visualization_operation_uuid)
            if start_frame is not None:
                return self._visualize(
                    start_frame.copy(),
                    visualization_operation_uuid,
                )

        return None

    def _find_upstream_device_input(
        self, operation_uuid: str | None
    ) -> Operation | None:
        """Find the nearest upstream device_input operation for a target operation.

        Args:
            operation_uuid: Target operation UUID.

        Returns:
            Upstream device_input operation if found, otherwise None.
        """
        if operation_uuid is None:
            return None
        start_operation = self.operations.get(operation_uuid)
        if start_operation is None:
            return None
        visited: set[str] = set()
        queue: deque[Operation] = deque([start_operation])

        while queue:
            operation = queue.popleft()
            if operation.uuid in visited:
                continue
            visited.add(operation.uuid)

            if operation.name == "device_input":
                return operation

            for connection in operation.get_input_connections():
                queue.append(connection.from_operation)

        return None

    def _get_device_input_frame(
        self, target_operation_uuid: str | None
    ) -> np.ndarray | None:
        """Get the current frame from device_input operation.

        Returns:
            The frame from device_input, or None if not found or no frame available.
        """
        target_device_input = self._find_upstream_device_input(target_operation_uuid)
        if target_device_input is not None:
            return self.flow_manager.operation_outputs.get(target_device_input.uuid)

        for operation in self.operations.values():
            if operation.name == "device_input":
                return self.flow_manager.operation_outputs.get(operation.uuid)
        return None

    def get_operation_by_uuid(self, operation_uuid: str) -> Operation | None:
        """Get an operation by its UUID.

        Args:
            operation_uuid: The UUID of the operation.

        Returns:
            The operation with the given UUID, or None if not found.
        """
        return self.operations.get(operation_uuid)

    def update_operations_config(self, operations_config: list[Dict[str, Any]]) -> str:
        """Update the configuration of multiple operations in the pipeline.

        This method allows live updating of operation parameters that are marked as
        restart_for_change: false in their configuration definition files.

        Args:
            operations_config: list of operation configurations, where each config
                is a dictionary with 'uuid', 'action_name', and 'action_params' keys.
                Format should match the pipeline configuration JSON format.

        Returns:
            "applied" if all live updates succeeded, "unsupported" if one or more
            operations cannot update live, and "failed" if an update errored or an
            operation was missing.
        """
        status = "applied"

        for operation_config in operations_config:
            action_uuid = operation_config.get("uuid")
            action_name = operation_config.get("action_name")
            action_params = operation_config.get("action_params", {})

            if not action_uuid:
                continue

            # Find the operation instance by UUID
            operation_wrapper = self.get_operation_by_uuid(action_uuid)

            if operation_wrapper is not None:
                operation = operation_wrapper.instance
                if hasattr(operation, "update_config"):
                    try:
                        operation.update_config(action_params)
                        if debug_mode and self.logger:
                            self.logger.log(
                                f"{Colors.GREEN}Updated config for {action_name} ({action_uuid}): {action_params}{Colors.RESET}"
                            )
                    except Exception as e:
                        status = "failed"
                        if self.logger:
                            self.logger.log(
                                f"{Colors.RED}Error updating config for {action_name} ({action_uuid}): {e}{Colors.RESET}"
                            )
                else:
                    if status != "failed":
                        status = "unsupported"
                    if debug_mode and self.logger:
                        self.logger.log(
                            f"{Colors.YELLOW}Operation {action_name} ({action_uuid}) does not support config updates{Colors.RESET}"
                        )
            else:
                status = "failed"
                if debug_mode and self.logger:
                    self.logger.log(
                        f"{Colors.RED}Operation {action_name} ({action_uuid}) not found in pipeline{Colors.RESET}"
                    )

        return status

    def thread_run(self, camera_thread_manager: CameraThreadManager) -> None:
        """Run the pipeline continuously in a thread.

        Args:
            camera_thread_manager: The camera thread manager.
        """
        with self.thread_state_lock:
            self.thread_running = True
            self.thread_active = False
            self.thread = threading.Thread(
                target=self._thread_run, args=(camera_thread_manager,)
            )
            self.thread.start()

    def _thread_run(self, camera_thread_manager: CameraThreadManager) -> None:
        """Run the pipeline continuously in a thread.

        Args:
            camera_thread_manager: The camera thread manager.
        """
        if self.logger:
            self.logger.log(f"{Colors.CYAN}Starting pipeline thread{Colors.RESET}")
        time.sleep(0.1)

        while self.thread_running:
            try:
                if not self._is_enabled_from_networktables():
                    with self.thread_state_lock:
                        self.thread_active = False
                    time.sleep(0.05)
                    continue

                with self.thread_state_lock:
                    self.thread_active = True

                # Snapshot visualize state and target name atomically
                with self.visualization_data_lock:
                    should_visualize = self.set_visualize
                    operation_uuid_snapshot = self.visualization_operation_uuid

                if should_visualize:
                    visualization_frame = self.run(
                        visualize=True,
                        visualization_operation_uuid=operation_uuid_snapshot,
                    )
                    # Get the original frame from device_input for display
                    frame = self._get_device_input_frame(operation_uuid_snapshot)
                    if frame is not None and visualization_frame is not None:
                        # Only hold the lock for the assignment
                        with self.visualization_data_lock:
                            self.visualization_data = {
                                "frame": frame.copy(),
                                "visualization_data": visualization_frame,
                            }
                else:
                    self.run()
            except Exception as _:
                with self.thread_state_lock:
                    self.thread_active = False
                if self.logger:
                    self.logger.log(
                        f"{Colors.RED}Error in pipeline itself: {traceback.format_exc()}{Colors.RESET}"
                    )

            time.sleep(0.001)

        with self.thread_state_lock:
            self.thread_active = False

    def _is_enabled_from_networktables(self) -> bool:
        """Return whether this pipeline should actively process frames.

        Returns:
            bool: True when the command topic is enabled or unavailable.
        """
        command_topic = f"{NT_COMMAND_PREFIX}/{self.pipeline_name}/{NT_ACTIVE_COMMAND}"
        try:
            active_value = bool(
                self.network_table.getEntry(command_topic).getBoolean(True)
            )
        except Exception:
            return True

        if active_value != self._last_nt_active_state and self.logger:
            self.logger.log(
                f"{Colors.CYAN}Pipeline {self.pipeline_name} active command set to {active_value}{Colors.RESET}"
            )
            self._last_nt_active_state = active_value

        return active_value

    def _visualize(
        self, start_frame: np.ndarray, action_uuid: str | None
    ) -> np.ndarray | None:
        """Visualize a single operation using the provided frame.

        Args:
            action_uuid: The UUID of the action to visualize.

        Returns:
            The visualized frame, or None if no visualization is available.
        """
        if action_uuid is None:
            return None

        operation = self.operations.get(action_uuid)
        if operation is None:
            return None

        if not hasattr(operation.instance, "visualize"):
            return None

        return operation.instance.visualize(start_frame)

    def start_visualize(self, visualization_operation_uuid: str) -> None:
        """Start visualizing the pipeline."""
        # Ensure operation UUID is set before enabling visualization
        with self.visualization_data_lock:
            self.visualization_operation_uuid = visualization_operation_uuid
            self.set_visualize = True

    def stop_visualize(self) -> None:
        """Stop visualizing the pipeline."""
        with self.visualization_data_lock:
            self.set_visualize = False

    def stop(self) -> None:
        """Stop the pipeline thread."""
        with self.thread_state_lock:
            self.thread_running = False
            self.thread_active = False
            thread = self.thread
        if thread is not None:
            thread.join()
            with self.thread_state_lock:
                if self.thread is thread:
                    self.thread = None

    def is_active(self) -> bool:
        """Return whether the pipeline processing thread is currently active.

        Returns:
            bool: True when the pipeline thread is alive and actively processing.
        """
        with self.thread_state_lock:
            thread = self.thread
            thread_active = self.thread_active

        if not thread_active:
            return False
        if thread is None:
            return False
        try:
            return bool(thread.is_alive())
        except Exception:
            return False

    def get_pipeline_thread_info(self) -> dict[str, Any]:
        """Get total number of threads, thread assignment, and execution timestep for each operation.

        Returns:
            dict[str, Any]: Dictionary containing:
                - 'total_threads': Total number of threads in the pipeline
                - 'operations': Dictionary mapping operation UUID to dict with 'thread' and 'timestep'
        """
        thread_info = self.flow_manager.get_thread_and_timestep_info()

        return {
            "total_threads": self.flow_manager.num_threads,
            "operations": thread_info,
        }

    def get_latest_profile_snapshot(self) -> dict[str, Any] | None:
        """Get the latest per-frame profiling snapshot.

        Returns:
            Latest profiling payload or None when unavailable.
        """
        return self.flow_manager.get_latest_profile_snapshot()

    def record_operation_error(self, operation: Operation, message: str) -> None:
        """Record an operation error entry.

        Args:
            operation: The operation that failed.
            message: The error message or traceback string.
        """
        if operation is None:
            return
        trimmed_message = message.strip() if message else ""
        if not trimmed_message:
            return
        with self.operation_errors_lock:
            record = self.operation_errors.get(operation.uuid)
            if record is None:
                self.operation_errors[operation.uuid] = {
                    "uuid": operation.uuid,
                    "name": operation.name,
                    "message": trimmed_message,
                    "last_seen_ts": time.time(),
                    "count": 1,
                }
            else:
                record["message"] = trimmed_message
                record["last_seen_ts"] = time.time()
                record["count"] = int(record.get("count", 0)) + 1

        self._publish_operation_error_update(operation)

    def clear_operation_error(self, operation: Operation) -> None:
        """Clear an operation error entry after success.

        Args:
            operation: The operation that succeeded.
        """
        if operation is None:
            return
        with self.operation_errors_lock:
            self.operation_errors.pop(operation.uuid, None)

        self._publish_operation_error_update(operation)

    def get_operation_errors(self) -> list[dict[str, Any]]:
        """Get the current list of operation error entries.

        Returns:
            List of error records.
        """
        with self.operation_errors_lock:
            errors = [record.copy() for record in self.operation_errors.values()]
        return sorted(
            errors, key=lambda record: record.get("last_seen_ts", 0), reverse=True
        )

    def _publish_operation_error_update(self, operation: Operation) -> None:
        """Publish SSE update for the current operation error state.

        Args:
            operation: The operation that triggered the update.
        """
        try:
            if not self.web_interface:
                return
            error_payload = {
                "pipeline_name": self.pipeline_name,
                "operation_uuid": operation.uuid,
                "errors": self.get_operation_errors(),
            }
            self.web_interface.publish_operation_errors(error_payload)
        except Exception:
            return

    def _publish_operation_error_snapshot(self) -> None:
        """Publish SSE update for the current operation error state."""
        try:
            if not self.web_interface:
                return
            error_payload = {
                "pipeline_name": self.pipeline_name,
                "errors": self.get_operation_errors(),
            }
            self.web_interface.publish_operation_errors(error_payload)
        except Exception:
            return
