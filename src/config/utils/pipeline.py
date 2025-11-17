import importlib
import threading
import time
import traceback
from collections import deque
from typing import Any, Dict, List

import cv2
import numpy as np
from networktables import NetworkTable

from src.config.utils.print_timing_summary import print_timing_summary
from src.utils.camera_utils.camera_thread_manager import CameraThreadManager
from src.utils.colors import Colors
from src.utils.device_management_utils.compute_pool import ComputePool
from src.webui.web_server import EagleEyeInterface

debug_mode = False


class Pipeline:
    """Pipeline for processing data through a sequence of operations."""

    def __init__(
        self,
        pipeline_config: dict,
        web_interface: EagleEyeInterface,
        camera_bus_id: str,
        compute_pool: ComputePool,
        network_table: NetworkTable,
        camera_manager: CameraThreadManager | None = None,
    ) -> None:
        """Initialize the pipeline with configuration.

        Args:
            pipeline_config: Dictionary containing pipeline configuration.
            web_interface: The web interface to use for the pipelines.
            camera_bus_id: The bus ID of the camera to run the pipeline on.
            compute_pool: The compute pool to use for the pipelines.
            network_table: The network table to use for the pipelines.
        """
        self.pipeline_config = pipeline_config
        self.web_interface = web_interface
        self.camera_bus_id = camera_bus_id
        self.compute_pool = compute_pool
        self.network_table = network_table
        self.camera_manager = camera_manager

        self.thread_running = False
        self.thread = None
        self.operations = self._initialize_operations()
        self.operation_time_history: List[deque[float]] = [
            deque(maxlen=50) for _ in range(len(self.operations))
        ]
        self.total_time_history: deque[float] = deque(maxlen=50)
        self.total_time_history_lock = threading.Lock()
        self.all_total_times: List[float] = []

        self.set_visualize = False
        self.visualization_data = None
        self.visualization_data_lock = threading.Lock()
        self.visualization_operation_name = None

    def _snake_to_camel(self, snake_str: str) -> str:
        """Convert snake_case string to CamelCase.

        Args:
            snake_str: String in snake_case format.

        Returns:
            String in CamelCase format.
        """
        components = snake_str.split("_")
        return "".join(word.capitalize() for word in components)

    def _initialize_operations(self) -> List[Any]:
        """Initialize operation instances based on configuration.

        Returns:
            List of initialized operation instances.
        """
        operations = []

        for operation_config in self.pipeline_config:
            action_name = operation_config["action_name"]
            action_params = operation_config.get("action_params", {})

            operation_instance = self._create_operation_instance(
                action_name, action_params
            )
            operations.append(operation_instance)

        return operations

    def _create_operation_instance(
        self, action_name: str, action_params: Dict[str, Any]
    ) -> Any:
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
                        f"Could not find class for action: {class_name} at {action_name}"
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

            return operation_class(**init_params)

        except TypeError as e:
            raise ValueError(f"Invalid parameters for {action_name}: {str(e)}")

    def run(
        self,
        input_data: np.ndarray,
        visualize: bool = False,
        visualization_operation_name: str = None,
    ) -> Any | None:
        """Run the pipeline with the given input data.

        Args:
            input_data: Input data to process through the pipeline.
            visualize: Whether to visualize the pipeline.

        Returns:
            If visualize is True, returns a dictionary with the frame and visualization data.
            Otherwise, returns nothing

        Raises:
            ValueError: If pipeline operations are empty or input validation fails.
        """
        if not self.operations:
            raise ValueError("No operations configured in pipeline")

        current_data = input_data

        time_elapsed = 0.0

        for i, operation in enumerate(self.operations):
            try:
                start_time = time.time()
                current_data = operation.run(current_data)
                if current_data is None and i != len(self.operations) - 1:
                    if debug_mode:
                        print(
                            f"{Colors.YELLOW}Operation {i} ({type(operation).__name__}) returned None, skipping the rest of the pipeline{Colors.RESET}"
                        )
                    break
                end_time = time.time()
                elapsed = end_time - start_time
                self.operation_time_history[i].append(elapsed)
                time_elapsed += elapsed
            except Exception as _:
                raise RuntimeError(
                    f"Error in operation {i} ({type(operation).__name__})"
                ) from _
        with self.total_time_history_lock:
            self.total_time_history.append(time_elapsed)
        self.all_total_times.append(time_elapsed)

        if debug_mode:
            print_timing_summary(
                self.operations, self.operation_time_history, self.total_time_history
            )

        if visualize:
            return self._visualize(input_data.copy(), visualization_operation_name)

    def get_operation_by_class_name(self, class_name: str) -> Any:
        """Get an operation by its class name.

        Args:
            class_name: The name of the operation class.
        """
        return next(
            (
                op
                for op in self.operations
                if op.__class__.__name__.strip("Definition")
                == class_name.strip("Definition")
            ),
            None,
        )

    def get_pipeline_by_name(
        self,
        pipeline_name: str,
        camera_name: str | None = None,
    ) -> None:
        """Retrieve a pipeline by name for the specified camera.

        Args:
            pipeline_name: Name of the pipeline to retrieve.
            camera_name: Optional camera identifier; defaults to the current pipeline camera.

        Returns:
            Pipeline | None: The matching pipeline instance if found, otherwise None.
        """
        pipeline_objects_callback = getattr(
            self.web_interface, "pipeline_objects_callback", None
        )
        if pipeline_objects_callback is None:
            return None
        pipeline_objects = pipeline_objects_callback()
        target_camera_name = (
            camera_name if camera_name is not None else self.camera_bus_id
        )
        if target_camera_name not in pipeline_objects:
            return None
        return pipeline_objects[target_camera_name].get(pipeline_name)

    def update_operations_config(self, operations_config: List[Dict[str, Any]]) -> None:
        """Update the configuration of multiple operations in the pipeline.

        This method allows live updating of operation parameters that are marked as
        restart_for_change: false in their configuration definition files.

        Args:
            operations_config: List of operation configurations, where each config
                is a dictionary with 'action_name' and 'action_params' keys.
                Format should match the pipeline configuration JSON format.
        """
        for operation_config in operations_config:
            action_name = operation_config.get("action_name")
            action_params = operation_config.get("action_params", {})

            if not action_name:
                continue

            # Convert action_name to class name format for lookup
            class_name = self._snake_to_camel(action_name)

            # Find the operation instance
            operation = self.get_operation_by_class_name(class_name)

            if operation is not None and hasattr(operation, "update_config"):
                try:
                    operation.update_config(action_params)
                    if debug_mode:
                        print(
                            f"{Colors.GREEN}Updated config for {action_name}: {action_params}{Colors.RESET}"
                        )
                except Exception as e:
                    print(
                        f"{Colors.RED}Error updating config for {action_name}: {e}{Colors.RESET}"
                    )
            elif operation is not None:
                if debug_mode:
                    print(
                        f"{Colors.YELLOW}Operation {action_name} does not support config updates{Colors.RESET}"
                    )
            else:
                if debug_mode:
                    print(
                        f"{Colors.RED}Operation {action_name} not found in pipeline{Colors.RESET}"
                    )

    def thread_run(
        self, camera_thread_manager: CameraThreadManager, camera_bus_id: str
    ) -> None:
        """Run the pipeline continuously in a thread.

        Args:
            camera_thread_manager: The camera thread manager.
            camera_bus_id: The bus ID of the camera to run the pipeline on.
        """
        self.thread_running = True
        self.thread = threading.Thread(
            target=self._thread_run, args=(camera_thread_manager, camera_bus_id)
        )
        self.thread.start()

    def _thread_run(
        self, camera_thread_manager: CameraThreadManager, camera_bus_id: str
    ) -> None:
        """Run the pipeline continuously in a thread.

        Args:
            camera_thread_manager: The camera thread manager.
            camera_bus_id: The bus ID of the camera to run the pipeline on.
        """
        if not camera_thread_manager.get_camera_ready(camera_bus_id):
            print(
                f"{Colors.YELLOW}Camera bus id: {camera_bus_id} is not ready, waiting for camera to be ready{Colors.RESET}"
            )
            while not camera_thread_manager.get_camera_ready(camera_bus_id):
                time.sleep(0.01)
            print(
                f"{Colors.GREEN}Camera bus id: {camera_bus_id} is ready{Colors.RESET}"
            )

        print(
            f"{Colors.CYAN}Starting pipeline for camera bus id: {camera_bus_id}{Colors.RESET}"
        )
        time.sleep(0.1)

        while self.thread_running:
            camera_frame_result = camera_thread_manager.get_current_frame(camera_bus_id)
            if camera_frame_result is not None:
                frame, _ = camera_frame_result
                try:
                    # Snapshot visualize state and target name atomically
                    with self.visualization_data_lock:
                        should_visualize = self.set_visualize
                        operation_name_snapshot = self.visualization_operation_name

                    if should_visualize:
                        frame_copy = frame.copy()
                        visualization_frame = self.run(
                            frame,
                            visualize=True,
                            visualization_operation_name=operation_name_snapshot,
                        )
                        # Only hold the lock for the assignment
                        with self.visualization_data_lock:
                            self.visualization_data = {
                                "frame": frame_copy,
                                "visualization_data": visualization_frame,
                            }
                    else:
                        self.run(frame)
                except Exception as _:
                    print(
                        f"{Colors.RED}Error in pipeline itself: {traceback.format_exc()}{Colors.RESET}"
                    )
            else:
                time.sleep(0.01)

    def _visualize(self, start_frame: np.ndarray, action_name: str) -> np.ndarray:
        """Visualize the pipeline up to the given action name.

        Args:
            action_name: The name of the action to visualize up to.

        Returns:
            The visualized frame.
        """
        # Normalize the target operation name; if None/empty, visualize full pipeline
        normalized_target = (
            action_name.lower().replace("_", "") if action_name else None
        )
        for operation in self.operations:
            start_frame = operation.visualize(start_frame)
            current_op_name = operation.__class__.__name__.lower().replace(
                "definition", ""
            )
            if normalized_target and current_op_name == normalized_target:
                break

        # Add FPS display in top left corner
        with self.total_time_history_lock:
            if self.total_time_history:
                avg_time = sum(self.total_time_history) / len(self.total_time_history)
            else:
                avg_time = 0.0
        fps = 1.0 / avg_time if avg_time > 0 else 0.0
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            start_frame,
            fps_text,
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),  # Yellow color in BGR
            2,
        )

        return start_frame

    def start_visualize(self, visualization_operation_name: str) -> None:
        """Start visualizing the pipeline."""
        # Ensure operation name is set before enabling visualization
        with self.visualization_data_lock:
            self.visualization_operation_name = visualization_operation_name
            self.set_visualize = True

    def stop_visualize(self) -> None:
        """Stop visualizing the pipeline."""
        with self.visualization_data_lock:
            self.set_visualize = False

    def stop(self) -> None:
        """Stop the pipeline thread."""
        self.thread_running = False
        if self.thread is not None:
            self.thread.join()
            self.thread = None
