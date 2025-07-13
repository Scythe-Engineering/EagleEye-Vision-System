import time
from typing import Any, Dict, List
import numpy as np
import importlib
from src.webui.web_server import EagleEyeInterface
from src.utils.camera_utils.camera_thread_manager import CameraThreadManager
import threading

debug_mode = True


class Pipeline:
    """Pipeline for processing data through a sequence of operations."""

    def __init__(self, pipeline_config: dict, web_interface: EagleEyeInterface, camera_bus_id: str) -> None:
        """Initialize the pipeline with configuration.

        Args:
            pipeline_config: Dictionary containing pipeline configuration.
        """
        self.pipeline_config = pipeline_config
        self.web_interface = web_interface
        self.camera_bus_id = camera_bus_id
        self.thread_running = False
        self.thread = None
        self.operations = self._initialize_operations()
        
    def _snake_to_camel(self, snake_str: str) -> str:
        """Convert snake_case string to CamelCase.

        Args:
            snake_str: String in snake_case format.

        Returns:
            String in CamelCase format.
        """
        components = snake_str.split('_')
        return ''.join(word.capitalize() for word in components)

    def _initialize_operations(self) -> List[Any]:
        """Initialize operation instances based on configuration.

        Returns:
            List of initialized operation instances.
        """
        operations = []
        
        for operation_config in self.pipeline_config:
            action_name = operation_config["action_name"]
            action_params = operation_config.get("action_params", {})
            
            operation_instance = self._create_operation_instance(action_name, action_params)
            operations.append(operation_instance)
        
        return operations

    def _create_operation_instance(self, action_name: str, action_params: Dict[str, Any]) -> Any:
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
                    raise ValueError(f"Could not find class for action: {class_name} at {action_name}")
                
            # Add web_interface parameter if the operation requires it
            if hasattr(operation_class.__init__, '__code__') and 'web_interface' in operation_class.__init__.__code__.co_varnames:
                action_params['web_interface'] = self.web_interface
                
            return operation_class(**action_params)
            
        except TypeError as e:
            raise ValueError(f"Invalid parameters for {action_name}: {str(e)}")

    def run(self, input_data: np.ndarray) -> Any:
        """Run the pipeline with the given input data.

        Args:
            input_data: Input data to process through the pipeline.

        Returns:
            Final output after processing through all operations.

        Raises:
            ValueError: If pipeline operations are empty or input validation fails.
        """
        if not self.operations:
            raise ValueError("No operations configured in pipeline")

        current_data = input_data
        
        time_elapsed = 0
        
        for i, operation in enumerate(self.operations):
            try:
                start_time = time.time()
                current_data = operation.run(current_data)
                if current_data is None and i != len(self.operations) - 1:
                    if debug_mode:
                        print(f"Operation {i} ({type(operation).__name__}) returned None, skipping the rest of the pipeline")
                    break
                end_time = time.time()
                if debug_mode:
                    print(f"Operation {i} ({type(operation).__name__}) time: {round((end_time - start_time) * 1000, 2)} ms")
                    time_elapsed += end_time - start_time
            except Exception as e:
                raise RuntimeError(f"Error in operation {i} ({type(operation).__name__}): {str(e)}")
        if debug_mode:
            print(f"Total time elapsed: {round(time_elapsed * 1000, 2)} ms")
            print(f"Fps: {round(1 / time_elapsed, 2)}")
            print("\n")

        return current_data
    
    def thread_run(self, camera_thread_manager: CameraThreadManager, camera_bus_id: str) -> None:
        """Run the pipeline continuously in a thread.

        Args:
            camera_thread_manager: The camera thread manager.
            camera_bus_id: The bus ID of the camera to run the pipeline on.
        """
        self.thread = threading.Thread(target=self._thread_run, args=(camera_thread_manager, camera_bus_id))
        self.thread.start()
        self.thread_running = True

    def _thread_run(self, camera_thread_manager: CameraThreadManager, camera_bus_id: str) -> None:
        """Run the pipeline continuously in a thread.

        Args:
            camera_thread_manager: The camera thread manager.
            camera_bus_id: The bus ID of the camera to run the pipeline on.
        """
        if not camera_thread_manager.get_camera_ready(camera_bus_id):
            print(f"Camera bus id: {camera_bus_id} is not ready, waiting for camera to be ready")
            while not camera_thread_manager.get_camera_ready(camera_bus_id):
                time.sleep(0.01)
            print(f"Camera bus id: {camera_bus_id} is ready")
        
        print(f"Starting pipeline for camera bus id: {camera_bus_id}")
        while self.thread_running:
            result = camera_thread_manager.get_current_frame(camera_bus_id)
            if result is not None:
                frame, _ = result
                try:
                    self.run(frame)
                except Exception as e:
                    print(f"Error in pipeline: {e}")
            else:
                time.sleep(0.01)

    def stop(self) -> None:
        """Stop the pipeline thread."""
        self.thread_running = False
        if self.thread is not None:
            self.thread.join()
            self.thread = None
