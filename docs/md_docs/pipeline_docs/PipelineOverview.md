## Pipeline Overview

EagleEye-Object-Detection runs its data processing as a sequence of operations called a pipeline. Each pipeline ingests camera frames, processes them through a chain of operations, and yields outputs such as pose estimates or processed frames. The system is designed to be extensible: new operations can be added either as main operations (definitions) or as secondary operations, and then wired into pipelines via configuration.

The divide between secondary and primary operations is complexity, secondary operations must be in one file and under 200 lines of code. If a operation is larger than this it should be re-implemented as a primary operation.

## Core concepts

- **Pipeline**: a chain of operations applied to each input frame. Implemented by `class Pipeline` in `src/config/utils/pipeline.py`.
- **Operation types**:
  - **Main operations (definitions)**: implemented under `src/main_operations/definitions`. Each operation class typically ends with `Definition` (e.g., `ApriltagCnnPreprocessorDefinition`).
  - **Secondary operations**: implemented under `src/secondary_operations`. Each operation is a standalone class (e.g., `FlattenPose`, `RobotPoseOutput`).
- **Configuration-driven wiring**: Pipelines are created by reading `src/config/pipeline_config.json` via `generate_all_pipelines.py`.
- **Compute resources**: The system relies on a `ComputePool` (from `src.utils.device_management_utils.compute_pool`) to allocate devices (CPU, MX3, CUDA, etc.). A `web_interface` (from `src.webui.web_server`) provides integration with the UI.

## File layout and roles

- `src/config/utils/pipeline.py` — Core orchestration and the `Pipeline` class. Responsible for:
  - loading pipeline configuration,
  - instantiating operation objects via dynamic importing,
  - running a chain of operations on each input frame,
  - optionally emitting timing statistics.
- `src/config/utils/generate_all_pipelines.py` — Builds pipelines for all configured cameras by reading `pipeline_config.json` and creating `Pipeline` instances, then wiring compute devices.
- `src/main_operations/definitions/` — Definition classes for main operations. Examples include:
  - `apriltag_cnn_preprocessor.py` (definition class likely named `ApriltagCnnPreprocessorDefinition`)
  - `detect_apriltags.py` (definition class `DetectApriltagsDefinition`)
  - `pnp_camera_localization.py` (definition class `PnpCameraLocalizationDefinition`)
- `src/secondary_operations/` — Secondary post-processing steps such as:
  - `flatten_pose.py` (class `FlattenPose`)
  - `velocity_based_filtering.py` (class `VelocityBasedFiltering`)
  - `robot_pose_output.py` (class `RobotPoseOutput`)
- `src/config/pipeline_config.json` — The per-camera configuration mapping each pipeline step to an action name and parameters.

## How configuration drives pipelines

Pipelines are instantiated by reading the pipeline config in `generate_all_pipelines.py`. The config file comprises per-camera entries whose values are lists of operation specs:

```json
{
  "CAM0": [
    {
      "action_name": "apriltag_cnn_preprocessor",
      "action_params": {
        "model_path": "/path/to/model.onnx",
        "device_id": "MX3",
        "conf_threshold": 0.15
      }
    },
    {
      "action_name": "detect_apriltags",
      "action_params": {
        "families": "tag36h11"
      }
    },
    {
      "action_name": "pnp_camera_localization",
      "action_params": {
        "camera_parameters_path": "/path/to/camera_parameters.yaml",
        "apriltag_map_path": "/path/to/apriltag_map.fmap"
      }
    }
  ],
  "CAM1": [
    {
      "action_name": "apriltag_cnn_preprocessor",
      "action_params": {
        "model_path": "/path/to/model.onnx",
        "device_id": "MX3",
        "conf_threshold": 0.15
      }
    }
  ]
}
```

Notes:
- The framework will inject `web_interface` and `compute_pool` into operation constructors if their signatures include `web_interface` and/or `compute_pool`.
- The modules are loaded in this order: first try `src.main_operations.definitions.{action_name}`; if not found, fall back to `src.secondary_operations.{action_name}`.

## End-to-end example

Below is a minimal fully-working example of wiring and running pipelines for all configured cameras. This assumes you have properly initialized `EagleEyeInterface` and `ComputePool` in your environment.

```python
from src.config.utils.generate_all_pipelines import generate_all_pipelines
from src.webui.web_server import EagleEyeInterface
from src.utils.device_management_utils.compute_pool import ComputePool

# Instantiate the shared components
web_interface = EagleEyeInterface()
compute_pool = ComputePool()

# Generate pipelines for all configured cameras
pipelines = generate_all_pipelines(web_interface, compute_pool)

# Now `pipelines` holds Pipeline objects ready to be started by your camera threads/manager.
```

## Typical execution flow

1) The camera thread manager feeds frames to the appropriate pipeline via `Pipeline.run(frame)`.
2) Each operation processes the frame and returns the output to the next operation.
3) The final output could be a transformed frame, a pose matrix, or a domain-specific object, depending on your configuration.

## Debugging and timing

- Set `debug_mode = True` inside `src/config/utils/pipeline.py` to print a timing summary after runs. The summary includes per-operation average times, total time, and FPS.

For more details, see related files:
- `src/config/utils/pipeline.py`
- `src/config/utils/generate_all_pipelines.py`
- `src/main_operations/definitions/*`
- `src/secondary_operations/*`
- `src/config/pipeline_config.json`
