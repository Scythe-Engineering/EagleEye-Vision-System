## Pipeline Overview

EagleEye-Object-Detection runs its data processing as a sequence of operations called a pipeline. Each pipeline ingests camera frames, processes them through a chain of operations, and yields outputs such as pose estimates or processed frames. The system is designed to be extensible: new operations can be added either as main operations (definitions) or as secondary operations, and then wired into pipelines via configuration.

The divide between secondary and primary operations is complexity, secondary operations must be in one file and under 200 lines of code. If a operation is larger than this it should be re-implemented as a primary operation.

## Core concepts

- **Pipeline**: a chain of operations applied to each input frame. Implemented by `class Pipeline` in `src/config/utils/pipeline.py`.
- **Operation types**:
    - **Main operations (definitions)**: implemented under `src/main_operations/definitions`. Each operation class typically ends with `Definition` (e.g., `ApriltagCnnPreprocessorDefinition`).
    - **Secondary operations**: implemented under `src/secondary_operations`. Each operation is a standalone class (e.g., `FlattenPose`, `RobotPoseOutput`).
    - **Data source operations**: operations that generate their own data independently (no input connections). These are marked with `is_data_source: true` in their config_def.json file and execute one timestep before their data is needed to get the most up-to-date value possible. Examples include `GetNetworktablesValue` which reads from NetworkTables.
- **Configuration-driven wiring**: Pipelines are created by reading `src/config/pipeline_config.json` via `generate_all_pipelines.py`.
- **Inference resources**: `DeviceRegistry` provides an immutable startup inventory using canonical IDs, while `ModelLibrary` owns managed model metadata and artifacts. Neither service allocates, balances, migrates, or falls back between devices.

## File layout and roles

- `src/config/utils/pipeline.py` — Core orchestration and the `Pipeline` class. Responsible for:
    - loading pipeline configuration,
    - instantiating operation objects via dynamic importing,
    - running a chain of operations on each input frame,
    - optionally emitting timing statistics.
- `src/config/utils/generate_all_pipelines.py` — Builds pipelines for all configured cameras by reading `pipeline_config.json` and creating `Pipeline` instances.
- `src/main_operations/definitions/` — Definition classes for main operations. Examples include:
    - `color_threshold_detection.py` (definition class `ColorThresholdDetectionDefinition`)
    - `detect_apriltags.py` (definition class `DetectApriltagsDefinition`)
    - `object_detection.py` (definition class `ObjectDetectionDefinition`)
    - `pnp_camera_localization.py` (definition class `PnpCameraLocalizationDefinition`)
    - `temporal_acceleration_preprocessor_rust.py` (definition class `TemporalAccelerationPreprocessorRustDefinition`)
- `src/secondary_operations/` — Secondary post-processing steps such as:
    - `device_input.py` (class `DeviceInput`) - Pipeline entry point
    - `camera_to_robot_pose.py` (class `CameraToRobotPose`)
    - `flatten_pose.py` (class `FlattenPose`)
    - `ground_plane_intersection.py` (class `GroundPlaneIntersection`)
    - `robot_pose_output.py` (class `RobotPoseOutput`)
    - `tag_filter.py` (class `TagFilter`)
    - `publish_to_networktables.py` (class `PublishToNetworktables`)
    - `extract_pose.py` (class `ExtractPose`)
    - `angle_to_objects.py` (class `AngleToObjects`)
    - `camera_adjust.py` (class `CameraAdjust`)
    - `fps_limiter.py` (class `FpsLimiter`)
    - `pose_outlier_filter_rust.py` (class `PoseOutlierFilterRust`)
    - `robot_local_to_field_transform.py` (class `RobotLocalToFieldTransform`)
    - `detected_objects_output.py` (class `DetectedObjectsOutput`)
    - `get_networktables_value.py` (class `GetNetworktablesValue`) - Data source operation that reads from NetworkTables
- `src/config/pipeline_config.json` — The per-camera configuration mapping each pipeline step to an action name and parameters.

## Pipeline Configuration and UI

### Configuration-Driven Instantiation

Pipelines are instantiated by reading the pipeline config in `generate_all_pipelines.py`. The config file comprises per-camera entries whose values are lists of operation specs. Each operation spec includes positioning information for the visual flowchart interface:

```json
{
    "CAM0": [
        {
            "action_name": "device_input.py",
            "action_params": {},
            "position": {
                "x": 100,
                "y": 100
            }
        },
        {
            "action_name": "detect_apriltags.py",
            "action_params": {
                "families": "tag36h11"
            },
            "position": {
                "x": 400,
                "y": 120
            }
        },
        {
            "action_name": "pnp_camera_localization.py",
            "action_params": {
                "camera_bus_id": "front_cam",
                "apriltag_map_path": "/path/to/apriltag_map.json"
            },
            "position": {
                "x": 700,
                "y": 140
            }
        },
        {
            "action_name": "camera_to_robot_pose.py",
            "action_params": {
                "camera_bus_id": "front_cam"
            },
            "position": {
                "x": 1000,
                "y": 160
            }
        },
        {
            "action_name": "flatten_pose.py",
            "action_params": {},
            "position": {
                "x": 1150,
                "y": 160
            }
        },
        {
            "action_name": "robot_pose_output.py",
            "action_params": {},
            "position": {
                "x": 1450,
                "y": 180
            }
        }
    ]
}
```

### Visual Pipeline Editor

The EagleEye WebUI provides a flowchart-style visual pipeline editor that allows users to:

- **Drag and Drop**: Operations can be dragged from a side panel onto a grid-based canvas
- **Visual Layout**: Operations appear as nodes positioned according to their `position` coordinates
- **Grid Snapping**: Nodes snap to a grid for consistent, professional layouts
- **Interactive Editing**: Click nodes to configure settings, drag to reposition, or remove operations
- **Real-time Saving**: Pipeline changes are automatically saved to the configuration file
- **Scrollable Canvas**: Large 4000x4000px workspace for complex pipeline layouts

The `position` field stores the x,y coordinates of each operation node in the visual editor. Operations without position data default to calculated positions.

Notes:

- The framework inspects actual constructor parameters and injects only requested backend services such as `web_interface`, `device_registry`, `model_library`, `network_table`, `mx3_coordinator`, camera services, and `logger`.
- The modules are loaded in this order: first try `src.main_operations.definitions.{action_name}`; if not found, fall back to `src.secondary_operations.{action_name}`.
- Operation input/output declarations are validated when pipelines are saved and constructed. Connections from multi-output operations route only the selected `from_port`; a dictionary returned by a single-output operation remains one payload.

## Typical execution flow

1. The camera thread manager feeds frames to the appropriate pipeline via `Pipeline.run(frame)`.
2. Each operation processes the frame and returns the output to the next operation.
3. The final output could be a transformed frame, a pose matrix, or a domain-specific object, depending on your configuration.

Pipeline Settings includes an optional **Limit frames to camera capture speed**
mode. All `device_input` operations in a named pipeline must belong to the same
connected graph. When enabled, the scheduler runs the complete pipeline again
only after every `device_input` has published a new frame. Pipelines without a
`device_input` continue running continuously. The setting is persisted in
`src/config/pipeline_settings.json` and takes effect after a backend restart.

For localization pipelines that publish to the 3D frontend, insert
`camera_to_robot_pose.py` after `pnp_camera_localization.py` so the frontend
receives robot pose rather than raw camera pose.

## Debugging and timing

- Set `debug_mode = True` inside `src/config/utils/pipeline.py` to print a timing summary after runs. The summary includes per-operation average times, total time, and FPS.

For more details, see related files:

- `src/config/utils/pipeline.py`
- `src/config/utils/generate_all_pipelines.py`
- `src/main_operations/definitions/*`
- `src/secondary_operations/*`
- `src/config/pipeline_config.json`
