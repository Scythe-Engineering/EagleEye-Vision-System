# EagleEyeInterface API Documentation

## Overview

The EagleEyeInterface is a Flask-based web server with SocketIO support that provides camera feed streaming, settings management, and real-time robot position tracking for FIRST robotics applications.

**Base URL:** `http://localhost:5001`

## Table of Contents

- [Authentication](#authentication)
- [HTTP Endpoints](#http-endpoints)
    - [Static File Endpoints](#static-file-endpoints)
    - [Settings Management](#settings-management)
    - [Camera Management](#camera-management)
    - [Robot Management](#robot-management)
    - [Pipeline Management](#pipeline-management)
    - [Pipeline Visualization](#pipeline-visualization)
    - [System Management](#system-management)
- [WebSocket Events](#websocket-events)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Usage Examples](#usage-examples)

## Authentication

Currently, the API does not require authentication. All endpoints are publicly accessible.

## HTTP Endpoints

### Static File Endpoints

#### GET `/`

**Description:** Serves the main web interface  
**Response:** HTML page  
**Content-Type:** `text/html`

#### GET `/script.js`

**Description:** Serves the main JavaScript bundle  
**Response:** JavaScript file  
**Content-Type:** `application/javascript`

#### GET `/main.css`

**Description:** Serves the main CSS stylesheet  
**Response:** CSS file  
**Content-Type:** `text/css`

#### GET `/background.png`

**Description:** Serves the background image from static directory  
**Response:** PNG image file  
**Content-Type:** `image/png`

#### GET `/favicon.ico`

**Description:** Serves the favicon from assets directory  
**Response:** ICO image file  
**Content-Type:** `image/x-icon`

#### GET `/frc2025r2.json`

**Description:** Serves the FRC 2025 AprilTag configuration  
**Response:** JSON configuration file  
**Content-Type:** `application/json`

#### GET `/src/webui/assets/apriltags/<filename>`

**Description:** Serves AprilTag image assets  
**Parameters:**

- `filename` (path): Name of the AprilTag image file
  **Response:** Image file  
  **Content-Type:** `image/*`

### Settings Management

#### GET `/get-settings`

**Description:** Retrieves current application settings from the Constants object  
**Response:**

```json
{
    "setting_name": "value",
    "another_setting": "value"
}
```

**Status Codes:**

- `200`: Success

#### POST `/save-settings`

**Description:** Updates application settings in the Constants object  
**Request Body:**

```json
{
    "setting_name": "new_value",
    "another_setting": "new_value"
}
```

**Response:**

```json
{
    "message": "Settings updated successfully"
}
```

**Status Codes:**

- `200`: Settings updated successfully
- `500`: Failed to update settings

### Camera Management

#### GET `/get-available-cameras`

**Description:** Retrieves list of available cameras with URL-safe names  
**Response:**

```json
{
    "Camera Name": "Camera_Name",
    "Another Camera": "Another_Camera"
}
```

**Status Codes:**

- `200`: Success

#### GET `/feed/<camera_name>`

**Description:** Streams live camera feed using multipart HTTP streaming  
**Parameters:**

- `camera_name` (path): URL-safe camera name (spaces replaced with underscores)
  **Response:** Multipart HTTP stream with JPEG frames  
  **Content-Type:** `multipart/x-mixed-replace; boundary=frame`  
  **Frame Rate:** Up to 120 FPS (throttled based on processing time)  
  **Fallback:** Returns no_image.png stream at 30 FPS if camera not found

### Robot Management

#### GET `/get-available-robots`

**Description:** Retrieves list of available robot 3D models  
**Response:**

```json
{
    "robots": ["robot1.glb", "robot2.glb"]
}
```

**Status Codes:**

- `200`: Success

#### GET `/get-robot-file/<filename>`

**Description:** Serves robot 3D model files
**Parameters:**

- `filename` (path): Name of the robot GLB file
  **Response:** GLB 3D model file
  **Content-Type:** `model/gltf-binary`

#### GET `/draco/<filename>`

**Description:** Serves Draco 3D compression library files
**Parameters:**

- `filename` (path): Name of the Draco library file (JavaScript/WebAssembly)
  **Response:** Draco library file
  **Content-Type:** `application/javascript` or `application/wasm`

### Pipeline Management

#### GET `/get-available-operations`

**Description:** Retrieves list of available pipeline operations (main and secondary)
**Response:**

```json
{
    "operations": [
        {
            "name": "operation_name.py",
            "path": "/path/to/operation",
            "config_data_path": "/path/to/config.json",
            "description": "Operation description",
            "category": "Operation category",
            "is_secondary": false
        }
    ]
}
```

**Status Codes:**

- `200`: Success

#### GET `/get-operation-config-data/<operation_name>/<is_secondary>`

**Description:** Retrieves configuration data for a specific operation
**Parameters:**

- `operation_name` (path): Name of the operation (without .py extension)
- `is_secondary` (int): 0 for main operations, 1 for secondary operations
  **Response:** Operation configuration JSON object
  **Status Codes:**
- `200`: Success

#### GET `/get-pipeline-names-for-camera/<camera_name>`

**Description:** Retrieves list of pipeline names for a specific camera
**Parameters:**

- `camera_name` (path): Name of the camera
  **Response:**

```json
["pipeline_name_1", "pipeline_name_2"]
```

**Status Codes:**

- `200`: Success

#### GET `/get-pipeline-config/<camera_name>/<pipeline_name>`

**Description:** Retrieves configuration for a specific pipeline
**Parameters:**

- `camera_name` (path): Name of the camera
- `pipeline_name` (path): Name of the pipeline
  **Response:** Pipeline configuration JSON object
  **Status Codes:**
- `200`: Success

#### POST `/save-pipeline-config/<camera_name>/<pipeline_name>`

**Description:** Saves/updates pipeline configuration
**Parameters:**

- `camera_name` (path): Name of the camera
- `pipeline_name` (path): Name of the pipeline
  **Request Body:**

```json
[
    {
        "action_name": "operation_name",
        "action_params": {
            "param1": "value1",
            "param2": "value2"
        }
    }
]
```

**Response:**

```json
{
    "message": "Pipeline config saved successfully"
}
```

**Status Codes:**

- `200`: Success

#### DELETE `/delete-pipeline/<camera_name>/<pipeline_name>`

**Description:** Deletes a pipeline configuration
**Parameters:**

- `camera_name` (path): Name of the camera
- `pipeline_name` (path): Name of the pipeline to delete
  **Response:**

```json
{
    "message": "Pipeline deleted successfully"
}
```

**Status Codes:**

- `200`: Success

### Pipeline Visualization

#### POST `/start-visualize/<camera_name>/<pipeline_name>`

**Description:** Starts visualization mode for a pipeline
**Parameters:**

- `camera_name` (path): Name of the camera
- `pipeline_name` (path): Name of the pipeline
  **Response:**

```json
{
    "message": "Pipeline visualized successfully"
}
```

**Status Codes:**

- `200`: Success

#### POST `/stop-visualize/<camera_name>/<pipeline_name>`

**Description:** Stops visualization mode for a pipeline
**Parameters:**

- `camera_name` (path): Name of the camera
- `pipeline_name` (path): Name of the pipeline
  **Response:**

```json
{
    "message": "Pipeline visualized stopped"
}
```

**Status Codes:**

- `200`: Success

#### GET `/visualize/<camera_name>/<pipeline_name>/<action_name>`

**Description:** Returns visualization image for a specific pipeline action
**Parameters:**

- `camera_name` (path): Name of the camera
- `pipeline_name` (path): Name of the pipeline
- `action_name` (path): Name of the action/operation to visualize
  **Response:** JPEG image of the visualization
  **Content-Type:** `image/jpeg`
  **Status Codes:**
- `200`: Success
- `500`: Function has no visualization or encoding failed

### System Management

#### POST `/restart-backend`

**Description:** Triggers a backend system restart
**Response:**

```json
{
    "message": "Backend restarted successfully"
}
```

**Status Codes:**

- `200`: Success

## WebSocket Events

The server uses SocketIO for real-time communication.

### Server-to-Client Events

#### `update_robot_transform`

**Description:** Broadcasts updated robot transformation matrix to all connected clients  
**Payload:**

```json
{
    "transform_matrix": [
        [1.0, 0.0, 0.0, 16.96816403],
        [0.0, 1.0, 0.0, 6.57341747],
        [0.0, 0.0, 1.0, 0.66152486],
        [0.0, 0.0, 0.0, 1.0]
    ]
}
```

## Data Models

### Transformation Matrix

A 4x4 matrix representing position and rotation in 3D space:

```json
[
  [r11, r12, r13, tx],
  [r21, r22, r23, ty],
  [r31, r32, r33, tz],
  [0.0, 0.0, 0.0, 1.0]
]
```

Where:

- `r11-r33`: Rotation matrix components
- `tx, ty, tz`: Translation (position) components

### Camera Information

```json
{
    "original_camera_name": "url_safe_camera_name"
}
```

### Settings Object

The structure depends on the Constants class configuration. Retrieved via `get_config()` method.

### Robot List

```json
{
    "robots": ["filename1.glb", "filename2.glb"]
}
```

### Operations List

```json
{
    "operations": [
        {
            "name": "operation_name.py",
            "path": "/full/path/to/operation.py",
            "config_data_path": "/full/path/to/config.json",
            "description": "Human-readable operation description",
            "category": "Operation category (e.g., 'Detection', 'Processing')",
            "is_secondary": false
        }
    ]
}
```

### Pipeline Configuration

```json
[
    {
        "action_name": "operation_name",
        "action_params": {
            "threshold": 0.8,
            "blur_kernel": 5,
            "min_area": 100
        }
    }
]
```

### Pipeline Names List

```json
["default_pipeline", "tracking_pipeline", "debug_pipeline"]
```

## Error Handling

### Standard Error Response

```json
{
    "message": "Error description"
}
```

### Common Status Codes

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `500`: Internal Server Error

### Error Logging

All errors are logged using the configured logging function (defaults to `print`).

## Usage Examples

### JavaScript Client Examples

#### Connecting to WebSocket

```javascript
const socket = io("http://localhost:5001");

socket.on("connect", () => {
    console.log("Connected to server");
});

socket.on("update_robot_transform", (data) => {
    console.log("Robot transform updated:", data.transform_matrix);
});
```

#### Fetching Settings

```javascript
fetch("/get-settings")
    .then((response) => response.json())
    .then((settings) => {
        console.log("Current settings:", settings);
    });
```

#### Updating Settings

```javascript
const newSettings = {
    camera_resolution: "1920x1080",
    detection_threshold: 0.8,
};

fetch("/save-settings", {
    method: "POST",
    headers: {
        "Content-Type": "application/json",
    },
    body: JSON.stringify(newSettings),
})
    .then((response) => response.json())
    .then((result) => {
        console.log("Settings update result:", result.message);
    });
```

#### Getting Available Cameras

```javascript
fetch("/get-available-cameras")
    .then((response) => response.json())
    .then((cameras) => {
        console.log("Available cameras:", cameras);
        // cameras is an object like {"Camera Name": "Camera_Name"}
    });
```

#### Displaying Camera Feed

```html
<img src="/feed/Camera_Name" alt="Camera Feed" />
```

#### Getting Available Robots

```javascript
fetch("/get-available-robots")
    .then((response) => response.json())
    .then((data) => {
        console.log("Available robots:", data.robots);
    });
```

#### Getting Available Operations

```javascript
fetch("/get-available-operations")
    .then((response) => response.json())
    .then((data) => {
        console.log("Available operations:", data.operations);
        // Filter by category
        const detectionOps = data.operations.filter(
            (op) => op.category === "Detection",
        );
    });
```

#### Getting Operation Configuration

```javascript
// Get config for main operation
fetch("/get-operation-config-data/apriltag_detection/0")
    .then((response) => response.json())
    .then((config) => {
        console.log("Operation config:", config);
    });

// Get config for secondary operation
fetch("/get-operation-config-data/contour_filter/1")
    .then((response) => response.json())
    .then((config) => {
        console.log("Secondary operation config:", config);
    });
```

#### Pipeline Management

```javascript
const cameraName = "front_camera";
const pipelineName = "default_pipeline";

// Get pipeline names for a camera
fetch(`/get-pipeline-names-for-camera/${cameraName}`)
    .then((response) => response.json())
    .then((names) => {
        console.log("Pipeline names:", names);
    });

// Get pipeline configuration
fetch(`/get-pipeline-config/${cameraName}/${pipelineName}`)
    .then((response) => response.json())
    .then((config) => {
        console.log("Pipeline config:", config);
    });

// Save pipeline configuration
const pipelineConfig = [
    {
        action_name: "apriltag_detection",
        action_params: {
            threshold: 0.8,
            blur_kernel: 5,
        },
    },
];

fetch(`/save-pipeline-config/${cameraName}/${pipelineName}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(pipelineConfig),
})
    .then((response) => response.json())
    .then((result) => {
        console.log("Save result:", result.message);
    });
```

#### Pipeline Visualization

```javascript
const cameraName = "front_camera";
const pipelineName = "default_pipeline";

// Start visualization
fetch(`/start-visualize/${cameraName}/${pipelineName}`, {
    method: "POST",
})
    .then((response) => response.json())
    .then((result) => {
        console.log("Visualization started:", result.message);
    });

// Get visualization image for specific action
const actionName = "apriltag_detection";
const imgElement = document.getElementById("visualization");
imgElement.src = `/visualize/${cameraName}/${pipelineName}/${actionName}`;

// Stop visualization
fetch(`/stop-visualize/${cameraName}/${pipelineName}`, {
    method: "POST",
})
    .then((response) => response.json())
    .then((result) => {
        console.log("Visualization stopped:", result.message);
    });
```

### Python Client Examples

#### Basic HTTP Client

```python
import requests
import json

# Get settings
response = requests.get('http://localhost:5001/get-settings')
settings = response.json()
print(f"Current settings: {settings}")

# Update settings
new_settings = {"camera_resolution": "1920x1080"}
response = requests.post(
    'http://localhost:5001/save-settings',
    json=new_settings
)
print(f"Update result: {response.json()}")

# Get available cameras
response = requests.get('http://localhost:5001/get-available-cameras')
cameras = response.json()
print(f"Available cameras: {cameras}")

# Get available robots
response = requests.get('http://localhost:5001/get-available-robots')
robots = response.json()
print(f"Available robots: {robots}")

# Get available operations
response = requests.get('http://localhost:5001/get-available-operations')
operations = response.json()
print(f"Available operations: {operations}")

# Get operation configuration
response = requests.get('http://localhost:5001/get-operation-config-data/apriltag_detection/0')
config = response.json()
print(f"Operation config: {config}")

# Pipeline management
camera_name = 'front_camera'
pipeline_name = 'default_pipeline'

# Get pipeline names
response = requests.get(f'http://localhost:5001/get-pipeline-names-for-camera/{camera_name}')
pipeline_names = response.json()
print(f"Pipeline names: {pipeline_names}")

# Get pipeline configuration
response = requests.get(f'http://localhost:5001/get-pipeline-config/{camera_name}/{pipeline_name}')
pipeline_config = response.json()
print(f"Pipeline config: {pipeline_config}")

# Save pipeline configuration
new_config = [
    {
        'action_name': 'apriltag_detection',
        'action_params': {
            'threshold': 0.8,
            'blur_kernel': 5
        }
    }
]
response = requests.post(
    f'http://localhost:5001/save-pipeline-config/{camera_name}/{pipeline_name}',
    json=new_config
)
print(f"Save result: {response.json()}")

# Delete pipeline
response = requests.delete(f'http://localhost:5001/delete-pipeline/{camera_name}/{pipeline_name}')
print(f"Delete result: {response.json()}")

# Pipeline visualization
# Start visualization
response = requests.post(f'http://localhost:5001/start-visualize/{camera_name}/{pipeline_name}')
print(f"Start visualization: {response.json()}")

# Get visualization image
response = requests.get(f'http://localhost:5001/visualize/{camera_name}/{pipeline_name}/apriltag_detection')
if response.status_code == 200:
    with open('visualization.jpg', 'wb') as f:
        f.write(response.content)
    print("Visualization image saved")
else:
    print(f"Visualization failed: {response.text}")

# Stop visualization
response = requests.post(f'http://localhost:5001/stop-visualize/{camera_name}/{pipeline_name}')
print(f"Stop visualization: {response.json()}")

# Restart backend
response = requests.post('http://localhost:5001/restart-backend')
print(f"Restart result: {response.json()}")
```

#### SocketIO Client

```python
import socketio

sio = socketio.Client()

@sio.event
def connect():
    print('Connected to server')

@sio.event
def update_robot_transform(data):
    print(f'Robot transform updated: {data["transform_matrix"]}')

sio.connect('http://localhost:5001')
sio.wait()
```

#### Updating Robot Position (Server-side method)

```python
import numpy as np

# This is called from within the EagleEyeInterface class
transform_matrix = np.array([
    [1.0, 0.0, 0.0, 16.96816403],
    [0.0, 1.0, 0.0, 6.57341747],
    [0.0, 0.0, 1.0, 0.66152486],
    [0.0, 0.0, 0.0, 1.0]
])

interface.update_robot_position(transform_matrix)
```

## Server Configuration

### Initialization Parameters

```python
interface = EagleEyeInterface(
    settings_object=None,  # Optional Constants object
    dev_mode=False,        # Run in development mode
    log=None              # Optional logging function (defaults to print)
)
```

### Camera Feed Configuration

- **Format:** JPEG
- **Max Frame Rate:** 120 FPS (throttled by processing time)
- **Fallback Frame Rate:** 30 FPS (for no_image when camera not found)
- **Thread Safety:** All camera operations use locks

### WebSocket Configuration

- **CORS:** Enabled for all origins (`*`)
- **Ping Timeout:** 60 seconds
- **Ping Interval:** 25 seconds
- **Async Mode:** Threading
- **Logging:** Disabled for both SocketIO and EngineIO

## Internal Methods

### Camera Frame Management

- `update_camera_frame(camera_name: str, frame: bytes)`: Updates frame for specific camera
- Camera names with spaces are converted to underscores for URL safety
- Frame list is thread-safe using locks
- Fallback to no_image.png when camera not available

### Robot Position Tracking

- `update_robot_position(transformation_matrix: np.ndarray)`: Emits robot transform via WebSocket
- Validates matrix is 4x4 before processing
- Converts numpy array to list for JSON serialization

### Pipeline Management Methods

- `get_available_operations()`: Returns list of main and secondary operations with metadata
- `get_operation_config_data(operation_name, is_secondary)`: Retrieves operation configuration
- `get_pipeline_config(camera_name, pipeline_name)`: Returns pipeline configuration from JSON file
- `get_pipeline_names_for_camera(camera_name)`: Lists all pipelines for a camera
- `save_pipeline_config(camera_name, pipeline_name)`: Saves pipeline configuration to JSON file
- `delete_pipeline(camera_name, pipeline_name)`: Removes pipeline from configuration

### Pipeline Visualization Methods

- `start_visualize(camera_name, pipeline_name)`: Enables visualization mode for pipeline
- `stop_visualize(camera_name, pipeline_name)`: Disables visualization mode
- `visualize(camera_name, pipeline_name, action_name)`: Returns JPEG image of operation result
- Converts numpy arrays to JPEG format for web transmission
- Handles cases where operations don't have visualizations

## Notes

- Server runs on all interfaces (0.0.0.0) on port 5001
- Camera detection occurs on startup and when `/get-available-cameras` is called
- All static files are served from the webui directory structure
- Robot files must be in GLB format and located in `assets/robots/`
- AprilTag images are served from `assets/apriltags/`
- Pipeline configurations are stored in `src/config/pipeline_config.json`
- Operations are categorized as main (primary) and secondary operations
- Draco 3D compression library files are served for web-based 3D model loading
- Pipeline visualization provides real-time debugging capabilities
- Error handling includes global exception handler that logs and returns 500 status
