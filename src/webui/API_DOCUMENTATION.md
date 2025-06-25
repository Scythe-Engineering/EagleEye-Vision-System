# EagleEyeInterface API Documentation

## Overview

The EagleEyeInterface is a Flask-based web server with SocketIO support that provides camera feed streaming, settings management, and real-time robot position tracking for FIRST robotics applications.

**Base URL:** `http://localhost:5001`

## Table of Contents

- [Authentication](#authentication)
- [HTTP Endpoints](#http-endpoints)
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
  "robots": [
    "robot1.glb",
    "robot2.glb"
  ]
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
const socket = io('http://localhost:5001');

socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('update_robot_transform', (data) => {
    console.log('Robot transform updated:', data.transform_matrix);
});
```

#### Fetching Settings
```javascript
fetch('/get-settings')
    .then(response => response.json())
    .then(settings => {
        console.log('Current settings:', settings);
    });
```

#### Updating Settings
```javascript
const newSettings = {
    camera_resolution: "1920x1080",
    detection_threshold: 0.8
};

fetch('/save-settings', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify(newSettings)
})
.then(response => response.json())
.then(result => {
    console.log('Settings update result:', result.message);
});
```

#### Getting Available Cameras
```javascript
fetch('/get-available-cameras')
    .then(response => response.json())
    .then(cameras => {
        console.log('Available cameras:', cameras);
        // cameras is an object like {"Camera Name": "Camera_Name"}
    });
```

#### Displaying Camera Feed
```html
<img src="/feed/Camera_Name" alt="Camera Feed" />
```

#### Getting Available Robots
```javascript
fetch('/get-available-robots')
    .then(response => response.json())
    .then(data => {
        console.log('Available robots:', data.robots);
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

## Notes

- Server runs on all interfaces (0.0.0.0) on port 5001
- Camera detection occurs on startup and when `/get-available-cameras` is called
- All static files are served from the webui directory structure
- Robot files must be in GLB format and located in `assets/robots/`
- AprilTag images are served from `assets/apriltags/`
- Error handling includes global exception handler that logs and returns 500 status 