# EagleEyeInterface API Documentation

## Overview

The EagleEyeInterface is a Flask-based web server with SocketIO support that provides camera feed streaming, settings management, and real-time sphere position tracking for FIRST robotics applications.

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

#### GET `/bundle.js.map`
**Description:** Serves the JavaScript source map  
**Response:** Source map file  
**Content-Type:** `application/json`

#### GET `/background.png`
**Description:** Serves the background image  
**Response:** PNG image file  
**Content-Type:** `image/png`

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
**Description:** Retrieves current application settings  
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
**Description:** Updates application settings  
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
**Description:** Retrieves list of available cameras  
**Response:**
```json
{
  "camera_name": "camera_id",
  "another_camera": "camera_id"
}
```
**Status Codes:**
- `200`: Success

#### GET `/feed/<camera_name>`
**Description:** Streams live camera feed  
**Parameters:**
- `camera_name` (path): Name of the camera (spaces replaced with underscores)
**Response:** Multipart HTTP stream with JPEG frames  
**Content-Type:** `multipart/x-mixed-replace; boundary=frame`  
**Frame Rate:** Up to 120 FPS (throttled based on processing time)

### Sphere Position Tracking

#### POST `/update-sphere-position`
**Description:** Updates the tracked sphere's position via transformation matrix  
**Request Body:**
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
**Response:**
```json
{
  "message": "Sphere position updated successfully"
}
```
**Status Codes:**
- `200`: Position updated successfully
- `400`: Missing transform_matrix in request
- `500`: Failed to update sphere position

## WebSocket Events

The server uses SocketIO for real-time communication.

### Server-to-Client Events

#### `update_sphere_position`
**Description:** Broadcasts updated sphere position to all connected clients  
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
  "camera_name": "camera_identifier"
}
```

### Settings Object
The structure depends on the Constants class configuration. Typically includes:
```json
{
  "camera_settings": {},
  "detection_settings": {},
  "processing_settings": {}
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

socket.on('update_sphere_position', (data) => {
    console.log('Sphere position updated:', data.transform_matrix);
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

#### Displaying Camera Feed
```html
<img src="/feed/Camera_Name" alt="Camera Feed" />
```

#### Updating Sphere Position
```javascript
const transformMatrix = [
    [1.0, 0.0, 0.0, 16.96816403],
    [0.0, 1.0, 0.0, 6.57341747],
    [0.0, 0.0, 1.0, 0.66152486],
    [0.0, 0.0, 0.0, 1.0]
];

fetch('/update-sphere-position', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        transform_matrix: transformMatrix
    })
})
.then(response => response.json())
.then(result => {
    console.log('Position update result:', result.message);
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

# Update sphere position
transform_matrix = [
    [1.0, 0.0, 0.0, 16.96816403],
    [0.0, 1.0, 0.0, 6.57341747],
    [0.0, 0.0, 1.0, 0.66152486],
    [0.0, 0.0, 0.0, 1.0]
]

response = requests.post(
    'http://localhost:5001/update-sphere-position',
    json={"transform_matrix": transform_matrix}
)
print(f"Position update result: {response.json()}")
```

#### SocketIO Client
```python
import socketio

sio = socketio.Client()

@sio.event
def connect():
    print('Connected to server')

@sio.event
def update_sphere_position(data):
    print(f'Sphere position updated: {data["transform_matrix"]}')

sio.connect('http://localhost:5001')
sio.wait()
```

## Server Configuration

### Initialization Parameters
```python
interface = EagleEyeInterface(
    settings_object=None,  # Optional Constants object
    dev_mode=False,        # Run in development mode
    log=None              # Optional logging function
)
```

### Camera Feed Configuration
- **Resolution:** 640x480 (default)
- **Format:** JPEG
- **Max Frame Rate:** 120 FPS (throttled by processing time)
- **Min Frame Rate:** 30 FPS (for camera updates)

### WebSocket Configuration
- **CORS:** Enabled for all origins
- **Ping Timeout:** 60 seconds
- **Ping Interval:** 25 seconds
- **Async Mode:** Threading

## Notes

- Camera names with spaces are converted to underscores in URLs
- The server automatically detects available cameras on startup
- Frame updates are throttled to prevent overwhelming the client
- All camera operations are thread-safe using locks
- The server runs on all interfaces (0.0.0.0) on port 5001 