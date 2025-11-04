# API Endpoints Summary

## Overview

The EagleEye WebUI provides a comprehensive REST API and WebSocket interface for managing camera feeds, settings, and real-time robot tracking. This summary covers the main endpoints and their functionality.

**Base URL:** `http://localhost:5001`

## HTTP Endpoints

### Static File Serving

#### `GET /`

- **Purpose**: Serves the main web interface
- **Response**: HTML page with the complete application
- **Use**: Primary entry point for the web application

#### `GET /script.js`

- **Purpose**: Serves the main JavaScript bundle
- **Response**: Minified JavaScript containing all frontend logic
- **Use**: Client-side application code

#### `GET /main.css`

- **Purpose**: Serves the main CSS stylesheet
- **Response**: Compiled CSS with all styling
- **Use**: Application styling and layout

#### `GET /background.png`

- **Purpose**: Serves the application background image
- **Response**: PNG image file
- **Use**: Visual background for the application

#### `GET /favicon.ico`

- **Purpose**: Serves the browser favicon
- **Response**: ICO image file
- **Use**: Browser tab icon

#### `GET /frc2025r2.json`

- **Purpose**: Serves FRC 2025 AprilTag configuration
- **Response**: JSON configuration file
- **Use**: AprilTag layout and calibration data

#### `GET /src/webui/assets/apriltags/<filename>`

- **Purpose**: Serves AprilTag image assets
- **Parameters**: `filename` - Name of the AprilTag image file
- **Response**: Image file (PNG)
- **Use**: Fiducial markers for robot localization

### Settings Management

#### `GET /get-settings`

- **Purpose**: Retrieves current application settings
- **Response**: JSON object with all current settings
- **Status**: 200 OK
- **Use**: Load current configuration into the UI

#### `POST /save-settings`

- **Purpose**: Updates application settings
- **Request Body**: JSON object with settings to update
- **Response**: Success/error message
- **Status**: 200 OK / 500 Internal Server Error
- **Use**: Persist user configuration changes

### Camera Management

#### `GET /get-available-cameras`

- **Purpose**: Retrieves list of available cameras
- **Response**: JSON array of camera names
- **Status**: 200 OK
- **Use**: Populate camera selection dropdowns

#### `GET /feed/<camera_name>`

- **Purpose**: Returns MJPEG video stream for a camera
- **Parameters**: `camera_name` - URL-safe camera identifier
- **Response**: MJPEG stream
- **Status**: 200 OK / 404 Not Found
- **Use**: Real-time video feed display

#### `GET /camera/<camera_name>/snapshot`

- **Purpose**: Returns a single frame from a camera
- **Parameters**: `camera_name` - URL-safe camera identifier
- **Response**: JPEG image
- **Status**: 200 OK / 404 Not Found
- **Use**: Static camera image capture

### System Management

#### `POST /restart-backend`

- **Purpose**: Triggers a backend system restart
- **Response**: Success/error message
- **Status**: 200 OK / 500 Internal Server Error
- **Use**: Apply configuration changes requiring restart

#### `POST /set-restart-required`

- **Purpose**: Sets flag indicating backend restart is required
- **Response**: Success message
- **Status**: 200 OK
- **Use**: Mark configuration as requiring restart

#### `GET /get-restart-required`

- **Purpose**: Gets current restart required status
- **Response**: JSON with restart_required boolean
- **Status**: 200 OK
- **Use**: Check if restart is needed before applying changes

#### `GET /get-pipeline-objects`

- **Purpose**: Retrieves current pipeline configuration
- **Response**: JSON object with pipeline structure
- **Status**: 200 OK
- **Use**: Load pipeline configuration into the editor

## WebSocket Events (SocketIO)

### Connection Management

- **`connect`**: Client successfully connects to server
- **`disconnect`**: Client disconnects from server
- **`connect_error`**: Connection attempt fails
- **`reconnect`**: Client successfully reconnects after disconnection

### Real-time Data

- **`update_robot_transform`**: Robot position and orientation updates
    - **Data**: Transform matrix with position and rotation
    - **Use**: Update 3D visualization with current robot pose

## Data Formats

### Settings JSON Structure

```json
{
  "setting_name": "value",
  "camera_resolution": "640x480",
  "pipeline_enabled": true,
  "calibration_data": {
    "camera_matrix": [[...], [...], [...]],
    "distortion_coefficients": [...]
  }
}
```

### Camera List Response

```json
["camera_1", "camera_2", "overhead_camera"]
```

### Robot Transform Data

```json
{
    "transform_matrix": [
        [1.0, 0.0, 0.0, 1000.0],
        [0.0, 1.0, 0.0, 2000.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]
}
```

## Error Handling

### HTTP Status Codes

- **200 OK**: Successful operation
- **404 Not Found**: Camera or resource not found
- **500 Internal Server Error**: Server-side error

### Error Response Format

```json
{
    "error": "Description of the error",
    "details": "Additional error information"
}
```

## Authentication & Security

### Current State

- **Authentication**: None required (development mode)
- **CORS**: Configured for `http://localhost:5173`
- **Security**: Basic HTTP, no encryption in development

### Production Considerations

- Add authentication headers
- Implement HTTPS
- Configure CORS for production domains
- Add rate limiting and input validation

## Performance Considerations

### Streaming

- MJPEG streams provide low-latency video
- Frame rate depends on camera and network
- Connection pooling for multiple clients

### Real-time Updates

- SocketIO provides efficient bidirectional communication
- Configured ping timeout: 60s, ping interval: 25s
- Automatic reconnection with exponential backoff

### Caching

- Static assets cached by browser
- Settings cached in memory
- Camera frames served fresh (no caching)

## Usage Examples

### JavaScript (Frontend)

```javascript
// Load settings
fetch("/get-settings")
    .then((response) => response.json())
    .then((settings) => console.log(settings));

// Save settings
fetch("/save-settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ setting: "value" }),
});

// Camera stream
const img = document.getElementById("camera-stream");
img.src = "/camera/front_camera";
```

### Python (External Integration)

```python
import requests

# Get settings
response = requests.get('http://localhost:5001/get-settings')
settings = response.json()

# Update settings
requests.post('http://localhost:5001/save-settings',
              json={'new_setting': 'value'})
```

## Integration Guidelines

### For External Applications

1. Use appropriate content-type headers for POST requests
2. Handle connection failures gracefully
3. Implement reconnection logic for WebSocket connections
4. Cache frequently accessed data locally

### For Development

1. Test endpoints with tools like Postman or curl
2. Monitor network traffic for performance optimization
3. Use browser developer tools for WebSocket debugging
4. Implement proper error handling in client applications

## Future Enhancements

### Planned Features

- Authentication and authorization
- API versioning
- Bulk operations for multiple cameras
- Advanced filtering and search for settings
- Webhook support for external integrations
- API rate limiting and throttling
- Comprehensive logging and monitoring

### API Evolution

- Maintain backward compatibility
- Deprecate old endpoints with clear migration paths
- Provide comprehensive documentation updates
- Implement proper semantic versioning
