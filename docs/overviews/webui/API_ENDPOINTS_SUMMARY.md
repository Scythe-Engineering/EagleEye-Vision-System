# API Endpoints Summary

## Overview

The EagleEye WebUI provides a comprehensive REST API and Server-Sent Events (SSE) interface for managing camera feeds, settings, pipelines, and real-time robot tracking. This summary covers all available endpoints and their functionality.

**Base URL:** `http://localhost:5001`

## HTTP Endpoints

### Static File Serving

#### `GET /`

- **Purpose**: Serves the main web interface
- **Response**: HTML page with the complete application
- **Use**: Primary entry point for the web application

#### `GET /js/main.js`

- **Purpose**: Serves the main JavaScript bundle (Vite output `bundle.js` in static assets)
- **Response**: JavaScript file containing all frontend logic
- **Use**: Client-side application code

#### `GET /style.css`

- **Purpose**: Serves the main CSS stylesheet (built `main.css` in static assets)
- **Response**: CSS file with all styling
- **Use**: Application styling and layout

#### `GET /background.webp`

- **Purpose**: Serves the application background image
- **Response**: WebP image file
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
- **Response**: Image files for AprilTag visualization
- **Use**: AprilTag marker display

#### `GET /get-robot-file/<filename>`

- **Purpose**: Serves 3D robot model files
- **Response**: GLTF/GLB robot model files
- **Use**: 3D robot visualization

#### `GET /draco/<filename>`

- **Purpose**: Serves Draco-compressed 3D geometry files
- **Response**: Draco-compressed files for 3D rendering
- **Use**: Optimized 3D model loading

### Camera Management

#### `GET /get-available-cameras`

- **Purpose**: Lists all configured cameras
- **Response**: JSON array of camera names
- **Use**: Camera selection and configuration

#### `GET /feed/<camera_name>`

- **Purpose**: Streams MJPEG video feed from camera
- **Response**: MJPEG stream
- **Use**: Real-time camera preview

### Robot Management

#### `GET /get-available-robots`

- **Purpose**: Lists available robot configurations
- **Response**: JSON array of robot names
- **Use**: Robot model selection

### Pipeline Management

#### `GET /get-available-operations`

- **Purpose**: Lists all available pipeline operations
- **Response**: JSON array of operation types
- **Use**: Pipeline configuration

#### `GET /get-operation-config-data/<operation_name>/<is_secondary>`

- **Purpose**: Gets configuration schema for an operation
- **Parameters**: `is_secondary` (0=main, 1=secondary)
- **Response**: JSON configuration schema
- **Use**: Dynamic form generation

#### `GET /get-pipeline-names-for-camera/<camera_name>`

- **Purpose**: Lists pipeline names for a camera
- **Response**: JSON array of pipeline names
- **Use**: Pipeline selection

#### `GET /get-pipeline-config/<camera_name>/<pipeline_name>`

- **Purpose**: Gets full pipeline configuration
- **Response**: JSON pipeline configuration
- **Use**: Pipeline editing and display

#### `POST /save-pipeline-config/<camera_name>/<pipeline_name>`

- **Purpose**: Saves pipeline configuration
- **Body**: JSON pipeline configuration
- **Response**: Success/error message
- **Use**: Pipeline creation and updates

#### `DELETE /delete-pipeline/<camera_name>/<pipeline_name>`

- **Purpose**: Deletes a pipeline configuration
- **Response**: Success/error message
- **Use**: Pipeline removal

### Pipeline Visualization

#### `POST /start-visualize/<camera_name>/<pipeline_name>/<operation_name>`

- **Purpose**: Starts visualization for a specific operation
- **Response**: Success/error message
- **Use**: Debug pipeline operation output

#### `GET /visualize/<camera_name>/<pipeline_name>`

- **Purpose**: Gets visualization image
- **Response**: JPEG image of operation output
- **Use**: Visual debugging of pipelines

#### `POST /stop-visualize/<camera_name>/<pipeline_name>`

- **Purpose**: Stops pipeline visualization
- **Response**: Success/error message
- **Use**: End visualization session

### System Management

#### `POST /restart-backend`

- **Purpose**: Triggers backend system restart
- **Response**: Success message
- **Use**: Apply configuration changes

#### `POST /set_restart_required`

- **Purpose**: Sets restart required flag
- **Response**: Success message
- **Use**: Mark system for restart

#### `GET /get_restart_required`

- **Purpose**: Gets restart required status
- **Response**: JSON boolean status
- **Use**: Check if restart is needed

#### `POST /shutdown`

- **Purpose**: Shuts down the web server
- **Response**: Success message before shutdown
- **Use**: System shutdown

### Real-time Data (SSE)

#### `GET /sse/stream`

- **Purpose**: Server-Sent Events stream
- **Response**: Event stream with real-time updates
- **Events**: `heartbeat`, `update_robot_transform`, `update_detected_objects`, `log_update`, `system_update_progress`, `pipeline_operation_errors`, `profiling_update`
- **Use**: Real-time data subscription

#### `GET /system-update/status`

- **Purpose**: Reports whether WiFi + internet allow a system update
- **Response**: `{ available, reason }`

#### `POST /system-update/run`

- **Purpose**: Starts git pull / apt update / apt upgrade in a background thread
- **Response**: `202` with `{ started: true }` when accepted; progress streams over SSE `system_update_progress`
- **Use**: System update from the WebUI
