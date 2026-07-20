# Backend Components Overview

## Web Server (`web_server.py`)

### Class: EagleEyeInterface
The main Flask application class that provides the web interface for the EagleEye system.

#### Key Features
- **Flask Web Server**: Serves the web interface and API endpoints
- **SocketIO Integration**: Real-time communication for robot position updates
- **Camera Management**: Handles multiple camera feeds and streaming
- **Settings Management**: Persists and retrieves application settings
- **Static File Serving**: Serves assets, JavaScript, CSS, and other static files

#### Initialization Parameters
- `restart_callback`: Function to restart the backend system
- `pipeline_objects_callback`: Function to get pipeline objects
- `settings_object`: Optional Constants object for settings
- `dev_mode`: Boolean flag for development mode
- `log`: Optional logging function

#### Core Methods

##### Camera Operations
- `add_camera(name, camera_object)`: Adds a camera to the system
- `remove_camera(name)`: Removes a camera from the system
- `get_camera_feed(camera_name)`: Returns MJPEG stream for camera
- `get_available_cameras()`: Returns list of available cameras

##### Settings Management
- `get_settings()`: Retrieves current application settings
- `save_settings()`: Updates application settings from JSON payload
- `restart_backend()`: Triggers backend restart via callback

##### Real-time Communication
- SocketIO events for robot transform updates
- Connection management and error handling
- Frame broadcasting to connected clients

## Web Server Utils

### `serve_static_files.py`
Utility module for serving static files with proper MIME types.

#### Functions
- `serve_index()`: Serves the main index.html
- `serve_js()`: Serves JavaScript files
- `serve_css()`: Serves CSS files

### Draco Loader (`drako_loader/`)
Handles 3D model compression and decompression for web delivery.

#### Components
- **draco_wasm_wrapper.js**: WebAssembly wrapper for Draco operations
- **draco_encoder.js**: Draco compression functionality
- **draco_decoder.js**: Draco decompression functionality
- **draco_decoder.wasm**: WebAssembly binary for decoding
- **rhino3dm/**: Rhino 3D model utilities
- **gltf/**: GLTF format handling utilities

#### Purpose
- Reduces 3D model file sizes for faster web loading
- Enables real-time 3D visualization in the browser
- Supports GLTF/GLB format conversion and optimization

## API Endpoints

### Static File Endpoints
- `GET /` - Main web interface
- `GET /js/main.js` - Main JavaScript bundle (serves Vite `bundle.js` from static output)
- `GET /style.css` - Main CSS stylesheet (serves built `main.css` from static output)
- `GET /background.webp` - Background image
- `GET /favicon.ico` - Favicon
- `GET /frc2025r2.json` - FRC 2025 AprilTag configuration
- `GET /src/webui/assets/apriltags/<filename>` - AprilTag images

### Settings Endpoints
- `GET /get-general-conf` - Read `general_conf.json` merged with defaults
- `POST /save-general-conf` - Merge JSON into general configuration (`network_table_address`, `view_stream_downscale`, etc.)

### Camera Endpoints
- `GET /get-available-cameras` - List available cameras
- `GET /camera/<camera_name>` - MJPEG stream for specific camera
- `GET /camera/<camera_name>/snapshot` - Single camera frame

### Utility Endpoints
- `POST /restart-backend` - Trigger backend restart
- `GET /get-pipeline-objects` - Retrieve pipeline configuration

## Realtime updates

### Server-Sent Events (`GET /sse/stream`)
- **Primary WebUI channel:** `EventSource` in the browser for heartbeats, `update_robot_transform`, `update_camera_pose`, `update_detected_objects`, `log_update`, `profiling_update`, `pipeline_operation_errors`, and related payloads.
- **Multi-client:** Each browser connection gets its own SSE queue. Events are fan-out published to every connected subscriber so multiple viewers can share one live demo session.

### Demo / read-only mode
- Enabled by `"demo_mode": true` in `src/general_conf.json`, or by env `EAGLEEYE_DEMO_MODE=1` (env overrides config).
- Mutating HTTP methods (`POST`/`PUT`/`DELETE`) return `403`, except ephemeral viewing helpers (`get_operation_config_data_batch`, `start_visualize`, `stop_visualize`).
- Frontend hides save/edit controls and opens operation settings as view-only.
- Simulated cameras come from `src/utils/sim_videos/*.mp4` (drop an MP4 named to match the pipeline `camera_bus_id`, then restart).

### Socket.IO (optional)
- Flask-SocketIO runs the server; some tooling may attach a Socket.IO client (for example `profiling_update` when `globalThis.socket` is present).
- **Server to client examples:** `update_robot_transform`, connection lifecycle, errors (when using a Socket.IO client rather than SSE).

## Data Flow

### Camera Pipeline
1. Camera objects register with the web server
2. Frames are captured and stored in frame_list
3. MJPEG streams are generated on-demand
4. SocketIO broadcasts position updates

### Settings Flow
1. Frontend requests current settings via HTTP GET
2. Backend retrieves settings from Constants object
3. Frontend displays settings in UI
4. User modifies settings and saves via HTTP POST
5. Backend updates Constants object and persists changes

### Real-time Updates
1. Backend receives robot position data
2. Position data is broadcast via SocketIO
3. Frontend receives updates and updates 3D visualization
4. UI reflects current robot state in real-time

## Error Handling

### HTTP Error Responses
- 200: Success responses with appropriate data
- 404: Camera not found, Pipeline not found
- 500: Internal server error during settings save

### Pipeline Error Handling
- Returns empty arrays for cameras without pipelines (graceful degradation)
- Automatically creates camera/pipeline entries when saving new pipelines
- Validates pipeline existence before deletion
- Safe pipeline visualization for non-existent pipelines

### SocketIO Error Handling
- Connection timeouts (60s ping timeout, 25s ping interval)
- Reconnection logic with exponential backoff
- Error logging and user notifications

### Logging
- Comprehensive logging of camera operations
- Settings changes logging
- Error conditions and stack traces
- Connection status monitoring
- Pipeline configuration changes

## Threading Model

### Main Thread
- Flask application initialization
- Route registration
- Settings object management

### Camera Threads
- Frame capture and processing
- MJPEG stream generation
- Frame buffer management with thread locks

### SocketIO Thread
- Real-time communication handling
- Event broadcasting
- Connection management

## Configuration

### Development Mode
- Runs Flask in debug mode
- Synchronous execution for easier debugging
- CORS configured for localhost:5173

### Production Mode
- Daemon thread execution
- Host binding to 0.0.0.0:5001
- Asynchronous SocketIO operation
- Error handling with logging
