# EagleEye WebUI Overview

## Overview

The EagleEye WebUI is a comprehensive web-based interface for FIRST Robotics Competition (FRC) object detection and robot tracking systems. It provides real-time camera feeds, 3D visualization, pipeline configuration, and settings management through a modern web interface.

## Architecture

### Backend (Flask + SocketIO)

- **Framework**: Flask web server with SocketIO for real-time communication
- **Port**: 5001
- **Features**: Camera streaming, settings management, robot position tracking

### Frontend (Vanilla JS + HTML + CSS)

- **Structure**: Modular JavaScript architecture with ES6 modules
- **Styling**: Tailwind CSS classes with custom styling
- **Real-time**: SocketIO client for live updates
- **3D Visualization**: Three.js for robot and field rendering

## Main Components

### 1. Web Server (`web_server.py`)

- Flask application serving the web interface
- SocketIO integration for real-time communication
- Camera feed management and streaming
- Settings persistence and retrieval
- Static file serving

### 2. Frontend Structure

- **HTML**: Template-based structure with Handlebars-style partials
- **JavaScript**: Modular architecture with separate concerns
- **CSS**: Component-specific styling with Tailwind utilities

### 3. Assets

- **3D Models**: Robot models (.glb), field layouts, game pieces
- **AprilTags**: Fiducial markers for localization
- **Images**: Background images, icons, UI elements

### 4. Configuration

- **Pipeline Config**: JSON-based pipeline configuration
- **Settings**: Application settings management
- **Camera Config**: Camera calibration and setup

## Key Features

### Real-time Camera Streaming

- Multiple camera feed support
- MJPEG streaming for low-latency video
- Camera discovery and management

### 3D Visualization

- Robot position tracking
- Field layout visualization
- Game piece rendering
- AprilTag-based localization

### Pipeline Management

- **Flowchart-style Visual Editor**: Interactive canvas-based pipeline creation with grid-based placement
- **Drag-and-Drop Operations**: Drag operations from the operations panel onto the flowchart canvas
- **Grid-Snapping**: Operations snap to a grid for organized layout and precise positioning
- **Visual Node Interface**: Each operation displays as a node with input/output ports based on configuration data
- **Connection Visualization**: SVG-based curved lines connecting operation nodes (future enhancement)
- **Scrollable Canvas**: Large 4000x4000px canvas area for complex pipeline layouts
- **Automatic Pipeline Saving**: Real-time saving on all structure changes (add, remove, reposition)
- **Settings Configuration**: Click nodes to configure operation parameters
- **Backend State Monitoring**: Automatic detection of restart requirements after pipeline changes
- **Graceful Handling of Cameras**: Cameras without pipelines display appropriately (see [Pipeline Error Handling](PIPELINE_ERROR_HANDLING.md))
- **Dynamic Pipeline Creation**: Automatic creation of camera/pipeline entries when saving new pipelines
- **Detailed Documentation**: Comprehensive flowchart interface guide (see [Pipeline Flowchart Interface](PIPELINE_FLOWCHART_INTERFACE.md))

### Settings Management

- Application configuration
- Camera calibration settings
- System preferences

### Backend State Monitoring

- Automatic detection of backend restart requirements
- System time comparison between frontend and backend
- Visual restart indicator with custom messaging
- Periodic state checking (every 30 seconds)
- Integration with pipeline configuration changes

## Directory Structure

```
webui/
├── web_server.py              # Main Flask application
├── index.html                  # Main HTML template
├── style.css                   # Global styles
├── API_DOCUMENTATION.md        # API documentation
├── assets/                     # Static assets
│   ├── robots/                # Robot 3D models
│   ├── apriltags/             # AprilTag images
│   ├── fields/                # Field layouts and game pieces
│   ├── *.png/svg              # UI images and icons
├── js/                        # Frontend JavaScript
│   ├── main.js                # Application entry point
│   ├── init3DView.js          # 3D visualization setup
│   ├── pipeline/              # Pipeline management
│   │   ├── flowchartCanvas.js    # Canvas and grid management
│   │   ├── flowchartNode.js      # Operation node components
│   │   ├── flowchartConnections.js # Connection visualization
│   │   ├── rendering.js          # Flowchart renderer
│   │   ├── pipelineCreator.js    # Pipeline state management
│   │   └── dragDrop.js           # Drag-and-drop utilities
│   ├── settings/              # Settings management
│   ├── ui/                    # UI components
│   ├── feeds/                 # Camera feed handling
│   └── dropdown/              # Dropdown components
├── html/                      # HTML templates and partials
│   ├── tabs/                  # Tab content templates
│   └── partials/              # Reusable HTML components
├── css/                       # Component-specific styles
├── static/                    # Built static files
└── web_server_utils/         # Server utilities
    ├── serve_static_files.py  # Static file serving
    └── drako_loader/          # 3D model compression
```

## Technology Stack

### Backend

- **Python**: Core application logic
- **Flask**: Web framework
- **Server-Sent Events (SSE)**: Real-time communication
- **Flask-CORS**: Cross-origin resource sharing
- **OpenCV**: Camera processing and streaming
- **NumPy**: Numerical computations

### Frontend

- **JavaScript (ES6)**: Client-side logic
- **Three.js**: 3D visualization
- **EventSource API**: Real-time communication (SSE)
- **Tailwind CSS**: Utility-first styling
- **HTML5**: Semantic markup

### Assets & Data

- **GLTF/GLB**: 3D model formats
- **PNG/SVG**: Image formats
- **JSON**: Configuration and settings
- **AprilTags**: Fiducial marker system

## Development

### Local Development

- Run Flask server on port 5001
- Hot reload for development mode
- CORS configured for localhost:5173

### Production

- Daemon thread execution
- Error handling and logging
- Static file optimization
- Build process required for pipeline tab inclusion (`npm run build`)

## Integration Points

### Main Application

- Camera feed providers
- Object detection pipelines
- Robot localization systems
- Settings persistence layer

### External Systems

- FIRST Robotics NetworkTables
- Camera hardware interfaces
- 3D model repositories
- Configuration management systems
