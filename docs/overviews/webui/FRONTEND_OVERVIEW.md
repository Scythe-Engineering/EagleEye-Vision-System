# Frontend Components Overview

## Main Entry Point (`main.js`)

### Core Functionality

The main JavaScript file that initializes the entire frontend application.

#### Key Features

- **Module Imports**: Imports all major frontend components
- **Application Initialization**: Sets up all UI components on page load
- **SocketIO Client**: Manages real-time communication with the backend
- **Coordinate Transformation**: Converts robot position data to field coordinates
- **Connection Management**: Handles connection status and error recovery

#### Coordinate System Conversion

```javascript
const convertDataToFieldSpace = (data) => {
    // Converts robot transform matrix to field coordinate system
    // Handles mm to meter conversion and axis transformations
};
```

#### SocketIO Event Handlers

- `connect`: Logs connection and hides connection lost overlay
- `disconnect`: Shows connection lost overlay
- `connect_error`: Handles connection errors
- `reconnect`: Reloads page on successful reconnection
- `update_robot_transform`: Updates robot position in 3D view

## JavaScript Module Structure

### Pipeline Module (`js/pipeline/`)

Handles visual pipeline creation and configuration.

#### Files

- **`pipelineCreator.js`**: Main pipeline creation interface (875 lines)

    - Profiling timestep panel: one circular badge per timestep with the timestep index in the center; the ring uses equal-angle slices (one per distinct thread on that timestep), colored by thread id, not weighted by runtime. The adjacent value is the summed `execution_time_ms` for that timestep (tooltip may include wall-clock duration when it differs).
    - Visual drag-and-drop pipeline builder
    - Pipeline node management
    - Settings integration
    - Save/load pipeline configurations

- **`settingsPopup.js`**: Pipeline node settings configuration (666 lines)

    - Dynamic settings forms
    - Parameter validation
    - Settings persistence
    - UI state management

- **`dragDrop.js`**: Sets drag payload for the operations panel so the flowchart canvas can add operations on drop.

    - The canvas (`FlowchartRenderer`) owns drop targets, pan/zoom, and connection UI; there is no separate list-based pipeline view.

- **`rendering.js`**: Pipeline visualization rendering (213 lines)

    - Canvas-based rendering
    - Node positioning and layout
    - Connection line drawing
    - Removing a flowchart edge calls through to `PipelineStore` so drag-to-disconnect matches auto-save
    - Adding or removing an operation updates the flowchart incrementally (no full node tear-down) and only recenters when the first node is placed or on a full load/switch via `renderPipeline` with `centerView`
    - Visual state updates

- **`utils.js`**: Pipeline utility functions (47 lines)
    - Helper functions for pipeline operations
    - Data validation and formatting
    - Common pipeline calculations

### Settings Module (`js/settings/`)

Manages application settings and configuration.

#### Files

- **`restartBackend.js`**: Backend restart functionality (30 lines)

    - REST API calls to restart backend
    - User confirmation dialogs
    - Error handling for restart operations

- **`loadSettings.js`**: Settings loading and display (39 lines)

    - Fetches settings from backend
    - Populates UI with current values
    - Handles settings data parsing

- **`saveSettings.js`**: Settings persistence (64 lines)
    - Validates user input
    - Sends settings to backend via API
    - Success/error feedback to user

### UI Components (`js/ui/`)

User interface management and navigation.

#### Files

- **`sidebar.js`**: Sidebar navigation and tab management (97 lines)
    - Tab switching logic
    - Active state management
    - Content area updates
    - Navigation event handling

### Camera Feed Management (`js/feeds/`)

Handles camera stream display and management.

#### Files

- **`cameraFeedHandlers.js`**: Camera feed operations (148 lines)
    - Camera stream initialization
    - Feed switching and management
    - Stream error handling
    - MJPEG stream processing

### Dropdown Components (`js/dropdown/`)

Selection dropdown management.

#### Files

- **`robotDropdown.js`**: Robot selection dropdown (49 lines)

    - Robot model selection
    - 3D model loading
    - Selection state management

- **`fieldDropdown.js`**: Field selection dropdown (46 lines)
    - Field layout selection
    - 3D field model loading
    - Coordinate system updates

### 3D Visualization (`init3DView.js`)

Three.js-based 3D rendering system (462 lines).

#### Key Features

- **Three.js Scene Setup**: Camera, renderer, lighting configuration
- **Robot Model Loading**: GLTF/GLB model loading and positioning
- **Field Rendering**: Game field and AprilTag visualization
- **Real-time Updates**: Robot position and orientation updates
- **Coordinate Systems**: Field and robot coordinate space management

#### Core Functions

- `updateRobotTransform()`: Updates robot position from SocketIO data
- Scene initialization and rendering loop
- Camera controls and viewport management

#### Rotation Conventions

The 3D view keeps the robot and camera markers aligned by applying the same
shared transform chain in `init3DView.js`:

```javascript
matrix
    .multiply(robotPitchRollSwap)
    .multiply(visualOrientationMatrix)
    .multiply(extraVisualRollMatrix)
    .multiply(robotPitchRollSwapInverse);
```

- `robotPitchRollSwap`: switches the incoming pose into the robot/Three.js
  visual basis
- `visualOrientationMatrix`: the fixed `-90°` X rotation used by the scene
- `extraVisualRollMatrix`: the additional `-90°` Z rotation

Keep the basis swap separate from the roll corrections. In this coordinate
setup, folding the extra roll into the X rotation makes it behave like a yaw
instead of a roll.

## HTML Structure

### Main Template (`index.html`)

The root HTML template with the application layout.

#### Structure

- **Connection Lost Overlay**: Displays when backend connection is lost
- **Main Container**: Application layout container
- **Header**: "EAGLE EYE" title bar
- **Content Area**: Split layout with left content and right sidebar
- **Sidebar**: Navigation tabs (Views, 3D View, Pipeline, Settings)

#### Template System

Uses Handlebars-style partials for modular content:

```
{{> camera_views_tab_content}}
{{> 3d_tab_content}}
{{> pipeline_tab_content}}
{{> settings_tab_content}}
```

### HTML Templates (`html/tabs/`)

Content templates for different application sections.

#### Files

- **`pipeline_tab_content.html`**: Pipeline creation interface (293 lines)

    - Visual pipeline builder canvas
    - Node palette and settings panels
    - Drag-and-drop zones
    - Pipeline configuration forms

- **`settings_tab_content.html`**: Settings management interface (108 lines)

    - Settings categories and forms
    - Input validation and feedback
    - Save/cancel controls
    - Settings preview

- **`3d_tab_content.html`**: 3D visualization interface (57 lines)

    - Three.js canvas container
    - Camera controls
    - Model loading status
    - Viewport controls

- **`camera_views_tab_content.html`**: Camera feed display (14 lines)
    - Camera stream containers
    - Feed selection controls
    - Stream status indicators

### HTML Partials (`html/partials/`)

Reusable HTML components.

#### Files

- **`file_upload.html`**: File upload component (14 lines)

    - Drag-and-drop file upload
    - File type validation
    - Upload progress feedback

- **`toast_success.html`**: Success notification (47 lines)

    - Success message display
    - Auto-dismiss functionality
    - Styling and animations

- **`toast_warning.html`**: Warning notification (47 lines)

    - Warning message display
    - User acknowledgment
    - Dismissible notifications

- **`toast_danger.html`**: Error notification (47 lines)
    - Error message display
    - Critical alert styling
    - Action buttons for error resolution

## CSS Styling

### Global Styles (`style.css`)

Main stylesheet with Tailwind CSS utilities and custom styles.

### Component Styles (`css/`)

Specific styling for UI components.

#### Files

- **`sidebar.css`**: Sidebar navigation styling (30 lines)

    - Tab appearance and states
    - Hover and active effects
    - Layout and positioning

- **`camera.css`**: Camera feed styling (46 lines)
    - Video stream containers
    - Feed layout and sizing
    - Status indicators
    - Responsive design

## Static Assets

### Bundle Files (`static/`)

Compiled and optimized static files for production. The Flask server maps them to stable URLs: `bundle.js` is served as **`/js/main.js`**, and `main.css` is served as **`/style.css`**.

#### Files

- **`bundle.js`**: Compiled JavaScript bundle (1.3MB)

    - All JavaScript modules combined
    - Minified for production
    - Dependency management

- **`index.html`**: Static HTML file (299 lines)

    - Self-contained HTML
    - Inline critical CSS/JS
    - Optimized for loading

- **`main.css`**: Compiled CSS (23KB)

    - All styles combined
    - Minified and optimized
    - Critical CSS inlined

- **`background.webp`**: Background image (53KB)
    - Application background
    - Optimized for web delivery

## Data Flow

### Initialization Flow

1. Page loads → `main.js` executes
2. Module imports and initialization
3. SocketIO connection established
4. UI components set up (sidebar, dropdowns, feeds)
5. Settings loaded from backend

### Real-time Updates

1. SocketIO receives robot position data
2. Coordinate transformation applied
3. 3D scene updated with new position
4. UI reflects current state

### User Interactions

1. User clicks sidebar tab
2. Content area updates with new template
3. JavaScript modules handle specific functionality
4. API calls made for data operations
5. UI updates based on responses

## Browser Compatibility

### Supported Features

- ES6 Modules (modern browsers)
- WebGL for 3D rendering
- WebSockets for real-time communication
- MJPEG streaming support
- Modern CSS features

### Fallbacks

- Connection retry logic
- Error overlays for connection issues
- Graceful degradation for missing features
