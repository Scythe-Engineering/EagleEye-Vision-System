# Pipeline Flowchart Interface

## Overview

The Pipeline Flowchart Interface is a modern, visual pipeline creation and editing system that replaces the traditional list-based approach with an interactive canvas-based editor. This interface allows users to visually construct complex data processing pipelines by dragging operations onto a grid and connecting them in a flowchart-style layout.

## Key Features

### Visual Canvas
- **Grid-based Layout**: 4000x4000px canvas with a subtle grid background for organized operation placement
- **Scrollable Area**: Large workspace that allows for complex pipeline layouts beyond the initial viewport
- **Minimap**: Small overview map (`flowchartMinimap.js`) synced with nodes, connections, and viewport for quick navigation
- **Clean Interface**: Minimal UI with no zoom/pan controls, focusing on simplicity and ease of use

### Drag-and-Drop Operations
- **Operation Panel**: Available operations displayed in a side panel with drag handles
- **Grid Snapping**: Operations automatically snap to the grid when dropped for consistent alignment
- **Visual Feedback**: Clear drag states and drop zones for intuitive interaction

### Node-based Architecture
- **Operation Nodes**: Each operation displays as a visually distinct node with:
  - Operation name and category (DET, LOC, PROC, etc.)
  - Input/output ports based on operation configuration
  - Settings button for configuration
  - Remove button for deletion
- **Port Visualization**: Future enhancement to show data flow connections between nodes

## Technical Implementation

### Core Components

#### FlowchartCanvas (`flowchartCanvas.js`)
- **Purpose**: Manages the scrollable canvas area with grid background
- **Key Methods**:
  - `screenToWorld()`: Converts screen coordinates to canvas coordinates accounting for scroll position
  - `snapPositionToGrid()`: Snaps positions to the nearest grid intersection
  - `getNodesLayer()`: Returns the layer where operation nodes are rendered
  - `getConnectionsLayer()`: Returns the SVG layer for connection lines (future use)

#### FlowchartRenderer (`rendering.js`)
- **Purpose**: Orchestrates the rendering of the entire flowchart interface
- **Key Responsibilities**:
  - Initializes the canvas, connections manager, minimap, and drop zone handling
  - Renders operation nodes at their specified positions
  - Manages node lifecycle (creation, updates, removal)
  - Handles drop events from the operations panel
  - Updates placeholder text based on pipeline state

#### FlowchartNode (`flowchartNode.js`)
- **Purpose**: Represents individual operation nodes on the canvas
- **Features**:
  - Dynamic port configuration based on operation's `input_nodes` and `output_nodes`
  - Drag-and-drop repositioning with grid snapping
  - Interactive buttons for settings and removal
  - Hover effects and visual feedback
  - Asynchronous config data loading from backend

#### FlowchartConnections (`flowchartConnections.js`)
- **Purpose**: Manages SVG-based connection lines between nodes
- **Current Status**: Framework in place for future connection visualization
- **Planned Features**: Bezier curves with data type labels connecting node ports

### Coordinate System

The interface uses a simplified coordinate system:
- **Screen Coordinates**: Raw mouse/touch coordinates
- **Canvas Coordinates**: Screen coordinates adjusted for scroll position
- **Node Positioning**: Absolute positioning within the canvas, snapped to grid

### Data Flow

1. **Operation Selection**: User drags an operation from the operations panel
2. **Coordinate Conversion**: Mouse position converted from screen to canvas coordinates
3. **Grid Snapping**: Position snapped to nearest grid intersection
4. **Node Creation**: New FlowchartNode created with operation data and position
5. **Pipeline Update**: Node added to pipeline state and saved to backend
6. **UI Update**: Canvas re-renders to show new node

## Configuration Integration

### Operation Config Data
Each operation node loads its configuration data from `/get-operation-config-data/{operationId}/{isSecondary}`:
```json
{
  "input_nodes": ["data"],
  "output_nodes": ["data"]
}
```

This data determines:
- Number of input ports displayed on the left side of the node
- Number of output ports displayed on the right side of the node
- Future connection validation between compatible data types

### Pipeline Persistence
- **Real-time Saving**: Pipeline changes automatically saved to backend
- **Position Storage**: Node positions stored alongside operation configuration
- **Restart Detection**: System detects when pipeline changes require backend restart

## Usage Guide

### Creating a Pipeline
1. Select a camera from the dropdown
2. Create a new pipeline or select an existing one
3. Drag operations from the operations panel onto the canvas
4. Position nodes by dragging them around the grid
5. Configure operation settings by clicking the settings button on each node
6. Remove operations by clicking the remove button on each node

### Best Practices
- **Layout Planning**: Arrange nodes logically from left to right (data flow)
- **Grid Alignment**: Use grid snapping for consistent, professional layouts
- **Node Spacing**: Leave adequate space between nodes for future connection lines
- **Scrollable Area**: Utilize the full canvas area for complex pipelines

## Future Enhancements

### Planned Features
- **Connection Lines**: Visual bezier curves connecting node ports
- **Data Type Labels**: Labels on connection lines indicating data flow types
- **Connection Validation**: Prevent invalid connections between incompatible ports
- **Auto-layout**: Automatic node positioning algorithms
- **Node Grouping**: Visual grouping of related operations
- **Search/Filter**: Quick operation finding in large operation sets

### Performance Considerations
- **Large Canvas**: 4000x4000px area allows for complex pipelines without performance issues
- **Lazy Loading**: Operation config data loaded on-demand
- **Efficient Rendering**: DOM-based rendering with CSS optimizations

## Integration Points

### Backend Communication
- **Operation List**: Fetched from `/get-available-operations`
- **Config Data**: Retrieved from `/get-operation-config-data/{id}/{secondary}`
- **Pipeline Saving**: POST to `/save-pipeline-config/{camera}/{pipeline}`
- **Camera Management**: Camera selection and pipeline association

### Frontend Architecture
- **Modular Design**: Separate concerns across multiple JavaScript modules
- **Event-driven**: Loose coupling through event callbacks
- **State Management**: Pipeline state synchronized with backend
- **Error Handling**: Graceful degradation and user feedback

## Troubleshooting

### Common Issues
- **Nodes not appearing**: Check console for JavaScript errors, verify operation config data loading
- **Dragging not working**: Ensure pipeline is selected, check event listener setup
- **Positioning issues**: Verify coordinate conversion and grid snapping logic
- **Performance problems**: Check for excessive DOM operations or memory leaks

### Debug Information
- **Console Logging**: Extensive logging throughout the drag/drop and rendering pipeline
- **Coordinate Tracking**: Mouse positions and conversions logged during interactions
- **State Inspection**: Pipeline state and node positions available in browser dev tools

## File Structure

```
src/webui/js/pipeline/
├── flowchartCanvas.js        # Core canvas management
├── flowchartNode.js          # Individual node components
├── flowchartConnections.js   # Connection line management (future)
├── flowchartMinimap.js       # Minimap overview and viewport indicator
├── interactiveGrid.js        # Grid snapping / interaction helpers used by the canvas
├── rendering.js              # Main flowchart renderer
├── pipelineCreator.js        # Pipeline state management
├── dragDrop.js               # Drag-and-drop utilities
└── utils.js                  # Shared utilities
```