# Assets Overview

## Directory Structure

Static assets are organized by type and purpose. Web interface assets live under
`assets/`, while pipeline replay assets live beside the managed simulation
videos.

```
assets/
├── robots/                    # Robot 3D models
├── apriltags/                 # AprilTag fiducial markers
├── fields/                    # Field layouts and game pieces
├── delete.svg                 # Delete icon
├── settings.svg               # Settings icon
├── favicon.ico                # Browser favicon
├── no_image.png               # Placeholder image
└── background.webp            # Application background
src/utils/sim_videos/
├── benchmark_manifest.json    # Benchmark replay metadata and thresholds
├── *_data.csv                 # Ground-truth replay annotations
└── *.mp4                      # Managed benchmark videos
```

## 3D Models (Robots)

### Location: `assets/robots/`
Contains 3D models of robots for visualization in the 3D view.

#### Files
- **`3322-24-25-DRACO.glb`** (9.4MB)
  - Team 3322 robot model for the 2024-2025 season
  - Draco-compressed GLTF format for web optimization
  - Includes robot geometry, materials, and textures

#### Purpose
- **3D Visualization**: Display robot position and orientation in the field
- **Real-time Tracking**: Update robot pose based on localization data
- **Collision Detection**: Visual representation for path planning
- **Team Identification**: Unique robot models for multi-robot scenarios

#### Technical Details
- **Format**: GLTF/GLB (Binary GLTF)
- **Compression**: Draco geometry compression for smaller file sizes
- **Coordinate System**: Aligned with FRC field coordinate system
- **Scale**: Real-world dimensions in millimeters

## AprilTags

### Location: `assets/apriltags/`
Contains AprilTag fiducial markers used for robot localization.

#### Tag Family: 36_11
- **Resolution**: 6x6 grid with 2-bit border
- **Family Size**: 36h11 - 2^11 = 2048 possible unique tags
- **Current Tags**: 00000 through 00040 (41 tags)

#### Individual Tag Files
- **Format**: PNG images
- **Resolution**: 64x64 pixels (typical)
- **Color**: Black tags on white background
- **Naming**: `tag36_11_XXXXX.png` where XXXXX is the tag ID

#### Usage
- **Localization**: Robot position estimation using computer vision
- **Field Calibration**: Known positions for coordinate system alignment
- **Navigation**: Waypoint definition and path following
- **Multi-Robot Coordination**: Unique identification for multiple robots

#### Technical Specifications
- **Detection**: Robust to lighting variations and partial occlusion
- **Orientation**: Can determine tag rotation and distance
- **Precision**: Sub-millimeter accuracy at close range
- **Processing**: Real-time detection at 30+ FPS

## Field Assets

### Location: `assets/fields/2025/`
Contains 3D models and assets for the 2025 FRC game field.

#### Field Files (`field_files/`)
- **`FE-2025-NGP.glb`** (71MB) - Full field model
  - Complete game field layout
  - All game elements and structures
  - High detail for accurate visualization

- **`FE-2025-NGP-Simple.glb`** (14MB) - Simplified field model
  - Reduced polygon count for better performance
  - Essential field elements only
  - Optimized for real-time rendering

#### Game Pieces (`game_pieces/`)
- **`FE-2025-GP.glb`** (17MB) - Game pieces collection
  - All movable game elements
  - Multiple instances of each piece type
  - Physics-ready geometry

- **`Algea.glb`** (648KB) - Algae game piece
  - Individual algae element model
  - Realistic geometry and materials
  - Optimized for instancing

- **`Coral.glb`** (13KB) - Coral game piece
  - Individual coral element model
  - Detailed geometry for scoring simulation

#### Purpose
- **Field Visualization**: Accurate 3D representation of the game environment
- **Game Simulation**: Physics-based interaction with game pieces
- **Strategy Planning**: Visual analysis of field layout and scoring positions
- **Robot Testing**: Virtual testing environment before physical prototyping

#### Technical Details
- **Format**: GLTF/GLB for web compatibility
- **Scale**: 1:1 real-world dimensions
- **Coordinate System**: FRC standard field coordinates
- **Materials**: PBR materials with textures and normal maps

## UI Assets

### Icons and Images
- **`delete.svg`** (23KB) - Delete action icon
  - Vector format for crisp rendering at all sizes
  - Consistent with application design language

- **`settings.svg`** (5.9KB) - Settings/gear icon
  - Used in settings panels and buttons
  - Scalable vector graphics

### Application Assets
- **`favicon.ico`** (205KB) - Browser tab icon
  - Multiple sizes for different devices
  - Brand identification in browser tabs

- **`no_image.png`** (1.5MB) - Placeholder/fallback image
  - Used when camera feeds are unavailable
  - Consistent with application color scheme

- **`background.webp`** (53KB) - Application background
  - Subtle texture/pattern for visual interest
  - Optimized for web delivery

## Asset Management

### Benchmark Replay Assets
- **Location**: `src/utils/sim_videos/`
- **Video Format**: MP4 files named after the camera bus ID, such as `basic_test.mp4`
- **Annotations**: Ground-truth CSV files kept beside the video, such as `basic_test_data.csv`
- **Manifest**: `benchmark_manifest.json` maps each video and annotation file to the pipeline, camera, and accuracy thresholds used by pytest replay coverage

### Loading Strategy
- **Lazy Loading**: 3D models loaded on-demand
- **Compression**: Draco compression for geometry optimization
- **Caching**: Browser caching for improved performance
- **Fallbacks**: Placeholder images for missing assets

### Performance Optimization
- **File Size**: Models compressed for web delivery
- **LOD System**: Multiple detail levels (full/simple models)
- **Texture Compression**: Optimized textures for web rendering
- **Progressive Loading**: Essential assets load first

### Version Control
- **Season-Specific**: Assets organized by FRC season/year
- **Modular Updates**: Individual assets can be updated independently
- **Backup Compatibility**: Fallback to previous versions if needed

## Integration Points

### 3D Visualization System
- Assets loaded by Three.js in `init3DView.js`
- Coordinate transformations applied for proper positioning
- Real-time updates from SocketIO position data

### Camera System
- AprilTags detected by computer vision pipelines
- Position data used to update robot models
- Field calibration using known tag positions

### Pipeline Configuration
- Assets referenced in pipeline settings
- Model selection through dropdown interfaces
- Dynamic loading based on user selection

## File Formats

### 3D Models
- **GLTF/GLB**: Primary format for web delivery
- **Draco Compression**: Geometry compression for smaller files
- **PBR Materials**: Physically-based rendering materials

### Images
- **PNG**: Lossless compression for UI elements and placeholders
- **SVG**: Vector format for icons and scalable graphics
- **ICO**: Multi-resolution favicon format

### Data Files
- **JSON**: Configuration files for AprilTag layouts
- **GLTF**: Text-based 3D model format for development

## Maintenance

### Asset Updates
- Regular updates for new game seasons
- Model optimization for performance improvements
- Bug fixes for rendering issues
- Compatibility updates for new browsers

### Quality Assurance
- Visual inspection of 3D models
- Performance testing on target devices
- Compatibility testing across browsers
- File size optimization checks

### Documentation
- Asset specifications and requirements
- Usage guidelines for developers
- Performance benchmarks
- Update procedures for maintainers
