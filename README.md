[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Scythe-Engineering_EagleEye-Object-Detection&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=Scythe-Engineering_EagleEye-Object-Detection) [![Bugs](https://sonarcloud.io/api/project_badges/measure?project=Scythe-Engineering_EagleEye-Object-Detection&metric=bugs)](https://sonarcloud.io/summary/new_code?id=Scythe-Engineering_EagleEye-Object-Detection) [![Duplicated Lines (%)](https://sonarcloud.io/api/project_badges/measure?project=Scythe-Engineering_EagleEye-Object-Detection&metric=duplicated_lines_density)](https://sonarcloud.io/summary/new_code?id=Scythe-Engineering_EagleEye-Object-Detection) 

# EagleEye Object Detection System

**EagleEye** is a comprehensive computer vision system designed for FIRST Robotics Competition (FRC) applications, featuring AI-accelerated AprilTag detection, dynamic processing pipelines, and real-time robot localization. Built by FRC Team 3322 Eagle Evolution, this system provides robust object detection and pose estimation capabilities for autonomous robotics.

## 🚀 Key Features

### 🤖 AI-Accelerated AprilTag Detection
- **CNN-Based Preprocessing**: Custom convolutional neural network for rapid AprilTag region detection
- **Grid-Based Prediction**: Efficient grid-based approach to identify potential AprilTag locations
- **Multi-Device Support**: GPU acceleration with CUDA, CPU fallback, and edge device optimization
- **Real-Time Performance**: Sub-millisecond inference times for live video processing
- **Adaptive Thresholding**: Dynamic confidence thresholds for optimal detection accuracy

### 🔄 Dynamic Processing Pipelines
- **Modular Architecture**: Pluggable pipeline components for flexible processing workflows
- **JSON Configuration**: Declarative pipeline definitions for easy customization
- **Real-Time Switching**: Hot-swappable pipeline configurations without system restart
- **Multi-Camera Support**: Independent pipeline execution per camera with thread management
- **Error Recovery**: Graceful handling of pipeline failures with automatic recovery

### 📊 Real-Time Robot Localization
- **Multi-Camera Fusion**: Combines data from multiple cameras for robust pose estimation
- **Velocity-Based Filtering**: Intelligent filtering to remove erroneous pose estimates
- **NetworkTables Integration**: Seamless communication with FRC robot control systems
- **3D Visualization**: Web-based 3D view of robot position and field elements
- **Pose Smoothing**: Advanced algorithms for stable position tracking

### 🌐 Web-Based Interface
- **Live Camera Feeds**: Real-time video streaming from multiple cameras
- **3D Field Visualization**: Interactive 3D view of robot position and AprilTags
- **Settings Management**: Dynamic configuration updates through web interface
- **Socket.IO Integration**: Real-time updates and bidirectional communication
- **Responsive Design**: Modern UI with Tailwind CSS and Three.js

## 🏗️ Architecture Overview

### Core Components

#### 1. **Pipeline Engine** (`src/config/utils/pipeline.py`)
- Dynamic pipeline execution with configurable operations
- Thread-safe processing with independent camera management
- Hot-reloadable configurations for rapid development

#### 2. **AprilTag Detection System** (`src/main_operations/modules/apriltags/`)
- **Traditional Detection**: `apriltag_detector.py` - Standard AprilTag detection using pupil-apriltags
- **AI Acceleration**: `pre_processing/ai_accelleration/` - CNN-based region proposal
- **Pose Estimation**: `apriltag_pose_estimator.py` - Camera pose calculation from detections

#### 3. **Object Detection** (`src/main_operations/modules/object_detection/`)
- YOLO-based game piece detection
- Multi-device support (GPU/CPU)
- Real-time inference with NetworkTables integration

#### 4. **Web Interface** (`src/webui/`)
- Flask-based web server with Socket.IO
- Real-time video streaming and 3D visualization
- RESTful API for system configuration

### Processing Pipeline Example

```json
{
    "basic_test": [
        {
            "action_name": "detect_apriltags",
            "action_params": {}
        },
        {
            "action_name": "camera_localization",
            "action_params": {
                "camera_parameters_path": "/path/to/intrinsics.json",
                "apriltag_map_path": "/path/to/frc2025r2.json"
            }
        },
        {
            "action_name": "velocity_based_filtering",
            "action_params": {}
        },
        {
            "action_name": "flatten_pose",
            "action_params": {}
        },
        {
            "action_name": "robot_pose_output",
            "action_params": {}
        }
    ]
}
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- OpenCV 4.12+
- PyTorch 2.7+
- CUDA (optional, for GPU acceleration)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Scythe-Engineering/EagleEye-Object-Detection.git
   cd EagleEye-Object-Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   npm install
   ```

3. **Configure cameras**
   - Edit camera configurations in `src/config/`
   - Add camera calibration files to `src/utils/camera_utils/camera_calibrations/`

4. **Start the system**
   ```bash
   python src/main_backend.py
   ```

5. **Access web interface**
   - Open `http://localhost:5001` in your browser
   - View live camera feeds and 3D visualization

## 🔧 Configuration

### Camera Setup
- Add camera configurations to pipeline config files
- Include camera calibration parameters
- Configure AprilTag field maps for localization

### AI Model Configuration
- Place trained CNN models in `src/main_operations/modules/apriltags/pre_processing/ai_accelleration/`
- Configure confidence thresholds and grid parameters
- Set device preferences (GPU/CPU/Edge)

### Pipeline Customization
- Define custom pipelines in JSON format
- Add new operations to `src/main_operations/definitions/` or `src/secondary_operations/`
- Configure operation parameters for your specific use case

## 📈 Performance Features

### AI Acceleration Benefits
- **10x Faster Detection**: CNN preprocessing reduces AprilTag search space
- **Reduced CPU Load**: GPU acceleration for neural network inference
- **Adaptive Processing**: Dynamic region-of-interest selection
- **Multi-Scale Detection**: Handles various AprilTag sizes efficiently

### Dynamic Pipeline Advantages
- **Zero-Downtime Updates**: Hot-swappable configurations
- **Resource Optimization**: Per-camera pipeline optimization
- **Fault Tolerance**: Automatic recovery from pipeline failures
- **Scalable Architecture**: Easy addition of new processing steps

## 🎯 Use Cases

### FRC Applications
- **Autonomous Navigation**: Real-time robot localization using AprilTags
- **Game Piece Detection**: YOLO-based detection of game objects
- **Multi-Camera Systems**: Fusion of multiple camera feeds
- **Edge Computing**: Deployable on embedded systems

### Research & Development
- **Computer Vision Research**: Modular architecture for algorithm testing
- **Robotics Education**: Educational platform for vision systems
- **Prototype Development**: Rapid iteration with dynamic pipelines

## 🤝 Contributing

We welcome contributions to improve EagleEye! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive tests for new features
- Update documentation for API changes
- Ensure backward compatibility

## 📊 Code Quality

We maintain high code quality standards with:
- **SonarCloud Integration**: Automated code quality analysis
- **Type Hints**: Comprehensive type annotations
- **Documentation**: Detailed docstrings and API documentation
- **Testing**: Unit tests for critical components

View our code quality metrics: [SonarCloud Dashboard](https://sonarcloud.io/project/overview?id=Scythe-Engineering_EagleEye-Object-Detection)

## 📄 License

EagleEye Framework © 2025 by (ScytheEngineering/FRC3322) is licensed under CC BY-NC 4.0. See the [LICENSE](LICENSE) file for details.

## 👥 Contributors

- [DarkEden-coding](https://github.com/DarkEden-coding) - Main Contributor & Architect
- FRC Team 3322 Eagle Evolution - Development Team

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the [wiki page](https://github.com/frc3322/EagleEye-Object-Detection/wiki)
- Review the [API documentation](src/webui/API_DOCUMENTATION.md)

---

**Built with ❤️ by FRC Team 3322 Eagle Evolution**

[![Quality gate](https://sonarcloud.io/api/project_badges/quality_gate?project=Scythe-Engineering_EagleEye-Object-Detection)](https://sonarcloud.io/summary/new_code?id=Scythe-Engineering_EagleEye-Object-Detection) [![SonarQube Cloud](https://sonarcloud.io/images/project_badges/sonarcloud-dark.svg)](https://sonarcloud.io/summary/new_code?id=Scythe-Engineering_EagleEye-Object-Detection)
