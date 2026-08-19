# EagleEye Object Detection - Documentation Overviews

This directory contains comprehensive overviews of the major components and systems within the EagleEye Object Detection framework.

## Documentation Structure

### [WebUI System](./webui/)

Complete documentation for the web-based user interface and API:

- **[WebUI Overview](./webui/WEBUI_OVERVIEW.md)** - High-level architecture and features
- **[Backend Overview](./webui/BACKEND_OVERVIEW.md)** - Server-side components and architecture
- **[Frontend Overview](./webui/FRONTEND_OVERVIEW.md)** - Client-side implementation details
- **[API Documentation](./webui/API_DOCUMENTATION.md)** - Complete REST API reference
- **[API Endpoints Summary](./webui/API_ENDPOINTS_SUMMARY.md)** - Quick reference for all endpoints
- **[Assets Overview](./webui/ASSETS_OVERVIEW.md)** - 3D models, images, and static resources
- **[Pipeline Error Handling](./webui/PIPELINE_ERROR_HANDLING.md)** - Error management for camera pipelines

### Inference devices and models

The old mutable compute-pool abstraction has been removed. See the [Object Detection operation](../md_docs/pipeline_docs/main_operations/ObjectDetection.md) for canonical device IDs, managed model artifacts, and deterministic CPU/CUDA resolution.

### [Time Synchronization](./TIME_SYNCHRONIZATION.md)

How frame capture timestamps are produced, propagated through the pipeline, and
published over NetworkTables, plus how robot code should consume them in a
WPILib pose estimator.

### [Pipeline Operations](../md_docs/pipeline_docs/)

Complete documentation for all pipeline operations:

- **[Pipeline Overview](../md_docs/pipeline_docs/PipelineOverview.md)** - Pipeline architecture and configuration
- **[Implement Pipeline Operation](../md_docs/pipeline_docs/ImplementPipelineOperation.md)** - Guidelines for creating new operations
- **[Main Operations](../md_docs/pipeline_docs/main_operations/)** - Primary computer vision operations
- **[Secondary Operations](../md_docs/pipeline_docs/secondary_operations/)** - Post-processing and utility operations

### [Rust Implementations](./rust_implementations/)

High-performance Rust modules for critical vision processing operations:

- **[Overview](./rust_implementations/RUST_MODULES_OVERVIEW.md)** - Rust modules architecture and build system
- **[Pose Outlier Filter](../src/rust_implementations/modules/pose_outlier_filter/README.md)** - Pose validation and filtering
- **[Temporal Acceleration](../src/rust_implementations/modules/temporal_acceleration/README.md)** - Region-of-interest prediction

## Quick Start

1. **Pipeline Operations**: Start with [Pipeline Overview](../md_docs/pipeline_docs/PipelineOverview.md) for understanding the pipeline architecture
2. **WebUI**: Refer to [WebUI Overview](./webui/WEBUI_OVERVIEW.md) for understanding the interface
3. **API Integration**: Check [API Documentation](./webui/API_DOCUMENTATION.md) for programmatic access
4. **Inference Setup**: Refer to [Object Detection](../md_docs/pipeline_docs/main_operations/ObjectDetection.md) for device and model configuration

## Last Updated

- WebUI Documentation: January 2025
- Pipeline Documentation: January 2025
- Rust Implementations: January 2025
- Overview Organization: January 2025

---

_For detailed implementation guides, see the source code in `src/` directory._
