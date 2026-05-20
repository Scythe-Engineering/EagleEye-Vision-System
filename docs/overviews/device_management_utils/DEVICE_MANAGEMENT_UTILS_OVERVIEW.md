# EagleEye Device Management Utils Overview

## Overview

The EagleEye Device Management Utils provide a unified interface for managing various compute devices used in object detection pipelines. The system abstracts hardware-specific implementations behind a common API, enabling seamless switching between CPU, GPU, and specialized accelerators like the MemryX MX3.

## Architecture

### Abstract Base Class Design

The system uses an abstract base class (`ComputeDevice`) that defines the interface for all compute devices. This allows for polymorphic usage of different hardware accelerators while maintaining consistent API contracts.

### Pool Management

A `ComputePool` class manages collections of compute devices, providing methods for adding, removing, and querying devices by ID or type.

### Device-Specific Implementations

Each compute device type has its own implementation class that inherits from `ComputeDevice` and provides hardware-specific model loading and inference execution.

## Main Components

### 1. ComputeDevice (Abstract Base Class)

- **Purpose**: Defines the common interface for all compute devices
- **Methods**:
  - `load_model()`: Abstract method for loading models
  - `run()`: Abstract method for executing inference
  - `stop()`: Abstract method for stopping device operations
- **Attributes**:
  - `device_id`: Unique identifier for the device
  - `device_type`: String indicating device type ('CPU', 'GPU_CUDA', 'MX3', etc.)

### 2. ComputePool (Device Management)

- **Purpose**: Manages collections of compute devices
- **Features**:
  - Add/remove devices dynamically
  - Query devices by ID or type
  - Stop all devices simultaneously
- **Methods**:
  - `add_compute_device()`
  - `remove_compute_device()`
  - `get_compute_device()`
  - `get_compute_devices_by_type()`
  - `stop_all_devices()`

### 3. CPU Implementation

- **Backend**: ONNX Runtime with CPU execution provider
- **Model Format**: ONNX models (.onnx)
- **Features**: Optimized graph execution, multi-threading support
- **Optimization**: Full graph optimization enabled

### 4. GPU Implementation

- **Backend**: PyTorch with CUDA support
- **Model Format**: PyTorch models (.pt/.pth)
- **Features**: CUDA acceleration, automatic device placement
- **Requirements**: CUDA-compatible GPU, PyTorch installation

### 5. MX3 Accelerator Implementation

- **Backend**: MemryX MultiStreamAsyncAccl
- **Model Format**: ONNX models optimized for MX3 hardware
- **Features**: Asynchronous multi-stream processing, low-latency inference
- **Architecture**: Hardware-accelerated neural processing with dedicated memory

## Key Features

### Unified API

All compute devices implement the same interface, allowing applications to switch between different hardware accelerators without code changes.

### Dynamic Device Management

The ComputePool allows for runtime addition and removal of compute devices, enabling dynamic scaling and hardware failover capabilities.

### Hardware-Specific Optimizations

Each device implementation leverages hardware-specific optimizations:
- CPU: ONNX Runtime graph optimizations
- GPU: CUDA parallel processing
- MX3: Dedicated neural processing hardware

### Asynchronous Processing (MX3)

The MX3 implementation supports asynchronous multi-stream processing, allowing multiple inference requests to be processed concurrently.

### Device-Agnostic Async Wrapper

`AsyncComputeWrapper` wraps any `ComputeDevice` and exposes the same callback contract to pipelines regardless of hardware type:

- `on_frame(model_path, input_data, input_data_shape, stream_idx)`: camera or pipeline code submits a preprocessed image buffer without waiting for the device implementation to finish inference. The call returns a request id that identifies the eventual result.
- `on_result(callback)`: pipeline code registers a thread-safe result hook. The callback receives an `AsyncComputeResult` containing the request id, output data, latency, and any device exception.
- `run(...)`: existing synchronous callers are bridged through `on_frame` and `wait_for_result`, so older pipeline code can keep the `ComputeDevice.run()` shape while new ML-heavy operations opt in to event-driven flow.

The wrapper is applied by `ComputePool.add_compute_device()` by default so dynamic device lifecycle handling still uses the compute pool. `connect_streams()`, `load_model()`, `stop()`, and MX3-style `register_thread_access()` calls are forwarded to the wrapped device.

Callback and device exceptions are preserved and surfaced back through the requesting operation. Object-detection operations store async callback exceptions and re-raise them on the next pipeline invocation, allowing the existing `FlowManager` and `Pipeline.record_operation_error()` path to publish centralized operation errors. The wrapper does not add retries or hardware failover; device replacement and retry policy remain compute-pool or operation-level responsibilities.

The async wrapper owns a small worker thread per wrapped device. `on_frame()` queue insertion is non-blocking, result callbacks run on the wrapper worker thread, and callback registration/result state are protected by locks. Callers that share buffers across threads must treat submitted input buffers as owned by the async request until its result callback fires. `AsyncComputeResult.latency_s` captures device execution time for future latency logging.

### Error Handling

Robust error handling for hardware unavailability, model loading failures, and inference execution errors.

### Resource Management

Proper cleanup and resource management through the `stop()` method implementations.

## Directory Structure

```
device_management_utils/
├── compute_device.py          # Abstract base class for compute devices
├── compute_pool.py            # Pool management for multiple devices
├── async_compute_wrapper.py   # Device-agnostic event-driven async wrapper
├── cpu.py                     # CPU-based inference implementation
├── gpu.py                     # GPU-based inference implementation
├── mx3_accelerator.py         # MemryX MX3 accelerator implementation
└── __pycache__/               # Python bytecode cache
```

## Technology Stack

### Core Dependencies

- **NumPy**: Numerical computations and array operations
- **Python ABC**: Abstract base class support

### CPU Implementation

- **ONNX Runtime**: Cross-platform inference engine
- **PyTorch**: Tensor operations and model loading

### GPU Implementation

- **PyTorch**: Deep learning framework with CUDA support
- **CUDA**: NVIDIA parallel computing platform

### MX3 Implementation

- **MemryX SDK**: MultiStreamAsyncAccl for MX3 hardware acceleration
- **ONNX Runtime**: Model loading and preprocessing

### Development Tools

- **Type Hints**: Full type annotation support
- **Google Style Docstrings**: Comprehensive documentation

## Development

### Adding New Device Types

1. Inherit from `ComputeDevice` abstract base class
2. Implement required abstract methods: `load_model()`, `run()`, `stop()`
3. Set appropriate `device_id` and `device_type` in constructor
4. Handle device-specific error conditions and cleanup

### Model Format Support

The system currently supports:
- **ONNX**: Universal format supported by CPU and MX3 implementations
- **PyTorch**: Native format for GPU implementation

### Testing Device Availability

Each device implementation includes availability checks:
- CPU: Always available (fallback device)
- GPU: CUDA availability and device count verification
- MX3: Hardware presence and library initialization

### Performance Considerations

- MX3 provides lowest latency for compatible models
- GPU offers highest throughput for large batches
- CPU serves as reliable fallback for all scenarios

## Integration Points

### Object Detection Pipelines

The device management utils integrate with object detection pipelines by providing hardware abstraction for model inference operations.

### Main Application

- Model loading and caching
- Dynamic device selection based on workload
- Performance monitoring and device failover

### Configuration System

Device selection and pool configuration through pipeline configuration files.

### Error Handling Systems

Integration with application-level error handling for device failures and hardware unavailability.

### Monitoring and Logging

Device performance metrics and hardware status reporting for system monitoring dashboards.
