# Device Management Utils - Device Implementations

## CPU Implementation

### Overview

The CPU implementation provides reliable, always-available inference capabilities using ONNX Runtime. It serves as the fallback device when specialized hardware is unavailable.

### Technical Details

**Backend:** ONNX Runtime with CPUExecutionProvider
**Model Format:** ONNX (.onnx files)
**Threading:** Multi-threaded execution
**Optimization:** Full graph optimization enabled

### Key Features

- **Universal Compatibility:** Runs on any system with Python support
- **Model Caching:** Loaded models are cached by filename key
- **Automatic Optimization:** Graph optimization level set to ORT_ENABLE_ALL
- **Fallback Device:** Always available when other devices fail

### Performance Characteristics

- **Latency:** Variable, depends on CPU performance and model complexity
- **Throughput:** Limited by CPU core count and clock speed
- **Memory Usage:** Models loaded into system RAM
- **Power Consumption:** Low power usage, suitable for continuous operation

### Usage Example

```python
from src.utils.device_management_utils.cpu import CPU

# Initialize CPU device
cpu_device = CPU()

# Load ONNX model
cpu_device.load_model("path/to/model.onnx")

# Run inference
import torch
input_tensor = torch.randn(1, 3, 640, 480)  # Example input
output = cpu_device.run("model", input_tensor, (640, 480))
```

### Limitations

- No `stop()` method implementation (not required for CPU inference as ONNX Runtime handles resource cleanup automatically)
- Synchronous execution only
- Performance limited by CPU capabilities

## GPU Implementation

### Overview

The GPU implementation leverages NVIDIA CUDA for accelerated inference using PyTorch. It provides high-throughput processing for compute-intensive models.

### Technical Details

**Backend:** PyTorch with CUDA support
**Model Format:** PyTorch (.pt/.pth files)
**Acceleration:** CUDA parallel processing
**Memory:** GPU VRAM for model storage and computation

### Key Features

- **CUDA Acceleration:** Leverages GPU parallel processing
- **Automatic Device Placement:** Models automatically moved to GPU
- **Memory Management:** PyTorch handles GPU memory allocation
- **Eval Mode:** Models set to evaluation mode for inference

### Performance Characteristics

- **Latency:** Low latency for large batch sizes
- **Throughput:** High throughput for parallel processing
- **Memory Usage:** Models stored in GPU VRAM
- **Power Consumption:** Higher power usage, optimized for burst processing

### Requirements

- CUDA-compatible NVIDIA GPU
- PyTorch with CUDA support installed
- Sufficient GPU memory for models and data

### Usage Example

```python
from src.utils.device_management_utils.gpu import GPU

# Initialize GPU device (checks CUDA availability)
gpu_device = GPU()

# Load PyTorch model
gpu_device.load_model("path/to/model.pth")

# Run inference
import numpy as np
input_data = np.random.randn(1, 3, 640, 480).astype(np.float32)
output = gpu_device.run("path/to/model.pth", input_data, (640, 480), 0)
```

### Error Handling

- **CUDA Unavailable:** RuntimeError if CUDA not detected
- **No GPUs:** RuntimeError if no CUDA devices found
- **Model Loading:** RuntimeError for file/loading errors

## MX3 Accelerator Implementation

### Overview

The MX3 implementation utilizes MemryX hardware accelerators for ultra-low latency neural processing. It features asynchronous multi-stream processing for high-performance inference.

### Technical Details

**Backend:** MemryX MultiStreamAsyncAccl
**Model Format:** ONNX models compiled for MX3 hardware
**Processing:** Asynchronous multi-stream architecture
**Memory:** Dedicated neural processing memory

### Key Features

- **Asynchronous Processing:** Multiple streams processed concurrently
- **Hardware Acceleration:** Dedicated neural processing hardware
- **Low Latency:** Optimized for real-time applications
- **Multi-Stream Support:** Parallel processing of multiple inputs

### Architecture Components

#### MX3ModelIO Class

Manages the interface between application and MX3 hardware:

- **Input Generation:** Provides data to MX3 processing streams
- **Output Collection:** Retrieves processed results
- **Stream Management:** Handles multiple concurrent streams
- **Stop Signaling:** Graceful shutdown coordination

#### Processing Flow

1. **Model Compilation:** ONNX models compiled for MX3 architecture
2. **Stream Initialization:** Multiple processing streams created
3. **Async Processing:** Inputs fed to streams asynchronously
4. **Result Retrieval:** Outputs collected as they become available

### Performance Characteristics

- **Latency:** Ultra-low latency (microseconds range)
- **Throughput:** High throughput with multi-stream processing
- **Memory Usage:** Efficient dedicated neural memory
- **Power Consumption:** Optimized for embedded applications

### Usage Example

```python
from src.utils.device_management_utils.mx3_accelerator import MX3Accelerator

# Initialize MX3 device
mx3_device = MX3Accelerator(
    model_path="path/to/model.onnx",
    input_data_shape=(640, 480),
    is_grayscale=False
)

# Load and compile model
mx3_device.load_model("path/to/model.onnx")

# Run inference (asynchronous)
input_data = np.random.randn(1, 3, 640, 480).astype(np.float32)
output = mx3_device.run("path/to/model.onnx", input_data, (640, 480), 0)
```

### Configuration Options

- **Input Shape:** Configurable tensor dimensions
- **Grayscale Support:** Single-channel input option
- **Stream Count:** Multiple concurrent processing streams
- **Polling Interval:** Configurable result checking frequency

### Error Handling

- **Hardware Initialization:** RuntimeError for MX3 hardware issues
- **Library Loading:** ImportError for missing MemryX SDK
- **Model Compilation:** RuntimeError for incompatible models

## Device Comparison

| Feature          | CPU                  | GPU                | MX3                   |
| ---------------- | -------------------- | ------------------ | --------------------- |
| **Availability** | Always               | Requires CUDA GPU  | Requires MX3 hardware |
| **Model Format** | ONNX                 | PyTorch            | ONNX                  |
| **Processing**   | Synchronous          | Synchronous        | Asynchronous          |
| **Latency**      | Medium               | Low-Medium         | Ultra-Low             |
| **Throughput**   | Low-Medium           | High               | High                  |
| **Power Usage**  | Low                  | High               | Medium                |
| **Use Case**     | Fallback/Development | Training/Inference | Real-time Production  |

## Selection Guidelines

### Choose CPU when:

- Developing and testing applications
- Running on systems without GPU/MX3 hardware
- Power consumption is a primary concern
- Model complexity is low to medium

### Choose GPU when:

- High throughput is required
- Large batch processing is needed
- CUDA-compatible hardware is available
- Power budget allows for higher consumption

### Choose MX3 when:

- Ultra-low latency is critical
- Real-time processing requirements
- Embedded or edge deployment
- Dedicated neural hardware is available

## Future Extensions

The modular architecture supports easy addition of new device types:

- **TPU Implementation:** Google Tensor Processing Units
- **Coral Implementation:** Google Coral Edge TPU
- **OpenVINO Implementation:** Intel OpenVINO toolkit
- **ROCm Implementation:** AMD GPU computing

Each new device type would follow the same pattern: inherit from `ComputeDevice`, implement the three abstract methods, and integrate device-specific optimizations.
