# Device Management Utils API Documentation

## ComputeDevice Abstract Base Class

### Constructor

```python
ComputeDevice(device_id: str, device_type: str)
```

**Parameters:**
- `device_id` (str): Unique identifier for the compute device
- `device_type` (str): Type of compute device ('CPU', 'GPU_CUDA', 'MX3', etc.)

### Methods

#### `load_model(model_path: str, input_data_shape: tuple[int, int], post_processing_model_path: str | None = None, is_grayscale: bool = False) -> None`
Abstract method for loading a model into the compute device.

**Parameters:**
- `model_path` (str): Path to the model file
- `input_data_shape` (tuple[int, int]): Expected input shape (height, width)
- `post_processing_model_path` (str | None): Optional path to post-processing model
- `is_grayscale` (bool): Whether input is grayscale (default: False)

**Raises:**
- `NotImplementedError`: Must be implemented by subclasses

#### `run(model_path: str, input_data: np.ndarray | torch.Tensor, input_data_shape: tuple[int, int], stream_idx: int = 0) -> np.ndarray`
Abstract method for running inference on the loaded model.

**Parameters:**
- `model_path` (str): Path to the model (used as key for model lookup)
- `input_data` (np.ndarray | torch.Tensor): Input data for inference (numpy array or torch tensor)
- `input_data_shape` (tuple[int, int]): Shape of input data (height, width)
- `stream_idx` (int): Stream index for multi-stream processing (default: 0)

**Returns:**
- `np.ndarray`: Model output

**Raises:**
- `NotImplementedError`: Must be implemented by subclasses

#### `connect_streams() -> None`
Abstract method for managing multi-stream processing.

**Raises:**
- `NotImplementedError`: Must be implemented by subclasses

#### `stop() -> None`
Abstract method for stopping the compute device and cleaning up resources.

**Raises:**
- `NotImplementedError`: Must be implemented by subclasses

## ComputePool Class

### Constructor

```python
ComputePool()
```

Initializes an empty compute pool.

### Methods

#### `add_compute_device(compute_device: ComputeDevice) -> None`
Add a compute device to the pool.

**Parameters:**
- `compute_device` (ComputeDevice): Device instance to add

#### `remove_compute_device(compute_device: ComputeDevice) -> None`
Remove a specific compute device from the pool.

**Parameters:**
- `compute_device` (ComputeDevice): Device instance to remove

#### `remove_compute_device_by_id(compute_device_id: str) -> None`
Remove a compute device by its ID.

**Parameters:**
- `compute_device_id` (str): ID of device to remove

**Raises:**
- `ValueError`: If device with given ID is not found

#### `get_compute_device(compute_device_id: str) -> ComputeDevice`
Retrieve a compute device by its ID.

**Parameters:**
- `compute_device_id` (str): ID of device to retrieve

**Returns:**
- `ComputeDevice`: The requested device instance

**Raises:**
- `ValueError`: If device with given ID is not found

#### `get_compute_devices_by_type(compute_device_type: str) -> list[ComputeDevice]`
Get all devices of a specific type.

**Parameters:**
- `compute_device_type` (str): Type of devices to retrieve

**Returns:**
- `list[ComputeDevice]`: List of devices matching the type

#### `stop_all_devices() -> None`
Stop all devices in the pool and clean up resources.

## CPU Implementation

### Constructor

```python
CPU()
```

Creates a CPU compute device with ID "CPU_001" and type "CPU".

### Methods

#### `load_model(model_path: str, input_data_shape: tuple[int, int], post_processing_model_path: str | None = None, is_grayscale: bool = False) -> None`
Load an ONNX model for CPU inference.

**Parameters:**
- `model_path` (str): Path to .onnx model file
- `input_data_shape` (tuple[int, int]): Expected input shape (height, width)
- `post_processing_model_path` (str | None): Optional path to post-processing model
- `is_grayscale` (bool): Whether input is grayscale (default: False)

**Raises:**
- `RuntimeError`: If model loading fails

#### `run(model_path: str, input_data: np.ndarray | torch.Tensor, input_data_shape: tuple[int, int], stream_idx: int = 0) -> np.ndarray`
Run inference on CPU using ONNX Runtime.

**Parameters:**
- `model_path` (str): Path to model (used as key)
- `input_data` (np.ndarray | torch.Tensor): Input data array or tensor
- `input_data_shape` (tuple[int, int]): Input shape (height, width)
- `stream_idx` (int): Stream index for multi-stream processing (default: 0)

**Returns:**
- `np.ndarray`: Inference output

**Raises:**
- `ValueError`: If model is not loaded

#### `connect_streams() -> None`
Manage multi-stream processing for CPU device.

#### `stop() -> None`
Implemented as a no-op. No resources require cleanup for CPU inference as ONNX Runtime handles resource cleanup automatically.

## GPU Implementation

### Constructor

```python
GPU(device_id: str = "GPU_001")
```

Creates a GPU compute device with CUDA support.

**Parameters:**
- `device_id` (str): Unique GPU device identifier

**Raises:**
- `RuntimeError`: If CUDA is unavailable or no GPUs detected

### Methods

#### `load_model(model_path: str, input_data_shape: tuple[int, int], post_processing_model_path: str | None = None, is_grayscale: bool = False) -> None`
Load a PyTorch model for GPU inference.

**Parameters:**
- `model_path` (str): Path to .pt/.pth PyTorch model file
- `input_data_shape` (tuple[int, int]): Expected input shape (height, width)
- `post_processing_model_path` (str | None): Optional path to post-processing model
- `is_grayscale` (bool): Whether input is grayscale (default: False)

**Raises:**
- `RuntimeError`: If model loading fails

#### `run(model_path: str, input_data: np.ndarray | torch.Tensor, input_data_shape: tuple[int, int], stream_idx: int = 0) -> np.ndarray`
Run inference on GPU using PyTorch CUDA.

**Parameters:**
- `model_path` (str): Path to model (used as key)
- `input_data` (np.ndarray | torch.Tensor): Input data array or tensor
- `input_data_shape` (tuple[int, int]): Input shape (height, width)
- `stream_idx` (int): Stream index for multi-stream processing (default: 0)

**Returns:**
- `np.ndarray`: Inference output on CPU

**Raises:**
- `ValueError`: If model is not loaded

#### `connect_streams() -> None`
Manage multi-stream processing for GPU device. Coordinate multiple processing streams when needed, though PyTorch's synchronous execution model limits stream-level parallelism.

#### `stop() -> None`
Clear loaded models from GPU memory.

## MX3 Accelerator Implementation

### MX3ModelIO Helper Class

#### Constructor

```python
MX3ModelIO(model_object: MultiStreamAsyncAccl, input_data_shape: tuple[int, int], is_grayscale: bool = False)
```

**Parameters:**
- `model_object` (MultiStreamAsyncAccl): MX3 model instance
- `input_data_shape` (tuple[int, int]): Input tensor shape (height, width)
- `is_grayscale` (bool): Whether input is grayscale (affects channel count)

### MX3Accelerator Class

#### Constructor

```python
MX3Accelerator(device_id: str = "MX3_001", logger=None)
```

**Parameters:**
- `device_id` (str): Unique MX3 device identifier
- `logger`: Optional logger instance for debugging output

**Raises:**
- `RuntimeError`: If MX3 hardware initialization fails

#### Methods

#### `load_model(model_path: str, input_data_shape: tuple[int, int], post_processing_model_path: str | None = None, is_grayscale: bool = False) -> None`
Load and compile model for MX3 hardware.

**Parameters:**
- `model_path` (str): Path to .onnx model file
- `input_data_shape` (tuple[int, int]): Expected input shape (height, width)
- `post_processing_model_path` (str | None): Optional path to post-processing model
- `is_grayscale` (bool): Whether model expects grayscale input

#### `run(model_path: str, input_data: np.ndarray | torch.Tensor, input_data_shape: tuple[int, int], stream_idx: int = 0) -> np.ndarray`
Run asynchronous inference on MX3 accelerator.

**Parameters:**
- `model_path` (str): Model path (used as key for model lookup)
- `input_data` (np.ndarray | torch.Tensor): Input data array or tensor
- `input_data_shape` (tuple[int, int]): Input shape (height, width)
- `stream_idx` (int): Stream index for multi-stream processing (default: 0)

**Returns:**
- `np.ndarray`: Inference results

#### `connect_streams() -> None`
Manage multi-stream processing for MX3 device. Coordinates multiple concurrent processing streams, enabling efficient parallel processing.

#### `stop() -> None`
Stop MX3 processing and clean up resources.

## Error Handling

### Common Exceptions

- **RuntimeError**: Hardware unavailability, model loading failures
- **ValueError**: Invalid model names, missing devices
- **NotImplementedError**: Abstract method not implemented

### Device-Specific Errors

- **CPU**: ONNX Runtime session creation failures
- **GPU**: CUDA unavailability, PyTorch loading errors
- **MX3**: Hardware initialization failures, MemryX SDK errors

## Type Definitions

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class ComputeDeviceProtocol(Protocol):
    device_id: str
    device_type: str

    def load_model(self, model_path: str) -> None: ...
    def run(self, model_path: str, input_data: np.ndarray, input_data_shape: tuple[int, int], stream_idx: int) -> np.ndarray: ...
    def stop(self) -> None: ...
```
