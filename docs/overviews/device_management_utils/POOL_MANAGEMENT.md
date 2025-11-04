# Device Management Utils - Pool Management

## Overview

The `ComputePool` class provides comprehensive management capabilities for collections of compute devices. It enables dynamic device allocation, querying, and lifecycle management across multiple hardware accelerators.

## Architecture

### Core Design

- **Container Pattern:** Maintains a list of `ComputeDevice` instances
- **Type Safety:** Enforces `ComputeDevice` interface compliance
- **Dynamic Management:** Runtime addition and removal of devices
- **Query Interface:** Multiple methods for device retrieval

### Data Structure

```python
class ComputePool:
    def __init__(self):
        self.compute_pool: list[ComputeDevice] = []
```

## Core Operations

### Device Addition

```python
def add_compute_device(self, compute_device: ComputeDevice) -> None:
    """Add a compute device to the pool."""
    self.compute_pool.append(compute_device)
```

**Features:**
- Direct list append operation
- No duplicate checking (devices can be added multiple times)
- Immediate availability after addition

### Device Removal

#### By Instance
```python
def remove_compute_device(self, compute_device: ComputeDevice) -> None:
    """Remove a specific compute device instance."""
    self.compute_pool.remove(compute_device)
```

**Behavior:**
- Removes first occurrence of the device instance
- Raises `ValueError` if device not found
- Reference-based removal (not ID-based)

#### By ID
```python
def remove_compute_device_by_id(self, compute_device_id: str) -> None:
    """Remove device by unique identifier."""
    for compute_device in self.compute_pool:
        if compute_device.device_id == compute_device_id:
            self.compute_pool.remove(compute_device)
            return
    raise ValueError(f"Compute device with id {compute_device_id} not found")
```

**Features:**
- ID-based lookup and removal
- Early return after successful removal
- Comprehensive error messaging

## Device Querying

### Retrieval by ID

```python
def get_compute_device(self, compute_device_id: str) -> ComputeDevice:
    """Retrieve device by unique identifier."""
    for compute_device in self.compute_pool:
        if compute_device.device_id == compute_device_id:
            return compute_device
    raise ValueError(f"Compute device with id {compute_device_id} not found")
```

**Use Cases:**
- Direct device access for inference operations
- Configuration updates for specific devices
- Status monitoring of individual devices

### Retrieval by Type

```python
def get_compute_devices_by_type(self, compute_device_type: str) -> list[ComputeDevice]:
    """Get all devices of specified type."""
    return [device for device in self.compute_pool
            if device.device_type == compute_device_type]
```

**Returns:** List of devices matching the specified type
**Use Cases:**
- Load balancing across similar devices
- Type-specific optimizations
- Hardware capability assessments

## Lifecycle Management

### Mass Device Control

```python
def stop_all_devices(self) -> None:
    """Stop all devices in the pool."""
    for compute_device in self.compute_pool:
        compute_device.stop()
```

**Features:**
- Graceful shutdown of all devices
- Resource cleanup across the entire pool
- Error resilience (continues stopping other devices if one fails)

## Usage Patterns

### Basic Pool Setup

```python
from src.utils.device_management_utils.compute_pool import ComputePool
from src.utils.device_management_utils.cpu import CPU
from src.utils.device_management_utils.gpu import GPU

# Initialize pool
pool = ComputePool()

# Add devices
cpu_device = CPU()
pool.add_compute_device(cpu_device)

try:
    gpu_device = GPU()
    pool.add_compute_device(gpu_device)
except RuntimeError:
    print("GPU not available, continuing with CPU only")

# Use devices
devices = pool.get_compute_devices_by_type("CPU")
if devices:
    cpu = devices[0]
    # Perform inference...
```

### Dynamic Scaling

```python
# Add devices based on workload
def scale_pool(pool: ComputePool, target_gpu_count: int):
    current_gpus = len(pool.get_compute_devices_by_type("GPU_CUDA"))

    # Add more GPUs if needed
    while current_gpus < target_gpu_count:
        try:
            gpu = GPU(device_id=f"GPU_{current_gpus + 1:03d}")
            pool.add_compute_device(gpu)
            current_gpus += 1
        except RuntimeError:
            break

    return current_gpus
```

### Device Failover

```python
def get_available_device(pool: ComputePool, preferred_type: str) -> ComputeDevice:
    """Get first available device of preferred type, fallback to CPU."""
    devices = pool.get_compute_devices_by_type(preferred_type)
    if devices:
        return devices[0]  # Return first available

    # Fallback to CPU
    cpu_devices = pool.get_compute_devices_by_type("CPU")
    if cpu_devices:
        return cpu_devices[0]

    raise RuntimeError("No compute devices available")
```

### Resource Cleanup

```python
def cleanup_pool(pool: ComputePool):
    """Safely shutdown and clear all devices."""
    pool.stop_all_devices()
    pool.compute_pool.clear()  # Remove all references
```

## Performance Considerations

### Lookup Efficiency

- **Linear Search:** ID-based lookups are O(n) complexity
- **Type Filtering:** List comprehensions create new lists
- **Memory Overhead:** Minimal, only stores device references

### Scaling Limits

- **Device Count:** Limited by available hardware and system resources
- **Concurrent Access:** Not thread-safe, requires external synchronization
- **Memory Usage:** Scales linearly with device count

## Thread Safety

### Current Limitations

The `ComputePool` class is not thread-safe. Concurrent access requires external synchronization:

```python
import threading

pool_lock = threading.Lock()

def thread_safe_add_device(pool: ComputePool, device: ComputeDevice):
    with pool_lock:
        pool.add_compute_device(device)
```

### Future Enhancements

Potential thread-safety improvements:
- Internal locking mechanisms
- Atomic operations for device management
- Concurrent data structures for device storage

## Error Handling

### Common Error Scenarios

- **Device Not Found:** `ValueError` for missing device IDs
- **Empty Pool:** Custom exceptions for pool exhaustion
- **Device Failures:** Individual device errors during operations

### Robust Error Handling

```python
def safe_stop_all(pool: ComputePool):
    """Stop all devices with error resilience."""
    errors = []
    for device in pool.compute_pool:
        try:
            device.stop()
        except Exception as e:
            errors.append(f"Failed to stop {device.device_id}: {e}")

    if errors:
        # Log or handle errors appropriately
        print(f"Errors during shutdown: {errors}")

    return len(errors) == 0
```

## Integration Patterns

### With Object Detection Pipelines

```python
class InferencePipeline:
    def __init__(self, device_pool: ComputePool):
        self.device_pool = device_pool
        self.active_device = None

    def select_optimal_device(self, model_requirements: dict):
        """Select best available device for model requirements."""
        device_type = model_requirements.get('preferred_device', 'CPU')

        try:
            self.active_device = get_available_device(self.device_pool, device_type)
        except RuntimeError:
            # Fallback logic
            self.active_device = get_available_device(self.device_pool, 'CPU')

        return self.active_device
```

### With Configuration Systems

```python
def configure_pool_from_config(config: dict) -> ComputePool:
    """Create and configure pool from configuration."""
    pool = ComputePool()

    # Configure CPU (always available)
    if config.get('enable_cpu', True):
        pool.add_compute_device(CPU())

    # Configure GPUs
    gpu_count = config.get('gpu_count', 0)
    for i in range(gpu_count):
        try:
            gpu = GPU(device_id=f"GPU_{i+1:03d}")
            pool.add_compute_device(gpu)
        except RuntimeError as e:
            print(f"Failed to initialize GPU {i+1}: {e}")

    # Configure MX3 devices
    mx3_count = config.get('mx3_count', 0)
    for i in range(mx3_count):
        try:
            mx3 = MX3Accelerator(device_id=f"MX3_{i+1:03d}")
            pool.add_compute_device(mx3)
        except RuntimeError as e:
            print(f"Failed to initialize MX3 {i+1}: {e}")

    return pool
```

## Monitoring and Observability

### Pool Statistics

```python
def get_pool_stats(pool: ComputePool) -> dict:
    """Get comprehensive pool statistics."""
    stats = {
        'total_devices': len(pool.compute_pool),
        'device_types': {},
        'device_ids': []
    }

    for device in pool.compute_pool:
        device_type = device.device_type
        stats['device_types'][device_type] = stats['device_types'].get(device_type, 0) + 1
        stats['device_ids'].append(device.device_id)

    return stats
```

### Health Monitoring

```python
def check_pool_health(pool: ComputePool) -> dict:
    """Check health status of all devices in pool."""
    health_status = {}

    for device in pool.compute_pool:
        try:
            # Perform basic health check (device-specific)
            health_status[device.device_id] = 'healthy'
        except Exception as e:
            health_status[device.device_id] = f'unhealthy: {e}'

    return health_status
```

## Best Practices

### Pool Management

1. **Initialize Early:** Set up device pool during application startup
2. **Resource Cleanup:** Always call `stop_all_devices()` during shutdown
3. **Error Resilience:** Handle device initialization failures gracefully
4. **Configuration-Driven:** Use configuration files for device setup

### Device Selection

1. **Capability Matching:** Select devices based on model requirements
2. **Load Balancing:** Distribute workload across similar devices
3. **Failover Planning:** Always have CPU fallback available
4. **Performance Monitoring:** Track device utilization and performance

### Maintenance

1. **Regular Health Checks:** Monitor device availability and health
2. **Dynamic Reconfiguration:** Support runtime device addition/removal
3. **Resource Limits:** Monitor memory and power usage
4. **Logging:** Comprehensive logging of device operations and errors
