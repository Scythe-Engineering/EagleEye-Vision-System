from src.utils.device_management_utils.compute_device import ComputeDevice
from src.utils.device_management_utils.async_compute_wrapper import AsyncComputeWrapper


class ComputePool:
    def __init__(self, enable_async_wrappers: bool = True) -> None:
        """
        Initialize the compute pool.

        Args:
            enable_async_wrappers (bool): Whether added devices should be wrapped
                with the event-driven async compute contract.
        """
        self.enable_async_wrappers = enable_async_wrappers
        self.compute_pool: list[ComputeDevice] = []
        
    def add_compute_device(self, compute_device: ComputeDevice) -> None:
        """
        Add a compute device to the compute pool.

        Args:
            compute_device (ComputeDevice): The compute device to be added.
        """
        self.compute_pool.append(self._wrap_compute_device(compute_device))
        
    def remove_compute_device(self, compute_device: ComputeDevice) -> None:
        """
        Remove a compute device from the compute pool.

        Args:
            compute_device (ComputeDevice): The compute device to be removed.
        """
        if compute_device in self.compute_pool:
            self.compute_pool.remove(compute_device)
            return
        for pooled_compute_device in self.compute_pool:
            if (
                isinstance(pooled_compute_device, AsyncComputeWrapper)
                and pooled_compute_device.delegate is compute_device
            ):
                self.compute_pool.remove(pooled_compute_device)
                return
        raise ValueError(f"Compute device with id {compute_device.device_id} not found")

    def remove_compute_device_by_id(self, compute_device_id: str) -> None:
        """
        Remove a compute device from the compute pool by its id.

        Args:
            compute_device_id (str): The id of the compute device to be removed.
        """
        for compute_device in self.compute_pool:
            if compute_device.device_id == compute_device_id:
                self.compute_pool.remove(compute_device)
                return
        raise ValueError(f"Compute device with id {compute_device_id} not found")
        
    def get_compute_device(self, compute_device_id: str) -> ComputeDevice:
        """
        Get a compute device from the compute pool by its id.

        Args:
            compute_device_id (str): The id of the compute device to be retrieved.

        Returns:
            ComputeDevice: The compute device with the given id.
        """
        for compute_device in self.compute_pool:
            if compute_device.device_id == compute_device_id:
                return compute_device
        raise ValueError(f"Compute device with id {compute_device_id} not found")
    
    def get_compute_devices_by_type(self, compute_device_type: str) -> list[ComputeDevice]:
        """
        Get compute devices from the compute pool by their type.

        Args:
            compute_device_type (str): The type of the compute device to be retrieved.

        Returns:
            list[ComputeDevice]: A list of compute devices of the given type.
        """
        return [compute_device for compute_device in self.compute_pool if compute_device.device_type == compute_device_type]
    
    def stop_all_devices(self) -> None:
        """
        Stop all devices in the compute pool.
        """
        for compute_device in self.compute_pool:
            compute_device.stop()

    def _wrap_compute_device(self, compute_device: ComputeDevice) -> ComputeDevice:
        """Wrap compute devices with the async event contract when enabled.

        Args:
            compute_device (ComputeDevice): Device to wrap.

        Returns:
            ComputeDevice: Wrapped or original compute device.
        """
        if not self.enable_async_wrappers:
            return compute_device
        if isinstance(compute_device, AsyncComputeWrapper):
            return compute_device
        return AsyncComputeWrapper(compute_device)
