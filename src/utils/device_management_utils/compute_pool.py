from src.utils.device_management_utils.compute_device import ComputeDevice


class ComputePool:
    def __init__(self) -> None:
        """
        Initialize the compute pool.
        """
        self.compute_pool = []
        
    def add_compute_device(self, compute_device: ComputeDevice) -> None:
        """
        Add a compute device to the compute pool.

        Args:
            compute_device (ComputeDevice): The compute device to be added.
        """
        self.compute_pool.append(compute_device)
        
    def remove_compute_device(self, compute_device: ComputeDevice) -> None:
        """
        Remove a compute device from the compute pool.

        Args:
            compute_device (ComputeDevice): The compute device to be removed.
        """
        self.compute_pool.remove(compute_device)

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
        """
        for compute_device in self.compute_pool:
            if compute_device.device_id == compute_device_id:
                return compute_device
        raise ValueError(f"Compute device with id {compute_device_id} not found")
