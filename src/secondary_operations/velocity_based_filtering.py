import numpy as np


class VelocityBasedFiltering:
    def __init__(self, velocity_threshold: float = 0.1, history_size: int = 10, max_poses_outside_threshold: int = 5):
        """Filter out poses that are too far outside the historical norm.

        Args:
            velocity_threshold (float, optional): The threshold for the velocity. Defaults to 0.1.
            history_size (int, optional): The size of the history. Defaults to 10.
            max_poses_outside_threshold (int, optional): The number of poses outside the threshold to reset the history. Defaults to 5.
        """
        self.velocity_threshold = velocity_threshold
        self.history_size = history_size
        self.max_poses_outside_threshold = max_poses_outside_threshold
        
        self.previous_poses: np.ndarray = np.zeros((history_size, 3))
        self.num_non_zero_poses = 0
        self.num_poses_outside_threshold = 0

    def run(self, pose: np.ndarray) -> np.ndarray | None:
        """Filter out poses that are too far outside the historical norm.

        Args:
            pose (np.ndarray): The pose to filter.

        Returns:
            np.ndarray | None: The filtered pose. None if the pose is too far outside the historical norm.
        """
        # extract the position from the pose
        position = pose[:3, 3]
        
        # Prevent continuous failures from causing the pipeline to fail
        if self.num_poses_outside_threshold == self.max_poses_outside_threshold:
            self.num_poses_outside_threshold = 0
            self.num_non_zero_poses = 0
            self.previous_poses = np.zeros((self.history_size, 3))
            return pose
        
        # If the history is not full, add the pose to the history
        if self.num_non_zero_poses < self.history_size:
            self.previous_poses[self.num_non_zero_poses] = position
            self.num_non_zero_poses += 1
            return pose
        
        # Calculate the velocity of the pose
        velocity = position - self.previous_poses[-1]
        
        if np.linalg.norm(velocity) < self.velocity_threshold:
            # Update the history if it is within the velocity threshold, do this by shifting the history to the left and adding the new pose to the end
            self.previous_poses = np.roll(self.previous_poses, -1)
            self.previous_poses[-1] = position
            
            return pose
        
        self.num_poses_outside_threshold += 1
        return None
