import numpy as np


class FlattenPose:
    """Flatten a pose matrix to have no z position component and preserve only y-axis rotation."""

    def run(self, pose: np.ndarray) -> np.ndarray:
        """Flatten the pose matrix to 2D by removing z position and x/y rotations.

        This method sets z position to 0 and removes x-axis (roll) and 
        y-axis (pitch) rotations while preserving z-axis (yaw) rotation.

        Args:
            pose: 4x4 transformation matrix representing a 3D pose.

        Returns:
            4x4 pose matrix flattened to 2D with only z-axis rotation preserved.
        """
        flattened_pose = pose.copy()
        
        # Set z position to 0
        flattened_pose[2, 3] = 0.0
        
        # Zero out x and y rotations
        flattened_pose = self._zero_x_rotation(flattened_pose)
        flattened_pose = self._zero_y_rotation(flattened_pose)
        #flattened_pose = self._zero_z_rotation(flattened_pose)
        
        return flattened_pose

    def _zero_x_rotation(self, pose: np.ndarray) -> np.ndarray:
        """Zero out x-axis rotation (roll) from the pose matrix.

        Args:
            pose: 4x4 transformation matrix.

        Returns:
            4x4 pose matrix with x-axis rotation removed.
        """
        modified_pose = pose.copy()
        
        # Extract current yaw and pitch angles
        current_yaw = np.arctan2(modified_pose[1, 0], modified_pose[0, 0])
        current_pitch = np.arctan2(-modified_pose[2, 0], 
                                  np.sqrt(modified_pose[2, 1]**2 + modified_pose[2, 2]**2))
        
        # Create rotation matrix with only yaw and pitch (no roll)
        rotation_matrix = self._create_rotation_matrix(0.0, current_pitch, current_yaw)
        modified_pose[:3, :3] = rotation_matrix
        
        return modified_pose

    def _zero_y_rotation(self, pose: np.ndarray) -> np.ndarray:
        """Zero out y-axis rotation (pitch) from the pose matrix.

        Args:
            pose: 4x4 transformation matrix.

        Returns:
            4x4 pose matrix with y-axis rotation removed.
        """
        modified_pose = pose.copy()
        
        # Extract current yaw and roll angles
        current_yaw = np.arctan2(modified_pose[1, 0], modified_pose[0, 0])
        current_roll = np.arctan2(modified_pose[2, 1], modified_pose[2, 2])
        
        # Create rotation matrix with only yaw and roll (no pitch)
        rotation_matrix = self._create_rotation_matrix(current_roll, 0.0, current_yaw)
        modified_pose[:3, :3] = rotation_matrix
        
        return modified_pose

    def _zero_z_rotation(self, pose: np.ndarray) -> np.ndarray:
        """Zero out z-axis rotation (yaw) from the pose matrix.

        Args:
            pose: 4x4 transformation matrix.

        Returns:
            4x4 pose matrix with z-axis rotation removed.
        """
        modified_pose = pose.copy()
        
        # Extract current roll and pitch angles
        current_roll = np.arctan2(modified_pose[2, 1], modified_pose[2, 2])
        current_pitch = np.arctan2(-modified_pose[2, 0], 
                                  np.sqrt(modified_pose[2, 1]**2 + modified_pose[2, 2]**2))
        
        # Create rotation matrix with only roll and pitch (no yaw)
        rotation_matrix = self._create_rotation_matrix(current_roll, current_pitch, 0.0)
        modified_pose[:3, :3] = rotation_matrix
        
        return modified_pose

    def _create_rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Create a 3x3 rotation matrix from roll, pitch, and yaw angles.

        Args:
            roll: Rotation around x-axis in radians.
            pitch: Rotation around y-axis in radians.
            yaw: Rotation around z-axis in radians.

        Returns:
            3x3 rotation matrix.
        """
        cos_roll = np.cos(roll)
        sin_roll = np.sin(roll)
        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        rotation_matrix = np.array([
            [cos_pitch * cos_yaw, -cos_pitch * sin_yaw, sin_pitch],
            [cos_roll * sin_yaw + sin_roll * sin_pitch * cos_yaw, 
             cos_roll * cos_yaw - sin_roll * sin_pitch * sin_yaw, 
             -sin_roll * cos_pitch],
            [sin_roll * sin_yaw - cos_roll * sin_pitch * cos_yaw, 
             sin_roll * cos_yaw + cos_roll * sin_pitch * sin_yaw, 
             cos_roll * cos_pitch]
        ])
        
        return rotation_matrix
