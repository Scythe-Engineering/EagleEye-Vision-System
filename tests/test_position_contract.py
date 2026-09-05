"""Physical robot/camera fixtures independent of the implementation's Euler remapping."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from src.secondary_operations.camera_to_robot_pose import CameraToRobotPose
from src.secondary_operations.publish_to_networktables import _matrix_to_pose3d
from src.utils.camera_utils.camera_config_manager import CameraConfigRegistry

# Camera right/down/forward expressed in a forward/left/up robot frame.
B = np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])

@pytest.mark.parametrize('yaw,pitch,roll,offset', [
    (0,0,0,(0,.4,0)), (0,0,0,(.3,0,.6)), (90,0,0,(0,0,0)),
    (0,30,0,(0,0,0)), (0,0,25,(0,0,0)), (37,19,-23,(.3,.4,.6)),
])
def test_mount_recovers_known_robot_pose(tmp_path,yaw,pitch,roll,offset):
    reg=CameraConfigRegistry(base_path=str(tmp_path))
    reg.get_config('cam').update_extrinsics_live(dict(yaw=yaw,pitch=pitch,roll=roll,
        x_offset=offset[0],y_offset=offset[1],z_offset=offset[2]))
    robot=np.eye(4); robot[:3,:3]=Rotation.from_euler('ZYX',[63,7,-11],degrees=True).as_matrix()
    robot[:3,3]=[4,2,.1]
    mount=np.eye(4); mount[:3,:3]=Rotation.from_euler('ZYX',[yaw,pitch,roll],degrees=True).as_matrix() @ B
    mount[:3,3]=offset
    camera=robot @ mount
    result=_matrix_to_pose3d(CameraToRobotPose('cam',reg).run(camera))
    np.testing.assert_allclose([result.X(),result.Y(),result.Z()],robot[:3,3],atol=1e-9)
    q=result.rotation().getQuaternion()
    actual=Rotation.from_quat([q.X(),q.Y(),q.Z(),q.W()]).as_matrix()
    np.testing.assert_allclose(actual,robot[:3,:3],atol=1e-9)

def test_saved_extrinsics_refresh_existing_operation(tmp_path):
    reg=CameraConfigRegistry(base_path=str(tmp_path))
    config=reg.get_config('cam')
    config.update_extrinsics_live(dict(x_offset=0,y_offset=0,z_offset=0,pitch=0,yaw=0,roll=0))
    op=CameraToRobotPose('cam',reg)
    camera=np.eye(4); camera[:3,:3]=B; camera[:3,3]=[4,2,.6]
    op.run(camera)
    config.update_extrinsics_live(dict(x_offset=.3,y_offset=0,z_offset=.6,pitch=0,yaw=0,roll=0))
    result=_matrix_to_pose3d(op.run(camera))
    np.testing.assert_allclose([result.X(),result.Y(),result.Z()],[3.7,2,0],atol=1e-9)
