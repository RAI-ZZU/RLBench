from typing import Any, Dict, List, Tuple, Union
import numpy as np
import transforms3d as tfs
from scipy.spatial.transform import Rotation as R

from pyrep import PyRep
from pyrep.robots.arms.arm import Arm
from pyrep.const import ObjectType
from pyrep.objects.dummy import Dummy

from pyrep.robots.end_effectors.gripper import Gripper

#from rlbench.action_modes.action_mode import ActionMode
from rlbench.const import ROBOT_ROLES




class Robot(object):
    """Simple container for the robot components.
    """

    def __init__(self, arm: Arm, gripper: Gripper, pyrep:PyRep=None, role:str=None, action_mode=None):
        self.arm = arm
        self.gripper = gripper
        self._pyrep = pyrep
        if role is not None:
            assert role in ROBOT_ROLES 
        self.role = role
        self.action_mode = action_mode

        self._initial_robot_state = (self.arm.get_configuration_tree(),
                                        self.gripper.get_configuration_tree())
        self._start_arm_joint_pos = self.arm.get_joint_positions()
        self._starting_gripper_joint_pos = self.gripper.get_joint_positions() 
        self.robot_shapes = self.arm.get_objects_in_tree(object_type=ObjectType.SHAPE)
        self._scene = None


        """
        if self.gripper.has_wirst_cam():
            wtc_trans,wtc_rot = self.gripper.get_wirst_cam_pose()
            wtg_trans,wtg_rot = self.arm.get_tip().get_position(), self.arm.get_tip().get_quaternion()

            wtg = tfs.affines.compose(wtg_trans,tfs.quaternions.quat2mat([wtg_rot[-1]]+ list(wtg_rot[:-1])),[1, 1, 1])
            wtc = tfs.affines.compose(wtc_trans,tfs.quaternions.quat2mat([wtc_rot[-1]]+ list(wtc_rot[:-1])),[1, 1, 1])

            self.ctg = np.linalg.inv(wtc).dot(wtg)"""
        

    def set_scene(self, scene):
        self._scene = scene

    def action(self, action:np.ndarray):
        assert self._scene is not None
        self.action_mode.action(self._scene, self, action)
    
    def set_control_mode(self):
        self.action_mode.arm_action_mode.set_control_mode(self)

    def reset(self):
        self.gripper.release()

        arm, gripper = self._initial_robot_state 
        self._pyrep.set_configuration_tree(arm)
        self._pyrep.set_configuration_tree(gripper)

        self.arm.set_joint_positions(self._start_arm_joint_pos, disable_dynamics=True)
        self.arm.set_joint_target_velocities([0] * len(self.arm.joints))

        self.gripper.set_joint_positions(self._starting_gripper_joint_pos, disable_dynamics=True)
        self.gripper.set_joint_target_velocities([0] * len(self.gripper.joints))
    
    def get_tip_from_view_point(self,viewpoint:Dummy)-> Dummy:
        assert self.gripper.has_wirst_cam()

        tip = Dummy.create()
        wtc_trans = viewpoint.get_position()
        wtc_quat = viewpoint.get_quaternion()
        wtc = tfs.affines.compose(wtc_trans,tfs.quaternions.quat2mat([wtc_quat[-1]]+ list(wtc_quat[:-1])),[1, 1, 1])
        wtg = wtc.dot(self.ctg)
        wtg_trans = wtg[:3,3]
        wtg_rot_quat = R.from_matrix(wtg[:3,:3]).as_quat()
        tip.set_position(wtg_trans)
        tip.set_quaternion(wtg_rot_quat)
        return tip

    def get_viewpoint_from_tip(self,tip:Dummy)-> Dummy:
        assert self.gripper.has_wirst_cam()
        viewpoint = Dummy.create()
        wtg_trans = tip.get_position()
        wtg_quat = tip.get_quaternion()
        wtg = tfs.affines.compose(wtg_trans,tfs.quaternions.quat2mat([wtg_quat[-1]]+ list(wtg_quat[:-1])),[1, 1, 1])
        wtc = wtg.dot(np.linalg.inv(self.ctg))
        wtc_trans = wtc[:3,3]
        wtc_rot_quat = R.from_matrix(wtc[:3,:3]).as_quat()
        viewpoint.set_position(wtc_trans)
        viewpoint.set_quaternion(wtc_rot_quat)
        return viewpoint




class Robots(object):
    """Robots in the scene
    """

    def __init__(self, worker_arm: Union[Robot,None]=None, 
                    vision_arm: Union[Robot,None]=None):
        self.worker_arm = worker_arm
        self.vision_arm = vision_arm
        #self.scene = None

    def yield_robot(self):
        if self.worker_arm is not None:
            yield self.worker_arm
        if self.vision_arm is not None:
            yield self.vision_arm

    def set_scene(self,scene):
        [robot.set_scene(scene) for robot in self.yield_robot()]

    @property
    def robot_num(self):
        return len(list(self.yield_robot()))

    def set_control_mode(self):
        [robot.set_control_mode() for robot in self.yield_robot()]

    def get_action_shape(self) -> Dict[str,int]:
        action_shape = [robot.action_mode.action_shape(robot) for robot in self.yield_robot()]
        return dict(zip(ROBOT_ROLES[:len(action_shape)],action_shape))
    
    def action(self,action:Dict[str,np.ndarray]):
        assert len(action.keys()) == self.robot_num 
        for role,action_ in action.items():
            assert role in ROBOT_ROLES
            if "worker" in role and action_ is not None:
                self.worker_arm.action(action_)
            elif "vision" in role and action_ is not None:
                self.vision_arm.action(action_)
    
    
    def reset(self):
        [robot.reset() for robot in self.yield_robot()]
    
