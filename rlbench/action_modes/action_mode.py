from abc import abstractmethod

import numpy as np

from rlbench.action_modes.arm_action_modes import ArmActionMode, JointPosition
from rlbench.action_modes.gripper_action_modes import GripperActionMode, GripperJointPosition
from rlbench.backend.scene import Scene
from rlbench.backend.robot import Robot
from rlbench.const import SUPPORTED_ROBOTS, ROBOT_ROLES

class ActionMode(object):

    def __init__(self,
                 arm_action_mode: 'ArmActionMode',
                 gripper_action_mode: 'GripperActionMode'):
        self.arm_action_mode = arm_action_mode
        self.gripper_action_mode = gripper_action_mode

    @abstractmethod
    def action(self, scene: Scene, robot: Robot, action: np.ndarray):
        pass

    @abstractmethod
    def action_shape(self, robot: Robot):
        pass

    def action_bounds(self):
        """Returns the min and max of the action mode."""
        raise NotImplementedError('You must define your own action bounds.')


class MoveArmThenGripper(ActionMode):
    """A customizable action mode.

    The arm action is first applied, followed by the gripper action.
    """

    def action(self, scene: Scene, robot: Robot, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(robot))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])
        self.arm_action_mode.action(scene, robot, arm_action)
        if ee_action.shape[-1] > 0:
            self.gripper_action_mode.action(scene, robot, ee_action)

    def action_shape(self, robot: Robot):
        return np.prod(self.arm_action_mode.action_shape(robot)) + np.prod(
            self.gripper_action_mode.action_shape(robot))


# RLBench is highly customizable, in both observations and action modes.
# This can be a little daunting, so below we have defined some
# common action modes for you to choose from.

class JointPositionActionMode(ActionMode):
    """A pre-set, delta joint position action mode or arm and abs for gripper.

    Both the arm and gripper action are applied at the same time.
    """

    def __init__(self):
        super(JointPositionActionMode, self).__init__(
            JointPosition(False), GripperJointPosition(True))

    def action(self, scene: Scene, robot: Robot, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape( robot))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])
        self.arm_action_mode.action_pre_step(scene, robot, arm_action)
        self.gripper_action_mode.action_pre_step(scene, robot, ee_action)
        scene.step()
        self.arm_action_mode.action_post_step(scene, robot, arm_action)
        self.gripper_action_mode.action_post_step(scene, robot, ee_action)

    def action_shape(self, robot: Robot):
        return np.prod(self.arm_action_mode.action_shape( robot)) + np.prod(
            self.gripper_action_mode.action_shape( robot))

    def action_bounds(self):
        """Returns the min and max of the action mode."""
        return np.array(7 * [-0.1] + [0.0]), np.array(7 * [0.1] + [0.04])
