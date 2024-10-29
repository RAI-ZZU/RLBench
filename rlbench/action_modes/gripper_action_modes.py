from abc import abstractmethod

import numpy as np

from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.scene import Scene
from rlbench.backend.robot import Robot


def assert_action_shape(action: np.ndarray, expected_shape: tuple):
    if np.shape(action) != expected_shape:
        raise InvalidActionError(
            'Expected the action shape to be: %s, but was shape: %s' % (
                str(expected_shape), str(np.shape(action))))


class GripperActionMode(object):

    @abstractmethod
    def action(self, scene: Scene, robot: Robot, action: np.ndarray):
        pass

    def action_step(self, scene: Scene, robot: Robot, action: np.ndarray):
        pass

    def action_pre_step(self, scene: Scene, robot: Robot, action: np.ndarray):
        pass

    def action_post_step(self, scene: Scene, robot: Robot, action: np.ndarray):
        pass

    @abstractmethod
    def action_shape(self, robot: Robot):
        pass

    @abstractmethod
    def action_bounds(self):
        pass


class Discrete(GripperActionMode):
    """Control if the gripper is open or closed in a discrete manner.

    Action values > 0.5 will be discretised to 1 (open), and values < 0.5
    will be  discretised to 0 (closed).
    """

    def __init__(self, attach_grasped_objects: bool = True,
                 detach_before_open: bool = True):
        self._attach_grasped_objects = attach_grasped_objects
        self._detach_before_open = detach_before_open

    def _actuate(self, scene, robot: Robot, action):
        done = False
        while not done:
            done = robot.gripper.actuate(action, velocity=0.2)
            scene.pyrep.step()
            scene.task.step()

    def action(self, scene: Scene, robot: Robot, action: np.ndarray):
        assert_action_shape(action, self.action_shape(robot))
        if 0.0 > action[0] > 1.0:
            raise InvalidActionError(
                'Gripper action expected to be within 0 and 1.')
        open_condition = all(
            x > 0.9 for x in robot.gripper.get_open_amount())
        current_ee = 1.0 if open_condition else 0.0
        action = float(action[0] > 0.5)

        if current_ee != action:
            done = False
            if not self._detach_before_open:
                self._actuate(scene, robot, action)
            if action == 0.0 and self._attach_grasped_objects:
                # If gripper close action, the check for grasp.
                for g_obj in scene.task.get_graspable_objects():
                    robot.gripper.grasp(g_obj)
            else:
                # If gripper open action, the check for un-grasp.
                robot.gripper.release()
            if self._detach_before_open:
                self._actuate(scene, robot, action)
            if action == 1.0:
                # Step a few more times to allow objects to drop
                for _ in range(10):
                    scene.pyrep.step()
                    scene.task.step()

    def action_shape(self, robot: Robot) -> tuple:
        return 1,

    def action_bounds(self):
        """Get the action bounds.

        Returns: Returns the min and max of the action.
        """
        return np.array([0]), np.array([1])


class GripperJointPosition(GripperActionMode):
    """Control the target joint positions absolute or delta) of the gripper.

    The action mode opoerates in absolute mode or delta mode, where delta
    mode takes the current joint positions and adds the new joint positions
    to get a set of target joint positions. The robot uses a simple control
    loop to execute until the desired poses have been reached.
    It os the users responsibility to ensure that the action lies within
    a usuable range.

    Control if the gripper is open or closed in a discrete manner.

    Action values > 0.5 will be discretised to 1 (open), and values < 0.5
    will be  discretised to 0 (closed).
    """

    def __init__(self, attach_grasped_objects: bool = True,
                 detach_before_open: bool = True,
                 absolute_mode: bool = True):
        self._attach_grasped_objects = attach_grasped_objects
        self._detach_before_open = detach_before_open
        self._absolute_mode = absolute_mode
        self._control_mode_set = False

    def action(self, scene: Scene, robot: Robot, action: np.ndarray):
        self.action_pre_step(scene, robot, action)
        self.action_step(scene, robot, action)
        self.action_post_step(scene, robot, action)

    def action_pre_step(self, scene: Scene, robot: Robot, action: np.ndarray):
        if not self._control_mode_set:
            robot.gripper.set_control_loop_enabled(True)
            self._control_mode_set = True
        assert_action_shape(action, self.action_shape(robot))
        action = action.repeat(2)  # use same action for both joints
        a = action if self._absolute_mode else np.array(
            robot.gripper.get_joint_positions())
        robot.gripper.set_joint_target_positions(a)

    def action_step(self, scene: Scene, robot: Robot, action: np.ndarray):
        scene.step()

    def action_post_step(self, scene: Scene, robot: Robot, action: np.ndarray):
        robot.gripper.set_joint_target_positions(
            robot.gripper.get_joint_positions())

    def action_shape(self, robot: Robot) -> tuple:
        return 1,

    def action_bounds(self):
        """Get the action bounds.

        Returns: Returns the min and max of the action.
        """
        return np.array([0]), np.array([0.04])
