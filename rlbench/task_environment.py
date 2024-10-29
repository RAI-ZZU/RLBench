import logging
from typing import List, Callable,Dict

import numpy as np
from pyrep import PyRep
from pyrep.const import ObjectType
from rlbench import utils
from rlbench.action_modes.action_mode import ActionMode
from rlbench.backend.exceptions import BoundaryError, WaypointError, \
    TaskEnvironmentError
from rlbench.backend.observation import Observation
from rlbench.backend.robot import Robot,Robots
from rlbench.backend.scene import Scene
from rlbench.backend.task import Task
from rlbench.backend.const import TABLE_COORD
from rlbench.demo import Demo
from rlbench.observation_config import ObservationConfig
import  rlbench.backend.utils as utils


_DT = 0.05
_MAX_RESET_ATTEMPTS = 40
_MAX_DEMO_ATTEMPTS = 20


class TaskEnvironment(object):

    def __init__(self,
                 pyrep: PyRep,
                 robot: Robots,
                 scene: Scene,
                 task: Task,
                 action_mode: Dict[str,ActionMode],
                 dataset_root: str,
                 obs_config: ObservationConfig,
                 static_positions: bool = False,
                 attach_grasped_objects: bool = True,
                 shaped_rewards: bool = False
                 ):
        self._pyrep = pyrep
        self._robot = robot
        self._scene = scene
        self._task = task
        self._variation_number = 0
        self._action_mode = action_mode
        self._dataset_root = dataset_root
        self._obs_config = obs_config
        self._static_positions = static_positions
        self._attach_grasped_objects = attach_grasped_objects
        self._shaped_rewards = shaped_rewards
        self._reset_called = False
        self._prev_ee_velocity = None
        self._enable_path_observations = False

        self._scene.load(self._task)
        self._pyrep.start()

    def get_name(self) -> str:
        return self._task.get_name()

    def sample_variation(self) -> int:
        self._variation_number = np.random.randint(
            0, self._task.variation_count())
        return self._variation_number

    def set_variation(self, v: int) -> None:
        if v >= self.variation_count():
            raise TaskEnvironmentError(
                'Requested variation %d, but there are only %d variations.' % (
                    v, self.variation_count()))
        self._variation_number = v

    def variation_count(self) -> int:
        return self._task.variation_count()

    def reset(self, demo = None) -> (List[str], Observation):
        self._scene.reset()
        try:
            place_demo = demo != None and hasattr(demo, 'num_reset_attempts') and demo.num_reset_attempts != None
            desc = self._scene.init_episode(
                self._variation_number, max_attempts=_MAX_RESET_ATTEMPTS,
                randomly_place=not self._static_positions, place_demo=place_demo)
        except (BoundaryError, WaypointError) as e:
            raise TaskEnvironmentError(
                'Could not place the task %s in the scene. This should not '
                'happen, please raise an issues on this task.'
                % self._task.get_name()) from e

        self._reset_called = True
        # Returns a list of descriptions and the first observation
        return desc, self._scene.get_observation()

    def get_observation(self) -> Observation:
        return self._scene.get_observation()

    def step(self, action:Dict[str,np.ndarray]) -> (Observation, int, bool):


        # returns observation, reward, done, info
        if not self._reset_called:
            raise RuntimeError(
                "Call 'reset' before calling 'step' on a task.")
        self._robot.action(action)
        success, terminate = self._task.success()
        reward = float(success)
        if self._shaped_rewards:
            reward = self._task.reward()
            if reward is None:
                raise RuntimeError(
                    'User requested shaped rewards, but task %s does not have '
                    'a defined reward() function.' % self._task.get_name())
        return self._scene.get_observation(), reward, terminate

    def step_active_cam(self, action:Dict[str,np.ndarray],eval:bool=False) -> (Observation, int, bool):


        # returns observation, reward, done, info
        if not self._reset_called:
            raise RuntimeError(
                "Call 'reset' before calling 'step' on a task.")
        #self._robot.action(action)
        assert action["vision_arm"].shape[0]==7
        viewpoint_pose = action["vision_arm"]

        if eval:
            current_vp_spher = utils.cartesian_to_spherical(wtc_pos=self._scene._cam_active.get_position(),
                                                            fixation=TABLE_COORD)
            target_vp_spher = utils.cartesian_to_spherical(wtc_pos=viewpoint_pose[:3],
                                                            fixation=TABLE_COORD)
            vp_trajectory = self._scene.get_next_floating_vp_trajectory(start_vp=current_vp_spher,
                                                                    target_vp=target_vp_spher,
                                                                    delta_phi=2.0,
                                                                    random_sample_theta=False)
            for vp_point in vp_trajectory:
                self._scene.show_virtual_cam(vp_point)

                self._scene._cam_active.set_pose(vp_point)
                #vp_trans_obs = self._scene.get_observation()
                self._scene.step()
        else:
            self._scene._cam_active.set_pose(viewpoint_pose)
            self._scene.show_point(viewpoint_pose)
            self._scene.step()
            


        success, terminate = self._task.success()
        reward = float(success)
        if self._shaped_rewards:
            reward = self._task.reward()
            if reward is None:
                raise RuntimeError(
                    'User requested shaped rewards, but task %s does not have '
                    'a defined reward() function.' % self._task.get_name())
        return self._scene.get_observation(), reward, terminate


    def get_demos(self, amount: int, live_demos: bool = False,
                  human_demos: bool = False,floating_cam:bool = False, 
                  image_paths: bool = False,
                  callable_each_step: Callable[[Observation], None] = None,
                  max_attempts: int = _MAX_DEMO_ATTEMPTS,
                  random_selection: bool = True,
                  from_episode_number: int = 0,
                  viewpoint_env_bounds: list=[1.3,20,-135,1.3,60,135],
                  ) -> List[Demo]:
        """Negative means all demos"""

        if not live_demos and (self._dataset_root is None
                               or len(self._dataset_root) == 0):
            raise RuntimeError(
                "Can't ask for a stored demo when no dataset root provided.")

        if not live_demos:
            if self._dataset_root is None or len(self._dataset_root) == 0:
                raise RuntimeError(
                    "Can't ask for stored demo when no dataset root provided.")
            demos = utils.get_stored_demos(
                amount, image_paths, self._dataset_root, self._variation_number,
                self._task.get_name(), self._obs_config,
                random_selection, from_episode_number)
        else:
            ctr_loop = [robot_.arm.joints[0].is_control_loop_enabled() for robot_ in self._robot.yield_robot()]

            [robot_.arm.set_control_loop_enabled(True) for robot_ in self._robot.yield_robot()]
            demos = self._get_live_demos(amount=amount, callable_each_step=callable_each_step,
                                         human_demos=human_demos, floating_cam = floating_cam, 
                                         viewpoint_env_bounds=viewpoint_env_bounds,
                                         max_attempts=max_attempts)
            [robot_.arm.set_control_loop_enabled(ctr_loop_)
                for ctr_loop_, robot_ in zip(ctr_loop,list(self._robot.yield_robot()))]

        return demos

    def _get_live_demos(self, amount: int,
                        callable_each_step: Callable[
                            [Observation], None] = None,
                        human_demos: bool = False,
                        floating_cam:bool = False, 
                        viewpoint_env_bounds: list=[1.3,20,-135,1.3,60,135],
                        max_attempts: int = _MAX_DEMO_ATTEMPTS) -> List[Demo]:
        demos = []
        for i in range(amount):
            attempts = max_attempts
            while attempts > 0:
                random_seed = np.random.get_state()
                self.reset()
                try:

                    demo = self._scene.get_demo(
                        callable_each_step=callable_each_step,
                        human_demos=human_demos,
                        viewpoint_env_bounds=viewpoint_env_bounds,
                        floating_cam=floating_cam)
                    demo.random_seed = random_seed
                    demos.append(demo)
                    break
                except Exception as e:
                    attempts -= 1
                    logging.info('Bad demo. ' + str(e))
            if attempts <= 0:
                raise RuntimeError(
                    'Could not collect demos. Maybe a problem with the task?')
        return demos

    def reset_to_demo(self, demo: Demo) -> (List[str], Observation):
        demo.restore_state()
        variation_index = demo._observations[0].misc["variation_index"]
        self.set_variation(variation_index)
        return self.reset(demo)
