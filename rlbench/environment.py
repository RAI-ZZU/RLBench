import importlib
import numpy as np
from os.path import exists, dirname, abspath, join
from typing import Type, List,Union

from pyrep import PyRep
from pyrep.objects import VisionSensor
from pyrep.robots.arms.panda import Panda
from pyrep.robots.arms.ur5 import UR5
from pyrep.const import ObjectType

from rlbench import utils
from rlbench.action_modes.action_mode import ActionMode
from rlbench.backend.const import *
from rlbench.backend.robot import Robot,Robots
from rlbench.backend.scene import Scene
from rlbench.backend.task import Task
from rlbench.const import SUPPORTED_ROBOTS, ROBOT_ROLES
from rlbench.demo import Demo
from rlbench.observation_config import ObservationConfig
from rlbench.sim2real.domain_randomization import RandomizeEvery, \
    VisualRandomizationConfig, DynamicsRandomizationConfig
from rlbench.sim2real.domain_randomization_scene import DomainRandomizationScene
from rlbench.task_environment import TaskEnvironment


DIR_PATH = dirname(abspath(__file__))

class Environment(object):
    """Each environment has a scene."""

    def __init__(self,
                 action_mode: Union[ActionMode,List[ActionMode]],
                 dataset_root: str = '',
                 obs_config: ObservationConfig = ObservationConfig(),
                 headless: bool = False,
                 static_positions: bool = False,
                 robot_setup: str = 'panda',
                 randomize_every: RandomizeEvery = None,
                 frequency: int = 1,
                 visual_randomization_config: VisualRandomizationConfig = None,
                 dynamics_randomization_config: DynamicsRandomizationConfig = None,
                 attach_grasped_objects: bool = True,
                 shaped_rewards: bool = False,
                 floating_cam:bool=False
                 ):

        self._dataset_root = dataset_root
        self._obs_config = obs_config
        self._headless = headless
        self._static_positions = static_positions

        robot_setup = list(filter(None,robot_setup.lower().split(',')))
        assert len(robot_setup) <= 2
        if len(robot_setup)==1:
            self._robot_setup = {ROBOT_ROLES[0]:robot_setup[0]}
            self._robot_default = {ROBOT_ROLES[0]:'panda'}

        elif len(robot_setup)>1:
            self._robot_setup = {ROBOT_ROLES[0]:robot_setup[0],ROBOT_ROLES[1]:robot_setup[1]}
            self._robot_default = {ROBOT_ROLES[0]:'panda_blind',ROBOT_ROLES[1]:'ur5'}



        action_mode = action_mode if isinstance(action_mode,List) else [action_mode]
        #assert(len(robot_setup) == len(action_mode))
        if len(robot_setup)>len(action_mode):
            action_mode = action_mode*len(robot_setup)
        self._action_mode = dict(zip(self._robot_setup.keys(), action_mode))
        # 

        self._randomize_every = randomize_every
        self._frequency = frequency
        self._visual_randomization_config = visual_randomization_config
        self._dynamics_randomization_config = dynamics_randomization_config
        self._attach_grasped_objects = attach_grasped_objects
        self._shaped_rewards = shaped_rewards
        self._floating_cam = floating_cam

        for robot_name in self._robot_setup.values():
            if robot_name not in SUPPORTED_ROBOTS.keys():
                raise ValueError('robot_configuration must be one of %s' %
                                str(SUPPORTED_ROBOTS.keys()))

        if (randomize_every is not None and
                visual_randomization_config is None and
                dynamics_randomization_config is None):
            raise ValueError(
                'If domain randomization is enabled, must supply either '
                'visual_randomization_config or dynamics_randomization_config')

        self._check_dataset_structure()
        self._pyrep = None
        self._robot = Robots()
        self._scene = None
        self._prev_task = None

    def _check_dataset_structure(self):
        if len(self._dataset_root) > 0 and not exists(self._dataset_root):
            raise RuntimeError(
                'Data set root does not exists: %s' % self._dataset_root)

    def _string_to_task(self, task_name: str):
        task_name = task_name.replace('.py', '')
        try:
            class_name = ''.join(
                [w[0].upper() + w[1:] for w in task_name.split('_')])
            mod = importlib.import_module("rlbench.tasks.%s" % task_name)
        except Exception as e:
            raise RuntimeError(
                'Tried to interpret %s as a task, but failed. Only valid tasks '
                'should belong in the tasks/ folder' % task_name) from e
        return getattr(mod, class_name)

    def _init_robot(self,role:str, robot_name:str, action_mode:ActionMode) -> Robot:
        arm_class, gripper_class, _ = SUPPORTED_ROBOTS[robot_name]
        if role == 'worker_arm':
            # We assume the panda is already loaded in the scene.
            if robot_name == self._robot_default['worker_arm']:
                # Remove the panda from the scene
                panda_arm = Panda()
                panda_pos = panda_arm.get_position()
                panda_ori = panda_arm.get_orientation()
                panda_arm.remove()

                arm_path = join(DIR_PATH, 'robot_ttms', robot_name + '.ttm')
                self._pyrep.import_model(arm_path)
                arm, gripper = arm_class(), gripper_class()
                arm.set_position(panda_pos)
                arm.set_orientation(panda_ori)
            else:
                arm, gripper = arm_class(), gripper_class()
        elif role =='vision_arm':
            # We assume the ur5_blind is already loaded in the scene.
            if robot_name == self._robot_default['vision_arm']:
                # Remove the ur5_blind from the scene
                ur5_arm = UR5()
                ur5_pos = ur5_arm.get_position()
                ur5_ori = ur5_arm.get_orientation()
                ur5_arm.remove()

                arm_path = join(DIR_PATH, 'robot_ttms', robot_name + '.ttm')
                self._pyrep.import_model(arm_path)
                arm, gripper = arm_class(), gripper_class()
                arm.set_position(ur5_pos)
                arm.set_orientation(ur5_ori)
            else:
                arm, gripper = arm_class(), gripper_class()

            gripper.set_model_collidable(False)
            gripper.set_model_detectable(False)
            gripper.set_model_measurable(False)
            gripper.set_model_renderable(False)
            gripper.set_model_respondable(False)
            components = gripper.get_objects_in_tree(ObjectType.SHAPE, exclude_base=False)
            [component.remove() for component in components 
                if 'Visible' in component.get_name()]
            #print("Here")


            

        return Robot(arm, gripper,self._pyrep, role, action_mode)

    def launch(self):
        if self._pyrep is not None:
            raise RuntimeError('Already called launch!')
        self._pyrep = PyRep()

        if "vision_arm" in self._robot_setup.keys():
            if self._floating_cam:
                self._pyrep.launch(join(DIR_PATH,FLOATING_VP_TTT_FILE), headless=self._headless)
            else:
                self._pyrep.launch(join(DIR_PATH, STATIC_TTT_FILE), headless=self._headless)
        else:
            self._pyrep.launch(join(DIR_PATH, TTT_FILE), headless=self._headless)

        # 
        for role,robot_name in self._robot_setup.items():
            if "worker" in role:
                self._robot.worker_arm = self._init_robot(role,robot_name,self._action_mode[role])
            elif "vision" in role:
                self._robot.vision_arm = self._init_robot(role,robot_name,self._action_mode[role])
        

        if self._randomize_every is None:
            self._scene = Scene(
                self._pyrep, self._robot, self._obs_config, self._robot_setup)
        else:
            self._scene = DomainRandomizationScene(
                self._pyrep, self._robot, self._obs_config, self._robot_setup,
                self._randomize_every, self._frequency,
                self._visual_randomization_config,
                self._dynamics_randomization_config)
        # 
        self._robot.set_scene(scene=self._scene)

        self._robot.set_control_mode()


    def shutdown(self):
        if self._pyrep is not None:
            self._pyrep.shutdown()
        self._pyrep = None

    def get_task(self, task_class: Type[Task]) -> TaskEnvironment:

        # If user hasn't called launch, implicitly call it.
        if self._pyrep is None:
            self.launch()

        self._scene.unload()
        task = task_class(self._pyrep, self._robot)
        self._prev_task = task
        return TaskEnvironment(
            self._pyrep, self._robot, self._scene, task,
            self._action_mode, self._dataset_root, self._obs_config,
            self._static_positions, self._attach_grasped_objects,
            self._shaped_rewards)

    @property
    def action_shape(self):
        action_shape = self._robot.get_action_shape()
        return action_shape

    def get_demos(self, task_name: str, amount: int,
                  variation_number=0,
                  image_paths=False,
                  random_selection: bool = True,
                  from_episode_number: int = 0) -> List[Demo]:

        if self._dataset_root is None or len(self._dataset_root) == 0:
            raise RuntimeError(
                "Can't ask for a stored demo when no dataset root provided.")
        demos = utils.get_stored_demos(
            amount, image_paths, self._dataset_root, variation_number,
            task_name, self._obs_config, random_selection, from_episode_number)
        return demos

    def get_scene_data(self) -> dict:
        """Get the data of various scene/camera information.

        This temporarily starts the simulator in headless mode.

        :return: A dictionary containing scene data.
        """

        def _get_cam_info(cam: VisionSensor):
            if not cam.still_exists():
                return None
            intrinsics = cam.get_intrinsic_matrix()
            return dict(
                intrinsics=intrinsics,
                near_plane=cam.get_near_clipping_plane(),
                far_plane=cam.get_far_clipping_plane(),
                extrinsics=cam.get_matrix())

        headless = self._headless
        self._headless = True
        self.launch()
        d = dict(
            left_shoulder_camera=_get_cam_info(
                self._scene._cam_over_shoulder_left),
            right_shoulder_camera=_get_cam_info(
                self._scene._cam_over_shoulder_right),
            front_camera=_get_cam_info(self._scene._cam_front),
            wrist_camera=_get_cam_info(self._scene._cam_wrist),
            active_camera=_get_cam_info(self._scene._cam_active),
            overhead_camera=_get_cam_info(self._scene._cam_overhead)
        )
        self.shutdown()
        self._headless = headless
        return d
