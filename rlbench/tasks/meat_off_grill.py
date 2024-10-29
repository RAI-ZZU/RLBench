from typing import List, Tuple
import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import NothingGrasped, DetectedCondition
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary

MEAT = ['chicken', 'steak']


class MeatOffGrill(Task):

    def init_task(self) -> None:
        self._steak = Shape('steak')
        self._chicken = Shape('chicken')
        self._success_sensor = ProximitySensor('success')
        self.register_graspable_objects([self._chicken, self._steak])
        self._w1 = Dummy('waypoint1')
        self._w1z= self._w1.get_position()[2]
        self._spawn_boundary = SpawnBoundary([Shape('spawn_boundary')])
    def init_episode(self, index: int) -> List[str]:
        conditions = [NothingGrasped(self.robot.worker_arm.gripper)]
        self._spawn_boundary.clear()
        self._spawn_boundary.sample(self._chicken,min_rotation=(0,0,0),max_rotation=(0,0,0))

        if index == 0:
            x, y, _ = self._chicken.get_position()
            self._w1.set_position([x, y, self._w1z])
            conditions.append(
                DetectedCondition(self._chicken, self._success_sensor))
        else:
            x, y, _ = self._steak.get_position()
            self._w1.set_position([x, y, self._w1z])
            conditions.append(
                DetectedCondition(self._steak, self._success_sensor))
        self.register_success_conditions(conditions)
        return ['take the %s off the grill' % MEAT[index],
                'pick up the %s and place it next to the grill' % MEAT[index],
                'remove the %s from the grill and set it down to the side'
                % MEAT[index]]

    def variation_count(self) -> int:
        return 2
    
    # def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
    #     return [0, 0, np.pi*5/4], [0, 0, np.pi*6/4]

