"""
RL-based drone controller for gym-style environment interface
"""

# Imports
import numpy as np
from typing import Optional
import gymnasium as gym

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.utils import constants
from swarm_rescue.solutions.my_drone_policies import RandomPolicy
from swarm_rescue.solutions.rl_utils import ACTION_SPACE, build_obs, obs_to_tensor, to_commands_dict
class MyDroneRL(DroneAbstract):
    """
    RL-based drone controller.
    """

    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                        misc_data=misc_data, display_lidar_graph=False,
                        **kwargs)

        self.size_area = misc_data.size_area if misc_data else None
        self.iteration:int = 0
        self.droneReady = False

        self.msg_data = None

        self.lidar_ray_angles = self.lidar().ray_angles
        self.rescue_center_pos = None
        self.init_position = None

        self.show_log_info = False
        self.show_log_warning = False
        self.show_log_error = False
        self.semantic_data = None
        self.grasped = False

        self.log_info("Drone is initialized")   

        self.action_space = ACTION_SPACE

        # Initialize RL Policy
        self.model = None
        self.policy = RandomPolicy(self.action_space)

        # Save heuristics
        self.prev_state = None
        self.prev_action = None
        # Episode stats
        self.total_reward = 0.0
        self.steps = 0

    def get_observation(self):
        return build_obs(self)

    def define_message_for_all(self):
        '''Define any custom messages to be sent to neighbors'''
        if not self.droneReady:
            return None
        self.msg_data = (self.identifier, (self.measured_gps_position(), self.measured_compass_angle()))
        return self.msg_data
    
    def is_collided(self):
        """
        Returns True if the drone collided a wall or other drones
        """
        if self.lidar_values() is None or self.semantic_values() is None:
            return False

        collided = False

        dist = min(self.lidar_values())
        semantic = self.semantic_values()
        if len(semantic) == 0:
            return dist < 30

        idx = np.argmin([x.distance for x in semantic])

        if dist < 30 and str(semantic[idx].entity_type) != "TypeEntity.WOUNDED_PERSON":
            collided = True
        return collided

    def touch_human(self):
        semantic = self.semantic_values()

        if semantic is None:
            return False

        for x in semantic:
            if x.distance < 30 and str(x.entity_type) == "TypeEntity.WOUNDED_PERSON":
                return True

        return False

    def get_distance(self, pos_a, pos_b):
        return np.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)


    def control(self) -> CommandsDict:
        """
        Main control loop for RL-based drone policy.        
        """
        # 1. Read current observation
        obs = self.get_observation()

        if self.model is not None:
            x = obs_to_tensor(obs)
            action = self.model(x).detach().cpu().numpy()[0]
        else: # Default to random policy
            action = self.policy.select_action(obs)

        action = np.clip(action, self.action_space.low, self.action_space.high)

        return to_commands_dict(action)
    
    
    def log_info(self, log=""):
        if self.show_log_info:
            print(
                f"{str(self.iteration).zfill(5)}][Drone {self.identifier}] {log}"
            )

    def log_warning(self, log=""):
        if self.show_log_warning:
            print(
                f"{str(self.iteration).zfill(5)}][Drone {self.identifier}] {log}"
            )

    def log_error(self, log=""):
        if self.show_log_error:
            print(
                f"{str(self.iteration).zfill(5)}][Drone {self.identifier}] {log}"
            )