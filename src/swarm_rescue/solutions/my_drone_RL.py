"""
RL-based drone controller for gym-style environment interface
"""

# Imports
import numpy as np
from typing import Optional
import gymnasium as gym
import arcade

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.solutions.my_drone_policies import RandomPolicy
from swarm_rescue.solutions.rl_utils import ACTION_SPACE, OBSERVATION_SPACE, build_obs, obs_to_tensor, to_commands_dict, flatten_observation

class MyDroneRL(DroneAbstract):
    """
    RL-based drone controller. Uses RL policy to select actions based on observations.
    """

    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                        misc_data=misc_data, display_lidar_graph=False,
                        **kwargs)
        # From super we get
        # _misc_data
        # self.elapsed_timestep (initialized to 0), self.elapsed_walltime (initialized to 0), self.size_area, self._drone_health (initialized to max health from constants), self.is_inside_return_area 
        # self.grasped_wounded_persons, self.communicator_is_disabled
        # semantic_values, lidar_values, lidar_ray_angles, gps_values, compass_values, odometer_values, measured_gps_position (if disabled none), measured_compass_angle (if disabled none), measured_velocity, measured_angular_velocity, 
        # For logging, true_position, true_angle, true_velocity, true_angular_velocity

        # Avoid redundant attributes
        
        # Define drones action and observation spaces
        self.action_space = ACTION_SPACE
        self.observation_space = OBSERVATION_SPACE
        # Initialize drone brain
        # Load trained model if path provided
        self.model_path = kwargs.get('model_path', None)
        self.model = None
        if self.model_path:
            self.load_model(self.model_path)
        if self.model is None:
            self.model = RandomPolicy(self.action_space)
        self.policy = self.model # May later update into a superset of self.model for drone action prediction

        # Current state
        self.current_state = None
        # Map env data
        self.grid = None
        # Comms data
        self.msg_data = None

    def load_model(self, model_path):
        """Load a trained SB3 model and configure observation space"""
        try:
            from stable_baselines3 import PPO # Assume PPO for now
            self.model = PPO.load(model_path)
        except Exception as e:
            self.model = None

    def get_observation(self):
        self.current_state = build_obs(self)
        return self.current_state

    def define_message_for_all(self):
        '''Define any custom messages to be sent to neighbors'''
        if self.current_state is not None:
            self.msg_data = (self.identifier, self.current_state["pose"]) 
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
        idx = np.argmin([x.distance for x in semantic])

        if dist < 30:# Collision with wall or drone
            entity_type = str(semantic[idx].entity_type)
            if entity_type != "TypeEntity.WOUNDED_PERSON" and entity_type != "TypeEntity.RESCUE_CENTER":
                collided = True
        
            # TODO: Override policy with rotation of +pi rad? Ensures drone gets unstuck? 
            # use self.policy for any override to self.model actions
        return collided

    def touch_human(self):
        '''Returns True if the drone is close enough to a wounded person'''
        semantic = self.semantic_values()
        if semantic is None:
            return False
        for x in semantic:
            if x.distance < 30 and str(x.entity_type) == "TypeEntity.WOUNDED_PERSON":
                return True
        return False

    def get_distance(self, pos_a, pos_b):
        return np.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)

    def timeout(self):
        '''Returns True if the drone is almost timed out'''
        timestep_out = self.elapsed_timestep > 0.9 * self._misc_data.max_timestep_limit
        walltime_out = self.elapsed_walltime > 0.9 * self._misc_data.max_walltime_limit

        # TODO: If you're out of time, need to head to return area. Override policy? Since, return area is not known and is to be checked via bool. self.is_inside_return_area
        return timestep_out or walltime_out
    
    def control(self) -> CommandsDict:
        """
        Main control loop for RL-based drone policy.        
        """
        # Read current observation
        obs = self.get_observation()

        if self.model is not None:
            flat_obs = flatten_observation(obs)
            action, _states = self.model.predict(flat_obs, deterministic=True)
        else: # Default to random policy
            action = self.policy.predict(obs)

        action = np.clip(action, self.action_space.low, self.action_space.high)
        return to_commands_dict(action)