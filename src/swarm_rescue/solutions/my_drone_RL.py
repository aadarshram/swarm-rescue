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

        self.msg_data = None

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
        self.model = None # Placeholder for loading a trained model
        self.policy = RandomPolicy(self.action_space)
        self.use_time_feature = False  # Will be set based on model observation space
        self.max_steps = misc_data.max_timestep_limit if misc_data else 500 # Default max steps for time normalization
        
        # Load trained model if path provided
        self.model_path = kwargs.get('model_path', None)
        if self.model_path:
            self.load_model(self.model_path)

        # Save heuristics
        self.prev_state = None
        self.prev_action = None
        
        # Episode stats
        self.total_reward = 0.0
        self.steps = 0
    
    def load_model(self, model_path):
        """Load a trained SB3 model and configure observation space"""
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path)
            
            # Verify observation space
            # Expected: 180 (lidar) + 105 (semantic) + 3 (pose) + 2 (vel) + 1 (grasper) = 291
            obs_dim = self.model.observation_space.shape[0]
            if obs_dim == 291:
                self.log_info(f"Loaded model with correct observation space (obs_dim={obs_dim})")
            else:
                self.log_warning(f"Model observation dimension {obs_dim} != expected 291")
            
            self.log_info(f"Loaded trained model from {model_path}")
        except Exception as e:
            self.log_error(f"Failed to load model from {model_path}: {e}")
            self.model = None

    def get_observation(self):
        return build_obs(self)

    def define_message_for_all(self):
        '''Define any custom messages to be sent to neighbors'''
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

        if dist < 30:
            entity_type = str(semantic[idx].entity_type)
            if entity_type != "TypeEntity.WOUNDED_PERSON" and entity_type != "TypeEntity.RESCUE_CENTER":
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
            # For SB3 models, flatten the observation (291 dimensions)
            from swarm_rescue.solutions.rl_utils import flatten_observation
            flat_obs = flatten_observation(obs)
            
            # No time feature - just use base observation (291 dims)
            # Simpler and works better for variable-length episodes
            
            # Predict action using SB3 model
            action, _states = self.model.predict(flat_obs, deterministic=True)
        else: # Default to random policy
            action = self.policy.select_action(obs)

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.steps += 1

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