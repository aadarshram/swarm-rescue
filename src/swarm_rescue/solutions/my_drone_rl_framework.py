"""
RL framework components for drone control. (NOT WORKING CODE. JUST AN IDEA)
Implements OpenAI Gym style environment, reward system, and policy interfaces.
"""
import math
import random
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.utils.constants import DRONE_INITIAL_HEALTH


class DroneAction(Enum):
    """Available drone actions in the discrete action space"""
    MOVE_FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    STOP = 3


@dataclass
class DroneState:
    """Represents the drone's state space"""
    lidar_data: np.ndarray  # 181 rays, 360° FOV
    gps_position: Tuple[float, float]  # x, y coordinates
    compass_angle: float  # orientation in radians
    drone_health: float  # remaining health points
    carrying_wounded: bool  # whether drone is carrying wounded person


class DroneReward:
    """Calculates rewards based on the drone's actions and state transitions"""

    def __init__(self):
        # Weights for different reward components
        # write code 
        
        # Track metrics for reward calculation
        # Write code
        pass

    def calculate(self, prev_state: DroneState, curr_state: DroneState, misc_data: MiscData) -> float:
        """Calculate reward based on state transition"""
        if prev_state is None:
            return 0.0  # No reward for initial state
        
        reward = 0.0
        
        # Implement reward components here

        return reward
 


class RandomPolicy:
    """Random action policy for initial testing"""
    
    def select_action(self, state: DroneState) -> DroneAction:
        """Randomly select an action from the action space"""
        return random.choice(list(DroneAction))


class DroneRLEnv:
    """OpenAI Gym style environment wrapper for the drone"""

    def __init__(self, drone: DroneAbstract):
        self.drone = drone
        self.reward_calculator = DroneReward()
        self.policy = RandomPolicy() # This shldnt be here. TODO
        self.current_state = None

    def get_state(self) -> DroneState:
        """Convert drone sensor data to state representation"""
        return DroneState( # Random data for now
            lidar_data=np.array(0),
            gps_position=(0.0, 0.0),
            compass_angle=0.0,
            drone_health=DRONE_INITIAL_HEALTH,
            carrying_wounded=False
        )

    def step(self, action: DroneAction) -> Tuple[DroneState, float, bool, Dict]:
        """Execute action and return new state, reward, done flag and info
        
        Args:
            action (DroneAction): Action to execute
            
        Returns:
            Tuple containing:
            - DroneState: New state after action execution
            - float: Reward for the transition
            - bool: Whether episode is done
            - Dict: Additional info (empty for now)
        """
        # Convert action enum to drone commands
        commands = CommandsDict()
        
        if action == DroneAction.MOVE_FORWARD:
            commands.forward = 1.0
        elif action == DroneAction.TURN_LEFT:
            commands.rotational_velocity = -1.0
        elif action == DroneAction.TURN_RIGHT:
            commands.rotational_velocity = 1.0
        # STOP action means no commands (zero velocity)

        # Save previous state for reward calculation
        prev_state = self.current_state
        
        # Execute action
        self.drone.controller.send_commands(commands)
        
        # Get new state
        self.current_state = self.get_state()
        
        # Calculate reward
        reward = self.reward_calculator.calculate(
            prev_state, 
            self.current_state,
            self.drone.misc_data
        )
        
        # Check if episode is done (drone destroyed)
        done = (self.current_state.drone_health <= 0)
        
        return self.current_state, reward, done, {}