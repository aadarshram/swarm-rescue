"""
RL-based drone controller for gym-style environment interface
"""

from typing import Optional, List, Tuple
import numpy as np

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.misc_data import MiscData

# New imports
from swarm_rescue.solutions.my_drone_rl_framework import DroneRLEnv, RandomPolicy


class MyDroneRL(DroneAbstract):
    """
    RL-based drone controller.
    Wraps DroneRLEnv and executes actions every control step.
    """

    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                        misc_data=misc_data, display_lidar_graph=False,
                        **kwargs)

        # RL Environment wrapper
        self.env = DroneRLEnv(self)
        self.current_state = None

        # RL Policy
        self.policy = RandomPolicy()

        # Episode stats
        self.total_reward = 0.0
        self.steps = 0

        # RL Env runs one step behind. Save previous states for reward calculation.
        self.prev_state = None
        self.prev_action = None
    
    def define_message_for_all(self) -> None:
        '''Define any custom messages to be sent to neighbors'''
        pass
    
    def control(self) -> CommandsDict:
        """
        Main control loop for RL-based drone policy.        
        This function DEVIATES from the standard RL cycle:
        (Since it implements a one-step behind logic to fit into the existing simulator control loop)
    
        1. Observe current state
        2. Get feedback for previous step and compute previous step heuristics (saves one-step history)
        3. Update internal stats and reset if episode ends for previous step
        4. Select current action using learned policy
        5. Execute action
        
        Returns:
            CommandsDict: The commands to execute on the drone
        """
        # Read current state
        if self.current_state is None:
            self.current_state = self.env.get_state()

        # Env based calculations 
        # TODO: [Arnav] Find gripper action from state observation instead of policy
        grasper = 0
        # Compute previous step reward if applicable
        # RL Env is one step behind the drone control loop
        reward, done, info = 0.0, False, {}
        if self.prev_state is not None and self.prev_action is not None:
            # Proxy env step for the RL framework
            reward, done, info = self.env.step(self.prev_state, self.prev_action, self.current_state) # TODO: implement info

        # Reset if episode termination
        if done:
            # TODO log statistics here
            self.total_reward = 0.0
            self.steps = 0
            self.prev_state = None
            self.prev_action = None
            self.current_state = None

        # Track episode stats
        self.total_reward += reward
        self.steps += 1  

        # Select action using policy
        action = self.policy.select_action(self.current_state) 
        action.grasper = grasper

        # Save current state and action for next step reward calculation
        self.prev_state = self.current_state
        self.prev_action = action

        return action.to_command_dict()
    