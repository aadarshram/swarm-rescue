"""
RL-based drone controller implementation using the modular RL framework.
Currently implements a random policy baseline for testing.
"""
import math
import random
from typing import Optional, Dict, Any

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.utils.utils import normalize_angle

# New imports
from swarm_rescue.solutions.my_drone_rl_framework import DroneRLEnv, DroneAction


class MyDroneRL(DroneAbstract):
    """
    Reinforcement Learning based drone controller.
    Uses OpenAI Gym style environment interface from my_drone_rl_framework.
    Currently implements a random policy baseline that will be replaced with 
    trained RL policies in future iterations.
    """
    counterStraight: int
    angleStopTurning: float
    distStopStraight: float
    isTurning: bool

    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        """
        Initialize the RL drone controller
        
        Args:
                identifier (Optional[int]): Drone identifier.
                misc_data (Optional[MiscData]): Miscellaneous data.
                **kwargs: Additional keyword arguments.
        """
        super().__init__(identifier=identifier,
                        misc_data=misc_data, display_lidar_graph=False,
                        **kwargs)
        
        self.counterStraight = 0
        self.angleStopTurning = random.uniform(-math.pi, math.pi)
        self.distStopStraight = random.uniform(10, 50)
        self.isTurning = False

        # Initialize RL environment
        # self.env = DroneRLEnv(self)
        # self.policy = RandomDronePolicy()
        # self.current_state = None
        
        # Episode stats
        # self.total_reward = 0.0
        # self.steps = 0

    def define_message_for_all(self) -> None:
        """
        No communication needed for a random drone.
        """
        pass

    def process_lidar_sensor(self) -> bool:
        """
        Returns True if the drone collided with an obstacle.

        Returns:
            bool: True if collision detected, False otherwise.
        """
        if self.lidar_values() is None:
            return False

        collided = False
        dist = min(self.lidar_values())

        if dist < 40:
            collided = True

        return collided
    
    def control(self) -> CommandsDict:
        """
        Main control loop for RL-based drone policy.
        
        This function follows the standard RL cycle:
        1. Observe current state
        2. Select action using learned policy
        3. Execute action and get feedback (next state, reward, done)
        4. Update internal stats and reset if episode ends

        Pipeline logic:
        -> Env returns current state of drone
        -> current state + history(previous state, action) is input to RLEnv class -> returns reward, done, info (next_state is known which is current_state)
        -> Policy selects action based on current state
        -> Action is converted to drone commands and executed
        -> Stats are updated and episode termination is handled
        
        ALL CLASSES AND LOGIC FOR THE RL FRAMEWORK IS TO BE IMPLEMENTED IN my_drone_rl_framework.py
        Returns:
            CommandsDict: The commands to execute on the drone
        """

        # Implement an abstract env class here that processes the current drone state into a gym style compatible env return.
        # Ie, if the swarm rescue framework returns X, Y , Z implement a conversion to the gym style state representation
        # Eg: next state, reward, done, info so that we use as env.step(action) 
        # For this possibly you need to save a history of previous state and action to compute reward properly.
        # The env abstraction should be implemented in my_drone_rl_framework.py
        # this file initializes it as self.env = DroneRLEnv(self) in the init function
        # and then only steps through it here in control() as env.step(action)

        # state from them -> our current state gym style self.current_state


        # Policy decides what to do given the current observation
        # action = self.policy.select_action(self.current_state)
        # Policy is initialized in init function. the select action function is defined in my_drone_rl_framework.py under your_policy class 
        # For now, do a random policy for testing
        command_straight: CommandsDict = {"forward": 1.0,
                                          "lateral": 0.0,
                                          "rotation": 0.0,
                                          "grasper": 0}
        action = command_straight

        # Convert abstract actions to command sent to them
        # currently it is directly a command as action for testing
        commands = action

        # ---- 5. Stats & Bookkeeping ----
        # accumulate reward
        # increment steps
        # update current state
        # terminal condition based on done

        # if done
        # Collect all stats and return and then reset everything.

        return commands