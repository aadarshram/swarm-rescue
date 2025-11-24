"""
RL framework components for drone control.
Implements OpenAI Gym style environment, reward system, and policy interfaces.
"""
import math
import random
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.utils.constants import DRONE_INITIAL_HEALTH


# ================================================================
# ACTION CLASS
# ================================================================
@dataclass
class DroneAction:
    """
    Continuous drone action space.

    forward, lateral, and rotation are all normalized to [-1, 1].
    The simulator will internally map these to physical forces/velocities.
    """
    forward: float
    lateral: float
    rotation: float
    grasper: int

    def clip(self) -> "DroneAction":
        """Clip each control signal to its valid range."""
        self.forward = float(np.clip(self.forward, -1.0, 1.0))
        self.lateral = float(np.clip(self.lateral, -1.0, 1.0))
        self.rotation = float(np.clip(self.rotation, -1.0, 1.0))
        return self

    def to_command_dict(self) -> CommandsDict:
        """
        Convert this action into the simulator's CommandsDict object,
        which is directly interpreted by the drone's low-level controller.
        """
        self.clip()
        commands : CommandsDict = {
            "forward": self.forward,
            "lateral": self.lateral,
            "rotation": self.rotation,
            "grasper": self.grasper
        }
        return commands        

# ================================================================
# STATE CLASS
# ================================================================
@dataclass
class DroneState:
    """
    Represents the full observable state of a drone in the swarm-rescue simulation.

    Attributes:
        gps_x, gps_y (float): Global GPS position in pixels.
        gps_theta (float): Orientation from compass sensor in radians [-π, π].

        dist_travel (float): Distance moved since last timestep (pixels).
        alpha (float): Relative movement direction since last timestep (radians).
        theta (float): Orientation change since last timestep (radians).

        semantic_data (List[Dict[str, Any]]): 35 semantic rays, each entry containing:
            {
              "distance": float,   # distance in pixels, [0, 200]
              "angle": float,      # ray angle in radians [-π, π]
              "entity_type": str,  # e.g., "WoundedPerson", "RescueCenter", "Drone"
              "grasped": bool      # whether that object is being grasped
            }
        comm_msg (Optional[Any]): Placeholder for communication data (unused for now).
        health (int): Drone health points (integer, decreases on collision).
        grasped (bool): True if currently carrying or holding an object.
        lidar_scan (np.ndarray): 181 Lidar distance readings (pixels), 360° FOV.
        map_matrix (Optional[np.ndarray]): Full exploration map (provided by environment).
    """

    gps_x: float
    gps_y: float
    gps_theta: float

    dist_travel: float
    alpha: float
    theta: float

    semantic_data: List[Dict[str, Any]]
    comm_msg: Optional[Any]

    health: int
    grasped: bool

    lidar_scan: np.ndarray
    map_matrix: Optional[np.ndarray] = None

    @classmethod
    def from_drone(cls, drone: DroneAbstract, map_matrix: Optional[np.ndarray] = None) -> "DroneState":
        """
        Construct a DroneState directly from the simulator's DroneAbstract object.
        Pulls all sensor and internal data to create a full observation snapshot.
        """
        gps_sensor = getattr(drone, "gps_sensor", None)
        compass_sensor = getattr(drone, "compass_sensor", None)
        odometer = getattr(drone, "odometer_sensor", None)
        lidar = getattr(drone, "lidar_sensor", None)
        semantic = getattr(drone, "semantic_sensor", None)

        # Default fallbacks
        gps_x, gps_y = (0.0, 0.0)
        gps_theta = 0.0
        dist_travel = alpha = theta = 0.0
        lidar_scan = np.zeros(181, dtype=np.float32)
        semantic_data: List[Dict[str, Any]] = []

        # GPS position (x, y)
        if gps_sensor is not None:
            gps_x, gps_y = gps_sensor.position

        # Orientation angle
        if compass_sensor is not None:
            gps_theta = compass_sensor.angle

        # Odometer delta values
        if odometer is not None:
            dist_travel = getattr(odometer, "delta_distance", 0.0)
            alpha = getattr(odometer, "delta_alpha", 0.0)
            theta = getattr(odometer, "delta_theta", 0.0)

        # Lidar readings (181 rays)
        if lidar is not None and hasattr(lidar, "data"):
            lidar_scan = np.array(lidar.data, dtype=np.float32)

        # Semantic sensor output (35 rays)
        if semantic is not None and hasattr(semantic, "data"):
            raw_data = semantic.data
            # Normalize to a consistent list of dicts
            semantic_data = [
                {
                    "distance": getattr(ray, "distance", 0.0),
                    "angle": getattr(ray, "angle", 0.0),
                    "entity_type": getattr(ray, "entity_type", None),
                    "grasped": getattr(ray, "grasped", False),
                }
                for ray in raw_data
            ]

        # Drone health and grasping state
        health = getattr(drone, "health_points", DRONE_INITIAL_HEALTH)
        grasped = getattr(drone, "grasped", False)

        # Communication placeholder
        comm_msg = None

        return cls(
            gps_x=gps_x,
            gps_y=gps_y,
            gps_theta=gps_theta,
            dist_travel=dist_travel,
            alpha=alpha,
            theta=theta,
            semantic_data=semantic_data,
            comm_msg=comm_msg,
            health=health,
            grasped=grasped,
            lidar_scan=lidar_scan,
            map_matrix=map_matrix,
        )

# ================================================================
# REWARD, POLICY, AND ENVIRONMENT
# ================================================================
class DroneReward:
    """Calculates rewards based on the drone's actions and state transitions"""

    def __init__(self):
        pass

    def calculate(self, current_state: DroneState, action: DroneAction, next_state: DroneState) -> float:
        """Calculate reward based on state transition"""
        if current_state is None:
            return 0.0  # No reward for initial state 
        
        reward = -1.0 # penalize time step
        return reward


class RandomPolicy:
    """Simple random continuous policy for smoke testing."""

    def select_action(self, state: DroneState | None) -> DroneAction:

        return DroneAction(
            forward=np.random.uniform(-1, 1),
            lateral=np.random.uniform(-1, 1),
            rotation=np.random.uniform(-1, 1),
            grasper=random.choice([0, 1])
        )

class DroneRLEnv:
    """Lightweight RL environment wrapper for a single drone."""

    def __init__(self, drone: DroneAbstract):
        self.drone = drone

        self.reward_calculator = DroneReward()
        self.current_state = None

    def get_state(self) -> DroneState:
        """Extract the full drone state from simulator sensors."""
        return DroneState.from_drone(self.drone)

    def step(self, current_state, action: DroneAction, next_state) -> Tuple[float, bool, Dict]:
        """
        Post-hoc step function to process action execution results. Returns:
        - reward
        - done flag (True if drone destroyed)
        - extra info dict (reserved)
        """

        reward = self.reward_calculator.calculate(
            current_state,
            action,
            next_state
        )

        done = (next_state.health <= 0)
        info = {} # TODO extra info
        return reward, done, info 
