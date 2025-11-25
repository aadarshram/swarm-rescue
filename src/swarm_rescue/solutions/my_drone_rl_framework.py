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
    """
    Reward class that uses exact fields you specified:
      - elapsed_timestep (from drone_abstract/state)
      - elapsed_walltime (from drone_abstract/state)
    Other repo-specific fields used unchanged:
      - misc_data.size_area
      - misc_data.max_timestep_limit
      - misc_data.max_walltime_limit
      - next_state.map_matrix (0 = unexplored)
      - next_state.semantic_data (RescueCenter detection)
      - next_state.lidar_scan (for wall avoidance)
    """

    def __init__(
        self,
        misc_data: Optional[MiscData] = None,
        R_pickup: float = 50.0,
        R_deliver: float = 200.0,
        R_bad_drop: float = -100.0,
        R_time_base: float = -0.1,
        time_scale: float = 5.0,
        R_exploration_coeff: float = -10.0,
        R_health_loss_coeff: float = 1.0,
        lidar_safe_distance: float = 50.0,
        lidar_penalty_coeff: float = 20.0,
        lidar_power: float = 3.0,
    ):
        # event rewards / penalties
        self.R_pickup = float(R_pickup)
        self.R_deliver = float(R_deliver)
        self.R_bad_drop = float(R_bad_drop)

        # time penalty (base and curvature)
        self.R_time_base = float(R_time_base)    # should be negative
        self.time_scale = float(time_scale)

        # exploration & health
        self.R_exploration_coeff = float(R_exploration_coeff)  # negative base
        self.R_health_loss_coeff = float(R_health_loss_coeff)  # positive multiplier for lost health

        # LIDAR params
        self.lidar_safe_distance = float(lidar_safe_distance)
        self.lidar_penalty_coeff = float(lidar_penalty_coeff)
        self.lidar_power = float(lidar_power)

        # misc_data fields (read exactly as in repo)
        self.size_area = None
        self.max_timestep_limit = None
        self.max_walltime_limit = None
        if misc_data is not None:
            self.size_area = getattr(misc_data, "size_area", None)
            try:
                self.max_timestep_limit = int(getattr(misc_data, "max_timestep_limit")) if hasattr(misc_data, "max_timestep_limit") else None
            except Exception:
                self.max_timestep_limit = None
            try:
                self.max_walltime_limit = float(getattr(misc_data, "max_walltime_limit")) if hasattr(misc_data, "max_walltime_limit") else None
            except Exception:
                self.max_walltime_limit = None

    # ---------- helpers using the exact elapsed names you gave ----------
    def _get_elapsed_timestep(self, state) -> Optional[int]:
        """Read elapsed_timestep from state (exact name)."""
        if state is None:
            return None
        if hasattr(state, "elapsed_timestep"):
            try:
                return int(getattr(state, "elapsed_timestep"))
            except Exception:
                return None
        if isinstance(state, dict) and "elapsed_timestep" in state:
            try:
                return int(state["elapsed_timestep"])
            except Exception:
                return None
        return None

    def _get_elapsed_walltime(self, state) -> Optional[float]:
        """Read elapsed_walltime from state (exact name)."""
        if state is None:
            return None
        if hasattr(state, "elapsed_walltime"):
            try:
                return float(getattr(state, "elapsed_walltime"))
            except Exception:
                return None
        if isinstance(state, dict) and "elapsed_walltime" in state:
            try:
                return float(state["elapsed_walltime"])
            except Exception:
                return None
        return None

    def _semantic_reports_rescue(self, next_state) -> bool:
        """Return True if any semantic ray reports RescueCenter with distance == 0.0."""
        semantic = getattr(next_state, "semantic_data", None)
        if not semantic:
            return False
        for ray in semantic:
            try:
                et = ray["entity_type"] if isinstance(ray, dict) else getattr(ray, "entity_type", None)
                dist = ray["distance"] if isinstance(ray, dict) else getattr(ray, "distance", None)
            except Exception:
                et = getattr(ray, "entity_type", None)
                dist = getattr(ray, "distance", None)
            try:
                if et == "RescueCenter" and float(dist) == 0.0:
                    return True
            except Exception:
                continue
        return False

    def _lidar_penalty(self, next_state) -> float:
        """
        LIDAR safety penalty: near wall => negative penalty.
        Smooth, steep "step-like" behavior using a power curve.
        """
        scan = getattr(next_state, "lidar_scan", None)
        if scan is None or len(scan) == 0:
            return 0.0
        try:
            nearest = float(np.min(scan))
        except Exception:
            try:
                nearest = float(min(scan))
            except Exception:
                return 0.0

        if nearest >= self.lidar_safe_distance:
            return 0.0
        normalized = max(0.0, 1.0 - (nearest / self.lidar_safe_distance))
        penalty = - self.lidar_penalty_coeff * (normalized ** self.lidar_power)
        return float(penalty)

    # ---------- main calculate ----------
    def calculate(self, current_state: DroneState, action: DroneAction, next_state: DroneState) -> float:
        """
        Compute scalar reward for transition (current_state -> next_state).
        Uses exact elapsed_timestep & elapsed_walltime names from drone_abstract.
        """
        if current_state is None:
            return 0.0

        # ----- 1) TIME: use both elapsed_timestep and elapsed_walltime (exact names) -----
        elapsed_step = self._get_elapsed_timestep(next_state) or self._get_elapsed_timestep(current_state)
        elapsed_wall = self._get_elapsed_walltime(next_state) or self._get_elapsed_walltime(current_state)

        max_steps = self.max_timestep_limit
        # fallback: use next_state attribute if misc_data didn't provide it
        if max_steps is None and hasattr(next_state, "max_timestep_limit"):
            try:
                max_steps = int(getattr(next_state, "max_timestep_limit"))
            except Exception:
                max_steps = None
        if max_steps is None and hasattr(current_state, "max_timestep_limit"):
            try:
                max_steps = int(getattr(current_state, "max_timestep_limit"))
            except Exception:
                max_steps = None

        max_wall = self.max_walltime_limit
        if max_wall is None and hasattr(next_state, "max_walltime_limit"):
            try:
                max_wall = float(getattr(next_state, "max_walltime_limit"))
            except Exception:
                max_wall = None
        if max_wall is None and hasattr(current_state, "max_walltime_limit"):
            try:
                max_wall = float(getattr(current_state, "max_walltime_limit"))
            except Exception:
                max_wall = None

        frac_step = None
        if (elapsed_step is not None) and (max_steps is not None) and max_steps > 0:
            frac_step = max(0.0, min(1.0, float(elapsed_step) / float(max_steps)))
        frac_wall = None
        if (elapsed_wall is not None) and (max_wall is not None) and max_wall > 0:
            frac_wall = max(0.0, min(1.0, float(elapsed_wall) / float(max_wall)))

        # conservative combine: use the larger fraction
        if frac_step is None and frac_wall is None:
            combined_frac = 0.0
        elif frac_step is None:
            combined_frac = frac_wall
        elif frac_wall is None:
            combined_frac = frac_step
        else:
            combined_frac = max(frac_step, frac_wall)

        try:
            per_step_penalty = float(self.R_time_base) * math.exp(self.time_scale * combined_frac)
        except Exception:
            per_step_penalty = float(self.R_time_base)

        reward = float(per_step_penalty)

        # ----- 2) Pickup detection (False -> True) -----
        prev_grasp = bool(getattr(current_state, "grasped", False))
        now_grasp = bool(getattr(next_state, "grasped", False))
        if (not prev_grasp) and now_grasp:
            reward += float(self.R_pickup)

        # ----- 3) Drop detection (True -> False) and rescue detection -----
        if prev_grasp and (not now_grasp):
            in_rescue = self._semantic_reports_rescue(next_state)
            if in_rescue:
                reward += float(self.R_deliver)
            else:
                reward += float(self.R_bad_drop)

        # ----- 4) Health loss penalty -----
        try:
            prev_h = float(getattr(current_state, "health", DRONE_INITIAL_HEALTH))
            next_h = float(getattr(next_state, "health", prev_h))
            health_lost = max(0.0, prev_h - next_h)
            if health_lost > 0.0:
                reward -= (self.R_health_loss_coeff * health_lost)
        except Exception:
            pass

        # ----- 5) Exploration penalty using next_state.map_matrix and misc_data.size_area -----
        exploration_contribution = 0.0
        try:
            mm = getattr(next_state, "map_matrix", None)
            total_pixels = None
            # prefer misc_data.size_area (tuple width, height)
            if self.size_area is not None and isinstance(self.size_area, (tuple, list)) and len(self.size_area) >= 2:
                try:
                    total_pixels = int(float(self.size_area[0]) * float(self.size_area[1]))
                except Exception:
                    total_pixels = None
            # fallback: next_state.size_area
            if total_pixels is None and hasattr(next_state, "size_area"):
                try:
                    sa = getattr(next_state, "size_area")
                    if isinstance(sa, (tuple, list)) and len(sa) >= 2:
                        total_pixels = int(float(sa[0]) * float(sa[1]))
                except Exception:
                    total_pixels = None
            # fallback: infer from map_matrix shape
            if total_pixels is None and (mm is not None):
                try:
                    total_pixels = int(mm.shape[0]) * int(mm.shape[1])
                except Exception:
                    total_pixels = None

            if (mm is not None) and (total_pixels is not None) and total_pixels > 0:
                try:
                    num_unexplored = float((mm == 0).sum())
                except Exception:
                    try:
                        num_unexplored = float(sum(1 for row in mm for v in row if v == 0))
                    except Exception:
                        num_unexplored = float(total_pixels)
                frac_unexplored = max(0.0, min(1.0, num_unexplored / float(total_pixels)))
                exploration_contribution = float(self.R_exploration_coeff) * frac_unexplored
            else:
                # conservative fallback
                exploration_contribution = float(self.R_exploration_coeff)
        except Exception:
            exploration_contribution = float(self.R_exploration_coeff)

        reward += float(exploration_contribution)

        # ----- 6) LIDAR wall avoidance penalty -----
        try:
            reward += float(self._lidar_penalty(next_state))
        except Exception:
            # ignore if lidar fails
            pass

        return float(reward)


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
