import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.utils import constants
# Note: We implement our own find_pose here instead of importing from mapping.py
# because the sensor return types are different (numpy arrays vs objects)

# Action space should be symmetric for better learning
# Grasper action is handled via separate logic
ACTION_SPACE = gym.spaces.Box(
    low=np.array([-1, -1, -1], dtype=np.float32), # Forward, Lateral, Rotation
    high=np.array([ 1,  1, 1], dtype=np.float32),
    shape=(3,), dtype=np.float32
)

# Observation space should be img or 1D vector (flattened- better for RL algorithms)
obs_range = (constants.RESOLUTION_LIDAR_SENSOR-1) + (constants.RESOLUTION_SEMANTIC_SENSOR*3) + 3 + 2  # lidar + semantic + pose + velocity; grasper handled separately
OBSERVATION_SPACE = spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(obs_range,),
    dtype=np.float32
)

class Pose:
    """Pose representation for tracking position and orientation"""
    def __init__(self, position=(0.0, 0.0), orientation=0.0):
        self.position = np.array(position, dtype=np.float32)
        self.orientation = float(orientation)

def find_pose(drone, prev_pose=None):
    """
    Find drone pose using GPS when available, or odometer integration in No-GPS zones.
    
    Args:
        drone: The drone object with sensors
        prev_pose: Previous pose for odometer integration, None for first call
    
    Returns:
        Updated pose
    """
    if prev_pose is None:
        prev_pose = Pose()
    
    # Try GPS and compass
    gps_data = drone.gps_values()
    compass_data = drone.compass_values()
    
    if gps_data is not None and compass_data is not None and not np.any(np.isnan(gps_data)):
        # GPS available: use absolute positioning
        return Pose(
            position=np.array(gps_data, dtype=np.float32),
            orientation=float(compass_data)
        )
    else:
        # No GPS: use odometer integration
        odometer = drone.odometer_values()
        
        if odometer is not None:
            # Extract odometer readings
            dist_travel = getattr(odometer, "delta_distance", 0.0)
            alpha = getattr(odometer, "delta_alpha", 0.0)
            theta = getattr(odometer, "delta_theta", 0.0)
            
            # Integrate odometry to estimate new pose
            beta = prev_pose.orientation + alpha
            x = prev_pose.position[0] + dist_travel * np.cos(beta)
            y = prev_pose.position[1] + dist_travel * np.sin(beta)
            angle = prev_pose.orientation + theta
            
            return Pose(position=(x, y), orientation=angle)
        else:
            # Fallback: return previous pose
            return prev_pose

def process_semantic(semantic_values):
    ''' Process raw semantic sensor to RLEnv compatible format.'''
    rows = []
    for item in semantic_values:
        rows.append([
            float(item.distance) / constants.MAX_RANGE_SEMANTIC_SENSOR,
            float(item.angle),
            1.0 if item.grasped else 0.0
        ])
    
    # Pad to RESOLUTION_SEMANTIC_SENSOR rows (some rays may not detect anything)
    result = np.zeros((constants.RESOLUTION_SEMANTIC_SENSOR, 3), dtype=np.float32)
    actual_rows = min(len(rows), constants.RESOLUTION_SEMANTIC_SENSOR)
    if actual_rows > 0:
        result[:actual_rows] = rows[:actual_rows]
    
    return result

def build_obs(drone):
    """Convert raw drone sensors → RLEnv format. Not flattened for model yet."""
    # LIDAR: 360° wall detection, normalized to [-1, 1]
    lidar = np.array(drone.lidar_values(), dtype=np.float32)
    if lidar is None or len(lidar) == 0:
        lidar = np.zeros(constants.RESOLUTION_LIDAR_SENSOR, dtype=np.float32)
    # Exclude last value (duplicate of first at ±π) -> 180 rays
    lidar = lidar[:-1]
    # Normalize to [-1, 1] range
    lidar = np.clip(lidar / constants.MAX_RANGE_LIDAR_SENSOR, 0, 1) * 2 - 1

    # Pose: position and orientation
    # Handles both GPS zones and No-GPS zones
    #   - GPS zone: Uses GPS + Compass for absolute positioning (accurate)
    #   - No-GPS zone: Uses odometer integration for dead reckoning (accumulated error)
    if not hasattr(drone, '_rl_pose'):
        # First call: initialize from GPS or default
        drone._rl_pose = find_pose(drone)
    else:
        # Subsequent calls: update pose (automatically switches based on GPS availability)
        drone._rl_pose = find_pose(drone, drone._rl_pose)
    
    pose = np.array([
        float(drone._rl_pose.position[0]),
        float(drone._rl_pose.position[1]),
        float(drone._rl_pose.orientation)
    ], dtype=np.float32)
    
    # Normalize pose if map size is available
    if hasattr(drone, 'size_area') and drone.size_area is not None:
        pose[0] /= drone.size_area[0]
        pose[1] /= drone.size_area[1]

    # Velocity
    vel_data = drone.measured_velocity()
    if vel_data is None:
        vel_data = (0.0, 0.0)
    vel = np.array([ # TODO: Velocity normalization?
        float(vel_data[0]),
        float(vel_data[1])
    ], dtype=np.float32)

    # Semantic sensor
    semantic = process_semantic(drone.semantic_values())

    # Send to drone
    drone.pose = pose
    
    obs = {
        "pose": pose,
        "velocity": vel,
        "lidar": lidar,
        "semantic": semantic,
    }
    return obs

def to_commands_dict(action: np.ndarray) -> CommandsDict:
    """
    Convert action array to CommandsDict.
    """
    command: CommandsDict = {
        "forward": float(action[0]),
        "lateral": float(action[1]),
        "rotation": float(action[2]),
        "grasper": int((action[3])) # Predicted as 0 or 1 by separate logic
    }
    return command


def flatten_observation(obs):
    """Flatten Dict observation to vector for SB3"""
    vec = np.concatenate([
        obs["lidar"],
        obs["semantic"].flatten(),
        obs["pose"],
        obs["velocity"],
    ]).astype(np.float32)
    return vec


def obs_to_tensor(obs):
    vec = flatten_observation(obs)
    return torch.tensor(vec, dtype=torch.float32).unsqueeze(0)  # (1, D)

