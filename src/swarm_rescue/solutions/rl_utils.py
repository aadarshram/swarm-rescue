import numpy as np
# import torch
import gymnasium as gym
from gymnasium import spaces
from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.utils import constants

# Global action space for each drone
ACTION_SPACE = gym.spaces.Box(
    low=np.array([-1, -1, -1, 0], dtype=np.float32),
    high=np.array([ 1,  1,  1, 1], dtype=np.float32),
    shape=(4,), dtype=np.float32
)

OBSERVATION_SPACE = spaces.Dict(
            {
                "lidar": spaces.Box(low=0.0, high=1.0, shape=(constants.RESOLUTION_LIDAR_SENSOR-1,), dtype=np.float32), # sub 1 to exclude max range ray (repeated)
                "semantic": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(constants.RESOLUTION_SEMANTIC_SENSOR, 3), dtype=np.float32 
                ),
                "pose": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
                "grasper": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

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
    """Convert raw drone sensors → RLEnv format"""
    # Get lidar values (handle None case)
    lidar_raw = drone.lidar_values()
    if lidar_raw is None:
        lidar = np.zeros(180, dtype=np.float32)
    else:
        lidar = np.array(lidar_raw, dtype=np.float32)
        lidar = lidar[:-1]  # Exclude last ray (repeated at 360 degrees)
        lidar = np.clip(lidar / constants.MAX_RANGE_LIDAR_SENSOR, 0, 1)

    # Get pose (handle None case)
    gps = drone.measured_gps_position()
    angle = drone.measured_compass_angle()
    if gps is None or angle is None:
        pose = np.zeros(3, dtype=np.float32)
    else:
        pose = np.array([gps[0], gps[1], angle], dtype=np.float32)
        # If map size exists in drone, normalize
        if drone.size_area is not None:
            pose[0] /= drone.size_area[0]
            pose[1] /= drone.size_area[1]

    # Get velocity (handle None case)
    velocity_raw = drone.measured_velocity()
    if velocity_raw is None:
        vel = np.zeros(2, dtype=np.float32)
    else:
        vel = np.array([velocity_raw[0], velocity_raw[1]], dtype=np.float32)
        # Clip velocity to reasonable range (drones shouldn't exceed ±10 m/s)
        vel = np.clip(vel / 10.0, -1.0, 1.0)

    # Get semantic values (handle None case)
    semantic_raw = drone.semantic_values()
    if semantic_raw is None or len(semantic_raw) == 0:
        semantic = np.zeros((constants.RESOLUTION_SEMANTIC_SENSOR, 3), dtype=np.float32)
    else:
        semantic = process_semantic(semantic_raw)

    grasper = np.array([1.0 if len(drone.grasped_wounded_persons()) > 0 else 0.0], dtype=np.float32)      

    return {
        "lidar": lidar,
        "pose": pose,
        "velocity": vel,
        "semantic": semantic,
        "grasper": grasper
    }


def flatten_observation(obs):
    vec = np.concatenate([
        obs["lidar"].ravel(),
        obs["semantic"].ravel(),
        obs["pose"],
        obs["velocity"],
        obs["grasper"]
    ]).astype(np.float32)
    return vec


def obs_to_tensor(obs):
    pass
    # vec = flatten_observation(obs)
    # return torch.tensor(vec, dtype=torch.float32).unsqueeze(0)  # (1, D)

def to_commands_dict(action: np.ndarray) -> CommandsDict:
    """
    Convert action array to CommandsDict.
    """
    command: CommandsDict = {
        "forward": float(action[0]),
        "lateral": float(action[1]),
        "rotation": float(action[2]),
        "grasper": int(action[3])
    }
    return command