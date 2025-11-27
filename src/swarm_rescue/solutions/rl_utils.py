import numpy as np
# import torch
import gymnasium as gym
from gymnasium import spaces
from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.utils import constants
# Note: We implement our own find_pose here instead of importing from mapping.py
# because the sensor return types are different (numpy arrays vs objects)

# Global action space for each drone
ACTION_SPACE = gym.spaces.Box(
    low=np.array([-1, -1, -1, 0], dtype=np.float32),
    high=np.array([ 1,  1,  1, 1], dtype=np.float32),
    shape=(4,), dtype=np.float32
)

OBSERVATION_SPACE = spaces.Dict(
        {
            "lidar": spaces.Box(
                low=0.0, 
                high=1.0, 
                shape=(constants.RESOLUTION_LIDAR_SENSOR - 1,),  # 180 rays (exclude duplicate at ±π)
                dtype=np.float32
            ),
            "semantic": spaces.Box(
                low=-np.inf, high=np.inf, shape=(constants.RESOLUTION_SEMANTIC_SENSOR, 3), dtype=np.float32 
            ),
            "pose": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "grasper": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        }
    )

# def build_observation_space() -> spaces.Dict:
#     """Build observation space for RL environment."""
#     return spaces.Dict(
#         {
#             "lidar": spaces.Box(
#                 low=0.0, 
#                 high=1.0, 
#                 shape=(constants.RESOLUTION_LIDAR_SENSOR - 1,),  # 180 rays (exclude duplicate at ±π)
#                 dtype=np.float32
#             ),
#             "semantic": spaces.Box(
#                 low=-np.inf, high=np.inf, shape=(constants.RESOLUTION_SEMANTIC_SENSOR, 3), dtype=np.float32 
#             ),
#             "pose": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
#             "velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
#             "grasper": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
#         }
#     )

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
    """Convert raw drone sensors → RLEnv format"""
    # LIDAR: 360° wall detection, normalized to [0, 1]
    lidar = np.array(drone.lidar_values(), dtype=np.float32)
    if lidar is None or len(lidar) == 0:
        lidar = np.zeros(constants.RESOLUTION_LIDAR_SENSOR, dtype=np.float32)
    # Exclude last value (duplicate of first at ±π) -> 180 rays
    lidar = lidar[:-1]
    # Normalize to [0, 1] range
    lidar = np.clip(lidar / constants.MAX_RANGE_LIDAR_SENSOR, 0, 1)

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

    # Grasper state
    grasper = np.array([1.0 if len(drone.grasped_wounded_persons()) > 0 else 0.0], dtype=np.float32)

    # Send to drone
    drone.pose = pose
    
    obs = {
        "lidar": lidar,
        "pose": pose,
        "velocity": vel,
        "semantic": semantic,
        "grasper": grasper
    }

    # --- MAPPING APPROACH --- # (Ad) Issue with variable size and redundant info
    # if not hasattr(build_obs, "has_run"):
    #     build_obs.has_run = True
    #     drone.pose = find_pose(drone)
    #     drone.grid = OccupancyGrid(drone.size_area, resolution=1, lidar=drone.lidar(), semantic=drone.semantic())
    # else:
    #     drone.pose = find_pose(drone, drone.pose) 
    #     drone.grid.update_grid(drone.pose)
    # obs["grid"] = np.array(drone.grid)
    # obs["pose"] = np.array(drone.pose)

    return obs

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


def flatten_observation(obs):
    """Flatten Dict observation to vector for SB3"""
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

