"""
Stable-Baselines3 compatible wrappers for DroneRLEnv.

This module provides wrapper classes to make DroneRLEnv compatible with SB3's
algorithms, particularly handling the Dict observation space.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from swarm_rescue.solutions.RL.rl_utils import flatten_observation


class FlattenObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper to flatten Dict observation space into a single Box space.
    
    Stable-Baselines3 works better with flattened observation spaces.
    This wrapper converts the Dict observation from DroneRLEnv into a
    single flat vector.
    
    The flattened observation contains:
    - lidar: 180 values (normalized distances)
    - semantic: 35 * 3 = 105 values (distance, angle, grasped)
    - pose: 3 values (x, y, angle - normalized)
    - velocity: 2 values (vx, vy)
    - grasper: 1 value (0 or 1)
    
    Total: 180 + 105 + 3 + 2 + 1 = 291 dimensions
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Calculate flattened observation space dimension
        self.obs_dim = self._calculate_obs_dim(env.observation_space)
        
        # Create new flattened observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )
    
    def _calculate_obs_dim(self, obs_space):
        """Calculate total dimension of flattened observation"""
        total_dim = 0
        for key, space in obs_space.spaces.items():
            if isinstance(space, spaces.Box):
                total_dim += int(np.prod(space.shape))
            else:
                raise ValueError(f"Unsupported space type for key {key}: {type(space)}")
        return total_dim
    
    def observation(self, obs):
        """Flatten the Dict observation into a vector"""
        return flatten_observation(obs)


class NormalizeRewardWrapper(gym.RewardWrapper):
    """
    Wrapper to normalize rewards for stable training.
    
    This uses a running mean and standard deviation to normalize
    rewards, which can significantly improve PPO training stability.
    """
    
    def __init__(self, env, gamma=0.99, epsilon=1e-8):
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.returns = 0.0
        self.return_mean = 0.0
        self.return_var = 1.0
        self.count = 0
    
    def reward(self, reward):
        """Normalize the reward"""
        self.returns = self.returns * self.gamma + reward
        self.count += 1
        
        # Update running statistics
        delta = self.returns - self.return_mean
        self.return_mean += delta / self.count
        delta2 = self.returns - self.return_mean
        self.return_var += delta * delta2
        
        # Normalize
        std = np.sqrt(self.return_var / max(1, self.count - 1) + self.epsilon)
        normalized_reward = reward / max(std, self.epsilon)
        
        return np.clip(normalized_reward, -10, 10)
    
    def reset(self, **kwargs):
        self.returns = 0.0
        return super().reset(**kwargs)


class TimeFeatureWrapper(gym.ObservationWrapper):
    """
    Add normalized timestep as a feature to the observation.
    
    This helps the agent learn time-dependent behaviors and understand
    when the episode is about to end.
    """
    
    def __init__(self, env, max_steps=None):
        super().__init__(env)
        self.max_steps = max_steps or env.max_steps
        self.current_step = 0
        
        # Get original observation space dimension
        if isinstance(env.observation_space, spaces.Box):
            orig_dim = env.observation_space.shape[0]
        else:
            raise ValueError("TimeFeatureWrapper requires Box observation space")
        
        # Add one dimension for time feature
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(orig_dim + 1,),
            dtype=np.float32
        )
    
    def observation(self, obs):
        """Add normalized timestep to observation"""
        time_feature = self.current_step / self.max_steps
        return np.append(obs, time_feature).astype(np.float32)
    
    def reset(self, **kwargs):
        self.current_step = 0
        return super().reset(**kwargs)
    
    def step(self, action):
        self.current_step += 1
        return super().step(action)


def make_sb3_env(
    map_name="Medium01",
    max_steps=500,
    fixed_step=10,
    headless=True,
    use_time_feature=True,
    normalize_reward=False,
):
    """
    Create a Stable-Baselines3 compatible environment with all necessary wrappers.
    
    Args:
        map_name: Name of the map to use
        max_steps: Maximum steps per episode
        fixed_step: Number of simulator ticks per action
        headless: Whether to run without GUI
        use_time_feature: Whether to add timestep as observation feature
        normalize_reward: Whether to normalize rewards
    
    Returns:
        Wrapped environment ready for SB3 training
    """
    from swarm_rescue.solutions.RL.my_drone_rl_framework import DroneRLEnv
    from swarm_rescue.solutions.RL.my_drone_RL import MyDroneRL
    
    # Create base environment
    env = DroneRLEnv(
        map_name=map_name,
        render_mode="human" if not headless else None,
        max_steps=max_steps,
        fixed_step=fixed_step,
        headless=headless,
        drone_cls=MyDroneRL,
    )
    
    # Apply wrappers
    env = FlattenObservationWrapper(env)
    
    # Time feature disabled - simpler observations
    # if use_time_feature:
    #     env = TimeFeatureWrapper(env, max_steps=max_steps)
    
    if normalize_reward:
        env = NormalizeRewardWrapper(env)
    
    return env
