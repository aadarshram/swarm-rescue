"""
RL framework components for drone control.
Implements OpenAI Gym style environment, reward system, and policy interfaces.
"""

import gc
import numpy as np
from typing import Optional

import gymnasium as gym
import arcade

from swarm_rescue.simulation.utils import constants
from swarm_rescue.solutions.RL.my_drone_RL import MyDroneRL
from swarm_rescue.simulation.gui_map.gui_sr import GuiSR
# Maps
from swarm_rescue.maps.map_medium_01 import MapMedium01
from swarm_rescue.maps.map_intermediate_01 import MapIntermediate01
from swarm_rescue.solutions.RL.rl_utils import ACTION_SPACE, OBSERVATION_SPACE, build_obs, to_commands_dict, flatten_observation

map_dict = {
    "Medium01": MapMedium01,
    "Intermediate01": MapIntermediate01,
}

class DroneRLEnv(gym.Env):
    """
    Variables:
    - max_steps: total timesteps to run before terminate the episode
    - fixed_steps: number of steps to step the playground per command produce by the agent
    - map_name: select the maps to run in.

    Oservation Space:
    - Pose: true_position and angle.
    - Velocity: velocity x and y axis.
    - Semantic: Rescue center, human, and drones. Data: distance, ray_angle, grased.
    - Lidar: 180 distance rays.

    Action Space: continuous or multi-discrete
    - Forward, Lateral, Rotation: [-1, 1]
    - Grasper: {0, 1}
    """

    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        map_name: str = "Medium01",
        render_mode: str = "rgb_array",
        max_steps: int = 100,
        fixed_step: int = 20,
        headless: bool = False,
        drone_cls=MyDroneRL,
    ):
        super().__init__()

        # Initialize map
        if map_name in map_dict:
            self.map_name = map_name
            self.map_cls = map_dict[map_name]
        else:
            raise ValueError(f"Unknown map_name {map_name}")

        # Config
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.fixed_step = fixed_step
        self.headless = bool(headless)
        self._map = None
        self._playground = None
        self.map_size = None
        self.total_rescued = 0

        # Drone config
        self.drone_cls = drone_cls
        self._agent = None
        self.action_space = ACTION_SPACE
        self.observation_space = OBSERVATION_SPACE
        
        # Bookkeeping
        self.ep_count = 0
        self.current_step = 0
        self.total_rescued = 0
        self.frames = []
        self.last_exp_score = None
        self.prev_obs = None
        self.prev_health = None
        
        # Episode statistics for evaluation

        self.re_init()

    def construct_action(self, action):
        """Convert action array to CommandsDict format expected by the simulator"""
        # Action is a 1D vector of le  ngth 4
        # Clip values to valid range [-1, 1] for continuous, [0, 1] for grasper
        action = np.clip(action, [-1, -1, -1, 0], [1, 1, 1, 1])
        return to_commands_dict(action)
        
    def get_distance(self, pos_a, pos_b):
        return np.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)
      
    def _get_obs(self):
        obs = build_obs(self._agent)
        # RL Env returns flattened observation
        obs_flat = flatten_observation(obs)
        return obs_flat

    def _get_info(self):
        info = {}
        info["map_name"] = self.map_name
        
        # Get wounded persons positions if available
        if hasattr(self._map, '_wounded_persons_pos'):
            info["wounded_people_pos"] = self._map._wounded_persons_pos
        
        # Get rescue center position if available
        if hasattr(self._map, '_rescue_center_pos'):
            info["rescue_zone"] = self._map._rescue_center_pos
        
        # Get drone position
        if self._agent is not None:
            info["drones_true_pos"] = self._agent.true_position()
        # Get drone orientation
            info["drones_true_angle"] = self._agent.true_angle()
            info["drone_true_velocity"] = self._agent.true_velocity()
            info["drone_true_angular_velocity"] = self._agent.true_angular_velocity()
            # TODO: Add more true values here as info to debug against policy
        return info

    def re_init(self):
        """(Re)create map, playground and main agent"""
        
        # Clean up previous resources
        if hasattr(self, 'gui') and self.gui is not None:
            try:
                self.gui.close()
            except Exception:
                pass
        
        # Explicitly close arcade window to avoid zombies
        try:
            import arcade
            arcade.close_window()
        except Exception:
            pass
        
        if hasattr(self, '_playground') and self._playground is not None:
            try:
                self._playground.cleanup()
            except Exception:
                pass
        
        gc.collect()
        
        # Create new map with the drone class
        self._map = self.map_cls(drone_type=self.drone_cls)
        # Get map properties
        self.map_size = self._map.size_area
        self._playground = self._map.playground
        
        # Get the first drone as the agent (single agent for now) # TODO: Extend to multi-agent later
        self._agent = self._playground.agents[0] if self._playground.agents else None
        
        # Create GUI for rendering (can be headless)
        if self.render_mode is not None:
            try:
                self.gui = GuiSR(the_map=self._map, headless=self.headless)
            except Exception as e:
                print(f"Warning: Could not create GuiSR: {e}")
                self.gui = None
        else:
            self.gui = None
        
        # Reset exploration tracking
        if hasattr(self._map, 'explored_map'): # TODO: Find a way to visualize this to know how our agent is exploring
            self._map.explored_map.reset()
            self.last_exp_score = self._map.explored_map.score()

    # Gym API

    def reset(self, seed = None, options = None):

        super().reset(seed=seed)

        self.re_init()

        self.current_step = 0
        self.total_rescued = 0
        self.ep_count += 1
        self.prev_obs = None
        self.prev_health = None
        
        # Reset episode statistics
        self.episode_stats = {
            'steps_taken': 0,
            'total_reward': 0.0,
            'rescued': 0,
            'success': False,
            'health_lost': 0.0,
            'exploration_progress': 0.0,
        }

        # Step the playground once to initialize sensors (they return NaN initially)
        if self._playground is not None and self._agent is not None:
            no_action_cmd = {self._agent: self.construct_action(np.zeros(4, dtype=np.float32))}
            self._playground.step(all_commands=no_action_cmd, all_messages={})

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def render(self):
        if self.render_mode == "rgb_array":
            try:
                return self.gui.get_playground_image()
            except Exception:
                return None
        elif self.render_mode == "human":
            return None

    def get_all_frames(self):
        return self.frames

    def step(self, action):
        """
        Execute one environment step with the given action.
        Follows the pattern from GuiSR.on_update() and Launcher.one_round()
        """
        frame_skip = 5
        counter = 0
        done = False

        terminated, truncated = False, False

        # Store previous state for reward calculation
        prev_grasped = len(self._agent.grasped_wounded_persons()) > 0 if self._agent else False
        prev_health = self._agent.drone_health if self._agent and hasattr(self._agent, 'drone_health') else constants.DRONE_INITIAL_HEALTH

        # Run several simulator ticks per action (like GuiSR.on_update loop)
        while counter < self.fixed_step and not done: # TODO: Is this fixed step business correct. if so what value is correct?
            # Construct command dict for the agent
            grasper = 1.0 if self._agent.found_wounded()[0] else 0.0
            cmd = {self._agent: self.construct_action(np.append(action, grasper))}
            
            # Step the playground (core simulation step)
            if self._playground is not None:
                self._playground.step(all_commands=cmd, all_messages=self._agent.msg_data)
            # # GUI update for rendering
            # if self.render_mode == "human":
            #     try:
            #         if self.gui is not None:
            #             # Check if window was closed externally
            #             if hasattr(self.gui._playground.window, 'has_exit') and self.gui._playground.window.has_exit:
            #                 terminated = True
            #                 truncated = True
            #                 break
                            
            #             self.gui.draw()
            #             self.gui._playground.window.flip()
            #             self.gui._playground.window.dispatch_events()
                        
            #             # Check again after dispatching events
            #             if hasattr(self.gui._playground.window, 'has_exit') and self.gui._playground.window.has_exit:
            #                 terminated = True
            #                 truncated = True
            #                 break
            #     except Exception:
            #         # If window is closed, accessing it might raise exception
            #         terminated = True
            #         truncated = True
            #         break



            # Capture frames for video if needed
            if counter % frame_skip == 0 and self.render_mode == "rgb_array":
                try:
                    self.frames.append(self.gui.get_playground_image())
                except Exception:
                    pass
            
            counter += 1

        # Calculate reward
        reward = self._calculate_reward(prev_grasped, prev_health, action)
        # Terminates if times up
        ep = 2 # Margin
        if self._agent.elapsed_timestep >= self._agent._misc_data.max_timestep_limit - ep:
            terminated = True
            truncated = True
            # High penalty for timeout
            reward -= 10.0
        if self._agent.elapsed_walltime >= self._agent._misc_data.max_walltime_limit - ep:
            terminated = True
            truncated = True
            # High penalty for timeout
            reward -= 10.0

        # Mission completion
        total_wounded = getattr(self._map, "_number_wounded_persons", 0)
        if self.total_rescued >= total_wounded and total_wounded > 0:
            self.episode_stats['success'] = True

        self.current_step += 1
        self.episode_stats['steps_taken'] = self.current_step
        self.episode_stats['total_reward'] += reward

        # When to truncate episode

        # Get observation and info
        obs = self._get_obs()
        info = self._get_info()
        info["reward"] = reward
        info["done"] = truncated or terminated
        info["total_rescued"] = self.total_rescued
        info["current_step"] = self.current_step
        
        # Add episode statistics to info for logging/evaluation
        info["episode_stats"] = dict(self.episode_stats)
        
        if info["done"]:
            info["ep_frames"] = list(self.frames)
            # Final episode statistics
            info["episode"] = {
                'r': self.episode_stats['total_reward'],
                'l': self.episode_stats['steps_taken'],
                'rescued': self.episode_stats['rescued'],
                'success': self.episode_stats['success'],
                'health_lost': self.episode_stats['health_lost'],
                'exploration': self.episode_stats['exploration_progress'],
            }

        return obs, float(reward), bool(terminated), bool(truncated), info

    def _calculate_reward(self, prev_grasped, prev_health, action):
        """Calculate comprehensive reward based on multiple factors"""

        # For agent exploration
        reward = 0.0
        
        # Timestep penalty (encourage efficiency)
        reward += -1
        
        # Rotation penalty (to avoid excessive spinning)
        reward += -0.08 * abs(action[2])  # Penalize based on rotation magnitude

        # Health penalty (collision damage)
        if self._agent and hasattr(self._agent, 'drone_health'):
            current_health = self._agent.drone_health
            health_loss = max(0.0, prev_health - current_health)
            # Normalize 
            health_loss /= constants.DRONE_INITIAL_HEALTH
            if health_loss > 0:
                reward -= 1 * health_loss
                self.episode_stats['health_lost'] += health_loss
        
        # LIDAR-based collision avoidance
        if self._agent:
            lidar_values = self._agent.lidar_values()
            if lidar_values is not None and len(lidar_values) > 0:
                min_distance = np.min(lidar_values)
                
                if min_distance < 40:
                    # Penalty increases as drone gets closer to obstacles
                    reward += -10.0 * (40 - min_distance - 5) / 40.0 # 5 is a small buffer
        
        # Exploration reward
        # Can use internal state map explored since only exposed to agent as reward and not state
        if hasattr(self._map, 'explored_map'):
            current_exp_score = self._map.explored_map.score()
            exp_progress = current_exp_score - self.last_exp_score if self.last_exp_score is not None else 0.0
            reward += 10.0 * exp_progress  # Scale exploration reward
            self.last_exp_score = current_exp_score
            self.episode_stats['exploration_progress'] += exp_progress
        
        # Reward to go close to wounded people
        if self._agent.found_wounded()[0]:
            reward += 5.0  # Reward for touching a wounded person; default rescue logic handles grasping and rescue
        
        # If drone destroyed, large penalty
        if self._agent and self._agent.removed:
            reward -= 50.0
        
        return reward

    def close(self):
        try:
            if self.gui:
                self.gui.close()
        except Exception:
            pass
        try:
            if self._playground:
                self._playground.cleanup()
        except Exception:
            pass
        gc.collect()
        try:
            arcade.close_window()
        except Exception:
            pass
