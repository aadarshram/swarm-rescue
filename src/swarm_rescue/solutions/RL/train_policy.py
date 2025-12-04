import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from swarm_rescue.solutions.RL.my_drone_rl_framework import DroneRLEnv

# Config
debug = True # Debug mode
map_name = "Medium01"
max_steps = 500
fixed_step = 10
headless = True # No GUI
device = "cpu"  # Use CPU for training MLP Policy

# Training
train_steps = 1000

# Video / rendering controls
# Set `capture_video` to True only when you intend to record a short debug run.
# Keeping `capture_video=False` avoids accumulating frames in memory which can OOM the process.
capture_video = False

# Set render mode based on debug + explicit capture flag
render_mode = "rgb_array" if (debug and capture_video) else None

# Safeguard: estimate frames that will be captured in-memory and disable video
# if the estimated frames are very large (likely to cause OOM / process kill).
if render_mode == "rgb_array":
    # Don't render during training to avoid OOM
    render_mode = None
    print("Warning: Video capture is disabled during training to prevent OOM errors.")

vec_env = make_vec_env(DroneRLEnv, n_envs=1, env_kwargs={
    "map_name": map_name,
    "max_steps": max_steps,
    "fixed_step": fixed_step,
    "headless": headless,
    "render_mode": render_mode
    })

if debug:
    # Check if the env follows the Gym API
    from stable_baselines3.common.env_checker import check_env
    check_env(DroneRLEnv(
        map_name=map_name,
        max_steps=max_steps,
        fixed_step=fixed_step,
        headless=headless,
        render_mode=None  # Don't render during env check
    ), warn=True)

obs = vec_env.reset()   

if debug:
    print(vec_env.observation_space)
    print(vec_env.action_space)

print("Training agent...")
# Train agent
model = A2C("MlpPolicy", vec_env, learning_rate=0.001, verbose=1, device=device)
model.learn(total_timesteps=train_steps, progress_bar=debug)


# # Save training video if debug mode
# if debug:
#     print("Saving training video...")
#     # Get frames from the last episode
#     env = vec_env.envs[0]
#     if hasattr(env, 'get_all_frames'):
#         frames = env.get_all_frames()
#         if len(frames) > 0:
#             import imageio
#             video_path = "src/swarm_rescue/solutions/videos/training_video.mp4"
#             imageio.mimsave(video_path, frames, fps=30)
#             print(f"Training video saved to {video_path}")
#         else:
#             print("No frames captured during training.")
#     else:
#         print("Environment does not support frame capture.")

# Save the trained model
model_path = "src/swarm_rescue/solutions/models/a2c"
model.save(model_path)

print("Training completed.")

