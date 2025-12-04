from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from swarm_rescue.solutions.RL.scripts.my_drone_rl_framework import DroneRLEnv

# Config
debug = True # Debug mode
map_name = "Medium01"
max_steps = 500
fixed_step = 10
headless = True # No GUI

eval_steps = 200
model_path = "src/swarm_rescue/solutions/models/a2c"  # Path to the trained model
device = "cpu"  # Use CPU for evaluation of MLP Policy

vec_env = make_vec_env(DroneRLEnv, n_envs=1, env_kwargs={
    "map_name": map_name,
    "max_steps": max_steps,
    "fixed_step": fixed_step,
    "headless": headless,
    })
model = A2C.load(model_path, env=vec_env, device=device)

obs = vec_env.reset()
steps = 0
done = False

print("Evaluating agent...")
# Evaluate agent
while steps < eval_steps and not done:
    steps += 1
    print(f"Step {steps}/{eval_steps}")
    action, _states = model.predict(obs)  # Shape: (1, 3): (N_envs, action_dim)
    obs, reward, done, info = vec_env.step(action)
    print(f"Reward: {reward}, Done: {done}")
    vec_env.render()
    if done:
        # VecEnv resets automatically
        print(f"Episode finished, reward: {reward}")
        break