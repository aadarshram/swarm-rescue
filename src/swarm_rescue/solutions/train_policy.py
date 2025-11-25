"""
Training script for RL policy using DroneRLEnv.

This script demonstrates how to train a policy using the gym-style DroneRLEnv
that wraps the actual swarm-rescue simulator.

Usage:
    python -m swarm_rescue.solutions.train_policy --episodes 100 --map Medium01
"""

import argparse
import numpy as np
import time

from swarm_rescue.solutions.my_drone_rl_framework import DroneRLEnv
from swarm_rescue.solutions.my_drone_RL import MyDroneRL


class RandomPolicy:
    """Simple random policy for baseline testing"""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def select_action(self, obs):
        """Sample random action from action space"""
        return self.action_space.sample()


def train_random_policy(
    num_episodes: int = 10,
    max_steps: int = 500,
    map_name: str = "Medium01",
    headless: bool = True,
    fixed_step: int = 10,
    use_exp_map: bool = False,
):
    """
    Train (or evaluate) a random policy in the DroneRLEnv.
    
    Args:
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        map_name: Name of the map to use
        headless: Whether to run without GUI
        fixed_step: Number of simulator ticks per action
        use_exp_map: Whether to use exploration map rewards
    """
    
    print(f"Starting training with {num_episodes} episodes on {map_name}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Headless mode: {headless}")
    print(f"Fixed step: {fixed_step}")
    print("-" * 60)
    
    # Create environment
    env = DroneRLEnv(
        map_name=map_name,
        render_mode="rgb_array" if not headless else None,
        max_steps=max_steps,
        fixed_step=fixed_step,
        use_exp_map=use_exp_map,
        headless=headless,
        drone_cls=MyDroneRL,
    )
    
    # Create random policy
    policy = RandomPolicy(env.action_space)
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    episode_rescues = []
    
    for episode in range(num_episodes):
        start_time = time.time()
        
        # Reset environment
        obs, info = env.reset()
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        # Run episode
        while not done:
            # Select action
            action = policy.select_action(obs)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            if done:
                break
        
        # Episode statistics
        elapsed = time.time() - start_time
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_rescues.append(info.get("total_rescued", 0))
        
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Steps: {episode_length}")
        print(f"  Rescued: {info.get('total_rescued', 0)}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Avg reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
        print("-" * 60)
    
    # Final statistics
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total episodes: {num_episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average rescues: {np.mean(episode_rescues):.2f} ± {np.std(episode_rescues):.2f}")
    print(f"Best episode reward: {np.max(episode_rewards):.2f}")
    print(f"Worst episode reward: {np.min(episode_rewards):.2f}")
    
    # Close environment
    env.close()
    
    return episode_rewards, episode_lengths, episode_rescues


def eval_policy(
    policy,
    num_episodes: int = 5,
    max_steps: int = 500,
    map_name: str = "Medium01",
    headless: bool = False,
    fixed_step: int = 10,
    use_exp_map: bool = False,
    render_video: bool = False,
):
    """
    Evaluate a policy in the DroneRLEnv.
    
    Args:
        policy: Policy object with select_action(obs) method
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        map_name: Name of the map to use
        headless: Whether to run without GUI
        fixed_step: Number of simulator ticks per action
        use_exp_map: Whether to use exploration map rewards
        render_video: Whether to save video frames
    """
    
    print(f"Evaluating policy for {num_episodes} episodes on {map_name}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Headless mode: {headless}")
    print("-" * 60)
    
    # Create environment
    env = DroneRLEnv(
        map_name=map_name,
        render_mode="rgb_array" if render_video else None,
        max_steps=max_steps,
        fixed_step=fixed_step,
        use_exp_map=use_exp_map,
        headless=headless,
        drone_cls=MyDroneRL,
    )
    
    # Evaluation statistics
    episode_rewards = []
    episode_lengths = []
    episode_rescues = []
    all_frames = []
    
    for episode in range(num_episodes):
        start_time = time.time()
        
        # Reset environment
        obs, info = env.reset()
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        # Run episode
        while not done:
            # Select action using policy
            action = policy.select_action(obs)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            if done:
                break
        
        # Episode statistics
        elapsed = time.time() - start_time
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_rescues.append(info.get("total_rescued", 0))
        
        if render_video and "ep_frames" in info:
            all_frames.append(info["ep_frames"])
        
        print(f"Eval Episode {episode + 1}/{num_episodes}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Steps: {episode_length}")
        print(f"  Rescued: {info.get('total_rescued', 0)}/{info.get('map_name', '')}")
        print(f"  Time: {elapsed:.2f}s")
        print("-" * 60)
    
    # Final statistics
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Episodes evaluated: {num_episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average rescues: {np.mean(episode_rescues):.2f} ± {np.std(episode_rescues):.2f}")
    print(f"Success rate: {100 * np.sum(np.array(episode_rescues) > 0) / num_episodes:.1f}%")
    print(f"Best episode reward: {np.max(episode_rewards):.2f}")
    print(f"Worst episode reward: {np.min(episode_rewards):.2f}")
    
    # Close environment
    env.close()
    
    results = {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "rescues": episode_rescues,
        "frames": all_frames if render_video else None,
    }
    if render_video:
        import imageio
        for i, frames in enumerate(all_frames):
            path = f"eval_episode_{i+1}.mp4"
            imageio.mimsave(path, frames)
            print(f"Saved video: {path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate RL policy for drone rescue")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                        help="Mode: train (random baseline) or eval (evaluate policy)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--map", type=str, default="Medium01", choices=["Medium01", "Intermediate01"],
                        help="Map to use")
    parser.add_argument("--fixed-step", type=int, default=10, help="Simulator ticks per action")
    parser.add_argument("--use-exp-map", action="store_true", help="Use exploration map rewards")
    parser.add_argument("--no-headless", action="store_true", help="Show GUI (slower)")
    parser.add_argument("--render-video", action="store_true", help="Save video frames during eval")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_random_policy(
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            map_name=args.map,
            headless=not args.no_headless,
            fixed_step=args.fixed_step,
            use_exp_map=args.use_exp_map,
        )
    else:  # eval mode
        # Create environment to get action space
        temp_env = DroneRLEnv(
            map_name=args.map,
            headless=True,
            drone_cls=MyDroneRL,
        )
        policy = RandomPolicy(temp_env.action_space)
        temp_env.close()
        
        # Run evaluation
        eval_policy(
            policy=policy,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            map_name=args.map,
            headless=not args.no_headless,
            fixed_step=args.fixed_step,
            use_exp_map=args.use_exp_map,
            render_video=args.render_video,
        )
