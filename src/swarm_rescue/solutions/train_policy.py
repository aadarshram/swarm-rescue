"""
RL training script using Stable-Baselines3.

This script trains RL agents (PPO, SAC, TD3, DQN, etc.) on the DroneRLEnv environment.

Usage:
    # Train with PPO (default)
    python -m swarm_rescue.solutions.train_ppo --total-timesteps 100000
    
    # Train with different algorithm
    python -m swarm_rescue.solutions.train_ppo --algo SAC --total-timesteps 100000
    
    # Training with custom hyperparameters
    python -m swarm_rescue.solutions.train_ppo --total-timesteps 500000 --learning-rate 0.0003 --n-steps 2048
    
    # Continue training from checkpoint
    python -m swarm_rescue.solutions.train_ppo --load-path models/ppo_drone_100000_steps.zip --total-timesteps 200000
    
    # Evaluate trained model
    python -m swarm_rescue.solutions.train_ppo --eval --load-path models/ppo_drone_best.zip --n-eval-episodes 10
"""

import argparse
import os
from typing import Optional, Type
import numpy as np

# SB3 imports
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Local imports
from swarm_rescue.solutions.sb3_wrappers import make_sb3_env


# Algorithm registry - import algorithms as needed
def get_algorithm_class(algo_name: str) -> Type[BaseAlgorithm]:
    """Get the algorithm class by name"""
    algo_name = algo_name.upper()
    
    if algo_name == "PPO":
        from stable_baselines3 import PPO
        return PPO
    elif algo_name == "SAC":
        from stable_baselines3 import SAC
        return SAC
    elif algo_name == "TD3":
        from stable_baselines3 import TD3
        return TD3
    elif algo_name == "A2C":
        from stable_baselines3 import A2C
        return A2C
    elif algo_name == "DQN":
        from stable_baselines3 import DQN
        return DQN
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}. Supported: PPO, SAC, TD3, A2C, DQN")


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to tensorboard.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_rescues = []
    
    def _on_step(self) -> bool:
        # Log additional info from the environment
        if "episode" in self.locals.get("infos", [{}])[0]:
            info = self.locals["infos"][0]["episode"]
            ep_rew = info["r"]
            ep_len = info["l"]
            
            self.episode_rewards.append(ep_rew)
            self.episode_lengths.append(ep_len)
            
            # Log to tensorboard
            self.logger.record("rollout/ep_rew_mean", np.mean(self.episode_rewards[-100:]))
            self.logger.record("rollout/ep_len_mean", np.mean(self.episode_lengths[-100:]))
            
            # Log rescue info if available
            if "total_rescued" in self.locals.get("infos", [{}])[0]:
                rescues = self.locals["infos"][0]["total_rescued"]
                self.episode_rescues.append(rescues)
                self.logger.record("rollout/rescues_mean", np.mean(self.episode_rescues[-100:]))
        
        return True


class ProgressCallback(BaseCallback):
    """
    Callback for printing training progress.
    """
    
    def __init__(self, check_freq=10000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get statistics from logger
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                mean_length = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
                
                if self.verbose > 0:
                    print(f"Timesteps: {self.n_calls}")
                    print(f"  Mean reward: {mean_reward:.2f}")
                    print(f"  Mean length: {mean_length:.1f}")
                    
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        print(f"  🎉 New best mean reward: {self.best_mean_reward:.2f}")
                print("-" * 50)
        
        return True


def make_train_env(
    map_name="Medium01",
    max_steps=500,
    fixed_step=10,
    use_time_feature=True,
    normalize_reward=False,
):
    """Create training environment with Monitor wrapper"""
    
    def _init():
        env = make_sb3_env(
            map_name=map_name,
            max_steps=max_steps,
            fixed_step=fixed_step,
            headless=True,
            use_time_feature=use_time_feature,
            normalize_reward=normalize_reward,
        )
        env = Monitor(env)
        return env
    
    return _init


def make_eval_env(
    map_name="Medium01",
    max_steps=500,
    fixed_step=10,
    use_time_feature=True,
):
    """Create evaluation environment"""
    
    def _init():
        env = make_sb3_env(
            map_name=map_name,
            max_steps=max_steps,
            fixed_step=fixed_step,
            headless=True,
            use_time_feature=use_time_feature,
            normalize_reward=False,  # Don't normalize during eval
        )
        env = Monitor(env)
        return env
    
    return _init


def train_agent(
    algo_name="PPO",
    total_timesteps=100000,
    map_name="Medium01",
    max_steps=500,
    fixed_step=10,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_time_feature=False,
    normalize_reward=False,
    save_freq=10000,
    eval_freq=10000,
    n_eval_episodes=5,
    save_path="models",
    log_path="logs",
    load_path=None,
    tensorboard_log=True,
    verbose=1,
):
    """
    Train RL agent on DroneRLEnv.
    
    Args:
        algo_name: Algorithm to use (PPO, SAC, TD3, A2C, DQN)
    
    Args:
        total_timesteps: Total training timesteps
        map_name: Map to train on
        max_steps: Max steps per episode
        fixed_step: Simulator ticks per action
        learning_rate: PPO learning rate
        n_steps: Steps per rollout
        batch_size: Minibatch size
        n_epochs: Optimization epochs per rollout
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Gradient clipping
        use_time_feature: Add timestep feature to observations
        normalize_reward: Normalize rewards
        save_freq: Save checkpoint every N steps
        eval_freq: Evaluate every N steps
        n_eval_episodes: Episodes for evaluation
        save_path: Directory to save models
        log_path: Directory for tensorboard logs
        load_path: Path to load existing model
        tensorboard_log: Enable tensorboard logging
        verbose: Verbosity level
    
    Returns:
        Trained PPO model
    """
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Get algorithm class
    AlgoClass = get_algorithm_class(algo_name)
    
    print("=" * 60)
    print(f"{algo_name.upper()} TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Algorithm: {algo_name.upper()}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Map: {map_name}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Fixed step (simulator ticks): {fixed_step}")
    print(f"Learning rate: {learning_rate}")
    print(f"Use time feature: {use_time_feature}")
    print(f"Normalize reward: {normalize_reward}")
    print("=" * 60)
    
    # Create training environment
    print("\nCreating training environment...")
    env = DummyVecEnv([
        make_train_env(
            map_name=map_name,
            max_steps=max_steps,
            fixed_step=fixed_step,
            use_time_feature=use_time_feature,
            normalize_reward=normalize_reward,
        )
    ])
    env = VecMonitor(env)
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([
        make_eval_env(
            map_name=map_name,
            max_steps=max_steps,
            fixed_step=fixed_step,
            use_time_feature=use_time_feature,
        )
    ])
    eval_env = VecMonitor(eval_env)
    
    # Create or load model
    if load_path and os.path.exists(load_path):
        print(f"\nLoading model from {load_path}...")
        model = AlgoClass.load(load_path, env=env)
    else:
        print(f"\nCreating new {algo_name.upper()} model...")
        
        # Build model kwargs based on algorithm
        model_kwargs = {
            "policy": "MlpPolicy",
            "env": env,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "verbose": verbose,
            "tensorboard_log": log_path if tensorboard_log else None,
        }
        
        # Add algorithm-specific parameters
        if algo_name.upper() in ["PPO", "A2C"]:
            model_kwargs.update({
                "n_steps": n_steps,
                "ent_coef": ent_coef,
                "vf_coef": vf_coef,
                "max_grad_norm": max_grad_norm,
            })
        if algo_name.upper() == "PPO":
            model_kwargs.update({
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "gae_lambda": gae_lambda,
                "clip_range": clip_range,
            })
        
        model = AlgoClass(**model_kwargs)
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_path,
        name_prefix=f"{algo_name.lower()}_drone",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)
    
    # Progress callback
    progress_callback = ProgressCallback(check_freq=10000, verbose=verbose)
    callbacks.append(progress_callback)
    
    # Tensorboard callback
    if tensorboard_log:
        tb_callback = TensorboardCallback()
        callbacks.append(tb_callback)
    
    callback_list = CallbackList(callbacks)
    
    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Monitor training with: tensorboard --logdir {log_path}\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    # Save final model
    final_path = os.path.join(save_path, f"{algo_name.lower()}_drone_final.zip")
    model.save(final_path)
    print(f"\n✅ Final model saved to {final_path}")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    return model


def evaluate_model(
    model_path,
    n_eval_episodes=10,
    map_name="Medium01",
    max_steps=500,
    fixed_step=10,
    use_time_feature=False,
    render=False,
    deterministic=True,
):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to saved model
        n_eval_episodes: Number of episodes to evaluate
        map_name: Map to evaluate on
        max_steps: Max steps per episode
        fixed_step: Simulator ticks per action
        use_time_feature: Whether model was trained with time feature
        render: Whether to render (not headless)
        deterministic: Use deterministic actions
    
    Returns:
        Dictionary with evaluation statistics
    """
    
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_eval_episodes}")
    print(f"Map: {map_name}")
    print(f"Deterministic: {deterministic}")
    print("=" * 60)
    
    # Load model (detect algorithm from file)
    print("\nLoading model...")
    # Try to load with BaseAlgorithm which works for all SB3 algorithms
    from stable_baselines3.common.base_class import BaseAlgorithm
    model = BaseAlgorithm.load(model_path)
    
    # Create environment
    print("Creating environment...")
    env = make_sb3_env(
        map_name=map_name,
        max_steps=max_steps,
        fixed_step=fixed_step,
        headless=not render,
        use_time_feature=use_time_feature,
        normalize_reward=False,
    )
    env = Monitor(env)
    
    # Evaluate
    episode_rewards = []
    episode_lengths = []
    episode_rescues = []
    
    for episode in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_rescues.append(info.get("total_rescued", 0))
        
        print(f"\nEpisode {episode + 1}/{n_eval_episodes}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length}")
        print(f"  Rescued: {info.get('total_rescued', 0)}")
    
    # Statistics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Mean rescues: {np.mean(episode_rescues):.2f} ± {np.std(episode_rescues):.2f}")
    print(f"Success rate: {100 * np.sum(np.array(episode_rescues) > 0) / n_eval_episodes:.1f}%")
    print(f"Best reward: {np.max(episode_rewards):.2f}")
    print(f"Worst reward: {np.min(episode_rewards):.2f}")
    
    env.close()
    
    return {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "rescues": episode_rescues,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate RL agent for drone rescue")
    
    # Mode
    parser.add_argument("--eval", action="store_true", help="Evaluation mode")
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "SAC", "TD3", "A2C", "DQN"],
                        help="RL algorithm to use")
    
    # Training parameters
    parser.add_argument("--total-timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per rollout (PPO/A2C)")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size (PPO)")
    parser.add_argument("--n-epochs", type=int, default=10, help="Optimization epochs per rollout (PPO)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda (PPO)")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    
    # Environment parameters
    parser.add_argument("--map", type=str, default="Medium01", choices=["Medium01", "Intermediate01"],
                        help="Map to use")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--fixed-step", type=int, default=10, help="Simulator ticks per action")
    parser.add_argument("--use-time-feature", action="store_true", help="Add time feature")
    parser.add_argument("--normalize-reward", action="store_true", help="Normalize rewards")
    
    # Evaluation parameters
    parser.add_argument("--n-eval-episodes", type=int, default=5, help="Episodes for evaluation")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions in eval")
    parser.add_argument("--render", action="store_true", help="Render during evaluation")
    
    # Paths
    parser.add_argument("--save-path", type=str, default="models", help="Model save directory")
    parser.add_argument("--log-path", type=str, default="logs", help="Tensorboard log directory")
    parser.add_argument("--load-path", type=str, default=None, help="Path to load model")
    
    # Misc
    parser.add_argument("--save-freq", type=int, default=10000, help="Save checkpoint frequency")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency during training")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    
    args = parser.parse_args()
    
    if args.eval:
        # Evaluation mode
        if not args.load_path:
            raise ValueError("Must provide --load-path for evaluation mode")
        
        evaluate_model(
            model_path=args.load_path,
            n_eval_episodes=args.n_eval_episodes,
            map_name=args.map,
            max_steps=args.max_steps,
            fixed_step=args.fixed_step,
            use_time_feature=args.use_time_feature,
            render=args.render,
            deterministic=args.deterministic,
        )
    else:
        # Training mode
        train_agent(
            algo_name=args.algo,
            total_timesteps=args.total_timesteps,
            map_name=args.map,
            max_steps=args.max_steps,
            fixed_step=args.fixed_step,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            use_time_feature=args.use_time_feature,
            normalize_reward=args.normalize_reward,
            save_freq=args.save_freq,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            save_path=args.save_path,
            log_path=args.log_path,
            load_path=args.load_path,
            verbose=args.verbose,
        )
