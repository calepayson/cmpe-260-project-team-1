#!/usr/bin/env python3
"""
train_fetch_ppo.py
Stable PPO training for FetchReach-v4 with collapse prevention
"""

import os
import yaml
import argparse
import logging
import time
from typing import Any, Dict, Optional
import matplotlib.pyplot as plt

import gymnasium as gym
import gymnasium_robotics  # ensure the robotics envs are registered
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed

class StabilityCallback(BaseCallback):
    """Custom callback to monitor training stability and prevent collapse"""
    
    def __init__(self, check_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.explained_variances = []
        
    def _on_step(self) -> bool:
        # Collect episode statistics
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
        
        # Check training stability every check_freq steps
        if self.n_calls % self.check_freq == 0:
            self._check_stability()
            
        return True
    
    def _check_stability(self):
        """Check for signs of training collapse"""
        if len(self.episode_rewards) < 10:
            return
            
        recent_rewards = self.episode_rewards[-10:]
        avg_recent_reward = np.mean(recent_rewards)
        std_recent_reward = np.std(recent_rewards)
        
        # Log statistics
        self.logger.record("stability/avg_episode_reward", avg_recent_reward)
        self.logger.record("stability/std_episode_reward", std_recent_reward)
        
        # Check for collapse indicators
        if std_recent_reward < 0.01 and avg_recent_reward < -40:  # Adjust thresholds based on your task
            logging.warning("Potential policy collapse detected: very low reward with no variance")
        
        if len(self.episode_rewards) > 50:
            long_term_avg = np.mean(self.episode_rewards[-50:-10])
            recent_avg = np.mean(self.episode_rewards[-10:])
            if recent_avg < 0.5 * long_term_avg and long_term_avg > -30:
                logging.warning("Performance degradation detected: recent performance much worse than before")

class TrainingMetricsCallback(BaseCallback):
    """Log train episode reward/length every `log_freq` callback steps."""
    def __init__(self, log_freq: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.ep_rewards = []
        self.ep_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])

        if self.n_calls % self.log_freq == 0 and self.ep_rewards:
            mean_r = float(np.mean(self.ep_rewards[-10:]))
            std_r = float(np.std(self.ep_rewards[-10:]))
            mean_l = float(np.mean(self.ep_lengths[-10:]))
            self.logger.record("train/mean_ep_reward", mean_r)
            self.logger.record("train/std_ep_reward", std_r)
            self.logger.record("train/mean_ep_length", mean_l)
            self.logger.record("timesteps", self.num_timesteps)
            if self.verbose:
                logging.info(f"[Train] t={self.num_timesteps:,}  R={mean_r:.2f} ± {std_r:.2f}  L={mean_l:.1f}")
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on FetchReach‑v4 with SB3")
    parser.add_argument("--config", type=str, default="configs/ppo_config.yaml", help="Path to config YAML file")
    return parser.parse_args()

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def validate_config(cfg: Dict[str, Any]) -> None:
    # Basic validation of required keys
    required_keys = ["env", "algo", "training", "seed"]
    for k in required_keys:
        if k not in cfg:
            raise ValueError(f"Missing required config key: {k}")
    
    # Validate hyperparameters for stability
    algo_cfg = cfg["algo"]
    
    # Check for potentially unstable combinations
    n_steps = algo_cfg.get("n_steps", 2048)
    batch_size = algo_cfg.get("batch_size", 64)
    n_epochs = algo_cfg.get("n_epochs", 10)
    learning_rate = algo_cfg.get("learning_rate", 3e-4)
    
    total_updates = (n_steps // batch_size) * n_epochs
    if total_updates > 100:
        logging.warning(f"High number of gradient updates per rollout ({total_updates}). Consider reducing n_epochs or increasing batch_size.")
    
    if learning_rate > 5e-4:
        logging.warning(f"Learning rate {learning_rate} might be too high for stable training.")
    
    logging.debug("Config successfully validated.")

def setup_logging_and_directories(cfg: Dict[str, Any]) -> None:
    """Setup logging and create necessary directories"""
    train_cfg = cfg["training"]
    
    # Create directories
    os.makedirs(train_cfg["tensorboard_log"], exist_ok=True)
    os.makedirs(train_cfg["checkpoint"]["save_path"], exist_ok=True)
    
    # Create CSV log directory if specified
    csv_log = train_cfg.get("csv_log")
    if csv_log:
        os.makedirs(os.path.dirname(csv_log), exist_ok=True)

def resolve_activation_fn(name: str):
    """Map string names to PyTorch activation functions"""
    name = name.lower()
    if name == "relu":
        return torch.nn.ReLU
    elif name == "tanh":
        return torch.nn.Tanh
    elif name == "elu":
        return torch.nn.ELU
    elif name == "leakyrelu":
        return torch.nn.LeakyReLU
    elif name == "sigmoid":
        return torch.nn.Sigmoid
    else:
        raise ValueError(f"Unknown activation function: {name}")

def make_envs(cfg: Dict[str, Any]) -> VecEnv:
    """Create vectorized training environments with proper wrappers"""
    env_cfg = cfg["env"]
    seed = cfg.get("seed", 0)
    n_envs = env_cfg["n_envs"]
    env_id = env_cfg["id"]
    env_kwargs = env_cfg.get("env_kwargs", {})

    # Set random seed for reproducibility
    set_random_seed(seed)

    env = make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
        wrapper_class=Monitor,
    )

    if cfg.get("vec_normalize", {}).get("enabled", False):
        vn_cfg = cfg["vec_normalize"]
        env = VecNormalize(
            env,
            norm_obs=vn_cfg.get("norm_obs", True),
            norm_reward=vn_cfg.get("norm_reward", False),
            clip_obs=vn_cfg.get("clip_obs", None),
            gamma=cfg["algo"].get("gamma", 0.99),
            training=True,
        )
    return env

def make_eval_env(cfg: Dict[str, Any], training_env: Optional[VecNormalize] = None) -> VecEnv:
    """Create evaluation environment with proper normalization"""
    env_cfg = cfg["env"]
    eval_cfg = cfg.get("evaluation", {})
    seed = cfg.get("seed", 0)
    env_id = env_cfg["id"]
    eval_env_kwargs = eval_cfg.get("eval_env_kwargs", {})

    eval_env = make_vec_env(
        env_id,
        n_envs=1,
        seed=seed + 1000,  # Different seed for eval
        env_kwargs=eval_env_kwargs,
        wrapper_class=Monitor,
    )

    # Apply VecNormalize if enabled
    if cfg.get("vec_normalize", {}).get("enabled", False):
        if training_env is not None and isinstance(training_env, VecNormalize):
            # Copy normalization stats from training environment
            eval_env = VecNormalize(eval_env, training=False, norm_reward=False)
            eval_env.obs_rms = training_env.obs_rms
            eval_env.ret_rms = training_env.ret_rms
        else:
            # Create new VecNormalize for final evaluation
            eval_env = VecNormalize(eval_env, training=False, norm_reward=False)
    
    return eval_env

def train(cfg: Dict[str, Any]) -> None:
    """Main training function with stability monitoring"""
    
    # Generate output directory structure: results/{algo_env}/{timestamp}/
    env_id_clean = cfg["env"]["id"].replace("-", "_").lower()
    algo_name = cfg["algo_name"].lower()
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Root output directory
    root_log_dir = os.path.join("results", f"{algo_name}_{env_id_clean}", timestamp)
    os.makedirs(root_log_dir, exist_ok=True)

    # Override config paths to use structured directories
    cfg["training"]["tensorboard_log"] = os.path.join(root_log_dir, "tensorboard")
    cfg["training"]["csv_log"] = os.path.join(root_log_dir, "progress", "log.csv")
    cfg["training"]["checkpoint"]["save_path"] = os.path.join(root_log_dir, "checkpoints")
    cfg["training"]["checkpoint"]["name_prefix"] = f"{algo_name}_{env_id_clean}"
    # Setup
    seed = cfg["seed"]
    set_random_seed(seed)
    setup_logging_and_directories(cfg)
    
    train_cfg = cfg["training"]
    ckpt_cfg = train_cfg["checkpoint"]
    
    # Configure SB3 logger
    csv_log = train_cfg.get("csv_log")
    log_folder = os.path.dirname(csv_log) if csv_log else train_cfg["tensorboard_log"]
    new_logger = configure(folder=log_folder, format_strings=["stdout", "tensorboard", "csv"])
    
    logging.info(f"TensorBoard logdir: {train_cfg['tensorboard_log']}")
    logging.info(f"Model save path: {ckpt_cfg['save_path']}")

    # Create environments
    env = make_envs(cfg)
    logging.info(f"Created {env.num_envs} training environments")

    # Handle resume logic
    model: Optional[PPO] = None
    if train_cfg.get("resume", False):
        resume_path = train_cfg.get("resume_model_path")
        if not resume_path or not os.path.exists(resume_path):
            raise ValueError(f"resume_model_path must exist when resume=true. Path: {resume_path}")

        # Load VecNormalize first so the model binds to the correct env
        if cfg.get("vec_normalize", {}).get("enabled", False):
            vec_path = train_cfg.get("resume_vecnormalize_path")
            if not vec_path or not os.path.exists(vec_path):
                raise ValueError("resume_vecnormalize_path must exist when resume=true")
            # Unwrap if already normalized
            if isinstance(env, VecNormalize):
                env = env.venv
            env = VecNormalize.load(vec_path, env)
            env.training = True
            env.norm_reward = cfg["vec_normalize"].get("norm_reward", False)

        logging.info(f"Resuming model from {resume_path}")
        model = PPO.load(resume_path, env=env, device="auto")
        model.set_env(env)
    else:
        # Create new model
        algo_cfg = cfg["algo"].copy()  # Make a copy to avoid modifying original
        
        policy_kwargs = algo_cfg.pop("policy_kwargs", {})

        # Convert activation_fn string to actual class
        if "activation_fn" in policy_kwargs and isinstance(policy_kwargs["activation_fn"], str):
            policy_kwargs["activation_fn"] = resolve_activation_fn(policy_kwargs["activation_fn"])
        
        # Log hyperparameters
        logging.info("PPO Hyperparameters:")
        for key, value in algo_cfg.items():
            if key != "policy_kwargs":
                logging.info(f"  {key}: {value}")
        
        model = PPO(
            algo_cfg["policy"],
            env,
            verbose=algo_cfg.get("verbose", 1),
            tensorboard_log=train_cfg["tensorboard_log"],
            policy_kwargs=policy_kwargs,
            **{k: v for k, v in algo_cfg.items() if k not in ["policy", "verbose", "policy_kwargs"]}
        )
    
    model.set_logger(new_logger)
    
    # Log model info
    logging.info(f"Model device: {model.device}")
    logging.info(f"Total parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

    # Setup callbacks
    callbacks = []

    # Compute a uniform logging frequency aligned with evaluation cadence
    raw_eval_freq = cfg["evaluation"].get("eval_freq_timesteps", env.num_envs * model.n_steps)
    uniform_cb_freq = max(raw_eval_freq // env.num_envs, 1)

    # Training metrics (TensorBoard)
    training_metrics_cb = TrainingMetricsCallback(log_freq=uniform_cb_freq, verbose=1)
    callbacks.append(training_metrics_cb)

    # Stability monitoring
    stability_cb = StabilityCallback(check_freq=uniform_cb_freq, verbose=1)
    callbacks.append(stability_cb)
    
    # Checkpoint callback
    if ckpt_cfg.get("enabled", False):
        checkpoint_cb = CheckpointCallback(
            save_freq=ckpt_cfg["save_freq_timesteps"],
            save_path=ckpt_cfg["save_path"],
            name_prefix=ckpt_cfg["name_prefix"],
            verbose=1
        )
        callbacks.append(checkpoint_cb)

    # Evaluation callback
    if cfg.get("evaluation", {}).get("enabled", False):
        eval_cfg = cfg["evaluation"]
        eval_env = make_eval_env(cfg, env if isinstance(env, VecNormalize) else None)

        # Use eval_freq_timesteps from config; convert timesteps -> callback steps
        # Because EvalCallback checks every `n_calls` (one per `env.step()` across all envs),
        # we divide by num_envs to align with timesteps.
        raw_eval_freq = eval_cfg.get("eval_freq_timesteps", env.num_envs * model.n_steps)
        eval_freq = max(raw_eval_freq // env.num_envs, 1)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=ckpt_cfg["save_path"],
            log_path=ckpt_cfg["save_path"],
            n_eval_episodes=eval_cfg["n_eval_episodes"],
            eval_freq=eval_freq,
            deterministic=eval_cfg["deterministic"],
            render=eval_cfg["render"],
            verbose=1
        )
        callbacks.append(eval_callback)


    # Start training
    total_timesteps = train_cfg["total_timesteps"]
    rollout_size = env.num_envs * model.n_steps
    total_updates = (rollout_size // model.batch_size) * model.n_epochs
    
    logging.info(f"Starting training for {total_timesteps:,} timesteps")
    logging.info(f"Rollout size: {env.num_envs} envs × {model.n_steps} steps = {rollout_size:,} timesteps")
    logging.info(f"Updates per rollout: {total_updates}")
    logging.info(f"Estimated number of rollouts: {total_timesteps // rollout_size}")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks) if callbacks else None,
            progress_bar=True
        )
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    
    end_time = time.time()
    training_time = end_time - start_time
    logging.info(f"Training completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")

    # Save final model and stats
    final_model_path = os.path.join(ckpt_cfg["save_path"], "final_model")
    model.save(final_model_path)
    logging.info(f"Saved final model to: {final_model_path}.zip")

    if cfg.get("vec_normalize", {}).get("enabled", False):
        vn_save_path = os.path.join(ckpt_cfg["save_path"], "vec_normalize_stats.pkl")
        env.save(vn_save_path)
        logging.info(f"Saved VecNormalize stats to: {vn_save_path}")

    env.close()


def evaluate(cfg: Dict[str, Any]) -> None:
    """Enhanced evaluation with detailed statistics, fixed for SB3 VecEnv API"""
    if not cfg.get("evaluation", {}).get("enabled", False):
        logging.info("Evaluation disabled in config; skipping.")
        return

    logging.info("Starting final evaluation...")
    eval_cfg = cfg["evaluation"]
    ckpt_cfg = cfg["training"]["checkpoint"]

    # Plain eval env, then load VecNormalize once if present
    env_id = cfg["env"]["id"]
    seed = cfg.get("seed", 0)
    eval_env_kwargs = eval_cfg.get("eval_env_kwargs", {})

    eval_env = make_vec_env(
        env_id,
        n_envs=1,
        seed=seed + 1000,
        env_kwargs=eval_env_kwargs,
        wrapper_class=Monitor,
    )

    if cfg.get("vec_normalize", {}).get("enabled", False):
        vn_path = os.path.join(ckpt_cfg["save_path"], "vec_normalize_stats.pkl")
        if os.path.exists(vn_path):
            logging.info(f"Loading VecNormalize stats from: {vn_path}")
            eval_env = VecNormalize.load(vn_path, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
        else:
            logging.warning(f"VecNormalize stats file not found at: {vn_path}")

    # Load model
    model_path = os.path.join(ckpt_cfg["save_path"], "final_model.zip")
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at: {model_path}")
        return

    logging.info(f"Loading model from: {model_path}")
    model = PPO.load(model_path, env=eval_env, device="auto")

    # Run evaluation episodes
    n_episodes = int(eval_cfg["n_eval_episodes"])
    deterministic = bool(eval_cfg["deterministic"])

    success_count = 0
    total_rewards: list[float] = []
    episode_lengths: list[int] = []

    for ep in range(n_episodes):
        # VecEnv.reset() returns only obs
        obs = eval_env.reset()
        episode_reward = 0.0
        step_count = 0

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            # VecEnv.step() returns 4 values: obs, rewards, dones, infos
            obs, rewards, dones, infos = eval_env.step(action)

            # Convert vectorized returns to scalars
            r = float(np.asarray(rewards).reshape(-1)[0])
            episode_reward += r
            step_count += 1

            done = bool(np.asarray(dones).reshape(-1)[0])

            if done:
                info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
                # Gymnasium robotics sets "is_success" in info
                is_success = info0.get("is_success", False)
                is_success = bool(np.asarray(is_success).reshape(-1)[0]) if isinstance(is_success, (np.ndarray, list, tuple)) else bool(is_success)

                if is_success:
                    success_count += 1

                total_rewards.append(float(episode_reward))
                episode_lengths.append(int(step_count))

                if (ep + 1) % 10 == 0:
                    logging.info(
                        f"Episode {ep+1}/{n_episodes}: Reward={episode_reward:.3f}, "
                        f"Steps={step_count}, Success={is_success}"
                    )
                break

    # Aggregate stats
    success_rate = success_count / n_episodes if n_episodes > 0 else 0.0
    avg_reward = float(np.mean(total_rewards)) if total_rewards else 0.0
    std_reward = float(np.std(total_rewards)) if total_rewards else 0.0
    avg_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0

    logging.info("=" * 60)
    logging.info("FINAL EVALUATION RESULTS:")
    logging.info(f"  Episodes: {n_episodes}")
    logging.info(f"  Success rate: {success_rate:.1%} ({success_count}/{n_episodes})")
    logging.info(f"  Average reward: {avg_reward:.3f} ± {std_reward:.3f}")
    logging.info(f"  Average episode length: {avg_length:.1f} steps")
    if total_rewards:
        logging.info(f"  Best reward: {max(total_rewards):.3f}")
        logging.info(f"  Worst reward: {min(total_rewards):.3f}")
    logging.info("=" * 60)

    eval_env.close()


def main():
    # Setup logging first
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s — %(levelname)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training.log")
        ]
    )
    
    args = parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        logging.error(f"Config file not found: {args.config}")
        return
    
    cfg = load_config(args.config)
    validate_config(cfg)
    
    logging.info("Starting PPO training on FetchReach-v4")
    logging.info(f"Config: {args.config}")
    
    try:
        train(cfg)
        evaluate(cfg)
        logging.info("Training and evaluation completed successfully!")
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()