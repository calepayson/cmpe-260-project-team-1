#!/usr/bin/env python3
"""
train_fetch_ppo.py
Stable PPO training for FetchReach-v4 with TD3-style logging
"""

import os
import yaml
import argparse
import logging
import time
from typing import Any, Dict, Optional

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecEnv, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed

from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.panel import Panel

# ============================================================================
#                                CUSTOM CALLBACKS
# ============================================================================

class RichDashboardCallback(BaseCallback):
    """Live training dashboard using Rich library (matching TD3 style)"""
    
    def __init__(self, total_timesteps: int, check_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.ep_rewards = []
        self.ep_lengths = []
        self.successes = []
        self.start_time = None
        self.live = None
        self.console = Console()
    
    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.live = Live(self.generate_table(), refresh_per_second=2, console=self.console)
        self.live.start()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            # Episode tracking
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])
            
            # Handle Gymnasium final_info first
            final_info = info.get("final_info")
            if final_info is not None and "is_success" in final_info:
                self.successes.append(float(final_info["is_success"]))
            elif "is_success" in info:
                self.successes.append(float(info["is_success"]))

        # Periodic UI update
        if self.n_calls % self.check_freq == 0:
            self.live.update(self.generate_table())

        return True

    def generate_table(self):
        current_step = self.num_timesteps
        progress = current_step / self.total_timesteps
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = int(current_step / elapsed) if elapsed > 0 else 0
        
        mean_r = np.mean(self.ep_rewards[-50:]) if self.ep_rewards else 0.0
        best_r = np.max(self.ep_rewards) if self.ep_rewards else 0.0
        mean_l = np.mean(self.ep_lengths[-50:]) if self.ep_lengths else 0.0
        success_rate = np.mean(self.successes[-100:]) if self.successes else 0.0
        
        table = Table(title=f"PPO Training ({progress:.1%})", box=None)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        table.add_row("Timesteps", f"{current_step:,} / {self.total_timesteps:,}")
        table.add_row("FPS", f"{fps}")
        table.add_row("Mean Reward (50 ep)", f"{mean_r:.2f}")
        table.add_row("Best Reward", f"{best_r:.2f}")
        table.add_row("Success Rate (100 ep)", f"{success_rate:.1%}")
        
        # Status indicator
        if success_rate > 0.8:
            status = "[bold green]SOLVED[/bold green]"
        elif success_rate > 0.5:
            status = "[green]EXCELLENT[/green]"
        elif success_rate > 0.2:
            status = "[yellow]IMPROVING[/yellow]"
        elif success_rate > 0.05:
            status = "[yellow]LEARNING[/yellow]"
        else:
            status = "[red]EXPLORING[/red]"
        
        table.add_row("Status", status)
        return Panel(table, title="FetchReach Training", border_style="blue")

    def _on_training_end(self) -> None:
        if self.live:
            self.live.stop()


class SuccessRateCallback(BaseCallback):
    """Track and log success rate (matching TD3 style)"""
    
    def __init__(self, log_freq: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.successes = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            # Handle Gymnasium final_info first
            final_info = info.get("final_info")
            if final_info is not None and "is_success" in final_info:
                self.successes.append(float(final_info["is_success"]))
            elif "is_success" in info:
                self.successes.append(float(info["is_success"]))

        if self.n_calls % self.log_freq == 0 and len(self.successes) >= 10:
            recent_sr = np.mean(self.successes[-100:])
            all_time_sr = np.mean(self.successes)
            
            self.logger.record("train/success_rate_recent", recent_sr)
            self.logger.record("train/success_rate_all", all_time_sr)
        return True


class StabilityCallback(BaseCallback):
    """Custom callback to monitor training stability and prevent collapse"""
    
    def __init__(self, check_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
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
        if std_recent_reward < 0.01 and avg_recent_reward < -40:
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


# ============================================================================
#                                CONFIG HELPERS
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on FetchReach‑v4 with SB3")
    parser.add_argument("--config", type=str, default="configs/ppo_config.yaml", help="Path to config YAML file")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Config file '{path}' is empty or invalid.")
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


def setup_directories(cfg: Dict[str, Any]) -> None:
    """Setup directories matching TD3 style"""
    env_id = cfg["env"]["id"].replace("-", "_").lower()
    algo = cfg["algo_name"].lower()
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    root = os.path.join("results", f"{algo}_{env_id}", timestamp)
    os.makedirs(root, exist_ok=True)

    cfg["training"]["tensorboard_log"] = os.path.join(root, "tensorboard")
    cfg["training"]["checkpoint"]["save_path"] = os.path.join(root, "checkpoints")
    cfg["training"]["checkpoint"]["name_prefix"] = f"{algo}_{env_id}"

    os.makedirs(cfg["training"]["tensorboard_log"], exist_ok=True)
    os.makedirs(cfg["training"]["checkpoint"]["save_path"], exist_ok=True)
    logging.info(f"📁 Output directory: {root}")


def resolve_activation_fn(name: str):
    """Map string names to PyTorch activation functions"""
    name = name.lower()
    mapping = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "elu": torch.nn.ELU,
        "leakyrelu": torch.nn.LeakyReLU,
        "sigmoid": torch.nn.Sigmoid,
    }
    if name not in mapping:
        raise ValueError(f"Unknown activation function: {name}")
    return mapping[name]


# ============================================================================
#                                ENVIRONMENT
# ============================================================================

def make_train_env(cfg: Dict[str, Any]) -> VecEnv:
    """Create training environment with proper VecNormalize wrapper"""
    env_cfg = cfg["env"]
    seed = cfg.get("seed", 0)
    n_envs = env_cfg.get("n_envs", 1)
    set_random_seed(seed)

    # Use SubprocVecEnv for parallel environments (faster than DummyVecEnv)
    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    
    if n_envs > 1:
        logging.info(f"🚀 Using {n_envs} parallel environments with SubprocVecEnv")

    env = make_vec_env(
        env_cfg["id"],
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_cfg.get("env_kwargs", {}),
        wrapper_class=Monitor,
        vec_env_cls=vec_env_cls,
    )

    if cfg.get("vec_normalize", {}).get("enabled", False):
        vn_cfg = cfg["vec_normalize"]
        logging.info("🔧 Applying VecNormalize wrapper")
        env = VecNormalize(
            env,
            norm_obs=vn_cfg.get("norm_obs", True),
            norm_reward=vn_cfg.get("norm_reward", False),
            clip_obs=vn_cfg.get("clip_obs", 10.0),
            gamma=cfg["algo"].get("gamma", 0.99),
            training=True,
        )
    return env


def make_eval_env(cfg: Dict[str, Any], train_env: Optional[VecEnv] = None) -> VecEnv:
    """Create evaluation environment with same normalization as training"""
    env_cfg = cfg["env"]
    eval_cfg = cfg.get("evaluation", {})
    seed = cfg.get("seed", 0)

    merged_kwargs = dict(env_cfg.get("env_kwargs", {}))
    merged_kwargs.update(eval_cfg.get("eval_env_kwargs", {}))

    eval_env = make_vec_env(
        env_cfg["id"],
        n_envs=1,
        seed=seed + 1000,
        env_kwargs=merged_kwargs,
        wrapper_class=Monitor,
        vec_env_cls=DummyVecEnv,
    )

    # Apply VecNormalize to eval env if training uses it
    vec_norm_cfg = cfg.get("vec_normalize", {})
    if vec_norm_cfg.get("enabled", False):
        eval_env = VecNormalize(
            eval_env,
            norm_obs=vec_norm_cfg.get("norm_obs", True),
            norm_reward=False,
            clip_obs=vec_norm_cfg.get("clip_obs", 10.0),
            gamma=vec_norm_cfg.get("gamma", 0.99),
            training=False
        )
        
        # Sync normalization stats from training env if available
        if train_env is not None and isinstance(train_env, VecNormalize):
            eval_env.obs_rms = train_env.obs_rms
            eval_env.ret_rms = train_env.ret_rms
            logging.info("✓ Synced normalization stats to eval env")
    
    return eval_env


# ============================================================================
#                                TRAINING
# ============================================================================

def train(cfg: Dict[str, Any], resume: bool = False) -> None:
    """Main training function with TD3-style logging"""
    
    setup_directories(cfg)
    train_cfg = cfg["training"]
    algo_cfg = cfg["algo"]
    
    # Log performance settings
    logging.info("=" * 60)
    logging.info("TRAINING CONFIGURATION")
    logging.info("=" * 60)
    
    # Setup
    seed = cfg["seed"]
    set_random_seed(seed)
    
    train_cfg = cfg["training"]
    ckpt_cfg = train_cfg["checkpoint"]
    
    # Configure logger
    logger = configure(
        folder=train_cfg["tensorboard_log"],
        format_strings=["stdout", "tensorboard", "csv"]
    )
    
    logging.info(f"TensorBoard logdir: {train_cfg['tensorboard_log']}")
    logging.info(f"Model save path: {ckpt_cfg['save_path']}")

    # Create environments
    env = make_train_env(cfg)
    logging.info(f"✓ Created {env.num_envs} training environment(s)")

    # Handle resume logic
    model: Optional[PPO] = None
    if resume or train_cfg.get("resume", False):
        resume_path = train_cfg.get("resume_model_path")
        if not resume_path or not os.path.exists(resume_path):
            raise ValueError(f"resume_model_path must exist when resume=true. Path: {resume_path}")

        logging.info(f"📂 Resuming from {resume_path}")
        
        # Load VecNormalize first
        if cfg.get("vec_normalize", {}).get("enabled", False):
            vec_path = train_cfg.get("resume_vecnormalize_path")
            if vec_path and os.path.exists(vec_path):
                if isinstance(env, VecNormalize):
                    env = env.venv
                env = VecNormalize.load(vec_path, env)
                env.training = True
                env.norm_reward = cfg["vec_normalize"].get("norm_reward", False)
                logging.info(f"✓ Loaded VecNormalize stats from {vec_path}")

        model = PPO.load(resume_path, env=env, device="auto")
        model.set_env(env)
    else:
        # Create new model
        algo_cfg_copy = cfg["algo"].copy()
        
        policy_kwargs = algo_cfg_copy.pop("policy_kwargs", {})

        # Convert activation_fn string to actual class
        if "activation_fn" in policy_kwargs and isinstance(policy_kwargs["activation_fn"], str):
            policy_kwargs["activation_fn"] = resolve_activation_fn(policy_kwargs["activation_fn"])
        
        # Log hyperparameters
        logging.info("PPO Hyperparameters:")
        for key, value in algo_cfg_copy.items():
            if key != "policy_kwargs":
                logging.info(f"  {key}: {value}")
        
        model = PPO(
            algo_cfg_copy["policy"],
            env,
            verbose=algo_cfg_copy.get("verbose", 1),
            tensorboard_log=train_cfg["tensorboard_log"],
            policy_kwargs=policy_kwargs,
            **{k: v for k, v in algo_cfg_copy.items() if k not in ["policy", "verbose", "policy_kwargs"]}
        )
    
    model.set_logger(logger)
    
    # Log model info
    logging.info(f"Model device: {model.device}")
    logging.info(f"Total parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

    # Setup callbacks (matching TD3 order and style)
    callbacks = [
        RichDashboardCallback(total_timesteps=train_cfg["total_timesteps"]),
        SuccessRateCallback(log_freq=2000),
    ]

    # Compute uniform logging frequency
    raw_eval_freq = cfg["evaluation"].get("eval_freq_timesteps", env.num_envs * model.n_steps)
    uniform_cb_freq = max(raw_eval_freq // env.num_envs, 1)

    # Training metrics
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

        raw_eval_freq = eval_cfg.get("eval_freq_timesteps", env.num_envs * model.n_steps)
        eval_freq = max(raw_eval_freq // env.num_envs, 1)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=ckpt_cfg["save_path"],
            log_path=ckpt_cfg["save_path"],
            n_eval_episodes=eval_cfg["n_eval_episodes"],
            eval_freq=eval_freq,
            deterministic=eval_cfg.get("deterministic", True),
            render=eval_cfg.get("render", False),
            verbose=1
        )
        callbacks.append(eval_callback)

    # Start training
    total_timesteps = train_cfg["total_timesteps"]
    rollout_size = env.num_envs * model.n_steps
    total_updates = (rollout_size // model.batch_size) * model.n_epochs
    
    logging.info(f"🚀 Starting training for {total_timesteps:,} timesteps")
    logging.info(f"⚡ TRAINING PARAMETERS:")
    logging.info(f"   - Parallel Envs: {env.num_envs}")
    logging.info(f"   - Rollout size: {rollout_size:,}")
    logging.info(f"   - Updates per rollout: {total_updates}")
    logging.info(f"   - Estimated rollouts: {total_timesteps // rollout_size}")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=False  # Use Rich dashboard instead
        )
    except KeyboardInterrupt:
        logging.info("⏸️  Training interrupted by user")
    except Exception as e:
        logging.error(f"❌ Training crashed: {e}", exc_info=True)
        raise
    
    end_time = time.time()
    training_time = end_time - start_time
    logging.info(f"Training completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")

    # Save final model and stats
    final_model_path = os.path.join(ckpt_cfg["save_path"], "final_model")
    model.save(final_model_path)
    logging.info(f"💾 Saved final model: {final_model_path}.zip")

    if cfg.get("vec_normalize", {}).get("enabled", False):
        vn_save_path = os.path.join(ckpt_cfg["save_path"], "vec_normalize_stats.pkl")
        env.save(vn_save_path)
        logging.info(f"💾 Saved VecNormalize stats: {vn_save_path}")

    env.close()


# ============================================================================
#                                EVALUATION
# ============================================================================

def evaluate(cfg: Dict[str, Any]) -> None:
    """CORRECTED evaluation with proper reset handling (matching TD3 style)"""
    if not cfg.get("evaluation", {}).get("enabled", False):
        logging.info("Evaluation disabled in config; skipping.")
        return

    logging.info("=" * 60)
    logging.info("FINAL EVALUATION")
    logging.info("=" * 60)
    
    eval_cfg = cfg["evaluation"]
    ckpt_cfg = cfg["training"]["checkpoint"]

    # Setup Env
    merged_kwargs = dict(cfg["env"].get("env_kwargs", {}))
    merged_kwargs.update(eval_cfg.get("eval_env_kwargs", {}))
    
    eval_env = make_vec_env(
        cfg["env"]["id"],
        n_envs=1,
        seed=cfg["seed"] + 2000,
        env_kwargs=merged_kwargs,
        wrapper_class=Monitor,
    )

    # Apply VecNormalize if used in training
    vec_norm_cfg = cfg.get("vec_normalize", {})
    if vec_norm_cfg.get("enabled", False):
        vec_norm_path = os.path.join(ckpt_cfg["save_path"], "vec_normalize_stats.pkl")
        
        if os.path.exists(vec_norm_path):
            eval_env = VecNormalize.load(vec_norm_path, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
            logging.info(f"✓ Loaded VecNormalize stats for evaluation")
        else:
            logging.warning("⚠️  VecNormalize stats not found, using fresh wrapper")
            eval_env = VecNormalize(
                eval_env,
                norm_obs=True,
                norm_reward=False,
                training=False
            )

    # Load model
    model_path = os.path.join(ckpt_cfg["save_path"], "final_model.zip")
    if not os.path.exists(model_path):
        logging.error(f"Model missing: {model_path}")
        return

    logging.info(f"Loading model from: {model_path}")
    model = PPO.load(model_path, env=eval_env, device="auto")

    # Run evaluation episodes
    n_episodes = eval_cfg.get("n_eval_episodes", 50)
    success_count = 0
    rewards = []
    
    # Proper reset handling for modern Gymnasium
    obs = eval_env.reset()
    
    # Handle both old and new Gymnasium API
    if isinstance(obs, tuple):
        obs, _ = obs
    
    for ep in range(n_episodes):
        done = False
        ep_reward = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, infos = eval_env.step(action)
            
            ep_reward += float(reward[0])
            done = done_arr[0]
            
            if done:
                # Handle Gymnasium final_info for success tracking
                fi = infos[0].get("final_info")
                if fi is not None and "is_success" in fi:
                    is_success = float(fi["is_success"])
                else:
                    is_success = float(infos[0].get("is_success", 0.0))

                success_count += int(is_success > 0.5)
                rewards.append(ep_reward)
        
        # VecEnv auto-resets, obs is already fresh
        if (ep + 1) % 10 == 0:
            logging.info(f"  Evaluated {ep+1}/{n_episodes}...")

    success_rate = success_count / n_episodes
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    logging.info("=" * 60)
    logging.info(f"📊 Success Rate: {success_rate:.1%}")
    logging.info(f"📊 Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    logging.info("=" * 60)
    
    eval_env.close()


# ============================================================================
#                                MAIN
# ============================================================================

def main():
    # Setup logging first
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training.log")
        ]
    )
    
    args = parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        logging.error(f"Config not found: {args.config}")
        return
    
    cfg = load_config(args.config)
    validate_config(cfg)
    
    logging.info("Starting PPO training on FetchReach-v4")
    logging.info(f"Config: {args.config}")
    
    try:
        train(cfg, resume=args.resume)
        evaluate(cfg)
        logging.info("Training and evaluation completed successfully!")
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()