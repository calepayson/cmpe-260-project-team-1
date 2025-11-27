#!/usr/bin/env python3
"""
ddpg_experiment.py
DDPG + HER training script for FetchReach.
"""

# ============================================================================
# CRITICAL: Thread limits MUST be set BEFORE any other imports
# This prevents PyTorch/NumPy from using all CPU cores per subprocess
# ============================================================================
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Standard Library
import yaml
import argparse
import logging
import time
from typing import Any, Dict, Optional

# Pytorch
import numpy as np
import torch

torch.set_num_threads(1)

# Gymnasium (gym)
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

# Rich
from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.panel import Panel

# Stable Baselines
from stable_baselines3 import DDPG
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    VecNormalize,
    VecEnv,
    DummyVecEnv,
    SubprocVecEnv,
)
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines3.common.noise import VectorizedActionNoise


# ============================================================================
#                                CUSTOM CALLBACKS
# ============================================================================


class RichDashboardCallback(BaseCallback):
    """Live training dashboard using Rich library."""

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
        self.live = Live(
            self.generate_table(), refresh_per_second=2, console=self.console
        )
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

        # Calculate stats
        mean_r = np.mean(self.ep_rewards[-50:]) if self.ep_rewards else 0.0
        best_r = np.max(self.ep_rewards) if self.ep_rewards else 0.0
        mean_l = np.mean(self.ep_lengths[-50:]) if self.ep_lengths else 0.0
        success_rate = np.mean(self.successes[-100:]) if self.successes else 0.0

        table = Table(title=f"DDPG + HER Training ({progress:.1%})", box=None)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        table.add_row("Timesteps", f"{current_step:,} / {self.total_timesteps:,}")
        table.add_row("FPS", f"{fps}")
        table.add_row("Mean Reward (50 ep)", f"{mean_r:.2f}")
        table.add_row("Best Reward", f"{best_r:.2f}")
        table.add_row("Success Rate (100 ep)", f"{success_rate:.1%}")

        # Status indicator based on success rate
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
    """Track and log success rate."""

    def __init__(self, log_freq: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.successes = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
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


class SyncEvalCallback(EvalCallback):
    """
    Custom Callback that syncs normalization stats from train env to eval env
    before every evaluation.
    """

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync Observation Statistics
            if hasattr(self.training_env, "obs_rms") and hasattr(
                self.eval_env, "obs_rms"
            ):
                self.eval_env.obs_rms = self.training_env.obs_rms.copy()
        return super()._on_step()


# ============================================================================
#                                CONFIG HELPERS
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDPG+HER on FetchReach")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ddpg_config_fetchreach.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Config file '{path}' is empty or invalid.")
    return cfg


def resolve_activation_fn(name: str):
    name = name.lower()
    mapping = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "elu": torch.nn.ELU,
        "leakyrelu": torch.nn.LeakyReLU,
    }
    if name not in mapping:
        raise ValueError(f"Unknown activation: {name}")
    return mapping[name]


def setup_directories(cfg: Dict[str, Any]) -> None:
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
    logging.info(f"Output directory: {root}")


# ============================================================================
#                                ENVIRONMENT
# ============================================================================


def make_train_env(cfg: Dict[str, Any]) -> VecEnv:
    """Create a training environment with the proper VecNormalize wrapper."""
    env_cfg = cfg["env"]
    seed = cfg.get("seed", 42)
    n_envs = env_cfg.get("n_envs", 1)
    set_random_seed(seed)

    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv

    if n_envs > 1:
        logging.info(f"Using {n_envs} parallel environments with SubprocVecEnv")

    env = make_vec_env(
        env_cfg["id"],
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_cfg.get("env_kwargs", {}),
        wrapper_class=Monitor,
        vec_env_cls=vec_env_cls,
    )

    vec_norm_cfg = cfg.get("vec_normalize", {})
    if vec_norm_cfg.get("enabled", False):
        logging.info("Applying VecNormalize wrapper")
        env = VecNormalize(
            env,
            norm_obs=vec_norm_cfg.get("norm_obs", True),
            norm_reward=vec_norm_cfg.get("norm_reward", False),
            clip_obs=vec_norm_cfg.get("clip_obs", 10.0),
            gamma=vec_norm_cfg.get("gamma", 0.99),
            training=True,
        )

    return env


def make_eval_env(cfg: Dict[str, Any], train_env: Optional[VecEnv] = None) -> VecEnv:
    """
    Create an evaluation environment with the same normalization as the training
    environment.
    """
    env_cfg = cfg["env"]
    eval_cfg = cfg.get("evaluation", {})
    seed = cfg.get("seed", 42)

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

    vec_norm_cfg = cfg.get("vec_normalize", {})
    if vec_norm_cfg.get("enabled", False):
        eval_env = VecNormalize(
            eval_env,
            norm_obs=vec_norm_cfg.get("norm_obs", True),
            norm_reward=False,
            clip_obs=vec_norm_cfg.get("clip_obs", 10.0),
            gamma=vec_norm_cfg.get("gamma", 0.99),
            training=False,
        )

        if train_env is not None and isinstance(train_env, VecNormalize):
            eval_env.obs_rms = train_env.obs_rms
            eval_env.ret_rms = train_env.ret_rms
            logging.info("Synced normalization stats to eval env")

    return eval_env


# ============================================================================
#                                TRAINING
# ============================================================================


def train(cfg: Dict[str, Any], resume: bool = False) -> None:
    setup_directories(cfg)
    train_cfg = cfg["training"]
    algo_cfg = cfg["algo"]

    # Log performance settings
    logging.info("=" * 60)
    logging.info("PERFORMANCE CONFIGURATION")
    logging.info("=" * 60)
    logging.info(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")
    logging.info(f"PyTorch threads: {torch.get_num_threads()}")
    logging.info(f"MuJoCo GL: {os.environ.get('MUJOCO_GL', 'default')}")
    logging.info("=" * 60)

    # Configure logger
    logger = configure(
        folder=train_cfg["tensorboard_log"],
        format_strings=["stdout", "tensorboard", "csv"],
    )

    env = make_train_env(cfg)
    logging.info(f"Created {env.num_envs} training environment(s)")

    n_actions = env.action_space.shape[-1]

    # Action Noise Configuration
    noise_sigma = algo_cfg.pop("action_noise_sigma", 0.1)
    sigma_vec = noise_sigma * np.ones(n_actions)

    env_id = cfg["env"]["id"]

    # Only mask gripper for tasks that manipulate objects
    needs_gripper_lock = any(task in env_id for task in ["Pick", "Place"])

    if needs_gripper_lock and n_actions >= 4:
        sigma_vec[3] = 0.0
        logging.info("Gripper noise locked for object manipulation task")
    else:
        logging.info(f"Full action noise enabled for {env_id}")

    # Use OU noise for pushing tasks, Gaussian for reaching/picking
    if "Push" in env_id or "Slide" in env_id:
        logging.info("Using Ornstein-Uhlenbeck Noise (smooth exploration for pushing)")
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=sigma_vec,
            dt=1e-2,
            theta=0.15,
        )
    else:
        logging.info("Using Normal (Gaussian) Noise")
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=sigma_vec,
        )

    action_noise = VectorizedActionNoise(action_noise, n_envs=env.num_envs)

    # HER setup
    use_her = algo_cfg.pop("use_her", False)
    her_kwargs = algo_cfg.pop("her_kwargs", {})
    replay_buffer_class = HerReplayBuffer if use_her else None

    policy_kwargs = algo_cfg.pop("policy_kwargs", {})
    net_arch = policy_kwargs.get("net_arch", {})

    if isinstance(net_arch, dict) and "pi" in net_arch:
        total_params = sum(net_arch["pi"])
        if total_params > 700:
            logging.info(f"Large network [{net_arch['pi']}] - Expect slow training.")
        if net_arch["pi"][0] > 512:
            logging.warning("Oversized network detected! Overriding to [400, 300]")
            policy_kwargs["net_arch"] = dict(pi=[400, 300], qf=[400, 300])

    if "activation_fn" in policy_kwargs and isinstance(
        policy_kwargs["activation_fn"], str
    ):
        policy_kwargs["activation_fn"] = resolve_activation_fn(
            policy_kwargs["activation_fn"]
        )

    if resume or train_cfg.get("resume", False):
        resume_path = train_cfg.get("resume_model_path")
        logging.info(f"Resuming from {resume_path}")
        model = DDPG.load(
            resume_path,
            env=env,
            device="auto",
            custom_objects={"action_noise": action_noise},
        )
        if isinstance(env, VecNormalize):
            vec_norm_path = train_cfg.get("resume_vecnormalize_path")
            if vec_norm_path and os.path.exists(vec_norm_path):
                env = VecNormalize.load(vec_norm_path, env)
                logging.info(f"Loaded VecNormalize stats from {vec_norm_path}")
    else:
        policy_type = algo_cfg.pop("policy", "MultiInputPolicy")

        if "train_freq" in algo_cfg and isinstance(algo_cfg["train_freq"], list):
            algo_cfg["train_freq"] = tuple(algo_cfg["train_freq"])

        model = DDPG(
            policy=policy_type,
            env=env,
            action_noise=action_noise,
            verbose=algo_cfg.get("verbose", 1),
            tensorboard_log=train_cfg["tensorboard_log"],
            policy_kwargs=policy_kwargs,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=her_kwargs,
            **{k: v for k, v in algo_cfg.items() if k not in ["verbose"]},
        )

    model.set_logger(logger)

    callbacks = [
        RichDashboardCallback(total_timesteps=train_cfg["total_timesteps"]),
        SuccessRateCallback(log_freq=2000),
    ]

    ckpt_cfg = train_cfg["checkpoint"]
    if ckpt_cfg.get("enabled", True):
        callbacks.append(
            CheckpointCallback(
                save_freq=ckpt_cfg["save_freq_timesteps"],
                save_path=ckpt_cfg["save_path"],
                name_prefix=ckpt_cfg["name_prefix"],
            )
        )

    if cfg.get("evaluation", {}).get("enabled", True):
        eval_env = make_eval_env(cfg, env)
        callbacks.append(
            SyncEvalCallback(
                eval_env,
                best_model_save_path=ckpt_cfg["save_path"],
                log_path=ckpt_cfg["save_path"],
                eval_freq=cfg["evaluation"]["eval_freq_timesteps"],
                n_eval_episodes=cfg["evaluation"]["n_eval_episodes"],
                deterministic=True,
                render=False,
            )
        )

    logging.info(f"Starting training for {train_cfg['total_timesteps']:,} timesteps")
    logging.info("SPEED OPTIMIZATIONS ACTIVE:")
    logging.info(f"   - Parallel Envs: {env.num_envs}")
    logging.info(f"   - Gradient Steps: {model.gradient_steps}")
    logging.info(f"   - Batch Size: {model.batch_size}")
    logging.info(f"   - Network Size: {policy_kwargs.get('net_arch', 'default')}")

    try:
        model.learn(
            total_timesteps=train_cfg["total_timesteps"],
            callback=CallbackList(callbacks),
            progress_bar=False,
        )
    except KeyboardInterrupt:
        logging.info("Training interrupted")
    except Exception as e:
        logging.error(f"Training crashed: {e}", exc_info=True)
        raise

    final_path = os.path.join(ckpt_cfg["save_path"], "final_model")
    model.save(final_path)
    logging.info(f"Saved final model: {final_path}")

    if isinstance(env, VecNormalize):
        vec_norm_path = os.path.join(ckpt_cfg["save_path"], "vec_normalize_stats.pkl")
        env.save(vec_norm_path)
        logging.info(f"Saved VecNormalize stats: {vec_norm_path}")

    env.close()


# ============================================================================
#                                EVALUATION
# ============================================================================


def evaluate(cfg: Dict[str, Any]) -> None:
    if not cfg.get("evaluation", {}).get("enabled", True):
        return

    logging.info("=" * 60)
    logging.info("FINAL EVALUATION")
    logging.info("=" * 60)

    eval_cfg = cfg["evaluation"]
    ckpt_cfg = cfg["training"]["checkpoint"]

    merged_kwargs = dict(cfg["env"].get("env_kwargs", {}))
    merged_kwargs.update(eval_cfg.get("eval_env_kwargs", {}))

    eval_env = make_vec_env(
        cfg["env"]["id"],
        n_envs=1,
        seed=cfg["seed"] + 2000,
        env_kwargs=merged_kwargs,
        wrapper_class=Monitor,
    )

    vec_norm_cfg = cfg.get("vec_normalize", {})
    if vec_norm_cfg.get("enabled", False):
        vec_norm_path = os.path.join(ckpt_cfg["save_path"], "vec_normalize_stats.pkl")

        if os.path.exists(vec_norm_path):
            eval_env = VecNormalize.load(vec_norm_path, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
            logging.info("Loaded VecNormalize stats for evaluation")
        else:
            logging.warning("VecNormalize stats not found, using fresh wrapper")
            eval_env = VecNormalize(
                eval_env, norm_obs=True, norm_reward=False, training=False
            )

    model_path = os.path.join(ckpt_cfg["save_path"], "final_model.zip")
    if not os.path.exists(model_path):
        logging.error(f"Model missing: {model_path}")
        return

    model = DDPG.load(model_path, env=eval_env)

    n_episodes = eval_cfg.get("n_eval_episodes", 50)
    success_count = 0
    rewards = []

    for ep in range(n_episodes):
        obs = eval_env.reset()
        if isinstance(obs, tuple):
            obs, _ = obs
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, infos = eval_env.step(action)

            ep_reward += float(reward[0])
            done = done_arr[0]

            if done:
                fi = infos[0].get("final_info")
                if fi is not None and "is_success" in fi:
                    is_success = float(fi["is_success"])
                else:
                    is_success = float(infos[0].get("is_success", 0.0))

                success_count += int(is_success > 0.5)
                rewards.append(ep_reward)

        if (ep + 1) % 10 == 0:
            logging.info(f"Evaluated {ep + 1}/{n_episodes}...")

    success_rate = success_count / n_episodes
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    logging.info("=" * 60)
    logging.info(f"Success Rate: {success_rate:.1%}")
    logging.info(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    logging.info("=" * 60)

    eval_env.close()


# ============================================================================
#                                MAIN
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_args()

    if not os.path.exists(args.config):
        logging.error(f"Config not found: {args.config}")
    else:
        cfg = load_config(args.config)
        train(cfg, resume=args.resume)
        evaluate(cfg)
