#!/usr/bin/env python3
"""
record_video_from_checkpoint.py

Generic video recorder for Stable-Baselines3 models (TD3, SAC, DDPG, PPO)
trained on Gymnasium / Gymnasium-Robotics environments.

Directory layout assumed (if --checkpoint-dir is NOT given):

results/
  <env-id>/
    <algo>_<env_id_sanitized>/
      <timestamp>/
        checkpoints/
          best_model.zip
          vec_normalize_stats.pkl

Example:
  results/Ant-v5/ddpg_ant_v5/20251127_124838/checkpoints
"""

import os
import argparse
import logging

import gymnasium as gym
import gymnasium_robotics

from stable_baselines3 import TD3, SAC, DDPG, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder

gym.register_envs(gymnasium_robotics)

ALGO_MAP = {
    "td3": TD3,
    "sac": SAC,
    "ddpg": DDPG,
    "ppo": PPO,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record video rollouts from a saved SB3 checkpoint"
    )

    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=list(ALGO_MAP.keys()),
        help="Algorithm used: td3 | sac | ddpg | ppo",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        required=True,
        help="Gymnasium environment ID (e.g. Ant-v5, HalfCheetah-v5, FetchReach-v4)",
    )

    # Option 1: explicit checkpoint dir
    parser.add_argument(
        "--checkpoint-dir",
        dest="checkpoint_dir",
        type=str,
        default=None,
        help="Directory with best_model.zip and vec_normalize_stats.pkl. "
             "If omitted, it is auto-resolved from --results-root / env / algo.",
    )

    # Option 2: resolve from a results root (matches your screenshot layout)
    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Root results directory used by training (default: results/)",
    )

    parser.add_argument(
        "--best-model-name",
        type=str,
        default="best_model.zip",
        help="Model filename (default: best_model.zip)",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="videos",
        help="Output directory for videos",
    )
    parser.add_argument(
        "--episodes",
        "--episode",
        dest="episodes",
        type=int,
        default=5,
        help="Number of episodes to record",
    )
    parser.add_argument(
        "--video-length",
        type=int,
        default=200,
        help="Max env steps per recorded episode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for env creation",
    )

    return parser.parse_args()


def resolve_checkpoint_dir(args) -> str:
    """
    If args.checkpoint_dir is set, use it.
    Otherwise infer:
      results_root / <env-id> / <algo>_<env_id_sanitized> / latest_timestamp / checkpoints
    """
    if args.checkpoint_dir is not None:
        return args.checkpoint_dir

    env_dir = os.path.join(args.results_root, args.env_id)
    if not os.path.isdir(env_dir):
        raise FileNotFoundError(f"Env directory not found: {env_dir}")

    env_id_sanitized = args.env_id.replace("-", "_").lower()
    algo_prefix = f"{args.algo.lower()}_{env_id_sanitized}"
    algo_root = os.path.join(env_dir, algo_prefix)
    if not os.path.isdir(algo_root):
        raise FileNotFoundError(f"Algo directory not found: {algo_root}")

    # find latest timestamp folder
    subdirs = [
        d for d in os.listdir(algo_root)
        if os.path.isdir(os.path.join(algo_root, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f"No run directories under: {algo_root}")

    latest_run = sorted(subdirs)[-1]  # timestamps are lexicographically ordered
    ckpt_dir = os.path.join(algo_root, latest_run, "checkpoints")

    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoints directory not found: {ckpt_dir}")

    logging.info(f"Auto-resolved checkpoint directory: {ckpt_dir}")
    return ckpt_dir


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()
    AlgoClass = ALGO_MAP[args.algo.lower()]

    ckpt_dir = resolve_checkpoint_dir(args)
    best_model_path = os.path.join(ckpt_dir, args.best_model_name)
    vecnorm_path = os.path.join(ckpt_dir, "vec_normalize_stats.pkl")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model not found: {best_model_path}")

    os.makedirs(args.video_dir, exist_ok=True)

    logging.info(f"Creating env: {args.env_id}")
    env = make_vec_env(
        args.env_id,
        n_envs=1,
        seed=args.seed,
        wrapper_class=Monitor,
    )

    if os.path.exists(vecnorm_path):
        logging.info(f"Loading VecNormalize stats from {vecnorm_path}")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
    else:
        logging.info("No VecNormalize stats found. Using raw env.")

    video_env = VecVideoRecorder(
        env,
        video_folder=args.video_dir,
        record_video_trigger=lambda step: step == 0,
        video_length=args.video_length,
        name_prefix=f"{args.algo.lower()}_{args.env_id}",
    )

    logging.info(f"Loading {args.algo.upper()} model from {best_model_path}")
    model = AlgoClass.load(best_model_path, env=video_env)

    obs = video_env.reset()
    ep = 0
    total_eps = args.episodes

    logging.info(
        f"Starting rollouts: {total_eps} episode(s), "
        f"video_length={args.video_length}, env={args.env_id}, algo={args.algo.upper()}"
    )

    while ep < total_eps:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = video_env.step(action)

        if dones[0]:
            ep += 1
            logging.info(f"Episode {ep}/{total_eps} finished")
            obs = video_env.reset()

    video_env.close()
    logging.info(f"Done. Videos saved to: {args.video_dir}")


if __name__ == "__main__":
    main()
