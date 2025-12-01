#!/usr/bin/env python3
"""
record_video_from_checkpoint.py

Load a TD3 model (optionally trained with VecNormalize) from an arbitrary
checkpoint directory and record evaluation episodes as video.

Supports generic Gymnasium env IDs like:
  - Ant-v5
  - HalfCheetah-v5
  - FetchPush-v2
etc.
"""

import os
import argparse
import logging

import gymnasium as gym
import gymnasium_robotics  # safe to import even if not using robotics envs

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder

# Register robotics envs (no-op for mujoco classic tasks but harmless)
gym.register_envs(gymnasium_robotics)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record video rollouts from a saved TD3 checkpoint"
    )

    parser.add_argument(
        "--checkpoint-dir",
        dest="checkpoint_dir",
        type=str,
        required=True,
        help="Directory that contains best_model.zip (and optionally vec_normalize_stats.pkl)",
    )
    parser.add_argument(
        "--best-model-name",
        type=str,
        default="best_model.zip",
        help="Model filename inside checkpoint-dir (default: best_model.zip)",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="Ant-v5",
        help="Gymnasium environment ID (e.g. Ant-v5, FetchPush-v2)",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="videos",
        help="Output directory for videos",
    )
    # accept both --episodes and --episode for convenience
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
        help="Max number of env steps per recorded episode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for env creation",
    )

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()

    # FIX 1: correct attribute name
    ckpt_dir = args.checkpoint_dir
    best_model_path = os.path.join(ckpt_dir, args.best_model_name)

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model not found: {best_model_path}")

    vecnorm_path = os.path.join(ckpt_dir, "vec_normalize_stats.pkl")

    os.makedirs(args.video_dir, exist_ok=True)

    # Create evaluation env
    logging.info(f"Creating env: {args.env_id}")
    env = make_vec_env(
        args.env_id,
        n_envs=1,
        seed=args.seed,
        wrapper_class=Monitor,
    )

    # Restore VecNormalize if stats exist
    if os.path.exists(vecnorm_path):
        logging.info(f"Found VecNormalize stats at {vecnorm_path}, loading...")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
    else:
        logging.info("No VecNormalize stats found. Using raw env.")

    # Wrap env with video recorder
    video_env = VecVideoRecorder(
        env,
        video_folder=args.video_dir,
        record_video_trigger=lambda step: step == 0,  # record first rollout
        video_length=args.video_length,
        name_prefix="best_model_eval",
    )

    # Load model with env attached
    logging.info(f"Loading model from {best_model_path}")
    model = TD3.load(best_model_path, env=video_env)

    obs = video_env.reset()
    ep = 0
    total_eps = args.episodes

    logging.info(
        f"Starting rollouts: {total_eps} episode(s), "
        f"video_length={args.video_length}, env={args.env_id}"
    )

    while ep < total_eps:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = video_env.step(action)

        # dones is a vector of shape (n_envs,), here n_envs = 1
        if dones[0]:
            ep += 1
            logging.info(f"Episode {ep}/{total_eps} finished")
            obs = video_env.reset()

    video_env.close()
    logging.info(f"Done. Videos saved to: {args.video_dir}")


if __name__ == "__main__":
    main()
