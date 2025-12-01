import os
import logging
import gymnasium as gym
import gymnasium_robotics  # noqa: F401 - registers robotics envs

from stable_baselines3 import PPO, DDPG, TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

MODELS_DIR = "models"
VIDEOS_DIR = "videos"

ALGO_MAP = {"ppo": PPO, "ddpg": DDPG, "td3": TD3, "sac": SAC}
ENV_MAP = {
    "ant": "Ant-v5",
    "halfcheetah": "HalfCheetah-v5",
    "fetchreach": "FetchReach-v4",
}


def get_recording_length(env_id):
    """
    Returns the number of steps to record.
    FetchReach is 50 steps/episode -> 500 steps = 10 episodes.
    Ant/HalfCheetah are 1000 steps/episode -> 1000 steps = 1 episode.
    """
    if "FetchReach" in env_id:
        return 500
    return 1000


def make_env(env_id):
    """Factory function to avoid lambda closure issues."""

    def _init():
        return gym.make(env_id, render_mode="rgb_array")

    return _init


def run_recording():
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".zip")]

    logging.info(f"Found {len(model_files)} models in {MODELS_DIR}")

    for i, filename in enumerate(model_files, 1):
        try:
            name_parts = filename.replace(".zip", "").split("_")
            if len(name_parts) != 3:
                logging.warning(
                    f"[{i}/{len(model_files)}] Skipping {filename}: unexpected naming format."
                )
                continue

            algo_name, env_key, model_type = name_parts

            if algo_name not in ALGO_MAP or env_key not in ENV_MAP:
                logging.warning(
                    f"[{i}/{len(model_files)}] Skipping {filename}: Unknown algo or env key."
                )
                continue

            env_id = ENV_MAP[env_key]
            model_class = ALGO_MAP[algo_name]
            model_path = os.path.join(MODELS_DIR, filename)
            video_prefix = f"{algo_name}_{env_key}_{model_type}"
            rec_length = get_recording_length(env_id)

            logging.info(
                f"[{i}/{len(model_files)}] Processing {algo_name.upper()} on {env_id} ({model_type})..."
            )

            env = DummyVecEnv([make_env(env_id)])
            env = VecVideoRecorder(
                env,
                VIDEOS_DIR,
                record_video_trigger=lambda x: x == 0,
                video_length=rec_length,
                name_prefix=video_prefix,
            )

            model = model_class.load(model_path, env=env)
            obs = env.reset()

            episodes_completed = 0
            for step in range(rec_length):
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)

                if dones[0]:
                    episodes_completed += 1

            env.close()
            logging.info(
                f"[{i}/{len(model_files)}] Recorded {video_prefix}: "
                f"{rec_length} steps, {episodes_completed} episodes"
            )

        except Exception as e:
            logging.error(f"[{i}/{len(model_files)}] Failed to record {filename}: {e}")


if __name__ == "__main__":
    run_recording()
