# RL Algorithm Benchmark: PPO, DDPG, TD3, SAC

Comparative benchmark of four reinforcement learning algorithms across MuJoCo locomotion and robotic manipulation tasks.

## Algorithms

| Algorithm | Type | Policy |
|-----------|------|--------|
| PPO | On-policy | Stochastic |
| DDPG | Off-policy | Deterministic |
| TD3 | Off-policy | Deterministic |
| SAC | Off-policy | Stochastic (Max-Entropy) |

## Environments

| Environment | Type | Observation | Action |
|-------------|------|-------------|--------|
| FetchReach-v4 | Robotic Manipulation | Dict (goal-conditioned) | Continuous |
| HalfCheetah-v5 | Locomotion | Box | Continuous |
| Ant-v5 | Locomotion | Box | Continuous |

## Project Structure

```
.
├── configs/
│   ├── ppo_config_antv.yaml
│   ├── ppo_config_fetchreach.yaml
│   ├── ppo_config_halfcheetah.yaml
│   ├── sac_config_antv.yaml
│   ├── sac_config_halfcheetah.yaml
│   ├── td3_config_antv.yaml
│   ├── td3_config_fetchreach.yaml
│   └── td3_config_halfcheetah.yaml
├── results/
│   ├── Ant-v5/
│   │   ├── ddpg_ant_v5/
│   │   ├── ppo_ant_v5/
│   │   ├── sac_ant_v5/
│   │   └── td3_ant_v5/
│   ├── FetchReach-v4/
│   │   ├── ddpg_fetchreach_v4/
│   │   ├── ppo_fetchreach_v4/
│   │   ├── sac_fetchreach_v4/
│   │   └── td3_fetchreach_v4/
│   ├── HalfCheetah-v5/
│   │   ├── ddpg_halfcheetah_v5/
│   │   ├── ppo_halfcheetah_v5/
│   │   ├── sac_halfcheetah_v5/
│   │   └── td3_halfcheetah_v5/
│   └── plots/
├── ppo_experiment.py      # FetchReach training
├── ppo_universal.py       # HalfCheetah/Ant training
├── sac_universal.py       # HalfCheetah/Ant training
├── td3_experiment.py      # FetchReach training
├── td3_universal.py       # HalfCheetah/Ant training
└── requirements.txt
```

## Installation

### 1. Create Python Environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install MuJoCo

#### Linux/macOS

```bash
# Download MuJoCo
mkdir -p ~/.mujoco
wget https://github.com/google-deepmind/mujoco/releases/download/3.1.1/mujoco-3.1.1-linux-x86_64.tar.gz
tar -xzf mujoco-3.1.1-linux-x86_64.tar.gz -C ~/.mujoco

# Set environment variables (add to ~/.bashrc)
export MUJOCO_HOME=~/.mujoco/mujoco-3.1.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO_HOME/lib
```

#### Windows

> ⚠️ **Windows Users**: MuJoCo installation requires Visual C++ Build Tools.

1. Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Download MuJoCo from [releases](https://github.com/google-deepmind/mujoco/releases)
3. Extract to `C:\Users\<username>\.mujoco\mujoco-3.1.1`
4. Set environment variables:

```powershell
# PowerShell (run as Administrator)
[Environment]::SetEnvironmentVariable("MUJOCO_HOME", "C:\Users\$env:USERNAME\.mujoco\mujoco-3.1.1", "User")
```

### 4. Verify Installation

```python
python -c "import gymnasium; import mujoco; env = gymnasium.make('HalfCheetah-v5'); print('MuJoCo OK')"
```

For FetchReach (requires gymnasium-robotics):
```python
python -c "import gymnasium_robotics; import gymnasium; gymnasium.register_envs(gymnasium_robotics); env = gymnasium.make('FetchReach-v4'); print('Robotics OK')"
```

## Usage

### Training

#### FetchReach-v4

```bash
# PPO
python ppo_experiment.py --config configs/ppo_config_fetchreach.yaml

# TD3
python td3_experiment.py --config configs/td3_config_fetchreach.yaml
```

#### HalfCheetah-v5

```bash
# PPO
python ppo_universal.py --config configs/ppo_config_halfcheetah.yaml

# SAC
python sac_universal.py --config configs/sac_config_halfcheetah.yaml

# TD3
python td3_universal.py --config configs/td3_config_halfcheetah.yaml
```

#### Ant-v5

```bash
# PPO
python ppo_universal.py --config configs/ppo_config_antv.yaml

# SAC
python sac_universal.py --config configs/sac_config_antv.yaml

# TD3
python td3_universal.py --config configs/td3_config_antv.yaml
```

### Resume Training

```bash
python ppo_universal.py --config configs/ppo_config_halfcheetah.yaml --resume
```

> Set `resume: true` and `resume_model_path` in the config file.

## Visualization

### View All Algorithms for One Environment

```bash
# HalfCheetah comparison
tensorboard --logdir results/HalfCheetah-v5/

# Ant comparison
tensorboard --logdir results/Ant-v5/

# FetchReach comparison
tensorboard --logdir results/FetchReach-v4/
```

Open `http://localhost:6006` in browser.

### View Single Algorithm Run

```bash
tensorboard --logdir results/HalfCheetah-v5/ppo_halfcheetah_v5/
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| `eval/mean_reward` | Mean evaluation episode reward |
| `eval/mean_ep_length` | Mean evaluation episode length |
| `train/actor_loss` | Actor network loss (off-policy) |
| `train/critic_loss` | Critic network loss (off-policy) |
| `train/loss` | Total policy loss (PPO) |

## Reproducibility Workflow

Complete workflow to reproduce HalfCheetah benchmark:

```bash
# 1. Setup
python -m venv venv
.\venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Train all algorithms
python ppo_universal.py --config configs/ppo_config_halfcheetah.yaml
python sac_universal.py --config configs/sac_config_halfcheetah.yaml
python td3_universal.py --config configs/td3_config_halfcheetah.yaml

# 3. Visualize comparison
tensorboard --logdir results/HalfCheetah-v5/
```

## Configuration

All hyperparameters are defined in YAML config files. Key sections:

```yaml
env:
  id: "HalfCheetah-v5"
  n_envs: 8                    # Parallel environments

vec_normalize:
  enabled: true
  norm_obs: true
  norm_reward: true

algo:
  learning_rate: 0.0003
  batch_size: 256
  # Algorithm-specific params...

training:
  total_timesteps: 1000000
  checkpoint:
    save_freq_timesteps: 50000

evaluation:
  eval_freq_timesteps: 10000
  n_eval_episodes: 20
```

## Output Structure

Each training run creates:

```
results/<env>/<algo>_<env>/<timestamp>/
├── checkpoints/
│   ├── <algo>_<env>_<steps>_steps.zip
│   ├── final_model.zip
│   └── vec_normalize_stats.pkl
└── tensorboard/
    └── events.out.tfevents.*
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- ~16GB RAM for parallel environments
- MuJoCo 3.x

## License

MIT