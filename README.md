# UAV Swarm Search and Rescue

Multi-UAV search and rescue simulation built for experimenting with cooperative reinforcement learning. The environment models a swarm of UAVs searching for victims while dealing with limited battery, obstacles, danger zones, charging stations, partial observations, and noisy sensing.

The current codebase includes three multi-agent reinforcement learning algorithms: **MAPPO**, **MASAC**, and **MATD3**.

## Video demo

MASAC policy running in the search-and-rescue environment:

https://github.com/user-attachments/assets/6495fa98-8ec6-45cc-9528-d96561a6aec3

## Environment

The simulator is implemented with the PettingZoo Parallel API so all UAVs act within the same environment step.

The default HARD scenario uses:

| Setting | Value |
|---|---:|
| Map size | 250 m × 250 m |
| UAVs | 4 |
| Victims | 40–55 |
| Obstacles | 30 |
| Danger zones | 12 |
| Charging stations | 2 |
| Episode length | 2,500 steps |
| Simulation timestep | 1 s |
| Horizontal speed | up to 5 m/s |
| Vertical speed | up to 2 m/s |
| Altitude | 3–40 m |
| Camera horizontal FOV | 90° |

Each UAV uses the continuous action

```text
[vx, vy, vz, land]
```

where the first three values control 3D motion and `land` is used for landing and charging behavior.

The local observation contains the UAV state together with nearby stations, teammates, obstacles, victims, coverage information, and remaining episode time. With the default configuration, the actor observation has 80 features. Training can also use a centralized critic state for CTDE.

## Algorithms

| Algorithm | Implementation |
|---|---|
| MAPPO | `training/algorithms/mappo/` |
| MASAC | `training/algorithms/masac/` |
| MATD3 | `training/algorithms/matd3/` |

The algorithms share the same environment and action interface so they can be trained and evaluated under the same scenario configuration.

## Reward functions

Two reward paths are available:

- `rewards/baseline_reward.py` — manually designed reward shaping.
- `rewards/llm_reward.py` — adapter for experiments with an LLM-generated reward function.

The reward includes task progress such as victim discovery and map coverage together with operational constraints such as battery use, collision avoidance, proximity, and charging behavior.

## Project structure

```text
uav_swarm/
├── config/                  # Environment and training configuration
├── core/                    # Coverage map, fleet management, map generation
├── entities/                # UAV, victim, obstacle, charging station
├── env_setup/               # Environment and PettingZoo wrapper
│   └── backends/            # Simulation backend
├── observation/             # Actor and centralized critic observations
├── rewards/                 # Baseline and LLM reward functions
├── sensors/                 # FOV and communication sensing
├── training/algorithms/     # MAPPO, MASAC and MATD3
├── visualization/           # 2D and 3D visualization
├── train_mappo.py
├── train_masac.py
├── train_matd3.py
├── evaluate.py
└── run_visualize.py
```

## Installation

Python 3.10+ is recommended.

```bash
git clone https://github.com/truongdaoanhduy/uav_swarm.git
cd uav_swarm

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Train one of the supported algorithms:

```bash
python train_mappo.py --total-episodes 3000 --seed 42
python train_masac.py --total-episodes 3000 --seed 42
python train_matd3.py --total-episodes 3000 --seed 42
```

To run an experiment with the generated reward function:

```bash
python train_masac.py \
  --total-episodes 3000 \
  --seed 42 \
  --llm-reward llm_reward_generated.py
```

Additional training options are available from each script with `--help`.

## Evaluation and visualization

```bash
python evaluate.py --help
python run_visualize.py --help
python plot_compare.py --help
python plot_same_algo.py --help
```

Evaluation scripts support checkpoint evaluation, multi-seed experiments, statistical comparison, and result plotting. The visualization package contains both 2D and 3D renderers for inspecting trained policies.

## Notes

The current implementation uses the Python logic backend. PyBullet and Isaac-based backends are not part of the active training pipeline. The simulator is intended for reinforcement-learning research and does not represent a complete real-world UAV flight-control stack.

