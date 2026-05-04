🚁 SAR UAV SWARM — COMPLETE PROJECT DOCUMENTATION v12.0
Status: Phase 2 MAPPO 100% Complete + Production Ready | Auto-Balanced Vectorized Training

📋 TABLE OF CONTENTS
Project Overview
Research Goals
Overall Status
Key Metrics
Project Architecture
Detailed File Structure
Execution Flows
Training Results
Known Issues
Next Steps
🎯 PROJECT OVERVIEW
Domain: Multi-Agent Reinforcement Learning cho Search and Rescue (SAR) với UAV swarm

Core Task: 4 UAVs phối hợp tìm kiếm 10-36 victims trong disaster area có debris và danger zones, constraint battery (phải về trạm sạc)

Key Innovation:

Curriculum learning: EASY → MEDIUM → HARD (tăng dần map size)
Shared cooperative reward (tất cả agents nhận cùng reward)
Centralized Training Decentralized Execution (CTDE)
Auto-balanced vectorized training (n_envs=1-16)
Technology Stack:

Python 3.8+
PyTorch (neural networks)
PettingZoo (multi-agent env API)
Gymnasium (base env interface)
NumPy (vectorized geometry)
Matplotlib (visualization)
Multiprocessing (parallel training)
🔬 RESEARCH GOALS
Paper 1: Algorithm Comparison
Question: MAPPO vs MASAC vs MATD3 — Which is best for SAR coordination?
Method: 3 algorithms × 5 seeds × 3 curriculum stages × 3000 episodes
Metrics: Coverage rate, Victims found, Episode reward, Sample efficiency, Convergence speed
Paper 2: LLM-Generated vs Hand-Crafted Rewards
Question: Can LLM (GPT-4/Claude) generate better reward functions than human experts?
Baseline: BaselineReward v3.1 (hand-crafted, 14 components, research-grade)
LLM: Generate reward code from natural language task description
Metrics: Task performance, Training stability, Generalization
📊 OVERALL STATUS
Phase Description Status Completion
Phase 1 Core Infrastructure (48 files) ✅ Complete 100%
Phase 2 MAPPO Algorithm ✅ Complete 100%
Phase 2b Auto-Balanced Vectorized Training ✅ Complete 100%
Phase 3 MASAC & MATD3 Algorithms ⬜ Not Started 0%
Phase 4 LLM Reward Integration ⬜ Not Started 0%
Phase 5 Advanced Backends (PyBullet/Isaac) ⬜ Not Started 0%
Test Coverage: 26/26 core tests PASS (100%)

Training Infrastructure:

✅ Single-env training (baseline)
✅ Vectorized training (n_envs=1-16, auto-balanced)
✅ Curriculum learning (3 stages)
✅ Checkpointing & resumption
✅ Real-time visualization
✅ TensorBoard logging (pending)
📐 KEY METRICS
Observation & Action Spaces
Metric Value Description
Actor Obs Dim 68 Local observation per UAV (với n_stations=2)
Critic Obs Dim 554 Global observation (8×68 + 10 global features)
Action Dim 3 Continuous [vx, vy, vz] ∈ [-1, 1]
UAV States 5 ACTIVE/RETURNING/CHARGING/DEPLOYING/DISABLED
Reward System
Metric Value Description
Reward Type Shared Cooperative Tất cả agents nhận cùng reward value
Components 14 Coverage, Victims, Battery, Collision, Proximity, etc.
Range [-100, +100] Per-step clip range
Terminal Bonus +200 base +100 cap based on performance
Performance Benchmarks
Metric Value Notes
Backend Speed ~1000 steps/s Pure Python logic backend
Training FPS (n_envs=1) ~53 FPS CPU + GPU (actor/critic inference)
Training FPS (n_envs=6) ~105 FPS 2× speedup with vectorization
Episode Time (random) ~9s 300 steps @ 30ms/step
Episode Time (trained) ~6s Faster decisions
Training Time (3000 eps, n_envs=1) ~10 hours Full curriculum
Training Time (3000 eps, n_envs=6) ~4 hours 2.5× faster
Curriculum Stages
Stage Map Size Pressure (m²/UAV) Max Steps Advance Threshold
EASY 150×150m 5,625 300 cov≥70%, vic≥80%
MEDIUM 200×200m 10,000 350 cov≥65%, vic≥75%
HARD 250×250m 15,625 400 cov≥60%, vic≥70%
Victim Density: ~0.53 per 1000m² (constant across stages — controlled variable)

🏗️ PROJECT ARCHITECTURE
High-Level Components
text

┌─────────────────────────────────────────────────────────┐
│ TRAINING LOOP │
│ (train_mappo.py + MAPPOTrainer) │
└────────────────┬────────────────────────────────────────┘
│
┌───────────┴───────────┐
│ │
┌────▼────┐ ┌─────▼─────┐
│ Actor │ │ Critic │
│ Network │ │ Network │
│ (68→3) │ │ (554→1) │
└────┬────┘ └─────┬─────┘
│ │
└───────────┬───────────┘
│
┌────────▼────────┐
│ Rollout Buffer │
│ + GAE Compute │
└────────┬────────┘
│
┌───────────▼───────────┐
│ │
┌────▼────────┐ ┌───────▼──────┐
│ Single Env │ OR │ Vectorized │
│ (n_envs=1) │ │ Env (n=2-16) │
└────┬────────┘ └───────┬──────┘
│ │
└───────────┬───────────┘
│
┌────────▼────────┐
│ SARBaseEnv │
│ (Gymnasium) │
└────────┬────────┘
│
┌───────────┴───────────┐
│ │
┌────▼────────┐ ┌───────▼──────┐
│ Logic │ │ Observation │
│ Backend │◄─────┤ Builder │
│ (Physics) │ │ (68/554 dims)│
└────┬────────┘ └──────────────┘
│
├──► Coverage Map (grid tracking)
├──► Fleet Manager (constraints)
├──► Map Generator (procedural)
├──► Reward Function (14 components)
└──► Sensors (FOV + Comm)
Data Flow (Single Training Step)
text

1. OBSERVATION:
   ObservationBuilder.build_all()
   ├─► Actor obs (68-dim per UAV): self + stations + teammates + obstacles + victims + coverage
   └─► Critic obs (554-dim global): 8×UAV states + fleet stats

2. ACTION SELECTION:
   Actor.get_action(obs[68]) → Gaussian sample → action[3] ∈ [-1,1]
3. ENVIRONMENT STEP:
   LogicBackend.apply_actions() → update UAV positions
   LogicBackend.step_physics() → battery drain/charge
   LogicBackend.step_world() → victim detection, coverage marking
4. REWARD COMPUTATION:
   BaselineReward.compute() → 14 components → shared reward
5. BUFFER STORAGE:
   RolloutBuffer.add(obs, global_obs, action, reward, value, log_prob, done)
6. PPO UPDATE (after rollout_length steps):
   RolloutBuffer.compute_gae() → advantages, returns
   For epoch in n_epochs:
   For batch in minibatches:
   Actor: PPO clip loss + entropy
   Critic: MSE loss (value prediction)
   📁 DETAILED FILE STRUCTURE
   Directory Tree (56 files total)
   text

uav*swarm_pybullet/
├── config/ # Configuration system (9 files)
│ ├── **init**.py # Exports all configs
│ ├── config.py # AppConfig (master orchestrator)
│ ├── env.py # EnvConfig (map, time, fleet)
│ ├── uav.py # UAVConfig (physics, battery)
│ ├── sensor.py # SensorConfig (FOV, comm, noise)
│ ├── entity.py # VictimConfig, ObstacleConfig, DangerZoneConfig
│ ├── reward.py # RewardConfig (14 components)
│ ├── obs.py # ObsConfig (68/554 dims)
│ ├── train.py # TrainConfig (RL hyperparams)
│ └── curriculum_config.py # StageConfig (3 curriculum stages)
│
├── utils/ # Utilities (2 files)
│ ├── geometry.py # 9 vectorized geometry functions
│ └── logger.py # EpisodeLogger, TrainingLogger
│
├── entities/ # Game objects (4 files)
│ ├── uav.py # UAV class + UAVState enum
│ ├── victim.py # BaseVictim, InjuredVictim, MobileVictim
│ ├── charging_station.py # ChargingStation
│ └── obstacle.py # Debris, DangerZone (3 shapes: circle/rect/polygon)
│
├── core/ # Core systems (3 files)
│ ├── coverage_map.py # CoverageMap v2.0 (grid + temporal tracking)
│ ├── map_generator.py # MapGenerator v4.1 (procedural generation)
│ └── fleet_manager.py # FleetManager v2.0 (constraint enforcer)
│
├── sensors/ # Sensor models (2 files)
│ ├── fov_sensor.py # FOVSensor (detection with noise pipeline)
│ └── comm_sensor.py # CommSensor (V2V communication)
│
├── observation/ # Observation builder (1 file)
│ └── obs_builder.py # ObservationBuilder + ObsResult
│
├── rewards/ # Reward functions (1 file)
│ └── baseline_reward.py # BaselineReward v3.1 (hand-crafted, 14 components)
│
├── env_setup/ # Environments (4 files)
│ ├── base_env.py # SARBaseEnv (Gymnasium interface)
│ ├── sar_pettingzoo_env.py # SARPettingZooEnv (PettingZoo wrapper)
│ ├── vec_env.py # VectorizedEnv (parallel training, n_envs=1-16)
│ └── backends/
│ ├── base_backend.py # BaseBackend (ABC)
│ └── logic_backend.py # LogicBackend (pure Python physics)
│
├── visualization/ # Renderers (3 files)
│ ├── renderer_factory.py # Factory pattern
│ ├── visualizer2d.py # Visualizer2D (Matplotlib, ~50ms/frame)
│ └── visualizer3d.py # Visualizer3D (Matplotlib 3D, ~400ms/frame)
│
├── training/ # Training pipeline (10 files)
│ ├── curriculum.py # CurriculumManager (stage progression)
│ ├── curriculum_trainer.py # CurriculumTrainer (random policy baseline)
│ └── algorithms/
│ └── mappo/ # MAPPO implementation (6 files)
│ ├── **init**.py # Exports
│ ├── networks.py # MLP foundation + orthogonal init
│ ├── actor.py # ActorNetwork (Gaussian policy, 68→3)
│ ├── critic.py # CriticNetwork (centralized, 554→1)
│ ├── buffer.py # RolloutBuffer + GAE computation
│ └── trainer.py # MAPPOTrainer (main training loop)
│
├── tests/ # Unit tests (26 tests, 100% pass)
│ ├── test*\*.py # Various component tests
│ └── ...
│
├── examples/ # Example scripts (2 files)
│ ├── test_pettingzoo.py # Minimal PettingZoo API test
│ └── record_video.py # Video recording (MP4/GIF)
│
├── results/ # Auto-generated training outputs
│ └── mappo/
│ └── {run_name}/
│ ├── checkpoints/ # .pt checkpoint files
│ ├── viz/ # PNG snapshots
│ └── plots/ # Training curves
│
├── train_mappo.py # ✅ Main entry point (CLI)
├── test_trainer_smoke.py # ✅ Smoke test
└── README.md # Project documentation
📄 DETAILED FILE DOCUMENTATION
📁 config/ — Configuration System
config/**init**.py
Purpose: Central export point cho tất cả config classes

Exports:

AppConfig — Master config
EnvConfig, UAVConfig, SensorConfig — Environment configs
VictimConfig, ObstacleConfig, DangerZoneConfig — Entity configs
RewardConfig, ObsConfig, TrainConfig — RL configs
StageConfig, STAGE_EASY, STAGE_MEDIUM, STAGE_HARD, CURRICULUM_STAGES — Curriculum
Usage: from config import AppConfig, STAGE_EASY

config/config.py — AppConfig
Purpose: Master config orchestrator — single source of truth cho toàn bộ hệ thống

Attributes:

env: EnvConfig — Map size, timestep, fleet size
uav: UAVConfig — Physics, battery model
sensor: SensorConfig — FOV, comm range, noise params
victim: VictimConfig — Victim spawning rules
obstacle: ObstacleConfig — Debris generation
danger: DangerZoneConfig — Danger zone types & penalties
reward: RewardConfig — 14 reward components
obs: ObsConfig — Observation dimensions (68/554)
train: TrainConfig — RL hyperparameters (MAPPO/MASAC/MATD3)
viz_mode: str — "2d" / "3d" / "none"
viz_3d_cfg: dict — 3D renderer config
Methods:

**post_init**() — Auto-sync obs.n_stations = env.n_stations, validate consistency
apply_stage(stage: StageConfig) — Apply curriculum stage (modify map_size, n_victims, max_steps, etc. in-place)
map_diagonal (property) — sqrt(2) × map_size (diagonal distance)
grid_cell_size (property) — map_size / grid_size (meters per cell)
save(path) — Serialize toàn bộ config sang JSON
load(path) — Restore từ JSON file
Design Pattern: Dataclass-based composition, immutable during training (stage change tạo mới)

config/env.py — EnvConfig
Purpose: Environment parameters (map, time, fleet)

Key Attributes:

map_size: int = 100 — Map size in meters (square map)
grid_size: int = 100 — Coverage grid resolution (always sync with map_size)
dt_seconds: float = 1.0 — Simulation timestep (seconds)
max_steps: int = 600 — Max steps per episode (timeout)
n_uav: int = 4 — Number of UAVs in swarm
n_stations: int = 2 — Number of charging stations
charge_radius_m: float = 3.0 — Charging activation radius (meters)
station_capacity: int = 2 — Max UAVs per station simultaneously
min_station_spacing_m: float = 15.0 — Min distance between stations
deterministic_eval: bool = False — Fixed seed during evaluation
eval_seed: int = 42 — Seed for deterministic eval
Placement Constraints: (for map generation)

max_place_attempts: int = 500 — Max tries to place objects
min_object_spacing_m: float = 2.5 — Min distance between objects
victim_clearance_m: float = 1.5 — Clearance around victims
placement_relax_threshold: float = 0.7 — After 70% attempts → relax spacing
placement_relaxed_spacing_m: float = 1.5 — Relaxed spacing value
allow_partial_obstacles: bool = True — Skip instead of crash when placement fails
Backward Compatibility: Properties dt, charge_radius, min_station_spacing (alias cho \_m/\_seconds)

config/uav.py — UAVConfig
Purpose: UAV physics model + battery dynamics

Physics Parameters:

z_min_m: float = 3.0 — Minimum altitude (meters)
z_max_m: float = 40.0 — Maximum altitude (meters)
max_speed_xy_mps: float = 5.0 — Max horizontal speed (m/s)
max_speed_z_mps: float = 2.0 — Max vertical speed (m/s)
collision_radius_m: float = 0.5 — Collision detection radius
Battery Model:

drain_xy_pct_per_s: float = 0.10 — Horizontal movement drain (%/s)
drain_z_up_pct_per_s: float = 0.15 — Climbing drain (%/s)
drain_z_down_pct_per_s: float = 0.03 — Descending drain (%/s)
drain_idle_pct_per_s: float = 0.05 — Hovering drain (%/s)
charge_rate_pct_per_s: float = 1.5 — Charging rate (%/s)
Battery Thresholds:

battery_return_pct: float = 10.0 — Auto-return threshold (%)
battery_ready_pct: float = 80.0 — Ready to deploy threshold (%)
battery_dead_pct: float = 0.0 — Dead → DISABLED state
battery_warning_pct: float = 20.0 — Warning level
battery_critical_pct: float = 10.0 — Critical level
battery_emergency_pct: float = 5.0 — Emergency level (forced RETURNING)
Fleet Management:

reserve_ratio: float = 0.2 — 20% of swarm kept in reserve
min_reserve: int = 2 — Minimum 2 UAVs in reserve at all times
Backward Compatibility: Properties z_min, z_max, drain_xy_max, battery_dead, etc.

config/sensor.py — SensorConfig
Purpose: Sensor models (FOV geometry + detection noise)

FOV Parameters:

comm_range_m: float = 30.0 — V2V communication range (meters)
hfov_deg: float = 90.0 — Horizontal field-of-view angle (degrees)
Detection Noise Model:

p_detect_base: float = 0.95 — Base detection probability (at altitude=0)
p_detect_decay: float = 0.04 — Decay rate with altitude
enable_noise: bool = True — Enable stochastic detection
motion_blur_coeff: float = 0.06 — Penalty coefficient for high speed
base_miss_rate: float = 0.03 — Hardware false negative rate
Computed Properties:

fov_tan — tan(hfov_deg / 2) in radians
fov_radius_at_altitude(altitude) — FOV radius at given altitude
comm_range — Alias for comm_range_m
Noise Pipeline: P_final = P_altitude × env_factor × (1 - motion_penalty) × victim_factor × (1 - base_miss_rate)

config/entity.py — Entity Configs
VictimConfig
Purpose: Victim spawning parameters

Attributes:

n_victims_min: int = 5 — Min victims per episode
n_victims_max: int = 20 — Max victims per episode
injured_ratio_min: float = 0.4 — Min ratio of injured (stationary)
injured_ratio_max: float = 0.7 — Max ratio of injured
injured_urgency_min: float = 4.0 — Urgency range for injured [4.0, 5.0]
injured_urgency_max: float = 5.0
mobile_urgency_min: float = 1.0 — Urgency range for mobile [1.0, 3.0]
mobile_urgency_max: float = 3.0
mobile_speed_min_mps: float = 0.2 — Mobile victim speed (m/s)
mobile_speed_max_mps: float = 0.4
mobile_dir_change_steps: int = 20 — Direction change interval (steps)
ObstacleConfig
Purpose: Debris and danger zone generation

Attributes:

n_debris: int = 6 — Number of debris objects
debris_width_min_m: float = 2.0 — Debris footprint diameter (meters)
debris_width_max_m: float = 5.0
debris_height_min_m: float = 3.0 — 3D height (meters)
debris_height_max_m: float = 8.0
n_danger_total: int = 2 — Total danger zones per episode
DangerZoneConfig
Purpose: Danger zone types, heights, penalties

Attributes (Dicts):

heights: Dict[str, float] — Max height per type
gas: 3m, fire: 15m, smoke: 25m, collapse: 10m, radiation: inf
penalties: Dict[str, float] — Per-step penalty per type
gas: -3, fire: -3, smoke: -1.5, collapse: -1, radiation: -5
max_counts: Dict[str, int] — Max count per type in one episode
widths: Dict[str, Tuple[float, float]] — (min_diameter, max_diameter) per type
Important: Widths = diameters, NOT radii (generator converts radius = width / 2)
Methods:

validate() — Check consistency (all types have entries in all 4 dicts)
danger_types (property) — List of danger types (sorted)
config/reward.py — RewardConfig v3.1
Purpose: 14-component reward function (hand-crafted baseline for Paper 2)

Reward Components:

Component Default Description
r_coverage_delta +6.0 Per 1% global coverage increase
r_victim_base +50.0 Base × (urgency/5) when victim found
r_battery_20 0.0 Penalty <20% (disabled)
r_battery_10 -1.0 Penalty <10%
r_battery_5 -3.0 Penalty <5% (critical)
r_battery_dead -100.0 One-time when battery=0 → DISABLED
r_collision_obstacle -30.0 One-time collision with debris
r_proximity_1m -10.0 Per step when 2 UAVs within 1m
r_proximity_2m -3.0 Per step within 2m
r_proximity_3m -0.5 Per step within 3m
proximity_penalty_cap -15.0 Cap total proximity penalty per step
r_time_penalty -0.05 Per active UAV per step (efficiency)
r_terminal_base +200.0 Terminal bonus base
terminal_bonus_cap +100.0 Max terminal bonus
Caps & Clipping:

step_penalty_cap: float = -30.0 — Cap total negative penalties per step
step_reward_clip_min: float = -100.0 — Clip min per step
step_reward_clip_max: float = +100.0 — Clip max per step
Advanced Features:

enable_distance_shaping: bool = True — Delta-based distance shaping to nearest unfound victim
distance_shaping_max_per_uav: float = 1.0 — Cap shaping reward per UAV
Design Principles:

Cooperative shared reward (all agents receive same value)
Positive incentives > negative penalties (for RL gradient)
Terminal bonus rewards mission completion (70% coverage + 20% victims + 10% time)
config/obs.py — Observation Configs
ObsSchemaConfig
Purpose: Define observation slot dimensions (constants)

Constants:

SELF_FEATURES = 11 — Self state (pos, vel, battery, state_onehot)
STATION_FEATURES_PER = 4 — Per charging station (rel_x, rel_y, dist, occupancy)
TEAMMATE_FEATURES_PER = 3 — Per teammate (dist, bearing, rel_alt)
OBSTACLE_FEATURES_PER = 3 — Per obstacle (rel_x, rel_y, type_id)
VICTIM_FEATURES_PER = 5 — Per victim (rel_x, rel_y, urgency, dist, found)
COVERAGE_FEATURES = 3 — Local coverage (small_radius, large_radius, time_remaining)
GLOBAL_FEATURES = 10 — Fleet stats (n_active, n_charging, battery stats, coverage, etc.)
ObsConfig
Purpose: Runtime observation configuration

Attributes:

n_obs_victims: int = 5 — Max victims in observation (top-5 nearest)
n_obs_obstacles: int = 4 — Max obstacles in observation (top-4 nearest)
n_tracked_teammates: int = 3 — Max teammates tracked (top-3 nearest)
local_cov_small: int = 15 — Small radius for local coverage (meters)
local_cov_large: int = 30 — Large radius for local coverage (meters)
max_uav: int = 8 — Padding size for critic observation (zero-pad up to 8 UAVs)
n_stations: int = None — Auto-synced from EnvConfig.n_stations in AppConfig.**post_init**
Computed Properties (Dimensions):

self_dim = 11
station_dim = n_stations × 4 → 8 (with n_stations=2)
team_dim = n_tracked_teammates × 3 = 9
obstacle_dim = n_obs_obstacles × 3 = 12
victim_dim = n_obs_victims × 5 = 25
coverage_dim = 3
actor_dim = 68 (total actor observation)
global_dim = 10
critic_dim = 554 (8×68 + 10 global features)
Method:

validate() — Ensure n_stations is not None (must be synced)
config/train.py — TrainConfig
Purpose: RL training hyperparameters (multi-algorithm support)

General Training Params:

n_seeds: int = 5 — Number of random seeds for statistical robustness
seeds: List[int] = [42, 123, 456, 789, 1011] — Fixed seeds
confidence_level: float = 0.95 — For statistical tests (Wilcoxon, t-test)
total_episodes: int = 3000 — Total episodes per curriculum stage
eval_interval: int = 50 — Evaluate policy every N episodes
save_interval: int = 100 — Save checkpoint every N episodes
log_window: int = 100 — Rolling window for metrics (mean/std)
MAPPO Hyperparameters (17 params):

Parameter Default Description
mappo_rollout_length 2048 Steps collected per PPO update
mappo_n_epochs 10 Epochs per PPO update
mappo_batch_size 256 Minibatch size
mappo_clip_epsilon 0.2 PPO clip range ε
mappo_gamma 0.99 Discount factor γ
mappo_gae_lambda 0.95 GAE λ parameter
mappo_lr_actor 3e-4 Actor learning rate
mappo_lr_critic 1e-3 Critic learning rate
mappo_max_grad_norm 0.5 Gradient clipping threshold
mappo_entropy_coeff 0.01 Entropy bonus coefficient
mappo_actor_hidden (256, 256) Actor MLP hidden layers
mappo_critic_hidden (512, 256) Critic MLP hidden layers
mappo_activation 'tanh' Activation function
mappo_use_layer_norm False Layer normalization
mappo_log_interval 10 Console log every N updates
mappo_viz_interval 100 2D viz snapshot every N updates
mappo_checkpoint_interval 100 Checkpoint save every N updates
Future: MASAC/MATD3 hyperparams sẽ thêm vào file này

config/curriculum_config.py — Curriculum Stages
StageConfig
Purpose: Define một curriculum stage

Attributes:

name: str — "easy" / "medium" / "hard"
map_size: int — Map size (meters)
n_uav: int = 4 — Number of UAVs (constant)
n_victims_min: int — Min victims
n_victims_max: int — Max victims
n_debris: int — Number of debris
n_danger_total: int — Total danger zones
station_capacity: int = 2 — Charging station capacity
max_steps: int — Max steps per episode
min_episodes: int — Min episodes before advance
advance_coverage: float — Coverage threshold to advance (e.g., 0.70 = 70%)
advance_victims: float — Victim rate threshold to advance (e.g., 0.80 = 80%)
Computed Properties (for Paper Metrics):

map_area_m2 — map_size²
coverage_pressure_m2_per_uav — area / n_uav (key difficulty metric)
victim_density_per_1000m2 — Victim density (≈0.53, constant across stages)
obstacle_density_per_1000m2 — Obstacle density
steps_per_m2 — Time budget per square meter
describe() — Human-readable summary string
Predefined Stages
3 curriculum instances:

Python

STAGE_EASY = StageConfig(
name="easy",
map_size=150,
n_victims_min=10, n_victims_max=15,
n_debris=8, n_danger_total=3,
max_steps=300,
min_episodes=100,
advance_coverage=0.70, advance_victims=0.80
)

STAGE_MEDIUM = StageConfig(
name="medium",
map_size=200,
n_victims_min=18, n_victims_max=24,
n_debris=12, n_danger_total=5,
max_steps=350,
min_episodes=100,
advance_coverage=0.65, advance_victims=0.75
)

STAGE_HARD = StageConfig(
name="hard",
map_size=250,
n_victims_min=28, n_victims_max=36,
n_debris=15, n_danger_total=7,
max_steps=400,
min_episodes=100,
advance_coverage=0.60, advance_victims=0.70
)

CURRICULUM_STAGES = [STAGE_EASY, STAGE_MEDIUM, STAGE_HARD]
Difficulty Progression:

Stage Map Area/UAV Victims Debris Danger Steps Threshold
EASY 150m 5,625 m² 10-15 8 3 300 70% cov, 80% vic
MEDIUM 200m 10,000 m² 18-24 12 5 350 65% cov, 75% vic
HARD 250m 15,625 m² 28-36 15 7 400 60% cov, 70% vic
Validation: \_verify_stages() runs on import — checks victim density consistency

📁 utils/ — Utility Functions
utils/geometry.py
Purpose: 9 vectorized geometry functions (NumPy-optimized)

Functions:

dist_2d(pos1, pos2) → float

2D Euclidean distance (XY plane)
Input: pos1[x,y,...], pos2[x,y,...]
Returns: sqrt((x2-x1)² + (y2-y1)²)
dist_3d(pos1, pos2) → float

3D Euclidean distance (XYZ)
Returns: sqrt((x2-x1)² + (y2-y1)² + (z2-z1)²)
normalize_angle(angle) → float

Normalize angle to [-π, π]
Input: angle in radians
Returns: wrapped angle
compute_bearing(from_pos, from_vel, to_pos) → float

Compute bearing from current heading to target
Input: from_pos[x,y], from_vel[vx,vy], to_pos[x,y]
Returns: angle difference in radians ∈ [-π, π]
Used in observation (teammates)
check_los_2d(pos1, pos2, obstacles) → bool

Line-of-sight check in 2D
Checks if line segment intersects any obstacle footprint
Supports Shapely polygon intersection (fast) or fallback circle check
Returns: True if clear, False if blocked
get_circle_cells(center, radius, grid_size, map_size) → ndarray(N, 2)

Get grid cells within circle (for FOV coverage marking)
10× faster than loop version (vectorized)
Input: center[x,y], radius (meters), grid_size (cells), map_size (meters)
Returns: array of [row, col] indices
get_relative_position(from_pos, to_pos) → ndarray

Compute relative position vector
Returns: [dx, dy, dz] = to_pos - from_pos
clip_position(pos, min_bounds, max_bounds) → ndarray

Clip position to boundary box
Returns: pos clamped element-wise
get_circle_cells_legacy(...) → ndarray

Legacy loop version (kept for benchmarking)
~10× slower than vectorized version
Performance: Geometry operations are called millions of times per training → vectorization critical

utils/logger.py
Purpose: Episode and training logging utilities

EpisodeLogger
Purpose: Log metrics for một episode

Attributes:

episode_num: int — Episode number
step_count: int = 0 — Current step in episode
reward_sum: float = 0.0 — Cumulative reward
coverage_max: float = 0.0 — Max coverage achieved
total_victims: int = 0 — Total victims in episode
victims_found: int = 0 — Victims found so far
events: Dict[str, int] — Event counters
Methods:

log_step(rewards, coverage) — Update reward_sum và coverage_max
log_event(event_type: str) — Increment event counter
Event types: "collision_obstacle", "victim_found", "battery_death", "danger_zone", "hot_swap"
set_total_victims(n: int) — Set total victims count
finalize() → Dict — Return JSON-safe metrics dict
Keys: episode, steps, reward, coverage_rate (%), victims_found, victims_total, victim_rate, success, events
TrainingLogger
Purpose: Aggregate metrics across nhiều episodes

Attributes:

window_size: int = 100 — Rolling window size
episodes: List[int] — Episode numbers
rewards: List[float] — Episode rewards
coverages: List[float] — Episode coverage rates
victims: List[float] — Episode victim rates
successes: List[bool] — Episode success flags
convergence_threshold: float = 0.95 — Convergence detection threshold
convergence_window: int = 50 — Window for convergence check
verbose: bool = True — Print logs?
Methods:

log_episode(metrics: Dict) — Add episode metrics
Auto-detect convergence (if last 50 eps > threshold)
Print summary if verbose
get_stats(last_n: int = 100) → Dict — Compute stats (mean/std/min/max/success_rate/converged)
save(filepath) — Save to JSON
load(filepath) — Load from JSON (static method)
Helper Function:

compare_training_runs(runs: List[TrainingLogger], labels: List[str]) — Compare multiple runs (for Paper 1)
📁 entities/ — Game Objects
entities/uav.py
UAVState (Enum)
Purpose: 5 UAV states for state machine

States:

ACTIVE — Normal operation (RL controls)
RETURNING — Auto-flying to charging station (low battery)
CHARGING — Docked at station, charging
DEPLOYING — Auto-flying from station to mission area (after charge)
DISABLED — Battery dead, out of mission
Transitions:

CHARGING → DEPLOYING (when battery ≥ 80%)
DEPLOYING → ACTIVE (when reached target position)
ACTIVE → RETURNING (when battery ≤ 10% or manual command)
RETURNING → CHARGING (when reached station)
→ DISABLED (when battery = 0%)
UAV Class
Purpose: Quadcopter agent với physics và battery model

Attributes (State):

id: int — UAV ID (0-based)
pos: ndarray — Position [x, y, z] (meters)
vel: ndarray — Velocity [vx, vy, vz] (m/s)
battery: float — Battery percentage [0, 100]
state: UAVState — Current state
target_station: ChargingStation | None — Target station (when RETURNING/CHARGING)
victims_found: int — Counter
battery_death: bool — Flag (one-time penalty tracking)
Attributes (Config, read-only):

cfg: AppConfig — Shared config reference
uav_cfg, env_cfg, sensor_cfg — Convenient aliases
Tracking Attributes:

steps_alive: int — Steps since spawn
distance_xy: float — Total XY distance traveled
distance_3d: float — Total 3D distance traveled
Methods:

apply_action(action: ndarray) → None

Apply RL action (only when state=ACTIVE)
Input: action [ax, ay, az] ∈ [-1, 1]³
Process: Scale to velocity → clip to max_speed → update position → boundary clamp
Updates: pos, vel, tracking counters
auto_navigate(target_pos: ndarray) → None

Automatic navigation to target (used in RETURNING/DEPLOYING)
Uses proportional controller (no overshoot)
Slows down near target
Updates: pos, vel
update_battery(stations: List[ChargingStation]) → None

Update battery based on state
ACTIVE/RETURNING/DEPLOYING: Drain based on velocity
CHARGING: Charge if in station range
Drain formula: drain = base × sqrt(vx² + vy² + vz²) × dt
Clamp battery to [0, 100]
\_do_drain() → None

Internal: Compute battery drain proportional to velocity
Different rates for XY vs Z_up vs Z_down vs idle
\_do_charge(stations) → None

Internal: Charge battery if in target_station range
Rate: charge_rate_pct_per_s × dt_seconds
get_fov_radius() → float

Compute FOV radius at current altitude
Formula: altitude × tan(hfov / 2)
get_state_onehot() → ndarray(5,)

Convert state to one-hot encoding [ACTIVE, RETURNING, CHARGING, DEPLOYING, DISABLED]
set_state(new_state: UAVState) → None

Change state với validation
Clear target_station khi không cần (ACTIVE/DISABLED)
needs_charging() → bool

Check if battery ≤ battery_return_pct
is_ready_to_deploy() → bool

Check if battery ≥ battery_ready_pct AND state=CHARGING
find_nearest_station(stations) → ChargingStation | None

Find nearest available station
Returns: Station with free slot, sorted by distance
to_dict() → Dict

Serialize to JSON-safe dict (for logging/debugging)
Properties:

battery_pct — Alias for battery attribute
Design Notes:

RL agent ONLY controls velocity when state=ACTIVE
State transitions enforced by FleetManager.enforce_safety_constraints()
Battery model: Linear drain proportional to velocity magnitude
entities/victim.py
BaseVictim (Abstract Base Class)
Purpose: Base class cho 2 victim types (injured/mobile)

Attributes:

id: int — Victim ID
pos: ndarray — Position [x, y, 0] (always on ground)
urgency: float — Urgency score [1.0, 5.0]
is_found: bool = False — Detection flag
found_at_step: int = -1 — Step number when found
found_by_uav: int = -1 — UAV ID that found victim
Methods:

step(obstacles) → None

Abstract method: Update physics (mobile movement)
Must be implemented by subclasses
update(step_count, obstacles) → None

Alias for step(obstacles) (for backend compatibility)
mark_found(step: int, uav_id: int) → None

Mark victim as found
Set flags: is_found=True, found_at_step=step, found_by_uav=uav_id
Call hook: \_on_found() (subclass-specific behavior)
get_reward_value() → float

Compute reward when found
Formula: r_victim_base × (urgency / 5.0)
Higher urgency → higher reward
\_on_found() → None

Hook method: Called when victim found
Override in subclasses (e.g., mobile stops moving)
Legacy: is_detected_by(...) → bool

Old detection method (replaced by FOVSensor)
InjuredVictim
Purpose: Stationary victim (không di chuyển)

Characteristics:

urgency ∈ [4.0, 5.0] — High urgency
speed = 0.0 — No movement
step() — Empty (no physics update)
\_on_found() — No special behavior
MobileVictim
Purpose: Moving victim (random walk)

Additional Attributes:

speed: float — Movement speed [0.2, 0.4] m/s
direction: ndarray — Current direction vector (unit vector)
dir_change_interval: int = 20 — Steps between direction changes
\_steps_since_dir_change: int = 0 — Counter
Characteristics:

urgency ∈ [1.0, 3.0] — Lower urgency (can wait)
step(obstacles):
If found: Freeze (speed=0, no movement)
Else: Random walk
Change direction every 20 steps
Update position: pos += direction × speed × dt
Boundary clipping
Obstacle collision check (bounce back if hit)
\_on_found():
Freeze in place: speed = 0.0
Rationale: Emergency responders immobilize victim
Design Notes:

Mobile victims add dynamic challenge (moving targets)
Freeze behavior prevents "lost again" after detection
Obstacle collision: Simple bounce-back (no physics simulation)
entities/charging_station.py
ChargingStation
Purpose: Battery recharge station

Attributes:

id: int — Station ID
pos: ndarray — Position [x, y, 0] (on ground)
capacity: int — Max UAVs simultaneously (from config)
charge_radius: float — Activation radius (meters)
charge_rate: float — Charge rate (%/step)
current_occupants: List[UAV] — UAVs currently charging
occupant_ids: Set[int] — UAV IDs (for O(1) lookup)
Methods:

is_full() → bool

Check if capacity reached
is_available() → bool

Check if has free slot
in_range(uav_pos: ndarray) → bool

Check if UAV is within charging range
Conditions: dist_xy ≤ charge_radius AND z ≤ 0.5m (near ground)
try_occupy(uav: UAV) → bool

Try to occupy a slot
Returns: True if success, False if full or already occupied
release(uav: UAV) → bool

Release slot
Returns: True if was occupant, False otherwise
charge(uav: UAV) → None

Charge one step
Check range → try_occupy → add charge rate to battery
Formula: battery += charge_rate × dt
force_release_all() → None

Clear all occupants (called on episode reset)
get_occupancy_ratio() → float

Return [0.0, 1.0] occupancy ratio
Design Notes:

Concurrency control: capacity limit prevents over-assignment
Spatial constraint: Must be in range AND near ground
Auto-release when UAV leaves range (handled by UAV.update_battery)
entities/obstacle.py
Debris
Purpose: Static obstacle (building debris, rubble)

Attributes:

id: int — Debris ID
pos: ndarray — Center position [x, y, 0]
height_3d: float — 3D height (meters)
shape: str — "circle" / "rectangle" / "polygon"
Shape-specific:
Circle: radius: float
Rectangle: width, height_2d: float, rotation: float (radians)
Polygon: vertices: ndarray(N, 2)
polygon: shapely.Polygon | None — Shapely object (if available)
penalty: float — Collision penalty (from reward config)
Methods:

in_zone_2d(pos_2d: ndarray) → bool

Check if 2D position inside footprint
Circle: dist ≤ radius
Rectangle: Rotate point, check AABB
Polygon: Shapely contains check (fallback: bounding circle)
causes_collision(uav_pos: ndarray) → bool

Check if UAV collides
Condition: in_zone_2d(pos_xy) AND uav.z < height_3d
blocks_los(pos1, pos2) → bool

Check if line segment intersects footprint
Uses Shapely if available, else fallback circle check
get_distance_to_edge(pos_2d) → float

Compute distance from point to edge (used in shaping)
Shapely: polygon.exterior.distance(point)
Fallback: dist - radius (for circle)
\_get_fallback_radius() → float

Bounding circle radius (when Shapely unavailable)
Circle: radius
Rectangle: sqrt(w² + h²) / 2
Polygon: Max distance from center
Shape Distribution (Map Generator):

40% circle
40% rectangle
20% polygon (3-6 vertices)
DangerZone
Purpose: Hazardous area (fire/smoke/gas/radiation/collapse)

Inherits: Same structure as Debris

Additional Attributes:

danger_type: str — "fire" / "smoke" / "gas" / "radiation" / "collapse"
max_height: float — Danger zone height (from config)
penalty: float — Per-step penalty (from config)
Methods (overridden):

is_inside(uav_pos: ndarray) → bool

Renamed from causes_collision (semantic clarity)
Condition: in_zone_2d(pos_xy) AND uav.z < max_height
blocks_los(pos1, pos2) → bool

ONLY fire and smoke block line-of-sight
Others (gas/radiation/collapse): transparent
get_sensor_modifier() → float

Sensor degradation factor [0.4, 1.0]
Smoke: 0.40 (worst)
Fire: 0.55
Collapse: 0.70
Gas: 0.85
Radiation: 0.95 (least impact)
get_battery_modifier() → float

Battery drain modifier
Fire: 0.05 (additional 5% drain)
Others: 0.0
Design Notes:

3 shapes support realistic disaster geometry
Height-based collision (3D awareness)
Shapely optional (graceful fallback)
Danger zones add strategic complexity (avoid vs shortcut trade-off)
📁 core/ — Core Systems
core/coverage_map.py — CoverageMap v2.0
Purpose: Grid-based coverage tracking với temporal information

Attributes:

grid_size: int — Grid resolution (cells)
map_size: float — Map size (meters)
grid: ndarray(bool, [GS, GS]) — Explored cells
timestamps: ndarray(int32, [GS, GS]) — Last scan step per cell
first_scan: ndarray(int32, [GS, GS]) — First scan step (-1 if never)
scan_count: ndarray(int32, [GS, GS]) — Number of times scanned
Methods:

reset() → None

Reset all arrays to initial state
grid = False, timestamps = 0, first_scan = -1, scan_count = 0
mark_explored(uav_pos: ndarray, fov_radius: float, step: int) → None

Mark cells within FOV as explored
Vectorized: Use get_circle_cells() (10× faster than loop)
Update: grid[cells] = True, timestamps[cells] = step (only if newer), first_scan[cells] (if first time), scan_count[cells] += 1
Boundary check: Clip cells to grid bounds
get_coverage_rate() → float

Return coverage ratio [0.0, 1.0]
Formula: np.sum(grid) / grid.size
get_coverage_percent() → float

Return coverage percentage [0.0, 100.0]
Formula: get_coverage_rate() × 100
get_local_coverage(pos: ndarray, radius: float) → float

Compute coverage within radius around position
Get cells in circle → count explored → ratio
get_staleness(pos: ndarray, radius: float, step: int) → float

Compute average "staleness" (age) of cells in radius
Staleness = step - timestamps[cell] (unexplored = max_steps)
Returns: Mean staleness in steps
get_staleness_normalized(pos, radius, step, decay_threshold=200) → float

Normalized staleness [0.0, 1.0]
Formula: staleness / decay_threshold (clamped to [0,1])
get_freshness(pos, radius, step, decay=200) → float

Inverse of staleness: 1.0 - get_staleness_normalized(...)
get_coverage_with_decay(step: int, decay_threshold: int = 200) → float

Coverage considering only "fresh" cells (scanned within threshold)
Filter: timestamps ≥ step - decay_threshold
get_rescan_count(pos, radius) → float

Average number of times cells were scanned in radius
Useful for detecting redundant scanning
get_nearest_unexplored(pos: ndarray, min_distance: float = 0.0) → ndarray | None

Find nearest unexplored cell (grid=False)
O(N) scan — bottleneck for large grids
Filter: Cells with distance ≥ min_distance
Returns: [x, y] position or None
get_nearest_stale(pos, step, threshold=100) → ndarray | None

Find nearest "stale" cell (not scanned recently)
Filter: step - timestamps[cell] > threshold
Returns: [x, y] position or None
get_stats(step: int) → Dict

Return comprehensive statistics
Keys: coverage_rate, total_cells, explored_cells, avg_staleness, max_staleness, avg_scan_count, cells_never_scanned, cells_with_decay
get_grid_snapshot() → Dict

Export raw arrays for visualization
Keys: grid, timestamps, first_scan, scan_count, grid_size, map_size
Performance Notes:

Vectorized operations: O(1) for mark*explored with NumPy
get_nearest*\*: O(N) — OK for 100×100, bottleneck at 500×500 (consider KD-tree)
Temporal tracking enables advanced strategies (revisit stale areas)
core/map_generator.py — MapGenerator v4.1
Purpose: Procedural map generation với placement constraints

Key Fix v4.1: Config widths = diameters (not radii). Generator converts: radius = width / 2.0

Attributes:

cfg: AppConfig — Config reference
env_cfg, victim_cfg, obstacle_cfg, danger_cfg — Shortcuts
Methods:

generate(n_victims_override: int | None = None, seed: int | None = None) → Dict

Main method: Generate full map
Steps:
Initialize RNG from seed
Determine victim count (override or sample from range)
Place stations (min spacing constraint)
Place debris (progressive relaxation if fail)
Place danger zones (avoid overlap)
Spawn victims (group spawning near debris)
Generate UAV spawns (around stations)
Returns: Dict with keys:
stations: List[Dict] — Station data
debris: List[Dict] — Debris data (shape, dimensions, etc.)
danger_zones: List[Dict] — Danger zone data
victims: List[Dict] — Victim data (type, pos, urgency)
uav_spawns: List[Dict] — UAV spawn positions
seed: int — Actual seed used
n_victims: int — Actual victim count
\_place_stations(rng) → List[Dict]

Place n_stations với min spacing constraint
Tries: max_place_attempts
Returns: [{id, pos:[x,y,0]}, ...]
Raises: RuntimeError if fail
\_place_debris(stations, rng) → List[Dict]

Place n_debris objects
Progressive relaxation: After 70% attempts → reduce min_spacing to relaxed value
Allow partial: If allow_partial_obstacles=True, skip failed placements instead of crash
Shape distribution:
40% circle: {shape: "circle", radius}
40% rectangle: {shape: "rectangle", width, height, rotation}
20% polygon: {shape: "polygon", vertices: [[x,y], ...]}
Constraints:
Min distance from stations (avoid blocking charge zone)
Min spacing between debris
Returns: List[Dict] — Debris definitions
\_place_danger_zones(existing_obstacles, rng) → List[Dict]

Place n_danger_total danger zones
Shape distribution:
50% circle
50% rectangle
Type selection: Random from danger_types, respect max_counts
Dimensions: Sample from widths[type] → convert to radius
Height & Penalty: From heights[type], penalties[type]
Constraints:
Min spacing from existing obstacles
Avoid stations
Returns: List[Dict]
\_spawn_victims(n, obstacles, danger_zones, rng) → List[Dict]

Spawn n victims
Ratio: Sample injured ratio ∈ [injured_ratio_min, injured_ratio_max]
Group spawning: 60% near debris (realistic disaster scenario), 40% random
Constraints:
Min clearance from obstacles/danger_zones
Not inside danger zones
Type: Injured (stationary) vs Mobile (random walk)
Urgency: Sample from type-specific ranges
Returns: [{type: "injured"/"mobile", pos, urgency, speed (if mobile)}, ...]
\_find_valid_victim_pos(obstacles, danger_zones, rng, near_pos=None, max_attempts=100) → ndarray | None

Find valid victim position
Near mode: If near_pos provided → try within 10m radius
Constraints:
victim_clearance_m from obstacles
Not inside danger zones
Returns: [x, y, 0] or None if fail
\_spawn_group(n, type, obstacles, danger_zones, debris_list, rng) → List[Dict]

Spawn n victims of type near debris
For each victim:
Pick random debris
Try spawn within 10m radius
Fallback: Random position if fail
Returns: Victim dicts
get_uav_spawns(stations, n_total, rng) → List[Dict]

Generate UAV spawn positions around stations
Distribution: Evenly distribute around stations
Position: Random angle, distance 1-3m from station, altitude 5-10m
Returns: [{id, pos:[x,y,z]}, ...]
get_map_statistics(map_data: Dict) → Dict

Compute map statistics for logging/paper
Keys:
victim_density_per_1000m2
obstacle_density_per_1000m2
danger_zone_density_per_1000m2
avg_victim_urgency
injured_ratio
victim_clustering_score (std dev of pairwise distances)
obstacle_coverage_ratio (area covered / total area)
Design Philosophy:

Procedural + constraints: Reproducible from seed, realistic layouts
Progressive relaxation: Ensure placement success (avoid deadlock)
Group spawning: Realistic disaster scenario (victims cluster near collapsed buildings)
Fail-safe: allow_partial_obstacles prevents training crash
core/fleet_manager.py — FleetManager v2.0
Purpose: Fleet constraint enforcer (NOT rule-based controller)

Design Philosophy:

ENFORCE: Safety constraints (battery emergency → RETURNING)
SUGGEST: Deployment/return suggestions (RL can ignore)
RL CONTROLS: Final action decisions
Attributes:

n_total: int — Total UAVs
n_reserve: int — Reserve count (20% of fleet)
all_uavs: List[UAV] — All UAVs reference
stations: List[ChargingStation] — Stations reference
\_battery_death_penalized: Set[int] — UAV IDs already penalized (one-time)
\_uav_return_locks: Dict[int, bool] — Hysteresis state for return decisions
Methods:

reset(uavs, stations) → None

Initialize/reset for new episode
Clear penalized set, reset locks
get_deployable_uavs() → List[UAV]

Get UAVs eligible for deployment
Filters: state=CHARGING AND battery ≥ battery_ready_pct
Sorted: Highest battery first (deploy strongest)
get_best_deployable(prefer_station: ChargingStation | None = None, min_battery: float = None) → UAV | None

Get single best UAV to deploy
Preference: UAVs from prefer_station (if specified)
Filter: battery ≥ min_battery (if specified)
Returns: Highest battery UAV or None
enforce_safety_constraints() → None

ENFORCE safety rules (non-negotiable)
Rules:
battery = 0 → state = DISABLED (one-time penalty tracking)
battery < battery_emergency_pct (5%) AND state = ACTIVE → state = RETURNING (emergency)
Called every step (before RL actions)
suggest_deployments(target_active: int) → List[int]

SUGGEST UAV IDs to deploy (RL can ignore)
Input: Desired active count
Current: Count ACTIVE + RETURNING + DEPLOYING
Gap: target_active - current
If gap > 0: Get gap best deployable UAVs
Returns: List of UAV IDs to deploy
suggest_returns() → List[int]

SUGGEST UAV IDs to return (RL can ignore)
Criteria: battery ≤ battery_return_pct (10%)
Returns: List of UAV IDs
step() → Dict

Main step method (called every step)
Process:
Enforce safety constraints
Suggest deployments (target = n_total - n_reserve)
Suggest returns
Returns: Dict with:
enforced_disabled: List[int] — UAVs forced to DISABLED
enforced_returning: List[int] — UAVs forced to RETURNING
suggested_deploy: List[int] — Suggested deployments
suggested_return: List[int] — Suggested returns
get_mission_priority_hints() → Dict

Strategic hints for RL policy
Keys:
operational_ratio — (n_active + n_returning) / n_total
reserve_health — Mean battery of reserve UAVs
station_pressure — Max occupancy ratio across stations
need_deploy — Bool (active < target)
get_spatial_awareness() → Dict

Fleet spatial statistics
Keys:
active_positions — List of active UAV positions
center_of_mass — Mean position
spread_radius — Std dev of distances from center
count_by_state() → Dict[UAVState, int]

Count UAVs per state
get_battery_stats() → Dict

Battery statistics
Keys:
mean, min, max, std — Battery stats
n_critical — Count battery < 10%
n_low — Count battery < 20%
n_emergency — Count battery < 5%
get_stats() → Dict

Comprehensive fleet stats (all above combined)
get_fleet_incentives() → float

Legacy method (backward compat)
Returns: 0.0 (incentives now in reward function)
is_episode_over() → bool

Check if all UAVs disabled (mission failure)
Hysteresis Logic:

\_uav_return_locks prevent oscillation (once RETURNING, stay until CHARGING)
Design Notes:

Separation of concerns: Constraint enforcement vs RL control
Suggestions ignorable → RL can learn better strategies
Emergency constraints non-negotiable (safety)
📁 sensors/ — Sensor Models
sensors/fov_sensor.py — FOVSensor
Purpose: Field-of-view detection với realistic noise model

Noise Pipeline (5 stages):

text

P_final = P_altitude × env_factor × (1 - motion_penalty) × victim_factor × (1 - base_miss_rate)
Attributes:

cfg: AppConfig — Config reference
sensor_cfg: SensorConfig — Shortcut
\_rng: np.random.Generator | None — RNG for reproducible eval
Methods:

set_seed(seed: int) → None

Set RNG seed for deterministic evaluation
Creates: self.\_rng = np.random.default_rng(seed)
calculate_fov_radius(altitude: float) → float

Compute FOV radius at altitude
Formula: altitude × tan(hfov / 2)
calculate_detection_prob(altitude, speed, env_modifier, victim_type) → float

Full noise pipeline:
Stage 1 — Altitude decay:
P_altitude = p_detect_base × (1 - p_detect_decay × altitude / z_max)
Stage 2 — Environment modifier:
env_factor from danger zone (if inside): [0.4, 1.0]
Stage 3 — Motion blur:
motion_penalty = motion_blur_coeff × (speed / max_speed)
Stage 4 — Victim type:
Injured: 1.15 (easier to detect — stationary)
Mobile: 0.85 (harder — moving)
Stage 5 — Hardware miss rate:
P_final × (1 - base_miss_rate)
Clamp to [0.0, 1.0]
\_get_env_factor(victim_pos, obstacles) → float

Check if victim inside danger zone
Returns: Danger zone sensor modifier [0.4, 1.0] or 1.0 if clear
\_get_victim_factor(victim) → float

Victim type detection modifier
InjuredVictim: 1.15
MobileVictim: 0.85
check_detected(uav, victim, obstacles) → bool

Full detection pipeline:
Step 1: Check FOV (distance ≤ fov_radius)
Step 2: Check LOS (not blocked by obstacles)
Step 3: Compute P_detect (full noise pipeline)
Step 4: Stochastic sample (Bernoulli trial)
Returns: True if detected, False otherwise
scan_victims(uav, victims, obstacles) → ndarray(25,)

Build victim observation vector (5 victims × 5 features)
Process:
Compute distances to all victims
Sort by distance (nearest first)
Take top 5
For each: Check detection → extract features → zero if not detected
Features per victim: [rel_x, rel_y, urgency/5, dist/map_diagonal, found(0/1)]
Padding: Zero-fill if < 5 victims
scan_obstacles(uav, obstacles) → ndarray(12,)

Build obstacle observation vector (4 obstacles × 3 features)
Process:
Compute distances to all obstacles
Sort by distance
Take top 4
Features per obstacle: [rel_x/map_diagonal, rel_y/map_diagonal, type_id]
type_id: 0 = Debris, 1 = DangerZone
Padding: Zero-fill if < 4 obstacles
Design Notes:

Noise model calibrated from real UAV search studies
Altitude-dependent: Higher flight → wider FOV but lower detection prob
Motion blur: Fast movement → harder to detect
Environment effects: Smoke/fire degrade sensors
Victim type: Injured easier than mobile (realistic)
sensors/comm_sensor.py — CommSensor
Purpose: V2V (vehicle-to-vehicle) communication sensor

Attributes:

cfg: AppConfig
comm_range: float — From sensor_cfg.comm_range_m
Methods:

scan(ego_uav, all_active_uavs) → ndarray(9,)

Build teammate observation vector (3 teammates × 3 features)
Process:
Filter: UAVs within comm_range AND not self
Sort by distance (nearest first)
Take top 3
Extract features
Features per teammate: [norm_dist, norm_bearing, norm_alt_diff]
norm_dist = dist / comm_range ∈ [0,1]
norm_bearing = bearing / π ∈ [-1,1]
norm_alt_diff = (z_teammate - z_ego) / z_max ∈ [-1,1]
Padding: Zero-fill if < 3 teammates in range
Returns: Flattened vector [9]
get_n_in_range(ego_uav, all_uavs) → int

Count teammates in comm range
get_teammates_in_range(ego_uav, all_uavs) → List[UAV]

Get sorted list of teammates in range (by distance)
Returns: List of UAV objects
Design Notes:

Limited range → partial observability (realistic constraint)
Nearest-first prioritization (most relevant info)
Normalized features → scale invariance for RL
📁 observation/ — Observation Builder
observation/obs_builder.py
ObsResult (Dataclass)
Purpose: Container for actor and critic observations

Attributes:

actor_obs: Dict[int, ndarray] — Per-agent local observations {uav_id: obs[68]}
critic_obs: ndarray — Global centralized observation [554]
ObservationBuilder
Purpose: Build observations cho CTDE (Centralized Training Decentralized Execution)

Attributes:

cfg: AppConfig
obs_cfg: ObsConfig
env_cfg, uav_cfg, sensor_cfg — Shortcuts
fov_sensor: FOVSensor — Sensor instance
comm_sensor: CommSensor — Comm instance
Actor Observation Layout (68 dims với n_stations=2):

Slice Dims Features Normalization
[0:11] 11 Self: pos(3), vel(3), battery(1), state_onehot(4) pos/vel → /map_size, battery → /100
[11:19] 8 Stations (2): rel_x, rel_y, dist, occupancy rel/dist → /map_diagonal, occupancy → [0,1]
[19:28] 9 Teammates (3): dist, bearing, rel_alt From CommSensor.scan()
[28:40] 12 Obstacles (4): rel_x, rel_y, type_id From FOVSensor.scan_obstacles()
[40:65] 25 Victims (5): rel_x, rel_y, urgency, dist, found From FOVSensor.scan_victims()
[65:68] 3 Coverage: local_15m, local_30m, time_remaining [0,1]
Critic Observation Layout (554 dims):

Slice Dims Features
[0:544] 544 Agent states: 8 UAVs × 68 (zero-padded for disabled/reserve)
[544:554] 10 Global: n_active/n_charging/n_disabled/n_alive (÷n_total), battery_mean/std/min, global_coverage, victims_found_rate, time_remaining
Methods:

build_actor_obs(uav, all_uavs, stations, victims, obstacles, step) → ndarray(68)

Build local observation for một UAV
Process:
Allocate zero array [68]
Write self features [0:11]
Write station features [11:19]
Write teammate features [19:28] (from CommSensor)
Write obstacle features [28:40] (from FOVSensor)
Write victim features [40:65] (from FOVSensor)
Write coverage features [65:68]
All features normalized to roughly [-1, 1] or [0, 1]
build_all(all_uavs, stations, victims, obstacles, coverage_map, step) → ObsResult

Build observations cho tất cả UAVs + critic
Process:
Build actor_obs for each UAV → Dict[int, ndarray(68)]
Build critic_obs:
Stack all 8 UAV observations (zero-pad if < 8)
Append global features [10]
Returns: ObsResult(actor_obs, critic_obs)
Private helpers:

\_write_self(obs, uav, start_idx) — Write [0:11]
\_write_stations(obs, uav, stations, start_idx) — Write [11:19]
\_write_teammates(obs, uav, all_uavs, start_idx) — Write [19:28]
\_write_obstacles(obs, uav, obstacles, start_idx) — Write [28:40]
\_write_victims(obs, uav, victims, obstacles, start_idx) — Write [40:65]
\_write_coverage(obs, uav, coverage_map, step, start_idx) — Write [65:68]
Design Philosophy:

Actor (68): Decentralized execution — local partial observability
Critic (554): Centralized training — global full observability
Normalization: Critical for RL (stable gradients)
Zero-padding: Critic sees all agents (even disabled) → fixed dim
Nearest-first: Prioritize most relevant entities
📁 rewards/ — Reward Functions
rewards/baseline_reward.py — BaselineReward v3.1
Purpose: Hand-crafted reward function (14 components, research-grade baseline)

Key Fixes vs v3.0:

BUG-31: Penalty cap ADDITIVE (not multiplicative scale)
BUG-32: Proximity cap scale theo swarm size
BUG-33: Distance shaping DELTA-based với memory
BUG-34: Terminal bonus không saturate
BUG-35: Battery urgency shaping → distance-to-station incentive
Attributes:

cfg: AppConfig
reward_cfg: RewardConfig
\_battery_death_penalized: Set[int] — One-time penalty tracking
\_collision_penalized: Set[int] — One-time penalty tracking
\_prev_min_dist: Dict[int, float] — Distance memory for shaping
14 Reward Components:

Coverage delta — +6.0 per 1% global coverage increase
Victim found — +50.0 × (urgency/5) when detected
Battery <10% — -1.0 per step (warning)
Battery <5% — -3.0 per step (critical)
Battery death — -100.0 one-time when battery=0
Collision — -30.0 one-time when hit debris
Proximity <1m — -10.0 per step (danger)
Proximity <2m — -3.0 per step
Proximity <3m — -0.5 per step
Time penalty — -0.05 per active UAV per step (efficiency)
Terminal bonus — +200.0 base + bonus (70% coverage + 20% victims + 10% time)
Distance shaping — Delta-based incentive to nearest victim
Penalty cap — Limit total negative per step to -30.0
Step clip — Clip final reward to [-100, +100]
Methods:

reset() → None

Clear penalized sets
Clear distance memory
compute(uavs, victims, obstacles, coverage_map, fleet_manager, newly_found, prev_coverage, step, done, stations) → Dict

Global reward computation
Returns: Dict với 14 component values
Used for: Logging, debugging
compute_per_uav(uav, newly_found_by_uav, all_uavs, victims, obstacles, coverage_map, fleet_manager, prev_coverage, step, done, stations) → float

Per-agent reward (MAPPO/MASAC/MATD3)
IMPORTANT: Shared reward → tất cả agents nhận CÙNG GIÁ TRỊ
Process:
Compute all 14 components
Apply penalty cap (additive adjustment if total penalty < cap)
Clip to [-100, +100]
Returns: Single float (shared value)
\_apply_penalty_cap(components: Dict, cap: float) → None

Adjust penalties to satisfy cap
Logic: If sum(negative components) < cap → add cap - sum uniformly to negative components
Preserves relative ratios (important for learning signal)
\_terminal_bonus(coverage, victims_found_rate, step) → float

Terminal bonus formula:
Base: r_terminal_base = +200.0
Bonus: +100 × (0.7×coverage + 0.2×victims + 0.1×time_norm)
Cap: +100 max bonus
Returns: base + bonus (up to +300 total)
\_delta_shaping_fleet(uavs, victims) → float

Fleet-level distance shaping
Track minimum distance from any UAV to any unfound victim
Delta: prev_min_dist - current_min_dist
Reward: delta × distance_shaping_max (if delta > 0)
Cap: Per-UAV max
\_delta_shaping_single(uav, victims, unfound_victims) → float

Per-UAV distance shaping
Track UAV's distance to nearest unfound victim
Reward approaching, penalize moving away
\_battery_rewards(uavs, stations) → float

Progressive battery penalties (<10%, <5%, death)
Battery urgency shaping:
If battery < 20% → incentive to move toward nearest station
Formula: +0.5 × (1 - dist_to_station / map_diagonal) × (1 - battery/20)
\_collision_reward(uavs, obstacles) → float

One-time -30.0 per UAV first collision
Track in \_collision_penalized set
\_danger_reward(uavs, obstacles) → float

Per-step penalty when inside danger zone
Penalty from danger_zone.penalty (e.g., -3 for fire)
get_component_names() → List[str]

Return list of 14 component names (for logging)
summarize(reward_dict: Dict) → str

Compact log string
Format: "cov+12.0 vic+100.0 bat-5.0 prox-8.0 total+99.0"
Module-Level Functions (Stateless):

\_coverage_delta_reward(prev, cur, weight) — Coverage delta component
\_victim_found_reward(newly_found, r_base) — Sum victim rewards
\_battery_penalty_single(uav, reward_cfg, uav_cfg) — Battery penalties
\_battery_urgency_shaping(uav, stations, map_size) — Urgency shaping
\_proximity_reward(active_uavs, ...) — Proximity penalties
\_proximity_reward_single(uav, active_uavs, ...) — Single UAV proximity
Design Philosophy:

Positive >> negative (RL prefers positive gradients)
Shared reward → cooperative behavior
Terminal bonus >> step rewards → focus on mission success
Delta-based shaping → stable across episodes
One-time penalties → prevent repeated penalization
Caps → prevent reward explosion/implosion
📁 DETAILED FILE DOCUMENTATION (Continued)
📁 env_setup/ — Environments
Note: Folder ban đầu tên env/, đổi thành env_setup/ trong quá trình development để tránh conflict với Python package tên env.

env_setup/base_env.py — SARBaseEnv
Purpose: Gymnasium-compatible environment (single-env interface)

Inheritance: gymnasium.Env

Attributes:

cfg: AppConfig — Master config
backend: LogicBackend — Physics backend
\_reward_fn: BaselineReward — Reward function instance
\_obs_builder: ObservationBuilder — Observation builder
\_map_gen: MapGenerator — Map generator
\_renderer: Visualizer2D | Visualizer3D | None — Renderer (if render_mode set)
\_step_count: int — Current step in episode
\_prev_coverage: float — Coverage at previous step (for delta)
\_episode_reward_sum: float — Cumulative reward
\_map_data: Dict | None — Current map data
\_done: bool — Episode done flag
Spaces:

observation_space: Dict[str, Box] — Dict space với keys "uav_0", "uav_1", ... (Box(68,) float32)
action_space: Dict[str, Box] — Dict space (Box(3,) ∈ [-1,1] float32)
Methods:

reset(seed: int | None = None, options: Dict | None = None) → Tuple[Dict, Dict]

Reset environment to initial state
Process:
Set RNG seed (if provided)
Generate new map: \_map_gen.generate(seed=seed)
Reset backend: backend.reset(map_data)
Reset reward function: reward_fn.reset()
Reset counters: \_step_count=0, \_prev_coverage=0
Build initial observations
Returns:
obs_dict: Dict[str, ndarray(68)] — Actor observations keyed by "uav_0", "uav_1", ...
info: Dict — Episode metadata
Info dict keys (reset):
seed: int — Actual seed used
n_uav, n_stations, n_victims, n_obstacles: int — Entity counts
map_size: float — Map size (meters)
coverage: float — Initial coverage (always 0.0)
coverage_rate: float — Initial coverage rate (0.0)
victims_found, victims_total: int — Victim counts
global_obs: ndarray(554) — ✅ CRITIC OBSERVATION (Phase 2 addition)
step(actions: Dict[str, ndarray]) → Tuple[Dict, Dict, bool, bool, Dict]

Execute one simulation step
Input: actions = {"uav_0": action[3], "uav_1": ...} (string keys)
Process (order critical):
Convert actions: str keys → int keys
Apply actions: backend.apply_actions(actions)
Step physics: backend.step_physics() (battery update)
Step world: backend.step_world() (victims, coverage, detection)
Increment step counter: \_step_count += 1
Check done FIRST (before reward): \_check_done()
Compute rewards: reward_fn.compute_per_uav(...) (shared)
Build observations: \_build_obs_dict()
Update prev_coverage
Returns:
obs_dict: Dict[str, ndarray(68)]
rewards: Dict[str, float] — Same value for all agents (shared)
terminated: bool — Episode success (coverage/victims)
truncated: bool — Timeout (step ≥ max_steps)
info: Dict — Step metadata
Info dict keys (step):
step: int — Current step
coverage_rate: float — Current coverage [0,1]
victims_found, victims_total: int
n_active, n_charging, n_disabled: int — Fleet state
success: bool — Episode success flag
done_reason: str — "coverage" / "victims" / "disabled" / "timeout" / ""
rewards_breakdown: Dict — 14 component values
newly_found_ids: List[int] — Victim IDs found this step
global_obs: ndarray(554) — ✅ CRITIC OBSERVATION
render() → ndarray | None

Render current state
Returns: RGB array [H,W,3] or None (if render_mode="none")
Call: \_renderer.render(uavs, victims, obstacles, stations, coverage_map, step, metrics)
close() → None

Cleanup resources
Close renderer (if exists)
\_build_obs_dict(backend_state, step) → Tuple[Dict[str, ndarray(68)], ndarray(554)]

Build observations from backend state
Process:
Extract entities: uavs, victims, obstacles, stations, coverage_map
Call: obs_builder.build_all(...) → ObsResult
Convert actor obs: int keys → str keys
Returns: (actor_obs_dict, critic_obs)
Phase 2 change: Return tuple instead of just dict
\_check_done(coverage_rate, victims, uavs) → bool

Check termination conditions
Conditions:
Coverage ≥ 90% → done (success)
All victims found → done (success)
All UAVs disabled → done (failure)
Returns: True if any condition met
make(cls, cfg, render_mode, \*\*kwargs) → SARBaseEnv

Classmethod factory
Usage: env = SARBaseEnv.make(cfg, render_mode="human")
Properties:

n_agents — Number of active UAVs
active_uav_ids — List of ACTIVE UAV IDs
coverage_rate — Current coverage rate [0,1]
Step Flow (Critical Order):

text

1. apply_actions(actions) # UAV movement
2. step_physics() # Battery drain/charge
3. step_world() # Detection, coverage marking
4. step_count += 1
5. CHECK done # ← BEFORE reward (BUG-ENV-06 fix)
6. compute_rewards(is_terminal=done)
7. build_observations()
8. return (obs, rewards, done, truncated, info)
   Design Notes:

Gymnasium API compliance (standard RL interface)
Info dict contains rich metadata (for logging/debugging)
Global obs in info → MAPPO critic access
Order matters: done check BEFORE reward (terminal bonus logic)
env_setup/sar_pettingzoo_env.py — SARPettingZooEnv
Purpose: PettingZoo ParallelEnv wrapper (multi-agent standard API)

Inheritance: pettingzoo.ParallelEnv

Key Difference vs Base:

Agent IDs: String format "uav_0", "uav_1", ... (PettingZoo standard)
Observation/action/reward dicts: All keyed by string agent IDs
Info dicts: Per-agent info (each agent has own dict)
Attributes:

base_env: SARBaseEnv — Wrapped Gymnasium env
possible_agents: List[str] — All agent IDs (fixed at init)
agents: List[str] — Current active agents (dynamic, updated each step)
Methods:

reset(seed: int | None = None, options: Dict | None = None) → Tuple[Dict, Dict]

Delegate to base_env.reset()
Returns:
observations: Dict[str, ndarray(68)] — {"uav_0": obs, ...}
infos: Dict[str, Dict] — {"uav_0": {...}, "uav_1": {...}}
Info structure (reset):
Python

infos = {
"uav_0": {
"seed": int,
"n_uav": int,
"n_stations": int,
"n_victims": int,
"n_obstacles": int,
"map_size": float,
"coverage": float, # Grid object
"coverage_rate": float, # [0,1]
"victims_found": int,
"victims_total": int,
"global_obs": ndarray(554) # ← CRITIC OBS
}, # Same dict copied to all agents
}
MAPPO access: global_obs = infos['uav_0']['global_obs']
step(actions: Dict[str, ndarray]) → Tuple[Dict, Dict, Dict, Dict, Dict]

Execute one step
Input: actions = {"uav_0": action[3], ...} (string keys)
Delegate to base_env.step(actions)
Returns (PettingZoo standard):
observations: Dict[str, ndarray(68)]
rewards: Dict[str, float]
terminations: Dict[str, bool] — Per-agent done flags (all same value)
truncations: Dict[str, bool] — Per-agent timeout flags (all same value)
infos: Dict[str, Dict] — Per-agent info dicts
Info structure (step):
Python

infos = {
"uav_0": {
"coverage": CoverageMap, # Grid object
"victims_found": int,
"victims_total": int,
"step": int,
"coverage_rate": float,
"n_active": int,
"n_charging": int,
"n_disabled": int,
"success": bool,
"done_reason": str,
"rewards_breakdown": Dict, # 14 components
"newly_found_ids": List[int], # Victims found this step
"global_obs": ndarray(554) # ← CRITIC OBS
}, # Same dict for all agents
}
MAPPO access: global_obs = infos['uav_0']['global_obs']
observation_space(agent: str) → gymnasium.Space

Return observation space for agent
Returns: Box(low=-inf, high=inf, shape=(68,), dtype=float32)
action_space(agent: str) → gymnasium.Space

Return action space for agent
Returns: Box(low=-1, high=1, shape=(3,), dtype=float32)
render() → ndarray | None

Delegate to base_env.render()
close() → None

Delegate to base_env.close()
Properties:

unwrapped — Return wrapped SARBaseEnv instance
Factory Functions:

make_parallel_env(cfg, render_mode, **kwargs) → SARPettingZooEnv — Create parallel env
make_aec_env(cfg, render_mode, **kwargs) — Create AEC env (not implemented, raises NotImplementedError)
MAPPO Integration:

Python

env = SARPettingZooEnv(cfg, render_mode=None)

# Reset

obs, infos = env.reset(seed=42)
global_obs = infos['uav_0']['global_obs'] # ndarray(554)

# Step

actions = {"uav_0": action0, "uav_1": action1, ...}
obs, rewards, terms, truncs, infos = env.step(actions)
global_obs = infos['uav_0']['global_obs'] # ndarray(554)

# Critic update

value = critic(global_obs) # Centralized value function
Design Notes:

PettingZoo API: Standard for multi-agent RL libraries
String keys: PettingZoo convention (int IDs in base env)
Global obs in per-agent info: Enables CTDE (Centralized Training Decentralized Execution)
Shared reward: All agents receive same value in rewards dict
Shared termination: All agents done simultaneously (cooperative task)
env_setup/vec_env.py — VectorizedEnv ✅ COMPLETE
Purpose: Parallel environment execution (n_envs=2-16) để tăng tốc training

Key Innovation: Auto-balanced rollout length theo n_envs (Phase 2b)

Architecture: Multiprocessing-based

Main process: Training loop (actor/critic forward pass)
Worker processes: N environment instances (physics simulation)
Communication: multiprocessing.Pipe (low latency)
Start method: spawn (CUDA-safe, avoid fork issues)
env_worker(pipe, config, seed) — Worker Process Function
Purpose: Environment worker chạy trong process riêng

Process:

Import modules (isolated namespace)
Create SARPettingZooEnv(config, render_mode=None)
Initialize cache for last valid state (handle done edge case)
Enter command loop: Listen on pipe for commands
Commands:

"reset" — Reset env, return (obs_array[n_agents, 68], global_obs[554], info)
"step" — Step env with actions, return (obs, global_obs, rewards, done, info)
"close" — Cleanup and exit
Key Features:

Cache Last Valid State:

Problem: PettingZoo may return empty obs/info dict when episode done
Solution: Cache last valid obs_array, global_obs, info and reuse if empty
Initialize cache:
Python

last_obs_array = np.zeros((n_agents, obs_dim), dtype=np.float32)
last_global_obs = np.zeros(global_obs_dim, dtype=np.float32)
last_info = {'uav_0': {'coverage_rate': 0.0, 'victims_found': 0, ...}}
Auto Reset on Done:

When episode terminates → auto reset env
Send done=True to trainer BEFORE reset
Next step() command receives fresh episode state
Action Conversion:

Receive: actions ndarray [n_agents, 3]
Convert: actions*dict = {f"uav*{i}": actions[i] for i in range(n_agents)}
PettingZoo expects string keys
Global Obs Extraction:

Extract from info['uav_0']['global_obs']
Fallback to cache if info empty
Reward Array:

Extract from rewards dict (string keys)
Convert to ndarray[n_agents]
Fallback to zeros if empty
Error Handling:

Try-except around full loop
Catch: EOFError, BrokenPipeError, KeyboardInterrupt (graceful exit)
Catch: General exceptions → print traceback → send None sentinel → continue
Finally: Close env and pipe
Code Structure:

Python

def env_worker(pipe, config, seed): # Init
env = SARPettingZooEnv(config, render_mode=None)
last_obs_array, last_global_obs, last_info = ... # Cache

    # Command loop
    while True:
        cmd, data = pipe.recv()

        if cmd == "reset":
            obs, info = env.reset(seed=seed)
            # Update cache
            # Send (obs_array, global_obs, info)

        elif cmd == "step":
            actions = data  # [n_agents, 3]
            obs, rewards, terms, truncs, info = env.step(actions_dict)
            done = any(terms) or any(truncs)
            # Update cache if valid
            # Send (obs_array, global_obs, rewards_array, done, info)
            if done:
                env.reset(seed=seed)  # Auto reset

        elif cmd == "close":
            break

VectorizedEnv Class
Purpose: Manage N worker processes, batch communication

Attributes:

n_envs: int — Number of parallel environments
n_agents: int — Agents per env (=4)
obs_dim: int — Actor obs dim (=68)
global_obs_dim: int — Critic obs dim (=554)
action_dim: int — Action dim (=3)
config: AppConfig — Config reference
pipes: List[Connection] — Parent pipes (one per worker)
processes: List[Process] — Worker processes
Methods:

**init**(config, n_envs=8, start_seed=0)

Create N worker processes
Process:
Get spawn context: ctx = mp.get_context("spawn") (CUDA-safe)
For each env:
Create pipe: parent_pipe, child_pipe = ctx.Pipe()
Create process: p = ctx.Process(target=env_worker, args=(child_pipe, config, start_seed+i))
Start process: p.start()
Close child pipe in parent: child_pipe.close() (ownership transfer)
Store: pipes.append(parent_pipe), processes.append(p)
Wait 0.5s for workers to initialize
Verify: Count alive processes
Raise error if any worker failed to start
Print: "✅ {alive}/{n_envs} environment workers ready!"
reset() → Tuple[ndarray, ndarray]

Reset all environments
Process:
Send "reset" command to all workers: pipe.send(("reset", None))
Receive results: obs, global_obs, info = pipe.recv()
Check for crash: if result is None: raise RuntimeError("Worker crashed")
Stack results
Returns:
obs_batch: ndarray[n_envs, n_agents, obs_dim] — Shape (8, 4, 68)
global_obs_batch: ndarray[n_envs, global_obs_dim] — Shape (8, 554)
step(actions_batch: ndarray) → Tuple

Step all environments in parallel
Input: actions_batch shape [n_envs, n_agents, action_dim] → (8, 4, 3)
Process:
Send actions to all workers (async): pipe.send(("step", actions_batch[i]))
Receive results (blocking): obs, global_obs, rewards, done, info = pipe.recv()
Check for crash: if result is None: raise RuntimeError("Worker crashed")
Stack results
Returns:
obs_batch: ndarray[n_envs, n_agents, obs_dim] — (8, 4, 68)
global_obs_batch: ndarray[n_envs, global_obs_dim] — (8, 554)
rewards_batch: ndarray[n_envs, n_agents] — (8, 4)
dones: List[bool] — Length n_envs
infos: List[Dict] — Length n_envs
close()

Shutdown all workers gracefully
Process:
Send "close" command: pipe.send(("close", None))
Wait for process exit: p.join(timeout=3)
If still alive: p.terminate() → p.join(timeout=1)
Print: "✅ All env workers closed."
**del**()

Destructor: Call close() if not already closed
Ensures cleanup even if user forgets to call close()
Performance:

Speedup: ~2× with n_envs=6 (vs single env)
FPS: 105 steps/s (vs 53 steps/s single env)
Overhead: Pipe communication + process context switching
Optimal: n_envs=4-8 (diminishing returns beyond 8)
Design Notes:

Spawn method: Avoid CUDA fork issues (PyTorch warning)
Async send + sync receive: Minimize idle time
Crash detection: None sentinel from worker → raise error
Auto reset: Workers handle episode boundaries automatically
Cache mechanism: Handle PettingZoo edge case (empty obs on done)
env_setup/backends/base_backend.py — BaseBackend
Purpose: Abstract backend interface (support multiple physics engines)

Methods (Abstract):

reset(map_data: Dict) → None — Initialize from map data
apply_actions(actions: Dict[int, ndarray]) → None — Apply UAV actions
step_physics() → None — Update physics (battery, etc.)
step_world() → None — Update world state (victims, detection, coverage)
get_state() → Dict — Return current state (for observation builder)
Design: Backend-agnostic → easy to swap (Logic → PyBullet → Isaac Gym)

env_setup/backends/logic_backend.py — LogicBackend
Purpose: Pure Python physics backend (fast prototyping, ~1000 steps/s)

Attributes:

cfg: AppConfig
uavs: List[UAV] — Fleet
victims: List[BaseVictim] — Injured + Mobile
stations: List[ChargingStation] — Charging stations
obstacles: List[Debris | DangerZone] — All obstacles
coverage_map: CoverageMap — Coverage grid
fleet_manager: FleetManager — Fleet controller
fov_sensor: FOVSensor — Detection sensor
\_map_data: Dict | None — Current map data
Methods:

reset(map_data: Dict) → None

Build world from map data
Process:
Store map_data
Build stations: \_build_stations(map_data["stations"])
Build UAVs: \_build_uavs(map_data.get("uav_spawns", []), stations)
Build obstacles: \_build_obstacles(map_data["debris"], map_data["danger_zones"])
Build victims: \_build_victims(map_data["victims"])
Reset coverage map: coverage_map.reset()
Reset fleet manager: fleet_manager.reset(uavs, stations)
apply_actions(actions: Dict[int, ndarray]) → None

Apply RL actions to UAVs
Process:
For each UAV:
If state=ACTIVE AND id in actions: uav.apply_action(actions[id])
Elif state=RETURNING: uav.auto_navigate(target_station.pos)
Elif state=DEPLOYING: uav.auto_navigate(deploy_target)
Else: No action (CHARGING/DISABLED)
step_physics() → None

Update physics (currently only battery)
Process: for uav in uavs: uav.update_battery(stations)
step_world() → None

Update world state (detection, coverage, victims)
Process:
Fleet manager step: fleet_manager.step() (enforce constraints)
Update victims: for v in victims: v.update(obstacles)
Update coverage: For each ACTIVE UAV → coverage_map.mark_explored(...)
Detection: For each ACTIVE UAV → for v in victims: if fov_sensor.check_detected(uav, v, obstacles): v.mark_found(...)
get_state() → Dict

Return current state for observation builder
Keys: uavs, victims, stations, obstacles, coverage_map, fleet_manager
Private builders:

\_build_stations(data) → List[ChargingStation] — Create from dict list
\_build_uavs(spawn_data, stations) → List[UAV] — Spawn UAVs
If uav_spawns provided: Use positions from map_data (FIX-P10)
Else: Fallback spawn near stations
⚠️ Known issue: All UAVs spawn with state=ACTIVE (no reserve pool initially)
\_build_obstacles(debris_data, danger_data) → List[Debris | DangerZone] — Create obstacles
\_build_victims(data) → List[InjuredVictim | MobileVictim] — Create victims
Performance: ~1000 steps/s (pure Python, no physics engine overhead)

Design Notes:

Modular: Easy to replace with PyBullet/Isaac backend
State-based UAV control: RL controls ACTIVE, auto-navigate handles RETURNING/DEPLOYING
Detection every step: Check all active UAVs against all unfound victims (O(M×N))
📁 visualization/ — Renderers
visualization/renderer_factory.py
Purpose: Factory pattern for renderer creation

Function:

Python

def create_renderer(cfg, render_mode, viz_mode) -> Visualizer2D | Visualizer3D | None:
if render_mode == "none":
return None

    if viz_mode == "2d":
        return Visualizer2D(cfg, render_mode)

    elif viz_mode == "3d":
        try:
            return Visualizer3D(cfg, render_mode)
        except ImportError:  # Matplotlib 3D not available
            return Visualizer2D(cfg, render_mode)  # Fallback

Usage: renderer = create_renderer(cfg, render_mode="human", viz_mode="2d")

visualization/visualizer2d.py — Visualizer2D
Purpose: 2D Matplotlib renderer (~50ms/frame, production-ready)

Optimization: Figure reuse (create once, update data every frame)

Layout: [3:1] ratio

Left panel (75%): Map view (top-down)
Right panel (25%): Info panel (metrics)
Attributes:

cfg: AppConfig
render_mode: str — "human" (window) / "rgb_array" (return image)
fig, ax_map, ax_info — Matplotlib objects
\_artists: Dict — Cached plot elements (untuk update efficiency)
Methods:

render(uavs, victims, obstacles, stations, coverage_map, step, metrics=None) → ndarray | None

Main render method
Process:
Clear axes
Draw coverage grid (gray scale by staleness)
Draw obstacles (brown hatches for debris, colored fills for danger zones)
Draw stations (green boxes + capacity text)
Draw victims:
Not found: Orange "X" marker
Found: Green circle
Size proportional to urgency
Draw UAVs:
Color by state: ACTIVE=blue, RETURNING=orange, CHARGING=green, DEPLOYING=purple, DISABLED=red
Arrow for velocity direction
FOV circle (transparent blue)
Battery bar above UAV
ID label
Draw info panel:
Step, Time remaining
Coverage %, Victims found
Fleet status (n_active/charging/disabled)
Battery stats (mean/min)
Custom metrics (if provided)
Render to canvas
Returns:
render_mode="rgb_array": ndarray[H,W,3] uint8
render_mode="human": None (display in window)
close()

Close Matplotlib figure
Call: plt.close(self.fig)
Private draw methods:

\_draw_coverage(ax, coverage_map, step) — Heatmap (gray = unexplored, darker = stale)
\_draw_obstacles(ax, obstacles) — Debris hatches, danger fills
\_draw_stations(ax, stations) — Green rectangles + occupancy text
\_draw_victims(ax, victims) — X markers vs circles
\_draw_uavs(ax, uavs) — State-based colors, FOV, battery bars
\_draw_battery_bar(ax, pos, battery) — Above UAV, green→yellow→red gradient
\_draw_info_panel(ax, step, max_steps, metrics) — Text block
Color Scheme:

Coverage: White (unexplored) → Gray (explored) → Dark gray (stale)
UAV states: Blue (ACTIVE) / Orange (RETURNING) / Green (CHARGING) / Purple (DEPLOYING) / Red (DISABLED)
Victims: Orange (missing) / Green (found)
Danger zones: Red (fire) / Gray (smoke) / Yellow (gas) / Brown (collapse) / Purple (radiation)
Debris: Brown hatch pattern
Performance: ~50ms/frame (reuse figure + selective redraw)

visualization/visualizer3d.py — Visualizer3D
Purpose: 3D Matplotlib renderer (~400ms/frame, demo quality)

Warning: Slow — NOT recommended for training (use viz_mode="none")

Layout: [3:1] ratio

Left panel (75%): 3D isometric view
Right panel (25%): Dashboard (metrics)
Attributes:

cfg: AppConfig
render_mode: str
fig, ax_3d, ax_info — Matplotlib 3D axes
export_method: str — "fig2data" / "canvas" / "fig2img" / "buffer" (fallbacks)
Methods:

render(uavs, victims, obstacles, stations, coverage_map, step) → ndarray
Main render (new figure every frame — expensive!)
Process:
Clear 3D axes
Draw debris: Cylinders (circle) / Boxes
📁 visualization/ — Renderers (continued)
visualization/visualizer3d.py — Visualizer3D (continued)
Purpose: 3D Matplotlib renderer (~400ms/frame, demo quality)

Warning: Slow — NOT recommended for training (use viz_mode="none")

Layout: [3:1] ratio

Left panel (75%): 3D isometric view
Right panel (25%): Dashboard (metrics)
Methods (continued from Part 2):

render(uavs, victims, obstacles, stations, coverage_map, step) → ndarray

Main render (new figure every frame — expensive!)
Process:
Clear 3D axes
Draw debris: Cylinders (circle) / Boxes (rectangle) / Line3D meshes (polygon)
Draw danger zones: Semi-transparent colored volumes (red/gray/yellow/brown/purple)
Draw coverage grid: Surface plot on ground plane (gray scale)
Draw stations: Green boxes with capacity text
Draw victims:
Not found: Orange 3D crosses (x,y,z lines)
Found: Green spheres
Draw UAVs:
Scatter points colored by state
Quiver arrows for velocity vectors
FOV cones (transparent blue triangular meshes)
Battery bars (vertical lines next to UAV)
Draw dashboard (text annotations):
Step, Coverage, Victims, Fleet stats, Battery stats
Export to image (4 fallback methods)
Returns: RGB array
Private draw methods:

_draw_debris(ax, debris) — Cylinders/boxes using _cylinder_faces(), _box_faces()
_draw_danger_zones(ax, zones) — Colored transparent meshes
_draw_stations(ax, stations) — Boxes + text
_draw_victims(ax, victims) — Crosses (3D) or spheres
_draw_uavs(ax, uavs) — Scatter + quiver + FOV cones
_draw_fov(ax, uav) — Cone mesh: _cone_faces()
_draw_dashboard(ax, step, max_steps, metrics) — Text block
Geometry helpers:

_circle_xy(center, radius, n_points=20) → ndarray(N,2) — Circle points for base
_cylinder_faces(center, radius, height, n_points=20) → Tuple[Poly3DCollection] — Cylinder mesh
_box_faces(center, w, h_2d, height_3d, rotation) → Tuple[Poly3DCollection] — Box mesh
_cone_faces(apex, base_center, radius, n_points=20) → Tuple[Poly3DCollection] — Cone mesh
_to_rgb(fig) → ndarray

Export Matplotlib figure to RGB array (uint8)
4 fallback methods:
fig2data(fig) — Fastest (if installed)
fig.canvas.tostring_rgb() — Canvas render
fig.savefig() to BytesIO → read as PIL/Pillow
fig.canvas.buffer_rgba() — Last resort
Performance: ~400ms/frame (3D rendering is inherently slow in Matplotlib)
Usage: Demo purposes only, not for training monitoring

📁 training/ — Training Pipeline
training/curriculum.py — CurriculumManager
Purpose: Manage curriculum learning progression (EASY → MEDIUM → HARD)

Data Structures:

StageStats (Dataclass): Per-stage metrics tracking

stage_name: str — "easy"/"medium"/"hard"
episodes_done: int — Episodes in this stage
coverage_list: List[float] — Coverage rates for last N episodes
victims_list: List[float] — Victim rates for last N episodes
reward_list: List[float] — Episode rewards for last N episodes
CurriculumManager Class:

Attributes:

stages: List[StageConfig] — [EASY, MEDIUM, HARD]
_stage_idx: int = 0 — Current stage index
_stats_cache: Dict[str, StageStats] — Per-stage stats
window_size: int = 50 — Rolling window for advance check
Properties:

current_stage: StageConfig — Current stage instance
stage_idx: int — Current index
is_final_stage: bool — At HARD?
total_episodes: int — Sum of all stage episodes
Methods:

update(coverage: float, victims_rate: float, reward: float) → None

Input: coverage in [0,1], victims_rate in [0,1], reward in float
Append to current stage stats
Trim to window_size (keep last 50 entries)
should_advance() → bool

Check if ready to advance to next stage
Conditions:
episodes_done >= min_episodes (minimum episodes)
avg_coverage >= advance_coverage (coverage threshold)
avg_victims >= advance_victims (victim rate threshold)
Not already at final stage
Averages computed over rolling window
advance() → None

Increment _stage_idx
Log transition message
apply_to_config(cfg: AppConfig) → None

Apply current stage params to AppConfig
Calls: cfg.apply_stage(current_stage)
Modifies: map_size, n_victims, n_debris, n_danger_total, max_steps
get_status() → Dict

Return comprehensive status dict
Keys: stage_name, episodes_done, avg_coverage, avg_victims, avg_reward, thresholds, ready_to_advance
print_status() → None

Formatted console output
Usage in Training Loop:

Python

# After each episode
curriculum_manager.update(coverage, victims_rate, reward)

if curriculum_manager.should_advance():
    curriculum_manager.advance()
    curriculum_manager.apply_to_config(cfg)
    env = create_new_env(cfg)  # Rebuild env with new stage
Design Notes:

Rolling window: Smooth metrics, prevent noise triggering
Min episodes: Prevent early advancement
Controlled difficulty progression: EASY → MEDIUM → HARD
Stage params modify AppConfig in-place (single source of truth)
training/curriculum_trainer.py — CurriculumTrainer
Purpose: Training loop with random policy (Phase 1 placeholder, replaced by MAPPO in Phase 2)

Attributes:

cfg: AppConfig
render_every: int — Render interval (0=none)
_history: Dict — Episode metrics history
_episode_rewards, _episode_coverages, _episode_victims, _episode_steps — Lists
Methods:

train(total_episodes: int) → None

Main training loop
Process:
Initialize curriculum manager
Apply initial stage (EASY)
For each episode:
Check curriculum advance
Create env if stage changed
Run episode: _run_episode()
Update curriculum
Log/plot periodically
Render frames (if enabled)
Print final summary
Save plots
_build_env() → SARBaseEnv

Create new environment instance with current config
_run_episode(episode_num, render_frames=False) → Dict

Run one episode with random actions
Process:
Reset env
Loop max_steps:
Sample random actions: _sample_actions(n_uav)
Step env
If render: capture frame
Update logger
If done: break
Finalize episode logger
Returns: Dict with metrics
_sample_actions(n_uav: int) → Dict[int, ndarray]

Generate random actions (uniform [-1,1]³)
Returns: {0: action[3], 1: action[3], ...} (int keys)
_save_episode_visualization(frames, episode, stage) → None

Save first/last frame as PNG
Optional: Create GIF if Pillow available
_plot_training_curves(history, episode, final=False) → None

4-panel plot: Coverage, Victims, Reward, Stage distribution
Save to results/curriculum/plots/
_save_gif(frames, path) → None

Create GIF from frames (slow, requires Pillow)
_print_summary(history) → None

Print final statistics
History Structure:

Python

history = {
    "episodes": List[int],
    "coverages": List[float],
    "victims": List[float],
    "rewards": List[float],
    "stages": List[str],
    "steps": List[int],
}
Status: Phase 1 placeholder — replaced by MAPPOTrainer in Phase 2

📁 training/algorithms/mappo/ — MAPPO Implementation (6 files)
training/algorithms/mappo/__init__.py
Purpose: Exports all MAPPO modules

Exports:

MLP, orthogonal_init, get_parameter_count — From networks.py
ActorNetwork — From actor.py
CriticNetwork — From critic.py
RolloutBuffer — From buffer.py
MAPPOTrainer — From trainer.py
training/algorithms/mappo/networks.py — MLP Foundation
Purpose: Reusable MLP module with orthogonal weight initialization (PPO best practice)

Functions:

orthogonal_init(layer, gain=1.0) → None

Initialize layer weights orthogonally (improves stability)
For Linear layers: Orthogonal weight matrix
For Conv2d layers: Orthogonal kernel
Bias: Initialize to 0
Typical gains:
Hidden layers: gain = sqrt(2) (ReLU-like activations)
Output layers: gain = 0.01 (small initial outputs)
get_parameter_count(model) → int

Count trainable parameters
Returns: Total count
print_network_summary(model, name="Network") → None

Print architecture summary
Includes: Layer name, input dim, output dim, parameter count
Class:

MLP (nn.Module)
Constructor: __init__(input_dim, hidden_dims, output_dim, activation='tanh', use_layer_norm=False, output_activation=None)

input_dim: int — Input dimension
hidden_dims: tuple — List of hidden layer sizes, e.g., (256, 256)
output_dim: int — Output dimension
activation: str — 'tanh', 'relu', 'elu'
use_layer_norm: bool — Add LayerNorm after each hidden layer
output_activation: str | None — Optional output activation
Build process:

Convert hidden_dims to list
Build sequential:
For each hidden layer: Linear → Activation → (optional LayerNorm)
Final layer: Linear → (optional output_activation)
Apply orthogonal init to all layers:
Hidden: gain=√2
Output: gain=0.01
Forward: forward(x) → [batch, output_dim]

Bug fix applied: dims = [input_dim] + list(hidden_dims) (hỗ trợ cả tuple và list)

Design Notes:

Orthogonal init: Standard for PPO (avoids exploding/vanishing gradients)
Small output scale: Prevents large initial actions/predictions
Flexible: Any activation, optional layer norm
Shared by both actor and critic
training/algorithms/mappo/actor.py — ActorNetwork
Purpose: Gaussian policy network (stochastic policy, decentralized execution)

Architecture: obs[68] → MLP(68, [256,256], 3, activation='tanh') → mean[3], learnable log_std[3]

Class: ActorNetwork

Attributes:

obs_dim: int = 68 — Input observation dimension
action_dim: int = 3 — Output action dimension (vx, vy, vz)
mean_net: MLP — Mean network (68 → 256 → 256 → 3)
log_std: nn.Parameter — Learnable log standard deviation [3] (state-independent)
Methods:

__init__(obs_dim=68, action_dim=3, hidden_dims=(256,256), activation='tanh', use_layer_norm=False)

Build mean_net: MLP with output_activation=None (unbounded mean)
Initialize log_std: nn.Parameter(torch.zeros(action_dim)) → std=1.0 initially
forward(obs) → Tuple[Tensor, Tensor]

Input: obs shape [batch, 68]
Output: (mean[batch,3], std[batch,3])
Process:
mean = mean_net(obs)
std = exp(log_std).expand(batch, -1) # Same std for all batch
get_action(obs, deterministic=False) → Tuple[Tensor, Tensor]

Input: obs [batch, 68]
Output: (action[batch,3], log_prob[batch])
Process:
Get mean, std via forward
If deterministic: action = mean
Else: Sample from Normal(mean, std) using reparameterization
Compute log_prob of sampled action (sum over dims)
Clamp action to [-1, 1] (safe action boundaries)
Returns: (action, log_prob_batch)
evaluate_actions(obs, actions) → Tuple[Tensor, Tensor]

Input: obs [batch, 68], actions [batch, 3]
Output: (log_prob[batch], entropy[batch])
Process:
Get mean, std via forward
Create Normal(mean, std)
Compute log_prob of given actions
Compute entropy of distribution
Used for PPO update (computing ratio)
get_log_std() → Tensor

Return current log_std parameter (for logging)
Returns: tensor([log_std_x, log_std_y, log_std_z])
set_log_std(value: float) → None

Set log_std uniformly
Usage: ActorNetwork.set_log_std(-0.5) → exploration control
Design Philosophy:

State-independent std: PPO standard, simpler than state-dependent
Learnable log_std: Entropy auto-adapts during training
Action clamping: Safety against extreme values
Reparameterization trick: Allows gradient through sampling
Shared weights across agents: All UAVs use same policy (parameter sharing)
training/algorithms/mappo/critic.py — CriticNetwork
Purpose: Centralized value function (CTDE — Centralized Training Decentralized Execution)

Architecture: global_obs[554] → MLP(554, [512,256], 1, activation='tanh') → value[1]

Class: CriticNetwork

Attributes:

global_obs_dim: int = 554 — Centralized observation dimension
value_net: MLP — Value network (554 → 512 → 256 → 1)
Methods:

__init__(global_obs_dim=554, hidden_dims=(512,256), activation='tanh', use_layer_norm=False)

Build value_net: MLP with output_activation=None (unbounded value)
forward(global_obs) → Tensor

Input: global_obs [batch, 554]
Output: value [batch, 1]
get_value(global_obs) → Tensor

Input: global_obs [batch, 554]
Output: value [batch] (squeezed from [batch,1])
Convenience: Squeeze last dim
compute_loss(global_obs, returns) → Tensor

Input: global_obs [batch, 554], returns [batch] (target values)
Output: Scalar MSE loss
Process:
Get predicted values
Calculate MSE: mean((values - returns)²)
compute_value_metrics(global_obs, returns) → Dict

Input: global_obs [batch, 554], returns [batch]
Output: Dict with metrics
Keys:
value_loss: float — MSE loss
explained_variance: float — 1 - Var(returns - values) / Var(returns)
mean_pred, std_pred: float
mean_target, std_target: float
Helper Functions (module-level):

test_critic_accuracy(critic, test_obs, test_value) — Test prediction accuracy
initialize_critic_for_env(env, critic) — Extract global_obs from env for initialization
Design Philosophy:

Centralized: Sees full global state (all UAVs + fleet stats)
Value function V(s): Estimates expected return from global state
Shared critic: Single critic for all agents (MAPPO standard)
MSE loss: Fits value function to TD(λ) returns
Explained variance: Key metric for value function quality (>0.5 is good, >0.8 is excellent)
training/algorithms/mappo/buffer.py — RolloutBuffer ✅ COMPLETE (Fixed)
Purpose: Experience storage + GAE (Generalized Advantage Estimation) computation

Storage Layout:

Array	Shape	Description
observations	[rollout_length, n_agents, 68]	Actor observations
global_obs	[rollout_length, 554]	Critic observations
actions	[rollout_length, n_agents, 3]	Actions taken
rewards	[rollout_length, n_agents]	Rewards received (shared)
values	[rollout_length, n_agents]	Value estimates from critic
log_probs	[rollout_length, n_agents]	Log probabilities of actions
dones	[rollout_length]	Episode termination flags
advantages	[rollout_length, n_agents]	Computed GAE advantages
returns	[rollout_length, n_agents]	Computed TD(λ) returns
Class: RolloutBuffer

Attributes:

capacity: int — Max steps (rollout_length × n_envs khi vectorized)
n_agents: int = 4
gamma: float = 0.99 — Discount factor
gae_lambda: float = 0.95 — GAE trace parameter
ptr: int = 0 — Current insertion index
(all storage arrays initialized to zeros)
Methods:

add(obs, global_obs, actions, rewards, values, log_probs, done) → None

Add one transition (one step from one env)
Input: All as numpy arrays
Check: ptr < capacity (raise RuntimeError if overflow)
Increment: ptr += 1
compute_gae(last_values, last_done) → None

Compute Generalized Advantage Estimation
Supports partial buffer (early stop)
Process:
Determine actual length: actual_length = min(ptr, capacity)
If actual_length == 0: Return (empty buffer)
Initialize GAE array: gae = zeros(actual_length, n_agents)
Backward iteration (vectorized — all agents simultaneously):
For t from actual_length-1 down to 0:
If last step (t == actual_length-1):
next_value = last_values # Bootstrap
next_non_terminal = 1.0 - float(last_done)
next_gae = 0.0
Else:
next_value = self.values[t+1]
next_non_terminal = 1.0 - self.dones[t+1]
next_gae = gae[t+1]
TD residual: delta = rewards[t] + gamma × next_value × next_non_terminal - values[t]
GAE: gae[t] = delta + gamma × gae_lambda × next_non_terminal × next_gae
Store: advantages[:actual_length] = gae
Compute returns: returns[:actual_length] = advantages + values
Normalize advantages: (adv - mean) / (std + 1e-8) across all agents × steps
get_batches(batch_size) → Iterator[Dict]

Yield random minibatches for PPO update
Supports partial buffer
Process:
Determine actual_length = min(ptr, capacity)
Flatten:
obs: [actual_length, n_agents, 68] → [actual_length×n_agents, 68]
actions: [actual_length, n_agents, 3] → [actual_length×n_agents, 3]
log_probs: [actual_length, n_agents] → [actual_length×n_agents]
advantages: [actual_length, n_agents] → [actual_length×n_agents]
returns: [actual_length, n_agents] → [actual_length×n_agents]
global_obs: [actual_length, 554] → repeat for agents → [actual_length×n_agents, 554]
Permutation: indices = np.random.permutation(n_samples)
Yield batched slices:
Python

yield {
    'obs': obs_flat[batch_indices],
    'global_obs': global_obs_flat[batch_indices],
    'actions': actions_flat[batch_indices],
    'old_log_probs': log_probs_flat[batch_indices],
    'advantages': advantages_flat[batch_indices],
    'returns': returns_flat[batch_indices],
}
clear() → None

Reset pointer: ptr = 0
get_stats() → Dict

Return buffer statistics
Keys:
buffer_size: int — Actual filled size
buffer_fill: float — Fill ratio [0,1]
mean_reward, mean_value, mean_advantage, std_advantage, mean_return: float
Key Fixes (Phase 2b):

✅ Supports partial buffer (early stop) — actual_length = min(ptr, capacity) instead of self.full
✅ Vectorized GAE: Compute for all agents simultaneously (4× speedup vs per-agent loop)
✅ Advantage normalization: Improves training stability
✅ Graceful empty buffer: actual_length == 0 → return
Design Notes:

Pre-allocated numpy arrays: Zero copy, fast access
Flatten + shuffle: Standard PPO minibatch sampling
GAE formula: Classical TD(λ) trace
Advantage normalization: Critical for stable gradients
training/algorithms/mappo/trainer.py — MAPPOTrainer ✅ COMPLETE
Purpose: Main MAPPO training loop (single + vectorized env support, auto-balanced)

Class: MAPPOTrainer

Attributes:

Config: Full AppConfig reference
Device: torch device (CPU/CUDA)
Networks:
actor: ActorNetwork — Policy network
critic: CriticNetwork — Value network
Optimizers:
actor_optimizer: Adam — LR from config
critic_optimizer: Adam — LR from config (higher than actor)
Buffer: RolloutBuffer — Experience buffer
Tracking:
episode_rewards, episode_lengths, episode_coverage, episode_victims — Deques (maxlen=100)
total_episodes_done: int — Cumulative episodes
total_steps_collected: int — Cumulative steps
Logging:
log_interval, viz_interval, checkpoint_interval — From config
Directories:
output_dir = results/mappo/{run_name}/
checkpoint_dir, viz_dir, plots_dir
Methods:

__init__(config, device, run_name, n_envs=1)

Create networks, optimizers, buffer
Buffer capacity = rollout_length × n_envs (khi vectorized)
Setup directories
Print network summary (actor params: ~84K, critic params: ~416K)
select_action(obs_dict, deterministic=False) → Tuple[Dict, ndarray, ndarray]

Select actions for all agents
Input: obs_dict = {"uav_0": obs[68], ...}
Process:
Stack observations: [n_agents, 68] → tensor
Forward actor: get_action(obs, deterministic)
Convert to numpy → dict
Returns:
actions_dict: Dict[str, ndarray(3)] — For PettingZoo env
actions_np: ndarray[n_agents, 3] — For buffer/tracking
log_probs_np: ndarray[n_agents]
get_values(global_obs) → ndarray[n_agents]

Get value estimates from critic
Input: global_obs [554]
Process: Forward critic → broadcast to n_agents (shared value)
Returns: ndarray[n_agents] (same value for all agents)
train(total_episodes, curriculum_manager=None, seed=42) → None

Main training loop
Creates tqdm progress bar with real-time metrics
Loop until total_episodes_done >= total_episodes:
Run rollout: rollout_metrics = self.rollout(env, pbar=pbar, max_episodes=total_episodes)
Update networks: train_metrics = self.update()
Update progress bar postfix with stable metrics
Log detailed stats every log_interval updates
Save viz/checkpoints every viz_interval/checkpoint_interval
Check curriculum advancement
On curriculum advance: Close old env, apply new stage, create new env
Final save and summary
rollout(env, pbar=None, max_episodes=None) → Dict

Dispatcher:
n_envs == 1 → _rollout_single()
n_envs > 1 → _rollout_vectorized()
_rollout_single(env, pbar=None, max_episodes=None) → Dict

Collect rollout_length steps from single env
Process:
Reset env
Loop: select_action → get_values → env.step → buffer.add
On episode done: log metrics, update pbar (+1 per episode), check early stop
If max_episodes reached: break
After collection: compute_gae → return metrics
_rollout_vectorized(env, pbar=None, max_episodes=None) → Dict

Collect steps from N parallel envs
Process:
Reset all envs → obs_batch [n_envs, n_agents, 68], global_obs_batch [n_envs, 554]
Loop rollout_length iterations:
Batch inference: Flatten obs → single forward pass → unflatten
Step all envs simultaneously: env.step(actions_batch)
Process each env:
buffer.add(...)
Track per-env rewards/lengths
On done: log, pbar.update(1), check early stop
Check stop flag → break outer loop
After collection: compute_gae with mean last_value → return metrics
update() → Dict

PPO update with n_epochs × minibatches
For each epoch:
For each minibatch from buffer:
Actor update:
Compute log_probs, entropy for stored actions
Compute ratio: exp(log_probs - old_log_probs)
PPO clipped loss: -min(ratio×adv, clip(ratio, 1-ε, 1+ε)×adv) - entropy_coeff×entropy
Optimize actor
Critic update:
Compute values for global_obs
MSE loss: mean((values - returns)²)
Optimize critic
Track losses, entropy, clip fraction
Clear buffer
Return metrics dict
_create_env(seed) → env

Create single or vectorized env
n_envs == 1: SARPettingZooEnv
n_envs > 1: VectorizedEnv(config, n_envs, start_seed=seed)
_save_visualization(env, update) → None

Render và lưu 2D snapshot
Only works with single env (SARPettingZooEnv has render())
Save to viz_dir/update_{update:05d}.png
save_checkpoint(update, curriculum_manager) → None

Save full checkpoint (.pt file)
Includes: networks state, optimizers state, config, episode stats
load_checkpoint(path) → int

Load checkpoint
Returns: update number
Training Flow:

text

while episodes < total:
    rollout_metrics = rollout(env, max_episodes=total)   # Collect experience
    train_metrics = update()                              # PPO optimization
    update += 1
    log metrics / save checkpoint / update curriculum pbar
Key Features:

✅ Auto-balanced rollout length (Phase 2b)
✅ Early stop support (max_episodes parameter)
✅ Real-time tqdm progress (episode-level bar, rollout-level metrics)
✅ Curriculum learning integration
✅ Checkpointing & resumption
✅ Memory management (env.close() on stage change)
📁 Root Files
train_mappo.py ✅ COMPLETE
Purpose: CLI entry point for MAPPO training

Command-Line Arguments:

Argument	Default	Description
--total-episodes	3000	Total episodes to train
--seed	42	Random seed
--device	auto	cpu/cuda/auto
--run-name	auto-generate	Run name for logging
--n-envs	1	Number of parallel environments
--no-balance	False	Disable auto-balance rollout
--no-curriculum	False	Disable curriculum learning
--stage	easy	Stage when no-curriculum
--max-steps	None	Override max episode steps
--map-size	None	Override map size
--n-victims	None	Override victim count
--n-debris	None	Override debris count
--rollout-length	None	Base rollout length (overrides config)
--batch-size	None	Minibatch size
--n-epochs	None	PPO epochs
--lr-actor	None	Actor learning rate
--lr-critic	None	Critic learning rate
--gamma	None	Discount factor
--gae-lambda	None	GAE lambda
--clip-epsilon	None	PPO clip epsilon
--entropy-coeff	None	Entropy bonus coefficient
Key Logic:

Auto-Balance (default ON):

Python

# if n_envs > 1 and not no_balance:
final_rollout = calculate_balanced_rollout(
    base_rollout=2048,      # hoặc user override
    n_envs=args.n_envs,
    batch_size=256,
    max_episode_steps=cfg.env.max_steps,
)
# Đảm bảo same số updates giữa các n_envs
Estimate Updates:

Python

eps_per_update = (final_rollout × n_envs) / avg_episode_length
est_updates = total_episodes / eps_per_update
Output Summary:

text

🚁 SAR UAV SWARM — MAPPO TRAINING
  n_envs:            6
  Auto-balance:      ON (auto)
  Target episodes:   3000
  Est. updates:      ~540
  Rollout (base):    2048
  Rollout (final):   384
  Rollout ≥ max_steps: 384 ≥ 400 ⚠️ (clamped to 512 if needed)
Usage Examples:

Bash

# Quick test
python train_mappo.py --total-episodes 50 --max-steps 400 --n-envs 6

# Full training
python train_mappo.py --total-episodes 3000 --n-envs 6 --device cuda

# Single env baseline
python train_mappo.py --total-episodes 3000 --n-envs 1 --device cuda

# Disable auto-balance (advanced)
python train_mappo.py --total-episodes 3000 --n-envs 6 --no-balance
test_trainer_smoke.py ✅ COMPLETE
Purpose: Quick smoke test for trainer (no crash check)

Configuration:

max_steps = 50 (short episodes)
rollout_length = 200 (small buffer)
total_updates = 5
No curriculum, EASY stage
Usage: python test_trainer_smoke.py

Checks:

Trainer initialization
Rollout collection
PPO update
No crash, no memory leak
🔄 EXECUTION FLOWS
Full Training Flow (n_envs=6, 3000 episodes)
text

1. PARSE ARGS & CONFIGURE
   ├─ Parse CLI arguments
   ├─ Create AppConfig
   ├─ Apply curriculum (start: EASY)
   └─ Auto-balance rollout_length: 2048 → 384 (or clamped)

2. INITIALIZE TRAINER
   ├─ Create ActorNetwork (68→256→256→3)
   ├─ Create CriticNetwork (554→512→256→1)
   ├─ Create RolloutBuffer (capacity=2304)
   ├─ Create Adam optimizers
   └─ Setup output dirs

3. TRAINING LOOP (while episodes < total)
   ├─ ROLLOUT (collect experience)
   │   ├─ Reset 6 vectorized envs → obs_batch[6,4,68], global_obs_batch[6,554]
   │   └─ For i in range(384):  # rollout_length
   │       ├─ BATCH INFERENCE
   │       │   ├─ Flatten obs: [6,4,68] → [24,68]
   │       │   ├─ Actor.get_action([24,68]) → actions[24,3], log_probs[24]
   │       │   ├─ Critic.get_value(global_obs) → values[6]
   │       │   └─ Reshape to [6,4,3], [6,4], [6,4]
   │       ├─ STEP ALL ENVS
   │       │   └─ env.step(actions_batch) → next_obs, next_global, rewards, dones
   │       ├─ STORE IN BUFFER
   │       │   └─ For each env: buffer.add(obs, global_obs, actions, rewards, values, log_probs, done)
   │       ├─ TRACK EPISODES
   │       │   └─ If done: log episode metrics, pbar.update(1), check early stop
   │       └─ CHECK EARLY STOP
   │           └─ If total_episodes_done >= target: break
   │   └─ Compute GAE: buffer.compute_gae(last_values, last_done)
   │
   ├─ PPO UPDATE
   │   ├─ For epoch = 1..10:
   │   │   └─ For minibatch in buffer.get_batches(256):
   │   │       ├─ ACTOR UPDATE
   │   │       │   ├─ Compute log_probs, entropy for stored actions
   │   │       │   ├─ ratio = exp(log_prob - old_log_prob)
   │   │       │   ├─ loss = -min(ratio×adv, clip(ratio)×adv) - entropy_coeff×entropy
   │   │       │   └─ actor_optimizer.step()
   │   │       └─ CRITIC UPDATE
   │   │           ├─ values = critic(global_obs)
   │   │           ├─ loss = MSE(values, returns)
   │   │           └─ critic_optimizer.step()
   │   └─ buffer.clear()
   │
   ├─ LOGGING
   │   ├─ Update pbar postfix (every update)
   │   └─ Detailed log (every log_interval)
   │
   ├─ SAVE
   │   ├─ Viz snapshot (every viz_interval)
   │   └─ Checkpoint (every checkpoint_interval)
   │
   └─ CURRICULUM
       ├─ Update metrics
       ├─ Check should_advance()
       └─ If advance: close env → apply new config → create new env

4. FINALIZE
   ├─ Save final checkpoint
   ├─ Close all envs
   └─ Print summary
Single Step Flow (inside env worker)
text

env.step(actions)
→ LogicBackend.apply_actions()
    ├─ ACTIVE UAVs: apply_action(action) → update vel → update pos
    ├─ RETURNING: auto_navigate(target_station.pos)
    └─ DEPLOYING: auto_navigate(deploy_target)
→ LogicBackend.step_physics()
    ├─ ACTIVE: _do_drain() (drain battery proportional to velocity)
    ├─ RETURNING: _do_drain()
    ├─ CHARGING: _do_charge() (charge if in station range)
    └─ DEPLOYING: _do_drain()
→ LogicBackend.step_world()
    ├─ FleetManager.enforce_safety_constraints()
    │    ├─ battery=0 → DISABLED
    │    └─ battery<5% → RETURNING
    ├─ Victim.update() (mobile: random walk)
    ├─ CoverageMap.mark_explored() for each ACTIVE UAV
    └─ FOVSensor.check_detected() for each UAV vs each unfound victim
→ CHECK done/truncated
→ BaselineReward.compute_per_uav() (shared reward)
→ ObservationBuilder.build_all()
→ Return (obs, rewards, done, truncated, info)
Auto-Balance Logic
text

Base rollout = 2048 (cho n_envs=1)
n_envs = 6
Current max_episode_steps = 400

CONSTRAINTS:
1. rollout_per_env = base // n_envs = 341
2. min_for_episode = max_episode_steps × 1.2 = 480  (phải dài hơn episode!)
3. min_for_batch = batch_size // n_agents = 64
4. absolute_min = 128

FINAL: max(341, 480, 64, 128) = 480 → round to 64 → 512

Result:
  n_envs=1: rollout=2048 → ~5.1 eps/update
  n_envs=6: rollout=512  → ~7.5 eps/update (vì 512×6=3072 steps, /408=7.5)
  → ~same number of updates! (actually ~600 vs ~540, acceptable)
📊 TRAINING RESULTS
Baselines (Random Policy)
Stage	Coverage	Victims Found	Reward
EASY	55% ± 11%	53% ± 19%	+150 ± 200
MEDIUM	41% ± 9%	44% ± 17%	+80 ± 180
HARD	32% ± 8%	36% ± 15%	+30 ± 160
MAPPO Results (50 episodes, HARD stage)
Config	Updates	Time	FPS	Reward	Coverage	Victims
n_envs=1 (baseline)	10	6.5 min	52.8	+124.3	55.0%	55.8%
n_envs=6 (auto-balanced)	9	4.3 min	105.4	+158.2	50.3%	55.1%
Key Findings:

n_envs=6: 34% faster, 27% higher reward (likely due to data diversity from 6 seeds)
Performance comparable despite 1 fewer update
Ready for full 3000 episode training
Target Performance (3000 episodes)
Stage	Coverage	Victims	Reward
EASY	82-88%	85-88%	+420-450
MEDIUM	68-72%	70-75%	+300-350
HARD	58-62%	60-65%	+200-250
⚠️ KNOWN ISSUES
Issue	Root Cause	Status	Workaround
UAV spawn all ACTIVE	UAV.__init__ default state=ACTIVE	Known	Reserve pool empty initially; fix in Phase 3
Reward positive with random	Coverage delta > time penalty	Not a bug	Correct task value representation
3D viz slow (~2-5 FPS)	Matplotlib 3D overhead	Workaround	Use viz_mode="none" for training
CurriculumManager.update() params	Parameters named victims_rate, reward	Fixed	Documented correctly
MLP hidden_dims tuple	MLP expects list for concatenation	Fixed	Convert to list in MLP.__init__
🎯 NEXT STEPS (Phase 3)
Immediate Tasks:
Run full training (5 seeds × curriculum × 3000 episodes) with n_envs=6
MASAC implementation — training/algorithms/masac/ (actor, twin Q-critic, replay buffer, trainer)
MATD3 implementation — training/algorithms/matd3/ (deterministic policy, delayed update, target smoothing)
Statistical comparison — Wilcoxon tests, t-tests, learning curves, performance profiles
<!-- Phase 3 (LLM Reward):
Prompt templates cho GPT-4/Claude (task description → reward code)
LLM reward code generation pipeline (API calls, code extraction)
Safety validation layer (syntax check, runtime sandbox, unit test generation)
Compare vs BaselineReward v3.1 (5 seeds, same curriculum)
Phase 4 (Advanced Backends):
PyBullet backend (realistic physics, higher fidelity)
Isaac Gym backend (GPU-accelerated, massive parallelization)
Compare training speed: Logic vs PyBullet vs Isaac
✅ PROJECT STATUS SUMMARY -->
text

Phase 1: Core Infrastructure      ✅ 100% (48 files, 26/26 tests)
Phase 2: MAPPO Implementation     ✅ 100% (trainable, stable)
Phase 2b: Auto-Balanced Vec Env   ✅ 100% (n_envs=1-16, same updates)
Phase 3: MASAC/MATD3              ⬜ Not started
Phase 4: LLM Reward               ⬜ Not started
Phase 5: Advanced Backends        ⬜ Not started

Total files: 56+
Test coverage: 26/26 PASS
Training time (est.): 4 hours (n_envs=6, 3000 eps, GPU)
Ready for research? YES! 🚀

vậy bạn hayx tổng hợp những cái t đã làm ở project này từ đầu tới cuối ghi chi tiết rõ ràng từng folder có file nào file có hàm và thuộc tính nào có tác dụng gì một cách chi tiết (ko cần code) và ghi rõ ràng chi tiết để khi qua đoạn chat ms thì chỉ cần đưa cái đó cho nớ là nó sẽ hiểu đang làm gì và đang thực hiện cái gì biết đang thực heienj cái gì theo format này