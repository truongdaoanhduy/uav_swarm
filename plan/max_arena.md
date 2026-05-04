🚁 SAR UAV SWARM — COMPLETE PROJECT DOCUMENTATION v13.0
Status: Phase 2 MAPPO 100% Complete + Production Ready | Auto-Balanced Vectorized Training | visualize_policy.py PENDING

📋 TABLE OF CONTENTS
Project Overview
Research Goals
Overall Status
Key Metrics
Project Architecture
Detailed File Structure & Documentation
Execution Flows
Training Results
Known Issues
Current Task & Next Steps
🎯 PROJECT OVERVIEW
Domain: Multi-Agent Reinforcement Learning cho Search and Rescue (SAR) với UAV swarm

Core Task: 4 UAVs phối hợp tìm kiếm 10-36 victims trong disaster area có debris và danger zones, constraint battery (phải về trạm sạc)

Key Innovation:

Curriculum learning: EASY → MEDIUM → HARD (tăng dần map size)
Shared cooperative reward (tất cả agents nhận cùng reward)
Centralized Training Decentralized Execution (CTDE)
Auto-balanced vectorized training (n_envs=1-16)
Technology Stack:

Python 3.8+, PyTorch, PettingZoo, Gymnasium, NumPy, Matplotlib, Multiprocessing
🔬 RESEARCH GOALS
Paper 1: MAPPO vs MASAC vs MATD3 — Which is best for SAR coordination?

Method: 3 algorithms × 5 seeds × 3 curriculum stages × 3000 episodes
Paper 2: LLM-Generated vs Hand-Crafted Rewards

Baseline: BaselineReward v3.1 (hand-crafted, 14 components)
LLM: GPT-4/Claude generate reward code from natural language
📊 OVERALL STATUS
Phase Description Status Completion
Phase 1 Core Infrastructure (48 files) ✅ Complete 100%
Phase 2 MAPPO Algorithm ✅ Complete 100%
Phase 2b Auto-Balanced Vectorized Training ✅ Complete 100%
Phase 2c visualize_policy.py (load checkpoint → render) ⬜ PENDING 0%
Phase 3 MASAC & MATD3 Algorithms ⬜ Not Started 0%
Phase 4 LLM Reward Integration ⬜ Not Started 0%
Phase 5 Advanced Backends (PyBullet/Isaac) ⬜ Not Started 0%
Test Coverage: 26/26 core tests PASS (100%)

Current Immediate Goal:

Train MAPPO → lấy checkpoint → visualize_policy.py load checkpoint đó render policy lên màn hình/video

📐 KEY METRICS
Observation & Action Spaces
Metric Value Description
Actor Obs Dim 68 Local observation per UAV (n_stations=2)
Critic Obs Dim 554 Global observation (8×68 + 10 global features)
Action Dim 3 Continuous [vx, vy, vz] ∈ [-1, 1]
UAV States 5 ACTIVE/RETURNING/CHARGING/DEPLOYING/DISABLED
Actor Observation Layout (68 dims)
Slice Dims Features
[0:11] 11 Self: pos(3), vel(3), battery(1), state_onehot(4)
[11:19] 8 Stations (2): rel_x, rel_y, dist, occupancy
[19:28] 9 Teammates (3): dist, bearing, rel_alt
[28:40] 12 Obstacles (4): rel_x, rel_y, type_id
[40:65] 25 Victims (5): rel_x, rel_y, urgency, dist, found
[65:68] 3 Coverage: local_15m, local_30m, time_remaining
Critic Observation Layout (554 dims)
Slice Dims Features
[0:544] 544 8 UAVs × 68 (zero-padded)
[544:554] 10 Global fleet stats
Reward System
Type: Shared Cooperative (tất cả agents nhận cùng value)
Components: 14
Range: [-100, +100] per step (clipped)
Terminal Bonus: +200 base + up to +100 cap
Curriculum Stages
Stage Map Size Pressure (m²/UAV) Max Steps Advance Threshold
EASY 150×150m 5,625 300 cov≥70%, vic≥80%
MEDIUM 200×200m 10,000 350 cov≥65%, vic≥75%
HARD 250×250m 15,625 400 cov≥60%, vic≥70%
Checkpoint Format (.pt file)
text

{
"episode": int,
"update": int,
"total_episodes_done": int,
"actor_state_dict": OrderedDict, ← ActorNetwork weights
"critic_state_dict": OrderedDict, ← CriticNetwork weights
"actor_optimizer_state_dict": ...,
"critic_optimizer_state_dict": ...,
"ep_rewards": List[float],
"ep_coverage": List[float],
"ep_victims": List[float],
"curriculum_stage": int,
}
Checkpoint Locations
Local: results/mappo/{run_name}/checkpoints/
Kaggle: /kaggle/working/results/mappo/{run_name}/checkpoints/
Final: checkpoint_final.pt
Periodic: checkpoint_ep{episode:06d}.pt
🏗️ PROJECT ARCHITECTURE
text

┌─────────────────────────────────────────────────────────┐
│ TRAINING LOOP │
│ (train_mappo.py + MAPPOTrainer) │
└────────────────┬────────────────────────────────────────┘
│
┌────────┴────────┐
▼ ▼
ActorNetwork CriticNetwork
(68→3) (554→1)
│ │
└────────┬────────┘
▼
RolloutBuffer + GAE Compute
│
┌────────┴────────┐
▼ ▼
Single Env (1) VectorizedEnv (N)
│ │
└────────┬────────┘
▼
SARBaseEnv (Gymnasium)
│
┌────────┴────────┐
▼ ▼
LogicBackend ObservationBuilder
(Physics) (68/554 dims)
│
├── CoverageMap
├── FleetManager
├── MapGenerator
├── BaselineReward
└── FOVSensor + CommSensor
📁 DETAILED FILE STRUCTURE & DOCUMENTATION
Directory Tree (56+ files)
text

uav_swarm_pybullet/
├── config/ # 9 files
├── utils/ # 2 files
├── entities/ # 4 files
├── core/ # 3 files
├── sensors/ # 2 files
├── observation/ # 1 file
├── rewards/ # 1 file
├── env_setup/ # 5 files
│ └── backends/ # 2 files
├── visualization/ # 3 files
├── training/ # 3 files
│ └── algorithms/mappo/ # 6 files
├── tests/ # 26 test files
├── results/mappo/{run_name}/
│ ├── checkpoints/ # .pt files
│ ├── viz/ # PNG snapshots
│ └── training_curves.png
├── train_mappo.py # ✅ Main training entry
├── test_trainer_smoke.py # ✅ Smoke test
└── visualize_policy.py # ⬜ PENDING — cần tạo
📁 config/ — Configuration System (9 files)
config/**init**.py
Purpose: Central export point cho tất cả config classes
Exports: AppConfig, EnvConfig, UAVConfig, SensorConfig, VictimConfig, ObstacleConfig, DangerZoneConfig, RewardConfig, ObsConfig, TrainConfig, StageConfig, STAGE_EASY, STAGE_MEDIUM, STAGE_HARD, CURRICULUM_STAGES

config/config.py — AppConfig
Purpose: Master config orchestrator — single source of truth

Attributes:

env: EnvConfig — Map, time, fleet params
uav: UAVConfig — Physics, battery model
sensor: SensorConfig — FOV, comm, noise
victim: VictimConfig — Victim spawning
obstacle: ObstacleConfig — Debris generation
danger: DangerZoneConfig — Danger zone types & penalties
reward: RewardConfig — 14 reward components
obs: ObsConfig — Observation dims (68/554)
train: TrainConfig — RL hyperparameters
viz_mode: str — "2d" / "3d" / "none"
viz_3d_cfg: dict — 3D renderer config
Methods:

**post_init**() — Auto-sync obs.n_stations = env.n_stations, validate
apply_stage(stage: StageConfig) — Apply curriculum stage in-place (modify map_size, n_victims, max_steps, etc.)
map_diagonal (property) — sqrt(2) × map_size
grid_cell_size (property) — map_size / grid_size
save(path) — Serialize to JSON
load(path) — Restore from JSON
config/env.py — EnvConfig
Purpose: Environment parameters (map, time, fleet)

Key Attributes:

map_size: int = 100 — Map size in meters
grid_size: int = 100 — Coverage grid resolution (sync with map_size)
dt_seconds: float = 1.0 — Simulation timestep
max_steps: int = 600 — Max steps per episode
n_uav: int = 4 — Number of UAVs
n_stations: int = 2 — Number of charging stations
charge_radius_m: float = 3.0 — Charging activation radius
station_capacity: int = 2 — Max UAVs per station
min_station_spacing_m: float = 15.0
deterministic_eval: bool = False
eval_seed: int = 42
max_place_attempts: int = 500
min_object_spacing_m: float = 2.5
victim_clearance_m: float = 1.5
placement_relax_threshold: float = 0.7
placement_relaxed_spacing_m: float = 1.5
allow_partial_obstacles: bool = True
Backward compat properties: dt, charge_radius, min_station_spacing, victim_clearance

config/uav.py — UAVConfig
Purpose: UAV physics model + battery dynamics

Physics:

z_min_m: float = 3.0, z_max_m: float = 40.0
max_speed_xy_mps: float = 5.0, max_speed_z_mps: float = 2.0
collision_radius_m: float = 0.5
Battery drain rates (%/second):

drain_xy_pct_per_s: float = 0.10
drain_z_up_pct_per_s: float = 0.15
drain_z_down_pct_per_s: float = 0.03
drain_idle_pct_per_s: float = 0.05
charge_rate_pct_per_s: float = 1.5
Battery thresholds (%):

battery_return_pct: float = 10.0 — Auto-return
battery_ready_pct: float = 80.0 — Ready to deploy
battery_dead_pct: float = 0.0 — DISABLED
battery_warning_pct: float = 20.0
battery_critical_pct: float = 10.0
battery_emergency_pct: float = 5.0 — Forced RETURNING
Fleet:

reserve_ratio: float = 0.2, min_reserve: int = 2
Backward compat properties: z_min, z_max, drain_xy_max, battery_dead, battery_ready, charge_rate

config/sensor.py — SensorConfig
Purpose: Sensor models (FOV geometry + detection noise)

Attributes:

comm_range_m: float = 30.0 — V2V communication range
hfov_deg: float = 90.0 — Horizontal FOV angle
p_detect_base: float = 0.95 — Base detection prob
p_detect_decay: float = 0.04 — Decay with altitude
enable_noise: bool = True
motion_blur_coeff: float = 0.06
base_miss_rate: float = 0.03
Computed properties:

fov_tan — tan(hfov/2)
fov_radius_at_altitude — closure: altitude → radius
comm_range — alias for comm_range_m
Noise Pipeline:
P_final = P_altitude × env_factor × (1 - motion_penalty) × victim_factor × (1 - base_miss_rate)

config/entity.py — Entity Configs
VictimConfig:

n_victims_min: int = 5, n_victims_max: int = 20
injured_ratio_min/max: float = 0.4/0.7
injured_urgency_min/max: float = 4.0/5.0
mobile_urgency_min/max: float = 1.0/3.0
mobile_speed_min/max_mps: float = 0.2/0.4
mobile_dir_change_steps: int = 20
ObstacleConfig:

n_debris: int = 6
debris_width_min/max_m: float = 2.0/5.0 (diameter)
debris_height_min/max_m: float = 3.0/8.0
n_danger_total: int = 2
DangerZoneConfig:

heights: Dict[str, float] — {gas:3, fire:15, smoke:25, collapse:10, radiation:inf}
penalties: Dict[str, float] — {gas:-3, fire:-3, smoke:-1.5, collapse:-1, radiation:-5}
max_counts: Dict[str, int]
widths: Dict[str, Tuple] — (min_diameter, max_diameter) per type (diameters, not radii)
validate() — Check all dicts have same keys
danger_types (property) — List of types
config/reward.py — RewardConfig v3.1
Purpose: 14-component reward config

Components:

Field Default Description
r_coverage_delta +6.0 Per 1% coverage increase
r_victim_base +30.0 Base × (urgency/5)
r_battery_20 -1.0 Penalty <20%
r_battery_10 -5.0 Penalty <10%
r_battery_5 -20.0 Penalty <5%
r_battery_dead -100.0 One-time dead
r_collision_obstacle -30.0 One-time collision
r_proximity_1m -10.0 Per step <1m
r_proximity_2m -3.0 Per step <2m
r_proximity_3m -0.5 Per step <3m
proximity_penalty_cap -15.0 Cap proximity total
r_time_penalty -0.1 Per active UAV per step
r_terminal_base +200.0 Terminal bonus base
terminal_bonus_cap +100.0 Max terminal bonus
Caps:

step_penalty_cap: float = -30.0
step_reward_clip_min/max: float = -100.0/+100.0
episode_reward_clip_min/max: float = -800.0/+600.0 (logging only)
Shaping:

enable_distance_shaping: bool = True
distance_shaping_max_per_uav: float = 1.0
config/obs.py — ObsConfig + ObsSchemaConfig
ObsSchemaConfig (constants):

SELF_FEATURES = 11
STATION_FEATURES_PER = 4
TEAMMATE_FEATURES_PER = 3
OBSTACLE_FEATURES_PER = 3
VICTIM_FEATURES_PER = 5
COVERAGE_FEATURES = 3
GLOBAL_FEATURES = 10
ObsConfig:

n_obs_victims: int = 5
n_obs_obstacles: int = 4
n_tracked_teammates: int = 3
local_cov_small: int = 15
local_cov_large: int = 30
max_uav: int = 8
n_stations: int = None — Auto-synced từ EnvConfig
Computed properties:

actor_dim = 68 (11+8+9+12+25+3)
critic_dim = 554 (8×68+10)
validate() — Ensure n_stations not None
config/train.py — TrainConfig
Purpose: RL training hyperparameters

General:

n_seeds: int = 5, seeds: List = [42,123,456,789,1011]
total_episodes: int = 3000
eval_interval: int = 50, save_interval: int = 100
MAPPO Hyperparameters:

Param Default
mappo_rollout_length 2048
mappo_n_epochs 10
mappo_batch_size 256
mappo_clip_epsilon 0.2
mappo_gamma 0.99
mappo_gae_lambda 0.95
mappo_lr_actor 3e-4
mappo_lr_critic 1e-3
mappo_max_grad_norm 0.5
mappo_entropy_coeff 0.01
mappo_actor_hidden (256, 256)
mappo_critic_hidden (512, 256)
mappo_activation 'tanh'
mappo_use_layer_norm False
config/curriculum_config.py — Curriculum Stages
StageConfig attributes:

name, map_size, n_uav, n_victims_min/max, n_debris, n_danger_total
station_capacity, max_steps, min_episodes
advance_coverage, advance_victims — Thresholds to next stage
Computed properties:

map_area_m2, coverage_pressure_m2_per_uav, victim_density_per_1000m2
obstacle_density_per_1000m2, steps_per_m2, describe()
3 predefined stages:

Stage map_size victims debris danger max_steps advance
EASY 150 10-14 6 2 300 70%cov,80%vic
MEDIUM 200 18-24 10 4 350 65%cov,75%vic
HARD 250 28-36 15 7 400 60%cov,70%vic
Note: \_verify_stages() runs on import — validates victim density ~0.53/1000m²

📁 utils/ — Utility Functions (2 files)
utils/geometry.py
Purpose: 9 vectorized geometry functions (NumPy-optimized, ~10× speedup)

Functions:

dist_2d(pos1, pos2) → float — 2D Euclidean distance (XY only)
dist_3d(pos1, pos2) → float — 3D Euclidean distance
normalize_angle(angle) → float — Normalize to [-π, π]
compute_bearing(from_pos, from_vel, to_pos) → float — Relative bearing to target ∈ [-π, π]
check_los_2d(pos1, pos2, obstacles) → bool — Line-of-sight check (True = clear)
get_circle_cells(center, radius, grid_size, map_size) → ndarray(N,2) — Vectorized grid cells in circle (10× faster)
get_relative_position(from_pos, to_pos) → ndarray — [dx, dy, dz]
clip_position(pos, min_bounds, max_bounds) → ndarray — Boundary clamp
\_line_intersects_circle(p1, p2, center, radius) → bool — Segment-circle intersection (helper)
get_circle_cells_legacy(...) — Old loop version (kept for benchmarking)
utils/logger.py
Purpose: Episode and training logging

EpisodeLogger:

Attributes: episode_id, seed, total_reward, coverage_rate, victims_found, total_victims, episode_length, collision_obstacle, collision_uav, battery_deaths, danger_zone_entries, hot_swaps
log_step(rewards, coverage) — Update cumulative reward, max coverage
log_event(event_type) — Count events (collision_obstacle, victim_found, battery_death, etc.)
set_total_victims(n) — Set episode victim count
finalize() → Dict — Return JSON-safe metrics (coverage as %, success if coverage≥90%)
TrainingLogger:

Attributes: verbose, window_size, all_metrics, recent_rewards/coverage/success/lengths, converged, convergence_episode
log_episode(metrics) — Add episode, check convergence, print if verbose
get_stats(last_n) → Dict — mean/std/success_rate/converged
save(filepath) / load(filepath) — JSON persistence
Helper: compare_training_runs(runs, labels) — Compare multiple TrainingLogger instances

📁 entities/ — Game Objects (4 files)
entities/uav.py
UAVState (Enum):

ACTIVE — RL controls
RETURNING — Auto-fly to station (low battery)
CHARGING — Docked at station
DEPLOYING — Auto-fly from station to mission
DISABLED — Battery dead
UAV class:

Attributes:

id: int, pos: ndarray[3], vel: ndarray[3]
battery: float — [0, 100]
battery_pct (property) — Alias for battery (semantic clarity)
state: UAVState
target_station: ChargingStation | None
victims_found: int, battery_death: bool
steps_alive: int, distance_xy: float, distance_3d: float
cfg: AppConfig
Methods:

apply_action(action: ndarray) — Apply RL action [vx,vy,vz]∈[-1,1]³, only when ACTIVE
auto_navigate(target_pos) — Proportional controller, no overshoot (RETURNING/DEPLOYING)
update_battery(stations) — Drain or charge based on state
\_do_drain() — Drain battery proportional to velocity (×dt_seconds)
\_do_charge(stations) — Charge via target_station
get_fov_radius() → float — altitude × fov_tan
get_state_onehot() → ndarray(5) — One-hot [ACTIVE, RETURNING, CHARGING, DEPLOYING, DISABLED]
set_state(new_state) — State transition with validation
needs_charging() → bool — battery ≤ battery_return_pct
is_ready_to_deploy() → bool — battery ≥ battery_ready_pct
find_nearest_station(stations) → ChargingStation | None — Nearest with free slot
mark_disabled() — Force DISABLED state
Predicates: is_active(), is_returning(), is_charging(), is_deploying(), is_disabled(), is_operational()
to_dict() → Dict — JSON-safe serialization
entities/victim.py
BaseVictim (ABC):

Attributes: id, pos[x,y,0], urgency[1-5], is_found, found_at_step, found_by_uav
step(obstacles) — Abstract: update physics
update(step_count, obstacles) — Alias for step() (backend compatibility)
mark_found(step, uav_id) — One-time, calls \_on_found() hook
get_reward_value() → float — r_victim_base × (urgency/5)
is_detected_by(...) — Legacy detection (replaced by FOVSensor)
InjuredVictim (stationary):

urgency ∈ [4.0, 5.0], speed=0.0
step() — Empty (no movement)
\_on_found() — No action
MobileVictim (random walk):

Additional: speed[0.2-0.4 m/s], direction[angle], move_timer
urgency ∈ [1.0, 3.0]
step(obstacles) — Random walk with boundary clip + obstacle bounce
\_on_found() — Freeze: speed=0.0 immediately
\_check_obstacle_block(new_pos, obstacles) — Only Debris blocks (DangerZone does not)
entities/charging_station.py — ChargingStation
Attributes:

id: int, pos: ndarray[x,y,0]
capacity: int — From cfg.env.station_capacity
charge_radius: float, charge_rate: float
current_occupants: List[UAV], occupant_ids: Set[int] (O(1) lookup)
Methods:

is_full() → bool, is_available() → bool, is_occupied() → bool
in_range(uav_pos) → bool — dist_xy ≤ charge_radius AND z ≤ 0.5m
try_occupy(uav) → bool — Occupy slot (idempotent if already occupant)
release(uav) → bool — Free slot
charge(uav) → float — Charge one step: check range → occupy → add rate. Returns amount charged
force_release_all() — Episode reset
get_occupancy_ratio() → float — [0.0, 1.0]
has_uav(uav) → bool — O(1) check
get_occupant_ids() → List[int]
to_dict() → Dict
entities/obstacle.py
Debris (static obstacle):

Attributes: id, pos[x,y,0], height_3d, shape("circle"/"rectangle"/"polygon"), penalty
Shape-specific: radius | width, height_2d, rotation | vertices, polygon(Shapely)
in_zone_2d(pos_2d) → bool — Inside footprint (Shapely if available)
causes_collision(uav_pos) → bool — in_zone_2d AND uav.z < height_3d
blocks_los(pos1, pos2) → bool — Line intersects footprint
get_distance_to_edge(pos_2d) → float
\_get_fallback_radius() → float — Bounding circle radius (no Shapely)
\_create_rotated_box() → ShapelyPolygon
to_dict() → Dict
DangerZone (hazardous area):

Same structure as Debris + danger_type: str
Additional attributes: max_height (from cfg.danger.heights), penalty (from cfg.danger.penalties)
is_inside(uav_pos) → bool — in_zone_2d AND uav.z < max_height (semantic rename)
blocks_los(pos1, pos2) → bool — Only fire and smoke block LOS
get_sensor_modifier() → float — {smoke:0.40, fire:0.55, collapse:0.70, gas:0.85, radiation:0.95}
get_battery_modifier() → float — {fire:0.05, others:0.0}
to_dict() → Dict
📁 core/ — Core Systems (3 files)
core/coverage_map.py — CoverageMap v2.0
Attributes:

grid_size: int, map_size: float
grid: ndarray(bool, [GS,GS]) — Explored cells
timestamps: ndarray(int32, [GS,GS]) — Last scan step
first_scan: ndarray(int32, [GS,GS]) — First scan step (-1 if never)
scan_count: ndarray(int32, [GS,GS]) — Times scanned
Methods:

reset() — Clear all arrays
mark_explored(uav_pos, fov_radius, step) — Vectorized circle marking (10×), only update timestamp if newer
get_coverage_rate() → float — [0,1]
get_coverage_percent() → float — [0,100]
get_local_coverage(pos, radius) → float — Coverage in circle
get_staleness(pos, radius, step) → float — Mean age of cells (unexplored = max_steps)
get_staleness_normalized(pos, radius, step, decay_threshold=200) → float — [0,1]
get_freshness(pos, radius, step, decay=200) → float — 1 - staleness_normalized
get_coverage_with_decay(step, decay_threshold=200) → float — Only fresh cells
get_rescan_count(pos, radius) → float — Mean scan count
get_nearest_unexplored(pos, min_distance=0) → ndarray | None — O(N) scan
get_nearest_stale(pos, step, threshold=200) → ndarray | None — O(N) scan
get_stats(step) → Dict, to_dict(step) → Dict, get_grid_snapshot() → Dict
core/map_generator.py — MapGenerator v4.1
Key Fix v4.1: Config widths = diameters → radius = width / 2.0

Attributes: cfg: AppConfig, \_shapely_cache: Dict

Methods:

generate(n_victims_override, seed) → Dict — Full map: {stations, debris, danger_zones, victims, uav_spawns, seed, n_victims}
\_place_stations(rng) — Min spacing constraint, fallback to corners
\_place_debris(stations, rng) — 40% circle, 40% rect, 20% polygon; progressive relaxation; allow partial
\_place_danger_zones(existing_objects, rng) — 50% circle, 50% rect; respect max_counts
\_spawn_victims(n_victims, obstacles, danger_zones, rng) — Group spawning (80% near debris for injured, 40% for mobile)
\_find_valid_victim_pos(...) — Clearance + not inside danger zones
\_spawn_group(n, type, ...) — Spawn near debris with fallback random
get_uav_spawns(stations, n_total, rng) — Around stations, altitude=z_min
get_map_statistics(map_data) → Dict — Densities, clustering, coverage ratio
\_generate_random_convex_polygon(...) — Convex hull with irregularity
Validation helpers: \_check_station_clearance, \_check_obstacle_spacing, \_check_spacing_fallback
Geometry: \_get_bounding_radius, \_get_or_create_polygon (with cache), \_create_shapely_polygon
core/fleet_manager.py — FleetManager v2.0
Design: ENFORCE constraints, SUGGEST actions, RL CONTROLS behavior

Attributes:

n_total: int, n_reserve: int
all_uavs: List[UAV], stations: List[ChargingStation]
\_enforced_disables/returns/deploys: int — Counters
\_uav_return_locks: Dict[int, bool] — Hysteresis
\_episode_forced_returns/disables: int — Per-episode counters
Methods:

reset(uavs, stations) — Initialize for new episode
get_deployable_uavs() → List[UAV] — CHARGING + battery≥ready, sorted by battery desc
get_best_deployable(prefer_station, require_min_battery) → UAV | None
enforce_safety_constraints() → Dict — SILENT: battery=0→DISABLED, battery<5%→RETURNING
suggest_deployments(target_active) → List[UAV]
suggest_returns() → List[UAV]
step() → Dict — Main: enforce + suggest, return dict
get_episode_summary() → Dict — {forced_returns, disables} (log once per episode)
get_mission_priority_hints() → Dict — {operational_ratio, reserve_health, station_pressure}
get_spatial_awareness() → Dict — {active_positions, center_of_mass, spread_radius}
count_by_state() → Dict, get_battery_stats() → Dict, get_stats() → Dict
get_fleet_incentives() — Legacy, returns 0.0
is_episode_over() → bool — All UAVs DISABLED
📁 sensors/ — Sensor Models (2 files)
sensors/fov_sensor.py — FOVSensor
Attributes:

cfg, \_fov_tan, \_p_base, \_p_decay
\_n_victims=5, \_n_obstacles=4
\_enable_noise, \_motion_blur_coeff, \_base_miss_rate
\_rng: np.random.Generator
Methods:

set_seed(seed) — Reproducible eval
calculate_fov_radius(altitude) → float — altitude × fov_tan
calculate_detection_prob(altitude, uav_speed, env_factor, victim_factor) → float — 5-stage noise pipeline
\_get_env_factor(victim_pos, obstacles) → float — Danger zone modifier [0.4, 1.0]
\_get_victim_factor(victim) → float — Injured:1.15, Mobile:0.75-0.95
check_detected(uav, victim, obstacles) → bool — FOV → LOS → P(detect) → Bernoulli
scan_victims(uav, victims, obstacles) → ndarray(25) — Top-5 nearest in FOV, features: [rel_x, rel_y, dist, urgency, found]
scan_obstacles(uav, obstacles) → ndarray(12) — Top-4 nearest, features: [rel_x, rel_y, type_id]
sensors/comm_sensor.py — CommSensor
Attributes: cfg, \_n_tracked=3, \_comm_range, \_z_max

Methods:

scan(ego_uav, all_active_uavs) → ndarray(9) — Top-3 nearest in comm_range, features: [norm_dist, norm_bearing, norm_alt]
get_n_in_range(ego_uav, all_uavs) → int
get_teammates_in_range(ego_uav, all_uavs) → List[UAV] — Sorted by distance
📁 observation/ — Observation Builder (1 file)
observation/obs_builder.py
ObsResult:

actor_obs: Dict[int, ndarray] — {uav_id: obs[68]}
critic_obs: ndarray — [554]
ObservationBuilder:

Attributes:

coverage_map: CoverageMap, cfg: AppConfig
fov_sensor: FOVSensor, comm_sensor: CommSensor
actor_dim=68, critic_dim=554, \_max_uav=8
slices: List[slice] — Precomputed index slices
\_actor_bufs: Dict[int, ndarray] — Pre-allocated buffers per UAV
Methods:

build_actor_obs(uav, all_uavs, stations, victims, obstacles, step) → ndarray(68) — Build local obs for 1 UAV
build_all(all_uavs, stations, victims, obstacles, step) → ObsResult — All UAVs + critic in one call
\_write_self(obs, uav) — [0:11]: pos/vel normalized, battery/100, state_onehot[:4]
\_write_stations(obs, uav, stations) — [11:19]: relative pos, dist/diagonal, occupancy_ratio
\_write_teammates(obs, uav, all_uavs) — [19:28]: CommSensor.scan()
\_write_obstacles(obs, uav, obstacles) — [28:40]: FOVSensor.scan_obstacles()
\_write_victims(obs, uav, victims, obstacles) — [40:65]: FOVSensor.scan_victims()
\_write_coverage(obs, uav, step) — [65:68]: local_cov_15m, local_cov_30m, time_remaining
Critic construction:

UAV part: Stack sorted actor_obs for 8 UAVs (zero-pad disabled/missing)
Global part [544:554]: n_active, n_charging, n_disabled, n_alive (÷n_total), mean/std/min battery, global_coverage, victim_found_rate, time_remaining
📁 rewards/ — Reward Functions (1 file)
rewards/baseline_reward.py — BaselineReward v3.1
Purpose: 14-component hand-crafted reward (research-grade baseline for Paper 2)

Key Fixes vs v3.0:

BUG-31: Penalty cap ADDITIVE (not multiplicative)
BUG-32: Proximity cap scales with swarm size
BUG-33: Distance shaping DELTA-based with memory
BUG-34: Terminal bonus doesn't saturate
BUG-35: Battery urgency shaping → distance-to-station
Attributes:

cfg: AppConfig
\_battery_death_penalized: Set[int] — One-time tracking
\_collision_penalized: Set[int] — One-time tracking
\_prev_min_dist: Dict[int, float] — Distance memory for delta shaping
Cached reward params from RewardConfig
Methods:

reset() — Clear sets + distance memory (call on episode start)
compute(uavs, victims, obstacles, coverage_map, fleet_manager, newly_found, prev_coverage, step, done, stations) → Dict — Global reward (all 14 components)
compute_per_uav(uav, newly_found_by_uav, all_uavs, ...) → Dict — Per-agent shared reward
\_apply_penalty_cap(components, cap) → Dict — Additive adjustment (preserve relative importance)
\_terminal_bonus(coverage_rate, victims, step, uavs) → float — Base+bonus: 60%cov+20%vic+10%time+10%battery
\_delta_shaping_fleet(uavs, victims) → float — Fleet-level approach reward
\_delta_shaping_single(uav, victims, unfound) → float — Per-UAV delta shaping with memory
\_battery_rewards(uavs, stations) → Tuple[float, float] — penalty + dead
\_collision_reward(uavs, obstacles) → float — One-time per UAV
\_danger_reward(uavs, obstacles) → float — Per-step inside zone
get_component_names() → List[str] — 14 names for logging
summarize(reward_dict) → str — Compact log string
Module-level functions (stateless):

\_coverage_delta_reward(prev, cur, weight), \_victim_found_reward(newly_found, r_base)
\_battery_penalty_single(uav, reward_cfg, uav_cfg), \_battery_urgency_shaping(uav, stations, map_size)
\_proximity_reward(active_uavs, ...), \_proximity_reward_single(uav, active_uavs, ...)
\_assert_no_nan_inf(value, label) — Sanity check
📁 env_setup/ — Environments (5 files)
env_setup/base_env.py — SARBaseEnv
Inheritance: gymnasium.Env

Attributes:

cfg, backend: LogicBackend, \_reward_fn: BaselineReward
\_obs_builder: ObservationBuilder, \_map_gen: MapGenerator
\_renderer: Visualizer2D | Visualizer3D | None
\_step_count, \_prev_coverage, \_episode_reward_sum, \_episode_id
\_ep_logger: EpisodeLogger, \_ep_start_time
\_step_rewards_history: List
Spaces:

observation_space: Box(68,) — float32
action_space: Box(3,) ∈ [-1,1] — float32
Methods:

reset(seed, options) → (Dict[int→ndarray68], Dict) — Generate map, reset backend/reward/obs_builder. Info contains global_obs: ndarray(554)
step(actions: Dict[int→ndarray3]) → (obs_dict, rewards_dict, done, truncated, info) — Critical order: apply→physics→world→step_count→check_done→reward→obs. Info contains global_obs: ndarray(554)
render() → ndarray | None — Delegate to renderer
close() — Cleanup renderer
\_build_obs_dict(uavs, stations, victims, obstacles) → Tuple[Dict, ndarray] — Returns (actor_obs_int_keys, critic_obs)
\_check_done(coverage, victims, uavs) → str | None — Returns "coverage"/"victims"/"disabled"/None
\_log_step(...), \_print_episode_summary(...), \_log_extreme_episode(...)
make(cls, cfg, render_mode, ...) → SARBaseEnv — Factory
Properties: n_agents, active_uav_ids, alive_uav_ids, step_count, coverage_rate

Step Flow (critical order):

text

apply_actions → step_physics → step_world → step_count++ →
CHECK done FIRST → compute_rewards(is_terminal) → build_obs → return
env_setup/sar_pettingzoo_env.py — SARPettingZooEnv
Inheritance: pettingzoo.ParallelEnv

Attributes:

\_base_env: SARBaseEnv
possible_agents: List[str] — ["uav_0", "uav_1", "uav_2", "uav_3"] (fixed)
agents: List[str] — Current active (updated each step)
\_observation_spaces, \_action_spaces: Dict[str, Space]
Methods:

reset(seed, options) → (obs_dict_str, infos_dict_str) — Converts int keys → str keys. Each agent info has global_obs: ndarray(554)
step(actions_str_keys) → (obs, rewards, terminations, truncations, infos) — str→int→base_env→str. All agents share same done/truncated. Info has global_obs
observation_space(agent) → Box(68,), action_space(agent) → Box(3,)
render(), close()
unwrapped (property) — Return SARBaseEnv
Factory functions:

make_parallel_env(cfg, **kwargs) → SARPettingZooEnv
make_aec_env(cfg, **kwargs) — Wraps parallel_to_aec
MAPPO access pattern:

Python

obs, infos = env.reset(seed=42)
global_obs = infos['uav_0']['global_obs'] # ndarray(554)
obs, rews, terms, truncs, infos = env.step(actions)
global_obs = infos['uav_0']['global_obs'] # ndarray(554)
env_setup/vec_env.py — VectorizedEnv
Architecture: N worker processes via multiprocessing.Pipe, start method = spawn (CUDA-safe)

env_worker(pipe, config, seed) — Worker function:

Creates SARPettingZooEnv in isolated process
Maintains cache: last_obs_array[n_agents,68], last_global_obs[554], last_info
Commands: "reset" → send (obs, global_obs, info); "step" → step env, auto-reset on done, send (obs, global_obs, rewards, done, info); "close" → exit
Error handling: None sentinel on crash
VectorizedEnv class:

Attributes: n_envs, n_agents=4, obs_dim=68, global_obs_dim=554, action_dim=3, pipes, processes

Methods:

**init**(config, n_envs=8, start_seed=0) — Spawn workers, verify alive
reset() → (obs_batch[n_envs,n_agents,68], global_obs_batch[n_envs,554])
step(actions_batch[n_envs,n_agents,3]) → (obs, global_obs, rewards[n_envs,n_agents], dones, infos)
close() — Graceful shutdown (send close, join 3s, terminate if needed)
**del**() — Cleanup on destroy
Performance: ~2× speedup with n_envs=6, optimal n_envs=4-8

env_setup/backends/base_backend.py — BaseBackend (ABC)
Abstract methods:

reset(map_data: Dict), apply_actions(actions: Dict[int, ndarray])
step_physics(), step_world(), get_state() → Dict
env_setup/backends/logic_backend.py — LogicBackend
Purpose: Pure Python physics (~1000 steps/s)

Attributes:

cfg, uavs, victims, stations, obstacles
\_cov_map: CoverageMap, \_fleet_mgr: FleetManager, \_fov_sensor: FOVSensor
\_step_count: int
Methods:

reset(map_data) — Build all entities from map_data dict. Sets FOVSensor seed. Resets cov_map + fleet_mgr
apply_actions(actions) — ACTIVE: apply_action(); RETURNING/DEPLOYING: auto_navigate()
step_physics() — Battery update for all UAVs
step_world() — fleet_mgr.step() → victim.update(obstacles) → mark_explored → check_detected → mark_found
get_state() → Dict — {uavs, victims, stations, obstacles, coverage_map, fleet_manager}
Builders: \_build_stations, \_build_obstacles, \_build_victims, \_build_uavs (uses map_data["uav_spawns"])
Known issue: All UAVs spawn ACTIVE (no reserve pool initially)

📁 visualization/ — Renderers (3 files)
visualization/renderer_factory.py
Purpose: Factory pattern

create_renderer(cfg, render_mode, viz_mode) → Visualizer2D | Visualizer3D

"2d" → Visualizer2D
"3d" → Visualizer3D (fallback to 2D if ImportError)
visualization/visualizer2d.py — Visualizer2D
Purpose: 2D Matplotlib renderer (~50ms/frame), figure reuse optimization

Attributes: cfg, render_mode, map_size, \_fig, \_ax_map, \_ax_info, \_initialized

Layout: [3:1] ratio — Map (75%) + Info panel (25%)

Methods:

render(uavs, victims, obstacles, stations, cov_map, step, metrics) → ndarray | None — Main render, calls all draw methods
close() — plt.close(fig)
\_init_figure() — Create figure once (1 time)
\_setup_map_axes() — Grid, limits, labels
\_draw_coverage(cov_map) — Yellow heatmap (grid, imshow, no transpose)
\_draw_obstacles(obstacles) — Dispatch debris vs danger_zone
\_draw_debris(debris) — Brown hatch pattern, shape-aware
\_draw_danger_zone(zone) — Colored fill + dashed border, type label
\_draw_stations(stations) — Blue rectangle, occupancy text, charge radius circle
\_draw_victims(victims) — Orange X (missing) / Green circle (found)
\_draw_uavs(uavs) — White circle+colored border, FOV circle, velocity arrow, battery bar, state label
\_draw_battery_bar(x, y, battery, color) — Green→orange→red gradient
\_draw_map_title(step, cov_map) — Coverage % in title
\_draw_info_panel(uavs, victims, cov_map, step, metrics) — Sections: MISSION, UAV STATUS, LEGEND
\_get_shape_patch(obs, x, y, \*\*kwargs) — Shapely-aware patch creation
\_to_rgb_array() → ndarray — buffer_rgba or tostring_rgb fallback
Color scheme:

UAV: ACTIVE=Blue, RETURNING=Orange, CHARGING=Green, DEPLOYING=Purple, DISABLED=BlueGrey
Victims: Orange X (missing), Green D (found)
Coverage: Light yellow (#FFF9C4)
Station: Dark blue (#1565C0)
visualization/visualizer3d.py — Visualizer3D
Purpose: 3D Matplotlib renderer (~400ms/frame, demo only)

Attributes: cfg, render_mode, \_map_size, \_z_max

Layout: [3:1] — 3D scene (75%) + Dashboard (25%)

Methods:

render(uavs, victims, obstacles, stations, cov_map, step) → ndarray — New figure every frame (expensive!)
\_make_figure(), \_make_axes(fig) → (ax3, ax_dash)
\_draw_scene(ax3, ...) — Orchestrates all 3D draw calls
\_scene_boundary(ax), \_scene_coverage(ax, cov_map) — Batch cell rendering
\_scene_obstacles(ax, obstacles), \_draw_debris(ax, d), \_draw_danger(ax, z)
\_scene_stations(ax, stations), \_scene_victims(ax, victims)
\_scene_uavs(ax, uavs) — Scatter + quiver + drop lines
\_scene_fov(ax, uavs) — Cone meshes
\_scene_title(ax, step, cov_map, victims)
\_draw_dashboard(ax, uavs, victims, cov_map, step, stations) — Progress bars, battery bars per UAV
\_to_rgb(fig) → ndarray — 4 fallback methods (buffer_rgba → PIL → tostring_rgb → tostring_argb)
Geometry helpers (module-level):

\_circle_xy(cx, cy, r, n=32), \_cylinder_faces(cx, cy, r, z0, z1, n=20)
\_box_faces(cx, cy, w, d, z0, z1), \_cone_faces(apex, r_base, n=16)
📁 training/ — Training Pipeline (3 files + mappo/)
training/curriculum.py — CurriculumManager
StageStats (dataclass):

stage_name, episodes_done
coverage_list, victims_list, reward_list: List
Properties: avg_coverage, avg_victims, avg_reward (last 50 entries)
CurriculumManager:

Attributes: stages: List[StageConfig], \_stage_idx: int = 0, \_stats: List[StageStats]

Properties: current_stage, current_stats, stage_idx, is_final_stage, total_episodes

Methods:

update(coverage, victims_rate, reward) — Append to current stage stats (NOTE: param names are coverage, victims_rate, reward — not victims_found_rate)
should_advance() → bool — episodes≥min AND avg_cov≥threshold AND avg_vic≥threshold
advance() — Increment stage_idx, log
apply_to_config(cfg) — Delegates to cfg.apply_stage(current_stage)
get_status() → Dict, print_status() — Console output with colored thresholds
training/curriculum_trainer.py — CurriculumTrainer
Purpose: Phase 1 placeholder — random policy training loop (NOT used for MAPPO)

Attributes: cfg, curriculum: CurriculumManager, env, render_every, save_gif

Methods:

train(total_episodes) → Dict — Main loop with history tracking
\_build_env() → SARBaseEnv — Create/recreate env
\_run_episode(episode_num, render_frames) → Dict — Random actions, return {coverage, victims_rate, reward, steps}
\_sample_actions(n_uav) → Dict[int, ndarray] — Uniform [-1,1]³
\_save_episode_visualization(frames, episode, stage_name) — PNG first+last frame
\_save_gif(frames, path) — Optional GIF (requires Pillow)
\_plot_training_curves(history, episode, final) — 4 panels (coverage, victims, reward, stage bar)
\_print_summary(history)
📁 training/algorithms/mappo/ — MAPPO Implementation (6 files)
training/algorithms/mappo/networks.py — MLP Foundation
Functions:

orthogonal_init(layer, gain=sqrt(2)) → layer — Orthogonal weights, bias=0
get_parameter_count(model) → int
print_network_summary(model, name)
MLP (nn.Module):

**init**(input_dim, hidden_dims, output_dim, activation='tanh', use_layer_norm=False, output_activation='none')
Builds: dims = [input_dim] + list(hidden_dims) → Linear+Activation(+LayerNorm) per layer → output Linear
Hidden layers: orthogonal gain=√2; Output: orthogonal gain=0.01
forward(x) → [batch, output_dim]
training/algorithms/mappo/actor.py — ActorNetwork
Architecture: obs[68] → MLP(68→256→256→3) + learnable log_std[3]

Attributes: obs_dim=68, action_dim=3, mean_net: MLP, log_std: nn.Parameter

Methods:

forward(obs) → (mean[batch,3], std[batch,3]) — std = exp(log_std).expand_as(mean)
get_action(obs, deterministic=False) → (action[batch,3], log_prob[batch]) — Sample Normal(mean,std), NO clamping in this version
evaluate_actions(obs, actions) → (log_prob[batch], entropy[batch]) — For PPO update
get_log_std() → Tensor, set_log_std(value: float) — Exploration control
Design: State-independent std, learnable, shared weights across agents

training/algorithms/mappo/critic.py — CriticNetwork
Architecture: global_obs[554] → MLP(554→512→256→1)

Attributes: global_obs_dim=554, value_net: MLP

Methods:

forward(global_obs) → [batch, 1]
get_value(global_obs) → [batch] — Squeezed
compute_loss(global_obs, returns) → Tensor — MSE scalar
compute_value_metrics(global_obs, returns) → Dict — {value_loss, explained_variance, mean/std pred/target}
Module helpers: test_critic_accuracy(...), initialize_critic_for_env(...)

training/algorithms/mappo/buffer.py — RolloutBuffer
Attributes:

capacity, n_agents, obs_dim, global_obs_dim, action_dim
gamma=0.99, gae_lambda=0.95, ptr=0
Arrays: observations[cap,n_agents,68], global_obs[cap,554], actions[cap,n_agents,3], rewards[cap,n_agents], values[cap,n_agents], log_probs[cap,n_agents], dones[cap], advantages[cap,n_agents], returns[cap,n_agents]
Methods:

add(obs, global_obs, actions, rewards, values, log_probs, done) — RuntimeError if overflow
compute_gae(last_values[n_agents], last_done: bool) — Vectorized backward pass, actual_length=min(ptr,cap), normalize advantages
get_batches(batch_size) → Iterator[Dict] — Flatten [actual×n_agents,*], permute, yield batches with keys: obs, global_obs, actions, old_log_probs, advantages, returns
clear() — ptr=0
get_stats() → Dict — buffer_size, fill, mean reward/value/advantage/return
training/algorithms/mappo/trainer.py — MAPPOTrainer
\_EnvWrapper (internal class):

Unified interface for single (SARPettingZooEnv) or vectorized (VectorizedEnv)
reset() → (obs[n_envs,n_agents,68], global[n_envs,554])
step(actions[n_envs,n_agents,3]) → (obs, global, rews, dones, infos)
render() — Single env only, close()
MAPPOTrainer:

Attributes:

config, n_envs, run_name, device
actor: ActorNetwork, critic: CriticNetwork
actor_opt, critic_opt: Adam
buffer: RolloutBuffer (capacity = rollout_length × n_envs)
ep_rewards, ep_lengths, ep_coverage, ep_victims: deque(maxlen=100)
total_episodes_done, total_steps, update_count
output_dir, checkpoint_dir, viz_dir — Kaggle-aware paths
Methods:

train(total_episodes, curriculum_manager, seed, log_every_n_eps, viz_every_n_eps, checkpoint_every_n_eps) — Main loop with tqdm
\_rollout(env, pbar, max_episodes) → Dict — Collect experience, return metrics
Batch inference: flatten [n_envs×n_agents, obs_dim] → actor → unflatten
Per-env tracking: coverage_rate from infos[ei].get('uav_0', {}).get('coverage_rate', 0.0)
On done: extract coverage/victims/done_reason from infos, pbar.update(1)
GAE bootstrap: mean of last_vals across envs
\_update() → Dict — n_epochs × minibatches: PPO clip loss + entropy; critic MSE; grad clip
plot_training_curves(save_path) — 4-panel: reward/coverage/victims/episode_length with rolling mean
\_log_detail(pbar, rollout, train, elapsed, fps, curriculum_manager) — pbar.write detailed stats
\_save_viz(env, episode) — Single env only, save PNG to viz_dir
save_checkpoint(episode, curriculum_manager, tag) → None — .pt file with full state
load_checkpoint(path) → int — Load actor/critic/optimizers, return episode number
\_print_init(), \_print_final(elapsed, metrics)
Checkpoint content:

Python

{
"episode": int, "update": int, "total_episodes_done": int,
"actor_state_dict": ..., "critic_state_dict": ...,
"actor_optimizer_state_dict": ..., "critic_optimizer_state_dict": ...,
"ep_rewards": List, "ep_coverage": List, "ep_victims": List,
"curriculum_stage": int,
}
📁 Root Files
train_mappo.py ✅ COMPLETE
Purpose: CLI entry for HARD stage MAPPO training (no curriculum)

Key functions:

auto_compute_config(max_steps, n_envs, n_uav, batch_size_hint, safety_factor) → AutoConfig — Compute rollout_length ≥ max_steps × safety_factor, aligned to 64
parse_args() — CLI args: total-episodes, seed, device, run-name, n-envs, max-steps, map-size, batch-size, safety-factor, n-epochs, lr-actor, lr-critic, log-interval, viz-interval, checkpoint-interval
main() — Apply STAGE_HARD, auto-compute config, create MAPPOTrainer, call train()
Key logic:

Always uses STAGE_HARD (no curriculum in this script)
viz_mode = "2d" enabled
Rollout assertion: rollout_length ≥ max_steps
Viz interval defaults to 5 × log_interval
Prints summary with checkpoint path at end
test_trainer_smoke.py ✅ COMPLETE
Purpose: Quick smoke test

max_steps=50, rollout_length=200, 5 updates, EASY stage
Checks: init, rollout, update, no crash
visualize_policy.py ⬜ PENDING — CHƯA TẠO
Purpose: Load MAPPO checkpoint → run policy → render visualization

Workflow cần implement:

Parse args: --checkpoint path.pt, --episodes N, --seed, --stage, --viz-mode 2d|3d, --save-video, --deterministic
Load AppConfig (recreate from checkpoint stage info)
Create ActorNetwork(obs_dim=68, action_dim=3), load actor_state_dict
Create SARPettingZooEnv(cfg, render_mode="rgb_array")
Run episodes:
obs, info = env.reset(seed=seed)
Loop: actor.get_action(obs_tensor, deterministic=True) → env.step(actions) → env.render() → collect frames
Save frames as PNG sequence or MP4/GIF
Print metrics: coverage, victims, reward per episode
Key details để implement:

Actor input: stack obs_dict values → [n_agents, 68] tensor
Deterministic mode: actor.get_action(obs, deterministic=True)
Render: env.render() returns ndarray[H,W,3] khi render_mode="rgb_array"
Checkpoint load: torch.load(path, map_location=device) → actor.load_state_dict(ckpt["actor_state_dict"])
Stage recovery: ckpt["curriculum_stage"] → CURRICULUM_STAGES[stage_idx] → cfg.apply_stage(...)
🔄 EXECUTION FLOWS
Training Flow
text

train_mappo.py → auto_compute_config → MAPPOTrainer
→ \_rollout: [n_envs×n_agents, 68] → actor → actions → env.step → buffer.add
→ compute_gae → \_update: PPO clip + critic MSE
→ log/checkpoint/viz every N episodes
Visualization Flow (PENDING)
text

visualize_policy.py → load checkpoint → ActorNetwork
→ SARPettingZooEnv(render_mode="rgb_array")
→ episode loop: obs → actor(deterministic) → env.step → env.render() → frames
→ save PNG/MP4/GIF + print metrics
Single Step Flow
text

env.step(actions)
→ LogicBackend.apply_actions() → step_physics() → step_world()
(fleet_mgr.step → victim.update → mark_explored → check_detected)
→ CHECK done FIRST
→ BaselineReward.compute_per_uav() → ObservationBuilder.build_all()
→ return (obs, rewards, done, truncated, info + global_obs)
📊 TRAINING RESULTS
Baselines (Random Policy)
Stage Coverage Victims Reward
EASY 55%±11% 53%±19% +150±200
MEDIUM 41%±9% 44%±17% +80±180
HARD 32%±8% 36%±15% +30±160
MAPPO Results (50 episodes, HARD stage)
Config FPS Reward Coverage Victims
n_envs=1 52.8 +124.3 55.0% 55.8%
n_envs=6 105.4 +158.2 50.3% 55.1%
Target (3000 episodes)
Stage Coverage Victims Reward
EASY 82-88% 85-88% +420-450
MEDIUM 68-72% 70-75% +300-350
HARD 58-62% 60-65% +200-250
⚠️ KNOWN ISSUES
Issue Root Cause Status Workaround
UAV spawn all ACTIVE UAV default state=ACTIVE Known Reserve pool empty initially
Reward positive with random Coverage delta > time penalty Not a bug Correct behavior
3D viz slow (~2-5 FPS) Matplotlib 3D overhead Workaround viz_mode="none" for training
visualize_policy.py missing Not yet created PENDING Next task
base_env step info: some fields may be missing Defensive coding needed Minor Use .get() with defaults
🎯 CURRENT TASK & NEXT STEPS
🔴 Immediate: visualize_policy.py
Goal: Load trained MAPPO checkpoint → render policy visually

Required capabilities:

CLI: --checkpoint path/to/checkpoint_final.pt
Reconstruct config (use AppConfig + apply correct stage from checkpoint)
Load actor weights: actor.load_state_dict(ckpt["actor_state_dict"])
Run deterministic episodes: actor.get_action(obs, deterministic=True)
Render each step: env.render() → RGB frames
Save output: PNG sequence + training curves overlay + optional MP4/GIF
Print episode metrics: coverage, victims, reward, done_reason
Interface:

Python

# Load

actor = ActorNetwork(obs_dim=68, action_dim=3, hidden_dims=(256,256))
ckpt = torch.load(checkpoint_path, map_location=device)
actor.load_state_dict(ckpt["actor_state_dict"])
actor.eval()

# Env

cfg = AppConfig()
stage_idx = ckpt.get("curriculum_stage", 2) # default HARD
cfg.apply_stage(CURRICULUM_STAGES[stage_idx])
cfg.viz_mode = "2d"
env = SARPettingZooEnv(cfg, render_mode="rgb_array")

# Episode

obs*dict, info = env.reset(seed=seed)
obs_tensor = torch.FloatTensor(np.stack([obs_dict[f"uav*{i}"] for i in range(4)]))
actions, _ = actor.get_action(obs_tensor, deterministic=True)
actions_dict = {f"uav_{i}": actions[i].numpy() for i in range(4)}
obs_dict, rews, terms, truncs, info = env.step(actions_dict)
frame = env.render() # ndarray[H,W,3]
🟡 After Visualization: Phase 3
Full training: 5 seeds × 3000 episodes × n_envs=6
MASAC implementation: training/algorithms/masac/
MATD3 implementation: training/algorithms/matd3/
Statistical comparison (Wilcoxon tests, learning curves)
✅ PROJECT STATUS SUMMARY
text

Phase 1: Core Infrastructure ✅ 100% (48 files, 26/26 tests)
Phase 2: MAPPO Implementation ✅ 100% (trainable, stable)
Phase 2b: Auto-Balanced Vec Env ✅ 100% (n_envs=1-16)
Phase 2c: visualize_policy.py ⬜ PENDING (next task)
Phase 3: MASAC/MATD3 ⬜ Not started
Phase 4: LLM Reward ⬜ Not started
Phase 5: Advanced Backends ⬜ Not started

Total files: 56+
Test coverage: 26/26 PASS
Training time (est.): 4 hours (n_envs=6, 3000 eps, GPU)
Next action: Create visualize_policy.py
