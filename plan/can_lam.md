🚁 SAR UAV SWARM — PROJECT STATE SNAPSHOT v4.0
Dựa trên code thực tế | MAPPO Phase 2 Complete | HARD Stage Fixed | No Curriculum

1. MỤC TIÊU DỰ ÁN
   Task: 4 UAV tự động phối hợp tìm kiếm nạn nhân trong khu vực thảm họa 250×250m.

Research Plan (Paper 1):

Phase 2 (Hiện tại): Train MAPPO trên HARD stage cố định, 5 seeds × 3000 eps/seed
Phase 3 (Tiếp theo): Train MASAC + MATD3 cùng cấu hình → so sánh thống kê (Wilcoxon test)
Paper 2: LLM-generated reward vs BaselineReward v3.1 (hand-crafted)

Không dùng curriculum learning. curriculum_manager=None trong trainer.train().

2. THÔNG SỐ HỆ THỐNG (Verified từ code)
   Observation Space — Actor: 68 dims
   Slice Dims Nội dung Normalization
   [0:11] 11 Self: pos(3), vel(3), battery(1), state_onehot(4) — chỉ 4 dims đầu pos/map_size, vel/max_speed, bat/100
   [11:19] 8 Stations (2×4): rel_x, rel_y, dist, occupancy_ratio rel/map_size, dist/map_diagonal
   [19:28] 9 Teammates (3×3): norm_dist, norm_bearing, norm_alt CommSensor.scan()
   [28:40] 12 Obstacles (4×3): rel_x/fov_r, rel_y/fov_r, type_id FOVSensor.scan_obstacles()
   [40:65] 25 Victims (5×5): rel_x, rel_y, dist, urgency/5, is_found FOVSensor.scan_victims()
   [65:68] 3 Coverage: local_15m, local_30m, time_remaining [0,1]
   Observation Space — Critic: 554 dims
   text

[0:544] = 8 UAVs × 68 (zero-padded, sorted by uav.id)
[544:554] = 10 global: n_active/n, n_charging/n, n_disabled/n, n_alive/n,
bat_mean, bat_std, bat_min (÷100),
global_coverage, victims_found_rate, time_remaining
Action Space
text

Actor output: action[4] = [vx, vy, vz, land]

- [vx, vy, vz] ~ Normal(mean[3], exp(log_std)[3]) — Gaussian continuous
- [land] ~ Bernoulli(sigmoid(logit)) — discrete {0,1}

Env API (PettingZoo): Box(3,) — land được xử lý trong LogicBackend, không expose
Key Numbers — HARD Stage
Param Value
map_size 250m
max_steps 400
n_uav 4
n_stations 2
n_victims 28–36
n_debris 15
n_danger_total 7
actor_dim 68
critic_dim 554
dt_seconds 1.0
charge_radius 3.0m
station_capacity 2
battery_emergency_pct 30.0% (force RETURNING) 3. CẤU TRÚC THƯ MỤC
text

uav_swarm_pybullet/
├── config/
│ ├── **init**.py
│ ├── config.py # AppConfig (master)
│ ├── env.py # EnvConfig
│ ├── uav.py # UAVConfig
│ ├── sensor.py # SensorConfig
│ ├── entity.py # VictimConfig, ObstacleConfig, DangerZoneConfig
│ ├── reward.py # RewardConfig
│ ├── obs.py # ObsConfig
│ ├── train.py # TrainConfig
│ └── curriculum_config.py # StageConfig, STAGE_HARD (dùng để apply_stage)
├── utils/
│ ├── geometry.py # 9 hàm vectorized
│ └── logger.py # EpisodeLogger, TrainingLogger
├── entities/
│ ├── uav.py # UAV, UAVState
│ ├── victim.py # InjuredVictim, MobileVictim
│ ├── charging_station.py # ChargingStation
│ └── obstacle.py # Debris, DangerZone
├── core/
│ ├── coverage_map.py # CoverageMap
│ ├── map_generator.py # MapGenerator v4.1
│ └── fleet_manager.py # FleetManager
├── sensors/
│ ├── fov_sensor.py # FOVSensor
│ └── comm_sensor.py # CommSensor
├── observation/
│ └── obs_builder.py # ObservationBuilder, ObsResult
├── rewards/
│ └── baseline_reward.py # BaselineReward v3.1 (16 components)
├── env_setup/
│ ├── base_env.py # SARBaseEnv (Gymnasium)
│ ├── sar_pettingzoo_env.py # SARPettingZooEnv (PettingZoo wrapper)
│ ├── vec_env.py # VectorizedEnv (multiprocessing)
│ └── backends/
│ ├── base_backend.py # BaseBackend (ABC)
│ └── logic_backend.py # LogicBackend (~1000 steps/s)
├── visualization/
│ ├── renderer_factory.py
│ ├── visualizer2d.py # ~50ms/frame
│ └── visualizer3d.py # ~400ms/frame (demo only)
├── training/
│ ├── curriculum.py # CurriculumManager (không dùng hiện tại)
│ └── algorithms/mappo/
│ ├── **init**.py
│ ├── networks.py # MLP
│ ├── actor.py # ActorNetwork (hybrid: 68→4)
│ ├── critic.py # CriticNetwork (554→1)
│ ├── buffer.py # RolloutBuffer + GAE
│ └── trainer.py # MAPPOTrainer + \_EnvWrapper
├── train_mappo.py # CLI entry point
└── test_trainer_smoke.py 4. CHI TIẾT TỪNG FILE
📁 config/
config/config.py — AppConfig
Mục đích: Master config, tổ hợp tất cả sub-config.

Attributes:

text

env: EnvConfig
uav: UAVConfig
sensor: SensorConfig
victim: VictimConfig
obstacle: ObstacleConfig
danger: DangerZoneConfig
reward: RewardConfig
obs: ObsConfig
train: TrainConfig
viz_mode: str = "2d"
Methods:

**post_init**() — Auto-sync obs.n_stations = env.n_stations
apply_stage(stage: StageConfig) — Ghi đè map_size, max_steps, n_victims, n_debris, n_danger_total
map_diagonal (property) — sqrt(2) × map_size
grid_cell_size (property) — map_size / grid_size
save(path) / load(path) — JSON serialization
config/env.py — EnvConfig
Attributes:

text

map_size: int = 100 → HARD override: 250
grid_size: int = 100 → luôn = map_size
dt_seconds: float = 1.0
max_steps: int = 600 → HARD override: 400
n_uav: int = 4
n_stations: int = 2
charge_radius_m: float = 3.0 → alias: charge_radius
station_capacity: int = 2
min_station_spacing_m: float = 15.0
max_place_attempts: int = 1000
min_object_spacing_m: float = 2.5
victim_clearance_m: float = 1.5
placement_relax_threshold: float = 0.7
placement_relaxed_spacing_m: float = 1.5
allow_partial_obstacles: bool = True
deterministic_eval: bool = False
eval_seed: int = 42
Properties (backward compat): dt, charge_radius, min_station_spacing

config/uav.py — UAVConfig
Physics:

text

z_min_m: float = 3.0
z_max_m: float = 40.0
max_speed_xy_mps: float = 5.0
max_speed_z_mps: float = 2.0
collision_radius_m: float = 0.5
Battery drain (% per SECOND × dt_seconds = per-step):

text

drain_xy_pct_per_s: float = 0.10
drain_z_up_pct_per_s: float = 0.15
drain_z_down_pct_per_s: float = 0.03
drain_idle_pct_per_s: float = 0.05
charge_rate_pct_per_s: float = 1.5
Battery thresholds (%):

text

battery_return_pct: float = 10.0 → auto-return ≤ 10%
battery_ready_pct: float = 80.0 → deploy khi ≥ 80%
battery_dead_pct: float = 0.0 → DISABLED
battery_warning_pct: float = 20.0 → r_battery_20 trigger
battery_critical_pct: float = 10.0 → r_battery_10 trigger
battery_emergency_pct: float = 30.0 → r_battery_5 trigger + force RETURNING
⚠️ QUAN TRỌNG: battery_emergency_pct = 30.0 — FleetManager force RETURNING khi battery < 30%. Reward urgency shaping cũng trigger ở 30%.

Fleet policy:

text

reserve_ratio: float = 0.2
min_reserve: int = 2
Properties (backward compat): z_min, z_max, max_speed_xy, max_speed_z, charge_rate, battery_return_threshold, battery_ready_threshold, battery_dead_threshold, battery_penalty_emergency

config/sensor.py — SensorConfig
Attributes:

text

comm_range_m: float = 30.0
hfov_deg: float = 90.0
p_detect_base: float = 0.95
p_detect_decay: float = 0.04 → dùng exp(-decay × alt)
enable_noise: bool = True
motion_blur_coeff: float = 0.06
base_miss_rate: float = 0.03
Properties: fov_tan = tan(hfov/2), comm_range

config/reward.py — RewardConfig
Reward values (thực tế trong code):

text

r_coverage_delta: float = 8.0 → per 1% coverage increase
r_victim_base: float = 30.0 → × (urgency/5)

# Battery penalties (per step)

r_battery_20: float = -5.0 → battery ≤ 20%
r_battery_10: float = -20.0 → battery ≤ 10%
r_battery_5: float = -50.0 → battery ≤ 30% (emergency_pct)
r_battery_dead: float = -200.0 → one-time khi battery = 0

r_collision_obstacle: float = -35.0 → one-time per UAV

# Proximity

r_proximity_1m: float = -5.0
r_proximity_2m: float = -1.0
r_proximity_3m: float = -0.2
proximity_penalty_cap: float = -10.0

r_time_penalty: float = -0.02 → per active UAV per step

r_terminal_base: float = 200.0
terminal_bonus_cap: float = 100.0

step_penalty_cap: float = -30.0
step_reward_clip_min: float = -100.0
step_reward_clip_max: float = +100.0

enable_distance_shaping: bool = True
distance_shaping_max_per_uav: float = 1.0
config/obs.py — ObsConfig
Attributes:

text

n_obs_victims: int = 5
n_obs_obstacles: int = 4
n_tracked_uavs: int = 3 → CommSensor dùng tên này (KHÔNG phải n_tracked_teammates)
local_cov_small: int = 15 → meters
local_cov_large: int = 30 → meters
max_uav: int = 8 → critic padding
n_stations: int = None → auto-sync từ env.n_stations trong AppConfig.**post_init**
Computed dims:

text

self_dim = 11
station_dim = n_stations × 4 = 8 (với n_stations=2)
team_dim = n_tracked_uavs × 3 = 9
obstacle_dim = n_obs_obstacles × 3 = 12
victim_dim = n_obs_victims × 5 = 25
coverage_dim = 3
actor_dim = 68
global_dim = 10
critic_dim = 554 (= 8×68 + 10)
config/train.py — TrainConfig
MAPPO hyperparams:

text

mappo_rollout_length: int = 2048 → Overridden bởi auto_compute_config()
mappo_n_epochs: int = 10
mappo_batch_size: int = 256 → Overridden bởi auto_compute_config()
mappo_clip_epsilon: float = 0.2
mappo_gamma: float = 0.99
mappo_gae_lambda: float = 0.95
mappo_lr_actor: float = 3e-4
mappo_lr_critic: float = 1e-3
mappo_max_grad_norm: float = 0.5
mappo_entropy_coeff: float = 0.01
mappo_actor_hidden: tuple = (256, 256)
mappo_critic_hidden: tuple = (512, 256)
mappo_activation: str = 'tanh'
mappo_use_layer_norm: bool = False
Reproducibility:

text

n_seeds: int = 5
seeds: List[int] = [42, 123, 456, 789, 1011]
confidence_level: float = 0.95
config/curriculum_config.py — StageConfig + STAGE_HARD
StageConfig attributes:

text

name: str
map_size: int
n_uav: int = 4
n_victims_min, n_victims_max: int
n_debris, n_danger_total: int
station_capacity: int = 2
max_steps: int
min_episodes: int
advance_coverage, advance_victims: float
STAGE_HARD (dùng để apply_stage, không dùng curriculum):

Python

STAGE_HARD = StageConfig(
name="hard", map_size=250, n_uav=4,
n_victims_min=28, n_victims_max=36,
n_debris=15, n_danger_total=7,
max_steps=400, min_episodes=100,
advance_coverage=0.60, advance_victims=0.70,
)
📁 utils/
utils/geometry.py
9 hàm vectorized (NumPy):

Hàm Input Output Ghi chú
dist_2d(pos1, pos2) [x,y,...] × 2 float XY only
dist_3d(pos1, pos2) [x,y,z] × 2 float 3D
normalize_angle(angle) float rad float ∈ [-π,π]
compute_bearing(from_pos, from_vel, to_pos) arrays float ∈ [-π,π] relative bearing
check_los_2d(pos1, pos2, obstacles) arrays, list bool True = clear
get_circle_cells(center, radius, grid_size, map_size) array, floats ndarray(N,2) Vectorized 10×
get_circle_cells_legacy(...) same ndarray(N,2) Loop version, deprecated
get_relative_position(from_pos, to_pos) arrays ndarray(3,) [dx,dy,dz]
clip_position(pos, min_bounds, max_bounds) arrays ndarray Clamp
get_circle_cells optimization: meshgrid + squared distance, không sqrt → 10× faster vs loop.

utils/logger.py
EpisodeLogger:

Attributes:

text

episode_id: int
seed: int | None
total_reward: float = 0.0
coverage_rate: float = 0.0 → lưu [0,1], convert sang % khi finalize
victims_found: int = 0
total_victims: int = 0
episode_length: int = 0

# Safety

collision_events: List[Dict] → [{step, uav_id, obstacle_id, type, pos, height}]
collision_obstacle: int
collision_uav: int
collision_proximity: int
battery_deaths: int
danger_zone_entries: int
hot_swaps: int
events: Dict[str, int]

# Landing tracking (mới)

landing_events: List[Dict] → [{uav_id, step, battery_before, battery_after, charge_amount}]
total_landings: int = 0
total_charge_time: int = 0
per_uav_landings: Dict[int, int] → {uav_id: count}
Methods:

log_step(rewards: Dict, coverage: float) — Cộng dồn reward, max coverage
log_event(event_type: str) — Phân loại: collision_obstacle/uav/proximity, victim_found, battery_death, danger_zone, hot_swap
set_total_victims(n: int)
log_landing(uav_id, step, battery_before, battery_after)
log_charging_step(uav_id)
log_collision(uav_id, step, obstacle_info: dict)
finalize() → Dict — ⚠️ Có 2 definitions, Python dùng thứ 2 (override). Thứ 2 KHÔNG có landing fields.
finalize() thứ 2 (active) trả về:

text

episode_id, seed, duration, episode_length,
total_reward, avg_reward_per_step,
coverage_rate (percent 0-100),
victims_found, total_victims, victims_found_rate,
collision_obstacle, collision_uav, collision_proximity, total_collisions,
battery_deaths, danger_zone_entries, hot_swaps,
success: bool(coverage_ratio >= 0.9)
⚠️ BUG: Landing fields chỉ trong finalize() thứ nhất (bị shadow). Cần merge khi sửa.

TrainingLogger:

text

verbose: int → 0=silent, 1=basic, 2=detail
window_size: int = 100

log_episode(metrics: Dict) → auto print nếu verbose
get_stats(last_n: int) → Dict
save(filepath: str)
load(filepath: str)
Module-level: compare_training_runs(runs, labels) — So sánh nhiều run

📁 entities/
entities/uav.py — UAV, UAVState
UAVState enum:

text

ACTIVE → RL controls
RETURNING → auto-navigate to station
CHARGING → docked, charging
DEPLOYING → auto-navigate to mission (instant → ACTIVE trong backend)
DISABLED → terminal
UAV attributes:

text

id: int
pos: ndarray[3] → [x, y, z]
vel: ndarray[3] → [vx, vy, vz]
battery: float → [0.0, 100.0]
battery_pct (property) → alias cho battery (FIX BUG-36)
state: UAVState
target_station: ChargingStation | None
battery_death: bool → one-time flag khi battery = 0
steps_alive: int
distance_xy: float
distance_3d: float
victims_found: int
Key methods:

apply_action(action: ndarray) — Chỉ khi ACTIVE. Scale [-1,1] → velocity → clip → update pos. Clamp pos [0, map_size], altitude [z_min, z_max].

auto_navigate(target_pos) — RETURNING/DEPLOYING. No-overshoot proportional control. Altitude clip theo state:

RETURNING → [0, z_max]
CHARGING → [0, 0.5m]
DEPLOYING → [0, z_max]
\_do_drain() — Battery drain × dt_seconds. Proportional to velocity.

\_do_charge(stations) — Via target_station hoặc nearest in-range.

set_state(new_state):

Guard: DISABLED terminal (không thể đổi)
CHARGING→ACTIVE chỉ khi battery ≥ battery_ready_pct
get_state_onehot() → ndarray(5,) — [ACTIVE, RETURNING, CHARGING, DEPLOYING, DISABLED]

⚠️ Note: \_write_self() trong obs_builder chỉ dùng [:4] của onehot, không phải cả 5.

find_nearest_station(stations) → Ưu tiên available, fallback nearest bất kể full.

entities/charging_station.py — ChargingStation
Attributes:

text

id: int
pos: ndarray[3] → z = 0.0
capacity: int = 2
charge_radius: float = 3.0m
charge_rate: float = 1.5 %/step
current_occupants: List[UAV]
occupant_ids: Set[int] → O(1) lookup
Methods:

in_range(uav_pos) — dist_xy ≤ charge_radius AND z ≤ 0.5m
try_occupy(uav) → bool — False nếu full hoặc đã occupied
release(uav) → bool
has_uav(uav) → bool — O(1) via occupant_ids
charge(uav):
Out of range → auto release
Pin đầy → auto release
Sạc min(charge_rate, 100-battery)
force_release_all() — Dùng khi reset episode
get_occupancy_ratio() → float [0,1]
entities/victim.py — InjuredVictim, MobileVictim
BaseVictim attributes:

text

id: int
pos: ndarray[3] → z = 0.0 (trên mặt đất)
urgency: float → [1.0, 5.0]
is_found: bool = False
found_at_step: int = -1
found_by_uav: int = -1
Methods:

mark_found(step, uav_id) — Set flags + gọi \_on_found()
get_reward_value() → r_victim_base × (urgency/5.0)
update(step_count, obstacles) → alias cho step(obstacles)
InjuredVictim: Stationary, urgency ∈ [4.0, 5.0]

MobileVictim: Random walk, urgency ∈ [1.0, 3.0]

speed ∈ [0.2, 0.4] m/s
Đổi hướng mỗi 20 steps
Bounce back khi hit obstacle
\_on_found() → Freeze (speed=0)
entities/obstacle.py — Debris, DangerZone
Debris — Static obstacle:

text

id: int
pos: ndarray[3] → center, z=0
height_3d: float
shape: str → "circle" / "rectangle" / "polygon"
Shape-specific: radius | width, height_2d, rotation | vertices
Methods:

causes_collision(uav_pos) → in_zone_2d(xy) AND uav.z < height_3d
blocks_los(pos1, pos2) → bool
\_get_fallback_radius() → bounding circle radius
DangerZone (kế thừa Debris):

text

danger_type: str → "fire"/"smoke"/"gas"/"radiation"/"collapse"
max_height: float → từ DangerZoneConfig.heights
penalty: float → per-step penalty
Methods:

is_inside(uav_pos) → in_zone_2d(xy) AND uav.z < max_height
blocks_los() → chỉ fire và smoke block LOS
get_sensor_modifier() → float [0.4, 1.0]:
smoke: 0.40, fire: 0.55, collapse: 0.70, gas: 0.85, radiation: 0.95
get_battery_modifier() → fire: 0.05, others: 0.0
Danger zone penalties:

text

gas: -3.0, fire: -3.0, smoke: -1.5, collapse: -1.0, radiation: -5.0
📁 core/
core/coverage_map.py — CoverageMap
Attributes:

text

grid: ndarray(bool, [GS, GS])
timestamps: ndarray(int32, [GS, GS]) → last scan step
first_scan: ndarray(int32, [GS, GS]) → first scan step (-1 if never)
scan_count: ndarray(int32, [GS, GS])
Key methods:

reset() — grid=False, timestamps=0, first_scan=-1, scan_count=0
mark_explored(uav_pos, fov_radius, step) — Vectorized via get_circle_cells()
get_coverage_rate() → float [0,1]
get_coverage_percent() → float [0,100]
get_local_coverage(pos, radius) → float [0,1] — Dùng trong obs_builder
get_nearest_unexplored(pos) → ndarray | None — O(N)
get_staleness(pos, radius, step) → float
get_stats(step) → Dict
core/map_generator.py — MapGenerator v4.1
⚠️ Key fix v4.1: Config widths = diameters, generator converts: radius = width / 2.0

generate(n_victims_override, seed) → Dict:

text

stations: List[Dict] → [{id, pos}]
debris: List[Dict] → [{id, pos, shape, radius/width/vertices, height_3d}]
danger_zones: List[Dict] → [{id, pos, shape, radius/width, danger_type, max_height, penalty}]
victims: List[Dict] → [{id, pos, victim_type, urgency, speed?}]
uav_spawns: List[Dict] → [{id, pos:[x,y,z]}]
seed: int
n_victims: int
Shape distribution debris: 40% circle, 40% rectangle, 20% polygon

Victim spawning: 60% near debris (group), 40% random

Progressive relaxation: After 70% attempts → reduce min_spacing

core/fleet_manager.py — FleetManager
Vai trò: Enforce safety (không thể ignore) + suggest deployments (RL có thể ignore).

Attributes:

text

n_total, n_reserve: int
all_uavs: List[UAV]
stations: List[ChargingStation]
\_uav_return_locks: Dict[int, bool] → hysteresis
\_episode_forced_returns: int
\_episode_disables: int
enforce_safety_constraints() — Gọi mỗi step, KHÔNG thể ignore:

battery ≤ 0 → DISABLED + battery_death = True
battery < 30% AND ACTIVE → RETURNING (hysteresis lock)
battery ≥ 80% AND CHARGING AND n_active < n_total-1 → ACTIVE (auto-deploy)
step() → Dict:

text

enforced: {enforced_disables, enforced_returns, auto_deploys, total_enforced}
suggestions: {deploy: [uav_ids], return: [uav_ids]}
priority_hints: {operational_ratio, reserve_health, station_pressure}
spatial: {center_of_mass, spread_radius, n_active_positions}
get_battery_stats() → Dict: mean/min/max/std/critical_count/low_count/emergency_count

get_episode_summary() → Dict: forced_returns, disables

📁 sensors/
sensors/fov_sensor.py — FOVSensor
calculate_fov_radius(altitude) = altitude × fov_tan

check_detected(uav, victim, obstacles) — Pipeline:

dist_2d ≤ fov_radius (fast reject)
check_los_2d (fast reject)
calculate_detection_prob(altitude, uav_speed, env_factor, victim_factor)
Bernoulli sample với \_rng
calculate_detection_prob() — 5 stages:

text

p = p_base × exp(-decay × altitude) → altitude factor
p = p × env_factor → smoke/fire degradation
p = p × (1 - motion_blur_coeff × speed_ratio) → motion blur
p = p × victim_factor → type: injured=1.15, mobile≈0.75-0.95
p = p × (1 - base_miss_rate) → hardware
clip to [0, 1]
scan_victims(uav, victims, obstacles) → ndarray(25,):

Top-5 nearest in FOV
Features per victim: [rel_x/fov_r, rel_y/fov_r, dist/fov_r, urgency/5, is_found]
scan_obstacles(uav, obstacles) → ndarray(12,):

Top-4 nearest
Features: [rel_x/fov_r, rel_y/fov_r, type_id (0=Debris, 1=DangerZone)]
set_seed(seed) — Reproducible evaluation

sensors/comm_sensor.py — CommSensor
scan(ego_uav, all_active_uavs) → ndarray(9,):

Top-3 teammates trong comm_range (30m)
Features per teammate: [norm_dist, norm_bearing, norm_alt]
📁 observation/
observation/obs_builder.py — ObservationBuilder
ObsResult: actor_obs: Dict[int, ndarray(68)], critic_obs: ndarray(554)

Slices (precomputed):

text

slices[0]: self [0:11]
slices[1]: stations [11:19]
slices[2]: teammates [19:28]
slices[3]: obstacles [28:40]
slices[4]: victims [40:65]
slices[5]: coverage [65:68]
build_all(all_uavs, stations, victims, obstacles, current_step) → ObsResult:

DISABLED UAVs → zero obs
Critic: Stack 8 UAV obs (sorted by uav.id, zero-padded) + 10 global dims
build_actor_obs(uav, all_uavs, stations, victims, obstacles, current_step) → ndarray(68):

Private writers: \_write_self, \_write_stations, \_write_teammates, \_write_obstacles, \_write_victims, \_write_coverage
Debug mode: Sanity check NaN/Inf/shape (chỉ khi cfg.env.debug_obs=True)
📁 rewards/
rewards/baseline_reward.py — BaselineReward v3.1
16 reward components:

Component Value Loại
coverage_delta +8.0 per 1% Dense, shared
victim_found +30 × urgency/5 Sparse, individual
distance_shaping ±1.0 cap, weight=0.1 Dense, delta-based
battery_penalty -5/-20/-50 per step Dense, individual
battery_dead -200 one-time Sparse, individual
collision_obstacle -35 one-time Sparse, individual
proximity -5/-1/-0.2 per step Dense, pairwise
danger_zone -1.0 to -5.0 per step Dense
fleet_incentive 0.0 Deprecated
time_penalty -0.02 per active UAV Dense
terminal +200 base + ≤100 bonus Sparse, terminal
penalty_cap_adjustment additive Capping mechanism
landing_reward +120 one-time Tier 3
hover_penalty -3.0 per step Tier 2
approach_reward +0.3×(1-norm_dist) Tier 1
raw_total, total computed —
Landing reward 3 tiers:

Python

# Tier 3: One-time khi chuyển CHARGING (per episode per UAV)

if uav.state == CHARGING and uav.id not in \_landed_uavs:
landing_total += 120

# Tier 1: battery ≤ 40% AND (ACTIVE or RETURNING) → approach station

approach_rew = 0.3 × (1.0 - min_dist / max_dist)

# Tier 2: ACTIVE + trong landing_range (charge_radius×2) + battery ≤ 40% nhưng không land

hover_total += -3.0
Terminal bonus formula:

Python

coverage_bonus = terminal_cap × 0.60 × coverage_rate
victim_bonus = terminal_cap × 0.20 × found_ratio
time_bonus = terminal_cap × 0.10 × (1-time_ratio) # chỉ khi cov ≥ 80%
battery_bonus = terminal_cap × 0.10 × mean_battery/100

# clipped to [0, terminal_cap=100]

# Total terminal = 200 + bonus (max 300)

Delta shaping:

Python

# Memory: \_prev_min_dist[uav_id]

delta = prev_min_dist - current_min_dist
reward = delta × 0.1 # shaping_weight
clipped to [-1.0, +1.0] # shaping_max
Penalty cap (BUG-31 fix) — Additive:

Python

penalty_sum = sum(negative components)
if penalty_sum < cap (-30):
components["penalty_cap_adjustment"] = cap - penalty_sum
Per-UAV vs Global:

compute() → Global reward (16 components dict, cho logging)
compute_per_uav() → Per-agent: coverage/n_active, victim chỉ agent tìm thấy, penalties riêng
Module-level stateless functions:

\_coverage_delta_reward(prev, cur, weight)
\_victim_found_reward(newly_found, r_base)
\_battery_penalty_single(uav, reward_cfg, uav_cfg)
\_battery_urgency_shaping(uav, stations, map_size) — Penalty tỉ lệ dist × severity
\_proximity_reward(active_uavs, ...) — Pairwise fleet
\_proximity_reward_single(uav, active_uavs, ...) — Per-UAV
📁 env_setup/
env_setup/backends/logic_backend.py — LogicBackend
apply_actions(actions: Dict[int, ndarray(4)]):

Python

# action[4] = [vx, vy, vz, land]

move_action = action[:3]
land_signal = float(action[3])

# ACTIVE UAV landing conditions (ALL must hold):

if land_signal > 0.5 AND battery ≤ 40.0:
nearest = find_station_in_range(charge_radius × 2.0 = 6m)
if nearest → set RETURNING + auto_navigate(target z=0.0)
else → apply_action(move_action) # fallback
else:
→ apply_action(move_action)

# RETURNING → auto_navigate(target_station, z=0)

# DEPLOYING → set_state(ACTIVE) ngay lập tức

# CHARGING/DISABLED → no movement

step_physics():

RETURNING in_range(station) → try_occupy() → CHARGING
CHARGING → station.charge(uav)
ACTIVE/DEPLOYING → uav.update_battery() (drain)
step_world():

fleet_manager.step() (enforce_safety_constraints)
victim.update() (mobile movement)
coverage_map.mark_explored() cho non-DISABLED UAVs
fov_sensor.check_detected() cho ACTIVE + RETURNING UAVs
Victim detection scope: ACTIVE + RETURNING (không phải CHARGING/DEPLOYING/DISABLED)

\_build_uavs(map_data): Dùng map_data["uav_spawns"] trực tiếp (FIX 4.2). Tất cả UAV khởi tạo state=ACTIVE.

env_setup/base_env.py — SARBaseEnv
reset(seed) → (obs_dict: Dict[int, ndarray(68)], info: Dict):

info['global_obs'] = ndarray(554)
obs_dict keys = int (uav_id)
step(actions: Dict[int, ndarray(4)]) — Execution order (CRITICAL):

text

1. apply_actions(actions)
2. step_physics()
3. step_world()
4. step_count += 1
5. \_check_done() ← TRƯỚC reward (BUG-ENV-06 fix)
6. compute_per_uav() ← Per-agent rewards
7. compute() ← Global reward (logging only)
8. \_log_step()
9. \_build_obs_dict()
10. return
    \_check_done() returns:

Python

"coverage" → win: coverage >= 90%
"victims" → win: all found
"disabled:battery_death" → fail: all dead pin
"disabled:other" → fail: mixed
None → continue
info keys trong step:

text

coverage, victims_found, victims_total,
step, coverage_rate, n_active, n_charging, n_disabled,
success, done_reason, rewards_breakdown,
newly_found_ids, battery_stats, global_obs,
episode: ep_metrics → chỉ khi is_terminal
⚠️ DISABLED UAVs: reward = 0.0 (không gọi compute_per_uav)

env_setup/sar_pettingzoo_env.py — SARPettingZooEnv
Thin wrapper: convert int keys ↔ str keys ("uav_0", "uav_1", ...).

reset() → (Dict[str, ndarray], Dict[str, Dict])
step(actions: Dict[str, ndarray]) → (obs, rewards, terminations, truncations, infos)
action_space(agent) = Box(3,) — KHÔNG phải 4! land dim transparent
infos['uav_0']['global_obs'] = ndarray(554) — MAPPO critic access
Tất cả agents terminate đồng thời (cooperative task)
env_setup/vec_env.py — VectorizedEnv
env_worker(pipe, config, seed) — Worker process:

Python

rng = np.random.default_rng(seed) # RNG riêng mỗi worker

# Initial reset với seed gốc

env.reset(seed=seed)

# Auto reset với seed MỚI sau mỗi episode

if done:
current_seed = int(rng.integers(0, 2\*\*31))
env.reset(seed=current_seed)
Commands: "reset", "step", "close"

Cache last valid obs/global_obs/info — Handle PettingZoo edge case (empty dict khi done).

VectorizedEnv(config, n_envs, start_seed):

Spawn N workers via mp.get_context("spawn") (CUDA-safe)
reset() → (obs[n_envs, n_agents, 68], global_obs[n_envs, 554])
step(actions[n_envs, n_agents, 4]) → (obs, global_obs, rewards, dones, infos)
close() — Graceful shutdown
📁 training/algorithms/mappo/
networks.py — MLP
text

class MLP(nn.Module):
input_dim → [Linear → Activation → (LayerNorm?)] × N → Linear → (output_activation?)
activation: 'tanh' / 'relu' / 'elu'
output_activation: 'none' = Identity
⚠️ Note: orthogonal_init defined nhưng bị comment out trong MLP constructor. Standard PyTorch init được dùng.

actor.py — ActorNetwork
Architecture:

text

obs[batch, 68]
→ backbone: MLP(68 → 256 → 256, output_activation='tanh') [shared]
→ movement_head: Linear(256 → 3) [orthogonal gain=0.01]
→ land_head: Linear(256 → 1) [bias=-2.0 → P(land)≈0.12 initial]
→ log_std: Parameter(zeros(3)) [learnable, state-independent]
get_action(obs[batch,68], deterministic=False) → (action[batch,4], log_prob[batch]):

Python

# Movement: Normal(move_mean, exp(log_std)) → sample → clamp [-1,1]

# Landing: Bernoulli(logits=land_logit) → sample {0.0, 1.0}

# log_prob = move_log_prob + land_log_prob

# action = cat([move_action, land_action], dim=-1) → [batch, 4]

evaluate_actions(obs[batch,68], actions[batch,4]) → (log_prob[batch], entropy[batch]):

Python

# Split: actions[:,:3] = move, actions[:,3:] = land

# move: Normal log_prob + entropy

# land: Bernoulli log_prob + entropy

# return sum

get_land_prob(obs) → float [0,1] — Dùng cho logging/viz

critic.py — CriticNetwork
Architecture:

text

global_obs[batch, 554] → MLP(554 → 512 → 256 → 1) → value[batch, 1]
Methods:

forward(global_obs) → [batch, 1]
get_value(global_obs) → [batch] (squeezed)
compute_loss(global_obs, returns) → MSE scalar
compute_value_metrics(global_obs, returns) → Dict: value_loss, explained_variance, mean/std pred/target
buffer.py — RolloutBuffer
Storage arrays:

text

observations[capacity, n_agents, 68]
global_obs[capacity, 554]
actions[capacity, n_agents, 4] → action_dim = 4 (hybrid)
rewards[capacity, n_agents]
values[capacity, n_agents]
log_probs[capacity, n_agents]
dones[capacity]
advantages[capacity, n_agents]
returns[capacity, n_agents]
ptr = 0
add(...) — Raise RuntimeError nếu overflow.

compute_gae(last_values[n_agents], last_done):

actual_length = min(ptr, capacity) — Hỗ trợ buffer không đầy
Vectorized backward loop: Tất cả agents cùng lúc (4× faster)
Normalize advantages: (adv - mean) / (std + 1e-8)
get_batches(batch_size) → Iterator[Dict]:

Flatten [actual_length, n_agents, ...] → [actual_length×n_agents, ...]
global_obs repeat cho n_agents
Random permutation → yield batches với keys: obs, global_obs, actions, old_log_probs, advantages, returns
clear() — Reset ptr = 0

trainer.py — MAPPOTrainer + \_EnvWrapper
\_EnvWrapper:

text

n_envs, n_agents, obs_dim
\_is_vec: bool
\_current_obs, \_current_global → Cache (BUG-5 fix)
\_needs_reset: bool = True

reset() → Hard reset (lần đầu + curriculum change)
get_current_obs() → Từ cache, không reset (BUG-5 fix)
step(actions_batch[n_envs, n_agents, 4])
reset_hard() → Force reset
render()
close()
MAPPOTrainer.**init**(config, device, run_name, n_envs):

text

actor: ActorNetwork(obs_dim=68, action_dim=4, hidden=(256,256))
critic: CriticNetwork(global_obs_dim=554, hidden=(512,256))
buffer: RolloutBuffer(capacity=rollout_length×n_envs, action_dim=4)
actor_opt: Adam(lr=3e-4)
critic_opt: Adam(lr=1e-3)

ep_rewards, ep_lengths, ep_coverage, ep_victims: deque(maxlen=100)
total_episodes_done, total_steps, update_count: int

# Trigger tracking (FIX-T3)

\_next_log_ep, \_next_viz_ep, \_next_checkpoint_ep: int

# Dirs

output_dir = results/mappo/{run_name}/
checkpoint_dir, viz_dir
train(total_episodes, curriculum_manager=None, seed=42, log_every_n_eps, viz_every_n_eps, checkpoint_every_n_eps):

Reset env 1 lần duy nhất ở đầu (BUG-5 fix)
curriculum_manager=None → No curriculum
Loop: \_rollout() → \_update() → logging/viz/checkpoint
Trigger dùng >= thay vì % (FIX-T1): if ep >= self.\_next_log_ep
\_rollout(env, pbar, max_episodes) — Dispatcher:

n_envs==1 → \_rollout_single()
n_envs>1 → \_rollout_vectorized()
\_rollout_vectorized():

Python

obs_batch, g_batch = env.get_current_obs() # No reset
for step in range(rollout_length): # Batch inference
obs_flat = obs_batch.reshape(n_envs × n_agents, 68)
act, lp = actor.get_action(obs_flat) → [n_envs×n_agents, 4]
val = critic.get_value(g_batch) → [n_envs]
act_batch = reshape to [n_envs, n_agents, 4]

    # Step all envs
    next_obs, next_g, rews, dones, infos = env.step(act_batch)

    # Per-env tracking
    for ei in range(n_envs):
        buffer.add(...)
        if dones[ei]:
            # Log landing stats từ infos[ei]['episode']
            pbar.update(1)
            check max_episodes

# GAE bootstrap

bootstrap = mean(critic.get_value(last_g))
buffer.compute_gae(bootstrap, last_done)
\_update() — PPO:

Python

for epoch in n_epochs:
for batch in buffer.get_batches(batch_size): # Actor
lp, entropy = actor.evaluate_actions(obs, actions)
ratio = exp(lp - old_lp)
loss = -min(ratio×adv, clip(ratio,1-ε,1+ε)×adv) - entropy_coeff×entropy
actor_opt.step()

        # Critic
        values = critic.get_value(global_obs)
        loss = MSE(values, returns)
        critic_opt.step()

buffer.clear()
save_checkpoint(episode, curriculum_manager, tag):

Lưu: actor/critic state_dict, optimizers, ep_rewards/coverage/victims, curriculum_stage
Print absolute path (FIX-T2)
load_checkpoint(path) → int (episode number)

📁 Root files
train_mappo.py — CLI Entry Point
auto_compute_config(max_steps, n_envs, n_uav, batch_size_hint, safety_factor):

Python

min_rollout = ceil(max_steps × safety_factor) # e.g. 400 × 1.5 = 600
rollout_length = ceil(min_rollout/64) × 64 # align to 64 → 640
buffer_capacity = rollout_length × n_envs
batch_size = auto or clip(hint, 64, buffer//2)
Main flow:

Python

cfg = AppConfig()
cfg.viz_mode = "2d"
cfg.apply_stage(STAGE_HARD) # map_size=250, max_steps=400, etc.
cfg.env.n_uav = 4 # Fixed

auto_cfg = auto_compute_config(max_steps=400, n_envs=N, ...)
cfg.train.mappo_rollout_length = auto_cfg.rollout_length
cfg.train.mappo_batch_size = auto_cfg.batch_size

trainer = MAPPOTrainer(config=cfg, device=device, n_envs=N)
trainer.train(
total_episodes=3000,
curriculum_manager=None, # ← No curriculum
seed=42,
log_every_n_eps=50,
viz_every_n_eps=250,
checkpoint_every_n_eps=100,
)
CLI args: --total-episodes, --seed, --device, --run-name, --n-envs, --max-steps, --map-size, --batch-size, --safety-factor, --n-epochs, --lr-actor, --lr-critic, --log-interval, --viz-interval, --checkpoint-interval

5. TRAINING FLOW (Tổng thể)
   text

train_mappo.py
├── AppConfig + cfg.apply_stage(STAGE_HARD)
├── auto_compute_config() → rollout_length, batch_size
└── MAPPOTrainer(config, device, n_envs)
├── ActorNetwork(68 → 4) ~params: backbone+heads
├── CriticNetwork(554 → 1) ~params: MLP(512,256)
└── RolloutBuffer(capacity=rollout_len×n_envs, action_dim=4)

    _EnvWrapper (n_envs=1→PettingZoo, n_envs>1→VectorizedEnv)
    env.reset() [1 lần duy nhất]

    WHILE episodes < total_episodes:
    ├── _rollout():
    │   ├── get_current_obs() [no reset, BUG-5 fix]
    │   ├── FOR step in rollout_length:
    │   │   ├── actor.get_action(obs[N×4, 68]) → action[N×4, 4], log_prob[N×4]
    │   │   ├── critic.get_value(global_obs[N, 554]) → value[N]
    │   │   ├── env.step(actions[N, 4, 4])
    │   │   ├── buffer.add(...)
    │   │   └── IF done: log + pbar.update(1) + check early stop
    │   └── buffer.compute_gae(bootstrap, last_done)
    │
    ├── _update():
    │   ├── n_epochs=10 × minibatches(batch_size)
    │   ├── Actor: PPO clip + entropy bonus
    │   ├── Critic: MSE loss
    │   └── buffer.clear()
    │
    ├── Logging (FIX-T1: >= trigger)
    ├── Viz snapshot (ep_xxx.png)
    └── Checkpoint save (absolute path, FIX-T2)

    FINALIZE:
    ├── save_checkpoint(tag="final")
    ├── plot_training_curves()
    └── env.close()

6. SINGLE STEP FLOW
   text

env.step(actions: Dict[str, ndarray(3)]) ← PettingZoo API (str keys, 3-dim)
↓ Convert str→int keys + pad to 4-dim
LogicBackend.apply*actions(Dict[int, ndarray(4)]):
ACTIVE: land>0.5 AND bat≤40% AND station_in_range(6m) → RETURNING
else → apply_action(move[:3])
RETURNING: auto_navigate(station, z=0)
DEPLOYING: set_state(ACTIVE) immediately
↓
step_physics():
RETURNING in_range(3m, z≤0.5) → try_occupy → CHARGING
CHARGING: station.charge(uav)
ACTIVE/DEPLOYING: \_do_drain()
↓
step_world():
fleet_manager.enforce_safety_constraints()
battery=0 → DISABLED
battery<30% AND ACTIVE → RETURNING
battery≥80% AND CHARGING AND n_active<n_total-1 → ACTIVE
victim.update() → mobile movement
coverage_map.mark_explored() for non-DISABLED
fov_sensor.check_detected() for ACTIVE+RETURNING
↓
step_count += 1
\_check_done() → "coverage"/"victims"/"disabled:\*"/None
compute_per_uav() → rewards_dict[int, float]
compute() → global_reward (logging)
\_build_obs_dict() → (actor_obs_dict, critic_obs[554])
return (obs, rewards, done, truncated, info)
info['global_obs'] = critic_obs[554]
info['uav_0']['global_obs'] via PettingZoo wrapper 7. TRẠNG THÁI HIỆN TẠI
✅ Đã hoàn thành
Phase 1: Core infrastructure đầy đủ
Phase 2: MAPPO hoàn chỉnh, trainable, stable
Hybrid action space [vx, vy, vz, land]
Landing reward 3 tiers
BaselineReward v3.1 (16 components)
BUG-5 fix: Không reset env mỗi rollout
FIX-T1: >= trigger cho log/viz/checkpoint
FIX-T2: Absolute path khi save checkpoint
FIX-T3: \_next*\*\_ep tracking
BUG-ENV-06: done check TRƯỚC reward
VectorizedEnv seed progression per-episode
⚠️ Known Issues
Issue File Severity
EpisodeLogger.finalize() có 2 defs — landing fields bị shadow logger.py Medium
orthogonal_init defined nhưng commented out trong MLP networks.py Low
\_write_self() dùng state_onehot[:4] thay vì cả 5 dims obs_builder.py Low
Tất cả UAV spawn ACTIVE (không có reserve pool ban đầu) logic_backend.py Low
⬜ Chưa làm
MASAC implementation
MATD3 implementation
Statistical comparison (Wilcoxon tests)
LLM reward generation (Phase 4)
PyBullet/Isaac backends (Phase 5) 8. CÁCH CHẠY
Bash

# Single env, 3000 eps, HARD stage

python train_mappo.py --total-episodes 3000 --n-envs 1 --seed 42

# Vectorized (6 envs), 2× faster

python train_mappo.py --total-episodes 3000 --n-envs 6 --seed 42

# Quick test

python train_mappo.py --total-episodes 50 --n-envs 1

# Multi-seed training (5 seeds)

for seed in 42 123 456 789 1011; do
python train_mappo.py --total-episodes 3000 --seed $seed \
        --run-name "hard_mappo_s${seed}" --n-envs 6
done
Auto-compute cho HARD:

text

max_steps = 400, safety_factor = 1.5
→ min_rollout = 600
→ rollout_length = 640 (aligned to 64)
→ buffer_capacity = 640 × n_envs 9. CHECKLIST IMPLEMENT MASAC/MATD3
text

training/algorithms/masac/
actor.py → Squashed Gaussian (SAC-style), obs_dim=68, act_dim=4
twin_critic.py → Q1, Q2 networks (sa_dim=68+4=72 → Q-value)
replay_buffer.py → Off-policy buffer (capacity 1M+)
trainer.py → SAC update: actor + twin Q + entropy temperature α

training/algorithms/matd3/
actor.py → Deterministic policy, obs_dim=68, act_dim=4
twin_critic.py → Twin Q-networks
replay_buffer.py → Off-policy buffer
trainer.py → TD3: delayed policy update, target smoothing

Cùng config: - action_dim = 4 (hybrid) - obs_dim = 68, global_obs_dim = 554 - BaselineReward v3.1 - HARD stage - Same 5 seeds

train_masac.py, train_matd3.py → tương tự train_mappo.py

vậy bạn hayx tổng hợp những cái t đã làm ở project này từ đầu tới cuối ghi chi tiết rõ ràng từng folder có file nào file có hàm và thuộc tính nào có tác dụng gì một cách chi tiết (ko cần code) và ghi rõ ràng chi tiết để khi qua đoạn chat ms thì chỉ cần đưa cái đó cho nớ là nó sẽ hiểu đang làm gì và đang thực hiện cái gì biết đang thực heienj cái gì theo format này
