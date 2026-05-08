🚁 SAR UAV SWARM — PROJECT STATE SNAPSHOT v5.0
Dựa trên code thực tế | MAPPO + MASAC + MATD3 Complete | HARD Stage | No Curriculum | Kaggle-optimized

1. 🎯 MỤC TIÊU DỰ ÁN
Task: 4 UAV tự động phối hợp tìm kiếm nạn nhân trong khu vực thảm họa 250×250m.
Research Plan:

Paper 1: So sánh MAPPO vs MASAC vs MATD3 trên HARD stage cố định, 5 seeds × 3000 eps/seed → Wilcoxon test
Paper 2: LLM-generated reward vs BaselineReward v4.0 (hand-crafted)
Không dùng curriculum learning. curriculum_manager=None trong tất cả trainer.


2. 📐 THÔNG SỐ HỆ THỐNG
Observation Space — Actor: 68 dims
SliceDimsNội dungNormalization[0:11]11Self: pos(3), vel(3), battery(1), state_onehot(4) — chỉ 4 dims đầupos/map_size, vel/max_speed, bat/100[11:19]8Stations (2×4): rel_x, rel_y, dist, occupancy_ratiorel/map_size, dist/map_diagonal[19:28]9Teammates (3×3): norm_dist, norm_bearing, norm_altCommSensor.scan()[28:40]12Obstacles (8×3): rel_x/fov_r, rel_y/fov_r, type_idFOVSensor.scan_obstacles()[40:65]25Victims (5×5): rel_x, rel_y, dist/fov_r, urgency/5, is_foundFOVSensor.scan_victims()[65:68]3Coverage: local_15m, local_30m, time_remaining[0,1]
Observation Space — Critic: 554 dims

[0:544] = 8 UAVs × 68 (zero-padded, sorted by uav.id)
[544:554] = 10 global: n_active/n, n_charging/n, n_disabled/n, n_alive/n, bat_mean, bat_std, bat_min (÷100), global_coverage, victims_found_rate, time_remaining

Action Space

Actor output: action[4] = [vx, vy, vz, land]
[vx, vy, vz] — continuous, khác nhau theo algo:

MAPPO: Gaussian Normal, log_prob tính trước khi clip
MASAC: Squashed Gaussian (TanhNormal) → ∈ [-1,1]
MATD3: Deterministic + Gaussian exploration noise


[land] — Bernoulli {0,1}, bias=-2.0 → P(land)≈0.12 initial
⚠️ LogicBackend chỉ chấp nhận land nếu battery ≤ 40%

Key Numbers — HARD Stage
ParamValuemap_size250mmax_steps2500n_uav4n_stations2n_victims30–40n_debris30n_danger_total8actor_dim68critic_dim554dt_seconds1.0charge_radius3.0mstation_capacity1battery_emergency_pct40.0% (force RETURNING)

3. 🗂️ CẤU TRÚC THƯ MỤC
uav_swarm_pybullet/
├── config/
│   ├── __init__.py
│   ├── config.py              # AppConfig (master)
│   ├── env.py                 # EnvConfig
│   ├── uav.py                 # UAVConfig
│   ├── sensor.py              # SensorConfig
│   ├── entity.py              # VictimConfig, ObstacleConfig, DangerZoneConfig
│   ├── reward.py              # RewardConfig v4.0
│   ├── obs.py                 # ObsConfig + ObsSchemaConfig
│   ├── train.py               # TrainConfig (MAPPO + MASAC + MATD3 hyperparams)
│   └── curriculum_config.py  # StageConfig, STAGE_HARD
├── utils/
│   ├── geometry.py            # 9 hàm vectorized
│   └── logger.py              # EpisodeLogger v2.0, TrainingLogger
├── entities/
│   ├── uav.py                 # UAV, UAVState
│   ├── victim.py              # InjuredVictim, MobileVictim, BaseVictim
│   ├── charging_station.py    # ChargingStation
│   └── obstacle.py            # Debris, DangerZone (multi-shape)
├── core/
│   ├── coverage_map.py        # CoverageMap (temporal-aware)
│   ├── map_generator.py       # MapGenerator v4.1
│   └── fleet_manager.py       # FleetManager
├── sensors/
│   ├── fov_sensor.py          # FOVSensor (với noise pipeline)
│   └── comm_sensor.py         # CommSensor
├── observation/
│   └── obs_builder.py         # ObservationBuilder, ObsResult
├── rewards/
│   └── baseline_reward.py     # BaselineReward v4.0 (anti-exploit)
├── env_setup/
│   ├── base_env.py            # SARBaseEnv (Gymnasium)
│   ├── sar_pettingzoo_env.py  # SARPettingZooEnv (PettingZoo wrapper)
│   ├── vec_env.py             # VectorizedEnv (multiprocessing)
│   └── backends/
│       ├── base_backend.py    # BaseBackend (ABC)
│       └── logic_backend.py   # LogicBackend (~1000 steps/s)
├── visualization/
│   ├── renderer_factory.py
│   ├── visualizer2d.py
│   └── visualizer3d.py
├── training/
│   ├── curriculum.py          # CurriculumManager (không dùng hiện tại)
│   └── algorithms/
│       ├── mappo/
│       │   ├── __init__.py
│       │   ├── networks.py    # MLP, orthogonal_init
│       │   ├── actor.py       # ActorNetwork (Gaussian hybrid)
│       │   ├── critic.py      # CriticNetwork (centralized)
│       │   ├── buffer.py      # RolloutBuffer + GAE (multi-env)
│       │   └── trainer.py     # MAPPOTrainer + _EnvWrapper
│       ├── masac/
│       │   ├── actor.py       # SACActorNetwork (Squashed Gaussian)
│       │   ├── twin_critic.py # TwinCriticNetwork (Q1+Q2)
│       │   ├── replay_buffer.py # ReplayBuffer (off-policy)
│       │   └── trainer.py     # MASACTrainer + _EnvWrapper
│       └── matd3/
│           ├── actor.py       # TD3ActorNetwork (Deterministic)
│           ├── twin_critic.py # TD3TwinCriticNetwork
│           ├── replay_buffer.py # ReplayBuffer (reuse từ masac)
│           └── trainer.py     # MATD3Trainer + _EnvWrapper
├── hf_upload.py               # HFUploader, HFDownloader
├── train_mappo.py             # CLI entry point MAPPO
├── train_masac.py             # CLI entry point MASAC
├── train_matd3.py             # CLI entry point MATD3
├── run_visualization.py       # CLI visualization + GIF export
├── plot_compare.py            # Download HF + plot comparison
└── plot_local.py              # Plot từ local metrics.json

4. 📁 CHI TIẾT TỪNG FILE

📁 config/
config/config.py — AppConfig
Mục đích: Master config, tổ hợp tất cả sub-config. Entry point duy nhất cho toàn bộ hệ thống.
Attributes:

env: EnvConfig
uav: UAVConfig
sensor: SensorConfig
victim: VictimConfig
obstacle: ObstacleConfig
danger: DangerZoneConfig
reward: RewardConfig
obs: ObsConfig
train: TrainConfig
viz_mode: str = "none" — "2d" | "3d" | "none"
viz_3d_cfg: dict = {} — passed to RenderConfig3D

Methods:

__post_init__() — Auto-sync obs.n_stations = env.n_stations, validate danger + obs
apply_stage(stage: StageConfig) — Ghi đè map_size, max_steps, n_victims, n_debris, n_danger_total, station_capacity, re-sync
map_diagonal (property) — sqrt(2) × map_size
grid_cell_size (property) — map_size / grid_size
save(path) / load(path) — JSON serialization (handle np.inf cho radiation)


config/env.py — EnvConfig
Attributes:

map_size: int = 100 → HARD: 250
grid_size: int = 100 → luôn = map_size
dt_seconds: float = 1.0
max_steps: int = 600 → HARD: 2500
n_uav: int = 4
n_stations: int = 2
charge_radius_m: float = 3.0 → alias charge_radius
station_capacity: int = 1 (HARD)
min_station_spacing_m: float = 15.0
max_place_attempts: int = 1000
min_object_spacing_m: float = 2.5
victim_clearance_m: float = 1.5
victim_near_dist_m: float = 6.0
victim_min_dist_m: float = 1.0
uav_spawn_radius_m: float = 3.0
placement_relax_threshold: float = 0.7
placement_relaxed_spacing_m: float = 1.5
allow_partial_obstacles: bool = True
warn_on_skipped_objects: bool = False
deterministic_eval: bool = False
eval_seed: int = 42

Properties (backward compat): dt, charge_radius, min_station_spacing, station_min_boundary_dist, victim_clearance, victim_near_dist, victim_min_dist, uav_spawn_radius, min_object_spacing, map_area_m2, cell_size_m

config/uav.py — UAVConfig
Physics:

z_min_m: float = 3.0, z_max_m: float = 40.0
max_speed_xy_mps: float = 5.0, max_speed_z_mps: float = 2.0
collision_radius_m: float = 0.5

Battery drain (% per SECOND):

drain_xy_pct_per_s: float = 0.10 — horizontal movement
drain_z_up_pct_per_s: float = 0.15 — climbing
drain_z_down_pct_per_s: float = 0.03 — descending
drain_idle_pct_per_s: float = 0.05 — hovering
charge_rate_pct_per_s: float = 1.5

Battery thresholds (%):

battery_return_pct: float = 10.0 — auto-return ≤ 10%
battery_ready_pct: float = 80.0 — deploy khi ≥ 80%
battery_dead_pct: float = 0.0 — DISABLED
battery_warning_pct: float = 20.0 — r_battery_20 trigger
battery_critical_pct: float = 10.0 — r_battery_10 trigger
battery_emergency_pct: float = 40.0 — ⚠️ FleetManager force RETURNING < 40%

Fleet policy: reserve_ratio: float = 0.2, min_reserve: int = 2
Properties (backward compat): z_min, z_max, max_speed_xy, max_speed_z, collision_radius, charge_rate, battery_return_threshold, battery_ready_threshold, battery_dead_threshold, battery_penalty_emergency, battery_low_20, battery_critical_10, battery_critical_5

config/sensor.py — SensorConfig
Attributes:

comm_range_m: float = 30.0
hfov_deg: float = 90.0
p_detect_base: float = 0.95
p_detect_decay: float = 0.04 — exp(-decay × altitude)
enable_noise: bool = True
motion_blur_coeff: float = 0.06
base_miss_rate: float = 0.03

Properties: fov_tan = tan(hfov/2), fov_radius_at_altitude (closure), comm_range

config/entity.py — VictimConfig, ObstacleConfig, DangerZoneConfig
VictimConfig:

n_victims_min: int = 5, n_victims_max: int = 20
injured_ratio_min/max: float = 0.4/0.7
injured_urgency_min/max: float = 4.0/5.0
mobile_urgency_min/max: float = 1.0/3.0
mobile_speed_min/max_mps: float = 0.2/0.4
mobile_dir_change_steps: int = 20

ObstacleConfig:

n_debris: int = 6 → HARD: 30
debris_width_min/max_m: float = 2.0/5.0 — diameter
debris_height_min/max_m: float = 3.0/8.0
n_danger_total: int = 2 → HARD: 8

DangerZoneConfig:

heights: Dict — gas:3, fire:15, smoke:25, collapse:10, radiation:inf
penalties: Dict — gas:-3, fire:-3, smoke:-2.5, collapse:-1, radiation:-5
max_counts: Dict — gas:3, fire:2, smoke:2, collapse:3, radiation:1
widths: Dict — diameter ranges per type
validate() — kiểm tra key consistency giữa heights/penalties/counts/widths
danger_types (property)


config/reward.py — RewardConfig v4.0
Design philosophy (Anti-exploit): Coverage là signal CHÍNH. Landing là survival, không phải goal.
Chính:

r_coverage_delta: float = 30.0 — dominant signal
r_victim_base: float = 50.0
r_terminal_base: float = 300.0, terminal_bonus_cap: float = 200.0

Battery penalties (per step):

r_battery_20: float = -0.5 (≤ 20%), r_battery_10: float = -2.0 (≤ 10%), r_battery_5: float = -8.0 (≤ 5%)
r_battery_dead: float = -50.0 — one-time

Collision/Proximity:

r_collision_obstacle: float = -15.0 — one-time
r_proximity_1m/2m/3m: float = -1.0/-0.3/-0.05
proximity_penalty_cap: float = -3.0

Time: r_time_penalty: float = -0.2
Landing v4.0 (anti-exploit):

r_landing_success: float = 5.0 — fixed, no bonus (giảm từ 20)
r_approach_weight: float = 0.05 — tiny nudge (giảm từ 0.5)
r_hover_penalty: float = -2.0

Caps/Clips:

step_penalty_cap: float = -8.0
step_reward_clip_min/max: float = -30.0/+200.0
enable_distance_shaping: bool = True
distance_shaping_max_per_uav: float = 2.0


config/obs.py — ObsConfig + ObsSchemaConfig
ObsSchemaConfig — định nghĩa số dims per feature group:

SELF_FEATURES: int = 11
STATION_FEATURES_PER: int = 4
TEAMMATE_FEATURES_PER: int = 3
OBSTACLE_FEATURES_PER: int = 3
VICTIM_FEATURES_PER: int = 5
COVERAGE_FEATURES: int = 3
GLOBAL_FEATURES: int = 10

ObsConfig:

n_obs_victims: int = 5, n_obs_obstacles: int = 8
n_tracked_teammates: int = 3 — CommSensor dùng tên này
local_cov_small: int = 15, local_cov_large: int = 30
max_uav: int = 8 — critic padding
n_stations: int = None — auto-sync từ AppConfig.post_init
schema: ObsSchemaConfig

Computed dims (properties):

self_dim = 11, station_dim = 8 (2×4), team_dim = 9 (3×3)
obstacle_dim = 24 (8×3), victim_dim = 25 (5×5), coverage_dim = 3
actor_dim = 68 (tổng)
global_dim = 10, critic_dim = 554 (8×68+10)

⚠️ NOTE: n_tracked_uavs (property) = alias của n_tracked_teammates cho backward compat.

config/train.py — TrainConfig
MAPPO hyperparams:

mappo_rollout_length: int = 2048 — override bởi auto_compute_config
mappo_n_epochs: int = 8
mappo_batch_size: int = 256 — override bởi auto_compute_config
mappo_clip_epsilon: float = 0.15
mappo_gamma: float = 0.99, mappo_gae_lambda: float = 0.95
mappo_lr_actor: float = 2e-4, mappo_lr_critic: float = 5e-4
mappo_max_grad_norm: float = 0.5, mappo_entropy_coeff: float = 0.005
mappo_actor_hidden: tuple = (256, 256), mappo_critic_hidden: tuple = (512, 256)
mappo_activation: str = 'tanh', mappo_use_layer_norm: bool = True

MASAC hyperparams:

masac_buffer_capacity: int = 500_000, masac_batch_size: int = 256
masac_lr_actor/critic/alpha: float = 3e-4
masac_gamma: float = 0.99, masac_tau: float = 0.005
masac_alpha_init: float = 0.2, masac_auto_alpha: bool = True
masac_warmup_steps: int = 1000, masac_update_every: int = 1
masac_actor_hidden: tuple = (256, 256), masac_critic_hidden: tuple = (400, 300)

MATD3 hyperparams:

matd3_buffer_capacity: int = 500_000, matd3_batch_size: int = 256
matd3_lr_actor/critic: float = 3e-4
matd3_gamma: float = 0.99, matd3_tau: float = 0.005
matd3_policy_delay: int = 2
matd3_explore_noise: float = 0.1, matd3_target_noise: float = 0.2, matd3_noise_clip: float = 0.5
matd3_warmup_steps: int = 1000
matd3_actor_hidden: tuple = (256, 256), matd3_critic_hidden: tuple = (400, 300)

Reproducibility: n_seeds: int = 5, seeds: [42, 123, 456, 789, 1011]

config/curriculum_config.py — StageConfig + STAGE_HARD
StageConfig attributes: name, map_size, n_uav, n_victims_min/max, n_debris, n_danger_total, station_capacity, max_steps, min_episodes, advance_coverage, advance_victims
StageConfig properties: map_area_m2, coverage_pressure_m2_per_uav, victim_density_per_1000m2, obstacle_density_per_1000m2, steps_per_m2, describe()
STAGE_HARD (active):

map_size=250, n_uav=4, n_victims_min=30, n_victims_max=40
n_debris=30, n_danger_total=8, station_capacity=1, max_steps=2500

_verify_stages() — Auto-run on import, kiểm tra victim density ∈ [0.45, 0.65]/1000m² và coverage pressure tăng dần.
⚠️ NOTE: STAGE_EASY và STAGE_MEDIUM đang bị comment. CURRICULUM_STAGES list hiện chỉ reference STAGE_HARD nhưng _verify_stages sẽ fail vì thiếu EASY/MEDIUM. Chỉ dùng apply_stage(STAGE_HARD) trực tiếp.

📁 utils/
utils/geometry.py — 9 hàm vectorized
HàmInputOutputGhi chúdist_2d(pos1, pos2)[x,y,...] × 2floatXY onlydist_3d(pos1, pos2)[x,y,z] × 2float3Dnormalize_angle(angle)float radfloat ∈ [-π,π]O(1) modulocompute_bearing(from_pos, from_vel, to_pos)arraysfloat ∈ [-π,π]relative bearingcheck_los_2d(pos1, pos2, obstacles)arrays, listboolTrue = clear; hỗ trợ tuple/blocks_los/polygon_line_intersects_circle(p1, p2, center, radius)ndarraysboolhelper nội bộget_circle_cells(center, radius, grid_size, map_size)array, floatsndarray(N,2)Vectorized meshgrid, 10× fasterget_circle_cells_legacy(...)samendarray(N,2)Loop version, DEPRECATEDget_relative_position(from_pos, to_pos)arraysndarray(3,)[dx,dy,dz]clip_position(pos, min_bounds, max_bounds)arraysndarrayClamp
get_circle_cells optimization: meshgrid + squared distance (tránh sqrt) → ~0.2ms vs ~2ms.

utils/logger.py — EpisodeLogger v2.0, TrainingLogger
EpisodeLogger:
Attributes:

episode_id, seed, start_time
total_reward, coverage_rate, victims_found, total_victims, episode_length
collision_events: List[Dict] — {step, uav_id, obstacle_id, type, pos, height}
collision_obstacle, collision_uav, collision_proximity, battery_deaths, danger_zone_entries, hot_swaps
landing_events: List[Dict] — {uav_id, step, battery_before, battery_after, charge_amount}
total_landings, total_charge_time, per_uav_landings: Dict[int, int]
_reward_breakdown_accum: Dict[str, float] — NEW v2.0: accumulate breakdown per component

Methods:

log_step(rewards, coverage, breakdown=None) — cộng dồn reward, max coverage, accumulate breakdown nếu có
log_event(event_type) — collision_obstacle/uav/proximity, victim_found, battery_death, danger_zone, hot_swap
set_total_victims(n)
log_landing(uav_id, step, battery_before, battery_after)
log_charging_step(uav_id)
log_collision(uav_id, step, obstacle_info)
finalize() → Dict — trả về metrics đầy đủ bao gồm landing fields, rewards_breakdown (abs sum), rewards_breakdown_pct (% của |total|), success: bool(coverage ≥ 0.9)

TrainingLogger:

verbose, window_size, all_metrics, recent_rewards/coverage/success/lengths, converged, convergence_episode
log_episode(metrics), _check_convergence(episode), _print_summary(last_n), get_stats(last_n), save(filepath), load(filepath)

Module-level: compare_training_runs(runs, labels) — in bảng so sánh reward/coverage/success/convergence

📁 entities/
entities/uav.py — UAV, UAVState
UAVState enum: ACTIVE, RETURNING, CHARGING, DEPLOYING, DISABLED
UAV attributes:

id, pos: ndarray[3], vel: ndarray[3], battery: float [0,100]
battery_pct (property) — alias cho battery
state: UAVState, target_station: ChargingStation | None
battery_death: bool, steps_alive, distance_xy, distance_3d, victims_found: int
_prev_state: UAVState — track previous state
pybullet_body_id: Optional[int]

Methods:

apply_action(action[3]) — chỉ ACTIVE; scale [-1,1] → velocity; clamp pos [0,map_size], altitude [z_min, z_max]
auto_navigate(target_pos) — RETURNING/DEPLOYING; no-overshoot; altitude clip per state: RETURNING→[0,z_max], CHARGING→[0,0.5m], DEPLOYING→[0,z_max]
update_battery(stations) — CHARGING: charge; ACTIVE/RETURNING/DEPLOYING: drain; DISABLED: skip
_do_drain() — proportional to velocity, × dt_seconds
_do_charge(stations) — via target_station hoặc nearest in-range
get_battery_penalty() — legacy method, progressive penalty
get_fov_radius() — pos[2] × fov_tan
get_state_onehot() → ndarray(5,) — [ACTIVE, RETURNING, CHARGING, DEPLOYING, DISABLED]
set_state(new_state) — DISABLED terminal; CHARGING→ACTIVE chỉ khi battery ≥ battery_ready_pct
find_nearest_station(stations) — ưu tiên available, fallback nearest
needs_charging(), is_ready_to_deploy() (predicates)
is_active/returning/charging/deploying/disabled/operational() (predicates)
to_dict() — JSON-safe

⚠️ NOTE: _write_self() trong obs_builder chỉ dùng get_state_onehot()[:4], không phải cả 5 dims.

entities/charging_station.py — ChargingStation
Attributes:

id, pos: ndarray[3] (z=0), capacity, charge_radius, charge_rate
current_occupants: List[UAV], occupant_ids: Set[int] — O(1) lookup

Methods:

in_range(uav_pos) — dist_xy ≤ charge_radius AND z ≤ 0.5m
try_occupy(uav) → bool — False nếu full
release(uav) → bool
has_uav(uav) → bool — O(1) via occupant_ids
charge(uav) → float — out of range → release; full → release; sạc min(rate, 100-bat)
force_release_all() — dùng khi reset episode
is_full(), is_available(), get_occupancy(), get_occupancy_ratio()


entities/victim.py — BaseVictim, InjuredVictim, MobileVictim
BaseVictim attributes:

id, pos: ndarray[3] (z=0), urgency: float [1,5]
is_found: bool = False, found_at_step: int = None, found_by_uav: int = None

BaseVictim methods:

step(obstacles) — abstract
update(step_count, obstacles) — alias cho step (bridge interface cho logic_backend)
mark_found(step, uav_id) — set flags, gọi _on_found()
_on_found() — hook cho subclass
get_reward_value() — r_victim_base × (urgency/5.0)

InjuredVictim: Stationary, urgency ∈ [4.0, 5.0], speed=0.0
MobileVictim: Random walk, urgency ∈ [1.0, 3.0]

speed ∈ [0.2, 0.4] m/s, direction, move_timer
_on_found() — freeze speed=0
step() — check is_found TRƯỚC khi move; boundary bounce; obstacle check (chỉ Debris, không DangerZone)
_check_obstacle_block(new_pos, obstacles) — chỉ isinstance Debris


entities/obstacle.py — Debris, DangerZone
Debris — static obstacle, multi-shape:

id, pos: ndarray[3] (z=0), height_3d, shape: str — "circle"/"rectangle"/"polygon"
Shape params: radius | width, height_2d, rotation | vertices
polygon: ShapelyPolygon | None — built on init nếu Shapely available
penalty = cfg.reward.r_collision_obstacle

Methods:

in_zone_2d(pos_2d) — circle: dist check; rect/poly: Shapely covers() (includes boundary)
causes_collision(uav_pos) — in_zone_2d AND uav.z < height_3d
blocks_los(pos1, pos2) — altitude check → XY intersection (circle: _line_intersects_circle; rect/poly: Shapely)
get_distance_to_edge(pos_2d) — khoảng cách đến cạnh gần nhất
_get_fallback_radius() — bounding circle khi không có Shapely

DangerZone — multi-shape:

id, pos, danger_type, max_height, penalty, shape, polygon
is_inside(uav_pos) — in_zone_2d AND uav.z < max_height
blocks_los() — chỉ fire và smoke block LOS
get_sensor_modifier() — smoke:0.40, fire:0.55, collapse:0.70, gas:0.85, radiation:0.95
get_battery_modifier() — fire:0.05, others:0.0


📁 core/
core/coverage_map.py — CoverageMap
Attributes:

grid: ndarray(bool, [GS, GS]) — explored cells
timestamps: ndarray(int32, [GS, GS]) — last scan step
first_scan: ndarray(int32, [GS, GS]) — first scan step (-1 if never)
scan_count: ndarray(int32, [GS, GS])

Methods:

reset() — grid=False, timestamps=0, first_scan=-1, scan_count=0
mark_explored(uav_pos, fov_radius, step) — vectorized via get_circle_cells; FIX: timestamps chỉ update nếu mới hơn (preserve gradient); track first_scan
get_coverage_rate() → float [0,1]
get_coverage_percent() → float [0,100]
get_local_coverage(pos, radius) → float [0,1]
get_staleness(pos, radius, step) — unexplored cells = max_steps (không bỏ qua)
get_staleness_normalized(...) — normalize theo decay_threshold (không phải max_steps)
get_freshness(...) — 1 - staleness_normalized
get_coverage_with_decay(step, decay_threshold=200) — vùng quét > threshold → không tính
get_rescan_count(pos, radius) → float — trả về float (không int) để preserve precision
get_nearest_unexplored(pos) → ndarray | None — O(N) full scan
get_nearest_stale(pos, step, threshold=200) → ndarray | None — O(N)
get_stats(step) → Dict, to_dict(step) → Dict, get_grid_snapshot() → Dict


core/map_generator.py — MapGenerator v4.1
⚠️ Key fix v4.1: Config widths = diameters. Generator converts: radius = width / 2.0
Constants:

_VICTIM_BOUNDARY_MARGIN = 2.0, _OBSTACLE_MIN_STATION = 3.0
_ROTATION_ANGLES = [0, 90, 180, 270]

Methods:

generate(n_victims_override, seed) → Dict — trả về {stations, debris, danger_zones, victims, uav_spawns, seed, n_victims}
_place_stations(rng) — min_spacing validation, fallback to corners
_place_debris(stations, rng) — shape 40% circle, 40% rect, 20% poly; radius=width/2; progressive relaxation
_place_danger_zones(existing_objects, rng) — shape 50% circle, 50% rect; radius=width/2
_spawn_victims(n, obstacles, danger_zones, rng) — 80% near debris cho injured, 40% near cho mobile
_find_valid_victim_pos(...) — tránh obstacles + danger zones + other victims
_spawn_group(n, victim_type, ...) — batch spawn với fallback
get_uav_spawns(stations, n_total, rng) — spawn quanh stations
get_map_statistics(map_data) → Dict — density metrics + clustering + danger_coverage_pct
_check_station_clearance, _check_obstacle_spacing, _check_spacing_fallback
_get_bounding_radius, _get_or_create_polygon (Shapely cache), _create_shapely_polygon
_generate_random_convex_polygon(center, avg_radius, irregularity, spikiness, n_vertices, rng)


core/fleet_manager.py — FleetManager
Vai trò: Enforce safety (không thể ignore bởi RL) + suggest deployments (RL có thể ignore).
Attributes:

n_total, n_reserve
all_uavs: List[UAV], stations: List[ChargingStation]
_uav_return_locks: Dict[int, bool] — hysteresis cho force RETURNING
_episode_forced_returns, _episode_disables: int
_cached_n_active: int — cache để tránh O(n²)

Methods:

reset(all_uavs, stations) — set n_reserve = max(min_reserve, ceil(n_total × reserve_ratio))
enforce_safety_constraints() → Dict — gọi mỗi step, KHÔNG thể ignore:

battery ≤ 0 → DISABLED + battery_death = True
battery < emergency_pct AND ACTIVE → RETURNING (hysteresis lock)
battery ≥ ready AND CHARGING AND n_active < n_total-1 → ACTIVE (auto-deploy)
PERF FIX: tính n_active 1 lần, update incremental


suggest_deployments(target_active), suggest_returns()
step() → Dict — {enforced, suggestions, priority_hints, spatial}
get_deployable_uavs() — sorted by battery desc, trừ n_reserve
get_best_deployable(prefer_station, require_min_battery)
get_battery_stats() → Dict — mean/min/max/std/critical/low/emergency counts
get_mission_priority_hints() → Dict — operational_ratio, reserve_health, station_pressure
get_spatial_awareness() → Dict — active_positions, center_of_mass, spread_radius
get_episode_summary() → Dict — forced_returns, disables
count_by_state(), get_stats(), get_fleet_incentives(), is_episode_over()


📁 sensors/
sensors/fov_sensor.py — FOVSensor
Attributes: cfg, _fov_tan, _p_base, _p_decay, _n_victims, _n_obstacles, _enable_noise, _motion_blur_coeff, _base_miss_rate, _rng
Methods:

set_seed(seed) — reproducible evaluation
calculate_fov_radius(altitude) → float
calculate_detection_prob(altitude, uav_speed, env_factor, victim_factor) → float [0,1]

Step 1: p_base × exp(-decay × alt)
Step 2: × env_factor (smoke/fire degradation)
Step 3: × (1 - motion_blur × speed_ratio)
Step 4: × victim_factor (injured=1.15, mobile≈0.75-0.95)
Step 5: × (1 - base_miss_rate)


_get_env_factor(victim_pos, obstacles) — min modifier nếu trong danger zone
_get_victim_factor(victim) — InjuredVictim:1.15, MobileVictim: 1-speed×0.5
check_detected(uav, victim, obstacles) → bool — dist → LOS → noise prob → Bernoulli
scan_victims(uav, victims, obstacles) → ndarray(25,) — top-5 nearest in FOV; features: [rel_x/fov_r, rel_y/fov_r, dist/fov_r, urgency/5, is_found]
scan_obstacles(uav, obstacles) → ndarray(24,) — top-8 nearest; features: [rel_x/fov_r, rel_y/fov_r, type_id (0=Debris, 1=DangerZone)]


sensors/comm_sensor.py — CommSensor
Attributes: _n_tracked, _dims_per_uav = 3, _comm_range, _z_max
Methods:

scan(ego_uav, all_active_uavs) → ndarray(9,) — top-3 trong comm_range; features: [norm_dist, norm_bearing, norm_alt]
get_n_in_range(ego_uav, all_uavs) → int
get_teammates_in_range(ego_uav, all_uavs) → List[UAV]


📁 observation/
observation/obs_builder.py — ObservationBuilder, ObsResult
ObsResult: actor_obs: Dict[int, ndarray(68)], critic_obs: ndarray(554)
ObservationBuilder:
Attributes: coverage_map, cfg, fov_sensor, comm_sensor, precomputed slices[6], _actor_bufs: Dict[int, ndarray], _critic_buf
Slices:

slices[0]: [0:11] self, slices[1]: [11:19] stations, slices[2]: [19:28] teammates
slices[3]: [28:40] obstacles, slices[4]: [40:65] victims, slices[5]: [65:68] coverage

Methods:

build_actor_obs(uav, all_uavs, stations, victims, obstacles, step) → ndarray(68) — gọi 6 private writers
build_all(all_uavs, stations, victims, obstacles, step) → ObsResult — DISABLED → zero obs; critic: stack 8 UAVs (sorted by id) + 10 global
Private writers: _write_self, _write_stations, _write_teammates, _write_obstacles, _write_victims, _write_coverage
Debug mode: NaN/Inf/shape check chỉ khi cfg.env.debug_obs = True
Critic global dims [544:554]: n_active/n, n_charging/n, n_disabled/n, n_alive/n, mean_bat, std_bat, min_bat (÷100), global_cov, victims_found_rate, time_remaining


📁 rewards/
rewards/baseline_reward.py — BaselineReward v4.0
Design: Anti-exploit. Coverage dominant. Landing là survival.
Reward components (17 total):
ComponentValueLoạicoverage_delta+30 per 1% increaseDense, shared/n_activevictim_found+50 × scale × urgency/5; scale = 2-coverage_rateSparse, individualdistance_shaping±2.0 cap, weight=0.1, delta-basedDense, individualbattery_penalty-0.5/-2.0/-8.0 per stepDense, individualbattery_dead-50 one-timeSparse, individualcollision_obstacle-15 one-timeSparse, individualproximity-1.0/-0.3/-0.05 per stepDense, pairwisedanger_zone-1.0 to -5.0 per stepDense, individualfleet_incentive0.0 (deprecated)—time_penalty-0.2 per active UAVDenseterminal300 base + ≤200 bonusSparsepenalty_cap_adjustmentadditiveCappinglanding_reward+5 one-time (v4.0, no early bonus)Sparsehover_penalty-2.0 per stepDenseapproach_reward0.05 × (1-norm_dist)Denseraw_total, totalcomputed—
Terminal bonus formula:

coverage_bonus = terminal_cap × 0.50 × coverage_rate
victim_bonus = terminal_cap × 0.30 × found_ratio
time_bonus = terminal_cap × 0.10 × (1-time_ratio)
battery_bonus = terminal_cap × 0.10 × mean_bat/100
clipped to [0, terminal_cap=200]

Landing v4.0:

Tier 3: one-time +5 khi CHARGING (not landed before this episode)
Tier 1: approach_weight=0.05 × (1-norm_dist) nếu battery ≤ 30% AND (ACTIVE/RETURNING)
Tier 2: -2.0 nếu ACTIVE + trong landing_range + battery ≤ 30%

Public API:

reset() — clear per-episode sets (_battery_death_penalized, _collision_penalized, _prev_min_dist, _landed_uavs)
compute(...) — global reward dict (cho logging)
compute_per_uav(uav, ...) — per-agent reward (cho training); coverage/n_active, victim individual
get_component_names(), summarize(reward_dict)

Module-level stateless functions:

_coverage_delta_reward(prev, cur, weight)
_victim_found_reward(newly_found, r_base, coverage_rate) — dynamic scaling: scale = 2.0 - coverage_rate
_battery_penalty_single(uav, reward_cfg, uav_cfg)
_battery_urgency_shaping(uav, stations, map_size) — penalty tỉ lệ dist × severity, clip [-4, 0]
_proximity_reward(active_uavs, ...), _proximity_reward_single(uav, ...)
_assert_no_nan_inf(value, label)


📁 env_setup/
env_setup/backends/base_backend.py — BaseBackend (ABC)
Abstract interface với 4 methods: reset(map_data), apply_actions(actions), step_physics(), step_world(), get_state() → Dict

env_setup/backends/logic_backend.py — LogicBackend
Entities: uavs, victims, stations, obstacles: List
Sub-systems: _cov_map: CoverageMap, _fleet_mgr: FleetManager, _fov_sensor: FOVSensor
reset(map_data):

Deterministic eval: seed np.random và FOVSensor
Build entities từ map_data
_cov_map.reset(), _fleet_mgr.reset()

apply_actions(actions: Dict[int, ndarray(4)]):

ACTIVE: parse [vx,vy,vz] + land_signal

land > 0.5 AND battery ≤ 40.0 → find_nearest_station → set RETURNING + auto_navigate(z=0)
else → apply_action(move[:3])
pin còn nhiều → ignore land, apply move; log 1 lần per episode


RETURNING: auto_navigate(target_station, z=0) hoặc find nearest
DEPLOYING: set_state(ACTIVE) immediately
CHARGING/DISABLED: no movement

step_physics():

RETURNING + in_range(station) → try_occupy → CHARGING
CHARGING → station.charge(uav)
ACTIVE/DEPLOYING → uav.update_battery(stations)

step_world():

_fleet_mgr.step() (enforce_safety_constraints)
v.update(step_count, obstacles=self.obstacles) — obstacle-aware movement
_cov_map.mark_explored() cho non-DISABLED
_fov_sensor.check_detected() cho ACTIVE + RETURNING

build helpers: _build_stations, _build_obstacles (multi-shape Debris+DangerZone), _build_victims, _build_uavs (từ map_data["uav_spawns"] trực tiếp — FIX 4.2; tất cả spawn ACTIVE)

env_setup/base_env.py — SARBaseEnv
Spaces: observation_space = Box(-inf, inf, (68,)), action_space = Box(-1, 1, (4,))
reset(seed) → (Dict[int, ndarray(68)], Dict):

Generate map, reset backend, build obs_builder
info có: seed, n_uav, n_victims, n_obstacles, global_obs

step(actions) — Execution order (CRITICAL):

apply_actions → step_physics → step_world
_step_count += 1
Tracking landing transitions (prev_state → CHARGING)
Coverage update
Collect newly_found
_check_done() TRƯỚC reward (BUG-ENV-06 fix)
compute_per_uav() — per-agent rewards (DISABLED → 0)
compute() — global reward (logging only)
Accumulate episode_reward_sum
_log_step() với global_breakdown
_build_obs_dict()
Return

_check_done() → str | None:

"coverage" nếu coverage ≥ 0.9
"victims" nếu all found
"disabled:battery_death" / "disabled:other"
None → tiếp tục

⚠️ KHÔNG auto-reset: Reset được quản lý bởi vec_env worker hoặc _EnvWrapper.
info keys: coverage, victims_found, victims_total, step, n_active/charging/disabled/returning/deploying, success, done_reason, rewards_breakdown, newly_found_ids, battery_stats, global_obs, episode (chỉ khi terminal)

env_setup/sar_pettingzoo_env.py — SARPettingZooEnv
Thin wrapper over SARBaseEnv, convert int keys ↔ str keys ("uav_0", ...).

possible_agents = ["uav_0", ..., "uav_3"]
action_space(agent) = Box(-1, 1, (4,)) — 4-dim hybrid
observation_space(agent) = Box(-inf, inf, (68,))
reset() → (Dict[str, ndarray], Dict[str, Dict])
step(actions: Dict[str, ndarray]) → (obs, rewards, terminations, truncations, infos)
Tất cả agents terminate đồng thời
infos['uav_0']['global_obs'] = ndarray(554)
make_parallel_env(), make_aec_env() — factory functions


env_setup/vec_env.py — VectorizedEnv
env_worker(pipe, config, seed) — Worker process:

rng = np.random.default_rng(seed) — RNG riêng mỗi worker
Initial reset với seed gốc
Auto reset với new seed sau mỗi episode (seed từ rng)
Cache last valid obs/global_obs/info
Commands: "reset", "step", "close"
Graceful error handling (EOFError, BrokenPipeError)

VectorizedEnv(config, n_envs, start_seed):

Spawn N workers via mp.get_context("spawn") (CUDA-safe)
SeedSequence để generate independent seeds per worker
reset() → (obs[E,A,68], global_obs[E,554])
step(actions[E,A,4]) → (obs, global_obs, rewards, dones, infos)
close() — send "close" + join + terminate


📁 training/algorithms/

training/algorithms/mappo/networks.py — MLP, orthogonal_init
orthogonal_init(layer, gain) — orthogonal weight init, bias=0; gain=sqrt(2) cho hidden, 0.01 cho output
MLP(input_dim, hidden_dims, output_dim, activation, use_layer_norm, output_activation):

Builds Sequential: [Linear → (LayerNorm?) → Activation] × N → Linear → (output_activation?)
activations: tanh / relu / elu
get_parameter_count(model), print_network_summary(model, name)


training/algorithms/mappo/actor.py — ActorNetwork
Architecture:

backbone: MLP(68 → 256 → 256, output_activation='tanh') — shared
movement_head: Linear(256 → 3) — gain=0.01
land_head: Linear(256 → 1) — bias=-1.0 → P(land)≈0.27 initial
log_std: Parameter(ones(3) × -0.5) — learnable, state-independent

Methods:

forward(obs) → (move_mean[B,3], move_std[B,3], land_logit[B,1])
get_action(obs, deterministic=False) → (action[B,4], log_prob[B]) — ⚠️ KHÔNG clamp trước log_prob (FIX P0): backend clip sau khi nhận action
evaluate_actions(obs, actions) → (log_prob[B], entropy[B]) — split move[:3] + land[3:]
get_land_prob(obs) → [B] — logging/viz
get_log_std(), set_log_std(value)


training/algorithms/mappo/critic.py — CriticNetwork
Architecture: global_obs[B, 554] → MLP(554→512→256→1) → value[B,1]
Methods:

forward(global_obs) → [B,1]
get_value(global_obs) → [B] — squeezed
compute_loss(global_obs, returns) → MSE scalar
compute_value_metrics(global_obs, returns) → Dict — value_loss, explained_variance, mean/std


training/algorithms/mappo/buffer.py — RolloutBuffer
Shape: [T, E, A, ...] — T=rollout_length, E=n_envs, A=n_agents
Storage arrays: observations, global_obs, actions (action_dim=4), rewards, values, log_probs, dones, advantages, returns
Methods:

add(obs, global_obs, actions, rewards, values, log_probs, dones) — thêm 1 timestep, raise RuntimeError nếu overflow
compute_gae(last_values, last_dones) — FIX P0-2: next_non_terminal[t] = 1 - dones[t+1] (không phải dones[t]); vectorized delta; GAE per-env normalize
get_batches(batch_size) → Iterator[Dict] — flatten + shuffle; FIX P1-1: n_envs=1 dùng repeat thay broadcast (40% faster); keys: obs, global_obs, actions, old_log_probs, advantages, returns
clear() — reset ptr
get_stats() → Dict


training/algorithms/mappo/trainer.py — MAPPOTrainer
_EnvWrapper(config, n_envs, seed):

n_envs=1: SARPettingZooEnv; n_envs>1: VectorizedEnv
Cache _current_obs, _current_global — không reset nếu không cần
reset(), get_current_obs(), step(actions[E,A,4]), reset_hard(), close()
FIX-P1: Khi done=True với n_envs=1 → reset ngay, cache obs mới, trả về terminal obs cho buffer sau đó không dùng

MAPPOTrainer attrs:

actor: ActorNetwork(68→4), critic: CriticNetwork(554→1)
buffer: RolloutBuffer(T×E, n_agents=4, action_dim=4)
actor_opt, critic_opt: Adam
ep_rewards, ep_lengths, ep_coverage, ep_victims: deque(maxlen=100)
_all_rewards, _all_coverage, _all_victims, _all_lengths: List — full history cho HF upload
_persist_ep_len/rew: ndarray[n_envs]
_next_log_ep, _next_checkpoint_ep: int — trigger tracking (FIX-T1: >= thay %)
hf_uploader: HFUploader | None
output_dir, checkpoint_dir: Path — Kaggle: /kaggle/working/results/mappo/{run_name}/

train(total_episodes, curriculum_manager=None, seed, log_every, checkpoint_every):

Single tqdm pbar
Loop: _rollout() → _update() → logging → checkpoint
Curriculum support nhưng mặc định None

_rollout(env, pbar, max_episodes) → Dict:

get_current_obs() — không reset
Batch inference: actor.get_action(obs_flat[E×A, 68])
critic.get_value(g_batch[E, 554])
FIX-P0-3: Clip action → re-compute log_prob với clipped action trước khi buffer.add
rews_team.reshape(n_envs) — FIX-P2: không dùng squeeze()
Shared team reward: rews_team broadcast to all agents
Per-env tracking khi dones[ei]; extract từ infos['uav_0']['episode']
GAE bootstrap với critic.get_value(last_g)

_update() → Dict:

n_epochs × minibatches
Actor: PPO clip + entropy bonus
Critic: MSE / n_agents
Track actor_loss, critic_loss, entropy, clip_fraction

save_checkpoint(episode, curriculum_manager, tag):

Lưu actor/critic state_dict + optimizers + full history
Upload HF nếu có hf_uploader
tag="final" → upload_final() + _cleanup_local()

load_checkpoint(path) → int

training/algorithms/masac/actor.py — SACActorNetwork
Architecture: backbone MLP → move_mean_head, move_log_std_head, land_head

LOG_STD_MIN = -5.0, LOG_STD_MAX = 2.0
land_head bias = -2.0 → P(land) ≈ 0.12 initial

Methods:

forward(obs) → (move_mean[B,3], move_log_std[B,3], land_logit[B,1])
get_action(obs, deterministic=False) → (action[B,4], log_prob[B]):

Movement: Normal.rsample → tanh squash → log_prob correction: log π = log N(u) - Σ log(1-a²)
Landing: Bernoulli


get_log_prob(obs, actions) → (log_prob, entropy) — un-squash via atanh, dùng trong SAC update
get_land_prob(obs)


training/algorithms/masac/twin_critic.py — TwinCriticNetwork
Input: concat(global_obs[554], action[4]) = 558-dim
Output: Q1, Q2 (twin critics để giảm overestimation)
Methods:

forward(global_obs, actions) → (Q1[B,1], Q2[B,1])
q1_value(...) — chỉ Q1 (cho actor update)
min_q(...) — min(Q1, Q2) cho target computation
compute_loss(global_obs, actions, targets) → (q1_loss, q2_loss)


training/algorithms/masac/replay_buffer.py — ReplayBuffer
Off-policy buffer. Stores (obs, global_obs, actions, rewards, next_obs, next_global, done).
Shape: [N, n_agents, obs_dim], [N, global_obs_dim], etc.
Methods:

add(obs, global_obs, actions, rewards, next_obs, next_global, done) — circular buffer
add_batch(...) — batch add từ vectorized env
sample(batch_size) → Dict — uniform random
is_ready(min_size), get_stats()


training/algorithms/masac/trainer.py — MASACTrainer
Attrs giống MAPPO nhưng thêm:

critic_target: TwinCriticNetwork — frozen target
log_alpha: Parameter, alpha_opt: Adam (nếu auto_alpha)
alpha: float — entropy temperature
target_entropy: float = -3.0 (= -(action_dim-1))
buffer: ReplayBuffer

train(total_episodes, ...):

Online collection: 1 step per env per iteration (không rollout batch)
Warmup: uniform random actions trước warmup_steps
Update mỗi update_every steps × updates_per_step lần

_update() → Dict:

Sample batch từ buffer
TD target: r + γ × (1-done) × (min_Q' - α × log π')
Update critics: MSE(Q, target)
Update actor: -mean(α × log_π - min_Q)
Update α (auto): -(log_α × (log_π + target_entropy)).mean()
Soft update target critics


training/algorithms/matd3/actor.py — TD3ActorNetwork
Deterministic policy. obs → tanh(MLP) → action[4]

Movement: tanh squash ∈ [-1,1]
Land: sigmoid output ∈ [0,1]

Methods:

forward(obs) → action[B,4]
get_action(obs, explore_noise, noise_clip, deterministic) → (action, zeros) — add Gaussian noise for exploration; binarize land at threshold
get_target_action(obs, target_noise, noise_clip) — TD3 smoothing: clip(N(0,σ), -c, c); no grad


training/algorithms/matd3/twin_critic.py — TD3TwinCriticNetwork
Giống MASAC TwinCriticNetwork. Thêm:

q1_value() — TD3: actor update dùng Q1 only (không min)
min_q() — TD3: target computation dùng min(Q1, Q2)


training/algorithms/matd3/trainer.py — MATD3Trainer
Attrs thêm:

actor_target: TD3ActorNetwork — frozen
_critic_update_count: int — đếm critic updates để trigger delayed actor update

_update() → Dict:

TD target: r + γ × (1-done) × min_Q_target(s', target_action_smoothed)
Update critics: MSE
Delayed actor update (_critic_update_count % policy_delay == 0):

actor_loss = -Q1(s, μ(s)).mean()
Soft update actor_target + critic_target




📁 Root files
hf_upload.py — HFUploader, HFDownloader
HFUploader:

__init__(token, repo_id) — đọc từ env var HF_TOKEN nếu không truyền; _ensure_repo()
upload_checkpoint(checkpoint_path, run_name, episode, metrics) — upload .pt + metrics.json
upload_final(run_name, checkpoint_path, metrics, plot_path) — upload checkpoint + metrics + optional plot
_upload_metrics(run_name, metrics, episode) — serialize + upload metrics.json

HFDownloader:

list_runs(), download_metrics(run_name, local_dir), download_checkpoint(run_name, filename, local_dir), download_all_metrics(local_dir)

_serialize(obj) — convert numpy types → Python native cho JSON

train_mappo.py — CLI Entry Point MAPPO
auto_compute_config(max_steps, n_envs, n_uav, batch_size_hint, safety_factor):

min_rollout = ceil(max_steps × safety_factor)
rollout_length = ceil(min_rollout/64) × 64 (align to 64)
batch_size = auto hoặc clip hint
returns: AutoConfig(rollout_length, buffer_capacity, batch_size, safety_margin)

set_seed(seed) — seed random, numpy, torch, cudnn deterministic, CUBLAS env
main():
cfg = AppConfig()
cfg.apply_stage(STAGE_HARD)
cfg.env.n_uav = 4
auto_cfg = auto_compute_config(...)
cfg.train.mappo_rollout_length = auto_cfg.rollout_length
cfg.train.mappo_batch_size = auto_cfg.batch_size
trainer = MAPPOTrainer(cfg, device, run_name, n_envs, hf_token, hf_repo)
trainer.train(total_episodes, curriculum_manager=None, seed, log_interval, checkpoint_interval)
CLI args: --total-episodes, --seed, --device, --run-name, --n-envs, --max-steps, --map-size, --batch-size, --safety-factor, --n-epochs, --lr-actor, --lr-critic, --log-interval, --checkpoint-interval, --hf-token, --hf-upload, --hf-upload-every

train_masac.py và train_matd3.py — CLI Entry Points
Tương tự train_mappo.py nhưng:

Không cần auto_compute_config (off-policy, buffer tự quản)
Hyperparams đọc từ cfg.train.masac_* / cfg.train.matd3_* trực tiếp
trainer = MASACTrainer(...) / MATD3Trainer(...)


run_visualization.py — CLI Visualization
Output: results/viz/{timestamp}_{policy_label}/

Per-episode GIF, combined GIF, per-frame PNG, summary.png
latest.gif và latest.png luôn overwrite

Functions:

setup_output_dir(policy_label) — tạo run_dir với timestamp
get_scripted_action(policy, step, uav_id, n_agents) — random/hover/circle
load_actor(checkpoint_path, algo, config, device) — load any algo actor; weights_only=False
create_untrained_actor(algo, config, device) — random weights
save_frame_png(frame, path, title) — matplotlib Agg
save_gif_file(frames, path, fps) — try Pillow → imageio → matplotlib animation
save_summary_plot(results, run_dir, policy_label) — 4 bar charts
run_episode(config, seed, policy_type, policy_arg, algo, device, run_dir, ep_idx) → (frames, result) — render="rgb_array", capture every step

CLI: --mode, --checkpoint, --algo, --policy, --seed, --n-episodes, --max-steps, --device, --n-uav, --fps, --no-gif

plot_compare.py — Download HF + Plot Comparison
_detect_algo(run_name), _smooth(data, window)
_plot_metric(ax, runs_data, metric_key, ...) — group by algo; single seed: raw + smoothed; multi-seed: mean ± std
plot_comparison(runs_data, save_path, window, title) — 2×3 subplots:

Row 1: Episode Reward, Coverage Rate (target 70%), Victims Found (target 80%)
Row 2: Episode Length, Sample Efficiency (reward vs env steps), Summary Table

CLI: --runs, --local-dir, --save, --window, --from-local

plot_local.py — Plot từ local metrics.json
Đọc trực tiếp từ ./mappo_s42/metrics.json v.v., không cần HF. Tạo 2×3 plot giống plot_compare.py nhưng đơn giản hơn. Dùng trong notebook (craft.ipynb).

5. 🔄 TRAINING FLOW (Tổng thể)
train_{algo}.py
├── AppConfig() + cfg.apply_stage(STAGE_HARD)
├── [MAPPO only] auto_compute_config() → rollout_length, batch_size
└── {MAPPO/MASAC/MATD3}Trainer(config, device, n_envs)
    ├── Actor + Critic networks
    ├── Buffer (RolloutBuffer hoặc ReplayBuffer)
    └── _EnvWrapper (n_envs=1→PettingZoo, >1→VectorizedEnv)

WHILE episodes < total_episodes:
    [MAPPO]
    ├── _rollout() → collect T steps across E envs
    │   ├── get_current_obs() [no reset]
    │   ├── actor.get_action → clip → re-compute log_prob → buffer.add
    │   └── buffer.compute_gae(bootstrap, last_dones)
    ├── _update() → n_epochs × PPO clip + critic MSE

    [MASAC/MATD3]
    ├── 1 env step → buffer.add_batch()
    ├── IF ready AND past warmup AND step % update_every:
    │   ├── [MASAC] update twin_critic + actor + alpha + soft_update_target
    │   └── [MATD3] update twin_critic + [delayed] actor + soft_update targets

    ├── Logging (>= trigger, không dùng %)
    └── Checkpoint save (absolute path + optional HF upload)

FINALIZE:
├── save_checkpoint(tag="final")
└── env.close()

6. ⚡ SINGLE STEP FLOW
env.step(actions: Dict[str, ndarray(4)])  ← PettingZoo API (str keys)
↓ Convert str→int keys
LogicBackend.apply_actions(Dict[int, ndarray(4)]):
    ACTIVE: land>0.5 AND bat≤40% → set RETURNING + auto_navigate(z=0)
            else → apply_action(move[:3])
    RETURNING: auto_navigate(target_station, z=0)
    DEPLOYING: set_state(ACTIVE) immediately
    CHARGING/DISABLED: no movement
↓
step_physics():
    RETURNING + in_range(3m, z≤0.5) → try_occupy → CHARGING
    CHARGING → station.charge(uav)
    ACTIVE/DEPLOYING → uav.update_battery() (drain)
↓
step_world():
    fleet_manager.enforce_safety_constraints()
        battery=0 → DISABLED
        battery<40% AND ACTIVE → RETURNING
        battery≥80% AND CHARGING AND n_active<n_total-1 → ACTIVE
    victim.update(step, obstacles) → mobile movement + obstacle check
    coverage_map.mark_explored() for non-DISABLED
    fov_sensor.check_detected() for ACTIVE+RETURNING
↓
step_count += 1
_check_done() → "coverage"/"victims"/"disabled:*"/None
compute_per_uav() → rewards_dict[int, float]  (DISABLED → 0)
compute() → global_reward (logging only)
_log_step() với global_breakdown accumulation
_build_obs_dict() → (actor_obs_dict, critic_obs[554])
return (obs, rewards, done, truncated, info)
info['uav_0']['global_obs'] = critic_obs[554]

7. 📊 TRẠNG THÁI HIỆN TẠI
✅ Đã hoàn thành

Phase 1: Core infrastructure đầy đủ (config, entities, core, sensors, obs, rewards, env)
Phase 2: MAPPO hoàn chỉnh, trainable, stable
Phase 3: MASAC + MATD3 hoàn chỉnh (Kaggle-optimized, no viz during training)
Hybrid action space [vx, vy, vz, land] cho cả 3 algorithms
BaselineReward v4.0 anti-exploit (landing = 5, coverage dominant = 30)
HuggingFace upload/download infrastructure
Visualization CLI với GIF export
Plot comparison (local + HF download)
VectorizedEnv seed progression per-episode
Multi-shape obstacles (circle/rect/polygon) với Shapely support
EpisodeLogger v2.0 với reward breakdown accumulation

⚠️ Known Issues
IssueFileSeveritySTAGE_EASY/MEDIUM commented → _verify_stages sẽ fail nếu dùng CURRICULUM_STAGEScurriculum_config.pyLow (không dùng curriculum)_write_self() dùng state_onehot[:4] thay vì cả 5 dimsobs_builder.pyLowTất cả UAV spawn ACTIVE (không có reserve pool ban đầu)logic_backend.pyLowget_sensor_modifier() nằm nhầm trong ObsConfig (là method của DangerZone)obs.pyLow
⬜ Chưa làm

Statistical comparison (Wilcoxon tests, 5 seeds)
LLM reward generation (Paper 2)
PyBullet/Isaac backends
Formal test suite