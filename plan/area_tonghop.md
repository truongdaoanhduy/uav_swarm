🚁 SAR UAV SWARM — PROJECT HANDOFF DOCUMENT v10.0
Cập nhật: Phase 1 hoàn thành 100% | Phase 2 đang triển khai (MAPPO 5/8 steps)

🎯 MỤC TIÊU NGHIÊN CỨU
Paper 1 — Algorithm Comparison
Câu hỏi: MAPPO vs MASAC vs MATD3 — thuật toán nào tốt nhất cho SAR?
Phương pháp: 3 algorithms × 5 seeds × 3 curriculum stages × 3000 episodes
Metrics: Coverage rate, Victims found, Episode reward, Sample efficiency
Paper 2 — LLM Reward vs Hand-Crafted
Câu hỏi: LLM có generate reward function tốt hơn human expert không?
Baseline: BaselineReward v3.1 (hand-crafted, nghiên cứu kỹ)
LLM: GPT-4/Claude generate reward code từ task description
Task Definition
Agents: 4 UAVs phối hợp tìm kiếm 10–36 victims (injured / mobile)
Môi trường: Debris (vật cản cứng), Danger Zones (fire/radiation/smoke/gas/collapse)
Constraint: Battery — UAV phải quay trạm sạc khi pin thấp
Reward: SHARED REWARD (cooperative) — tất cả UAVs nhận cùng reward
Curriculum: EASY (150×150m) → MEDIUM (200×200m) → HARD (250×250m)
📊 TRẠNG THÁI TỔNG QUAN
Phase Mô tả Trạng thái
Phase 1 Core Infrastructure (48 files) ✅ 100% hoàn thành
Phase 2 RL Algorithms (MAPPO/MASAC/MATD3) 🔄 Đang thực hiện (MAPPO)
Phase 3 LLM Reward Integration ⬜ Chưa bắt đầu
Phase 4 Advanced Backends (PyBullet/Isaac) ⬜ Chưa bắt đầu
Test status: 26/26 tests PASS (100% core coverage)
Actor obs dim: 68 (n_stations=2)
Critic obs dim: 554 (8×68+10)
Action dim: 3 (vx, vy, vz) ∈ [-1, 1]

📐 KEY NUMBERS
Metric Giá trị
Actor obs dim 68 (n_stations=2)
Critic obs dim 554 (8×68+10)
Action space 3 dims ∈ [-1,1]
UAV states 5: ACTIVE/RETURNING/CHARGING/DEPLOYING/DISABLED
Reward components 14
Reward type SHARED (cooperative, tất cả UAVs nhận cùng value)
Test pass 26/26
Backend speed ~1000 steps/s
Difficulty metric coverage_pressure = map_area / n_uav
Victim density ~0.53/1000m² (constant across stages)
📁 CẤU TRÚC ĐẦY ĐỦ
📁 config/ — Hệ thống cấu hình
config/**init**.py
Vai trò: Export tất cả config classes.
Export: AppConfig, EnvConfig, UAVConfig, SensorConfig, VictimConfig, ObstacleConfig, DangerZoneConfig, RewardConfig, ObsConfig, TrainConfig, StageConfig, STAGE_EASY, STAGE_MEDIUM, STAGE_HARD, CURRICULUM_STAGES

config/config.py — AppConfig
Vai trò: Master config — single source of truth, truyền vào toàn bộ hệ thống.

Thuộc tính:

Tên Kiểu Mô tả
env EnvConfig Map & physics params
uav UAVConfig Drone dynamics & battery
sensor SensorConfig FOV/Comm/Noise model
victim VictimConfig Victim spawning
obstacle ObstacleConfig Debris params
danger DangerZoneConfig Danger zone configs
reward RewardConfig 14 reward components
obs ObsConfig Observation dims
train TrainConfig RL training params + MAPPO hyperparams
viz_mode str "2d"/"3d"/"none"
viz_3d_cfg dict 3D renderer config
Methods:

Tên Mô tả
**post_init**() Auto-sync obs.n_stations = env.n_stations, validate
apply_stage(stage) Apply curriculum stage in-place (single source of truth)
map_diagonal Property: sqrt(2) × map_size
grid_cell_size Property: map_size / grid_size
save(path) Serialize sang JSON
load(path) Restore từ JSON
config/env.py — EnvConfig
Vai trò: Map, thời gian, fleet params.

Thuộc tính chính:

Tên Mặc định Mô tả
map_size 100 Kích thước map (m)
grid_size 100 Số ô lưới (sync với map_size)
dt_seconds 1.0 Timestep (s)
max_steps 600 Steps tối đa/episode
n_uav 4 Số UAV
n_stations 2 Số trạm sạc
charge_radius_m 3.0 Bán kính sạc (m)
station_capacity 2 UAVs tối đa/trạm
deterministic_eval False Fixed seed khi eval
eval_seed 42 Seed cho eval
config/uav.py — UAVConfig
Vai trò: Physics UAV + battery model.

Thuộc tính chính:

Tên Mặc định Mô tả
z_min_m 3.0 Độ cao tối thiểu (m)
z_max_m 40.0 Độ cao tối đa (m)
max_speed_xy_mps 5.0 Tốc độ ngang tối đa (m/s)
max_speed_z_mps 2.0 Tốc độ dọc tối đa (m/s)
drain_xy_pct_per_s 0.10 Drain ngang (%/s)
drain_z_up_pct_per_s 0.15 Drain leo cao (%/s)
charge_rate_pct_per_s 1.5 Tốc độ sạc (%/s)
battery_return_pct 10.0 Ngưỡng tự động về trạm (%)
battery_ready_pct 80.0 Sẵn sàng xuất phát (%)
battery_emergency_pct 5.0 Ngưỡng emergency (%)
reserve_ratio 0.2 20% swarm trong reserve
min_reserve 2 Tối thiểu 2 UAV reserve
Backward compat properties: z_min, z_max, max_speed_xy, drain_xy_max, battery_dead, charge_rate, v.v.

config/sensor.py — SensorConfig
Vai trò: FOV geometry + detection noise model.

Thuộc tính:

Tên Mặc định Mô tả
comm_range_m 30.0 Tầm liên lạc (m)
hfov_deg 90.0 Góc FOV ngang (°)
p_detect_base 0.95 P_detect tại altitude=0
p_detect_decay 0.04 Decay theo altitude
enable_noise True Bật noise model
motion_blur_coeff 0.06 Penalty khi bay nhanh
base_miss_rate 0.03 Hardware miss rate
Properties: fov_tan, comm_range

config/entity.py — VictimConfig, ObstacleConfig, DangerZoneConfig
VictimConfig:

Tên Mặc định Mô tả
n_victims_min/max 5/20 Range số victims
injured_ratio_min/max 0.4/0.7 Tỉ lệ injured
injured_urgency_min/max 4.0/5.0 Urgency injured
mobile_urgency_min/max 1.0/3.0 Urgency mobile
mobile_speed_min/max_mps 0.2/0.4 Tốc độ mobile
mobile_dir_change_steps 20 Đổi hướng mỗi N steps
ObstacleConfig:

Tên Mặc định Mô tả
n_debris 6 Số debris
debris_width_min/max_m 2.0/5.0 Footprint diameter (m)
debris_height_min/max_m 3.0/8.0 Chiều cao 3D (m)
n_danger_total 2 Tổng số danger zones
DangerZoneConfig:

heights: {gas:3, fire:15, smoke:25, collapse:10, radiation:inf}
penalties: {gas:-3, fire:-3, smoke:-1.5, collapse:-1, radiation:-5}
max_counts: Dict số lượng tối đa mỗi loại
widths: Dict {type: (min_diam, max_diam)} — đường kính, KHÔNG phải bán kính
validate(): Kiểm tra consistency
danger_types: Property list các loại
config/reward.py — RewardConfig v3.1
Vai trò: 14 reward components.

Component Giá trị Mô tả
r_coverage_delta +6.0 Per 1% coverage tăng
r_victim_base +50.0 × urgency/5 khi tìm thấy
r_battery_10 -1.0 Penalty <10%
r_battery_5 -3.0 Penalty <5%
r_battery_dead -100.0 One-time khi chết pin
r_collision_obstacle -30.0 One-time va chạm debris
r_proximity_1m -10.0 Per step khi 2 UAV <1m
r_proximity_2m -3.0 Per step <2m
r_proximity_3m -0.5 Per step <3m
proximity_penalty_cap -15.0 Cap proximity/step
r_time_penalty -0.05 Per active UAV per step
r_terminal_base +200.0 Base terminal bonus
terminal_bonus_cap +100.0 Max terminal bonus
step_penalty_cap -30.0 Tổng penalty tối đa/step
step_reward_clip_min/max -100/+100 Clip mỗi step
enable_distance_shaping True Delta-based shaping
config/obs.py — ObsSchemaConfig, ObsConfig
Vai trò: Observation dimensions.

ObsSchemaConfig constants:

SELF_FEATURES = 11
STATION_FEATURES_PER = 4
TEAMMATE_FEATURES_PER = 3
OBSTACLE_FEATURES_PER = 3
VICTIM_FEATURES_PER = 5
COVERAGE_FEATURES = 3
GLOBAL_FEATURES = 10
ObsConfig thuộc tính:

Tên Mặc định Mô tả
n_obs_victims 5 Max victims trong obs
n_obs_obstacles 4 Max obstacles trong obs
n_tracked_teammates 3 Max teammates track
local_cov_small 15 Radius nhỏ coverage (m)
local_cov_large 30 Radius lớn coverage (m)
max_uav 8 Padding critic obs
n_stations None Auto-sync từ EnvConfig
Computed properties:

self_dim = 11
station_dim = n_stations × 4 → 8 (n_stations=2)
team_dim = 9 (3×3)
obstacle_dim = 12 (4×3)
victim_dim = 25 (5×5)
coverage_dim = 3
actor_dim = 68 (tổng)
critic_dim = 554 (8×68+10)
validate(): Kiểm tra n_stations không None
config/train.py — TrainConfig ✅ UPDATED Phase 2
Vai trò: Training params + MAPPO hyperparameters.

Existing fields:

Tên Mặc định Mô tả
n_seeds 5 Số seeds
seeds [42,123,456,789,1011] Fixed seeds
confidence_level 0.95 Wilcoxon/t-test
total_episodes 3000 Episodes per stage
eval_interval 50 Eval mỗi N episodes
save_interval 100 Save checkpoint
log_window 100 Rolling mean window
MAPPO hyperparameters (Phase 2 additions):

Tên Mặc định Mô tả
mappo_rollout_length 2048 Steps per update
mappo_n_epochs 10 Epochs per update
mappo_batch_size 256 Minibatch size
mappo_clip_epsilon 0.2 PPO clip range
mappo_gamma 0.99 Discount factor
mappo_gae_lambda 0.95 GAE lambda
mappo_lr_actor 3e-4 Actor learning rate
mappo_lr_critic 1e-3 Critic learning rate
mappo_max_grad_norm 0.5 Gradient clipping
mappo_entropy_coeff 0.01 Entropy bonus
mappo_actor_hidden (256,256) Actor hidden dims
mappo_critic_hidden (512,256) Critic hidden dims
mappo_activation 'tanh' Activation function
mappo_use_layer_norm False Layer normalization
mappo_log_interval 10 Console log interval
mappo_viz_interval 100 2D viz interval
mappo_checkpoint_interval 100 Checkpoint interval
config/curriculum_config.py — StageConfig
Vai trò: Định nghĩa 3 curriculum stages.

StageConfig fields:

Field Mô tả
name "easy"/"medium"/"hard"
map_size Kích thước map (m)
n_uav Số UAV (luôn=4)
n_victims_min/max Range victims
n_debris Số debris
n_danger_total Tổng danger zones
max_steps Max steps/episode
min_episodes Tối thiểu episodes trước advance
advance_coverage Ngưỡng coverage để advance
advance_victims Ngưỡng victim rate để advance
Properties: map_area_m2, coverage_pressure_m2_per_uav, victim_density_per_1000m2, describe()

3 stages:

Stage Map Pressure Max Steps Advance
STAGE_EASY 150×150m 5,625 m²/UAV 300 cov≥70%, vic≥80%
STAGE_MEDIUM 200×200m 10,000 m²/UAV 350 cov≥65%, vic≥75%
STAGE_HARD 250×250m 15,625 m²/UAV 400 cov≥60%, vic≥70%
📁 utils/ — Tiện ích
utils/geometry.py
Vai trò: 9 geometry functions, vectorized với NumPy.

Hàm Output Mô tả
dist_2d(pos1, pos2) float Khoảng cách XY
dist_3d(pos1, pos2) float Khoảng cách XYZ
normalize_angle(angle) float Về [-π, π]
compute_bearing(from_pos, from_vel, to_pos) float Góc tương đối
check_los_2d(pos1, pos2, obstacles) bool Line-of-sight
get_circle_cells(center, radius, grid_size, map_size) ndarray(N,2) FOV cells (10× faster)
get_relative_position(from_pos, to_pos) ndarray Vector relative
clip_position(pos, min_bounds, max_bounds) ndarray Boundary clamp
utils/logger.py
Vai trò: Episode và training logging.

EpisodeLogger:

Method Mô tả
log_step(rewards, coverage) Cập nhật reward sum và coverage max
log_event(event_type) Track events: collision, victim_found, battery_death, danger_zone
set_total_victims(n) Set tổng victims
finalize() → Dict metrics: reward, coverage_rate, victims, collisions, success
TrainingLogger:

Method Mô tả
log_episode(metrics) Cập nhật windows, check convergence
get_stats(last_n) Stats dict (mean/std/success_rate)
save(filepath) Lưu JSON
load(filepath) Khôi phục từ JSON
📁 entities/ — Game Objects
entities/uav.py — UAV, UAVState
Vai trò: UAV agent với physics và state machine.

UAVState enum: ACTIVE, RETURNING, CHARGING, DEPLOYING, DISABLED

UAV thuộc tính:

Tên Mô tả
id int ID
pos ndarray [x,y,z]
vel ndarray [vx,vy,vz]
battery float [0,100]
state UAVState
target_station ChargingStation hoặc None
victims_found int
battery_death bool (pin chết lần đầu)
UAV methods:

Tên Mô tả
apply_action(action) action [-1,1]³ → vel → update pos
auto_navigate(target_pos) Bay tự động (không overshoot)
update_battery(stations) Drain hoặc charge
get_fov_radius() altitude × fov_tan
get_state_onehot() ndarray(5,) one-hot
set_state(new_state) Chuyển state có validation
needs_charging() battery ≤ battery_return_pct
is_ready_to_deploy() battery ≥ battery_ready_pct
find_nearest_station(stations) Tìm trạm gần nhất có chỗ
entities/victim.py — BaseVictim, InjuredVictim, MobileVictim
Vai trò: Victim entities.

BaseVictim thuộc tính: id, pos, urgency, is_found, found_at_step, found_by_uav

BaseVictim methods:

Tên Mô tả
mark_found(step, uav_id) Set is_found=True
get_reward_value() r_victim_base × urgency/5
update(step_count, obstacles) Alias của step()
InjuredVictim: Stationary, urgency [4.0,5.0], speed=0

MobileVictim: Random walk speed [0.2,0.4] m/s, freeze khi found, đổi hướng mỗi 20 steps

entities/charging_station.py — ChargingStation
Vai trò: Battery recharge station.

Thuộc tính: id, pos, capacity, charge_radius, charge_rate, current_occupants

Methods:

Tên Mô tả
in_range(uav_pos) dist_xy ≤ charge_radius AND z ≤ 0.5m
try_occupy(uav) Chiếm slot → bool
release(uav) Giải phóng slot → bool
charge(uav) Sạc 1 step
get_occupancy_ratio() [0.0, 1.0]
entities/obstacle.py — Debris, DangerZone
Vai trò: Static hazards. Hỗ trợ 3 shapes: circle/rectangle/polygon.

Debris methods:

Tên Mô tả
causes_collision(uav_pos) in_zone_2d AND uav.z < height_3d
blocks_los(pos1, pos2) LOS blocked?
DangerZone thêm:

Tên Mô tả
danger_type fire/smoke/gas/radiation/collapse
is_inside(uav_pos) Check containment
get_sensor_modifier() Float [0.4,1.0] — ảnh hưởng detection
blocks_los() Chỉ fire và smoke chặn LOS
📁 core/ — Hệ thống lõi
core/coverage_map.py — CoverageMap v2.0
Vai trò: Track khu vực đã khám phá với temporal info.

Thuộc tính: grid (bool[GS,GS]), timestamps (int32), first_scan (int32), scan_count (int32)

Methods:

Tên Mô tả
reset() Reset về 0/False/-1
mark_explored(uav_pos, fov_radius, step) Vectorized FOV marking
get_coverage_rate() [0,1] toàn map
get_local_coverage(pos, radius) Coverage trong vùng
get_staleness(pos, radius, step) Tuổi trung bình cells
get_nearest_unexplored(pos, min_distance) O(N) tìm ô chưa explore
get_stats(step) Dict metrics
core/map_generator.py — MapGenerator v4.1
Vai trò: Sinh map procedurally.

Methods:

Tên Mô tả
generate(n_victims_override, seed) Main: sinh toàn bộ map_data dict
\_place_stations(rng) Stations với min_spacing
\_place_debris(stations, rng) 40% circle, 40% rect, 20% polygon
\_place_danger_zones(existing, rng) 50% circle, 50% rect
\_spawn_victims(n, obstacles, danger_zones, rng) Group spawn
get_uav_spawns(stations, n_total, rng) Spawn positions quanh stations
Output generate(): Dict với stations, debris, danger_zones, victims, uav_spawns, seed, n_victims

core/fleet_manager.py — FleetManager v2.0
Vai trò: Constraint enforcer — RL agent tự quyết định.

Thuộc tính: n_total, n_reserve, all_uavs, stations, \_battery_death_penalized, \_uav_return_locks

Methods:

Tên Mô tả
enforce_safety_constraints() ENFORCE: battery=0→DISABLED, <5%→RETURNING
suggest_deployments(target_active) SUGGEST (RL có thể ignore)
suggest_returns() Gợi ý về trạm
step() enforce + suggest → Dict
get_mission_priority_hints() operational_ratio, reserve_health
get_battery_stats() mean/min/max/std + critical counts
📁 sensors/ — Sensor Models
sensors/fov_sensor.py — FOVSensor
Vai trò: FOV geometry + detection probability.

Noise pipeline: P_final = P_altitude × env_factor × (1-motion_penalty) × victim_factor × (1-base_miss_rate)

Methods:

Tên Mô tả
calculate_detection_prob(alt, speed, env, victim) Full noise pipeline
check_detected(uav, victim, obstacles) FOV → LOS → P_detect → sample
scan_victims(uav, victims, obstacles) → ndarray(25,) — 5 victims × 5 features
scan_obstacles(uav, obstacles) → ndarray(12,) — 4 obstacles × 3 features
sensors/comm_sensor.py — CommSensor
Vai trò: V2V communication trong comm_range.

Methods:

Tên Mô tả
scan(ego_uav, all_active_uavs) → ndarray(9,) — 3 teammates × 3 features
get_teammates_in_range(ego_uav, all_uavs) Sorted list by distance
📁 observation/ — Observation Builder
observation/obs_builder.py — ObservationBuilder, ObsResult
Vai trò: Build actor (68-dim) và critic (554-dim) observations.

ObsResult: Container với actor_obs: Dict[int, ndarray(68)] và critic_obs: ndarray(554)

Actor obs layout (68 dims):

Slice Dims Nội dung
[0:11] 11 Self: pos(3)/vel(3)/battery(1)/state_onehot(4)
[11:19] 8 Stations: 2×[rel_x, rel_y, dist, occupancy]
[19:28] 9 Teammates: 3×[dist, bearing, rel_alt]
[28:40] 12 Obstacles: 4×[rel_x, rel_y, type_id]
[40:65] 25 Victims: 5×[rel_x, rel_y, urgency, dist, found]
[65:68] 3 Coverage: [local_15m, local_30m, time_remaining]
Critic obs layout (554 dims):

[0:544] = 8 UAVs × 68 (zero-padded cho disabled/reserve)
[544:554] = Global 10-dim: n_active/n_charging/n_disabled/n_alive (÷n), mean/std/min battery, global_coverage, victim_found_rate, time_remaining
Methods:

Tên Mô tả
build_actor_obs(uav, all_uavs, stations, victims, obstacles, step) Build 68-dim cho 1 UAV
build_all(all_uavs, stations, victims, obstacles, step) Build tất cả → ObsResult
📁 rewards/ — Reward Functions
rewards/baseline_reward.py — BaselineReward v3.1
Vai trò: Hand-crafted reward, 14 components. SHARED REWARD — tất cả agents nhận cùng value.

Methods:

Tên Mô tả
reset() Clear penalized sets, prev_dist memory
compute(...) Global reward → dict 14 components
compute_per_uav(uav, newly_found_by_uav, ...) Per-agent (shared value cho MAPPO)
\_terminal_bonus(coverage, victims, step) 70% cov + 20% vic + 10% time
summarize(reward_dict) Compact log string
⚠️ Lưu ý MAPPO: compute_per_uav() trả về CÙNG GIÁ TRỊ cho tất cả agents (shared cooperative reward).

📁 env/ — Environments
env/base_env.py — SARBaseEnv ✅ UPDATED Phase 2
Vai trò: Gymnasium interface. ĐÃ SỬA: \_build_obs_dict() trả về (obs_dict, critic_obs), info có global_obs.

Thuộc tính: cfg, backend, \_reward_fn, \_obs_builder, \_map_gen, \_step_count, \_prev_coverage, \_episode_reward_sum

Methods:

Tên Mô tả
reset(seed, options) → (obs_dict, info) — info có global_obs: ndarray(554)
step(actions) → (obs, rewards, done, truncated, info) — info có global_obs
render() → ndarray hoặc None
\_build_obs_dict(...) → (dict[int,ndarray(68)], ndarray(554)) ← UPDATED
\_check_done(coverage, victims, uavs) coverage≥90% / all_found / all_disabled
Step flow: apply_actions → step_physics → step_world → check done TRƯỚC → compute rewards → build obs → return

Info dict keys: step, coverage_rate, victims_found, victims_total, n_active, n_charging, n_disabled, success, done_reason, rewards_breakdown, global_obs ← NEW

env/sar_pettingzoo_env.py — SARPettingZooEnv
Vai trò: PettingZoo ParallelEnv wrapper. Convert int keys → str ("uav_0").

Methods:

Tên Mô tả
reset(seed) → (obs[str], infos[str]) — infos có global_obs
step(actions[str]) → (obs, rewards, terms, truncs, infos) — tất cả keyed by str
observation_space(agent) → Box(68,)
action_space(agent) → Box(3,)
MAPPO access pattern:

Python

obs, infos = env.reset(seed=42)
global_obs = infos['uav_0']['global_obs'] # ndarray(554,)

obs, rewards, terms, truncs, infos = env.step(actions)
global_obs = infos['uav_0']['global_obs'] # ndarray(554,)
env/backends/logic_backend.py — LogicBackend
Vai trò: Pure Python backend ~1000 steps/s.

Methods:

Tên Mô tả
reset(map_data) Build entities từ map_data
apply_actions(actions) ACTIVE: velocity; RETURNING: auto_navigate
step_physics() Battery drain/charge
step_world() Fleet → victims → coverage → detection
get_state() Dict: uavs/victims/stations/obstacles/coverage_map/fleet_manager
⚠️ Known issue: Tất cả UAVs spawn ACTIVE, reserve pool empty ban đầu.

📁 visualization/ — Renderers
visualization/renderer_factory.py
Vai trò: Factory pattern.

create_renderer(cfg, render_mode, viz_mode) → Visualizer2D hoặc Visualizer3D
visualization/visualizer2d.py — Visualizer2D
Vai trò: Matplotlib 2D, ~50ms/frame, reuse figure.

render(uavs, victims, obstacles, stations, cov_map, step, metrics) → ndarray
Layout: [3:1] Map + Info panel
visualization/visualizer3d.py — Visualizer3D
Vai trò: Matplotlib 3D, ~400ms/frame.

render(uavs, victims, obstacles, stations, cov_map, step) → ndarray
Layout: [3:1] 3D scene + Dashboard
📁 training/ — Training Pipeline
training/curriculum.py — CurriculumManager
Vai trò: Stage progression logic.

Methods:

Tên Mô tả
current_stage Property: StageConfig hiện tại
update(coverage, victims_rate, reward) Cập nhật stats sau episode
should_advance() eps≥min AND cov≥threshold AND vic≥threshold
advance() Tăng stage_idx
apply_to_config(cfg) Gọi cfg.apply_stage(current_stage)
training/curriculum_trainer.py — CurriculumTrainer
Vai trò: Training loop với random policy (Phase 1 placeholder).

Methods: train(total_episodes), \_build_env(), \_run_episode(), \_sample_actions(), \_plot_training_curves()

📁 training/algorithms/mappo/ — MAPPO Implementation 🔄 ĐANG THỰC HIỆN
training/algorithms/mappo/networks.py ✅ DONE (Step 3)
Vai trò: MLP foundation cho Actor và Critic.

Functions:

Tên Mô tả
orthogonal_init(layer, gain) Orthogonal weight init (PPO best practice)
get_parameter_count(model) Đếm trainable params
print_network_summary(model, name) In architecture summary
MLP class:

Tên Mô tả
**init**(input_dim, hidden_dims, output_dim, activation, use_layer_norm, output_activation) Build flexible MLP
forward(x) [batch, input_dim] → [batch, output_dim]
Design: Orthogonal init (gain=√2 hidden, gain=0.01 output), nn.Sequential, supports tanh/relu/elu

training/algorithms/mappo/actor.py ✅ DONE (Step 4)
Vai trò: Gaussian policy network — Decentralized execution.

ActorNetwork class:

Thuộc tính: obs_dim=68, action_dim=3, mean_net (MLP), log_std (nn.Parameter, learnable)

Methods:

Tên Input Output Mô tả
forward(obs) [batch, 68] (mean[batch,3], std[batch,3]) Compute mean và std
get_action(obs, deterministic) [batch, 68] (action[batch,3], log_prob[batch]) Sample action (rollout)
evaluate_actions(obs, actions) [batch, 68], [batch, 3] (log_prob[batch], entropy[batch]) Evaluate cho PPO update
get_log_std() — tensor(3,) Current log_std
set_log_std(value) float — Set log_std
Design: State-independent std (log_std là parameter, không phải network output), Gaussian distribution, shared weights across agents

training/algorithms/mappo/critic.py ✅ DONE (Step 5)
Vai trò: Centralized value function — CTDE training.

CriticNetwork class:

Thuộc tính: global_obs_dim=554, value_net (MLP 554→512→256→1)

Methods:

Tên Input Output Mô tả
forward(global_obs) [batch, 554] [batch, 1] Raw value output
get_value(global_obs) [batch, 554] [batch] Squeezed value
compute_loss(global_obs, returns) [batch, 554], [batch] scalar MSE loss
compute_value_metrics(global_obs, returns) — Dict value_loss, explained_variance, stats
Helper functions: test_critic_accuracy(), initialize_critic_for_env()

Design: Centralized (sees global 554-dim), single critic shared across agents, outputs V(s) not Q(s,a)

training/algorithms/mappo/buffer.py ⬜ CHƯA IMPLEMENT (Step 6)
Vai trò: Rollout buffer + GAE computation.

Planned RolloutBuffer class:

Storage: observations[2048,4,68], global_obs[2048,554], actions[2048,4,3], rewards[2048,4], values[2048,4], log_probs[2048,4], dones[2048]
Computed: advantages[2048,4], returns[2048,4]
Methods: add(), compute_gae(), get_batches(), clear()
training/algorithms/mappo/trainer.py ⬜ CHƯA IMPLEMENT (Step 7)
Vai trò: Main MAPPO training loop.

Planned MAPPOTrainer class:

Methods: select_action(), rollout(), update(), train(), save_checkpoint(), load_checkpoint()
Console output: mỗi 10 episodes — task metrics + training metrics
Visualization: mỗi 100 episodes — 2D PNG snapshots
Checkpoints: mỗi 100 episodes
training/algorithms/mappo/**init**.py ✅ DONE
Export: ActorNetwork, CriticNetwork, RolloutBuffer (pending), MAPPOTrainer (pending)

📁 results/ — Auto-generated
text

results/
└── mappo/
├── easy/
│ ├── checkpoints/ # .pt files mỗi 100 eps
│ ├── plots/ # Training curves
│ └── viz/ # 2D snapshots mỗi 100 eps
├── medium/
└── hard/
🔄 EXECUTION FLOW
Reset Flow
text

env.reset(seed)
→ MapGenerator.generate(seed) # Sinh map
→ LogicBackend.reset(map_data) # Build entities
→ ObservationBuilder.build_all() # Build obs
→ return (obs_dict[int→68], info)
info['global_obs'] = ndarray(554) ← NEW
Step Flow
text

env.step(actions[int→3])
→ apply_actions() → step_physics() → step_world()
→ CHECK done/truncated TRƯỚC reward ← BUG-ENV-06 fix
→ BaselineReward.compute_per_uav() # SHARED reward
→ ObservationBuilder.build_all()
→ return (obs, rewards, done, truncated, info)
info['global_obs'] = ndarray(554) ← NEW
MAPPO Data Flow
text

ROLLOUT (collect experience):
env.step() → infos['uav_0']['global_obs'] → Critic.get_value()
env.step() → obs['uav_i'] → Actor.get_action()
→ Buffer.add(obs, global_obs, actions, rewards, values, log_probs, done)

UPDATE (learn from buffer):
Buffer.compute_gae(last_values, last_done)
→ advantages = δ + γλδ' + (γλ)²δ'' + ...
for epoch in 10:
for batch in Buffer.get_batches(256):
→ Actor.evaluate_actions() → PPO clip loss → update actor
→ Critic.get_value() → MSE loss → update critic
⚠️ KNOWN ISSUES
Issue 1: UAV Spawn Tất Cả ACTIVE
Root cause: UAV.**init** default state=ACTIVE
Impact: Reserve pool empty ban đầu
Fix Phase 2: Spawn n_uav - min_reserve ACTIVE + min_reserve CHARGING
Issue 2: Reward Dương Với Random Policy
Lý do: Coverage delta (+330) > time penalty (-60) → net dương
Không phải bug: Correct task value representation
Issue 3: 3D Viz Chậm (~2-5 FPS)
Workaround: Training viz_mode="none", demo viz_mode="3d"
📈 BASELINE PERFORMANCE (Random Policy)
Stage Coverage Victims Found Reward
EASY 55% ± 11% 53% ± 19% +150 ± 200
MEDIUM 41% ± 9% 44% ± 17% +80 ± 180
HARD 32% ± 8% 36% ± 15% +30 ± 160
Target MAPPO:

EASY: 82-88% coverage, 85-88% victims, +420-450 reward
MEDIUM: 68-72% coverage
HARD: 58-62% coverage
🎯 PHASE 2 PROGRESS — MAPPO
Completed ✅
text

STEP 0: env/base_env.py → \_build_obs_dict() trả về (obs, critic_obs)
info['global_obs'] = ndarray(554) ← CRITICAL
STEP 1: config/train.py → 17 MAPPO hyperparameters added
STEP 2: Folder structure → training/algorithms/mappo/ created
STEP 3: networks.py → MLP + orthogonal_init ✅ tested
STEP 4: actor.py → ActorNetwork Gaussian policy ✅ tested
STEP 5: critic.py → CriticNetwork centralized ✅ tested
Remaining ⬜
text

STEP 6: buffer.py → RolloutBuffer + GAE computation
STEP 7: trainer.py → MAPPOTrainer main loop - Console log mỗi 10 episodes - 2D viz mỗi 100 episodes - Checkpoint mỗi 100 episodes
STEP 8: train_mappo.py → Entry point script
🛠️ DEVELOPMENT COMMANDS

"
📦 FILE STATUS SUMMARY
File Status Mô tả
config/config.py ✅ AppConfig master
config/train.py ✅ Updated +17 MAPPO params
config/env.py ✅ EnvConfig
config/uav.py ✅ UAVConfig
config/sensor.py ✅ SensorConfig
config/entity.py ✅ Victim/Obstacle/Danger configs
config/reward.py ✅ RewardConfig v3.1
config/obs.py ✅ ObsConfig (68/554 dims)
config/curriculum_config.py ✅ 3 stages
env/base_env.py ✅ Updated +global_obs in info
env/sar_pettingzoo_env.py ✅ PettingZoo wrapper
env/backends/logic_backend.py ✅ Physics backend
observation/obs_builder.py ✅ Actor(68) + Critic(554) obs
rewards/baseline_reward.py ✅ 14 components shared reward
core/coverage_map.py ✅ CoverageMap v2.0
core/map_generator.py ✅ MapGenerator v4.1
core/fleet_manager.py ✅ FleetManager v2.0
sensors/fov_sensor.py ✅ FOVSensor noise model v2
sensors/comm_sensor.py ✅ CommSensor
training/curriculum.py ✅ CurriculumManager
training/curriculum_trainer.py ✅ Random policy trainer
training/algorithms/mappo/networks.py ✅ MLP foundation
training/algorithms/mappo/actor.py ✅ ActorNetwork
training/algorithms/mappo/critic.py ✅ CriticNetwork
training/algorithms/mappo/buffer.py ⬜ NEXT: Step 6
training/algorithms/mappo/trainer.py ⬜ Step 7
train_mappo.py ⬜ Step 8

vậy bạn hayx tổng hợp những cái t đã làm ở project này từ đầu tới cuối ghi chi tiết rõ ràng từng folder có file nào file có hàm và thuộc tính nào có tác dụng gì một cách chi tiết (ko cần code) và ghi rõ ràng chi tiết để khi qua đoạn chat ms thì chỉ cần đưa cái đó cho nớ là nó sẽ hiểu đang làm gì và đang thực hiện cái gì biết đang thực heienj cái gì theo format này
