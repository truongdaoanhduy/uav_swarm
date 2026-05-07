🚁 SAR UAV SWARM — PROJECT HANDOFF DOCUMENT v11.0
Cập nhật: Phase 2 MAPPO 100% hoàn thành | Vectorized Env đang fix bug

🎯 MỤC TIÊU NGHIÊN CỨU
Paper 1 — Algorithm Comparison
Câu hỏi: MAPPO vs MASAC vs MATD3 — thuật toán nào tốt nhất cho SAR?
Phương pháp: 3 algorithms × 5 seeds × 3 curriculum stages × 3000 episodes
Metrics: Coverage rate, Victims found, Episode reward, Sample efficiency
Paper 2 — LLM Reward vs Hand-Crafted
Câu hỏi: LLM có generate reward function tốt hơn human expert không?
Baseline: BaselineReward v3.1 (hand-crafted)
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
Phase 2 RL Algorithms — MAPPO ✅ 100% hoàn thành
Phase 2b Vectorized Env (parallel training) 🔄 Đang fix bug
Phase 3 LLM Reward Integration ⬜ Chưa bắt đầu
Phase 4 Advanced Backends (PyBullet/Isaac) ⬜ Chưa bắt đầu
Test status: 26/26 tests PASS
Actor obs dim: 68 (n_stations=2)
Critic obs dim: 554 (8×68+10)
Action dim: 3 (vx, vy, vz) ∈ [-1, 1]
Training speed: ~70 FPS (single env, CPU+GPU)
📐 KEY NUMBERS
Metric Giá trị
Actor obs dim 68
Critic obs dim 554
Action space 3 dims ∈ [-1,1]
UAV states 5: ACTIVE/RETURNING/CHARGING/DEPLOYING/DISABLED
Reward components 14
Reward type SHARED (cooperative)
Test pass 26/26
Backend speed ~1000 steps/s
Training FPS ~70 (single env)
n_envs parallel 8 (đang fix bug)
📁 CẤU TRÚC DỰ ÁN ĐẦY ĐỦ
Lưu ý: Tên folder gốc là env/ đã được đổi thành env_setup/ trong quá trình phát triển.

📁 config/ — Hệ thống cấu hình
config/**init**.py
Export tất cả config classes ra ngoài
Export: AppConfig, EnvConfig, UAVConfig, SensorConfig, VictimConfig, ObstacleConfig, DangerZoneConfig, RewardConfig, ObsConfig, TrainConfig, StageConfig, STAGE_EASY, STAGE_MEDIUM, STAGE_HARD, CURRICULUM_STAGES
config/config.py — AppConfig
Vai trò: Master config — single source of truth, truyền vào toàn bộ hệ thống
Thuộc tính:
env: EnvConfig — Map & physics params
uav: UAVConfig — Drone dynamics & battery
sensor: SensorConfig — FOV/Comm/Noise model
victim: VictimConfig — Victim spawning
obstacle: ObstacleConfig — Debris params
danger: DangerZoneConfig — Danger zone configs
reward: RewardConfig — 14 reward components
obs: ObsConfig — Observation dims
train: TrainConfig — RL training params + MAPPO hyperparams
viz_mode: str — "2d"/"3d"/"none"
Methods:
**post_init**() — Auto-sync obs.n_stations = env.n_stations, validate
apply_stage(stage) — Apply curriculum stage in-place (SINGLE SOURCE OF TRUTH)
map_diagonal — Property: sqrt(2) × map_size
grid_cell_size — Property: map_size / grid_size
save(path) — Serialize sang JSON
load(path) — Restore từ JSON
config/env.py — EnvConfig
Vai trò: Map, thời gian, fleet params
Thuộc tính chính:
map_size=100 — Kích thước map (m)
grid_size=100 — Số ô lưới (sync với map_size)
dt_seconds=1.0 — Timestep (s)
max_steps=600 — Steps tối đa/episode
n_uav=4 — Số UAV
n_stations=2 — Số trạm sạc
charge_radius_m=3.0 — Bán kính sạc (m)
station_capacity=2 — UAVs tối đa/trạm
config/uav.py — UAVConfig
Vai trò: Physics UAV + battery model
Thuộc tính chính:
z_min_m=3.0, z_max_m=40.0 — Độ cao (m)
max_speed_xy_mps=5.0 — Tốc độ ngang tối đa
max_speed_z_mps=2.0 — Tốc độ dọc tối đa
drain_xy_pct_per_s=0.10 — Drain ngang (%/s)
drain_z_up_pct_per_s=0.15 — Drain leo cao (%/s)
charge_rate_pct_per_s=1.5 — Tốc độ sạc
battery_return_pct=10.0 — Ngưỡng tự động về trạm
battery_ready_pct=80.0 — Sẵn sàng xuất phát
battery_emergency_pct=5.0 — Ngưỡng emergency
reserve_ratio=0.2, min_reserve=2
config/sensor.py — SensorConfig
Vai trò: FOV geometry + detection noise model
Thuộc tính:
comm_range_m=30.0 — Tầm liên lạc
hfov_deg=90.0 — Góc FOV ngang
p_detect_base=0.95 — P_detect tại altitude=0
p_detect_decay=0.04 — Decay theo altitude
enable_noise=True
motion_blur_coeff=0.06
base_miss_rate=0.03
config/entity.py — VictimConfig, ObstacleConfig, DangerZoneConfig
VictimConfig:
n_victims_min/max=5/20
injured_ratio_min/max=0.4/0.7
injured_urgency_min/max=4.0/5.0
mobile_urgency_min/max=1.0/3.0
mobile_speed_min/max_mps=0.2/0.4
mobile_dir_change_steps=20
ObstacleConfig:
n_debris=6
debris_width_min/max_m=2.0/5.0
debris_height_min/max_m=3.0/8.0
n_danger_total=2
DangerZoneConfig:
heights: {gas:3, fire:15, smoke:25, collapse:10, radiation:inf}
penalties: {gas:-3, fire:-3, smoke:-1.5, collapse:-1, radiation:-5}
widths: Dict {type: (min_diam, max_diam)}
validate() — Kiểm tra consistency
danger_types — Property list các loại
config/reward.py — RewardConfig v3.1
Vai trò: 14 reward components
Components:
r_coverage_delta=+6.0 — Per 1% coverage tăng
r_victim_base=+50.0 × urgency/5 khi tìm thấy
r_battery_10=-1.0 — Penalty <10%
r_battery_5=-3.0 — Penalty <5%
r_battery_dead=-100.0 — One-time khi chết pin
r_collision_obstacle=-30.0 — One-time va chạm debris
r_proximity_1m=-10.0, r_proximity_2m=-3.0, r_proximity_3m=-0.5
proximity_penalty_cap=-15.0
r_time_penalty=-0.05 — Per active UAV per step
r_terminal_base=+200.0, terminal_bonus_cap=+100.0
step_penalty_cap=-30.0
step_reward_clip_min/max=-100/+100
enable_distance_shaping=True
config/obs.py — ObsSchemaConfig, ObsConfig
ObsSchemaConfig constants:
SELF_FEATURES=11, STATION_FEATURES_PER=4, TEAMMATE_FEATURES_PER=3
OBSTACLE_FEATURES_PER=3, VICTIM_FEATURES_PER=5, COVERAGE_FEATURES=3, GLOBAL_FEATURES=10
ObsConfig thuộc tính:
n_obs_victims=5, n_obs_obstacles=4, n_tracked_teammates=3
local_cov_small=15, local_cov_large=30
max_uav=8, n_stations=None (auto-sync)
Computed properties:
actor_dim=68 (tổng), critic_dim=554 (8×68+10)
config/train.py — TrainConfig
Vai trò: Training params + MAPPO hyperparameters
Existing fields: n_seeds=5, seeds=[42,123,456,789,1011], total_episodes=3000, eval_interval=50, save_interval=100, log_window=100
MAPPO hyperparameters:
mappo_rollout_length=2048 — Steps per update
mappo_n_epochs=10 — Epochs per update
mappo_batch_size=256 — Minibatch size
mappo_clip_epsilon=0.2 — PPO clip range
mappo_gamma=0.99 — Discount factor
mappo_gae_lambda=0.95 — GAE lambda
mappo_lr_actor=3e-4, mappo_lr_critic=1e-3
mappo_max_grad_norm=0.5 — Gradient clipping
mappo_entropy_coeff=0.01 — Entropy bonus
mappo_actor_hidden=(256,256), mappo_critic_hidden=(512,256)
mappo_activation='tanh', mappo_use_layer_norm=False
mappo_log_interval=10, mappo_viz_interval=100, mappo_checkpoint_interval=100
config/curriculum_config.py — StageConfig
Vai trò: Định nghĩa 3 curriculum stages
StageConfig fields: name, map_size, n_uav=4, n_victims_min/max, n_debris, n_danger_total, max_steps, min_episodes, advance_coverage, advance_victims
3 stages:
Stage Map Pressure Max Steps Advance
STAGE_EASY 150×150m 5,625 m²/UAV 300 cov≥70%, vic≥80%
STAGE_MEDIUM 200×200m 10,000 m²/UAV 350 cov≥65%, vic≥75%
STAGE_HARD 250×250m 15,625 m²/UAV 400 cov≥60%, vic≥70%
📁 utils/ — Tiện ích
utils/geometry.py
Vai trò: 9 geometry functions, vectorized với NumPy
Hàm:
dist_2d(pos1, pos2) → float
dist_3d(pos1, pos2) → float
normalize_angle(angle) → float
compute_bearing(from_pos, from_vel, to_pos) → float
check_los_2d(pos1, pos2, obstacles) → bool
get_circle_cells(center, radius, grid_size, map_size) → ndarray(N,2)
get_relative_position(from_pos, to_pos) → ndarray
clip_position(pos, min_bounds, max_bounds) → ndarray
utils/logger.py
EpisodeLogger methods: log_step(), log_event(), set_total_victims(), finalize()→Dict
TrainingLogger methods: log_episode(), get_stats(), save(), load()
📁 entities/ — Game Objects
entities/uav.py — UAV, UAVState
UAVState enum: ACTIVE, RETURNING, CHARGING, DEPLOYING, DISABLED
UAV thuộc tính: id, pos[x,y,z], vel[vx,vy,vz], battery, state, target_station, victims_found, battery_death
UAV methods:
apply_action(action) — action [-1,1]³ → vel → update pos
auto_navigate(target_pos) — Bay tự động không overshoot
update_battery(stations) — Drain hoặc charge
get_fov_radius() — altitude × fov_tan
get_state_onehot() → ndarray(5,)
set_state(new_state) — Chuyển state có validation
needs_charging() — battery ≤ battery_return_pct
is_ready_to_deploy() — battery ≥ battery_ready_pct
find_nearest_station(stations) — Tìm trạm gần nhất có chỗ
entities/victim.py — BaseVictim, InjuredVictim, MobileVictim
BaseVictim: id, pos, urgency, is_found, found_at_step, found_by_uav
Methods: mark_found(step, uav_id), get_reward_value(), update(step_count, obstacles)
InjuredVictim: Stationary, urgency [4.0,5.0]
MobileVictim: Random walk [0.2,0.4] m/s, đổi hướng mỗi 20 steps
entities/charging_station.py — ChargingStation
Thuộc tính: id, pos, capacity, charge_radius, charge_rate, current_occupants
Methods: in_range(uav_pos), try_occupy(uav), release(uav), charge(uav), get_occupancy_ratio()
entities/obstacle.py — Debris, DangerZone
Debris methods: causes_collision(uav_pos), blocks_los(pos1, pos2)
DangerZone thêm: danger_type, is_inside(uav_pos), get_sensor_modifier(), blocks_los()
Shapes: circle/rectangle/polygon
📁 core/ — Hệ thống lõi
core/coverage_map.py — CoverageMap v2.0
Thuộc tính: grid(bool[GS,GS]), timestamps(int32), first_scan(int32), scan_count(int32)
Methods:
reset() — Reset về 0/False/-1
mark_explored(uav_pos, fov_radius, step) — Vectorized FOV marking
get_coverage_rate() → [0,1]
get_local_coverage(pos, radius) → float
get_staleness(pos, radius, step) → float
get_nearest_unexplored(pos, min_distance) — O(N)
get_stats(step) → Dict
core/map_generator.py — MapGenerator v4.1
Methods:
generate(n_victims_override, seed) → Dict (stations, debris, danger_zones, victims, uav_spawns, seed, n_victims)
\_place_stations(rng) — Min spacing
\_place_debris(stations, rng) — 40% circle, 40% rect, 20% polygon
\_place_danger_zones(existing, rng) — 50% circle, 50% rect
\_spawn_victims(n, obstacles, danger_zones, rng) — Group spawn
get_uav_spawns(stations, n_total, rng) — Spawn quanh stations
core/fleet_manager.py — FleetManager v2.0
Thuộc tính: n_total, n_reserve, all_uavs, stations, \_battery_death_penalized, \_uav_return_locks
Methods:
enforce_safety_constraints() — ENFORCE: battery=0→DISABLED, <5%→RETURNING
suggest_deployments(target_active) — SUGGEST (RL có thể ignore)
suggest_returns() — Gợi ý về trạm
step() → Dict
get_battery_stats() → Dict
📁 sensors/ — Sensor Models
sensors/fov_sensor.py — FOVSensor
Noise pipeline: P_final = P_altitude × env_factor × (1-motion_penalty) × victim_factor × (1-base_miss_rate)
Methods:
calculate_detection_prob(alt, speed, env, victim) → float
check_detected(uav, victim, obstacles) → bool
scan_victims(uav, victims, obstacles) → ndarray(25,)
scan_obstacles(uav, obstacles) → ndarray(12,)
sensors/comm_sensor.py — CommSensor
Methods:
scan(ego_uav, all_active_uavs) → ndarray(9,) — 3 teammates × 3 features
get_teammates_in_range(ego_uav, all_uavs) → sorted list
📁 observation/ — Observation Builder
observation/obs_builder.py — ObservationBuilder, ObsResult
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
[0:544] = 8 UAVs × 68 (zero-padded)
[544:554] = Global 10-dim: n_active/charging/disabled/alive, battery stats, coverage, victim_rate, time
Methods:
build_actor_obs(uav, all_uavs, stations, victims, obstacles, step) → ndarray(68)
build_all(all_uavs, stations, victims, obstacles, step) → ObsResult
📁 rewards/ — Reward Functions
rewards/baseline_reward.py — BaselineReward v3.1
Vai trò: Hand-crafted reward, 14 components, SHARED REWARD
Methods:
reset() — Clear penalized sets, prev_dist memory
compute(...) → Dict 14 components
compute_per_uav(uav, newly_found_by_uav, ...) → float (CÙNG GIÁ TRỊ cho tất cả agents)
\_terminal_bonus(coverage, victims, step) — 70% cov + 20% vic + 10% time
summarize(reward_dict) → str
📁 env_setup/ — Environments
Lưu ý: Folder gốc là env/, đã đổi thành env_setup/ trong quá trình dev

env*setup/base_env.py — SARBaseEnv
Vai trò: Gymnasium interface
Thuộc tính: cfg, backend, \_reward_fn, \_obs_builder, \_map_gen, \_step_count, \_prev_coverage, \_episode_reward_sum
Methods:
reset(seed, options) → (obs_dict, info) — info có global_obs: ndarray(554)
step(actions) → (obs, rewards, done, truncated, info) — info có global_obs
render() → ndarray hoặc None
\_build_obs_dict(...) → (dict[int,ndarray(68)], ndarray(554))
\_check_done(coverage, victims, uavs) → bool
Step flow: apply_actions → step_physics → step_world → check done TRƯỚC → compute rewards → build obs → return
Info dict keys: step, coverage_rate, victims_found, victims_total, n_active, n_charging, n_disabled, success, done_reason, rewards_breakdown, global_obs
env_setup/sar_pettingzoo_env.py — SARPettingZooEnv
Vai trò: PettingZoo ParallelEnv wrapper, convert int keys → str ("uav_0")
Methods:
reset(seed) → (obs[str], infos[str]) — infos có global_obs
step(actions[str]) → (obs, rewards, terms, truncs, infos) — keyed by str
observation_space(agent) → Box(68,)
action_space(agent) → Box(3,)
MAPPO access pattern:
obs, infos = env.reset(seed=42) → global_obs = infos['uav_0']['global_obs']
obs, rewards, terms, truncs, infos = env.step(actions) → global_obs = infos['uav_0']['global_obs']
Info structure reset: {uav_0: {seed, n_uav, n_stations, n_victims, n_obstacles, map_size, coverage, coverage_rate, victims_found, victims_total, global_obs}}
Info structure step: {uav_0: {coverage, victims_found, victims_total, step, coverage_rate, n_active, n_charging, n_disabled, success, done_reason, rewards_breakdown, newly_found_ids, global_obs}}
env_setup/vec_env.py — VectorizedEnv 🔄 ĐANG FIX
Vai trò: Chạy N envs song song bằng multiprocessing để tăng tốc training
env_worker(pipe, config, seed) — Worker process function:
Chạy trong process riêng
Nhận cmd qua pipe: "reset", "step", "close"
Cache last valid obs/global_obs khi done
Auto reset khi episode done
Convert actions ndarray → {f"uav*{i}": actions[i]} (string keys)
Extract global*obs từ info['uav_0']['global_obs']
VectorizedEnv class:
**init**(config, n_envs=8, start_seed=0) — Tạo N processes
reset() → (obs_batch[n_envs,n_agents,obs_dim], global_obs_batch[n_envs,global_obs_dim])
step(actions_batch[n_envs,n_agents,3]) → (obs_batch, global_obs_batch, rewards_batch, dones, infos)
close() — Đóng tất cả processes
⚠️ Known bug (đang fix): Episode done → obs/info dict có thể rỗng trong PettingZoo → đã fix bằng cache last valid state
env_setup/backends/logic_backend.py — LogicBackend
Vai trò: Pure Python backend ~1000 steps/s
Methods:
reset(map_data) — Build entities từ map_data
apply_actions(actions) — ACTIVE: velocity; RETURNING: auto_navigate
step_physics() — Battery drain/charge
step_world() — Fleet → victims → coverage → detection
get_state() → Dict: uavs/victims/stations/obstacles/coverage_map/fleet_manager
📁 visualization/ — Renderers
visualization/renderer_factory.py
create_renderer(cfg, render_mode, viz_mode) → Visualizer2D hoặc Visualizer3D
visualization/visualizer2d.py — Visualizer2D
~50ms/frame, reuse figure, layout [3:1] Map + Info panel
render(uavs, victims, obstacles, stations, cov_map, step, metrics) → ndarray
visualization/visualizer3d.py — Visualizer3D
~400ms/frame, layout [3:1] 3D scene + Dashboard
render(uavs, victims, obstacles, stations, cov_map, step) → ndarray
📁 training/ — Training Pipeline
training/curriculum.py — CurriculumManager
Vai trò: Stage progression logic
StageStats dataclass: stage_name, episodes_done, coverage_list, victims_list, reward_list
Properties: avg_coverage, avg_victims, avg_reward (rolling 50 episodes)
CurriculumManager methods:
current_stage — Property: StageConfig hiện tại
stage_idx — Property: int
is_final_stage — Property: bool
total_episodes — Property: int
update(coverage, victims_rate, reward) — Cập nhật metrics sau episode
should_advance() — eps≥min AND cov≥threshold AND vic≥threshold
advance() — Tăng stage_idx
apply_to_config(cfg) — Gọi cfg.apply_stage(current_stage)
get_status() → Dict
print_status() — In status ra console
⚠️ Lưu ý: Method update() nhận params: coverage, victims_rate, reward (KHÔNG phải victims_found_rate hay episode_reward)
training/curriculum_trainer.py — CurriculumTrainer
Vai trò: Training loop với random policy (Phase 1 placeholder)
Methods: train(total_episodes), \_build_env(), \_run_episode(), \_sample_actions(), \_plot_training_curves()
📁 training/algorithms/mappo/ — MAPPO Implementation ✅ HOÀN THÀNH
training/algorithms/mappo/networks.py — MLP Foundation
Vai trò: MLP foundation cho Actor và Critic
Functions:
orthogonal_init(layer, gain) — Orthogonal weight init (PPO best practice)
get_parameter_count(model) → int
print_network_summary(model, name) — In architecture
MLP class:
**init**(input_dim, hidden_dims, output_dim, activation, use_layer_norm, output_activation)
forward(x) → [batch, output_dim]
Design: Orthogonal init (gain=√2 hidden, 0.01 output), supports tanh/relu/elu
⚠️ Bug fix: hidden_dims phải convert sang list: dims = [input_dim] + list(hidden_dims)
training/algorithms/mappo/actor.py — ActorNetwork
Vai trò: Gaussian policy network — Decentralized execution
Thuộc tính: obs_dim=68, action_dim=3, mean_net(MLP), log_std(nn.Parameter)
Methods:
forward(obs[batch,68]) → (mean[batch,3], std[batch,3])
get_action(obs, deterministic) → (action[batch,3], log_prob[batch])
evaluate_actions(obs, actions) → (log_prob[batch], entropy[batch])
get_log_std(), set_log_std(value)
Design: State-independent std, shared weights across agents
training/algorithms/mappo/critic.py — CriticNetwork
Vai trò: Centralized value function — CTDE training
Thuộc tính: global_obs_dim=554, value_net(MLP 554→512→256→1)
Methods:
forward(global_obs[batch,554]) → [batch,1]
get_value(global_obs) → [batch] squeezed
compute_loss(global_obs, returns) → scalar MSE
compute_value_metrics(global_obs, returns) → Dict
Design: Centralized (sees global 554-dim), single critic shared across agents
training/algorithms/mappo/buffer.py — RolloutBuffer
Vai trò: Rollout buffer + GAE computation
Storage: observations[rollout_length, n_agents, obs_dim], global_obs[rollout_length, global_obs_dim], actions, rewards, values, log_probs[rollout_length, n_agents], dones[rollout_length]
Computed: advantages[rollout_length, n_agents], returns[rollout_length, n_agents]
Methods:
add(obs, global_obs, actions, rewards, values, log_probs, done) — Thêm 1 transition
compute_gae(last_values, last_done) — GAE backward iteration
get_batches(batch_size) → Iterator[Dict] — Flatten + shuffle + yield batches
clear() — Reset ptr
get_stats() → Dict
⚠️ Lưu ý: get_batches() yield batch cuối có thể < batch_size (check shape[1] không phải shape[0])
Buffer size khi n_envs>1: rollout_length * n*envs
training/algorithms/mappo/trainer.py — MAPPOTrainer
Vai trò: Main MAPPO training loop
**init**(config, device, run_name, n_envs=1):
Tạo ActorNetwork, CriticNetwork, RolloutBuffer
Tạo Adam optimizers cho actor/critic
Setup output dirs: results/mappo/{run_name}/checkpoints/, viz/, plots/
Buffer size = rollout_length * n_envs
select_action(obs_dict, deterministic) → (actions_dict, actions_np, log_probs_np)
Input: {"uav_0": obs[68], ...}
Stack → batch → forward actor → return dict + arrays
get_values(global_obs) → ndarray[n_agents]
Broadcast critic value cho tất cả agents (shared)
rollout(env) → Dict metrics
Dispatcher: n_envs==1 → \_rollout_single(), else → \_rollout_vectorized()
\_rollout_single(env) → Dict metrics
Thu thập rollout_length steps từ 1 env
Auto reset khi done, track episode metrics
Compute GAE sau khi đủ steps
\_rollout_vectorized(env) → Dict metrics
Thu thập từ N envs song song
Batch inference: flatten [n_envs,n_agents,68] → [n_envs*n_agents,68] → 1 forward pass
Compute GAE sau khi đủ steps
update() → Dict train_metrics
n_epochs × minibatch PPO update
Actor: PPO clipped loss + entropy bonus
Critic: MSE loss
Gradient clipping
Clear buffer sau update
train(total_updates, curriculum_manager, seed):
TQDM progress bar (real-time cập nhật mỗi update)
Detailed log mỗi log_interval updates (pbar.write)
Viz mỗi viz_interval updates
Checkpoint mỗi checkpoint_interval updates
Curriculum advancement check sau mỗi update
\_create_env(seed) — Single env hoặc VectorizedEnv tùy n_envs
save_checkpoint(update, curriculum_manager) — Lưu actor/critic weights + optimizer states
load_checkpoint(path) — Restore từ checkpoint
training/algorithms/mappo/**init**.py
Export: MLP, orthogonal_init, get_parameter_count, ActorNetwork, CriticNetwork, RolloutBuffer, MAPPOTrainer
📁 Root Files
train_mappo.py — Entry Point
Vai trò: CLI entry point cho MAPPO training
Arguments:
--seed (default: 42)
--total-updates (default: 500)
--device (default: auto)
--run-name (default: auto-generate)
--n-envs (default: 1) — Số envs song song
--no-curriculum — Disable curriculum
--stage (default: easy) — Fixed stage khi no-curriculum
--rollout-length, --batch-size, --lr-actor, --lr-critic — Config overrides
Usage:
Bash

python train_mappo.py --seed 42 --total-updates 500 --device cuda
python train_mappo.py --seed 42 --total-updates 500 --n-envs 8
python train_mappo.py --no-curriculum --stage easy --total-updates 100
test_trainer_smoke.py — Smoke Test
Vai trò: Verify trainer chạy được không crash
Config: max_steps=50, rollout_length=200, 5 updates
Usage: python test_trainer_smoke.py
🔄 EXECUTION FLOW
Reset Flow
text

env.reset(seed)
→ MapGenerator.generate(seed) # Sinh map
→ LogicBackend.reset(map_data) # Build entities
→ ObservationBuilder.build_all() # Build obs
→ return (obs_dict[str→68], info)
info['uav_0']['global_obs'] = ndarray(554)
Step Flow
text

env.step(actions[str→3])
→ apply_actions() → step_physics() → step_world()
→ CHECK done/truncated TRƯỚC reward
→ BaselineReward.compute_per_uav() # SHARED reward
→ ObservationBuilder.build_all()
→ return (obs, rewards, done, truncated, info)
info['uav_0']['global_obs'] = ndarray(554)
MAPPO Training Flow
text

ROLLOUT:
env.reset() → obs_dict, global_obs
while steps < rollout_length:
actor.get_action(obs) → actions, log_probs
critic.get_value(global_obs) → values
env.step(actions) → next_obs, rewards, done, info
buffer.add(obs, global_obs, actions, rewards, values, log_probs, done)
buffer.compute_gae(last_values, last_done)

UPDATE (n_epochs):
for batch in buffer.get_batches(batch_size):
actor.evaluate_actions(obs, actions) → log_probs, entropy
PPO clip loss → update actor
critic.get_value(global_obs) → values
MSE loss → update critic
buffer.clear()
Vectorized Env Flow
text

VectorizedEnv(n_envs=8):
8 worker processes (multiprocessing)

reset_all() → obs[8,4,68], global_obs[8,554]

each step:
flatten obs [8,4,68] → [32,68]
actor.get_action([32,68]) → actions[32,3] (1 GPU forward)
reshape → [8,4,3]
send to 8 workers simultaneously
receive results from 8 workers
→ obs[8,4,68], global_obs[8,554], rewards[8,4], dones[8]
⚠️ KNOWN ISSUES & FIXES
Issue Root Cause Status Fix
VectorizedEnv KeyError 'uav_0' PettingZoo trả obs rỗng khi done 🔄 Đang fix Cache last valid state trong worker
UAV spawn tất cả ACTIVE UAV.init default state=ACTIVE ⚠️ Known Fix Phase 2: spawn n_uav-min_reserve ACTIVE
Reward dương với random policy Coverage delta > time penalty ✅ Not a bug Correct task representation
3D viz chậm (~2-5 FPS) Matplotlib 3D overhead ✅ Workaround viz_mode="none" khi training
hidden_dims tuple vs list MLP nhận tuple nhưng dùng + ✅ Fixed list(hidden_dims) trong MLP.init
CurriculumManager.update() params victims_found_rate vs victims_rate ✅ Fixed Dùng victims_rate, reward
📈 PERFORMANCE
Baseline (Random Policy)
Stage Coverage Victims Found Reward
EASY 55% ± 11% 53% ± 19% +150 ± 200
MEDIUM 41% ± 9% 44% ± 17% +80 ± 180
HARD 32% ± 8% 36% ± 15% +30 ± 160
MAPPO Results (update 100, HARD stage)
Metric Value
Coverage 82.2%
Victims 81.3%
Reward +86.3
FPS ~70 (CPU+GPU)
Time/500 updates ~4 giờ (single env)
Target MAPPO
Stage Coverage Victims Reward
EASY 82-88% 85-88% +420-450
MEDIUM 68-72% 70-75% +300-350
HARD 58-62% 60-65% +200-250
📦 FILE STATUS SUMMARY
File Status Mô tả
config/config.py ✅ AppConfig master
config/train.py ✅ +17 MAPPO params
config/env.py ✅ EnvConfig
config/uav.py ✅ UAVConfig
config/sensor.py ✅ SensorConfig
config/entity.py ✅ Victim/Obstacle/Danger
config/reward.py ✅ RewardConfig v3.1
config/obs.py ✅ ObsConfig (68/554 dims)
config/curriculum_config.py ✅ 3 stages
env_setup/base_env.py ✅ +global_obs in info
env_setup/sar_pettingzoo_env.py ✅ PettingZoo wrapper
env_setup/vec_env.py 🔄 Đang fix KeyError done
env_setup/backends/logic_backend.py ✅ Physics backend
observation/obs_builder.py ✅ Actor(68)+Critic(554)
rewards/baseline_reward.py ✅ 14 components shared
core/coverage_map.py ✅ CoverageMap v2.0
core/map_generator.py ✅ MapGenerator v4.1
core/fleet_manager.py ✅ FleetManager v2.0
sensors/fov_sensor.py ✅ FOVSensor noise v2
sensors/comm_sensor.py ✅ CommSensor
training/curriculum.py ✅ CurriculumManager
training/curriculum_trainer.py ✅ Random policy
training/algorithms/mappo/networks.py ✅ MLP foundation
training/algorithms/mappo/actor.py ✅ ActorNetwork
training/algorithms/mappo/critic.py ✅ CriticNetwork
training/algorithms/mappo/buffer.py ✅ RolloutBuffer+GAE
training/algorithms/mappo/trainer.py ✅ MAPPOTrainer
train_mappo.py ✅ Entry point
test_trainer_smoke.py ✅ Smoke test
🎯 VIỆC CẦN LÀM TIẾP THEO
Immediate (đang dở)
Fix vec_env.py — VectorizedEnv hoạt động ổn định với 8 envs
Verify n_envs=8 — Smoke test với vectorized env
Push lên GitHub — git push
Phase 2 Complete → Phase 3
Run full training — 5 seeds × curriculum (Kaggle GPU)
MASAC implementation — training/algorithms/masac/
MATD3 implementation — training/algorithms/matd3/
Statistical comparison — Wilcoxon tests, plots
Phase 3 — LLM Reward
Prompt templates cho GPT-4/Claude
LLM reward code generation
Safety validation layer
Compare vs BaselineReward v3.1

ACTOR OBS (68 dims) - LOCAL ONLY:
┌─────────────────────────────────────────────┐
│ Part 1: Self State [0:11] 11 dims │
│ pos_x, pos_y, alt → [0,1] │
│ vel_x, vel_y, vel_z → [-1,1] │
│ battery → [0,1] │
│ state_onehot (4) → {0,1} │
├─────────────────────────────────────────────┤
│ Part 2: Stations [11:19] 8 dims │
│ 2 × [rel_x, rel_y, dist, occupancy] │
├─────────────────────────────────────────────┤
│ Part 3: Teammates [19:28] 9 dims │
│ 3 × [dist, bearing, rel_alt] │
├─────────────────────────────────────────────┤
│ Part 4: Obstacles FOV [28:40] 12 dims │
│ 4 × [rel_x, rel_y, type_id] │
├─────────────────────────────────────────────┤
│ Part 5: Victims FOV [40:65] 25 dims │
│ 5 × [rel_x, rel_y, dist, urgency, found] │
├─────────────────────────────────────────────┤
│ Part 6: Coverage [65:68] 3 dims │
│ local_15m, local_30m, time_remain │
└─────────────────────────────────────────────┘

CRITIC OBS: N_UAV × 68 + 7 global dims

import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import time
import json

"""
Hệ thống logging cho SAR UAV Swarm - Research Grade
Đã fix: coverage units, total_victims missing, convergence logic
"""

class EpisodeLogger:
"""
Logger cho một episode - CHỈ LƯU DATA, KHÔNG IN
Research-ready với đầy đủ metrics cần thiết
"""

text

def **init**(self, episode_id: int, seed: Optional[int] = None):
"""
Tham số:
episode_id: ID của episode
seed: Random seed (để track khi chạy nhiều seeds)
"""
self.episode_id = episode_id
self.seed = seed
self.start_time = time.time()

    # Metrics chính (4 metrics quan trọng nhất)
    self.total_reward = 0.0
    self.coverage_rate = 0.0  # Lưu dưới dạng [0, 1], sẽ convert sang % khi finalize
    self.victims_found = 0
    self.total_victims = 0

    # Episode length (QUAN TRỌNG cho research)
    self.episode_length = 0
    self.collision_events: List[Dict] = []
    self.events: Dict[str, int] = {}

    # ✅ NEW: Landing tracking
    self.landing_events: List[Dict] = []  # List of landing events
    self.total_landings = 0
    self.total_charge_time = 0  # Steps spent charging
    self.per_uav_landings: Dict[int, int] = {}  # UAV ID → count
    # Safety metrics (phân loại chi tiết)
    self.collision_obstacle = 0  # Va chạm debris
    self.collision_uav = 0        # Va chạm UAV khác
    self.collision_proximity = 0  # Proximity warning
    self.battery_deaths = 0
    self.danger_zone_entries = 0

    # Fleet metrics
    self.hot_swaps = 0

def log_step(self, rewards: Dict[str, float], coverage: float):
"""
Log step - CHỈ CẬP NHẬT DATA

    Tham số:
        rewards: Dict {agent_id: reward}
        coverage: Coverage rate hiện tại [0, 1]
    """
    self.total_reward += sum(rewards.values())

    # Coverage: lưu giá trị MAX (tránh noise), giữ [0,1]
    self.coverage_rate = max(self.coverage_rate, coverage)

    # Tăng episode length
    self.episode_length += 1

def log_event(self, event_type: str, \*\*kwargs):
"""
Log event với phân loại chi tiết

    Tham số:
        event_type: Loại event
        **kwargs: Thông tin bổ sung
    """
    if event_type == 'collision_obstacle':
        self.collision_obstacle += 1

    elif event_type == 'collision_uav':
        self.collision_uav += 1

    elif event_type == 'collision_proximity':
        self.collision_proximity += 1

    elif event_type == 'victim_found':
        self.victims_found += 1

    elif event_type == 'battery_death':
        self.battery_deaths += 1

    elif event_type == 'danger_zone':
        self.danger_zone_entries += 1

    elif event_type == 'hot_swap':
        self.hot_swaps += 1

def set_total_victims(self, n: int):
"""Đặt tổng số victims trong episode"""
self.total_victims = n

def log_landing(self, uav_id: int, step: int, battery_before: float, battery_after: float):
"""Log một landing event"""
self.landing_events.append({
"uav_id": uav_id,
"step": step,
"battery_before": battery_before,
"battery_after": battery_after,
"charge_amount": battery_after - battery_before,
})
self.total_landings += 1
self.per_uav_landings[uav_id] = self.per_uav_landings.get(uav_id, 0) + 1

def log_charging_step(self, uav_id: int):
"""Log một step CHARGING state"""
self.total_charge_time += 1
def log_collision(self, uav_id: int, step: int, obstacle_info: dict):
self.collision_events.append({
"step": step,
"uav_id": uav_id,
"obstacle_id": obstacle_info.get("id"),
"obstacle_type": obstacle_info.get("type"),
"position": obstacle_info.get("pos"),
"height": obstacle_info.get("height"),
})

def finalize(self) -> Dict[str, Any]:
"""
Hoàn tất episode, trả về metrics
ĐẢM BẢO tất cả values là Python native types (không phải numpy)

    FIX:
    - Tách rõ coverage_ratio [0,1] và coverage_percent [0,100]
    - Success dùng đúng ngưỡng 0.9 với coverage_ratio
    - Thêm total_victims vào metrics

    Trả về:
        Dict chứa tất cả metrics (JSON-safe)
    """
    duration = time.time() - self.start_time

    # Tách rõ coverage units (FIX LỖI 1)
    coverage_ratio = float(self.coverage_rate)  # [0, 1]
    coverage_percent = coverage_ratio * 100.0   # [0, 100]

    # Tính toán metrics
    victim_found_rate = (self.victims_found / max(1, self.total_victims)) * 100
    total_collisions = (self.collision_obstacle +
                       self.collision_uav +
                       self.collision_proximity)

    # ĐẢM BẢO tất cả là float/int, KHÔNG phải numpy types
    metrics = {
        # Episode info
        'episode_id': int(self.episode_id),
        'seed': int(self.seed) if self.seed is not None else None,
        'duration': float(duration),
        'episode_length': int(self.episode_length),

        # Performance metrics (CAST FLOAT)
        'total_reward': float(self.total_reward),
        'avg_reward_per_step': float(self.total_reward / max(1, self.episode_length)),
        'coverage_rate': float(coverage_percent),  # Store as percent [0, 100]
        'victims_found': int(self.victims_found),
        'total_victims': int(self.total_victims),  # FIX LỖI 2: Thêm field này
        'victims_found_rate': float(victim_found_rate),

        # Safety metrics (phân loại chi tiết)
        'collision_obstacle': int(self.collision_obstacle),
        'collision_uav': int(self.collision_uav),
        'collision_proximity': int(self.collision_proximity),
        'total_collisions': int(total_collisions),
        'battery_deaths': int(self.battery_deaths),
        'danger_zone_entries': int(self.danger_zone_entries),

        # Fleet metrics
        'hot_swaps': int(self.hot_swaps),

         # ✅ Landing stats (phải có)
        'total_landings': int(self.total_landings),
        'total_charge_time': int(self.total_charge_time),
        'avg_charge_per_landing': float(
            self.total_charge_time / max(self.total_landings, 1)
        ),
        'landings_per_uav': dict(self.per_uav_landings),

        # Success criteria (FIX LỖI 1: Dùng đúng coverage_ratio >= 0.9)
        'success': bool(coverage_ratio >= 0.9),
    }

    return metrics

class TrainingLogger:
"""
Logger CHÍNH cho training - Research Grade
Hỗ trợ multi-seed, convergence tracking, và phân tích chi tiết
"""

text

def **init**(self, verbose: int = 1, window_size: int = 100):
"""
Tham số:
verbose:
0 = Im lặng (chỉ lưu file)
1 = Cơ bản (mỗi episode in 1 dòng)
2 = Chi tiết (mỗi 100 episodes in summary)
window_size: Kích thước window cho moving average
"""
self.verbose = verbose
self.window_size = window_size

    # Lưu trữ TẤT CẢ episodes
    self.all_metrics = []

    # Moving windows cho monitoring
    self.recent_rewards = deque(maxlen=window_size)
    self.recent_coverage = deque(maxlen=window_size)
    self.recent_success = deque(maxlen=window_size)
    self.recent_episode_lengths = deque(maxlen=window_size)

    # Convergence tracking (FIX LỖI 3: Dùng relative threshold)
    self.converged = False
    self.convergence_episode = None
    self.convergence_std_threshold = 0.05  # 5% của mean reward

def log_episode(self, metrics: Dict[str, Any]):
"""
Log episode - TỰ ĐỘNG QUYẾT ĐỊNH IN GÌ

    Tham số:
        metrics: Dict từ EpisodeLogger.finalize()
    """
    self.all_metrics.append(metrics)

    # Update moving windows
    self.recent_rewards.append(metrics['total_reward'])
    self.recent_coverage.append(metrics['coverage_rate'])
    self.recent_success.append(1 if metrics['success'] else 0)
    self.recent_episode_lengths.append(metrics['episode_length'])

    ep_id = metrics['episode_id']

    # Check convergence (chỉ sau 100 episodes đầu)
    if not self.converged and len(self.recent_rewards) == self.window_size:
        self._check_convergence(ep_id)

    # LEVEL 1: Mỗi episode - 1 DÒNG
    if self.verbose >= 1:
        self._print_episode_line(metrics)

    # LEVEL 2: Mỗi 100 episodes - SUMMARY
    if self.verbose >= 2 and (ep_id + 1) % 100 == 0:
        self._print_summary(last_n=100)

def \_print_episode_line(self, metrics: Dict[str, Any]):
"""In một dòng cho episode (compact)"""
success_icon = "✅" if metrics['success'] else "❌"

    # Thêm icon convergence nếu đã converge
    conv_icon = "🎯" if self.converged else ""

    print(f"Ep {metrics['episode_id']:4d} | "
          f"R: {metrics['total_reward']:6.1f} | "
          f"Cov: {metrics['coverage_rate']:5.1f}% | "
          f"Vic: {metrics['victims_found']:2d}/{metrics['total_victims']} | "  # FIX: Không cần .get()
          f"Len: {metrics['episode_length']:3d} | "
          f"{success_icon}{conv_icon}")

def \_check_convergence(self, episode: int):
"""Check convergence với threshold phù hợp cho reward âm."""
if len(self.recent_rewards) < self.window_size:
return

    mean_reward = np.mean(self.recent_rewards)
    std_reward = np.std(self.recent_rewards)
    success_rate = np.mean(self.recent_success)

    # FIX: Dùng absolute std nếu mean gần 0 hoặc âm
    if abs(mean_reward) > 10.0:
        relative_std = std_reward / abs(mean_reward)
        threshold = self.convergence_std_threshold  # 5%
    else:
        # Mean reward gần 0 → dùng absolute threshold
        relative_std = std_reward / 10.0  # normalize by reasonable scale
        threshold = 0.5  # absolute threshold

    if relative_std < threshold and success_rate > 0.5:
        self.converged = True
        self.convergence_episode = episode

        if self.verbose >= 1:
            print(f"\n🎯 CONVERGENCE DETECTED at episode {episode}")
            print(f"   Mean reward: {mean_reward:.2f}")
            print(f"   Std: {std_reward:.2f} (relative: {relative_std*100:.1f}%)")
            print(f"   Success rate: {success_rate*100:.1f}%\n")

def \_print_summary(self, last_n: int = 100):
"""In summary ngắn gọn"""
if not self.all_metrics:
return

    recent = self.all_metrics[-last_n:]

    # Tính stats
    rewards = [e['total_reward'] for e in recent]
    coverage = [e['coverage_rate'] for e in recent]
    success = [1 if e['success'] else 0 for e in recent]
    lengths = [e['episode_length'] for e in recent]
    collisions = [e['total_collisions'] for e in recent]

    print(f"\n{'='*70}")
    print(f"SUMMARY - LAST {last_n} EPISODES:")
    print(f"{'='*70}")
    print(f"Reward      : {np.mean(rewards):6.1f} ± {np.std(rewards):5.1f}")
    print(f"Coverage    : {np.mean(coverage):5.1f}% ± {np.std(coverage):4.1f}%")
    print(f"Success Rate: {np.mean(success)*100:5.1f}%")
    print(f"Avg Length  : {np.mean(lengths):5.1f} steps")
    print(f"Collisions  : {np.mean(collisions):5.2f} ± {np.std(collisions):4.2f}")
    print(f"{'='*70}\n")

def get_stats(self, last_n: Optional[int] = None) -> Dict[str, float]:
"""
Lấy stats - KHÔNG IN

    Tham số:
        last_n: Số episodes gần nhất (None = tất cả)

    Trả về:
        Dict chứa statistics
    """
    if not self.all_metrics:
        return {}

    if last_n is None:
        episodes = self.all_metrics
    else:
        episodes = self.all_metrics[-last_n:]

    rewards = [e['total_reward'] for e in episodes]
    coverage = [e['coverage_rate'] for e in episodes]
    success = [1 if e['success'] else 0 for e in episodes]
    lengths = [e['episode_length'] for e in episodes]

    return {
        'n_episodes': len(episodes),
        'reward_mean': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'coverage_mean': float(np.mean(coverage)),
        'coverage_std': float(np.std(coverage)),
        'success_rate': float(np.mean(success) * 100),
        'avg_episode_length': float(np.mean(lengths)),
        'converged': bool(self.converged),
        # FIX LỖI 4: Dùng is not None thay vì truthy check
        'convergence_episode': int(self.convergence_episode) if self.convergence_episode is not None else None,
    }

def get_overall_stats(self) -> Dict[str, float]:
"""
Lấy overall stats cho TOÀN BỘ training
(Dùng cho final report)
"""
return self.get_stats(last_n=None)

def save(self, filepath: str):
"""
Lưu ra file JSON (JSON-safe)

    Tham số:
        filepath: Đường dẫn file
    """
    with open(filepath, 'w') as f:
        json.dump(self.all_metrics, f, indent=2)

    if self.verbose >= 1:
        print(f"✅ Saved {len(self.all_metrics)} episodes to {filepath}")

def load(self, filepath: str):
"""
Load từ file JSON

    Tham số:
        filepath: Đường dẫn file
    """
    with open(filepath, 'r') as f:
        self.all_metrics = json.load(f)

    # Rebuild moving windows từ episodes gần nhất
    recent = self.all_metrics[-self.window_size:]
    for ep in recent:
        self.recent_rewards.append(ep['total_reward'])
        self.recent_coverage.append(ep['coverage_rate'])
        self.recent_success.append(1 if ep['success'] else 0)
        self.recent_episode_lengths.append(ep['episode_length'])

    if self.verbose >= 1:
        print(f"✅ Loaded {len(self.all_metrics)} episodes from {filepath}")

============================================================================
HELPER FUNCTIONS
============================================================================
def compare_training_runs(runs: List[TrainingLogger], labels: List[str]):
"""
So sánh nhiều training runs (cho Phase 2 & 3)

text

Tham số:
runs: Danh sách TrainingLogger objects
labels: Tên của từng run (vd: ["MAPPO", "MASAC", "MATD3"])
"""
print(f"\n{'='*80}")
print(f"TRAINING COMPARISON - FINAL RESULTS")
print(f"{'='*80}")
print(f"{'Algorithm':<15} | {'Reward':<15} | {'Coverage':<15} | {'Success':<10} | {'Converged'}")
print(f"{'-'\*80}")

for run, label in zip(runs, labels):
stats = run.get_overall_stats()

    # FIX: Dùng is not None
    conv_text = f"Ep {stats['convergence_episode']}" if stats['convergence_episode'] is not None else "No"

    print(f"{label:<15} | "
          f"{stats['reward_mean']:6.1f} ± {stats['reward_std']:5.1f} | "
          f"{stats['coverage_mean']:5.1f}% ± {stats['coverage_std']:4.1f}% | "
          f"{stats['success_rate']:5.1f}% | "
          f"{conv_text}")

print(f"{'='\*80}\n")
import numpy as np
from typing import Tuple, List

"""
utils/geometry.py - OPTIMIZED VERSION
Công cụ hình học cho SAR UAV Swarm với NumPy vectorization.

OPTIMIZATIONS:

get_circle_cells(): Vectorized với meshgrid (10× faster)
\_line_intersects_circle(): Optimized distance computation
Removed all debug logging
PERFORMANCE:

Before: ~2ms per get_circle_cells() call
After: ~0.2ms per call
Speedup: 10×
"""
def dist_2d(pos1: List, pos2: List) -> float:
"""
Tính khoảng cách Euclidean 2D giữa hai vị trí (x, y)

text

Tham số:
pos1: [x, y] hoặc [x, y, z] (z sẽ bị bỏ qua)
pos2: [x, y] hoặc [x, y, z] (z sẽ bị bỏ qua)

Trả về:
Khoảng cách tính bằng mét
"""
p1 = np.array(pos1[:2])
p2 = np.array(pos2[:2])
return float(np.linalg.norm(p1 - p2))
def dist_3d(pos1: List, pos2: List) -> float:
"""
Tính khoảng cách Euclidean 3D giữa hai vị trí (x, y, z)

text

Tham số:
pos1: [x, y, z]
pos2: [x, y, z]

Trả về:
Khoảng cách tính bằng mét
"""
p1 = np.array(pos1[:3])
p2 = np.array(pos2[:3])
return float(np.linalg.norm(p1 - p2))
def normalize_angle(angle: float) -> float:
"""
Chuẩn hóa về [-π, π]

text

OPTIMIZED: O(1) modulo operation

Rule theo unit test:
π -> π
-π -> -π
3π -> π
-3π -> π
"""
wrapped = ((angle + np.pi) % (2 \* np.pi)) - np.pi

if np.isclose(wrapped, -np.pi) and not np.isclose(angle, -np.pi):
return np.pi

return wrapped
def compute_bearing(from_pos: List, from_vel: List, to_pos: List) -> float:
"""
Tính góc bearing tương đối từ vị trí/vận tốc hiện tại đến vị trí đích

text

Định nghĩa bearing (từ hướng UAV đang bay):

- 0 rad: mục tiêu ở phía trước (cùng hướng vận tốc)
- π/2 rad: mục tiêu ở bên phải (rẽ phải 90°)
- -π/2 rad: mục tiêu ở bên trái (rẽ trái 90°)
- ±π rad: mục tiêu ở phía sau

Nếu vận tốc bằng 0, sử dụng hướng mặc định = 0 (hướng về trục +X)

Tham số:
from_pos: [x, y, ...] vị trí hiện tại
from_vel: [vx, vy, ...] vận tốc hiện tại
to_pos: [x, y, ...] vị trí đích

Trả về:
Bearing tính bằng radian, trong khoảng [-π, π]
"""
from_pos = np.array(from_pos[:2])
from_vel = np.array(from_vel[:2])
to_pos = np.array(to_pos[:2])

# Vector hướng đến mục tiêu

to_target = to_pos - from_pos

# Nếu mục tiêu tại vị trí hiện tại, bearing không xác định (trả về 0)

if np.linalg.norm(to_target) < 1e-6:
return 0

# Góc heading (hướng của vận tốc)

velocity_norm = np.linalg.norm(from_vel)
if velocity_norm < 1e-6: # Nếu không di chuyển, giả sử heading = 0 (hướng về trục +X)
heading_angle = 0
else:
heading_angle = np.arctan2(from_vel[1], from_vel[0])

# Góc đến mục tiêu từ vị trí hiện tại

target_angle = np.arctan2(to_target[1], to_target[0])

# Bearing tương đối

bearing = normalize_angle(target_angle - heading_angle)

return bearing
def check_los_2d(pos1: List, pos2: List, obstacles: List) -> bool:
"""
Kiểm tra có line-of-sight giữa hai vị trí 2D hay không (không có vật cản chắn)

text

Hỗ trợ nhiều loại obstacle:

- Hình tròn: tuple (center, radius)
- Polygon: object có attribute .polygon
- Custom: object có method .blocks_los(pos1, pos2)

Tham số:
pos1: [x, y, ...] vị trí bắt đầu
pos2: [x, y, ...] vị trí kết thúc
obstacles: Danh sách obstacles

Trả về:
True nếu LOS thông thoáng, False nếu bị chắn
"""
p1 = np.array(pos1[:2], dtype=np.float64)
p2 = np.array(pos2[:2], dtype=np.float64)

# Vector hướng

line_vec = p2 - p1
line_len = np.linalg.norm(line_vec)

if line_len < 1e-6:
return True # Cùng vị trí, không cần LOS

# Kiểm tra từng vật cản

for obstacle in obstacles: # Case 1: Tuple (center, radius) - Hình tròn
if isinstance(obstacle, tuple) and len(obstacle) == 2:
obs_center, obs_radius = obstacle
if \_line_intersects_circle(p1, p2, np.array(obs_center), obs_radius):
return False

    # Case 2: Object có method blocks_los
    elif hasattr(obstacle, 'blocks_los'):
        if obstacle.blocks_los(pos1, pos2):
            return False

    # Case 3: Object có polygon (để mở rộng sau)
    elif hasattr(obstacle, 'polygon'):
        try:
            from shapely.geometry import LineString  # ← Lazy import
            line = LineString([p1, p2])
            if obstacle.polygon.intersects(line):
                return False
        except ImportError:
            # Nếu chưa cài shapely, bỏ qua
            pass

return True
def \_line_intersects_circle(
p1: np.ndarray,
p2: np.ndarray,
center: np.ndarray,
radius: float
) -> bool:
"""
Kiểm tra đoạn thẳng có giao với hình tròn không (hàm helper)

text

OPTIMIZED: Vectorized distance computation

Tham số:
p1, p2: Hai đầu đoạn thẳng
center: Tâm hình tròn
radius: Bán kính

Trả về:
True nếu giao nhau
"""
line_vec = p2 - p1
line_len = np.linalg.norm(line_vec)

if line_len < 1e-6: # Line is a point
return np.linalg.norm(center - p1) < radius

line_dir = line_vec / line_len

# Vector từ p1 đến tâm

to_center = center - p1

# Chiếu lên đường thẳng

proj_length = np.dot(to_center, line_dir)
proj_length = np.clip(proj_length, 0, line_len)

# Điểm gần nhất trên đoạn thẳng

closest_point = p1 + proj_length \* line_dir

# Khoảng cách đến tâm

dist = np.linalg.norm(center - closest_point)

return dist < radius
def get_circle_cells(
center: List,
radius: float,
grid_size: int = 100,
map_size: float = 100.0
) -> np.ndarray:
"""
Lấy tất cả các ô lưới trong một hình tròn (để mapping coverage)

text

OPTIMIZED VERSION - Vectorized với NumPy meshgrid

PERFORMANCE:
Before (nested loops): ~2ms per call
After (vectorized): ~0.2ms per call
Speedup: 10×

Sử dụng thuật toán rasterization hình tròn với meshgrid

Tham số:
center: [x, y] tâm hình tròn trong tọa độ thế giới (mét)
radius: Bán kính tính bằng mét
grid_size: Số ô lưới (mặc định 100x100)
map_size: Kích thước bản đồ tính bằng mét (mặc định 100m)

Trả về:
Mảng shape (N, 2) chứa các chỉ số [hàng, cột] = [y, x] của các ô trong hình tròn
"""
center = np.array(center[:2], dtype=np.float64)

# Chuyển đổi tọa độ thế giới sang tọa độ lưới

# Thế giới: [0, map_size] → Lưới: [0, grid_size-1]

scale = grid_size / map_size
center_grid = (center _ scale).astype(np.int32)
radius_grid = int(radius _ scale)

# Kẹp tâm vào giới hạn lưới

cx = int(np.clip(center_grid[0], 0, grid_size - 1))
cy = int(np.clip(center_grid[1], 0, grid_size - 1))

# Tạo bounding box

# center_grid[0] = x_grid, center_grid[1] = y_grid

min_row = max(0, cy - radius_grid) # row = y
max_row = min(grid_size - 1, cy + radius_grid)
min_col = max(0, cx - radius_grid) # col = x
max_col = min(grid_size - 1, cx + radius_grid)

# ═══════════════════════════════════════════════════════════════

# VECTORIZED VERSION - 10× FASTER

# ═══════════════════════════════════════════════════════════════

# Tạo arrays cho rows và cols

rows = np.arange(min_row, max_row + 1, dtype=np.int32)
cols = np.arange(min_col, max_col + 1, dtype=np.int32)

# Tạo meshgrid (broadcast)

# rr[i, j] = rows[i], cc[i, j] = cols[j]

rr, cc = np.meshgrid(rows, cols, indexing='ij')

# Tính distances từ tất cả cells đến center (vectorized)

dx = cc - cx
dy = rr - cy
distances_sq = dx _ dx + dy _ dy # Dùng squared distance (tránh sqrt)

# Filter theo radius

radius_sq = radius_grid \* radius_grid
mask = distances_sq <= radius_sq

# Extract cells thỏa điều kiện

cells = np.column_stack([rr[mask], cc[mask]])

return cells
def get_circle_cells_legacy(
center: List,
radius: float,
grid_size: int = 100,
map_size: float = 100.0
) -> np.ndarray:
"""
LEGACY VERSION - Nested loops (SLOW)

text

Giữ lại để so sánh performance hoặc fallback.

DEPRECATED: Use get_circle_cells() instead.
"""
center = np.array(center[:2])

center_grid = (center / map_size _ grid_size).astype(int)
radius_grid = int(radius / map_size _ grid_size)

center_grid = np.clip(center_grid, 0, grid_size - 1)

min_row = max(0, center_grid[1] - radius_grid)
max_row = min(grid_size - 1, center_grid[1] + radius_grid)
min_col = max(0, center_grid[0] - radius_grid)
max_col = min(grid_size - 1, center_grid[0] + radius_grid)

cells = []

# Nested loops (SLOW!)

for row in range(min_row, max_row + 1):
for col in range(min_col, max_col + 1):
cell_center = np.array([col, row])
dist = np.linalg.norm(cell_center - center_grid)

        if dist <= radius_grid:
            cells.append([row, col])

return np.array(cells) if cells else np.empty((0, 2), dtype=int)
def get_relative_position(from_pos: List, to_pos: List) -> np.ndarray:
"""
Lấy vector vị trí tương đối (to_pos - from_pos)

text

Tham số:
from_pos: [x, y, z, ...]
to_pos: [x, y, z, ...]

Trả về:
Vị trí tương đối [dx, dy, dz]
"""
from_pos = np.array(from_pos[:3])
to_pos = np.array(to_pos[:3])
return to_pos - from_pos
def clip_position(pos: List, min_bounds: List, max_bounds: List) -> np.ndarray:
"""
Kẹp vị trí vào giới hạn bản đồ

text

Tham số:
pos: [x, y, z]
min_bounds: [min_x, min_y, min_z]
max_bounds: [max_x, max_y, max_z]

Trả về:
Vị trí đã được kẹp
"""
pos = np.array(pos)
min_bounds = np.array(min_bounds)
max_bounds = np.array(max_bounds)
return np.clip(pos, min_bounds, max_bounds)
#!/usr/bin/env python3
"""
🚁 MAPPO Training — HARD Stage Focus
Full auto-config system for production training
"""

import argparse
import torch
from dataclasses import dataclass

from config import AppConfig, STAGE_HARD,STAGE_MEDIUM
from training.algorithms.mappo.trainer import MAPPOTrainer

@dataclass
class AutoConfig:
"""Auto-computed training configuration."""
rollout_length: int
buffer_capacity: int
batch_size: int
safety_margin: float

def auto_compute_config(max_steps: int, n_envs: int, n_uav: int,
batch_size_hint: int = None, safety_factor: float = 1.5):
"""
Auto-compute optimal training hyperparameters.

text

GUARANTEES:
rollout_length ≥ max_steps × safety_factor
buffer_capacity = rollout_length × n_envs
"""

# Rollout must be >= episode length × safety factor

min_rollout = int(max_steps _ safety_factor)
rollout_length = ((min_rollout + 63) // 64) _ 64 # Align to 64

# Buffer capacity

buffer_capacity = rollout_length \* n_envs

# Batch size

if batch_size_hint is None:
batch_size = max(64, buffer_capacity // 8)
batch_size = ((batch_size + 63) // 64) \* 64
else:
min_batch = 64
max_batch = buffer_capacity // 2
batch_size = max(min_batch, min(batch_size_hint, max_batch))

safety = buffer_capacity / max(max_steps \* n_envs, 1)

return AutoConfig(
rollout_length=rollout_length,
buffer_capacity=buffer_capacity,
batch_size=batch_size,
safety_margin=safety,
)
def parse_args():
parser = argparse.ArgumentParser(description="MAPPO Training — HARD Stage")

text

# Core params

parser.add_argument("--total-episodes", type=int, default=3000,
help="Total training episodes")
parser.add_argument("--seed", type=int, default=42,
help="Random seed")
parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
help="Training device")
parser.add_argument("--run-name", type=str, default=None,
help="Custom run name (auto-generated if None)")

# Environment

parser.add_argument("--n-envs", type=int, default=1,
help="Number of parallel environments")
parser.add_argument("--max-steps", type=int, default=None,
help="Episode timeout (default: 400 from HARD stage)")
parser.add_argument("--map-size", type=int, default=None,
help="Map size (default: 250m from HARD stage)")

# Training

parser.add_argument("--batch-size", type=int, default=None,
help="Batch size (auto-computed if None)")
parser.add_argument("--safety-factor", type=float, default=1.5,
help="Rollout safety margin")
parser.add_argument("--n-epochs", type=int, default=None,
help="PPO epochs per update")
parser.add_argument("--lr-actor", type=float, default=None,
help="Actor learning rate")
parser.add_argument("--lr-critic", type=float, default=None,
help="Critic learning rate")

# Logging

parser.add_argument("--log-interval", type=int, default=50,
help="Detailed log every N episodes")
parser.add_argument("--viz-interval", type=int, default=100,
help="2D viz every N episodes (default: 5 × log-interval)")
parser.add_argument("--checkpoint-interval", type=int, default=100,
help="Save checkpoint every N episodes")

return parser.parse_args()
def main():
args = parse_args()

text

# Device

device = "cuda" if torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device

# Config - HARD stage only

cfg = AppConfig()
cfg.viz_mode = "2d" # ✅ Enable 2D visualization
cfg.apply_stage(STAGE_HARD)

# Apply overrides

cfg.env.n_uav = 4 # ✅ Fixed to 4 UAVs (không dùng args.n_uav)
if args.max_steps:
cfg.env.max_steps = args.max_steps
if args.map_size:
cfg.env.map_size = args.map_size
cfg.env.grid_size = args.map_size

# Auto-compute training config

auto_cfg = auto_compute_config(
max_steps=cfg.env.max_steps,
n_envs=args.n_envs,
n_uav=cfg.env.n_uav,
batch_size_hint=args.batch_size,
safety_factor=args.safety_factor,
)

# Apply auto-computed values

cfg.train.mappo_rollout_length = auto_cfg.rollout_length
cfg.train.mappo_batch_size = auto_cfg.batch_size

# Apply optional overrides

if args.n_epochs:
cfg.train.mappo_n_epochs = args.n_epochs
if args.lr_actor:
cfg.train.mappo_lr_actor = args.lr_actor
if args.lr_critic:
cfg.train.mappo_lr_critic = args.lr_critic

# ✅ Viz interval: 5× log interval

viz_interval = args.viz_interval if args.viz_interval else (args.log_interval \* 5)

# Estimate training

avg_ep_len = cfg.env.max_steps _ 0.85
steps_per_update = auto_cfg.rollout_length _ args.n_envs
eps_per_update = steps_per_update / avg_ep_len
est_updates = max(1, int(args.total_episodes / eps_per_update))

# Print config

print(f"\n{'='*70}")
print(f"🚁 MAPPO TRAINING — HARD STAGE")
print(f"{'='*70}")
print(f"ENVIRONMENT:")
print(f" Map Size: {cfg.env.map_size}×{cfg.env.map_size}m")
print(f" Max Steps: {cfg.env.max_steps}")
print(f" n_envs: {args.n_envs}")
print(f" n_uav: {cfg.env.n_uav}") # Always 4
print(f" Victims: {cfg.victim.n_victims_min}-{cfg.victim.n_victims_max}")
print(f" Debris: {cfg.obstacle.n_debris}")
print(f"")
print(f"AUTO-CONFIG:")
print(f" Rollout Length: {auto_cfg.rollout_length:,}")
print(f" Buffer Capacity: {auto_cfg.buffer_capacity:,}")
print(f" Batch Size: {auto_cfg.batch_size}")
print(f" Safety Margin: {auto_cfg.safety_margin:.2f}×")
print(f"")
print(f"TRAINING:")
print(f" Target Episodes: {args.total_episodes:,}")
print(f" Est. Updates: ~{est_updates:,}")
print(f" Episodes/Update: ~{eps_per_update:.1f}")
print(f" n_epochs: {cfg.train.mappo_n_epochs}")
print(f" LR (actor/critic): {cfg.train.mappo_lr_actor}/{cfg.train.mappo_lr_critic}")
print(f" Device: {device}")
print(f"")
print(f"LOGGING:")
print(f" Log interval: {args.log_interval} episodes")
print(f" Viz interval: {viz_interval} episodes (5× log)")
print(f" Checkpoint interval: {args.checkpoint_interval} episodes")
print(f"{'='\*70}\n")

# Runtime assertions

assert auto_cfg.rollout_length >= cfg.env.max_steps, \
 "Rollout too short for episode length"
assert auto_cfg.buffer_capacity >= cfg.env.max_steps \* args.n_envs, \
 "Buffer too small"

# Create trainer

trainer = MAPPOTrainer(
config=cfg,
device=device,
run_name=args.run_name,
n_envs=args.n_envs,
)

# Train

print(f"🚀 Starting training...\n")
trainer.train(
total_episodes=args.total_episodes,
curriculum_manager=None, # No curriculum
seed=args.seed,
log_every_n_eps=args.log_interval,
viz_every_n_eps=viz_interval,
checkpoint_every_n_eps=args.checkpoint_interval,
)

# Summary

print(f"\n{'='*70}")
print(f"✅ TRAINING COMPLETE")
print(f"{'='*70}")
print(f" Episodes: {trainer.total_episodes_done:,}")
print(f" Steps: {trainer.total_steps:,}")
print(f" Updates: {trainer.update_count:,}")
print(f" Final Checkpoint: {trainer.checkpoint_dir}/checkpoint_final.pt")
print(f"")
print(f"📊 Final Metrics:")
if trainer.ep_rewards:
import numpy as np
print(f" Reward: {np.mean(trainer.ep_rewards):.2f} ± {np.std(trainer.ep_rewards):.2f}")
print(f" Coverage: {np.mean(trainer.ep_coverage):.1f}%")
print(f" Victims Found: {np.mean(trainer.ep_victims):.1f}%")
print(f"")
print(f"🎬 Visualize results:")
print(f" python visualize_policy.py --checkpoint {trainer.checkpoint_dir}/checkpoint_final.pt")
print(f"{'='\*70}\n")
if name == "main":
main()
"""
🔍 Landing Debug Script
Chạy 10 episodes với verbose logging để track landing behavior
"""

import sys
import numpy as np
from config import AppConfig, STAGE_HARD
from env_setup.sar_pettingzoo_env import SARPettingZooEnv

def debug_landing():
cfg = AppConfig()
cfg.viz_mode = "2d"
cfg.apply_stage(STAGE_HARD)

text

# ✅ Override để dễ test

cfg.env.max_steps = 500 # Ngắn hơn để test nhanh
cfg.env.n_uav = 4

env = SARPettingZooEnv(cfg, render_mode=None)

for ep in range(10):
print(f"\n{'='*70}")
print(f"Episode {ep+1}/10")
print(f"{'='*70}")

    obs_dict, info = env.reset(seed=42 + ep)

    # Track per-UAV
    uav_states = {}  # {uav_id: [(step, state, battery, land_signal)]}
    for uid in range(4):
        uav_states[uid] = []

    step_count = 0
    done = False

    while not done and step_count < cfg.env.max_steps:
        # ✅ Random actions với fixed land probability để test
        actions = {}
        for agent in obs_dict.keys():
            uid = int(agent.split("_")[1])

            # Movement: random
            move = np.random.uniform(-1, 1, size=3).astype(np.float32)

            # Land signal: force to 0.9 if battery < 50% để test
            uav = env._base_env.backend.uavs[uid]   # ✅ Dùng _base_env
            if uav.battery < 50.0:
                land = 0.9  # Force land signal
            else:
                land = 0.0

            actions[agent] = np.concatenate([move, [land]])

            # Log state
            uav_states[uid].append((
                step_count,
                uav.state.value,
                uav.battery,
                land,
            ))

        obs_dict, rewards, terminations, truncations, infos = env.step(actions)
        done = any(terminations.values()) or any(truncations.values())
        step_count += 1

        # ✅ Log khi có state transition
        for uid in range(4):
            uav = env._base_env.backend.uavs[uid]   # ✅ Dùng _base_env
            if len(uav_states[uid]) >= 2:
                prev_state = uav_states[uid][-2][1]
                curr_state = uav_states[uid][-1][1]

                if prev_state != curr_state:
                    print(f"  [Step {step_count:3d}] UAV {uid}: "
                          f"{prev_state:8s} → {curr_state:8s} | "
                          f"battery={uav.battery:5.1f}% | "
                          f"land_signal={uav_states[uid][-1][3]:.2f}")

    # ✅ Episode summary
    print(f"\n📊 Episode {ep+1} Summary:")
    print(f"  Steps: {step_count}")

    for uid in range(4):
        uav = env._base_env.backend.uavs[uid]   # ✅ Dùng _base_env

        # Count transitions
        transitions = []
        for i in range(1, len(uav_states[uid])):
            if uav_states[uid][i][1] != uav_states[uid][i-1][1]:
                transitions.append(
                    f"{uav_states[uid][i-1][1]}→{uav_states[uid][i][1]}"
                )

        # Battery stats
        batteries = [s[2] for s in uav_states[uid]]

        print(f"  UAV {uid}: final_state={uav.state.value:8s} | "
              f"battery={uav.battery:5.1f}% | "
              f"min_bat={min(batteries):5.1f}% | "
              f"transitions={len(transitions)} {transitions[:3]}")

env.close()
print(f"\n✅ Debug complete. Check logs above for landing behavior.")
debug_landing()
bạn coi trước th ko cần làm gì cả

claude-sonnet-4-5-20250929
✅ ĐÃ ĐỌC VÀ HIỂU TOÀN BỘ 10 FILE CUỐI CÙNG
📦 Tổng hợp nội dung đã tiếp nhận:

1. BaselineReward v3.1 (rewards/baseline_reward.py)
   16 Components:
   Component Type Value Notes
   coverage_delta Dense +8.0 per 1% Shared, max +800
   victim_found Sparse +30 × urgency/5 Individual
   distance_shaping Dense ±1.0 cap Delta-based với memory
   battery_penalty Dense -5/-20/-50 Progressive thresholds
   battery_dead Sparse -200 One-time
   collision_obstacle Sparse -35 One-time per UAV
   proximity Dense -5/-1/-0.2 Pairwise, normalized by swarm size
   danger_zone Dense -1.0 to -5.0 Per-step inside zone
   time_penalty Dense -0.02 × n_active Gentle pressure
   terminal Sparse +200 base + ≤100 bonus 60% cov + 20% vic + 10% time + 10% bat
   penalty_cap_adjustment Additive cap - penalty_sum BUG-31 fix
   landing_reward Sparse +120 Tier 3: one-time CHARGING
   hover_penalty Dense -3.0 Tier 2: gần trạm không land
   approach_reward Dense +0.3×(1-dist) Tier 1: approach station
   Key Fixes:
   BUG-31: Penalty cap additive (không scale components)
   BUG-32: Proximity cap scale theo n_pairs / 6.0
   BUG-33: Delta shaping với \_prev_min_dist memory
   BUG-34: Terminal bonus không saturate, 4 tiers (60/20/10/10)
   BUG-35: Battery urgency shaping với distance-to-station
   Landing Reward System (3 Tiers):
   Python

# Tier 3: One-time success (sparse, strong)

if state == CHARGING AND uav_id not in \_landed_uavs:
reward += 120.0

# Tier 1: Approach (dense, weak)

if battery <= 40% AND state in (ACTIVE, RETURNING):
reward += 0.3 × (1 - norm_dist)

# Tier 2: Hover penalty (dense, medium)

if ACTIVE AND dist_to_station <= landing_range AND battery <= 40%:
penalty += -3.0 2. CommSensor (sensors/comm_sensor.py)
Scan Output: 9 dims (3 teammates × 3 features)
[norm_dist, norm_bearing, norm_alt] per teammate
Top-3 teammates trong comm_range=30m
Bearing normalized: (rad + π) / 2π ∈ [0,1] 3. FOVSensor (sensors/fov_sensor.py)
Detection Probability Pipeline (5 stages):
Python

# 1. Altitude factor

p = p_base × exp(-decay × altitude)

# 2. Environmental degradation (smoke/fire)

p × env_factor # smoke: 0.4, fire: 0.55

# 3. Motion blur

p × (1 - motion_blur_coeff × speed_ratio)

# 4. Victim type

p × victim_factor # injured: 1.15, mobile: 0.75-0.95

# 5. Hardware base miss rate

p × (1 - base_miss_rate) # 3%
Noise Factors:
env_factor: DangerZone.get_sensor_modifier() (0.4-1.0)
motion_blur: UAV speed → nhòe theo chiều ngang
victim_factor: Injured stationary → 1.15, Mobile → 0.75-0.95
base_miss_rate: 3% hardware limitation
Scan Methods:
scan_victims(): Top-5 in FOV → 25 dims [rel_x, rel_y, dist, urgency/5, is_found]
scan_obstacles(): Top-4 → 12 dims [rel_x, rel_y, type_id] (NO noise) 4. ObservationBuilder (observation/obs_builder.py)
68-dim Actor Obs Slices:
text

[0:11] self (pos/map, vel/max, battery/100, state_onehot[:4])
[11:19] stations (2×4: rel_pos/map, dist/diagonal, occupancy)
[19:28] teammates (3×3: CommSensor.scan())
[28:40] obstacles (4×3: FOVSensor.scan_obstacles())
[40:65] victims (5×5: FOVSensor.scan_victims())
[65:68] coverage (local_15m, local_30m, time_remaining)
554-dim Critic Obs:
text

[0:544] = 8 UAVs × 68 (zero-padded, sorted by uav.id)
[544:554] = 10 global dims:
n_active/n, n_charging/n, n_disabled/n, n_alive/n,
bat_mean, bat_std, bat_min (÷100),
global_coverage, victims_found_rate, time_remaining
FIX 3.3: Debug mode flag cfg.env.debug_obs cho NaN/Inf check 5. ChargingStation (entities/charging_station.py)
Key Methods:
in_range(uav_pos): dist_xy ≤ 3m AND z ≤ 0.5m
try_occupy(uav): Return False nếu full
charge(uav): Auto release khi out-of-range hoặc battery=100
force_release_all(): Dùng khi reset episode
Occupancy Tracking:
current_occupants: List[UAV]
occupant_ids: Set[int] → O(1) lookup 6. Obstacle (Debris + DangerZone) (entities/obstacle.py)
Multi-Shape Support:
Shape Params Detection Method
circle radius dist_2d ≤ radius
rectangle width, height_2d, rotation Shapely polygon (rotated box)
polygon vertices Shapely contains/covers
Debris:
causes_collision(uav_pos): in_zone_2d(xy) AND uav.z < height_3d
blocks_los(): Line-circle/polygon intersection
DangerZone:
is_inside(): Same as Debris, check z < max_height
blocks_los(): Chỉ fire và smoke block LOS
get_sensor_modifier(): smoke=0.4, fire=0.55, collapse=0.7
get_battery_modifier(): fire=0.05, others=0.0
FIX 2.2: Dùng covers() thay vì contains() (include boundary) 7. UAV (entities/uav.py)
UAVState Enum:
Python

ACTIVE → RETURNING → CHARGING → DEPLOYING → ACTIVE
↓
DISABLED (terminal)
Battery Convention:
self.battery = float ∈ [0, 100]
battery_pct (property) = alias for self.battery (BUG-36 fix)
Movement:
apply_action(action[3]): ACTIVE only, clip diagonal XY speed
auto_navigate(target): RETURNING/DEPLOYING, no-overshoot
Battery Drain (per SECOND × dt):
Python

drain_xy = drain_xy_pct_per_s × (speed_xy / max_xy)
drain_up = drain_z_up_pct_per_s × (vz_up / max_z)
drain_down = drain_z_down_pct_per_s × (vz_down / max_z)
drain_idle = drain_idle_pct_per_s
total_drain = (drain_xy + drain_up + drain_down + drain_idle) × dt
State Transitions:
set_state(): Guard DISABLED terminal, CHARGING→ACTIVE chỉ khi ≥80%
\_prev_state tracking cho landing detection 8. Victim (InjuredVictim + MobileVictim) (entities/victim.py)
BaseVictim:
update(step_count, obstacles): Alias cho step(obstacles) (backend compat)
mark_found(): One-time, gọi \_on_found() hook
InjuredVictim:
Stationary, urgency ∈ [4.0, 5.0]
speed = 0.0 → detection factor = 1.15
MobileVictim:
Random walk, urgency ∈ [1.0, 3.0]
speed ∈ [0.2, 0.4] m/s
Đổi hướng mỗi 20 steps
\_on_found(): Freeze speed=0.0 (BUG-04 fix)
\_check_obstacle_block(): Chỉ Debris chặn, DangerZone không chặn 9. Logger (EpisodeLogger + TrainingLogger) (utils/logger.py)
EpisodeLogger:
Python

# Core metrics

total_reward: float
coverage_rate: float # [0,1], convert → % khi finalize
victims_found: int
total_victims: int
episode_length: int

# Safety

collision_obstacle, collision_uav, collision_proximity
battery_deaths, danger_zone_entries, hot_swaps

# Landing tracking (NEW)

landing_events: List[Dict]
total_landings: int
total_charge_time: int
per_uav_landings: Dict[int, int]
Finalize Output:
Python

{
'episode_id', 'seed', 'duration', 'episode_length',
'total_reward', 'avg_reward_per_step',
'coverage_rate', # [0, 100] percent
'victims_found', 'total_victims', 'victims_found_rate',
'collision_obstacle', 'collision_uav', 'collision_proximity',
'total_collisions', 'battery_deaths', 'danger_zone_entries',
'hot_swaps',
'total_landings', 'total_charge_time', 'avg_charge_per_landing',
'landings_per_uav',
'success': bool(coverage_ratio >= 0.9),
}
TrainingLogger:
verbose: 0=silent, 1=per-episode line, 2=summary mỗi 100 eps
Convergence detection: std < 5% mean reward AND success_rate > 50%
get_stats(last_n): Statistics cho n episodes gần nhất
save(filepath): JSON export (JSON-safe với Python native types)
Key Fixes:
FIX 1: Tách coverage_ratio [0,1] vs coverage_percent [0,100]
FIX 2: Thêm total_victims vào metrics
FIX 3: Convergence dùng relative threshold (5% mean)
FIX 4: is not None thay vì truthy check
🔍 CRITICAL INSIGHTS:

1. Landing Mechanism (3-Tier System):
   Python

# Tier 1 (Dense): Approach reward khi battery ≤ 40%

if battery <= 40% AND state in (ACTIVE, RETURNING):
reward += 0.3 × (1 - normalized_dist_to_station)

# Tier 2 (Dense): Hover penalty gần trạm không land

if ACTIVE AND dist_to_station <= 6m AND battery <= 40%:
penalty += -3.0

# Tier 3 (Sparse): One-time success khi CHARGING

if state == CHARGING AND uav_id not in \_landed_uavs:
reward += 120.0
\_landed_uavs.add(uav_id) 2. Detection Noise Pipeline:
text

Geometric FOV → LOS check →
P_altitude × E_smoke × E_motion × E_victim × (1 - miss_rate) →
Bernoulli sample → DETECTED 3. Penalty Cap (BUG-31 Fix):
Python

# ADDITIVE adjustment (không scale components)

penalty_sum = sum(negative components)
if penalty_sum < cap:
components["penalty_cap_adjustment"] = cap - penalty_sum

# RL sees EXACT component values + knows cap was applied

4. Observation Normalization:
   Pos: /map_size
   Vel: /max_speed_xy hoặc /max_speed_z
   Battery: /100.0
   Distance: /map_diagonal (stations), /fov_radius (victims/obstacles)
   Bearing: (rad + π) / 2π ∈ [0,1]
5. Episode Termination:
   Python

# Done reasons:

"coverage" → win: coverage >= 90%
"victims" → win: all found
"disabled:battery_death" → fail: all dead by battery
"disabled:other" → fail: mixed causes
None → continue
✅ TOÀN BỘ HỆ THỐNG ĐÃ NẮMƯỠNG:
Execution Flow (Đầy đủ):
text

1. reset() → MapGenerator.generate() → Backend.reset()
2. loop:
   a. get_current_obs() (NO reset, cache)
   b. actor.get_action(obs[68]) → action[4]
   c. critic.get_value(global_obs[554]) → value
   d. env.step(actions): - apply_actions() → landing logic - step_physics() → battery drain/charge - step_world() → fleet/victims/coverage/detection - \_check_done() ← TRƯỚC reward (BUG-ENV-06) - compute_per_uav() → rewards_dict - log landing transitions (ACTIVE/RETURNING → CHARGING)
   e. buffer.add(...)
   f. if done: log episode + auto reset with NEW seed
3. buffer.compute_gae() → update() → clear()
   📋 SẴN SÀNG CHO TASK TIẾP THEO:
   Tôi đã nắm đầy đủ:

16 reward components với 3-tier landing system
Noise model (5-stage detection pipeline)
Observation builder (68-dim actor, 554-dim critic)
Entity mechanics (UAV states, battery drain, victim movement)
Multi-shape obstacles (circle/rectangle/polygon)
Logging system (episode + training với convergence detection)
All bug fixes (BUG-31 đến BUG-36, FIX 1-4)
Chờ lệnh implement MASAC/MATD3 hoặc task analysis/debugging. 🚀

"""
rewards/baseline_reward.py
Baseline Reward Function - v3.1 RESEARCH-GRADE

FIXES FROM v3.0:
BUG-31: Penalty cap → adjustment component (không scale từng thành phần)
BUG-32: Proximity cap → scale theo max_pairs thực tế
BUG-33: Distance shaping → delta-based với memory
BUG-34: Terminal bonus → scale trực tiếp theo terminal_cap
BUG-35: Battery urgency shaping → distance-to-station incentive

DESIGN PRINCIPLES (unchanged):

1. Sparse signals > Dense penalties
2. No saturation (wide clip bounds)
3. Multi-agent aware
4. Step penalty cap (additive, không distort)
5. Ablation study ready (toggleable components)
   """
   from future import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np

from entities.uav import UAV, UAVState
from utils.geometry import dist_2d, dist_3d

if TYPE_CHECKING:
from config import AppConfig
from core.coverage_map import CoverageMap
from core.fleet_manager import FleetManager
from entities.victim import BaseVictim

logger = logging.getLogger(name)

Baseline pairs để normalize proximity (4 UAVs → 6 pairs)
\_BASELINE_PAIRS = 6.0

class BaselineReward:
"""
Baseline Reward Function v3.1 - Research-grade.

text

Multi-agent RL compatible (MAPPO / MASAC / MATD3).

Key fixes vs v3.0: - Penalty cap: additive adjustment (không distort components) - Proximity: normalized theo swarm size - Shaping: delta-based với prev_pos memory - Terminal: không saturate với terminal_cap
"""

def **init**(self, cfg: "AppConfig") -> None:
self.cfg = cfg

    # ── Cache reward params ──────────────────────────────────────────────
    self._r_victim_base     = cfg.reward.r_victim_base
    self._r_coverage_delta  = cfg.reward.r_coverage_delta
    self._r_battery_dead    = cfg.reward.r_battery_dead
    self._r_collision_obs   = cfg.reward.r_collision_obstacle
    self._r_proximity_1m    = cfg.reward.r_proximity_1m
    self._r_proximity_2m    = cfg.reward.r_proximity_2m
    self._r_proximity_3m    = cfg.reward.r_proximity_3m
    self._r_time_penalty    = cfg.reward.r_time_penalty
    self._r_terminal_base   = cfg.reward.r_terminal_base

    # ── Caps & limits ────────────────────────────────────────────────────
    self._step_penalty_cap     = cfg.reward.step_penalty_cap
    self._proximity_cap        = cfg.reward.proximity_penalty_cap
    self._terminal_cap         = cfg.reward.terminal_bonus_cap
    self._clip_min             = cfg.reward.step_reward_clip_min
    self._clip_max             = cfg.reward.step_reward_clip_max

    # ── Shaping ──────────────────────────────────────────────────────────
    self._enable_shaping       = cfg.reward.enable_distance_shaping
    self._shaping_weight       = 0.1   # ✅ FIX BUG-33: giảm từ 0.5 → 0.1
    self._shaping_max          = cfg.reward.distance_shaping_max_per_uav

    # ── Proximity thresholds ─────────────────────────────────────────────
    self._PROX_1M = 1.0
    self._PROX_2M = 2.0
    self._PROX_3M = 3.0

    # ── UAV / env params ─────────────────────────────────────────────────
    self._collision_radius = cfg.uav.collision_radius
    self._max_steps        = cfg.env.max_steps
    self._map_size         = cfg.env.map_size

    # ── Per-episode tracking ─────────────────────────────────────────────
    self._battery_death_penalized: Set[int] = set()
    self._collision_penalized:     Set[int] = set()

    # ── Delta shaping memory: uav_id → prev_min_dist_to_victim ──────────
    # ✅ FIX BUG-33: delta-based shaping requires state memory
    self._prev_min_dist: Dict[int, float] = {}
    self._r_landing_success = 200    # Tier 3: one-time khi CHARGING
    self._r_hover_penalty   = -3.0    # Tier 2: lơ lửng gần trạm
    self._approach_weight   = 0.3     # Tier 1: approach reward weight
    self._landing_range     = cfg.env.charge_radius * 2.0

    # Per-episode: track UAVs đã land thành công
    self._landed_uavs: Set[int] = set()
    logger.info(
        f"BaselineReward v3.1 initialized: "
        f"clip=[{self._clip_min:.0f}, {self._clip_max:.0f}], "
        f"penalty_cap={self._step_penalty_cap:.0f}, "
        f"proximity_cap={self._proximity_cap:.0f}"
    )

# ═════════════════════════════════════════════════════════════════════════

# PUBLIC API

# ═════════════════════════════════════════════════════════════════════════

def reset(self) -> None:
"""Reset per-episode state."""
self.\_battery_death_penalized.clear()
self.\_collision_penalized.clear()
self.\_prev_min_dist.clear()
self.\_landed_uavs.clear()

def compute(
self,
uavs: List[UAV],
victims: List["BaseVictim"],
obstacles: List,
coverage_map: "CoverageMap",
fleet_manager: "FleetManager",
newly_found: List["BaseVictim"],
prev_coverage: float,
current_step: int,
done: bool,
stations: Optional[List] = None,
) -> Dict[str, float]:
"""
Compute global reward (shared across all UAVs).

    Args:
        stations: List of ChargingStation objects (for battery urgency shaping)

    Returns:
        Dict with component breakdown + total
    """
    cur_coverage = coverage_map.get_coverage_rate()
    active_uavs  = [u for u in uavs if u.state != UAVState.DISABLED]
    n_active     = max(len(active_uavs), 1)

    components: Dict[str, float] = dict.fromkeys(
        self.get_component_names(), 0.0
    )

    # ── 1. Positive rewards ──────────────────────────────────────────────
    components["coverage_delta"] = _coverage_delta_reward(
        prev_coverage, cur_coverage, self._r_coverage_delta
    )

    components["victim_found"] = _victim_found_reward(
        newly_found, self._r_victim_base
    )

    if self._enable_shaping:
        components["distance_shaping"] = self._delta_shaping_fleet(uavs, victims)

    # ── 2. Negative penalties ────────────────────────────────────────────
    bat_pen, bat_dead = self._battery_rewards(uavs, stations)
    components["battery_penalty"] = bat_pen
    components["battery_dead"]    = bat_dead

    components["collision_obstacle"] = self._collision_reward(uavs, obstacles)

    # ✅ FIX BUG-32: Normalize proximity cap theo swarm size
    n_pairs = max(n_active * (n_active - 1) / 2, 1)
    scaled_prox_cap = self._proximity_cap * (n_pairs / _BASELINE_PAIRS)
    proximity_raw = _proximity_reward(
        active_uavs,
        self._PROX_1M, self._PROX_2M, self._PROX_3M,
        self._r_proximity_1m, self._r_proximity_2m, self._r_proximity_3m,
    )
    components["proximity"] = max(proximity_raw, scaled_prox_cap)

    components["danger_zone"] = self._danger_reward(uavs, obstacles)
     # ✅ NEW: Landing rewards (Cách 1 + Cách 2)
    land_r, hover_p, approach_r = self._landing_rewards(uavs, stations)
    components["landing_reward"] = land_r
    components["hover_penalty"]  = hover_p
    components["approach_reward"] = approach_r

    components["fleet_incentive"] = 0.0  # deprecated

    components["time_penalty"] = self._r_time_penalty * n_active

    # ── 3. Terminal bonus ────────────────────────────────────────────────
    if done:
        components["terminal"] = self._terminal_bonus(
            cur_coverage, victims, current_step,
            uavs=uavs  # ← Thêm
        )

    # ── 4. ✅ FIX BUG-31: Additive cap (không distort components) ───────
    components = self._apply_penalty_cap(components)

    # ── 5. Total + clipping ──────────────────────────────────────────────
    components = self._finalize(components)

    # ── 6. Debug logging ─────────────────────────────────────────────────
    self._log_extreme_reward(components, current_step, coverage_map, uavs, obstacles)

    return components

def compute_per_uav(
self,
uav: UAV,
newly_found_by_uav: List["BaseVictim"],
uavs: List[UAV],
victims: List["BaseVictim"],
obstacles: List,
coverage_map: "CoverageMap",
fleet_manager: "FleetManager",
prev_coverage: float,
current_step: int,
done: bool,
stations: Optional[List] = None,
) -> Dict[str, float]:
"""
Compute per-agent reward (MAPPO/MASAC/MATD3).

    Per-agent design:
        - Shared rewards (coverage) → chia đều cho active UAVs
        - Individual rewards (victim) → chỉ agent discover
        - Individual penalties → chỉ agent vi phạm
    """
    components: Dict[str, float] = dict.fromkeys(
        self.get_component_names(), 0.0
    )

    # DISABLED agent → zero reward
    if uav.state == UAVState.DISABLED:
        components["raw_total"] = 0.0
        components["total"]     = 0.0
        return components

    active_uavs = [u for u in uavs if u.state != UAVState.DISABLED]
    n_active    = max(len(active_uavs), 1)
    cur_coverage = coverage_map.get_coverage_rate()

    # ── 1. Shared rewards ────────────────────────────────────────────────
    components["coverage_delta"] = _coverage_delta_reward(
        prev_coverage, cur_coverage, self._r_coverage_delta
    ) / n_active

    # ── 2. Individual rewards ────────────────────────────────────────────
    components["victim_found"] = _victim_found_reward(
        newly_found_by_uav, self._r_victim_base
    )

    if self._enable_shaping:
        components["distance_shaping"] = self._delta_shaping_single(uav, victims)

    # ── 3. Individual penalties ──────────────────────────────────────────
    components["battery_penalty"] = _battery_penalty_single(
        uav, self.cfg.reward, self.cfg.uav
    )
    components["battery_penalty"] += _battery_urgency_shaping(
        uav, stations, self._map_size
    )

    if uav.battery_death and uav.id not in self._battery_death_penalized:
        components["battery_dead"] = self._r_battery_dead
        self._battery_death_penalized.add(uav.id)

    components["collision_obstacle"] = self._collision_reward([uav], obstacles)

    # ✅ FIX BUG-32: Per-UAV proximity (vs all others)
    proximity_raw = _proximity_reward_single(
        uav, active_uavs,
        self._PROX_1M, self._PROX_2M, self._PROX_3M,
        self._r_proximity_1m, self._r_proximity_2m, self._r_proximity_3m,
    )
    # Cap per-UAV = cap_total / n_active (chia đều upper bound)
    per_uav_prox_cap = self._proximity_cap / n_active
    components["proximity"] = max(proximity_raw, per_uav_prox_cap)

    components["danger_zone"] = self._danger_reward([uav], obstacles)

    # ✅ NEW: Per-UAV landing rewards
    land_r, hover_p, approach_r = self._landing_rewards([uav], stations)
    components["landing_reward"]  = land_r
    components["hover_penalty"]   = hover_p
    components["approach_reward"] = approach_r

    components["fleet_incentive"] = 0.0

    components["time_penalty"] = self._r_time_penalty

    # ── 4. Terminal (shared) ─────────────────────────────────────────────
    if done:
        components["terminal"] = self._terminal_bonus(
            cur_coverage, victims, current_step,
            uavs=uavs  # ← Thêm
        ) / n_active

    # ── 5. ✅ FIX BUG-31: Additive cap ──────────────────────────────────
    per_uav_cap = self._step_penalty_cap / n_active
    components  = self._apply_penalty_cap(components, cap=per_uav_cap)

    # ── 6. Total + clipping ──────────────────────────────────────────────
    components = self._finalize(components, label=f"uav_{uav.id}")

    return components

def \_landing_rewards(
self,
uavs: List[UAV],
stations: Optional[List],
) -> Tuple[float, float, float]:
"""
3-tier landing reward system.

    Tier 1 — Approach reward (dense, weak):
        Khi battery ≤ 40% và ACTIVE/RETURNING
        → Reward tỉ lệ nghịch với distance to nearest station
        → Gentle gradient hướng về trạm

    Tier 2 — Hover penalty (dense, medium):
        Khi UAV ACTIVE, gần trạm (≤ landing_range),
        battery ≤ 40% nhưng KHÔNG land
        → Phá local optimum "đứng im gần trạm"

    Tier 3 — Landing success (sparse, strong):
        Khi UAV vừa chuyển sang CHARGING (one-time per episode)
        → Lớn để signal rõ ràng

    Returns:
        (landing_reward, hover_penalty, approach_reward)
    """
    if not stations:
        return 0.0, 0.0, 0.0

    landing_total  = 0.0
    hover_total    = 0.0
    approach_total = 0.0

    max_dist = float(self._map_size) * 1.414   # diagonal

    for uav in uavs:
        if uav.state == UAVState.DISABLED:
            continue

        # Distance to nearest station
        min_dist = min(
            float(np.sqrt(
                (uav.pos[0] - s.pos[0]) ** 2 +
                (uav.pos[1] - s.pos[1]) ** 2
            ))
            for s in stations
        )

        # ── Tier 3: Landing success (one-time per episode) ────────────
        if (uav.state == UAVState.CHARGING
                and uav.id not in self._landed_uavs):
            landing_total += self._r_landing_success
            self._landed_uavs.add(uav.id)

        # ── Tier 1: Approach reward ───────────────────────────────────
        # Chỉ khi cần sạc (battery ≤ 40%) và đang bay
        if (uav.battery_pct <= 40.0
                and uav.state in (UAVState.ACTIVE, UAVState.RETURNING)):
            norm_dist     = min(min_dist / max(max_dist, 1.0), 1.0)
            approach_rew  = self._approach_weight * (1.0 - norm_dist)
            approach_total += approach_rew

        # ── Tier 2: Hover penalty ─────────────────────────────────────
        # UAV ACTIVE, gần trạm, cần sạc → nên land nhưng không land
        if (uav.state == UAVState.ACTIVE
                and min_dist <= self._landing_range
                and uav.battery_pct <= 40.0):
            hover_total += self._r_hover_penalty

    return landing_total, hover_total, approach_total

# ═════════════════════════════════════════════════════════════════════════

# PRIVATE: PENALTY CAP (FIX BUG-31)

# ═════════════════════════════════════════════════════════════════════════

def \_apply_penalty_cap(
self,
components: Dict[str, float],
cap: Optional[float] = None,
) -> Dict[str, float]:
"""
✅ FIX BUG-31: Apply penalty cap ADDITIVELY.

    KHÔNG scale từng component → giữ nguyên relative importance.
    Thêm "penalty_cap_adjustment" nếu tổng penalty vượt quá cap.

    Args:
        components: Current component dict
        cap: Override cap (default: self._step_penalty_cap)

    Returns:
        Updated component dict
    """
    if cap is None:
        cap = self._step_penalty_cap

    penalty_sum = sum(
        v for k, v in components.items()
        if v < 0 and k not in ("raw_total", "total", "penalty_cap_adjustment")
    )

    if penalty_sum < cap:
        # Additive adjustment → không thay đổi relative importance
        # RL sees exact component values + knows cap was applied
        adjustment = cap - penalty_sum
        components["penalty_cap_adjustment"] = adjustment
        logger.debug(
            f"Penalty cap applied: sum={penalty_sum:.2f} → "
            f"adjustment={adjustment:.2f} → effective={cap:.2f}"
        )

    return components

# ═════════════════════════════════════════════════════════════════════════

# PRIVATE: TERMINAL BONUS (FIX BUG-34)

# ═════════════════════════════════════════════════════════════════════════

def \_terminal_bonus(
self,
coverage_rate: float,
victims: List["BaseVictim"],
current_step: int,
uavs: Optional[List[UAV]] = None, # ← Thêm param
) -> float:
"""Terminal bonus với battery survival reward."""
n_total = max(len(victims), 1)
n_found = sum(1 for v in victims if v.is_found)
found_ratio = n_found / n_total
time_ratio = current_step / max(self.\_max_steps, 1)

    coverage_bonus = self._terminal_cap * 0.60 * coverage_rate   # ← 70% → 60%
    victim_bonus   = self._terminal_cap * 0.20 * found_ratio
    time_bonus     = (
        self._terminal_cap * 0.10 * (1.0 - time_ratio)
        if coverage_rate >= 0.80 else 0.0
    )

    # ✅ NEW: Battery survival bonus (10%)
    battery_bonus = 0.0
    if uavs is not None:
        alive = [u for u in uavs if u.state != UAVState.DISABLED]
        if alive:
            mean_bat = np.mean([u.battery_pct for u in alive])
            # Reward UAV còn pin nhiều khi kết thúc
            battery_bonus = self._terminal_cap * 0.10 * (mean_bat / 100.0)

    raw = coverage_bonus + victim_bonus + time_bonus + battery_bonus
    return float(np.clip(raw, 0.0, self._terminal_cap))

# ═════════════════════════════════════════════════════════════════════════

# PRIVATE: DELTA SHAPING (FIX BUG-33)

# ═════════════════════════════════════════════════════════════════════════

def \_delta_shaping_fleet(
self,
uavs: List[UAV],
victims: List["BaseVictim"],
) -> float:
"""
✅ FIX BUG-33: Delta-based shaping (fleet total).

    Reward = prev_min_dist - current_min_dist (per UAV)
    → Positive khi approaching, negative khi retreating
    → Không thể "farm" bằng cách đứng yên

    Theoretical basis: Potential-based reward shaping (Ng et al. 1999)
    → Không thay đổi optimal policy
    """
    unfound = [v for v in victims if not v.is_found]
    if not unfound:
        return 0.0

    total = 0.0
    for uav in uavs:
        total += self._delta_shaping_single(uav, victims, unfound)
    return total

def \_delta_shaping_single(
self,
uav: UAV,
victims: List["BaseVictim"],
unfound: Optional[List["BaseVictim"]] = None,
) -> float:
"""
✅ FIX BUG-33: Delta shaping per UAV.

    Algorithm:
        1. Compute current min distance to unfound victims
        2. Compare with prev (stored in self._prev_min_dist)
        3. Reward = delta (capped at shaping_max)
        4. Update prev

    Edge cases:
        - First step: no prev → store, return 0
        - No unfound: return 0
        - DISABLED/RETURNING: return 0
    """
    if uav.state not in (UAVState.ACTIVE, UAVState.DEPLOYING):
        # Don't update prev_dist for inactive UAVs
        return 0.0

    if unfound is None:
        unfound = [v for v in victims if not v.is_found]
    if not unfound:
        return 0.0

    # Current min distance
    current_min = min(dist_2d(uav.pos, v.pos) for v in unfound)

    # Delta shaping
    uav_id   = uav.id
    prev_min = self._prev_min_dist.get(uav_id, None)

    # Update memory
    self._prev_min_dist[uav_id] = current_min

    if prev_min is None:
        # First step → no reward, just initialize
        return 0.0

    # ✅ reward = approach (positive) or retreat (negative)
    delta = prev_min - current_min
    shaped = delta * self._shaping_weight

    # Cap per UAV để tránh dominate
    return float(np.clip(shaped, -self._shaping_max, self._shaping_max))

# ═════════════════════════════════════════════════════════════════════════

# PRIVATE: BATTERY REWARDS

# ═════════════════════════════════════════════════════════════════════════

def \_battery_rewards(
self,
uavs: List[UAV],
stations: Optional[List] = None,
) -> Tuple[float, float]:
"""Battery penalty + dead (one-time) + urgency shaping."""
bat_penalty = 0.0
bat_dead = 0.0

    for uav in uavs:
        if uav.state != UAVState.DISABLED:
            bat_penalty += _battery_penalty_single(
                uav, self.cfg.reward, self.cfg.uav
            )
            # ✅ BUG-35: Battery urgency shaping
            bat_penalty += _battery_urgency_shaping(
                uav, stations, self._map_size
            )

        if uav.battery_death and uav.id not in self._battery_death_penalized:
            bat_dead += self._r_battery_dead
            self._battery_death_penalized.add(uav.id)

    return bat_penalty, bat_dead

def \_collision_reward(self, uavs: List[UAV], obstacles: List) -> float:
"""Collision penalty (one-time per UAV)."""
from entities.obstacle import Debris
penalty = 0.0
for uav in uavs:
if uav.state == UAVState.DISABLED:
continue
if uav.id in self.\_collision_penalized:
continue
for obs in obstacles:
if isinstance(obs, Debris) and obs.causes_collision(uav.pos):
penalty += self.\_r_collision_obs
self.\_collision_penalized.add(uav.id)
break
return penalty

def \_danger_reward(self, uavs: List[UAV], obstacles: List) -> float:
"""Danger zone penalty (per step)."""
from entities.obstacle import DangerZone
penalty = 0.0
for uav in uavs:
if uav.state == UAVState.DISABLED:
continue
for obs in obstacles:
if isinstance(obs, DangerZone) and obs.is_inside(uav.pos):
penalty += obs.penalty
return penalty

# ═════════════════════════════════════════════════════════════════════════

# PRIVATE: FINALIZE

# ═════════════════════════════════════════════════════════════════════════

def \_finalize(
self,
components: Dict[str, float],
label: str = "global",
) -> Dict[str, float]:
"""Compute raw_total + apply clip."""
raw_total = sum(
v for k, v in components.items()
if k not in ("raw_total", "total")
)
components["raw_total"] = raw_total
components["total"] = float(
np.clip(raw_total, self.\_clip_min, self.\_clip_max)
)
\_assert_no_nan_inf(components["total"], f"finalize.{label}")
return components

# ═════════════════════════════════════════════════════════════════════════

# PRIVATE: LOGGING

# ═════════════════════════════════════════════════════════════════════════

def \_log_extreme_reward(
self,
components: Dict[str, float],
current_step: int,
coverage_map: "CoverageMap",
uavs: List[UAV],
obstacles: List,
) -> None:
"""Log extreme rewards cho debugging."""
total = components.get("total", 0.0)
if total > -50.0:
return

    from entities.obstacle import DangerZone
    n_in_danger = sum(
        1 for u in uavs
        for obs in obstacles
        if isinstance(obs, DangerZone) and obs.is_inside(u.pos)
    )

    # Format component breakdown
    breakdown = "\n".join(
        f"    {k:<28} {v:>+8.2f}"
        for k, v in components.items()
        if k not in ("raw_total", "total") and abs(v) > 0.001
    )

    # logger.warning(
    #     f"[STEP {current_step}] EXTREME REWARD: {total:.1f}\n"
    #     f"  Raw: {components['raw_total']:.1f} → clipped: {total:.1f}\n"
    #     f"  Breakdown:\n{breakdown}\n"
    #     f"  Context: danger={n_in_danger} UAVs, "
    #     f"collisions={len(self._collision_penalized)}, "
    #     f"coverage={coverage_map.get_coverage_rate():.1%}"
    # )

# ═════════════════════════════════════════════════════════════════════════

# PUBLIC UTILITIES

# ═════════════════════════════════════════════════════════════════════════

def get_component_names(self) -> List[str]:
return [
"coverage_delta",
"victim_found",
"distance_shaping",
"battery_penalty",
"battery_dead",
"collision_obstacle",
"proximity",
"danger_zone",
"fleet_incentive",
"time_penalty",
"terminal",
"penalty_cap_adjustment", # ✅ NEW: BUG-31 fix
"landing_reward", # ✅ NEW: Tier 3 (one-time success)
"hover_penalty", # ✅ NEW: Tier 2 (dense penalty)
"approach_reward", # ✅ NEW: Tier 1 (dense approach)
"raw_total",
"total",
]

def summarize(self, reward_dict: Dict[str, float]) -> str:
"""Compact summary cho logging."""
skip = {"raw_total", "total", "fleet_incentive"}
parts = [
f"{k}={v:+.2f}"
for k, v in reward_dict.items()
if k not in skip and abs(v) > 0.001
]
return (
f"[{', '.join(parts)}] → "
f"raw={reward_dict.get('raw_total', 0):+.2f} "
f"total={reward_dict.get('total', 0):+.2f}"
)

def **repr**(self) -> str:
return (
f"BaselineReward(v3.1, "
f"clip=[{self._clip_min:.0f}, {self._clip_max:.0f}], "
f"penalty_cap={self.\_step_penalty_cap:.0f}, "
f"shaping_weight={self.\_shaping_weight})"
)
=============================================================================
MODULE-LEVEL FUNCTIONS (stateless, unit-testable)
=============================================================================
def \_coverage_delta_reward(
prev_coverage: float,
cur_coverage: float,
weight: float,
) -> float:
"""Coverage delta reward (clamped to non-negative)."""
delta = max(0.0, cur_coverage - prev_coverage)
return delta \* weight

def \_victim_found_reward(
newly_found: List["BaseVictim"],
r_victim_base: float,
) -> float:
"""Urgency-weighted victim discovery reward."""
if not newly_found:
return 0.0
return sum(r_victim_base \* (v.urgency / 5.0) for v in newly_found)

def \_battery_penalty_single(
uav: UAV,
reward_cfg: "RewardConfig",
uav_cfg: "UAVConfig",
) -> float:
"""
Progressive battery penalty cho single UAV.

text

Thresholds từ UAVConfig (không hardcode):
20% → r_battery_20
10% → r_battery_10
5% → r_battery_5
"""
bat = uav.battery_pct
if bat <= uav_cfg.battery_emergency_pct:
return reward_cfg.r_battery_5
if bat <= uav_cfg.battery_critical_pct:
return reward_cfg.r_battery_10
if bat <= uav_cfg.battery_warning_pct:
return reward_cfg.r_battery_20
return 0.0
TRONG baseline_reward.py
Thay hàm \_battery_urgency_shaping():
def \_battery_urgency_shaping(uav, stations, map_size):

text

if stations is None or not stations:
return 0.0

bat = uav.battery_pct

# ✅ FIX: Không apply cho RETURNING (agent đang quay về rồi)

if bat > 30.0 or uav.state in (UAVState.CHARGING, UAVState.RETURNING, UAVState.DISABLED):
return 0.0

# Severity mạnh hơn

# ✅ FIX: Trigger sớm hơn + severity cao hơn

if bat <= 5.0:
severity = 15.0 # Tăng từ 10.0
elif bat <= 10.0:
severity = 8.0 # Tăng từ 5.0
elif bat <= 20.0:
severity = 3.0 # Tăng từ 2.0
elif bat <= 30.0:
severity = 1.0 # NEW: thêm tier 30%
else:
return 0.0 # Không penalty khi > 30%

min_dist = min(dist_2d(uav.pos, s.pos) for s in stations)
normalized_dist = min_dist / max(map_size, 1.0)

# Penalty tỉ lệ thuận với distance × severity

penalty = -normalized_dist _ severity _ 5.0 # Giữ nguyên multiplier

return float(np.clip(penalty, -10.0, 0.0)) # Tăng cap từ -5.0 lên -10.0
def \_proximity_reward(
active_uavs: List[UAV],
thresh_1m: float,
thresh_2m: float,
thresh_3m: float,
r_1m: float,
r_2m: float,
r_3m: float,
) -> float:
"""
Pairwise proximity penalty (fleet total, uncapped).

text

Caller handles cap (normalized theo swarm size).
"""
penalty = 0.0
for i in range(len(active_uavs)):
for j in range(i + 1, len(active_uavs)):
d = dist_3d(active_uavs[i].pos, active_uavs[j].pos)
if d <= thresh_1m:
penalty += r_1m
elif d <= thresh_2m:
penalty += r_2m
elif d <= thresh_3m:
penalty += r_3m
return penalty
def \_proximity_reward_single(
uav: UAV,
active_uavs: List[UAV],
thresh_1m: float,
thresh_2m: float,
thresh_3m: float,
r_1m: float,
r_2m: float,
r_3m: float,
) -> float:
"""Proximity penalty cho single UAV vs all others."""
penalty = 0.0
for other in active_uavs:
if other.id == uav.id:
continue
d = dist_3d(uav.pos, other.pos)
if d <= thresh_1m:
penalty += r_1m
elif d <= thresh_2m:
penalty += r_2m
elif d <= thresh_3m:
penalty += r_3m
return penalty

def \_assert_no_nan_inf(value: float, label: str) -> None:
"""Sanity check: no NaN/Inf in reward."""
assert not np.isnan(value), f"NaN detected in {label}"
assert not np.isinf(value), f"Inf detected in {label}"
from future import annotations

import logging
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from utils.geometry import dist_3d, compute_bearing

✅ REMOVED: sys.path.append hack
if TYPE_CHECKING:
from entities.uav import UAV
from config import AppConfig

logger = logging.getLogger(name)

"""
sensors/comm_sensor.py
Communication Sensor - V2V giữa các UAV trong COMM_RANGE
"""

class CommSensor:
"""V2V Communication Sensor."""

text

def **init**(self, cfg: "AppConfig") -> None:
self.cfg = cfg
self.\_n_tracked = cfg.obs.n_tracked_uavs
self.\_dims_per_uav = 3
self.obs_dim = self.\_n_tracked \* self.\_dims_per_uav
self.\_comm_range = float(cfg.sensor.comm_range)
self.\_z_max = float(cfg.uav.z_max)

def scan(
self,
ego_uav: "UAV",
all_active_uavs: List["UAV"],
) -> np.ndarray:
"""Quét teammates gần nhất trong COMM_RANGE."""
result = np.zeros(self.obs_dim, dtype=np.float32)
ego_pos = ego_uav.pos
ego_vel = ego_uav.vel

    candidates: List[Tuple[float, "UAV"]] = []
    for uav in all_active_uavs:
        if uav.id == ego_uav.id:
            continue
        d = dist_3d(ego_pos, uav.pos)
        if d > self._comm_range:
            continue
        candidates.append((d, uav))

    if not candidates:
        return result

    candidates.sort(key=lambda x: x[0])

    for i, (d, teammate) in enumerate(candidates[:self._n_tracked]):
        base         = i * self._dims_per_uav
        norm_dist    = float(np.clip(d / self._comm_range, 0.0, 1.0))
        bearing_rad  = compute_bearing(ego_pos, ego_vel, teammate.pos)
        norm_bearing = float(
            np.clip((bearing_rad + np.pi) / (2.0 * np.pi), 0.0, 1.0)
        )
        rel_alt  = float(teammate.pos[2]) - float(ego_pos[2])
        norm_alt = float(np.clip(rel_alt / self._z_max, -1.0, 1.0))

        result[base + 0] = norm_dist
        result[base + 1] = norm_bearing
        result[base + 2] = norm_alt

    return result

def get_n_in_range(self, ego_uav: "UAV", all_uavs: List["UAV"]) -> int:
return sum(
1 for uav in all_uavs
if uav.id != ego_uav.id
and dist_3d(ego_uav.pos, uav.pos) <= self.\_comm_range
)

def get*teammates_in_range(
self, ego_uav: "UAV", all_uavs: List["UAV"]
) -> List["UAV"]:
in_range = [
(dist_3d(ego_uav.pos, uav.pos), uav)
for uav in all_uavs
if uav.id != ego_uav.id
and dist_3d(ego_uav.pos, uav.pos) <= self._comm_range
]
in_range.sort(key=lambda x: x[0])
return [uav for *, uav in in_range]
sensors/fov_sensor.py - Enhanced với noise
from future import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from utils.geometry import dist_2d, check_los_2d

if TYPE_CHECKING:
from entities.uav import UAV
from entities.victim import BaseVictim
from config import AppConfig

logger = logging.getLogger(name)

class FOVSensor:
"""
Field-of-View Sensor với noise model.

text

Noise pipeline:
P_final = P_altitude × E_smoke × E_motion × E_victim

    P_altitude: cao → khó thấy (đã có)
    E_smoke:    victim trong smoke/fire → khó thấy hơn
    E_motion:   UAV bay nhanh → motion blur → khó thấy hơn
    E_victim:   Injured stationary → dễ thấy hơn mobile

"""

def **init**(self, cfg: "AppConfig") -> None:
self.cfg = cfg
self.\_fov_tan = float(cfg.sensor.fov_tan)
self.\_p_base = float(cfg.sensor.p_detect_base)
self.\_p_decay = float(cfg.sensor.p_detect_decay)
self.\_n_victims = cfg.obs.n_obs_victims
self.\_n_obstacles = cfg.obs.n_obs_obstacles
self.\_v_dims = 5
self.\_o_dims = 3
self.victim_obs_dim = self.\_n_victims _ self.\_v_dims
self.obstacle_obs_dim = self.\_n_obstacles _ self.\_o_dims

    # ── Noise params (từ SensorConfig hoặc default) ──────────
    self._enable_noise      = getattr(cfg.sensor, "enable_noise",       True)
    self._motion_blur_coeff = getattr(cfg.sensor, "motion_blur_coeff",  0.06)
    self._base_miss_rate    = getattr(cfg.sensor, "base_miss_rate",     0.03)

    # RNG riêng cho sensor (reproducible)
    self._rng = np.random.default_rng()

def set_seed(self, seed: int) -> None:
"""Set seed cho reproducible evaluation."""
self.\_rng = np.random.default_rng(seed)

# ── Core: FOV geometry ────────────────────────────────────────────────────

def calculate_fov_radius(self, altitude: float) -> float:
return float(altitude) \* self.\_fov_tan

# ── Core: Detection probability ───────────────────────────────────────────

def calculate_detection_prob(
self,
altitude: float,
uav_speed: float = 0.0,
env_factor: float = 1.0,
victim_factor: float = 1.0,
) -> float:
"""
Tính P(detect) với đầy đủ noise factors.

    Công thức:
        p = p_base × exp(-decay × alt)   ← altitude factor (đã có)
        p = p × env_factor               ← smoke/fire degradation
        p = p × (1 - motion_penalty)     ← motion blur
        p = p × victim_factor            ← victim type
        p = p × (1 - base_miss_rate)     ← hardware limitation

    Args:
        altitude:      UAV altitude (m)
        uav_speed:     UAV horizontal speed (m/s)
        env_factor:    Environmental visibility [0.0, 1.0]
                       1.0 = clear sky
                       0.4 = heavy smoke
                       0.2 = fire zone
        victim_factor: Victim detectability [0.75, 1.15]
                       1.15 = injured (stationary, dễ thấy)
                       0.75 = mobile (đang chạy, khó thấy)

    Returns:
        p ∈ [0.0, 1.0]
    """
    # ── Step 1: Base altitude probability ────────────────────
    # altitude=3m  → 0.95 × exp(-0.12) = 0.84
    # altitude=20m → 0.95 × exp(-0.80) = 0.43
    # altitude=40m → 0.95 × exp(-1.60) = 0.19
    p = self._p_base * np.exp(-self._p_decay * float(altitude))

    if not self._enable_noise:
        return float(np.clip(p, 0.0, 1.0))

    # ── Step 2: Environmental degradation ────────────────────
    # Victim đứng trong smoke → camera khó thấy
    # env_factor đến từ DangerZone.get_sensor_modifier()
    #   smoke:     0.40 → p giảm 60%
    #   fire:      0.55 → p giảm 45%
    #   gas:       0.80 → p giảm 20%
    #   radiation: 0.90 → p giảm 10%
    #   collapse:  0.95 → p giảm 5%
    p *= env_factor

    # ── Step 3: Motion blur ───────────────────────────────────
    # UAV bay nhanh → camera bị nhòe theo chiều ngang
    # max_speed = 5 m/s, coeff = 0.06
    # speed=0   → penalty = 0.0  → p không đổi
    # speed=2.5 → penalty = 0.03 → p giảm 3%
    # speed=5.0 → penalty = 0.06 → p giảm 6%
    max_speed = float(self.cfg.uav.max_speed_xy_mps)
    if max_speed > 0 and uav_speed > 0:
        speed_ratio    = np.clip(uav_speed / max_speed, 0.0, 1.0)
        motion_penalty = self._motion_blur_coeff * speed_ratio
        p *= (1.0 - motion_penalty)

    # ── Step 4: Victim type factor ────────────────────────────
    # Injured (nằm im) → dễ nhận dạng hơn (1.15)
    # Mobile (đang chạy) → khó nhận dạng hơn (0.75-0.90)
    p *= victim_factor

    # ── Step 5: Hardware base miss rate ──────────────────────
    # Dù điều kiện hoàn hảo, camera vẫn có 3% miss rate
    # (lens flare, compression artifacts, etc.)
    p *= (1.0 - self._base_miss_rate)

    return float(np.clip(p, 0.0, 1.0))

# ── Noise factor extractors ───────────────────────────────────────────────

def \_get_env_factor(
self,
victim_pos: np.ndarray,
obstacles: list,
) -> float:
"""
Tính environmental visibility tại vị trí victim.

    Logic:
        - Duyệt qua tất cả DangerZones
        - Nếu victim đứng trong zone → lấy sensor_modifier
        - Nếu nhiều zones overlap → lấy worst case (min)

    Returns:
        1.0 = clear (không có danger zone)
        0.40 = heavy smoke
        0.20 = fire
    """
    if not obstacles:
        return 1.0

    from entities.obstacle import DangerZone

    env_factor = 1.0
    for obs in obstacles:
        if not isinstance(obs, DangerZone):
            continue
        # Check nếu victim position nằm trong zone (2D check)
        if obs.is_inside(victim_pos):
            modifier   = obs.get_sensor_modifier()
            env_factor = min(env_factor, modifier)

    return float(env_factor)

def \_get_victim_factor(self, victim: "BaseVictim") -> float:
"""
Victim type ảnh hưởng detection:

    InjuredVictim (stationary):
        - Nằm im → thermal signature ổn định → dễ detect
        - factor = 1.15

    MobileVictim (moving):
        - Đang di chuyển → blur, khó lock
        - speed cao → factor thấp hơn
        - factor = 0.75 ~ 0.95

    BaseVictim (unknown):
        - factor = 1.0
    """
    from entities.victim import InjuredVictim, MobileVictim

    if isinstance(victim, InjuredVictim):
        return 1.15

    if isinstance(victim, MobileVictim):
        speed = float(getattr(victim, "speed", 0.3))
        # speed=0.2 m/s → factor=0.93
        # speed=0.4 m/s → factor=0.83
        # speed=0.6 m/s → factor=0.75
        factor = 1.0 - np.clip(speed * 0.5, 0.05, 0.25)
        return float(factor)

    return 1.0

# ── Core: Single victim detection ─────────────────────────────────────────

def check_detected(
self,
uav: "UAV",
victim: "BaseVictim",
obstacles: Optional[list] = None,
) -> bool:
"""
Kiểm tra UAV có detect được victim không.

    Pipeline:
        1. FOV geometric check  → fast reject
        2. LOS check            → fast reject
        3. Noise probability    → stochastic
    """
    fov_r = self.calculate_fov_radius(uav.pos[2])

    # ── 1. FOV check ─────────────────────────────────────────
    if dist_2d(uav.pos, victim.pos) > fov_r:
        return False

    # ── 2. LOS check ─────────────────────────────────────────
    if obstacles and not check_los_2d(uav.pos, victim.pos, obstacles):
        return False

    # ── 3. Compute P(detect) với noise ───────────────────────
    altitude = float(uav.pos[2])

    # Environmental factor (victim trong smoke/fire?)
    env_factor = self._get_env_factor(victim.pos, obstacles or [])

    # UAV speed (motion blur)
    vel       = getattr(uav, "vel", np.zeros(3))
    uav_speed = float(np.linalg.norm(vel[:2]))

    # Victim type factor
    victim_factor = self._get_victim_factor(victim)

    # Final probability
    p = self.calculate_detection_prob(
        altitude      = altitude,
        uav_speed     = uav_speed,
        env_factor    = env_factor,
        victim_factor = victim_factor,
    )

    # ── 4. Stochastic sample ──────────────────────────────────
    return bool(self._rng.random() < p)

# ── Batch scan: Victims ───────────────────────────────────────────────────

def scan_victims(
self,
uav: "UAV",
victims: List["BaseVictim"],
obstacles: Optional[list] = None,
) -> np.ndarray:
"""
Quét victims trong FOV → observation vector.

    Noise ảnh hưởng:
        - check_detected() dùng noise → newly found
        - obs vector vẫn show victims trong FOV
          (UAV "thấy" nhưng chưa chắc "nhận ra")
    """
    fov_r = self.calculate_fov_radius(uav.pos[2])
    if fov_r <= 0:
        return np.zeros(self.victim_obs_dim, dtype=np.float32)

    candidates: List[Tuple[float, "BaseVictim"]] = []
    for v in victims:
        d = dist_2d(uav.pos, v.pos)
        if d > fov_r:
            continue
        if obstacles and not check_los_2d(uav.pos, v.pos, obstacles):
            continue
        candidates.append((d, v))

    candidates.sort(key=lambda x: x[0])

    result = np.zeros(self.victim_obs_dim, dtype=np.float32)
    for i, (d, v) in enumerate(candidates[:self._n_victims]):
        base             = i * self._v_dims
        result[base + 0] = (v.pos[0] - uav.pos[0]) / fov_r
        result[base + 1] = (v.pos[1] - uav.pos[1]) / fov_r
        result[base + 2] = d / fov_r
        result[base + 3] = float(v.urgency) / 5.0
        result[base + 4] = 1.0 if v.is_found else 0.0

    return result

def scan_obstacles(
self,
uav: "UAV",
obstacles: list,
) -> np.ndarray:
"""Quét obstacles - KHÔNG có noise (obstacles là static, dễ thấy)."""
fov_r = self.calculate_fov_radius(uav.pos[2])
if fov_r <= 0:
return np.zeros(self.obstacle_obs_dim, dtype=np.float32)

    from entities.obstacle import Debris, DangerZone

    candidates: List[Tuple[float, object, float]] = []
    for obs in obstacles:
        d          = dist_2d(uav.pos, obs.pos)
        obs_radius = obs._get_fallback_radius()
        if d > fov_r + obs_radius:
            continue
        type_id = 1.0 if isinstance(obs, DangerZone) else 0.0
        candidates.append((d, obs, type_id))

    candidates.sort(key=lambda x: x[0])

    result = np.zeros(self.obstacle_obs_dim, dtype=np.float32)
    for i, (d, obs, tid) in enumerate(candidates[:self._n_obstacles]):
        base             = i * self._o_dims
        result[base + 0] = np.clip(
            (obs.pos[0] - uav.pos[0]) / fov_r, -1.5, 1.5
        )
        result[base + 1] = np.clip(
            (obs.pos[1] - uav.pos[1]) / fov_r, -1.5, 1.5
        )
        result[base + 2] = tid

    return result

from future import annotations

import logging
from typing import TYPE_CHECKING, Dict, List

import numpy as np

from core.coverage_map import CoverageMap
from sensors.fov_sensor import FOVSensor
from sensors.comm_sensor import CommSensor
from utils.geometry import dist_2d

✅ REMOVED: sys.path.append hack
if TYPE_CHECKING:
from entities.uav import UAV
from entities.victim import BaseVictim
from entities.charging_station import ChargingStation
from config import AppConfig

logger = logging.getLogger(name)

"""
observation/obs_builder.py
Observation Builder - Dec-POMDP + CTDE
"""

class ObsResult:
"""Container: actor obs + critic obs."""
slots = ("actor_obs", "critic_obs")

text

def **init**(
self,
actor_obs: Dict[int, np.ndarray],
critic_obs: np.ndarray,
) -> None:
self.actor_obs = actor_obs
self.critic_obs = critic_obs
class ObservationBuilder:
"""
Observation Builder cho SAR UAV Swarm.

text

Actor obs (68 dims) - LOCAL ONLY (Dec-POMDP compliant)
Critic obs (MAX_UAV×68 + 10) - GLOBAL STATE (CTDE)
"""

def **init**(self, coverage_map: CoverageMap, cfg: "AppConfig") -> None:
self.coverage_map = coverage_map
self.cfg = cfg

    self.fov_sensor  = FOVSensor(cfg)
    self.comm_sensor = CommSensor(cfg)

    # Dims từ cfg
    self._self_dim     = cfg.obs.self_dim
    self._station_dim  = cfg.obs.station_dim
    self._team_dim     = cfg.obs.team_dim
    self._obs_dim      = cfg.obs.obstacle_dim
    self._victim_dim   = cfg.obs.victim_dim
    self._coverage_dim = cfg.obs.coverage_dim
    self.actor_dim     = cfg.obs.actor_dim

    # Precompute slices
    dims = [
        self._self_dim,
        self._station_dim,
        self._team_dim,
        self._obs_dim,
        self._victim_dim,
        self._coverage_dim,
    ]
    self.slices: List[slice] = []
    start = 0
    for d in dims:
        self.slices.append(slice(start, start + d))
        start += d

    # Critic dims
    self._max_uav    = cfg.obs.max_uav
    self._global_dim = cfg.obs.global_dim
    self.critic_dim  = cfg.obs.critic_dim

    # Preallocate buffers
    self._actor_bufs: Dict[int, np.ndarray] = {}
    self._critic_buf  = np.zeros(self.critic_dim, dtype=np.float32)

    # Coverage radii
    self._cov_small = cfg.obs.local_cov_small
    self._cov_large = cfg.obs.local_cov_large

    # ✅ FIX 3.3: Debug flag cho sanity assert
    self._debug_obs = getattr(cfg.env, "debug_obs", False)

    logger.debug(
        "ObservationBuilder: actor_dim=%d, critic_dim=%d, max_uav=%d",
        self.actor_dim, self.critic_dim, self._max_uav,
    )

def \_get_actor_buf(self, uav_id: int) -> np.ndarray:
if uav_id not in self.\_actor_bufs:
self.\_actor_bufs[uav_id] = np.zeros(self.actor_dim, dtype=np.float32)
return self.\_actor_bufs[uav_id]

def \_write_self(self, obs: np.ndarray, uav: "UAV") -> None:
s = self.slices[0].start
obs[s + 0] = uav.pos[0] / self.cfg.env.map_size
obs[s + 1] = uav.pos[1] / self.cfg.env.map_size
obs[s + 2] = uav.pos[2] / self.cfg.uav.z_max
obs[s + 3] = uav.vel[0] / self.cfg.uav.max_speed_xy
obs[s + 4] = uav.vel[1] / self.cfg.uav.max_speed_xy
obs[s + 5] = uav.vel[2] / self.cfg.uav.max_speed_z
obs[s + 6] = uav.battery / 100.0
obs[s + 7: s + 11] = uav.get_state_onehot()[:4]

def \_write_stations(
self,
obs: np.ndarray,
uav: "UAV",
stations: List["ChargingStation"],
) -> None:
s = self.slices[1]
obs[s] = 0.0
for i, st in enumerate(stations[:self.cfg.env.n_stations]):
b = s.start + i \* 4
obs[b + 0] = (st.pos[0] - uav.pos[0]) / self.cfg.env.map_size
obs[b + 1] = (st.pos[1] - uav.pos[1]) / self.cfg.env.map_size
obs[b + 2] = dist_2d(uav.pos, st.pos) / self.cfg.map_diagonal
obs[b + 3] = st.get_occupancy_ratio()

def \_write_teammates(
self, obs: np.ndarray, uav: "UAV", all_uavs: List["UAV"]
) -> None:
obs[self.slices[2]] = self.comm_sensor.scan(uav, all_uavs)

def \_write_obstacles(
self, obs: np.ndarray, uav: "UAV", obstacles: list
) -> None:
obs[self.slices[3]] = self.fov_sensor.scan_obstacles(uav, obstacles)

def \_write_victims(
self,
obs: np.ndarray,
uav: "UAV",
victims: List["BaseVictim"],
obstacles: list,
) -> None:
obs[self.slices[4]] = self.fov_sensor.scan_victims(uav, victims, obstacles)

def \_write_coverage(
self, obs: np.ndarray, uav: "UAV", current_step: int
) -> None:
s = self.slices[5].start
obs[s + 0] = self.coverage_map.get_local_coverage(uav.pos, self.\_cov_small)
obs[s + 1] = self.coverage_map.get_local_coverage(uav.pos, self.\_cov_large)
obs[s + 2] = (
max(0, self.cfg.env.max_steps - current_step) / self.cfg.env.max_steps
)

def build_actor_obs(
self,
uav: "UAV",
all_uavs: List["UAV"],
stations: List["ChargingStation"],
victims: List["BaseVictim"],
obstacles: list,
current_step: int,
) -> np.ndarray:
"""Build actor obs cho 1 UAV."""
obs = self.\_get_actor_buf(uav.id)
obs[:] = 0.0

    self._write_self(obs, uav)
    self._write_stations(obs, uav, stations)
    self._write_teammates(obs, uav, all_uavs)
    self._write_obstacles(obs, uav, obstacles)
    self._write_victims(obs, uav, victims, obstacles)
    self._write_coverage(obs, uav, current_step)

    # ✅ FIX 3.3: Sanity check chỉ khi debug mode
    if self._debug_obs:
        assert obs.shape == (self.actor_dim,), \
            f"Shape error: {obs.shape} != ({self.actor_dim},)"
        assert not np.any(np.isnan(obs)), \
            f"NaN in actor obs UAV {uav.id}"
        assert not np.any(np.isinf(obs)), \
            f"Inf in actor obs UAV {uav.id}"

    return obs

def build_all(
self,
all_uavs: List["UAV"],
stations: List["ChargingStation"],
victims: List["BaseVictim"],
obstacles: list,
current_step: int,
) -> ObsResult:
"""Build TẤT CẢ obs trong 1 lần."""
from entities.uav import UAVState

    # ── 1. Actor obs ─────────────────────────────────────────────────────
    actor_obs: Dict[int, np.ndarray] = {}

    for uav in all_uavs:
        if uav.state == UAVState.DISABLED:
            buf      = self._get_actor_buf(uav.id)
            buf[:]   = 0.0
            actor_obs[uav.id] = buf
        else:
            actor_obs[uav.id] = self.build_actor_obs(
                uav, all_uavs, stations, victims, obstacles, current_step
            )

    # ── 2. Critic: UAV part ──────────────────────────────────────────────
    critic   = self._critic_buf
    critic[:] = 0.0

    # ✅ FIX 3.2: Stable ordering → sorted by uav.id
    all_uavs_sorted = sorted(all_uavs, key=lambda u: u.id)

    for i in range(self._max_uav):
        start = i * self.actor_dim
        end   = start + self.actor_dim
        if i < len(all_uavs_sorted):
            uav = all_uavs_sorted[i]
            critic[start:end] = actor_obs[uav.id]
        # else: đã là 0

    # ── 3. Global info (10 dims) ─────────────────────────────────────────
    n  = max(len(all_uavs), 1)
    g  = self._max_uav * self.actor_dim

    bats_live  = [
        u.battery for u in all_uavs
        if u.state != UAVState.DISABLED
    ]
    n_active   = sum(1 for u in all_uavs if u.state == UAVState.ACTIVE)
    n_charging = sum(1 for u in all_uavs if u.state == UAVState.CHARGING)
    n_disabled = sum(1 for u in all_uavs if u.state == UAVState.DISABLED)
    n_alive    = n - n_disabled

    mean_bat = float(np.mean(bats_live)) / 100.0 if bats_live else 0.0
    std_bat  = float(np.std(bats_live))  / 100.0 if bats_live else 0.0
    min_bat  = float(np.min(bats_live))  / 100.0 if bats_live else 0.0

    global_cov    = self.coverage_map.get_coverage_rate()
    n_victims     = max(len(victims), 1)
    victims_found = sum(1 for v in victims if v.is_found)
    time_rem      = max(0, self.cfg.env.max_steps - current_step)

    critic[g + 0] = n_active   / n
    critic[g + 1] = n_charging / n
    critic[g + 2] = n_disabled / n
    critic[g + 3] = n_alive    / n
    critic[g + 4] = mean_bat
    critic[g + 5] = std_bat
    critic[g + 6] = min_bat
    critic[g + 7] = global_cov
    critic[g + 8] = victims_found / n_victims
    critic[g + 9] = time_rem / self.cfg.env.max_steps

    return ObsResult(actor_obs=actor_obs, critic_obs=critic)

from future import annotations

import logging
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
from config import AppConfig

logger = logging.getLogger(name)

"""
entities/charging_station.py
Trạm sạc cho UAV trong hệ thống SAR
"""

class ChargingStation:
"""
Trạm sạc UAV với nhiều cổng sạc.

text

Đặc điểm: - Capacity = 2 UAVs cùng lúc (từ cfg) - UAV PHẢI HẠ THẤP (z ≤ 0.5m) mới sạc được - Auto release khi UAV ra khỏi range hoặc pin đầy - Trạm KHÔNG BỊ HƯ (simplified)

Args:
station_id: ID định danh
pos: Vị trí trạm [x, y] hoặc [x, y, z]
cfg: AppConfig object

Usage:
station = ChargingStation(0, [10.0, 20.0], cfg)
station.charge(uav) # sạc pin
station.release(uav) # giải phóng slot
"""

def **init**(
self,
station_id: int,
pos: list,
cfg: "AppConfig",
) -> None:
self.id = station_id
self.pos = np.array([pos[0], pos[1], 0.0], dtype=np.float64)
self.cfg = cfg

    # Lấy params từ cfg (không hardcode)
    self.capacity      = cfg.env.station_capacity  # 2
    self.charge_radius = cfg.env.charge_radius     # 3.0m
    self.charge_rate   = cfg.uav.charge_rate       # 1.5%/step

    # Occupant tracking
    self.current_occupants: List = []
    self.occupant_ids: set = set()  # O(1) lookup

    logger.debug(
        "ChargingStation %d khởi tạo tại (%.1f, %.1f), capacity=%d",
        self.id, self.pos[0], self.pos[1], self.capacity,
    )

# ─── Query methods ────────────────────────────────────────────────────────

def is_full(self) -> bool:
"""Trạm đã đầy chưa."""
return len(self.current_occupants) >= self.capacity

def is_occupied(self) -> bool:
"""Có ít nhất 1 UAV đang sạc."""
return len(self.current_occupants) > 0

def is_available(self) -> bool:
"""Trạm còn chỗ trống."""
return len(self.current_occupants) < self.capacity

def get_occupancy(self) -> int:
"""Số UAV đang sạc."""
return len(self.current_occupants)

def get_occupancy_ratio(self) -> float:
"""
Tỉ lệ lấp đầy.

    Returns:
        float: [0.0, 1.0]
    """
    return len(self.current_occupants) / self.capacity

def has_uav(self, uav) -> bool:
"""
Kiểm tra UAV có đang sạc ở trạm này không.

    Args:
        uav: UAV object

    Returns:
        bool: True nếu UAV đang ở trạm
    """
    return uav.id in self.occupant_ids

def in_range(self, uav_pos: np.ndarray) -> bool:
"""
Kiểm tra UAV có trong vùng sạc không.

    Điều kiện:
        - Khoảng cách XY ≤ charge_radius
        - Độ cao z ≤ 0.5m (UAV phải hạ xuống gần mặt đất)

    Args:
        uav_pos: np.ndarray [x, y, z]

    Returns:
        bool: True nếu trong vùng VÀ đủ thấp
    """
    dx = float(uav_pos[0]) - self.pos[0]
    dy = float(uav_pos[1]) - self.pos[1]
    dist_xy = np.sqrt(dx**2 + dy**2)

    is_close_enough = dist_xy <= self.charge_radius
    is_low_enough   = float(uav_pos[2]) <= 0.5

    return is_close_enough and is_low_enough

# ─── Action methods ───────────────────────────────────────────────────────

def try_occupy(self, uav) -> bool:
"""
UAV cố gắng chiếm 1 slot sạc.

    Args:
        uav: UAV object

    Returns:
        bool: True nếu thành công (kể cả đã occupy rồi)
              False nếu trạm đầy
    """
    # UAV đã đang sạc ở đây → OK
    if uav.id in self.occupant_ids:
        return True

    if self.is_full():
        return False

    self.current_occupants.append(uav)
    self.occupant_ids.add(uav.id)
    return True

def release(self, uav) -> bool:
"""
UAV rời khỏi trạm sạc.

    Args:
        uav: UAV object

    Returns:
        bool: True nếu release thành công
              False nếu UAV không ở đây
    """
    if uav.id not in self.occupant_ids:
        return False

    self.current_occupants = [
        u for u in self.current_occupants if u.id != uav.id
    ]
    self.occupant_ids.discard(uav.id)
    return True

def charge(self, uav) -> float:
"""
Sạc pin cho UAV 1 step.

    Logic:
        1. Out of range → auto release, return 0
        2. Pin đầy → auto release, return 0
        3. Occupy slot (nếu chưa)
        4. Sạc min(charge_rate, 100 - battery)

    Args:
        uav: UAV object cần sạc

    Returns:
        float: Lượng pin đã sạc thêm (%)
               0.0 nếu không sạc được
    """
    # [1] Out of range → auto release (tránh ghost occupancy)
    if not self.in_range(uav.pos):
        self.release(uav)
        return 0.0

    # [2] Pin đầy → auto release
    if uav.battery >= 100.0:
        self.release(uav)
        return 0.0

    # [3] Thử chiếm slot
    if not self.try_occupy(uav):
        return 0.0

    # [4] Sạc pin
    charge_amount = min(self.charge_rate, 100.0 - uav.battery)
    uav.battery  += charge_amount

    # Auto release khi vừa đầy
    if uav.battery >= 100.0:
        uav.battery = 100.0
        self.release(uav)

    return charge_amount

def force_release_all(self) -> None:
"""
Giải phóng tất cả slots.
Dùng khi reset episode.
"""
self.current_occupants.clear()
self.occupant_ids.clear()

# ─── Info methods ─────────────────────────────────────────────────────────

def get_occupant_ids(self) -> List[int]:
"""Danh sách ID của UAV đang sạc."""
return list(self.occupant_ids)

def to_dict(self) -> dict:
"""
Chuyển thành dict JSON-safe.
Dùng cho logging và visualization.
"""
return {
"id": int(self.id),
"pos": self.pos.tolist(),
"capacity": int(self.capacity),
"charge_radius": float(self.charge_radius),
"charge_rate": float(self.charge_rate),
"occupancy": int(self.get_occupancy()),
"occupancy_ratio": float(self.get_occupancy_ratio()),
"occupant_ids": self.get_occupant_ids(),
"is_full": bool(self.is_full()),
"is_available": bool(self.is_available()),
}

def **repr**(self) -> str:
return (
f"ChargingStation("
f"id={self.id}, "
f"pos=({self.pos[0]:.1f}, {self.pos[1]:.1f}), "
f"slots={self.get_occupancy()}/{self.capacity})"
)
from future import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple
import numpy as np

from utils.geometry import dist_2d, \_line_intersects_circle

if TYPE_CHECKING:
from config import AppConfig

logger = logging.getLogger(name)

─── Shapely import (lazy) ────────────────────────────────────────────────────
try:
from shapely.geometry import Point, LineString, Polygon as ShapelyPolygon
from shapely.geometry import box as shapely_box
from shapely.ops import nearest_points
SHAPELY_AVAILABLE = True
except ImportError:
SHAPELY_AVAILABLE = False
Point = None
LineString = None
ShapelyPolygon = None
shapely_box = None
nearest_points = None

Loại danger chắn LOS
\_LOS_BLOCKING_TYPES = {"fire", "smoke"}

═══════════════════════════════════════════════════════════════════════════
DEBRIS - Hỗ trợ nhiều hình dạng
═══════════════════════════════════════════════════════════════════════════
class Debris:
"""
Mảnh vỡ tòa nhà sập - HỖ TRỢ NHIỀU HÌNH DẠNG.

text

Hình dạng: - circle: Hình tròn (center + radius) - rectangle: Hình chữ nhật (center + width + height + rotation) - polygon: Hình đa giác bất kỳ (danh sách vertices)

UAV va chạm khi: - Trong vùng XY của debris - uav.z < height_3d

Args:
debris_id: ID định danh
pos: Vị trí tâm [x, y] hoặc [x, y, z]
height_3d: Độ cao debris (mét)
cfg: AppConfig object

    # Shape params (chọn 1 trong 3)
    shape:     "circle" | "rectangle" | "polygon"
    radius:    Bán kính (chỉ dùng với circle)
    width:     Chiều rộng (chỉ dùng với rectangle)
    height_2d: Chiều dài (chỉ dùng với rectangle)
    rotation:  Góc xoay theo độ (chỉ dùng với rectangle)
    vertices:  List vertices [[x1,y1], [x2,y2], ...] (chỉ dùng với polygon)

"""

def **init**(
self,
debris_id: int,
pos: List[float],
height_3d: float,
cfg: "AppConfig", # Shape params
shape: str = "circle",
radius: Optional[float] = None,
width: Optional[float] = None,
height_2d: Optional[float] = None,
rotation: float = 0.0,
vertices: Optional[List[List[float]]] = None,
) -> None:
self.id = debris_id
self.pos = np.array([pos[0], pos[1], 0.0], dtype=np.float64)
self.height_3d = float(height_3d)
self.cfg = cfg
self.penalty = cfg.reward.r_collision_obstacle # -50.0

    self.shape     = shape
    self.radius    = None
    self.width     = None
    self.height_2d = None
    self.rotation  = rotation
    self.vertices  = None
    self.polygon   = None  # Shapely polygon object

    # ── Validate shape params ──
    if shape == "circle":
        if radius is None:
            raise ValueError("Circle debris requires 'radius' parameter")
        self.radius = float(radius)
        if SHAPELY_AVAILABLE:
            self.polygon = Point(self.pos[:2]).buffer(self.radius)

    elif shape == "rectangle":
        if width is None or height_2d is None:
            raise ValueError("Rectangle debris requires 'width' and 'height_2d' parameters")
        self.width     = float(width)
        self.height_2d = float(height_2d)
        self.rotation  = float(rotation)
        if SHAPELY_AVAILABLE:
            # Create rotated rectangle
            self.polygon = self._create_rotated_box()

    elif shape == "polygon":
        if vertices is None or len(vertices) < 3:
            raise ValueError("Polygon debris requires 'vertices' with >= 3 points")
        self.vertices = [np.array(v[:2]) for v in vertices]
        if SHAPELY_AVAILABLE:
            self.polygon = ShapelyPolygon(self.vertices)

    else:
        raise ValueError(f"Unknown debris shape: {shape}")

    logger.debug(
        "Debris %d (%s) at (%.1f, %.1f), h=%.1f",
        self.id, self.shape, self.pos[0], self.pos[1], self.height_3d,
    )

# ─── Helper: Tạo rotated rectangle ───────────────────────────────────────

def \_create_rotated_box(self) -> "ShapelyPolygon":
"""Tạo hình chữ nhật xoay bằng Shapely."""
if not SHAPELY_AVAILABLE:
return None

    from shapely import affinity

    # Tạo box không xoay (centered at origin)
    box = shapely_box(
        -self.width / 2, -self.height_2d / 2,
        self.width / 2, self.height_2d / 2,
    )

    # Xoay quanh origin
    if self.rotation != 0:
        box = affinity.rotate(box, self.rotation, origin=(0, 0))

    # Dịch chuyển đến vị trí thực
    box = affinity.translate(box, self.pos[0], self.pos[1])

    return box

# ─── Geometry checks ──────────────────────────────────────────────────────

def in_zone_2d(self, pos_2d: np.ndarray) -> bool:
"""
Kiểm tra vị trí 2D có trong vùng debris không.

    Args:
        pos_2d: [x, y] hoặc [x, y, z]

    Returns:
        bool: True nếu trong vùng
    """
    point = np.array(pos_2d[:2])

    # ── Circle: Dùng distance check (fast) ──
    if self.shape == "circle":
        return dist_2d(point, self.pos) <= self.radius

    # ── Rectangle/Polygon: Dùng Shapely ──
    # ✅ FIX 2.2: covers() thay vì contains() (include boundary)
    if SHAPELY_AVAILABLE and self.polygon is not None:
        return self.polygon.covers(Point(point))

    # ── Fallback: Sử dụng radius tương đương ──
    fallback_radius = self._get_fallback_radius()
    return dist_2d(point, self.pos) <= fallback_radius

def causes_collision(self, uav_pos: np.ndarray) -> bool:
"""
UAV có va chạm với debris không.

    Args:
        uav_pos: [x, y, z]

    Returns:
        bool: True nếu va chạm
    """
    if uav_pos[2] >= self.height_3d:
        return False
    return self.in_zone_2d(uav_pos)

def blocks_los(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
"""
Debris có chắn line-of-sight không.

    Args:
        pos1: [x, y, z] điểm bắt đầu
        pos2: [x, y, z] điểm kết thúc

    Returns:
        bool: True nếu bị chắn
    """
    # [1] 3D altitude check
    if min(pos1[2], pos2[2]) >= self.height_3d:
        return False

    p1 = np.array(pos1[:2])
    p2 = np.array(pos2[:2])

    # [2] XY intersection check
    if self.shape == "circle":
        return _line_intersects_circle(p1, p2, self.pos[:2], self.radius)

    # Rectangle/Polygon: Dùng Shapely
    if SHAPELY_AVAILABLE and self.polygon is not None:
        line = LineString([p1, p2])
        return line.intersects(self.polygon)

    # Fallback
    fallback_radius = self._get_fallback_radius()
    return _line_intersects_circle(p1, p2, self.pos[:2], fallback_radius)

def get_distance_to_edge(self, pos_2d: np.ndarray) -> float:
"""
Khoảng cách từ pos đến cạnh gần nhất của debris.

    Args:
        pos_2d: [x, y] hoặc [x, y, z]

    Returns:
        float: khoảng cách (0 nếu đang trong debris)
    """
    point = Point(pos_2d[:2])

    if self.shape == "circle":
        return max(0.0, dist_2d(pos_2d, self.pos) - self.radius)

    if SHAPELY_AVAILABLE and self.polygon is not None:
        if self.polygon.contains(point):
            return 0.0
        near_pt = nearest_points(self.polygon.boundary, point)[0]
        return point.distance(near_pt)

    # Fallback
    fallback_radius = self._get_fallback_radius()
    return max(0.0, dist_2d(pos_2d, self.pos) - fallback_radius)

# ─── Helper: Fallback radius ──────────────────────────────────────────────

def \_get_fallback_radius(self) -> float:
"""Tính bán kính tương đương cho fallback (khi không có Shapely)."""
if self.shape == "circle":
return self.radius
elif self.shape == "rectangle": # Bounding circle radius = sqrt(w²+h²)/2
return np.sqrt(self.width**2 + self.height_2d**2) / 2
elif self.shape == "polygon": # Max distance từ center đến vertices
dists = [dist_2d(self.pos, v) for v in self.vertices]
return max(dists) if dists else 1.0
return 1.0

# ─── Serialization ────────────────────────────────────────────────────────

def to_dict(self) -> dict:
"""Chuyển thành dict JSON-safe."""
data = {
"id": int(self.id),
"type": "debris",
"pos": self.pos.tolist(),
"height_3d": float(self.height_3d),
"penalty": float(self.penalty),
"shape": self.shape,
}

    if self.shape == "circle":
        data["radius"] = float(self.radius)
    elif self.shape == "rectangle":
        data["width"]     = float(self.width)
        data["height_2d"] = float(self.height_2d)
        data["rotation"]  = float(self.rotation)
    elif self.shape == "polygon":
        data["vertices"] = [v.tolist() for v in self.vertices]

    return data

def **repr**(self) -> str:
if self.shape == "circle":
return (
f"Debris(id={self.id}, circle, "
f"pos=({self.pos[0]:.1f}, {self.pos[1]:.1f}), "
f"r={self.radius:.1f}, h={self.height_3d:.1f})"
)
elif self.shape == "rectangle":
return (
f"Debris(id={self.id}, rect, "
f"pos=({self.pos[0]:.1f}, {self.pos[1]:.1f}), "
f"w={self.width:.1f}, h={self.height_2d:.1f}, "
f"rot={self.rotation:.0f}°, h3d={self.height_3d:.1f})"
)
else:
return (
f"Debris(id={self.id}, polygon, "
f"pos=({self.pos[0]:.1f}, {self.pos[1]:.1f}), "
f"{len(self.vertices)} vertices, h={self.height_3d:.1f})"
)
═══════════════════════════════════════════════════════════════════════════
DANGERZONE - Giữ nguyên, chỉ thêm shape support (tương tự Debris)
═══════════════════════════════════════════════════════════════════════════
class DangerZone:
"""
Vùng nguy hiểm - HỖ TRỢ NHIỀU HÌNH DẠNG.

text

Shape params giống Debris.
"""

def **init**(
self,
zone_id: int,
pos: List[float],
danger_type: str,
cfg: "AppConfig", # Shape params
shape: str = "circle",
radius: Optional[float] = None,
width: Optional[float] = None,
height_2d: Optional[float] = None,
rotation: float = 0.0,
vertices: Optional[List[List[float]]] = None,
) -> None:
self.id = zone_id
self.pos = np.array([pos[0], pos[1], 0.0], dtype=np.float64)
self.danger_type = danger_type
self.cfg = cfg

    self.max_height = cfg.danger.heights.get(danger_type, 15.0)
    self.penalty    = cfg.danger.penalties.get(danger_type, -12.0)

    self.shape      = shape
    self.radius     = None
    self.width      = None
    self.height_2d  = None
    self.rotation   = rotation
    self.vertices   = None
    self.polygon    = None

    # ── Validate shape (giống Debris) ──
    if shape == "circle":
        if radius is None:
            raise ValueError("Circle zone requires 'radius'")
        self.radius = float(radius)
        if SHAPELY_AVAILABLE:
            self.polygon = Point(self.pos[:2]).buffer(self.radius)

    elif shape == "rectangle":
        if width is None or height_2d is None:
            raise ValueError("Rectangle zone requires 'width' and 'height_2d'")
        self.width     = float(width)
        self.height_2d = float(height_2d)
        self.rotation  = float(rotation)
        if SHAPELY_AVAILABLE:
            self.polygon = self._create_rotated_box()

    elif shape == "polygon":
        if vertices is None or len(vertices) < 3:
            raise ValueError("Polygon zone requires >= 3 vertices")
        self.vertices = [np.array(v[:2]) for v in vertices]
        if SHAPELY_AVAILABLE:
            self.polygon = ShapelyPolygon(self.vertices)

    else:
        raise ValueError(f"Unknown zone shape: {shape}")

# ── Copy helper methods từ Debris ──

def \_create_rotated_box(self):
"""Tạo rotated rectangle (same as Debris)."""
if not SHAPELY_AVAILABLE:
return None
from shapely import affinity
box = shapely_box(
-self.width / 2, -self.height_2d / 2,
self.width / 2, self.height_2d / 2,
)
if self.rotation != 0:
box = affinity.rotate(box, self.rotation, origin=(0, 0))
box = affinity.translate(box, self.pos[0], self.pos[1])
return box

def \_get_fallback_radius(self) -> float:
"""Fallback radius (same as Debris)."""
if self.shape == "circle":
return self.radius
elif self.shape == "rectangle":
return np.sqrt(self.width**2 + self.height_2d**2) / 2
elif self.shape == "polygon":
dists = [dist_2d(self.pos, v) for v in self.vertices]
return max(dists) if dists else 1.0
return 1.0

# ── Geometry methods (giống Debris) ──

def is_inside(self, uav_pos: np.ndarray) -> bool:
"""Check UAV trong zone không."""
if uav_pos[2] >= self.max_height:
return False

    point = np.array(uav_pos[:2])

    if self.shape == "circle":
        return dist_2d(point, self.pos) <= self.radius

    # ✅ FIX 2.2: covers() thay vì contains()
    if SHAPELY_AVAILABLE and self.polygon is not None:
        return self.polygon.covers(Point(point))

    fallback_radius = self._get_fallback_radius()
    return dist_2d(point, self.pos) <= fallback_radius

def blocks_los(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
"""Check chắn LOS không."""
if self.danger_type not in \_LOS_BLOCKING_TYPES:
return False

    if min(pos1[2], pos2[2]) >= self.max_height:
        return False

    p1 = np.array(pos1[:2])
    p2 = np.array(pos2[:2])

    if self.shape == "circle":
        return _line_intersects_circle(p1, p2, self.pos[:2], self.radius)

    if SHAPELY_AVAILABLE and self.polygon is not None:
        line = LineString([p1, p2])
        return line.intersects(self.polygon)

    fallback_radius = self._get_fallback_radius()
    return _line_intersects_circle(p1, p2, self.pos[:2], fallback_radius)

# ── Sensor/Battery modifiers (giữ nguyên) ──

def get_sensor_modifier(self) -> float:
return 0.4 if self.danger_type == "smoke" else 1.0

def get_battery_modifier(self) -> float:
return 0.05 if self.danger_type == "fire" else 0.0

# ── Serialization ──

def to_dict(self) -> dict:
data = {
"id": int(self.id),
"type": "danger_zone",
"danger_type": self.danger_type,
"pos": self.pos.tolist(),
"max_height": None if np.isinf(self.max_height) else float(self.max_height),
"penalty": float(self.penalty),
"shape": self.shape,
}

    if self.shape == "circle":
        data["radius"] = float(self.radius)
    elif self.shape == "rectangle":
        data["width"]     = float(self.width)
        data["height_2d"] = float(self.height_2d)
        data["rotation"]  = float(self.rotation)
    elif self.shape == "polygon":
        data["vertices"] = [v.tolist() for v in self.vertices]

    return data

def **repr**(self) -> str:
max_h = "inf" if np.isinf(self.max_height) else f"{self.max_height:.1f}"
if self.shape == "circle":
return (
f"DangerZone(id={self.id}, {self.danger_type}, circle, "
f"r={self.radius:.1f}, max_h={max_h})"
)
elif self.shape == "rectangle":
return (
f"DangerZone(id={self.id}, {self.danger_type}, rect, "
f"w={self.width:.1f}×h={self.height_2d:.1f}, "
f"rot={self.rotation:.0f}°, max_h={max_h})"
)
else:
return (
f"DangerZone(id={self.id}, {self.danger_type}, polygon, "
f"{len(self.vertices)} vertices, max_h={max_h})"
)
"""
entities/uav.py
UAV entity cho SAR UAV Swarm - v2.1

Changes vs v2.0:
BUG-36: Add battery_pct property (was missing, caused AttributeError)
CLEAN: Remove sys.path hack (dùng relative import ở caller)
CLEAN: Remove duplicate get_battery_penalty (delegate đến reward fn)
"""
from future import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
from entities.charging_station import ChargingStation
from config import AppConfig

logger = logging.getLogger(name)

class UAVState(Enum):
"""Trạng thái UAV."""
ACTIVE = "active"
RETURNING = "returning"
CHARGING = "charging"
DEPLOYING = "deploying"
DISABLED = "disabled"

class UAV:
"""
UAV trong SAR Swarm.

text

Battery convention:
self.battery → float ∈ [0.0, 100.0] (percentage)
self.battery_pct → same value, explicit name

    Dùng battery_pct trong reward functions để code tự-documenting.

"""

STATE_ORDER = [
UAVState.ACTIVE,
UAVState.RETURNING,
UAVState.CHARGING,
UAVState.DEPLOYING,
UAVState.DISABLED,
]

def **init**(
self,
uav_id: int,
pos: List[float],
cfg: "AppConfig",
battery: float = 100.0,
state: UAVState = UAVState.ACTIVE,
) -> None:
self.id = uav_id
self.pos = np.array(pos, dtype=np.float64)
self.vel = np.zeros(3, dtype=np.float64)
self.battery = float(np.clip(battery, 0.0, 100.0))
self.state = state
self.cfg = cfg

    self.target_station: Optional["ChargingStation"] = None
    self.pybullet_body_id: Optional[int] = None

    # Per-episode tracking
    self.steps_alive:   int   = 0
    self.distance_xy:   float = 0.0
    self.distance_3d:   float = 0.0
    self.victims_found: int   = 0
    self.battery_death: bool  = False
    self._prev_state: UAVState = state

# ─── Battery Properties ───────────────────────────────────────────────────

@property
def battery_pct(self) -> float:
"""
Battery percentage (0.0 → 100.0).

    ✅ FIX BUG-36: Add this property.

    Alias for self.battery with explicit semantic name.
    Used in reward functions for clarity:
        uav.battery_pct <= 5.0  (vs magic number check)
    """
    return self.battery

@battery_pct.setter
def battery_pct(self, value: float) -> None:
"""Allow setting via battery_pct for symmetry."""
self.battery = float(np.clip(value, 0.0, 100.0))

# ─── Action & Movement ────────────────────────────────────────────────────

def apply_action(self, action: np.ndarray) -> None:
"""
Áp dụng action từ policy (chỉ ACTIVE mới nhận).

    Args:
        action: [ax, ay, az] ∈ [-1, 1]³
    """
    if self.state != UAVState.ACTIVE:
        self.vel[:] = 0.0
        return

    action = np.clip(action, -1.0, 1.0)

    vx = action[0] * self.cfg.uav.max_speed_xy
    vy = action[1] * self.cfg.uav.max_speed_xy
    vz = action[2] * self.cfg.uav.max_speed_z

    # Cap diagonal speed XY
    speed_xy = np.sqrt(vx**2 + vy**2)
    if speed_xy > self.cfg.uav.max_speed_xy:
        scale = self.cfg.uav.max_speed_xy / speed_xy
        vx   *= scale
        vy   *= scale

    self.vel = np.array([vx, vy, vz], dtype=np.float64)

    prev_pos  = self.pos.copy()
    self.pos  = self.pos + self.vel * self.cfg.env.dt

    # Clip altitude + boundary
    self.pos[2] = np.clip(self.pos[2], self.cfg.uav.z_min, self.cfg.uav.z_max)
    self.pos[0] = np.clip(self.pos[0], 0.0, self.cfg.env.map_size)
    self.pos[1] = np.clip(self.pos[1], 0.0, self.cfg.env.map_size)

    # Track distance
    delta = self.pos - prev_pos
    self.distance_xy += float(np.sqrt(delta[0]**2 + delta[1]**2))
    self.distance_3d += float(np.linalg.norm(delta))

def auto_navigate(self, target_pos: np.ndarray) -> None:
"""
Tự động bay đến target (RETURNING / DEPLOYING).
No overshoot: step = min(max_speed × dt, dist).

    Args:
        target_pos: [x, y, z]
    """
    target = np.array(target_pos, dtype=np.float64)
    diff   = target - self.pos
    dist   = np.linalg.norm(diff)

    if dist < 0.05:
        self.vel = np.zeros(3, dtype=np.float64)
        return

    direction = diff / dist

    vx = direction[0] * self.cfg.uav.max_speed_xy
    vy = direction[1] * self.cfg.uav.max_speed_xy
    vz = direction[2] * self.cfg.uav.max_speed_z

    # Cap diagonal XY
    speed_xy = np.sqrt(vx**2 + vy**2)
    if speed_xy > self.cfg.uav.max_speed_xy:
        scale = self.cfg.uav.max_speed_xy / speed_xy
        vx   *= scale
        vy   *= scale

    self.vel      = np.array([vx, vy, vz], dtype=np.float64)

    # No-overshoot step
    step      = self.vel * self.cfg.env.dt
    step_dist = np.linalg.norm(step)
    if step_dist > dist:
        step = direction * dist

    prev_pos  = self.pos.copy()
    self.pos  = self.pos + step

    # Boundary clip
    self.pos[0] = np.clip(self.pos[0], 0.0, self.cfg.env.map_size)
    self.pos[1] = np.clip(self.pos[1], 0.0, self.cfg.env.map_size)

    # Altitude clip theo state
    if self.state == UAVState.ACTIVE:
        self.pos[2] = np.clip(self.pos[2], self.cfg.uav.z_min, self.cfg.uav.z_max)
    elif self.state == UAVState.RETURNING:
        self.pos[2] = np.clip(self.pos[2], 0.0, self.cfg.uav.z_max)
    elif self.state == UAVState.CHARGING:
        self.pos[2] = np.clip(self.pos[2], 0.0, 0.5)
    elif self.state == UAVState.DEPLOYING:
        self.pos[2] = np.clip(self.pos[2], 0.0, self.cfg.uav.z_max)

    # Track distance
    delta = self.pos - prev_pos
    self.distance_xy += float(np.sqrt(delta[0]**2 + delta[1]**2))
    self.distance_3d += float(np.linalg.norm(delta))

# ─── Battery ──────────────────────────────────────────────────────────────

def update_battery(self, stations: List["ChargingStation"]) -> None:
"""
Cập nhật pin theo state mỗi step.

    Drain rates (% per SECOND, × dt để ra per-step):
        ACTIVE:    drain_xy × speed_ratio + drain_z + drain_idle
        RETURNING: same as ACTIVE
        DEPLOYING: same as ACTIVE
        CHARGING:  +charge_rate via station
        DISABLED:  skip
    """
    if self.state == UAVState.DISABLED:
        return

    if self.state == UAVState.CHARGING:
        self._do_charge(stations)

    elif self.state in (UAVState.ACTIVE,
                        UAVState.RETURNING,
                        UAVState.DEPLOYING):
        self._do_drain()

    # Clamp
    self.battery = float(np.clip(self.battery, 0.0, 100.0))

    # Terminal: battery dead
    if self.battery <= 0.0 and not self.battery_death:
        self.battery_death = True
        self.state         = UAVState.DISABLED
        self.vel[:]        = 0.0
        logger.debug(f"UAV {self.id}: battery dead → DISABLED")

def \_do_charge(self, stations: List["ChargingStation"]) -> None:
"""Charge via target_station hoặc nearest in-range."""
if self.target_station is not None:
self.target_station.charge(self)
return

    for station in stations:
        if station.in_range(self.pos):
            station.charge(self)
            self.target_station = station
            return

def \_do_drain(self) -> None:
"""
Drain battery theo velocity.

    FIX BUG-30: Drain × dt_seconds (decoupled từ simulation step).
    """
    dt       = self.cfg.env.dt_seconds
    speed_xy = float(np.sqrt(self.vel[0]**2 + self.vel[1]**2))
    vz       = float(self.vel[2])
    vz_up    = max(0.0,  vz)
    vz_down  = max(0.0, -vz)

    max_xy  = self.cfg.uav.max_speed_xy_mps
    max_z   = self.cfg.uav.max_speed_z_mps

    # Proportional drain (0 speed → 0 motion drain, still idle drain)
    drain_xy   = self.cfg.uav.drain_xy_pct_per_s   * (speed_xy / max_xy if max_xy > 0 else 0.0)
    drain_up   = self.cfg.uav.drain_z_up_pct_per_s  * (vz_up   / max_z  if max_z  > 0 else 0.0)
    drain_down = self.cfg.uav.drain_z_down_pct_per_s * (vz_down / max_z  if max_z  > 0 else 0.0)
    drain_idle = self.cfg.uav.drain_idle_pct_per_s

    self.battery -= (drain_xy + drain_up + drain_down + drain_idle) * dt

def get_battery_penalty(self) -> float:
"""
Legacy method: progressive battery penalty.

    NOTE: Prefer _battery_penalty_single() trong baseline_reward.py
    vì nó nhận cfg explicitly → dễ test hơn.

    Kept for backward compatibility với code cũ.
    """
    cfg_r   = self.cfg.reward
    cfg_uav = self.cfg.uav

    if self.battery_pct <= cfg_uav.battery_emergency_pct:
        return cfg_r.r_battery_5
    if self.battery_pct <= cfg_uav.battery_critical_pct:
        return cfg_r.r_battery_10
    if self.battery_pct <= cfg_uav.battery_warning_pct:
        return cfg_r.r_battery_20
    return 0.0

# ─── Sensor ───────────────────────────────────────────────────────────────

def get_fov_radius(self) -> float:
"""FOV radius tại độ cao hiện tại (meters)."""
return float(self.pos[2]) \* self.cfg.sensor.fov_tan

# ─── State helpers ────────────────────────────────────────────────────────

def get_state_onehot(self) -> np.ndarray:
"""One-hot encoding: [ACTIVE, RETURNING, CHARGING, DEPLOYING, DISABLED]."""
onehot = np.zeros(len(self.STATE_ORDER), dtype=np.float32)
for i, s in enumerate(self.STATE_ORDER):
if self.state == s:
onehot[i] = 1.0
break
return onehot

def set_state(self, new_state: UAVState) -> None:
"""Chuyển state với validation."""
if self.state == UAVState.DISABLED:
return # terminal

    # CHARGING → ACTIVE: chỉ khi pin đủ
    if (self.state == UAVState.CHARGING and
            new_state == UAVState.ACTIVE and
            self.battery < self.cfg.uav.battery_ready):
        return

    # ✅ NEW: Track previous state trước khi chuyển
    self._prev_state = self.state
    self.state = new_state

# ─── Convenience predicates ───────────────────────────────────────────────

def is_active(self) -> bool: return self.state == UAVState.ACTIVE
def is_returning(self) -> bool: return self.state == UAVState.RETURNING
def is_charging(self) -> bool: return self.state == UAVState.CHARGING
def is_deploying(self) -> bool: return self.state == UAVState.DEPLOYING
def is_disabled(self) -> bool: return self.state == UAVState.DISABLED
def is_operational(self) -> bool: return self.state != UAVState.DISABLED

def needs_charging(self) -> bool:
"""Battery ≤ return threshold → cần về trạm."""
return self.battery_pct <= self.cfg.uav.battery_return_pct

def is_ready_to_deploy(self) -> bool:
"""Battery ≥ ready threshold → sẵn sàng nhiệm vụ."""
return self.battery_pct >= self.cfg.uav.battery_ready_pct

def find_nearest_station(
self,
stations: List["ChargingStation"],
) -> Optional["ChargingStation"]:
"""Tìm trạm gần nhất còn chỗ (fallback: gần nhất bất kể full)."""
if not stations:
return None

    def _dist(s):
        dx = self.pos[0] - s.pos[0]
        dy = self.pos[1] - s.pos[1]
        return float(np.sqrt(dx**2 + dy**2))

    sorted_s = sorted(stations, key=_dist)

    for s in sorted_s:
        if s.is_available():
            return s

    return sorted_s[0]  # fallback

# ─── Info ─────────────────────────────────────────────────────────────────

def get_speed_xy(self) -> float:
return float(np.sqrt(self.vel[0]**2 + self.vel[1]**2))

def get_speed_3d(self) -> float:
return float(np.linalg.norm(self.vel))

def to_dict(self) -> dict:
return {
"id": int(self.id),
"pos": self.pos.tolist(),
"vel": self.vel.tolist(),
"battery": float(self.battery),
"battery_pct": float(self.battery_pct),
"state": self.state.value,
"state_onehot": self.get_state_onehot().tolist(),
"fov_radius": float(self.get_fov_radius()),
"speed_xy": float(self.get_speed_xy()),
"steps_alive": int(self.steps_alive),
"distance_xy": float(self.distance_xy),
"distance_3d": float(self.distance_3d),
"victims_found": int(self.victims_found),
"battery_death": bool(self.battery_death),
"needs_charging": bool(self.needs_charging()),
"ready_to_deploy": bool(self.is_ready_to_deploy()),
"target_station": int(self.target_station.id)
if self.target_station else None,
}

def **repr**(self) -> str:
return (
f"UAV(id={self.id}, "
f"state={self.state.value}, "
f"bat={self.battery:.1f}%, "
f"pos=[{self.pos[0]:.1f}, {self.pos[1]:.1f}, {self.pos[2]:.1f}])"
)
from future import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from utils.geometry import dist_2d, check_los_2d

if TYPE_CHECKING:
from config import AppConfig

logger = logging.getLogger(name)

═══════════════════════════════════════════════════════════════════════
BASE CLASS
═══════════════════════════════════════════════════════════════════════
class BaseVictim(ABC):
"""
Abstract base class cho tất cả victim types.

text

FIXES: - update(step) alias cho step() → compatible với logic_backend - mark_found() → subclass override để freeze speed
"""

def **init**(
self,
victim_id: int,
pos: List[float],
urgency: float,
cfg: "AppConfig",
) -> None:
self.id = victim_id
self.pos = np.array([pos[0], pos[1], 0.0], dtype=np.float64)
self.urgency = float(np.clip(urgency, 1.0, 5.0))
self.cfg = cfg

    self.is_found:      bool          = False
    self.found_at_step: Optional[int] = None
    self.found_by_uav:  Optional[int] = None

# ── Abstract ──────────────────────────────────────────────────────────────

@abstractmethod
def step(self, obstacles: List = None) -> None:
"""Update physics mỗi timestep."""
pass

@property
@abstractmethod
def victim_type(self) -> str:
pass

# ── Alias: update() = step() ──────────────────────────────────────────────

# logic_backend gọi v.update(step_count)

# victim.py định nghĩa step(obstacles)

# → Cần bridge cả 2 interface

def update(self, step_count: int = 0, obstacles: List = None) -> None:
"""
Alias cho step() - compatible với logic_backend.

    logic_backend gọi: v.update(self._step_count)
    → obstacles=None (logic_backend không truyền obstacles vào update)
    → Nếu cần obstacle-aware movement, truyền từ backend

    Args:
        step_count: Step hiện tại (không dùng trực tiếp, để compatible)
        obstacles:  Optional obstacle list
    """
    self.step(obstacles)

# ── Detection ─────────────────────────────────────────────────────────────

def \_calc_detection_prob(self, uav_z: float) -> float:
"""P(detect) theo altitude. Chỉ dùng khi KHÔNG có FOVSensor noise."""
p = (
self.cfg.sensor.p_detect_base
_ np.exp(-self.cfg.sensor.p_detect_decay _ uav_z)
)
return float(np.clip(p, 0.0, 1.0))

def is_detected_by(
self,
uav_pos: np.ndarray,
fov_radius: float,
obstacles: List = None,
p_detect: float = None,
) -> bool:
"""
Legacy detection (không có noise model).
FOVSensor.check_detected() là preferred method với noise đầy đủ.
Giữ lại để backward compatibility.
"""
d = dist_2d(uav_pos, self.pos)
if d > fov_radius:
return False

    if obstacles:
        if not check_los_2d(uav_pos, self.pos, obstacles):
            return False

    if p_detect is None:
        p_detect = self._calc_detection_prob(uav_pos[2])

    return bool(np.random.rand() < p_detect)

def mark_found(self, step: int, uav_id: int) -> None:
"""
Đánh dấu found (chỉ 1 lần).
Subclass override để thêm behavior (freeze speed, etc.)
"""
if not self.is_found:
self.is_found = True
self.found_at_step = step
self.found_by_uav = uav_id
self.\_on_found() # ← Hook cho subclass

def \_on_found(self) -> None:
"""Hook gọi sau khi mark_found(). Override trong subclass."""
pass

# ── Info ──────────────────────────────────────────────────────────────────

def get_reward_value(self) -> float:
return self.cfg.reward.r_victim_base \* (self.urgency / 5.0)

def to_dict(self) -> dict:
return {
"id": int(self.id),
"type": self.victim_type,
"pos": self.pos.tolist(),
"urgency": float(self.urgency),
"is_found": bool(self.is_found),
"found_at_step": int(self.found_at_step) if self.found_at_step is not None else None,
"found_by_uav": int(self.found_by_uav) if self.found_by_uav is not None else None,
}

def **repr**(self) -> str:
status = "FOUND" if self.is_found else "MISSING"
return (
f"{self.**class**.**name**}("
f"id={self.id}, urgency={self.urgency:.1f}, "
f"pos=({self.pos[0]:.1f},{self.pos[1]:.1f}), "
f"status={status})"
)
═══════════════════════════════════════════════════════════════════════
INJURED VICTIM
═══════════════════════════════════════════════════════════════════════
class InjuredVictim(BaseVictim):
"""
Nạn nhân bị thương nặng - KHÔNG DI CHUYỂN.
Urgency cao [4, 5]. Detection factor = 1.15 (dễ thấy vì nằm im).
"""

text

def **init**(
self,
victim_id: int,
pos: List[float],
cfg: "AppConfig",
urgency: float = None,
) -> None:
if urgency is None:
urgency = np.random.uniform(
cfg.victim.injured_urgency_min,
cfg.victim.injured_urgency_max,
)
super().**init**(victim_id, pos, urgency, cfg)

    # Speed=0 cho FOVSensor._get_victim_factor()
    self.speed = 0.0

@property
def victim_type(self) -> str:
return "injured"

def step(self, obstacles: List = None) -> None:
"""InjuredVictim không di chuyển."""
pass

def \_on_found(self) -> None:
"""Đã found → không cần làm gì thêm (đã đứng im)."""
pass
═══════════════════════════════════════════════════════════════════════
MOBILE VICTIM
═══════════════════════════════════════════════════════════════════════
class MobileVictim(BaseVictim):
"""
Nạn nhân còn di chuyển - RANDOM WALK.
Urgency thấp [1, 3]. Detection factor tùy speed.

text

FIXES: - \_on_found(): freeze speed = 0.0 ngay lập tức - step(): check is_found trước khi move - update() nhận obstacles từ backend nếu có
"""

\_BOUNDARY_MARGIN: float = 2.0

def **init**(
self,
victim_id: int,
pos: List[float],
cfg: "AppConfig",
urgency: float = None,
) -> None:
if urgency is None:
urgency = np.random.uniform(
cfg.victim.mobile_urgency_min,
cfg.victim.mobile_urgency_max,
)
super().**init**(victim_id, pos, urgency, cfg)

    self.speed      = float(np.random.uniform(
        cfg.victim.mobile_speed_min,
        cfg.victim.mobile_speed_max,
    ))
    self.direction  = float(np.random.uniform(0, 2 * np.pi))
    self.move_timer = 0

@property
def victim_type(self) -> str:
return "mobile"

def \_on_found(self) -> None:
"""
FIX BUG-04: Freeze ngay khi found.
speed=0.0 → FOVSensor.\_get_victim_factor() trả về 1.0 (không penalty)
"""
self.speed = 0.0
logger.debug("MobileVictim %d frozen at step %d", self.id, self.found_at_step)

def step(self, obstacles: List = None) -> None:
"""
Random walk mỗi step.

    FIX: Check is_found/speed=0 TRƯỚC khi move.
    """
    # [1] Freeze nếu found hoặc speed=0
    if self.is_found or self.speed <= 0:
        return

    # [2] Timer đổi hướng
    self.move_timer += 1
    if self.move_timer >= self.cfg.victim.mobile_dir_change:
        self.direction  = float(np.random.uniform(0, 2 * np.pi))
        self.move_timer = 0

    # [3] Tính new_pos (dt từ config)
    dt = float(getattr(self.cfg.env, "dt_seconds",
                       getattr(self.cfg.env, "dt", 1.0)))
    dx = self.speed * np.cos(self.direction) * dt
    dy = self.speed * np.sin(self.direction) * dt

    new_pos    = self.pos.copy()
    new_pos[0] += dx
    new_pos[1] += dy

    # [4] Boundary clip
    ms     = float(self.cfg.env.map_size)
    margin = self._BOUNDARY_MARGIN

    clipped_x = float(np.clip(new_pos[0], margin, ms - margin))
    clipped_y = float(np.clip(new_pos[1], margin, ms - margin))

    if clipped_x != new_pos[0] or clipped_y != new_pos[1]:
        self.direction  = float(np.random.uniform(0, 2 * np.pi))
        self.move_timer = 0
        new_pos[0]      = clipped_x
        new_pos[1]      = clipped_y

    # [5] Obstacle check
    if obstacles and self._check_obstacle_block(new_pos, obstacles):
        self.direction  = float(np.random.uniform(0, 2 * np.pi))
        self.move_timer = 0
        return  # Giữ nguyên pos

    self.pos = new_pos

def \_check_obstacle_block(self, new_pos: np.ndarray, obstacles: List) -> bool:
"""Chỉ Debris chặn movement. DangerZone không chặn."""
from entities.obstacle import Debris
for obs in obstacles:
if isinstance(obs, Debris) and obs.in_zone_2d(new_pos):
return True
return False

def to_dict(self) -> dict:
d = super().to_dict()
d.update({
"speed": float(self.speed),
"direction": float(self.direction),
"move_timer": int(self.move_timer),
})
return d
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import time
import json

"""
Hệ thống logging cho SAR UAV Swarm - Research Grade
Đã fix: coverage units, total_victims missing, convergence logic
"""

class EpisodeLogger:
"""
Logger cho một episode - CHỈ LƯU DATA, KHÔNG IN
Research-ready với đầy đủ metrics cần thiết
"""

text

def **init**(self, episode_id: int, seed: Optional[int] = None):
"""
Tham số:
episode_id: ID của episode
seed: Random seed (để track khi chạy nhiều seeds)
"""
self.episode_id = episode_id
self.seed = seed
self.start_time = time.time()

    # Metrics chính (4 metrics quan trọng nhất)
    self.total_reward = 0.0
    self.coverage_rate = 0.0  # Lưu dưới dạng [0, 1], sẽ convert sang % khi finalize
    self.victims_found = 0
    self.total_victims = 0

    # Episode length (QUAN TRỌNG cho research)
    self.episode_length = 0
    self.collision_events: List[Dict] = []
    self.events: Dict[str, int] = {}

    # ✅ NEW: Landing tracking
    self.landing_events: List[Dict] = []  # List of landing events
    self.total_landings = 0
    self.total_charge_time = 0  # Steps spent charging
    self.per_uav_landings: Dict[int, int] = {}  # UAV ID → count
    # Safety metrics (phân loại chi tiết)
    self.collision_obstacle = 0  # Va chạm debris
    self.collision_uav = 0        # Va chạm UAV khác
    self.collision_proximity = 0  # Proximity warning
    self.battery_deaths = 0
    self.danger_zone_entries = 0

    # Fleet metrics
    self.hot_swaps = 0

def log_step(self, rewards: Dict[str, float], coverage: float):
"""
Log step - CHỈ CẬP NHẬT DATA

    Tham số:
        rewards: Dict {agent_id: reward}
        coverage: Coverage rate hiện tại [0, 1]
    """
    self.total_reward += sum(rewards.values())

    # Coverage: lưu giá trị MAX (tránh noise), giữ [0,1]
    self.coverage_rate = max(self.coverage_rate, coverage)

    # Tăng episode length
    self.episode_length += 1

def log_event(self, event_type: str, \*\*kwargs):
"""
Log event với phân loại chi tiết

    Tham số:
        event_type: Loại event
        **kwargs: Thông tin bổ sung
    """
    if event_type == 'collision_obstacle':
        self.collision_obstacle += 1

    elif event_type == 'collision_uav':
        self.collision_uav += 1

    elif event_type == 'collision_proximity':
        self.collision_proximity += 1

    elif event_type == 'victim_found':
        self.victims_found += 1

    elif event_type == 'battery_death':
        self.battery_deaths += 1

    elif event_type == 'danger_zone':
        self.danger_zone_entries += 1

    elif event_type == 'hot_swap':
        self.hot_swaps += 1

def set_total_victims(self, n: int):
"""Đặt tổng số victims trong episode"""
self.total_victims = n

def log_landing(self, uav_id: int, step: int, battery_before: float, battery_after: float):
"""Log một landing event"""
self.landing_events.append({
"uav_id": uav_id,
"step": step,
"battery_before": battery_before,
"battery_after": battery_after,
"charge_amount": battery_after - battery_before,
})
self.total_landings += 1
self.per_uav_landings[uav_id] = self.per_uav_landings.get(uav_id, 0) + 1

def log_charging_step(self, uav_id: int):
"""Log một step CHARGING state"""
self.total_charge_time += 1
def log_collision(self, uav_id: int, step: int, obstacle_info: dict):
self.collision_events.append({
"step": step,
"uav_id": uav_id,
"obstacle_id": obstacle_info.get("id"),
"obstacle_type": obstacle_info.get("type"),
"position": obstacle_info.get("pos"),
"height": obstacle_info.get("height"),
})

def finalize(self) -> Dict[str, Any]:
"""
Hoàn tất episode, trả về metrics
ĐẢM BẢO tất cả values là Python native types (không phải numpy)

    FIX:
    - Tách rõ coverage_ratio [0,1] và coverage_percent [0,100]
    - Success dùng đúng ngưỡng 0.9 với coverage_ratio
    - Thêm total_victims vào metrics

    Trả về:
        Dict chứa tất cả metrics (JSON-safe)
    """
    duration = time.time() - self.start_time

    # Tách rõ coverage units (FIX LỖI 1)
    coverage_ratio = float(self.coverage_rate)  # [0, 1]
    coverage_percent = coverage_ratio * 100.0   # [0, 100]

    # Tính toán metrics
    victim_found_rate = (self.victims_found / max(1, self.total_victims)) * 100
    total_collisions = (self.collision_obstacle +
                       self.collision_uav +
                       self.collision_proximity)

    # ĐẢM BẢO tất cả là float/int, KHÔNG phải numpy types
    metrics = {
        # Episode info
        'episode_id': int(self.episode_id),
        'seed': int(self.seed) if self.seed is not None else None,
        'duration': float(duration),
        'episode_length': int(self.episode_length),

        # Performance metrics (CAST FLOAT)
        'total_reward': float(self.total_reward),
        'avg_reward_per_step': float(self.total_reward / max(1, self.episode_length)),
        'coverage_rate': float(coverage_percent),  # Store as percent [0, 100]
        'victims_found': int(self.victims_found),
        'total_victims': int(self.total_victims),  # FIX LỖI 2: Thêm field này
        'victims_found_rate': float(victim_found_rate),

        # Safety metrics (phân loại chi tiết)
        'collision_obstacle': int(self.collision_obstacle),
        'collision_uav': int(self.collision_uav),
        'collision_proximity': int(self.collision_proximity),
        'total_collisions': int(total_collisions),
        'battery_deaths': int(self.battery_deaths),
        'danger_zone_entries': int(self.danger_zone_entries),

        # Fleet metrics
        'hot_swaps': int(self.hot_swaps),

         # ✅ Landing stats (phải có)
        'total_landings': int(self.total_landings),
        'total_charge_time': int(self.total_charge_time),
        'avg_charge_per_landing': float(
            self.total_charge_time / max(self.total_landings, 1)
        ),
        'landings_per_uav': dict(self.per_uav_landings),

        # Success criteria (FIX LỖI 1: Dùng đúng coverage_ratio >= 0.9)
        'success': bool(coverage_ratio >= 0.9),
    }

    return metrics

class TrainingLogger:
"""
Logger CHÍNH cho training - Research Grade
Hỗ trợ multi-seed, convergence tracking, và phân tích chi tiết
"""

text

def **init**(self, verbose: int = 1, window_size: int = 100):
"""
Tham số:
verbose:
0 = Im lặng (chỉ lưu file)
1 = Cơ bản (mỗi episode in 1 dòng)
2 = Chi tiết (mỗi 100 episodes in summary)
window_size: Kích thước window cho moving average
"""
self.verbose = verbose
self.window_size = window_size

    # Lưu trữ TẤT CẢ episodes
    self.all_metrics = []

    # Moving windows cho monitoring
    self.recent_rewards = deque(maxlen=window_size)
    self.recent_coverage = deque(maxlen=window_size)
    self.recent_success = deque(maxlen=window_size)
    self.recent_episode_lengths = deque(maxlen=window_size)

    # Convergence tracking (FIX LỖI 3: Dùng relative threshold)
    self.converged = False
    self.convergence_episode = None
    self.convergence_std_threshold = 0.05  # 5% của mean reward

def log_episode(self, metrics: Dict[str, Any]):
"""
Log episode - TỰ ĐỘNG QUYẾT ĐỊNH IN GÌ

    Tham số:
        metrics: Dict từ EpisodeLogger.finalize()
    """
    self.all_metrics.append(metrics)

    # Update moving windows
    self.recent_rewards.append(metrics['total_reward'])
    self.recent_coverage.append(metrics['coverage_rate'])
    self.recent_success.append(1 if metrics['success'] else 0)
    self.recent_episode_lengths.append(metrics['episode_length'])

    ep_id = metrics['episode_id']

    # Check convergence (chỉ sau 100 episodes đầu)
    if not self.converged and len(self.recent_rewards) == self.window_size:
        self._check_convergence(ep_id)

    # LEVEL 1: Mỗi episode - 1 DÒNG
    if self.verbose >= 1:
        self._print_episode_line(metrics)

    # LEVEL 2: Mỗi 100 episodes - SUMMARY
    if self.verbose >= 2 and (ep_id + 1) % 100 == 0:
        self._print_summary(last_n=100)

def \_print_episode_line(self, metrics: Dict[str, Any]):
"""In một dòng cho episode (compact)"""
success_icon = "✅" if metrics['success'] else "❌"

    # Thêm icon convergence nếu đã converge
    conv_icon = "🎯" if self.converged else ""

    print(f"Ep {metrics['episode_id']:4d} | "
          f"R: {metrics['total_reward']:6.1f} | "
          f"Cov: {metrics['coverage_rate']:5.1f}% | "
          f"Vic: {metrics['victims_found']:2d}/{metrics['total_victims']} | "  # FIX: Không cần .get()
          f"Len: {metrics['episode_length']:3d} | "
          f"{success_icon}{conv_icon}")

def \_check_convergence(self, episode: int):
"""Check convergence với threshold phù hợp cho reward âm."""
if len(self.recent_rewards) < self.window_size:
return

    mean_reward = np.mean(self.recent_rewards)
    std_reward = np.std(self.recent_rewards)
    success_rate = np.mean(self.recent_success)

    # FIX: Dùng absolute std nếu mean gần 0 hoặc âm
    if abs(mean_reward) > 10.0:
        relative_std = std_reward / abs(mean_reward)
        threshold = self.convergence_std_threshold  # 5%
    else:
        # Mean reward gần 0 → dùng absolute threshold
        relative_std = std_reward / 10.0  # normalize by reasonable scale
        threshold = 0.5  # absolute threshold

    if relative_std < threshold and success_rate > 0.5:
        self.converged = True
        self.convergence_episode = episode

        if self.verbose >= 1:
            print(f"\n🎯 CONVERGENCE DETECTED at episode {episode}")
            print(f"   Mean reward: {mean_reward:.2f}")
            print(f"   Std: {std_reward:.2f} (relative: {relative_std*100:.1f}%)")
            print(f"   Success rate: {success_rate*100:.1f}%\n")

def \_print_summary(self, last_n: int = 100):
"""In summary ngắn gọn"""
if not self.all_metrics:
return

    recent = self.all_metrics[-last_n:]

    # Tính stats
    rewards = [e['total_reward'] for e in recent]
    coverage = [e['coverage_rate'] for e in recent]
    success = [1 if e['success'] else 0 for e in recent]
    lengths = [e['episode_length'] for e in recent]
    collisions = [e['total_collisions'] for e in recent]

    print(f"\n{'='*70}")
    print(f"SUMMARY - LAST {last_n} EPISODES:")
    print(f"{'='*70}")
    print(f"Reward      : {np.mean(rewards):6.1f} ± {np.std(rewards):5.1f}")
    print(f"Coverage    : {np.mean(coverage):5.1f}% ± {np.std(coverage):4.1f}%")
    print(f"Success Rate: {np.mean(success)*100:5.1f}%")
    print(f"Avg Length  : {np.mean(lengths):5.1f} steps")
    print(f"Collisions  : {np.mean(collisions):5.2f} ± {np.std(collisions):4.2f}")
    print(f"{'='*70}\n")

def get_stats(self, last_n: Optional[int] = None) -> Dict[str, float]:
"""
Lấy stats - KHÔNG IN

    Tham số:
        last_n: Số episodes gần nhất (None = tất cả)

    Trả về:
        Dict chứa statistics
    """
    if not self.all_metrics:
        return {}

    if last_n is None:
        episodes = self.all_metrics
    else:
        episodes = self.all_metrics[-last_n:]

    rewards = [e['total_reward'] for e in episodes]
    coverage = [e['coverage_rate'] for e in episodes]
    success = [1 if e['success'] else 0 for e in episodes]
    lengths = [e['episode_length'] for e in episodes]

    return {
        'n_episodes': len(episodes),
        'reward_mean': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'coverage_mean': float(np.mean(coverage)),
        'coverage_std': float(np.std(coverage)),
        'success_rate': float(np.mean(success) * 100),
        'avg_episode_length': float(np.mean(lengths)),
        'converged': bool(self.converged),
        # FIX LỖI 4: Dùng is not None thay vì truthy check
        'convergence_episode': int(self.convergence_episode) if self.convergence_episode is not None else None,
    }

def get_overall_stats(self) -> Dict[str, float]:
"""
Lấy overall stats cho TOÀN BỘ training
(Dùng cho final report)
"""
return self.get_stats(last_n=None)

def save(self, filepath: str):
"""
Lưu ra file JSON (JSON-safe)

    Tham số:
        filepath: Đường dẫn file
    """
    with open(filepath, 'w') as f:
        json.dump(self.all_metrics, f, indent=2)

    if self.verbose >= 1:
        print(f"✅ Saved {len(self.all_metrics)} episodes to {filepath}")

def load(self, filepath: str):
"""
Load từ file JSON

    Tham số:
        filepath: Đường dẫn file
    """
    with open(filepath, 'r') as f:
        self.all_metrics = json.load(f)

    # Rebuild moving windows từ episodes gần nhất
    recent = self.all_metrics[-self.window_size:]
    for ep in recent:
        self.recent_rewards.append(ep['total_reward'])
        self.recent_coverage.append(ep['coverage_rate'])
        self.recent_success.append(1 if ep['success'] else 0)
        self.recent_episode_lengths.append(ep['episode_length'])

    if self.verbose >= 1:
        print(f"✅ Loaded {len(self.all_metrics)} episodes from {filepath}")

============================================================================
HELPER FUNCTIONS
============================================================================
def compare_training_runs(runs: List[TrainingLogger], labels: List[str]):
"""
So sánh nhiều training runs (cho Phase 2 & 3)

text

Tham số:
runs: Danh sách TrainingLogger objects
labels: Tên của từng run (vd: ["MAPPO", "MASAC", "MATD3"])
"""
print(f"\n{'='*80}")
print(f"TRAINING COMPARISON - FINAL RESULTS")
print(f"{'='*80}")
print(f"{'Algorithm':<15} | {'Reward':<15} | {'Coverage':<15} | {'Success':<10} | {'Converged'}")
print(f"{'-'\*80}")

for run, label in zip(runs, labels):
stats = run.get_overall_stats()

    # FIX: Dùng is not None
    conv_text = f"Ep {stats['convergence_episode']}" if stats['convergence_episode'] is not None else "No"

    print(f"{label:<15} | "
          f"{stats['reward_mean']:6.1f} ± {stats['reward_std']:5.1f} | "
          f"{stats['coverage_mean']:5.1f}% ± {stats['coverage_std']:4.1f}% | "
          f"{stats['success_rate']:5.1f}% | "
          f"{conv_text}")

print(f"{'='\*80}\n")
claude-sonnet-4-5-20250929
Cảm ơn bạn! Tôi đã đọc và hiểu các file này:

✅ ĐÃ TIẾP NHẬN:

1. RewardConfig (config/reward.py)
   16 reward components với rebalanced v3
   Landing reward 3 tiers (tier 3: +120 one-time)
   Battery penalties: -5/-20/-50/-200
   Proximity penalties: -5/-1/-0.2, cap -10
   Terminal bonus cap: 100
   Step penalty cap: -5.0 (quan trọng!)
   Episode clip bounds: [-800, +600] (chỉ logging, không affect learning)
2. FleetManager (core/fleet_manager.py)
   enforce_safety_constraints():
   Battery ≤ 0 → DISABLED
   Battery < 30% AND ACTIVE → RETURNING (hysteresis lock)
   Battery ≥ 80% AND CHARGING → ACTIVE (auto-deploy nếu n_active < n_total-1)
   Episode tracking: \_episode_forced_returns, \_episode_disables
   get_episode_summary() cho logging 1 lần cuối episode
3. LogicBackend (env_setup/backends/logic_backend.py)
   apply_actions(): Landing logic
   land > 0.5 AND state == ACTIVE AND battery ≤ 40% → RETURNING
   Tìm nearest station (KHÔNG giới hạn range) → navigate về z=0
   \_build_uavs(): FIX 4.2 - dùng map_data["uav_spawns"] trực tiếp
   Victim detection: ACTIVE + RETURNING (không phải CHARGING/DEPLOYING/DISABLED)
   FOVSensor seed: deterministic eval hoặc random mỗi episode
4. BaseBackend (abstract interface)
   5 methods: reset(), apply_actions(), step_physics(), step_world(), get_state()
   Temporal correctness: Fleet → Victims → Coverage → Detection
5. SARBaseEnv (env_setup/base_env.py)
   Step execution order (CRITICAL):
   apply_actions() → step_physics() → step_world() → step_count++
   \_check_done() TRƯỚC reward (BUG-ENV-06 fix)
   compute_per_uav() → compute() (global)
   Accumulate \_episode_reward_sum
   \_build_obs_dict() → return
   Landing tracking: Detect ACTIVE/RETURNING → CHARGING transition
   \_check_done() returns: "coverage", "victims", "disabled:battery_death", "disabled:other", None
   Extreme episode logging: reward < -500
6. SARPettingZooEnv (wrapper)
   Thin wrapper: int keys ↔ str keys ("uav_0", "uav_1", ...)
   action_space(agent) = Box(3,) (KHÔNG phải 4, land dim transparent)
   infos['uav_0']['global_obs'] cho MAPPO critic
   All agents terminate cùng lúc
7. VectorizedEnv (env_setup/vec_env.py)
   FIX: Seed progression
   Mỗi worker có RNG riêng: rng = np.random.default_rng(seed)
   Initial reset: seed gốc
   Auto reset sau episode: current_seed = int(rng.integers(0, 2\*\*31))
   Cache last valid obs/global_obs/info (handle PettingZoo edge case)
   Spawn method: mp.get_context("spawn") (CUDA-safe)
   🔍 KEY INSIGHTS ĐÃ NẮM:
   Landing Mechanism:
   Python

# Backend (logic_backend.py)

if land_signal > 0.5 AND state == ACTIVE:
nearest = find_nearest_station() # NO range limit
if nearest:
navigate to (station.pos[0], station.pos[1], 0.0)
target_station = nearest
state = RETURNING

# Physics step

if RETURNING AND in_range(station, 3m, z≤0.5):
try_occupy() → CHARGING
Execution Flow Critical Points:
Done check TRƯỚC reward → tránh terminal reward double-count
Seed progression → mỗi episode seed khác nhau trong vec_env
Landing tracking → log transition ACTIVE/RETURNING → CHARGING
Fleet auto-deploy → battery ≥ 80% + n_active < n_total-1 → ACTIVE
📦 SẴN SÀNG NHẬN THÊM FILE:
Tôi đang chờ các file còn lại (nếu cần):

rewards/baseline_reward.py (implementation 16 components)
observation/obs_builder.py (68 dims slicing)
sensors/fov_sensor.py, comm_sensor.py
entities/uav.py, victim.py, obstacle.py, charging_station.py
utils/logger.py (EpisodeLogger, TrainingLogger)
training/algorithms/mappo/\*.py (networks, actor, critic, buffer, trainer)
train_mappo.py
Hãy gửi file tiếp theo bạn muốn tôi đọc! 🚀

from dataclasses import dataclass

@dataclass
class RewardConfig:
"""
Reward function configuration for SAR task.

text

REBALANCED v3 - Research-grade:
Based on random policy baseline + multi-agent RL stability

    Design principles:
        1. Sparse signals (coverage, victim) > Dense penalties
        2. No reward saturation (clip bounds wide enough)
        3. Step penalty cap to prevent single-step collapse
        4. Multi-agent aware (proximity normalized by pair count)

"""

# ══════════════════════════════════════════════════════════

# 1. COVERAGE REWARD (dense, incremental)

# ══════════════════════════════════════════════════════════

r_coverage_delta: float = 8.0

# Reward per 1% coverage increase

# Max per episode: +600 (100% coverage) - achievable but not trivial

# Expected with random (55%): ~330

# ══════════════════════════════════════════════════════════

# 2. VICTIM DISCOVERY REWARD (sparse, high value)

# ══════════════════════════════════════════════════════════

r_victim_base: float = 30.0 # UNCHANGED

# Base × (urgency / 5.0)

# Range: +10 (urgency=1) to +50 (urgency=5)

# Expected with random (53%, avg urgency=3): ~318

# ══════════════════════════════════════════════════════════

# 3. BATTERY PENALTIES (progressive)

# ══════════════════════════════════════════════════════════

r_battery_20: float = -5.0 # Tăng từ -0.5
r_battery_10: float = -20.0 # Tăng từ -5.0
r_battery_5: float = -50.0 # Tăng từ -20.0
r_battery_dead: float = -200.0 # Tăng từ -100.0

# Rationale:

# - Remove penalty at 20% → encourage exploration

# - Keep strong penalty at <10% → force return behavior

# - Dead penalty prevents catastrophic failure

# ══════════════════════════════════════════════════════════

# 4. COLLISION PENALTY (one-time per obstacle)

# ══════════════════════════════════════════════════════════

r_collision_obstacle: float = -35.0 # UNCHANGED

# ══════════════════════════════════════════════════════════

# 5. DANGER ZONE PENALTY (per step inside zone)

# ══════════════════════════════════════════════════════════

# Applied via DangerZoneConfig.penalties (already rebalanced)

# ══════════════════════════════════════════════════════════

# 6. PROXIMITY PENALTY (multi-threshold, per UAV pair)

# ══════════════════════════════════════════════════════════

# ✅ ĐÚNG — relax thresholds

r_proximity_1m: float = -5.0 # giảm severity
r_proximity_2m: float = -1.0
r_proximity_3m: float = -0.2

proximity_penalty_cap: float = -10.0 # giảm cap xuống

# Prevents proximity spam from dominating signal

# With 6 pairs: worst case -60 → capped at -15

# ══════════════════════════════════════════════════════════

# 7. FLEET MANAGEMENT INCENTIVES (deprecated - set to 0)

# ══════════════════════════════════════════════════════════

r_fleet_deploy: float = 0.0  
r_fleet_recall: float = 0.0

# Rationale: Fleet behavior should emerge from RL, not hand-crafted

# ══════════════════════════════════════════════════════════

# 8. TIME PENALTY (gentle pressure)

# ══════════════════════════════════════════════════════════

r_time_penalty: float = -0.02 # ✅ CHANGED: -0.1 → -0.05

# With 4 UAVs × 300 steps: -60 total (manageable)

# ══════════════════════════════════════════════════════════

# 9. TERMINAL BONUS (mission success)

# ══════════════════════════════════════════════════════════

r_terminal_base: float = 200.0 # UNCHANGED
terminal_bonus_cap: float = 100.0 # ✅ CHANGED: 50 → 100 (more meaningful)

# ══════════════════════════════════════════════════════════

# 10. REWARD CLIPPING & CAPPING

# ══════════════════════════════════════════════════════════

# ✅ CRITICAL: Step penalty cap (applied BEFORE clipping)

# Step penalty cap (applied BEFORE clipping)

step_penalty_cap: float = -5.0

# Step-level clip (WIDENED to prevent saturation)

step_reward_clip_min: float = -100.0
step_reward_clip_max: float = +100.0

# ✅ FIX 3.4: Episode-level clip (FOR LOGGING/ANALYSIS ONLY - does NOT affect learning)

# These bounds are used for:

# - Extreme episode detection logging

# - Result visualization scaling

# - Statistical outlier filtering

# They are NOT enforced during training (would bias gradient)

episode_reward_clip_min: float = -800.0
episode_reward_clip_max: float = +600.0

# ══════════════════════════════════════════════════════════

# 11. DISTANCE SHAPING (sparse → dense bridge)

# ══════════════════════════════════════════════════════════

enable_distance_shaping: bool = True # ✅ NEW: toggleable
distance_shaping_max_per_uav: float = 1.0 # UNCHANGED

# Future: implement distance-delta shaping to avoid local optimum

# (requires state memory)

from future import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from entities.uav import UAV, UAVState

if TYPE_CHECKING:
from config import AppConfig
from entities.charging_station import ChargingStation

✅ FIX: Tắt logger ra console, chỉ lưu file nếu cần
logger = logging.getLogger(name)
logger.addHandler(logging.NullHandler()) # ← Không print ra console

class FleetManager:
def init(self, cfg: "AppConfig") -> None:
self.cfg = cfg
self.n_total: int = 0
self.n_reserve: int = 0
self.all_uavs: List[UAV] = []
self.stations: List["ChargingStation"] = []
self.\_enforced_disables: int = 0
self.\_enforced_returns: int = 0
self.\_suggested_deploys: int = 0
self.\_uav_return_locks: Dict[int, bool] = {}

text

    # ✅ FIX: Tracking cho episode summary (thay vì spam mỗi step)
    self._episode_forced_returns: int = 0
    self._episode_disables: int = 0

def reset(self, all_uavs, stations) -> None:
self.all_uavs = all_uavs
self.stations = stations
self.n_total = len(all_uavs)
self.n_reserve = max(
self.cfg.uav.min_reserve,
int(np.ceil(self.n_total \* self.cfg.uav.reserve_ratio)),
)
self.\_enforced_disables = 0
self.\_enforced_returns = 0
self.\_suggested_deploys = 0
self.\_uav_return_locks = {}

    # ✅ Reset episode counters
    self._episode_forced_returns = 0
    self._episode_disables = 0

def get_deployable_uavs(self) -> List[UAV]:
ready_pool = [
u for u in self.all_uavs
if (u.state == UAVState.CHARGING and
u.battery >= self.cfg.uav.battery_ready_threshold)
]
if not ready_pool:
return []
ready_pool = sorted(ready_pool, key=lambda u: u.battery, reverse=True)
deployable_count = max(0, len(ready_pool) - self.n_reserve)
if deployable_count == 0:
return []
return ready_pool[:deployable_count]

def get_best_deployable(
self,
prefer_station=None,
require_min_battery: float = 80.0,
):
deployable = self.get_deployable_uavs()
if not deployable:
return None
deployable = [u for u in deployable if u.battery >= require_min_battery]
if not deployable:
return None
if prefer_station is not None and prefer_station.is_full:
return None
if prefer_station is None:
return max(deployable, key=lambda u: u.battery)

    def score(uav):
        battery_score = uav.battery / 100.0
        dist = np.linalg.norm(
            np.array(uav.pos[:2]) - np.array(prefer_station.pos[:2])
        )
        dist_score = 1.0 / (1.0 + dist / 10.0)
        return 0.7 * battery_score + 0.3 * dist_score

    return max(deployable, key=score)

def enforce_safety_constraints(self) -> Dict[str, int]:
n_disabled = 0
n_forced_return = 0
n_auto_deploy = 0

    for uav in self.all_uavs:
        # [1] Battery dead → disable
        if uav.battery <= self.cfg.uav.battery_dead_threshold:
            if uav.state != UAVState.DISABLED:
                # ✅ FIX: dùng set_state thay vì mark_disabled()
                uav.state = UAVState.DISABLED
                uav.vel[:] = 0.0
                uav.battery_death = True
                n_disabled += 1
                self._enforced_disables += 1
                self._episode_disables += 1
                logger.debug(f"UAV {uav.id} disabled (battery=0)")
            continue

        # [2] Emergency return (ACTIVE → RETURNING khi battery < 30%)
        if uav.state == UAVState.ACTIVE:
            emergency_threshold = self.cfg.uav.battery_penalty_emergency  # 30%
            resume_threshold    = self.cfg.uav.battery_return_threshold   # 10%
            is_locked = self._uav_return_locks.get(uav.id, False)

            if not is_locked and uav.battery < emergency_threshold:
                target = uav.find_nearest_station(self.stations)
                if target is not None:
                    uav.target_station = target
                    uav.set_state(UAVState.RETURNING)
                    self._uav_return_locks[uav.id] = True
                    n_forced_return += 1
                    self._enforced_returns += 1
                    self._episode_forced_returns += 1
                    logger.debug(
                        f"UAV {uav.id} forced return "
                        f"(battery={uav.battery:.1f}%)"
                    )

            if is_locked and uav.battery > resume_threshold:
                self._uav_return_locks[uav.id] = False

        # ✅ [3] NEW: Auto-deploy khi CHARGING đủ pin
        # [3] Auto-deploy khi CHARGING đủ pin - respect n_reserve
        if uav.state == UAVState.CHARGING:
            if uav.battery >= self.cfg.uav.battery_ready_threshold:
                # ✅ Check: còn đủ UAV active không?
                n_active = sum(
                    1 for u in self.all_uavs
                    if u.state == UAVState.ACTIVE
                )
                n_charging = sum(
                    1 for u in self.all_uavs
                    if u.state == UAVState.CHARGING
                    and u.battery >= self.cfg.uav.battery_ready_threshold
                )

                # Deploy nếu thiếu active UAVs
                # Giữ ít nhất 1 UAV trong reserve (không deploy tất cả)
                if n_active < (self.n_total - 1):
                    # ✅ Release station slot trước khi deploy
                    if uav.target_station is not None:
                        uav.target_station.release(uav)

                    uav.set_state(UAVState.ACTIVE)
                    uav.target_station = None
                    n_auto_deploy += 1
                    logger.debug(
                        f"UAV {uav.id} auto-deployed "
                        f"(battery={uav.battery:.1f}%, "
                        f"n_active={n_active})"
                    )

    return {
        "enforced_disables": n_disabled,
        "enforced_returns":  n_forced_return,
        "auto_deploys":      n_auto_deploy,
        "total_enforced":    n_disabled + n_forced_return,
    }

def suggest_deployments(self, target_active=None):
if target_active is None:
target_active = max(2, int(self.n_total \* 0.3))
n_active = sum(1 for u in self.all_uavs if u.state == UAVState.ACTIVE)
need_deploy = max(0, target_active - n_active)
if need_deploy == 0:
return []
deployable = self.get_deployable_uavs()
if not deployable:
return []
total_station_capacity = sum(
station.capacity - len(station.current_occupants)
for station in self.stations
)
max_can_deploy = min(need_deploy, total_station_capacity, len(deployable))
suggestions = deployable[:max_can_deploy]
self.\_suggested_deploys += len(suggestions)
return suggestions

def suggest_returns(self):
suggestions = []
for uav in self.all_uavs:
if uav.state != UAVState.ACTIVE:
continue
if self.\_uav_return_locks.get(uav.id, False):
continue
if uav.battery < self.cfg.uav.battery_return_threshold:
target = uav.find_nearest_station(self.stations)
if target is not None and not target.is_full:
suggestions.append(uav)
return suggestions

def get_episode_summary(self) -> Dict[str, int]:
"""
✅ NEW: Trả về summary cho episode log
Thay vì spam mỗi step, trainer gọi hàm này 1 lần khi episode done
"""
return {
"forced_returns": self.\_episode_forced_returns,
"disables": self.\_episode_disables,
}

def get_mission_priority_hints(self):
n_active = sum(1 for u in self.all_uavs if u.state == UAVState.ACTIVE)
operational_ratio = n_active / max(self.n_total, 1)
ready_reserve = sum(
1 for u in self.all_uavs
if (u.state == UAVState.CHARGING and
u.battery >= self.cfg.uav.battery_ready_threshold)
)
reserve_health = min(ready_reserve / max(self.n_reserve, 1), 1.0)
congested_stations = sum(1 for s in self.stations if s.is_full)
station_pressure = congested_stations / max(len(self.stations), 1)
return {
"operational_ratio": float(operational_ratio),
"reserve_health": float(reserve_health),
"station_pressure": float(station_pressure),
}

def get_spatial_awareness(self):
active_uavs = [u for u in self.all_uavs if u.state == UAVState.ACTIVE]
charging_uavs = [u for u in self.all_uavs if u.state == UAVState.CHARGING]
active_pos = np.array([u.pos for u in active_uavs]) if active_uavs else np.zeros((0, 3))
charging_pos = np.array([u.pos for u in charging_uavs]) if charging_uavs else np.zeros((0, 3))
if len(active_pos) > 0:
center = np.mean(active_pos, axis=0)
spread = np.mean(np.linalg.norm(active_pos - center, axis=1))
else:
center = np.zeros(3)
spread = 0.0
return {
"active_positions": active_pos,
"charging_positions": charging_pos,
"center_of_mass": center,
"spread_radius": float(spread),
}

def step(self):
safety = self.enforce_safety_constraints()
deploy_suggestions = self.suggest_deployments()
return_suggestions = self.suggest_returns()
priority_hints = self.get_mission_priority_hints()
spatial = self.get_spatial_awareness()
return {
"enforced": safety,
"suggestions": {
"deploy": [u.id for u in deploy_suggestions],
"return": [u.id for u in return_suggestions],
},
"priority_hints": priority_hints,
"spatial": {
"center_of_mass": spatial["center_of_mass"].tolist(),
"spread_radius": spatial["spread_radius"],
"n_active_positions": len(spatial["active_positions"]),
},
}

def count_by_state(self):
counts = {s.value: 0 for s in UAVState}
for u in self.all_uavs:
counts[u.state.value] += 1
return counts

def get_battery_stats(self):
if not self.all_uavs:
return {}
batteries = [u.battery for u in self.all_uavs if u.state != UAVState.DISABLED]
if not batteries:
return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0,
"critical_count": 0, "low_count": 0, "emergency_count": 0}
return {
"mean": float(np.mean(batteries)),
"min": float(np.min(batteries)),
"max": float(np.max(batteries)),
"std": float(np.std(batteries)),
"critical_count": int(sum(1 for b in batteries if b <= self.cfg.uav.battery_return_threshold)),
"low_count": int(sum(1 for b in batteries if b <= self.cfg.uav.battery_penalty_low)),
"emergency_count": int(sum(1 for b in batteries if b <= self.cfg.uav.battery_penalty_emergency)),
}

def get_stats(self):
state_counts = self.count_by_state()
battery = self.get_battery_stats()
priority = self.get_mission_priority_hints()
spatial = self.get_spatial_awareness()
n_deployable = len(self.get_deployable_uavs())
return {
"n_total": int(self.n_total),
"n_reserve": int(self.n_reserve),
"n_active": int(state_counts.get(UAVState.ACTIVE.value, 0)),
"n_returning": int(state_counts.get(UAVState.RETURNING.value, 0)),
"n_charging": int(state_counts.get(UAVState.CHARGING.value, 0)),
"n_deploying": int(state_counts.get(UAVState.DEPLOYING.value, 0)),
"n_disabled": int(state_counts.get(UAVState.DISABLED.value, 0)),
"n_deployable": int(n_deployable),
"battery": battery,
"priority": priority,
"spatial": {"spread_radius": spatial["spread_radius"],
"n_active_positions": len(spatial["active_positions"])},
"enforced_disables": int(self.\_enforced_disables),
"enforced_returns": int(self.\_enforced_returns),
"suggested_deploys": int(self.\_suggested_deploys),
}

def get_fleet_incentives(self):
return {"deploy": 0.0, "recall": 0.0, "total": 0.0}

def is_episode_over(self):
return all(u.state == UAVState.DISABLED for u in self.all_uavs)

def **repr**(self):
counts = self.count_by_state()
priority = self.get_mission_priority_hints()
return (
f"FleetManager("
f"total={self.n_total}, "
f"active={counts.get(UAVState.ACTIVE.value, 0)}, "
f"reserve={self.n_reserve}, "
f"op_ratio={priority['operational_ratio']:.2f})"
)
"""
Logic backend for SAR UAV environment.

Pure Python physics simulation (CPU only).
~1000 steps/second on modern hardware.

Design:

- Deterministic given same seed
- No hidden heuristics (RL controls behavior)
- Clean separation: physics in step_physics, world in step_world
  """

from future import annotations

import logging
from typing import Any

import numpy as np

✅ FIX 4.1: REMOVED sys.path hack
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(file)))
from config import AppConfig
from core.coverage_map import CoverageMap
from core.fleet_manager import FleetManager
from entities.charging_station import ChargingStation
from entities.obstacle import Debris, DangerZone
from entities.uav import UAV, UAVState
from entities.victim import InjuredVictim, MobileVictim
from sensors.fov_sensor import FOVSensor

from .base_backend import BaseBackend

logger = logging.getLogger(name)

class LogicBackend(BaseBackend):
"""
Pure Python physics backend (~1000 steps/s).

text

UAV spawn:
✅ FIX 4.2: Dùng map_data["uav_spawns"] trực tiếp (từ MapGenerator.generate())
KHÔNG gọi lại get_uav_spawns() → tránh re-generate với RNG khác

Deterministic eval:
✅ FIX 4.3: Reset np.random seed khi cfg.env.deterministic_eval = True
"""

def **init**(self, cfg: AppConfig) -> None:
self.cfg = cfg

    # Entities (populated in reset)
    self.uavs:      list[UAV]                       = []
    self.victims:   list[InjuredVictim | MobileVictim] = []
    self.stations:  list[ChargingStation]           = []
    self.obstacles: list[Debris | DangerZone]       = []

    # Sub-systems
    self._cov_map   = CoverageMap(cfg)
    self._fleet_mgr = FleetManager(cfg)
    self._fov_sensor = FOVSensor(cfg)

    # ✅ FIX 4.2: REMOVED self._map_gen (MapGenerator)
    # MapGenerator chỉ dùng để generate map (base_env làm điều đó)
    # Backend chỉ cần nhận map_data đã generate sẵn
    # Spawn positions đến từ map_data["uav_spawns"]

    self._step_count = 0

# ══════════════════════════════════════════════════════════════════════

# RESET

# ══════════════════════════════════════════════════════════════════════

def reset(self, map_data: dict[str, Any]) -> None: # ✅ FIX 4.3: Deterministic eval mode
if self.cfg.env.deterministic_eval:
np.random.seed(self.cfg.env.eval_seed)
logger.debug(
"LogicBackend: deterministic_eval=True, seed=%d",
self.cfg.env.eval_seed,
)

    # ✅ NEW: Set FOVSensor RNG seed cho reproducible detection
    if self.cfg.env.deterministic_eval:
        self._fov_sensor.set_seed(self.cfg.env.eval_seed)
    else:
        # Random seed mỗi episode → stochastic training
        self._fov_sensor.set_seed(
            int(np.random.randint(0, 2**31))
        )

    # Build entities từ map_data (giữ nguyên)
    self.stations  = self._build_stations(map_data)
    self.obstacles = self._build_obstacles(map_data)
    self.victims   = self._build_victims(map_data)
    self.uavs      = self._build_uavs(map_data)

    self._cov_map.reset()
    self._fleet_mgr.reset(self.uavs, self.stations)
    self._step_count = 0

    logger.debug(
        "LogicBackend reset: %d UAVs, %d victims, %d stations, %d obstacles",
        len(self.uavs), len(self.victims),
        len(self.stations), len(self.obstacles),
    )

# ══════════════════════════════════════════════════════════════════════

# APPLY ACTIONS

# ══════════════════════════════════════════════════════════════════════

def \_find_nearest_station_in_range(
self,
uav,
range_m: float,
):
"""
Tìm station gần nhất trong range_m.
Returns None nếu không có station nào trong range.
"""
best = None
best_dist = float("inf")

    for station in self.stations:
        dx   = float(uav.pos[0]) - float(station.pos[0])
        dy   = float(uav.pos[1]) - float(station.pos[1])
        dist = float(np.sqrt(dx * dx + dy * dy))

        if dist <= range_m and dist < best_dist:
            best      = station
            best_dist = dist

    return best

def apply_actions(self, actions: dict[int, np.ndarray]) -> None:
"""
Apply velocity commands + landing signal to UAVs.

    Action format: [vx, vy, vz, land]
        - [vx, vy, vz] ∈ [-1, 1]³  : movement
        - [land]       ∈ {0.0, 1.0} : landing intent

    Landing logic (ALL conditions must hold):
        1. land > 0.5           → agent muốn land
        2. state == ACTIVE      → chỉ active mới được land
        3. station trong range  → charge_radius × 2.0 (wider detection)
        4. battery ≤ 40%        → chỉ land khi cần sạc
           (Tránh agent land khi đang survey = local optimum)

    Note:
        - land=1 nhưng không có station → move bình thường (graceful fallback)
        - land=1 nhưng battery cao → move bình thường (tránh false positive)
    """
    for uav in self.uavs:

        if uav.state == UAVState.ACTIVE:
            action = actions.get(uav.id, np.zeros(4, dtype=np.float32))
            action = np.clip(action, -1.0, 1.0).astype(np.float64)

            move_action = action[:3]    # [vx, vy, vz]
            land_signal = float(action[3])

            # ✅ Landing conditions
            wants_to_land = land_signal > 0.5

            if wants_to_land:
                # Tìm station gần nhất (KHÔNG giới hạn range)
                nearest = uav.find_nearest_station(self.stations)

                if nearest is not None:
                    # ✅ Hạ cánh: navigate xuống station
                    target = np.array([
                        nearest.pos[0],
                        nearest.pos[1],
                        0.0,   # Ground level
                    ], dtype=np.float64)
                    uav.auto_navigate(target)
                    uav.target_station = nearest
                    uav.set_state(UAVState.RETURNING)
                else:
                    # Không có station → move bình thường
                    uav.apply_action(move_action)
            else:
                # Normal movement
                uav.apply_action(move_action)
        elif uav.state == UAVState.RETURNING:
            if uav.target_station is not None:
                target = np.array([
                    uav.target_station.pos[0],
                    uav.target_station.pos[1],
                    0.0,
                ], dtype=np.float64)
                uav.auto_navigate(target)
            else:
                nearest = uav.find_nearest_station(self.stations)
                if nearest is not None:
                    target = np.array([
                        nearest.pos[0],
                        nearest.pos[1],
                        0.0,
                    ], dtype=np.float64)
                    uav.auto_navigate(target)

        elif uav.state == UAVState.DEPLOYING:
            # Simple: chuyển ACTIVE ngay
            uav.set_state(UAVState.ACTIVE)
            uav.target_station = None
        # CHARGING / DISABLED: no movement

# ══════════════════════════════════════════════════════════════════════

# STEP PHYSICS

# ══════════════════════════════════════════════════════════════════════

def step_physics(self) -> None:
for uav in self.uavs:
if uav.state == UAVState.DISABLED:
continue

        # ✅ RETURNING → CHARGING transition
        if uav.state == UAVState.RETURNING:
            if uav.target_station is not None:
                if uav.target_station.in_range(uav.pos):
                    if uav.target_station.try_occupy(uav):
                        uav.set_state(UAVState.CHARGING)
                        logger.debug(
                            f"UAV {uav.id} → CHARGING "
                            f"at station {uav.target_station.id}"
                        )
            # RETURNING vẫn drain
            uav.update_battery(self.stations)
            continue  # ← Skip phần dưới

        # ✅ CHARGING: charge via station
        if uav.state == UAVState.CHARGING:
            if uav.target_station is not None:
                uav.target_station.charge(uav)
            else:
                # Fallback: tìm station đang occupy
                for station in self.stations:
                    if station.has_uav(uav):
                        station.charge(uav)
                        break
            continue  # ← Skip update_battery

        # ACTIVE / DEPLOYING: drain
        uav.update_battery(self.stations)

# ══════════════════════════════════════════════════════════════════════

# STEP WORLD

# ══════════════════════════════════════════════════════════════════════

def step_world(self) -> None:
self.\_step_count += 1

    # 1. Fleet
    self._fleet_mgr.step()

    # 2. Victim movement - truyền obstacles để obstacle-aware
    for v in self.victims:
        v.update(self._step_count, obstacles=self.obstacles)
        #                          ↑ FIX: truyền obstacles

    # 3. Coverage
    for uav in self.uavs:
        if uav.state == UAVState.DISABLED:
            continue
        fov_r = self._fov_sensor.calculate_fov_radius(uav.pos[2])
        self._cov_map.mark_explored(uav.pos, fov_r, self._step_count)

    # 4. Detection với noise
    for uav in self.uavs:
        if uav.state not in (UAVState.ACTIVE, UAVState.RETURNING):
            continue
        for victim in self.victims:
            if victim.is_found:
                continue
            if self._fov_sensor.check_detected(uav, victim, self.obstacles):
                victim.mark_found(self._step_count, uav.id)

# ══════════════════════════════════════════════════════════════════════

# GET STATE

# ══════════════════════════════════════════════════════════════════════

def get_state(self) -> dict[str, Any]:
"""Return current state của tất cả entities."""
return {
"uavs": self.uavs,
"victims": self.victims,
"stations": self.stations,
"obstacles": self.obstacles,
"coverage_map": self.\_cov_map,
"fleet_manager": self.\_fleet_mgr,
}

# ══════════════════════════════════════════════════════════════════════

# BUILD ENTITIES (private helpers)

# ══════════════════════════════════════════════════════════════════════

def \_build_stations(self, map_data: dict) -> list[ChargingStation]:
"""Build ChargingStation objects từ map_data."""
stations = []
for s in map_data["stations"]:
station = ChargingStation(
station_id = s["id"],
pos = s["pos"],
cfg = self.cfg,
)
stations.append(station)
return stations

def \_build_obstacles(self, map_data: dict) -> list[Debris | DangerZone]:
"""
Build Debris + DangerZone objects từ map_data.

    Supports multi-shape: circle | rectangle | polygon
    Unknown shapes are skipped with warning.
    """
    obstacles = []

    # ── Debris ───────────────────────────────────────────────────────
    for d_dict in map_data.get("debris", []):
        shape     = d_dict.get("shape", "circle")
        debris_id = d_dict["id"]
        pos       = d_dict["pos"]
        height_3d = d_dict.get("height_3d", 10.0)

        if shape == "circle":
            obj = Debris(
                debris_id = debris_id, pos = pos,
                height_3d = height_3d, cfg = self.cfg,
                shape = "circle", radius = d_dict["radius"],
            )
        elif shape == "rectangle":
            obj = Debris(
                debris_id = debris_id, pos = pos,
                height_3d = height_3d, cfg = self.cfg,
                shape = "rectangle",
                width     = d_dict["width"],
                height_2d = d_dict["height_2d"],
                rotation  = d_dict["rotation"],
            )
        elif shape == "polygon":
            obj = Debris(
                debris_id = debris_id, pos = pos,
                height_3d = height_3d, cfg = self.cfg,
                shape = "polygon", vertices = d_dict["vertices"],
            )
        else:
            logger.warning("Unknown debris shape '%s', skipping id=%d", shape, debris_id)
            continue

        obstacles.append(obj)

    # ── Danger Zones ─────────────────────────────────────────────────
    for z_dict in map_data.get("danger_zones", []):
        shape       = z_dict.get("shape", "circle")
        zone_id     = z_dict["id"]
        pos         = z_dict["pos"]
        danger_type = z_dict["danger_type"]

        if shape == "circle":
            obj = DangerZone(
                zone_id     = zone_id, pos = pos,
                danger_type = danger_type, cfg = self.cfg,
                shape = "circle", radius = z_dict["radius"],
            )
        elif shape == "rectangle":
            obj = DangerZone(
                zone_id     = zone_id, pos = pos,
                danger_type = danger_type, cfg = self.cfg,
                shape = "rectangle",
                width     = z_dict["width"],
                height_2d = z_dict["height_2d"],
                rotation  = z_dict["rotation"],
            )
        else:
            logger.warning("Unknown zone shape '%s', skipping id=%d", shape, zone_id)
            continue

        obstacles.append(obj)

    return obstacles

def \_build_victims(self, map_data: dict) -> list[InjuredVictim | MobileVictim]:
"""Build Victim objects từ map_data."""
victims = []
for v in map_data["victims"]:
victim_type = v.get("victim_type", "injured")
cls = MobileVictim if victim_type == "mobile" else InjuredVictim

        victim = cls(
            victim_id = v["id"],
            pos       = v["pos"],
            cfg       = self.cfg,
            urgency   = v.get("urgency", 3.0),
        )
        victims.append(victim)
    return victims

def \_build_uavs(self, map_data: dict) -> list[UAV]:
"""
Build UAV objects từ map_data["uav_spawns"].

    ✅ FIX 4.2: Dùng pre-generated spawn positions từ MapGenerator.generate()
                KHÔNG gọi lại get_uav_spawns() để tránh RNG drift.

    map_data["uav_spawns"] format:
        [{"id": 0, "pos": [x, y, z]}, ...]
    """
    spawns = map_data.get("uav_spawns", [])

    # Fallback nếu uav_spawns empty (backward compat)
    if not spawns:
        logger.warning(
            "map_data['uav_spawns'] is empty, "
            "falling back to station-based spawn. "
            "Check MapGenerator.generate() FIX-P10."
        )
        # Emergency fallback: spawn tại stations
        for i, station in enumerate(self.stations):
            if i >= self.cfg.env.n_uav:
                break
            spawns.append({
                "id":  i,
                "pos": [station.pos[0], station.pos[1], self.cfg.uav.z_min],
            })

    uavs = []
    for s in spawns:
        uav = UAV(
            uav_id  = s["id"],
            pos     = s["pos"],
            cfg     = self.cfg,
            battery = 100.0,
        )
        uavs.append(uav)

    return uavs

"""
Abstract backend interface for SAR UAV environment.

Separates physics simulation from RL logic.
Allows switching between different physics engines (logic/pybullet/isaac).
"""

from future import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

✅ FIX 4.1: REMOVED sys.path hack
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(file)))
class BaseBackend(ABC):
"""
Abstract interface for environment physics backend.

text

Concrete implementations:

- LogicBackend: Pure Python simulation (CPU)
- PyBulletBackend: PyBullet physics (CPU, realistic)
- IsaacBackend: IsaacLab physics (GPU, parallel)
  """

@abstractmethod
def reset(self, map_data: dict[str, Any]) -> None:
"""
Initialize/reset backend with map from MapGenerator.

    Builds entities (UAVs, victims, stations, obstacles) from map_data
    and resets internal state (coverage, fleet manager, etc).

    Args:
        map_data: Dictionary containing:
            - "stations":     list of station dicts
            - "debris":       list of debris dicts
            - "danger_zones": list of danger zone dicts
            - "victims":      list of victim dicts
            - "uav_spawns":   list of spawn position dicts (from MapGenerator)
    """
    pass

@abstractmethod
def apply_actions(self, actions: dict[int, np.ndarray]) -> None:
"""
Apply velocity commands to UAVs.

    For ACTIVE UAVs:              apply action to set velocity.
    For RETURNING/DEPLOYING UAVs: auto-navigate towards target.
    For CHARGING/DISABLED UAVs:   no movement.

    Args:
        actions: {uav_id: action_array [vx, vy, vz] ∈ [-1, 1]³}
                 Actions are normalized, will be scaled by backend.
    """
    pass

@abstractmethod
def step_physics(self) -> None:
"""
Update physics state: movement, collision, battery.

    Execution order:
        1. Battery drain (ACTIVE / RETURNING / DEPLOYING)
        2. Battery charge (CHARGING at station)
        3. Collision detection (UAV vs Debris)
        4. State transitions (battery dead → DISABLED)

    NOTE: No reward logic here. Backend is pure simulation.
    """
    pass

@abstractmethod
def step_world(self) -> None:
"""
Update world state: victims, coverage, fleet management.

    Execution order (temporal correctness):
        1. Fleet manager step (UAV state transitions)
        2. Mobile victims movement
        3. Coverage map update (mark explored cells)
        4. Victim detection (FOV scan at current positions)

    NOTE: Order matters for temporal consistency.
          Detection uses current-step positions (no lag).
    """
    pass

@abstractmethod
def get_state(self) -> dict[str, Any]:
"""
Get current state of all entities.

    Used by base_env to compute observations and rewards.

    Returns:
        {
            "uavs":         list[UAV]
            "victims":      list[BaseVictim]
            "stations":     list[ChargingStation]
            "obstacles":    list[Debris | DangerZone]
            "coverage_map": CoverageMap
            "fleet_manager": FleetManager
        }
    """
    pass

"""
env/base_env.py
Base environment cho SAR UAV Swarm.

FIXES APPLIED:
✅ FIX-3.1: Episode reward accumulation
✅ FIX-3.2: Extreme episode logging
✅ FIX-3.3: Attribute errors
✅ FIX-3.4: Per-step extreme detection
✅ BUG-ENV-02: reward_fn.reset()
✅ BUG-ENV-03: Terminal reward không double-count
✅ BUG-ENV-04: max_steps REMOVED từ build_all()
✅ BUG-ENV-05: Episode metrics keys
✅ BUG-ENV-06: done/truncated MOVED trước reward computation ✅✅✅
"""

from future import annotations

import time
import logging

import numpy as np

try:
import gymnasium as gym
from gymnasium import spaces
except ImportError:
import gymnasium as gym
from gymnasium import spaces

from config import AppConfig
from core.map_generator import MapGenerator
from entities.uav import UAV, UAVState
from observation.obs_builder import ObservationBuilder
from rewards.baseline_reward import BaselineReward
from utils.logger import EpisodeLogger
from visualization.renderer_factory import create_renderer

logger = logging.getLogger(name)

─────────────────────────────────────────────────────────────────────────────
CONSTANTS
─────────────────────────────────────────────────────────────────────────────
\_DONE_COVERAGE_THRESHOLD = 0.90

\_INFO_STEP = "step"
\_INFO_COVERAGE = "coverage_rate"
\_INFO_VICTIMS_FOUND = "victims_found"
\_INFO_VICTIMS_TOTAL = "victims_total"
\_INFO_N_ACTIVE = "n_active"
\_INFO_N_CHARGING = "n_charging"
\_INFO_N_DISABLED = "n_disabled"
\_INFO_SUCCESS = "success"
\_INFO_DONE_REASON = "done_reason"
\_INFO_EPISODE_TIME = "episode_time_s"
\_INFO_REWARDS = "rewards_breakdown"

class SARBaseEnv(gym.Env):
"""
Search-and-Rescue UAV Swarm base environment.
Gymnasium multi-agent environment.
"""

text

metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

def **init**(
self,
cfg: AppConfig | None = None,
backend: str = "logic",
render_mode: str | None = None,
n_victims_override: int | None = None,
verbose: int = 0,
viz_mode: str = "2d",
):
super().**init**()
self.cfg = cfg or AppConfig()
self.render_mode = render_mode
self.\_n_victims_ov = n_victims_override
self.verbose = verbose
self.\_viz_mode = cfg.viz_mode

    actor_dim = self.cfg.obs.actor_dim
    self.observation_space = spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(actor_dim,), dtype=np.float32,
    )
    self.action_space = spaces.Box(
        low=-1.0, high=1.0,
        shape=(3,), dtype=np.float32,
    )

    self._map_gen   = MapGenerator(self.cfg)
    self._reward_fn = BaselineReward(self.cfg)

    if backend == "logic":
        from env_setup.backends.logic_backend import LogicBackend
        self.backend = LogicBackend(self.cfg)
    elif backend == "pybullet":
        raise NotImplementedError("PyBullet backend not yet implemented")
    elif backend == "isaac":
        raise NotImplementedError("IsaacLab backend not yet implemented")
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    self._obs_builder:         ObservationBuilder | None = None
    self._step_count:          int   = 0
    self._prev_coverage:       float = 0.0
    self._episode_seed:        int   = 0
    self._episode_id:          int   = 0
    self._ep_logger:           EpisodeLogger | None = None
    self._ep_start_time:       float = 0.0
    self._episode_reward_sum:  float = 0.0
    self._step_rewards_history: list  = []
    self._renderer = None

# ── Reset ────────────────────────────────────────────────────────────────

def reset(
self,
seed: int | None = None,
options: dict | None = None,
) -> tuple[dict[int, np.ndarray], dict]:
super().reset(seed=seed)

    if seed is None:
        if self.cfg.env.deterministic_eval:
            seed = self.cfg.env.eval_seed
            logger.info("Deterministic eval mode: seed=%d", seed)
        else:
            seed = int(time.time() * 1000) % (2 ** 31)

    self._episode_seed        = seed
    self._episode_id         += 1
    self._ep_start_time       = time.time()
    self._episode_reward_sum  = 0.0
    self._step_rewards_history = []

    # ✅ BUG-ENV-02: Reset reward state
    self._reward_fn.reset()

    map_data = self._map_gen.generate(
        n_victims_override=self._n_victims_ov,
        seed=seed,
    )
    self.backend.reset(map_data)

    self._step_count    = 0
    self._prev_coverage = 0.0

    state = self.backend.get_state()
    self._obs_builder = ObservationBuilder(
        state["coverage_map"],
        self.cfg,
    )

    self._ep_logger = EpisodeLogger(
        episode_id=self._episode_id,
        seed=seed,
    )
    self._ep_logger.set_total_victims(len(state["victims"]))

    obs_dict,critic_obs = self._build_obs_dict(
        state["uavs"],
        state["stations"],
        state["victims"],
        state["obstacles"],
    )



    if self._renderer is not None and hasattr(self._renderer, "reset_scene"):
        self._renderer.reset_scene()

    info = {
        "seed":              seed,
        "n_uav":             len(state["uavs"]),
        "n_stations":        len(state["stations"]),
        "n_victims":         len(state["victims"]),
        "n_obstacles":       len(state["obstacles"]),
        "map_size":          self.cfg.env.map_size,
        "coverage":          0.0,
        _INFO_COVERAGE:      0.0,
        _INFO_VICTIMS_FOUND: 0,
        _INFO_VICTIMS_TOTAL: len(state["victims"]),
    }
    info['global_obs'] = critic_obs

    if self.verbose >= 2:
        print(
            f"[ENV] Episode {self._episode_id} | seed={seed} "
            f"| {len(state['uavs'])} UAVs "
            f"| {len(state['victims'])} victims"
        )

    return obs_dict, info

# ── Step ─────────────────────────────────────────────────────────────────

def step(
self,
actions: dict[int, np.ndarray],
) -> tuple[
dict[int, np.ndarray],
dict[int, float],
bool,
bool,
dict,
]: # ✅ 1. Physics
self.backend.apply_actions(actions)
self.backend.step_physics()
self.backend.step_world()
self.\_step_count += 1

    # ✅ 2. State
    state         = self.backend.get_state()
    uavs          = state["uavs"]
    victims       = state["victims"]
    obstacles     = state["obstacles"]
    stations      = state["stations"]
    coverage_map  = state["coverage_map"]
    fleet_manager = state["fleet_manager"]

    # ✅ NEW: Track landing transitions
    for uav in uavs:
        prev_state = getattr(uav, '_prev_state', None)

        # Detect ACTIVE/RETURNING → CHARGING transition (landing event)
        if prev_state in (UAVState.ACTIVE, UAVState.RETURNING) and \
        uav.state == UAVState.CHARGING:
            # Log landing event
            battery_before = getattr(uav, '_battery_before_charge', uav.battery_pct)
            self._ep_logger.log_landing(
                uav_id=uav.id,
                step=self._step_count,
                battery_before=battery_before,
                battery_after=uav.battery_pct,
            )

        # Track CHARGING steps (for total_charge_time)
        if uav.state == UAVState.CHARGING:
            self._ep_logger.log_charging_step(uav.id)

        # Store battery before charge (for next step's landing event)
        if uav.state == UAVState.RETURNING:
            uav._battery_before_charge = uav.battery_pct
    # ✅ 3. Coverage
    cur_coverage = coverage_map.get_coverage_rate()

    # ✅ 4. Newly found
    newly_found = [
        v for v in victims
        if v.is_found and v.found_at_step == self._step_count
    ]

    # ✅ BUG-ENV-06: done/truncated TRƯỚC reward
    done_reason = self._check_done(cur_coverage, victims, uavs)
    done        = done_reason is not None
    truncated   = self._step_count >= self.cfg.env.max_steps
    is_terminal = done or truncated

    # ✅ 5. Per-agent rewards
    rewards_dict: dict[int, float] = {}
    for uav in uavs:
        if uav.state == UAVState.DISABLED:
            rewards_dict[uav.id] = 0.0
            continue

        newly_found_by_uav = [
            v for v in newly_found if v.found_by_uav == uav.id
        ]

        breakdown = self._reward_fn.compute_per_uav(
            uav                = uav,
            newly_found_by_uav = newly_found_by_uav,
            uavs               = uavs,
            victims            = victims,
            obstacles          = obstacles,
            coverage_map       = coverage_map,
            fleet_manager      = fleet_manager,
            prev_coverage      = self._prev_coverage,
            current_step       = self._step_count,
            done               = is_terminal,
            stations           = stations,
        )
        rewards_dict[uav.id] = breakdown["total"]

    # ✅ 6. Global reward
    global_reward = self._reward_fn.compute(
        uavs          = uavs,
        victims       = victims,
        obstacles     = obstacles,
        coverage_map  = coverage_map,
        fleet_manager = fleet_manager,
        newly_found   = newly_found,
        prev_coverage = self._prev_coverage,
        current_step  = self._step_count,
        done          = is_terminal,
        stations      = stations,
    )

    # ✅ 7. Accumulate episode reward
    self._episode_reward_sum += global_reward["total"]
    self._step_rewards_history.append({
        "step":   self._step_count,
        "total":  global_reward["total"],
        "global": global_reward["total"],
    })

    # ✅ BUG-1 FIX: Proper logging (không phải floating literals)
    if global_reward["total"] < -100:
        logger.warning(
            "[EP %d | STEP %d] Extreme step reward: %.1f | %s",
            self._episode_id,
            self._step_count,
            global_reward["total"],
            self._reward_fn.summarize(global_reward),
        )

    # ✅ 9. Logging
    self._log_step(rewards_dict, cur_coverage, newly_found, uavs, obstacles)
    self._prev_coverage = cur_coverage

    # ✅ 10. Observations
    obs_dict, critic_obs = self._build_obs_dict(uavs, stations, victims, obstacles)

    # ✅ 11. Info
    n_found  = sum(1 for v in victims if v.is_found)
    n_total  = len(victims)
    n_active = sum(1 for u in uavs if u.state == UAVState.ACTIVE)
    n_charge = sum(1 for u in uavs if u.state == UAVState.CHARGING)
    n_dead   = sum(1 for u in uavs if u.state == UAVState.DISABLED)
    success  = done_reason in ("coverage", "victims")
    fleet_stats = fleet_manager.get_battery_stats()

    info = {
        "coverage":          cur_coverage,
        "victims_found":     n_found,
        "victims_total":     n_total,
        _INFO_STEP:          self._step_count,
        _INFO_COVERAGE:      cur_coverage,
        _INFO_VICTIMS_FOUND: n_found,
        _INFO_VICTIMS_TOTAL: n_total,
        _INFO_N_ACTIVE:      n_active,
        _INFO_N_CHARGING:    n_charge,
        _INFO_N_DISABLED:    n_dead,
        _INFO_SUCCESS:       success,
        _INFO_DONE_REASON:   done_reason or ("truncated" if truncated else None),
        _INFO_REWARDS:       global_reward,
        "newly_found_ids":   [v.id for v in newly_found],
        "battery_stats":     fleet_stats,
        "global_obs":        critic_obs,
    }

    # ✅ 12. Episode-end
    if is_terminal:
        if self._ep_logger is not None:
            ep_metrics = self._ep_logger.finalize()
            info["episode"]          = ep_metrics
            info[_INFO_EPISODE_TIME] = time.time() - self._ep_start_time

            if self.verbose >= 1:
                self._print_episode_summary(ep_metrics, done_reason, truncated)

            if done_reason == "disabled":
                uav_batteries = {f"uav_{u.id}": u.battery_pct for u in uavs}
                logger.info(
                    "[EP %d] All UAVs disabled. Batteries: %s",
                    self._episode_id,
                    uav_batteries,
                )

        if self._episode_reward_sum < -500:
            self._log_extreme_episode(
                episode_reward = self._episode_reward_sum,
                cur_coverage   = cur_coverage,
                n_found        = n_found,
                n_total        = n_total,
                uavs           = uavs,
                obstacles      = obstacles,
                done_reason    = done_reason,
                truncated      = truncated,
            )

    return obs_dict, rewards_dict, done, truncated, info

# ── Render ───────────────────────────────────────────────────────────────

def render(self) -> np.ndarray | None:
if self.render_mode is None:
return None
if self.\_renderer is None:
self.\_renderer = self.\_init_renderer()

    state = self.backend.get_state()
    return self._renderer.render(
        uavs      = state["uavs"],
        victims   = state["victims"],
        obstacles = state["obstacles"],
        stations  = state["stations"],
        cov_map   = state["coverage_map"],
        step      = self._step_count,
    )

def \_init_renderer(self):
return create_renderer(
cfg = self.cfg,
render_mode = self.render_mode,
viz_mode = self.\_viz_mode,
)

def close(self):
if self.\_renderer is not None:
self.\_renderer.close()
self.\_renderer = None

# ── Properties ───────────────────────────────────────────────────────────

@property
def n_agents(self) -> int:
return sum(
1 for u in self.backend.get_state()["uavs"]
if u.state != UAVState.DISABLED
)

@property
def active_uav_ids(self) -> list[int]:
return [
u.id for u in self.backend.get_state()["uavs"]
if u.state == UAVState.ACTIVE
]

@property
def alive_uav_ids(self) -> list[int]:
return [
u.id for u in self.backend.get_state()["uavs"]
if u.state != UAVState.DISABLED
]

@property
def step_count(self) -> int:
return self.\_step_count

@property
def coverage_rate(self) -> float:
return self.backend.get_state()["coverage_map"].get_coverage_rate()

# ── Private ───────────────────────────────────────────────────────────────

def \_build_obs_dict(
self,
uavs: list,
stations: list,
victims: list,
obstacles: list,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
"""
✅ BUG-ENV-04 FIX: KHÔNG truyền max_steps vào build_all()
"""
obs_dict: dict[int, np.ndarray] = {}

    result = self._obs_builder.build_all(
        all_uavs     = uavs,
        stations     = stations,
        victims      = victims,
        obstacles    = obstacles,
        current_step = self._step_count,
        # ✅ max_steps KHÔNG có trong signature của build_all()
    )

    for uid, obs in result.actor_obs.items():
        uav = self._get_uav_from_list(uid, uavs)
        if uav is None or uav.state == UAVState.DISABLED:
            continue
        obs_dict[uid] = obs.astype(np.float32)

    return obs_dict,result.critic_obs.copy()

def \_check_done(self, coverage, victims, uavs) -> str | None:
"""
Returns done reason string (human-readable).
""" # Coverage win condition
if coverage >= 0.90:
return "coverage"

    # Victims win condition
    if victims and all(v.is_found for v in victims):
        return "victims"

    # All UAVs disabled
    all_disabled = all(u.state == UAVState.DISABLED for u in uavs)
    if all_disabled:
        # ✅ Detailed sub-reason
        dead_batteries = sum(1 for u in uavs if u.battery_death)
        if dead_batteries == len(uavs):
            return "disabled:battery_death"  # All died from battery
        else:
            return "disabled:other"  # Mixed causes

    return None  # Episode continues

def \_log_step(
self,
rewards: dict,
coverage: float,
newly_found: list,
uavs: list,
obstacles: list,
) -> None:
if self.\_ep_logger is None:
return

    self._ep_logger.log_step(rewards=rewards, coverage=coverage)

    for uav in uavs:
        for obs in obstacles:
            if hasattr(obs, "causes_collision") and obs.causes_collision(uav.pos):
                # Log chi tiết
                obs_info = {
                    "id": obs.id if hasattr(obs, 'id') else None,
                    "type": type(obs).__name__,
                    "pos": obs.pos.tolist() if hasattr(obs, 'pos') else None,
                    "height": getattr(obs, 'height_3d', None) if hasattr(obs, 'height_3d') else None
                }
                self._ep_logger.log_collision(uav.id, self._step_count, obs_info)
                self._ep_logger.log_event("collision_obstacle")
                break

def \_print_episode_summary(
self,
metrics: dict,
done_reason: str | None,
truncated: bool,
) -> None:
"""✅ BUG-ENV-05 FIX: Multiple key fallbacks."""
success = metrics.get("success", False)
status = "SUCCESS" if success else "FAIL"
reason = done_reason or ("truncated" if truncated else "unknown")

    cov = (
        metrics.get("coverage_rate")
        or metrics.get("coverage")
        or metrics.get("final_coverage")
        or 0.0
    )
    v_found = (
        metrics.get("victims_found")
        or metrics.get("n_found")
        or 0
    )
    v_total = (
        metrics.get("total_victims")
        or metrics.get("victims_total")
        or 0
    )

    print(
        f"[ENV] Ep {self._episode_id} {status} | "
        f"reason={reason} | "
        f"steps={self._step_count} | "
        f"cov={cov:.1%} | "
        f"victims={v_found}/{v_total} | "
        f"ep_reward={self._episode_reward_sum:.1f}"
    )

def \_log_extreme_episode(
self,
episode_reward: float,
cur_coverage: float,
n_found: int,
n_total: int,
uavs: list,
obstacles: list,
done_reason: str | None,
truncated: bool,
) -> None: # ✅ BUG-1 FIX: Proper logger.warning call
from entities.obstacle import DangerZone

    n_in_danger = sum(
        1 for u in uavs for obs in obstacles
        if isinstance(obs, DangerZone) and obs.is_inside(u.pos)
    )
    n_collisions   = len(getattr(self._reward_fn, "_collision_penalized",    set()))
    n_dead_battery = len(getattr(self._reward_fn, "_battery_death_penalized", set()))

    emergency_pct = getattr(
        self.cfg.uav, "battery_emergency_pct",
        getattr(self.cfg.uav, "battery_penalty_emergency", 5.0),
    )
    n_battery_critical = sum(1 for u in uavs if u.battery < emergency_pct)
    n_disabled         = sum(1 for u in uavs if u.state == UAVState.DISABLED)

    worst_str = "N/A"
    if self._step_rewards_history:
        worst     = sorted(self._step_rewards_history, key=lambda x: x["total"])[:3]
        worst_str = ", ".join(
            f"step {s['step']}={s['total']:.1f}" for s in worst
        )

    logger.warning(
        "\n%s\n"
        "⚠️  EXTREME EPISODE REWARD\n"
        "  Ep=%d | reward=%.1f | steps=%d | reason=%s\n"
        "  cov=%.1f%% | victims=%d/%d\n"
        "  in_danger=%d | collisions=%d | battery_dead=%d "
        "| battery_critical=%d | disabled=%d\n"
        "  Worst steps: %s\n"
        "%s",
        "=" * 60,
        self._episode_id, episode_reward, self._step_count,
        done_reason or ("truncated" if truncated else "N/A"),
        cur_coverage * 100, n_found, n_total,
        n_in_danger, n_collisions, n_dead_battery,
        n_battery_critical, n_disabled,
        worst_str,
        "=" * 60,
    )

    if self.verbose >= 3:
        import json
        import os
        os.makedirs("results/extreme_episodes", exist_ok=True)
        fname = (
            f"results/extreme_episodes/"
            f"ep_{self._episode_id}_{episode_reward:.0f}.json"
        )
        try:
            with open(fname, "w") as f:
                json.dump(
                    {
                        "episode_id":   self._episode_id,
                        "reward":       episode_reward,
                        "steps":        self._step_count,
                        "step_rewards": self._step_rewards_history,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.warning("Failed to save extreme episode: %s", e)

def \_get_uav_from_list(self, uid: int, uavs: list) -> UAV | None:
for u in uavs:
if u.id == uid:
return u
return None

@classmethod
def make(
cls,
cfg: AppConfig | None = None,
render_mode: str | None = None,
n_victims: int | None = None,
verbose: int = 0,
viz_mode: str = "2d",
) -> "SARBaseEnv":
return cls(
cfg = cfg,
render_mode = render_mode,
n_victims_override = n_victims,
verbose = verbose,
viz_mode = viz_mode,
)
"""
PettingZoo wrapper for SAR UAV Swarm environment.

Provides ParallelEnv API for multi-agent RL frameworks (EPyMARL, RLlib, MARLlib).

Key differences from base_env:

Agent IDs: int → str ("uav_0", "uav_1", ...)
Done signal: bool → dict (per-agent terminations/truncations)
Infos: dict → dict[str, dict] (per-agent infos)
Thin wrapper - no logic, just API conversion.
"""

from future import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

PettingZoo imports
try:
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
PETTINGZOO_AVAILABLE = True
except ImportError:

# Fallback if pettingzoo not installed

PETTINGZOO_AVAILABLE = False
ParallelEnv = object # Dummy base class

from config import AppConfig
from env_setup.base_env import SARBaseEnv
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(file)))

═══════════════════════════════════════════════════════════════════════
MAIN CLASS
═══════════════════════════════════════════════════════════════════════
class SARPettingZooEnv(ParallelEnv):
"""
PettingZoo ParallelEnv wrapper for SAR UAV Swarm.

text

Wraps SARBaseEnv (composition pattern) and converts API:

- reset() returns (obs_dict, infos_dict)
- step() returns (obs, rews, terms, truncs, infos)
- Agent names are strings: "uav_0", "uav_1", ...
- Per-agent terminations/truncations

Usage:
env = SARPettingZooEnv(cfg)
obs, infos = env.reset(seed=42)

    for step in range(1000):
        actions = {agent: env.action_space(agent).sample()
                  for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)

        if not env.agents:  # All agents done
            break

Compatible with:

- PettingZoo utilities (parallel_to_aec, SuperSuit)
- RLlib MultiAgentEnv
- EPyMARL
  """

metadata = {
"render_modes": ["human", "rgb_array"],
"name": "sar_uav_swarm_v0",
"is_parallelizable": True,
}

def **init**(
self,
cfg: AppConfig | None = None,
backend: str = "logic",
render_mode: str | None = None,
n_victims: int | None = None,
verbose: int = 0,
viz_mode: str = "2d"
):
"""
Initialize PettingZoo environment.

    Args:
        cfg: AppConfig instance (or None for default)
        backend: "logic" | "pybullet" | "isaac"
        render_mode: "human" | "rgb_array" | None
        n_victims: Override number of victims (for testing)
        verbose: 0=silent, 1=info, 2=debug
    """
    if not PETTINGZOO_AVAILABLE:
        raise ImportError(
            "PettingZoo not installed. Install with: pip install pettingzoo"
        )

    super().__init__()

    # Wrap base environment
    self._base_env = SARBaseEnv(
        cfg=cfg,
        backend=backend,
        render_mode=render_mode,
        n_victims_override=n_victims,
        verbose=verbose,
        viz_mode= viz_mode
    )

    self.cfg = self._base_env.cfg

    # PettingZoo metadata
    self.possible_agents = [f"uav_{i}" for i in range(self.cfg.env.n_uav)]
    self.agents: list[str] = []  # Updated in reset()

    # Action/Observation spaces (per agent)
    actor_dim = self.cfg.obs.actor_dim  # 68 with n_stations=2

    self._observation_spaces = {
        agent: spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(actor_dim,),
            dtype=np.float32,
        )
        for agent in self.possible_agents
    }

    self._action_spaces = {
        agent: spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),  # (vx, vy, vz)
            dtype=np.float32,
        )
        for agent in self.possible_agents
    }

    self.render_mode = render_mode
    self._episode_info: dict[str, Any] = {}

# ══════════════════════════════════════════════════════════════════════

# PETTINGZOO CORE API

# ══════════════════════════════════════════════════════════════════════

def reset(
self,
seed: int | None = None,
options: dict | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
"""
Reset environment.

    Returns:
        observations: {agent_name: obs_array(actor_dim,)}
        infos: {agent_name: info_dict}
    """
    # Reset base environment (returns int keys)
    obs_dict_int, info = self._base_env.reset(seed=seed)

    # Convert int keys → str keys
    self.agents = [f"uav_{uid}" for uid in sorted(obs_dict_int.keys())]

    observations = {
        f"uav_{uid}": obs
        for uid, obs in obs_dict_int.items()
    }

    # Per-agent infos (all agents share same global info for now)
    infos = {agent: info.copy() for agent in self.agents}

    # Store episode info
    self._episode_info = info

    return observations, infos

def step(
self,
actions: dict[str, np.ndarray],
) -> tuple[
dict[str, np.ndarray], # observations
dict[str, float], # rewards
dict[str, bool], # terminations
dict[str, bool], # truncations
dict[str, dict], # infos
]:
"""
Step environment with actions.

    Args:
        actions: {agent_name: action_array}
                 e.g., {"uav_0": [0.5, 0.3, -0.1], "uav_1": ...}

    Returns:
        observations: {agent_name: obs_array}
        rewards: {agent_name: reward_value}
        terminations: {agent_name: bool} - episode ended for agent
        truncations: {agent_name: bool} - episode truncated
        infos: {agent_name: info_dict}
    """
    # Convert str keys → int keys
    actions_int = {
        int(agent.split("_")[1]): action
        for agent, action in actions.items()
    }

    # Step base environment
    obs_dict, rewards_dict, done, truncated, info = self._base_env.step(actions_int)

    # Convert int keys → str keys
    observations = {
        f"uav_{uid}": obs
        for uid, obs in obs_dict.items()
    }

    rewards = {
        f"uav_{uid}": rew
        for uid, rew in rewards_dict.items()
    }

    # PettingZoo requires per-agent terminations/truncations
    # In our env, all agents terminate together (shared done)
    # But we support per-agent for API compliance

    current_agents = list(observations.keys())

    all_agents_for_signal = self.possible_agents

    terminations = {agent: done     for agent in all_agents_for_signal}
    truncations  = {agent: truncated for agent in all_agents_for_signal}
    infos        = {agent: info.copy() for agent in all_agents_for_signal}

    if done or truncated:
        self.agents = []
    else:
        self.agents = current_agents

    return observations, rewards, terminations, truncations, infos

# ══════════════════════════════════════════════════════════════════════

# PETTINGZOO REQUIRED PROPERTIES

# ══════════════════════════════════════════════════════════════════════

def observation_space(self, agent: str) -> spaces.Space:
"""Get observation space for agent."""
return self.\_observation_spaces[agent]

def action_space(self, agent: str) -> spaces.Space:
"""Get action space for agent."""
return self.\_action_spaces[agent]

# ══════════════════════════════════════════════════════════════════════

# OPTIONAL METHODS

# ══════════════════════════════════════════════════════════════════════

def render(self) -> np.ndarray | None:
"""
Render environment.

    Delegates to base_env.render().

    Returns:
        np.ndarray: RGB array if render_mode="rgb_array"
        None: if render_mode="human" (displays window)
    """
    return self._base_env.render()

def close(self):
"""Close environment and cleanup resources."""
self.\_base_env.close()

@property
def unwrapped(self) -> SARBaseEnv:
"""Return unwrapped base environment."""
return self.\_base_env

# ══════════════════════════════════════════════════════════════════════

# CUSTOM PROPERTIES (for convenience)

# ══════════════════════════════════════════════════════════════════════

@property
def num_agents(self) -> int:
"""Current number of active agents."""
return len(self.agents)

@property
def max_num_agents(self) -> int:
"""Maximum possible agents."""
return len(self.possible_agents)
═══════════════════════════════════════════════════════════════════════
FACTORY FUNCTIONS
═══════════════════════════════════════════════════════════════════════
def make_parallel_env(
cfg: AppConfig | None = None,
\*\*kwargs,
) -> SARPettingZooEnv:
"""
Factory function for parallel env.

text

Usage:
env = make_parallel_env(verbose=1)
env = make_parallel_env(cfg=my_cfg, n_victims=10)

Args:
cfg: AppConfig instance
\*\*kwargs: Passed to SARPettingZooEnv

Returns:
SARPettingZooEnv instance
"""
return SARPettingZooEnv(cfg=cfg, **kwargs)
def make_aec_env(
cfg: AppConfig | None = None,
**kwargs,
) -> wrappers.OrderEnforcingWrapper:
"""
Factory function for AEC (agent-environment-cycle) env.

text

Some frameworks prefer AEC API over parallel API.

Usage:
env = make_aec_env(verbose=1)

Args:
cfg: AppConfig instance
\*\*kwargs: Passed to SARPettingZooEnv

Returns:
AEC-wrapped environment
"""
if not PETTINGZOO_AVAILABLE:
raise ImportError("PettingZoo not installed")

parallel_env = SARPettingZooEnv(cfg=cfg, \*\*kwargs)
aec_env = parallel_to_aec(parallel_env)
return aec_env
"""
env_setup/vec_env.py
FIXED: Seed progression cho mỗi episode
"""

import multiprocessing as mp
import numpy as np
from typing import List, Tuple, Dict

def env_worker(pipe, config_dict, seed):
"""
Worker process với seed progression.

text

FIXED: Mỗi episode dùng seed khác nhau
"""
try:
from env_setup.sar_pettingzoo_env import SARPettingZooEnv
from config import AppConfig

    config = config_dict
    n_agents = config.env.n_uav
    obs_dim = config.obs.actor_dim
    global_obs_dim = config.obs.critic_dim

    env = SARPettingZooEnv(config, render_mode=None)

    # Cache
    last_obs_array = np.zeros((n_agents, obs_dim), dtype=np.float32)
    last_global_obs = np.zeros(global_obs_dim, dtype=np.float32)
    last_info = {
        'uav_0': {
            'coverage_rate': 0.0,
            'victims_found': 0,
            'victims_total': 1,
            'global_obs': np.zeros(global_obs_dim, dtype=np.float32)
        }
    }

    # ✅ FIX: Episode counter + RNG cho seed progression
    episode_count = 0
    rng = np.random.default_rng(seed)  # ← RNG riêng cho worker

    # Initial reset với seed gốc
    current_seed = seed
    obs, info = env.reset(seed=current_seed)
    episode_count += 1

    if obs:
        agent_ids = sorted(obs.keys())
        last_obs_array = np.stack([obs[aid] for aid in agent_ids], axis=0)
    if info and 'uav_0' in info:
        last_global_obs = info['uav_0']['global_obs'].copy()
        last_info = info

    while True:
        cmd, data = pipe.recv()

        if cmd == "reset":
            # ✅ FIX: Generate seed mới cho mỗi episode
            current_seed = int(rng.integers(0, 2**31))
            obs, info = env.reset(seed=current_seed)
            episode_count += 1

            if obs:
                agent_ids = sorted(obs.keys())
                last_obs_array = np.stack(
                    [obs[aid] for aid in agent_ids], axis=0
                ).astype(np.float32)
            if info and 'uav_0' in info:
                last_global_obs = info['uav_0']['global_obs'].copy()
                last_info = info

            pipe.send((
                last_obs_array.copy(),
                last_global_obs.copy(),
                last_info
            ))

        elif cmd == "step":
            actions = data  # [n_agents, 3]
            actions_dict = {
                f"uav_{i}": actions[i] for i in range(n_agents)
            }

            obs, rewards, terms, truncs, info = env.step(actions_dict)
            done = any(terms.values()) or any(truncs.values())

            # Update cache nếu obs hợp lệ
            if obs and len(obs) > 0:
                valid_agents = sorted(obs.keys())
                if len(valid_agents) > 0:
                    obs_arrays = []
                    for i in range(n_agents):
                        aid = f"uav_{i}"
                        if aid in obs:
                            obs_arrays.append(obs[aid])
                        else:
                            obs_arrays.append(
                                np.zeros(obs_dim, dtype=np.float32)
                            )
                    last_obs_array = np.stack(
                        obs_arrays, axis=0
                    ).astype(np.float32)

            if info and 'uav_0' in info:
                last_global_obs = info['uav_0']['global_obs'].copy()
                last_info = info

            # Extract rewards
            rewards_array = np.zeros(n_agents, dtype=np.float32)
            if rewards:
                for i in range(n_agents):
                    aid = f"uav_{i}"
                    if aid in rewards:
                        rewards_array[i] = rewards[aid]

            # Send TRƯỚC khi reset
            pipe.send((
                last_obs_array.copy(),
                last_global_obs.copy(),
                rewards_array.copy(),
                done,
                last_info
            ))

            # ✅ FIX: Auto reset với seed MỚI
            if done:
                current_seed = int(rng.integers(0, 2**31))
                obs_new, info_new = env.reset(seed=current_seed)
                episode_count += 1

                if obs_new:
                    agent_ids = sorted(obs_new.keys())
                    last_obs_array = np.stack(
                        [obs_new[aid] for aid in agent_ids], axis=0
                    ).astype(np.float32)
                if info_new and 'uav_0' in info_new:
                    last_global_obs = info_new['uav_0']['global_obs'].copy()
                    last_info = info_new

        elif cmd == "close":
            break


except (EOFError, BrokenPipeError, KeyboardInterrupt):
pass
except Exception as e:
import traceback
print(f"\n[Worker ERROR] {e}")
traceback.print_exc()
try:
pipe.send(None)
except:
pass
finally:
try:
env.close()
except:
pass
try:
pipe.close()
except:
pass
class VectorizedEnv:
"""Vectorized environment với seed progression fix."""

text

def **init**(self, config, n_envs: int = 8, start_seed: int = 0):
self.n_envs = n_envs
self.n_agents = config.env.n_uav
self.obs_dim = config.obs.actor_dim
self.global_obs_dim = config.obs.critic_dim
self.action_dim = 3
self.config = config
self.start_seed = start_seed

    ctx = mp.get_context("spawn")

    self.pipes = []
    self.processes = []

    print(f"  🔧 Creating {n_envs} parallel environments (spawn method)...")

    for i in range(n_envs):
        parent_pipe, child_pipe = ctx.Pipe()
        p = ctx.Process(
            target=env_worker,
            args=(child_pipe, config, start_seed + i),
            daemon=True
        )
        p.start()
        child_pipe.close()
        self.pipes.append(parent_pipe)
        self.processes.append(p)

    import time
    time.sleep(0.5)
    alive = sum(1 for p in self.processes if p.is_alive())
    print(f"  ✅ {alive}/{n_envs} environment workers ready!")

    if alive < n_envs:
        raise RuntimeError(f"Only {alive}/{n_envs} workers started!")

def reset(self) -> Tuple[np.ndarray, np.ndarray]:
for pipe in self.pipes:
pipe.send(("reset", None))

    obs_list = []
    global_obs_list = []

    for i, pipe in enumerate(self.pipes):
        result = pipe.recv()
        if result is None:
            raise RuntimeError(f"Worker {i} crashed during reset!")
        obs, global_obs, info = result
        obs_list.append(obs)
        global_obs_list.append(global_obs)

    return (
        np.stack(obs_list, axis=0),
        np.stack(global_obs_list, axis=0)
    )

def step(
self,
actions_batch: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[bool], List[Dict]]:
for i, pipe in enumerate(self.pipes):
pipe.send(("step", actions_batch[i]))

    obs_list, global_obs_list, rewards_list = [], [], []
    dones, infos = [], []

    for i, pipe in enumerate(self.pipes):
        result = pipe.recv()
        if result is None:
            raise RuntimeError(f"Worker {i} crashed during step!")
        obs, global_obs, rewards, done, info = result
        obs_list.append(obs)
        global_obs_list.append(global_obs)
        rewards_list.append(rewards)
        dones.append(done)
        infos.append(info)

    return (
        np.stack(obs_list, axis=0),
        np.stack(global_obs_list, axis=0),
        np.stack(rewards_list, axis=0),
        dones,
        infos
    )

def close(self):
for i, pipe in enumerate(self.pipes):
try:
pipe.send(("close", None))
except Exception:
pass

    for p in self.processes:
        p.join(timeout=3)
        if p.is_alive():
            p.terminate()
            p.join(timeout=1)

    print("  ✅ All env workers closed.")

def **del**(self):
try:
self.close()
except Exception:
pass
bạn cần file nào hãy nói cho t biết giờ t sẽ gửi các file cho bạn bạn đọc và hiểu tiếp thoe t gửi tieps đây

claude-sonnet-4-5-20250929
✅ ĐÃ ĐỌC VÀ HIỂU TOÀN BỘ TÀI LIỆU
📋 Tóm tắt nội dung đã tiếp nhận:
Dự án: SAR UAV SWARM - Multi-Agent Search & Rescue
Phase hiện tại: Phase 2 hoàn thành (MAPPO trained)
Phase tiếp theo: Phase 3 (MASAC + MATD3 implementation)
Môi trường: 4 UAVs, 250×250m, 28-36 victims, HARD stage cố định
Không dùng curriculum learning
🎯 Kiến trúc hệ thống đã nắm:

1. State/Action Spaces
   Actor obs: 68 dims (self + stations + teammates + obstacles + victims + coverage)
   Critic obs: 554 dims (8 UAVs × 68 + 10 global features)
   Action: [vx, vy, vz, land] - hybrid continuous + discrete
   Landing logic: Bernoulli + battery ≤40% + station in 6m range
2. Core Components
   13 config files trong config/ (đã hiểu hết params)
   Entities: UAV (5 states), Victim (Injured/Mobile), Obstacles (Debris/DangerZone)
   Sensors: FOVSensor (detection prob), CommSensor (30m range)
   Rewards: BaselineReward v3.1 với 16 components
   Backend: LogicBackend (~1000 steps/s)
3. Training Infrastructure
   MAPPO: Hoàn chỉnh với ActorNetwork(68→4), CriticNetwork(554→1)
   Buffer: GAE-based RolloutBuffer, action_dim=4
   VectorizedEnv: Multi-process với seed progression per-episode
   Fixes đã apply: BUG-5, FIX-T1/T2/T3, BUG-ENV-06
   ⚠️ Known Issues đã ghi nhận:
   EpisodeLogger.finalize() - 2 definitions, landing fields bị shadow (Medium)
   orthogonal_init - commented out trong MLP (Low)
   state_onehot - dùng [:4] thay vì [:5] (Low)
   UAV spawn - tất cả ACTIVE ban đầu (Low)
   📦 Checklist Phase 3 (MASAC/MATD3):
   training/algorithms/masac/ (actor, twin_critic, replay_buffer, trainer)
   training/algorithms/matd3/ (deterministic actor, twin_critic, replay_buffer, trainer)
   train_masac.py + train_matd3.py CLI
   Cùng config: action_dim=4, HARD stage, BaselineReward v3.1, 5 seeds
   Statistical comparison với Wilcoxon test
   ✅ SẴN SÀNG CHO TASK TIẾP THEO
   Đã hiểu đầy đủ:

Execution flow (single step + training loop)
Data pipeline (observation → action → reward → update)
Tất cả hyperparameters và normalization schemes
Bug history và fixes đã apply
Chờ lệnh implement MASAC/MATD3 hoặc task khác.

đọc và hiểu ko cần làm gì
🚁 SAR UAV SWARM — PROJECT STATE SNAPSHOT v4.0
Dựa trên code thực tế | MAPPO Phase 2 Complete | HARD Stage Fixed | No Curriculum

MỤC TIÊU DỰ ÁN
Task: 4 UAV tự động phối hợp tìm kiếm nạn nhân trong khu vực thảm họa 250×250m.
Research Plan (Paper 1):

Phase 2 (Hiện tại): Train MAPPO trên HARD stage cố định, 5 seeds × 3000 eps/seed
Phase 3 (Tiếp theo): Train MASAC + MATD3 cùng cấu hình → so sánh thống kê (Wilcoxon test)
Paper 2: LLM-generated reward vs BaselineReward v3.1 (hand-crafted)

Không dùng curriculum learning. curriculum_manager=None trong trainer.train().

THÔNG SỐ HỆ THỐNG (Verified từ code)
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

[vx, vy, vz] ~ Normal(mean[3], exp(log_std)[3]) — Gaussian continuous
[land] ~ Bernoulli(sigmoid(logit)) — discrete {0,1}
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
│ ├── init.py
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
│ ├── init.py
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

post_init() — Auto-sync obs.n_stations = env.n_stations
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

Battery penalties (per step)
r_battery_20: float = -5.0 → battery ≤ 20%
r_battery_10: float = -20.0 → battery ≤ 10%
r_battery_5: float = -50.0 → battery ≤ 30% (emergency_pct)
r_battery_dead: float = -200.0 → one-time khi battery = 0

r_collision_obstacle: float = -35.0 → one-time per UAV

Proximity
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
n_stations: int = None → auto-sync từ env.n_stations trong AppConfig.post_init
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

Safety
collision_events: List[Dict] → [{step, uav_id, obstacle_id, type, pos, height}]
collision_obstacle: int
collision_uav: int
collision_proximity: int
battery_deaths: int
danger_zone_entries: int
hot_swaps: int
events: Dict[str, int]

Landing tracking (mới)
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

Tier 3: One-time khi chuyển CHARGING (per episode per UAV)
if uav.state == CHARGING and uav.id not in \_landed_uavs:
landing_total += 120

Tier 1: battery ≤ 40% AND (ACTIVE or RETURNING) → approach station
approach_rew = 0.3 × (1.0 - min_dist / max_dist)

Tier 2: ACTIVE + trong landing_range (charge_radius×2) + battery ≤ 40% nhưng không land
hover_total += -3.0
Terminal bonus formula:

Python

coverage_bonus = terminal_cap × 0.60 × coverage_rate
victim_bonus = terminal_cap × 0.20 × found_ratio
time_bonus = terminal_cap × 0.10 × (1-time_ratio) # chỉ khi cov ≥ 80%
battery_bonus = terminal_cap × 0.10 × mean_battery/100

clipped to [0, terminal_cap=100]
Total terminal = 200 + bonus (max 300)
Delta shaping:

Python

Memory: \_prev_min_dist[uav_id]
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

action[4] = [vx, vy, vz, land]
move_action = action[:3]
land_signal = float(action[3])

ACTIVE UAV landing conditions (ALL must hold):
if land_signal > 0.5 AND battery ≤ 40.0:
nearest = find_station_in_range(charge_radius × 2.0 = 6m)
if nearest → set RETURNING + auto_navigate(target z=0.0)
else → apply_action(move_action) # fallback
else:
→ apply_action(move_action)

RETURNING → auto_navigate(target_station, z=0)
DEPLOYING → set_state(ACTIVE) ngay lập tức
CHARGING/DISABLED → no movement
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

apply_actions(actions)
step_physics()
step_world()
step_count += 1
\_check_done() ← TRƯỚC reward (BUG-ENV-06 fix)
compute_per_uav() ← Per-agent rewards
compute() ← Global reward (logging only)
\_log_step()
\_build_obs_dict()
return
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

Initial reset với seed gốc
env.reset(seed=seed)

Auto reset với seed MỚI sau mỗi episode
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

Movement: Normal(move_mean, exp(log_std)) → sample → clamp [-1,1]
Landing: Bernoulli(logits=land_logit) → sample {0.0, 1.0}
log_prob = move_log_prob + land_log_prob
action = cat([move_action, land_action], dim=-1) → [batch, 4]
evaluate_actions(obs[batch,68], actions[batch,4]) → (log_prob[batch], entropy[batch]):

Python

Split: actions[:,:3] = move, actions[:,3:] = land
move: Normal log_prob + entropy
land: Bernoulli log_prob + entropy
return sum
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
MAPPOTrainer.init(config, device, run_name, n_envs):

text

actor: ActorNetwork(obs_dim=68, action_dim=4, hidden=(256,256))
critic: CriticNetwork(global_obs_dim=554, hidden=(512,256))
buffer: RolloutBuffer(capacity=rollout_length×n_envs, action_dim=4)
actor_opt: Adam(lr=3e-4)
critic_opt: Adam(lr=1e-3)

ep_rewards, ep_lengths, ep_coverage, ep_victims: deque(maxlen=100)
total_episodes_done, total_steps, update_count: int

Trigger tracking (FIX-T3)
\_next_log_ep, \_next_viz_ep, \_next_checkpoint_ep: int

Dirs
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

text

# Step all envs

next_obs, next_g, rews, dones, infos = env.step(act_batch)

# Per-env tracking

for ei in range(n_envs):
buffer.add(...)
if dones[ei]: # Log landing stats từ infos[ei]['episode']
pbar.update(1)
check max_episodes
GAE bootstrap
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

text

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

TRAINING FLOW (Tổng thể)
text
train_mappo.py
├── AppConfig + cfg.apply_stage(STAGE_HARD)
├── auto_compute_config() → rollout_length, batch_size
└── MAPPOTrainer(config, device, n_envs)
├── ActorNetwork(68 → 4) ~params: backbone+heads
├── CriticNetwork(554 → 1) ~params: MLP(512,256)
└── RolloutBuffer(capacity=rollout_len×n_envs, action_dim=4)

text

\_EnvWrapper (n_envs=1→PettingZoo, n_envs>1→VectorizedEnv)
env.reset() [1 lần duy nhất]

WHILE episodes < total_episodes:
├── \_rollout():
│ ├── get_current_obs() [no reset, BUG-5 fix]
│ ├── FOR step in rollout_length:
│ │ ├── actor.get_action(obs[N×4, 68]) → action[N×4, 4], log_prob[N×4]
│ │ ├── critic.get_value(global_obs[N, 554]) → value[N]
│ │ ├── env.step(actions[N, 4, 4])
│ │ ├── buffer.add(...)
│ │ └── IF done: log + pbar.update(1) + check early stop
│ └── buffer.compute_gae(bootstrap, last_done)
│
├── \_update():
│ ├── n_epochs=10 × minibatches(batch_size)
│ ├── Actor: PPO clip + entropy bonus
│ ├── Critic: MSE loss
│ └── buffer.clear()
│
├── Logging (FIX-T1: >= trigger)
├── Viz snapshot (ep_xxx.png)
└── Checkpoint save (absolute path, FIX-T2)

FINALIZE:
├── save_checkpoint(tag="final")
├── plot_training_curves()
└── env.close()
SINGLE STEP FLOW
text
env.step(actions: Dict[str, ndarray(3)]) ← PettingZoo API (str keys, 3-dim)
↓ Convert str→int keys + pad to 4-dim
LogicBackend.applyactions(Dict[int, ndarray(4)]):
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
\_check_done() → "coverage"/"victims"/"disabled:_"/None
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
FIX-T3: \_next_\_ep tracking
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

Single env, 3000 eps, HARD stage
python train_mappo.py --total-episodes 3000 --n-envs 1 --seed 42

Vectorized (6 envs), 2× faster
python train_mappo.py --total-episodes 3000 --n-envs 6 --seed 42

Quick test
python train_mappo.py --total-episodes 50 --n-envs 1

Multi-seed training (5 seeds)
for seed in 42 123 456 789 1011; do
python train_mappo.py --total-episodes 3000 --seed
s
e
e
d

−
−
r
u
n
−
n
a
m
e
"
h
a
r
d
m
a
p
p
o
s
seed −−run−name"hard
m
​
appo
s
​
{seed}" --n-envs 6
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

Ask followup…

Add files

Inputs are processed by third-party AI and responses may be inaccurate.
