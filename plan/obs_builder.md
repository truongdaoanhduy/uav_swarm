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
Extract global_obs từ info['uav_0']['global_obs']
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
Buffer size khi n_envs>1: rollout_length _ n_envs
training/algorithms/mappo/trainer.py — MAPPOTrainer
Vai trò: Main MAPPO training loop
**init**(config, device, run_name, n_envs=1):
Tạo ActorNetwork, CriticNetwork, RolloutBuffer
Tạo Adam optimizers cho actor/critic
Setup output dirs: results/mappo/{run_name}/checkpoints/, viz/, plots/
Buffer size = rollout_length _ n_envs
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
