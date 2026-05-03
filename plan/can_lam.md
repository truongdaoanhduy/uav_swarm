SAR UAV SWARM - KẾ HOẠCH NGHIÊN CỨU
TỔNG QUAN
text

QUY TRÌNH NGHIÊN CỨU:

PHASE 1: XÂY DỰNG MÔI TRƯỜNG
└── PettingZoo + Logic Python + PyBullet

PHASE 2: SO SÁNH THUẬT TOÁN (PyBullet)
├── MAPPO + CTDE
├── MASAC + CTDE
└── MATD3 + CTDE
→ Chọn Winner

PHASE 3: SO SÁNH REWARD (PyBullet)
├── Winner + Hand-crafted Reward
└── Winner + LLM Reward
→ Chứng minh LLM có ích

PHASE 4: MIGRATION
└── Best Setup → IsaacLab
    (Hoàn thành tất cả trước khi lên IsaacLab)

FRAMEWORK: Dec-POMDP + CTDE
├── Actor: Local obs only (52 dims)
├── Critic: Global state (training only)
└── CTDE nhất quán trong tất cả experiments
BỐI CẢNH & Ý TƯỞNG
text

KỊCH BẢN:
├── Động đất xảy ra tại khu vực 100×100m
├── N UAVs (input bất kỳ) tìm kiếm nạn nhân
├── Không biết tổng số nạn nhân
├── Hoạt động 24/7 liên tục
└── Mục tiêu: Tìm nạn nhân + Coverage ≥ 90%

THÁCH THỨC:
├── Partial Observability (Dec-POMDP)
├── Variable Fleet Size
├── Energy Management 24/7
├── Unknown Number of Victims
└── Multi-Agent Coordination

GIẢI PHÁP:
├── CTDE: Critic thấy global state khi training
├── Actor: Chỉ dùng local obs khi execution
├── Fleet Manager: Rule-based energy cycling
└── Soft Constraints: UAV tự học battery mgmt
PHASE 1 - XÂY DỰNG MÔI TRƯỜNG
Checklist Phase 1
text

□ Nhóm 1: Config
□ Nhóm 2: Entities
□ Nhóm 3: Core Systems
□ Nhóm 4: Sensors
□ Nhóm 5: Observation Builder
□ Nhóm 6: Reward
□ Nhóm 7: Environment
□ Nhóm 8: Curriculum
□ Nhóm 9: Utils & Visualization
Nhóm 1: Config
text

FILE: config.py

□ MAP CONFIG:
  ├── MAP_SIZE = 100
  ├── GRID_SIZE = 100
  ├── DT = 1.0
  └── MAX_STEPS = 1000

□ UAV CONFIG:
  ├── N_TOTAL = input bất kỳ (không random)
  ├── Z_MIN = 3.0, Z_MAX = 40.0
  ├── MAX_SPEED_XY = 5.0
  ├── MAX_SPEED_Z = 2.0
  └── COLLISION_RADIUS = 0.5

□ BATTERY CONFIG:
  ├── DRAIN_XY_MAX = 0.10%
  ├── DRAIN_Z_UP_MAX = 0.15%
  ├── DRAIN_Z_DOWN_MAX = 0.03%
  ├── DRAIN_IDLE = 0.05%
  ├── CHARGE_RATE = 1.5%/step
  ├── THRESHOLD_LOW = 20%
  └── THRESHOLD_READY = 80%

□ FLEET CONFIG:
  ├── N_RESERVE = max(2, ceil(N_TOTAL × 0.2))
  ├── HOT_THRESHOLD = 90%
  ├── WARM_THRESHOLD = 80%
  ├── COLD_THRESHOLD = 60%
  └── HOT_SWAP_TRIGGER = 25%

□ SENSOR CONFIG:
  ├── COMM_RANGE = 30.0
  ├── N_TRACKED_UAVS = 3
  ├── HFOV = 90°
  └── FOV_TAN = 1.0

□ VICTIM CONFIG:
  ├── N_VICTIMS_MIN = 5
  ├── N_VICTIMS_MAX = 20
  ├── INJURED_RATIO = [0.4, 0.7]
  ├── INJURED_URGENCY = [4, 5]
  ├── MOBILE_URGENCY = [1, 3]
  └── MOBILE_SPEED = [0.2, 0.4]

□ OBSTACLE CONFIG:
  ├── N_DEBRIS = 6
  ├── N_DANGER = 2
  ├── DEBRIS_WIDTH = [2, 5]
  ├── DEBRIS_HEIGHT = [3, 8]
  ├── DANGER_WIDTH = [3, 8]
  └── DANGER_HEIGHT = [5, 12]

□ STATION CONFIG:
  ├── N_STATIONS = 2
  ├── CHARGE_RADIUS = 3.0
  └── MIN_STATION_SPACING = 15.0

□ REWARD CONFIG:
  ├── R_VICTIM_FOUND = 50.0
  ├── R_COVERAGE_DELTA = 5.0
  ├── R_BATTERY_20 = -2.0
  ├── R_BATTERY_10 = -5.0
  ├── R_BATTERY_5 = -10.0
  ├── R_BATTERY_DEAD = -200.0
  ├── R_OBSTACLE = -50.0
  ├── R_DANGER = -12.0
  ├── R_PROXIMITY_1M = -50.0
  ├── R_PROXIMITY_2M = -10.0
  ├── R_PROXIMITY_3M = -2.0
  ├── R_TIME = -0.1
  └── R_TERMINAL = 200.0

□ CURRICULUM CONFIG:
  ├── PHASE_1A_EPISODES = 500
  ├── PHASE_1B_EPISODES = 1000
  └── PHASE_1C_EPISODES = 2000
Nhóm 2: Entities
text

FILE: entities/uav.py

□ ENUM UAVState:
  ├── ACTIVE
  ├── RETURNING
  ├── CHARGING
  └── DEPLOYING

□ CLASS UAV:
  ATTRIBUTES:
  ├── id, pos, vel, battery
  ├── state: UAVState
  ├── last_heading: float
  ├── target_station: ChargingStation
  └── pybullet_body_id: int

  METHODS:
  □ apply_action(action):
    ├── Clip action [-1, 1]
    ├── Scale vx,vy × 5.0, vz × 2.0
    ├── Cap diagonal speed ≤ 5.0
    ├── Update pos = pos + vel × dt
    ├── Clip altitude [3, 40]
    ├── Clip boundary [0, 100]
    └── Update last_heading

  □ update_battery(stations):
    ├── drain_xy = 0.10% × (speed_xy/5.0)
    ├── drain_z_up = 0.15% × (vz/2.0)
    ├── drain_z_down = 0.03% × (|vz|/2.0)
    ├── drain_idle = 0.05%
    ├── If near station: charge += 1.5%
    └── battery = clip(battery, 0, 100)

  □ get_battery_penalty():
    ├── battery ≤ 5%:  return -10.0
    ├── battery ≤ 10%: return -5.0
    ├── battery ≤ 20%: return -2.0
    └── return 0

  □ get_fov_radius():
    └── return pos[2]

  □ auto_navigate(target_pos):
    ├── Tính vector đến target
    ├── Normalize và scale
    └── Update vel

  □ to_dict()

---

FILE: entities/victim.py

□ CLASS InjuredVictim:
  ├── pos: fixed (không di chuyển)
  ├── urgency: random [4, 5]
  ├── is_found: bool
  └── step(): pass

□ CLASS MobileVictim:
  ├── pos: thay đổi mỗi step
  ├── urgency: random [1, 3]
  ├── speed: random [0.2, 0.4]
  ├── direction: random [0, 2π]
  ├── move_timer: int
  └── step(obstacles):
      ├── move_timer += 1
      ├── if timer ≥ 20: random direction mới
      ├── new_pos = pos + [cos,sin] × speed × dt
      └── clip boundary [2, 98]

---

FILE: entities/obstacle.py

□ CLASS Debris(Obstacle):
  ├── pos, width [2,5], height_3d [3,8]
  ├── penalty = -1.5/step
  ├── in_zone(pos_2d): dist ≤ width/2
  └── causes_collision(uav_pos):
      ├── uav.z < height_3d
      └── dist_2d < width/2

□ CLASS DangerZone(Obstacle):
  ├── pos, width [3,8], height_3d [5,12]
  └── penalty = -12.0/step

---

FILE: entities/charging_station.py

□ CLASS ChargingStation:
  ├── id, pos
  ├── charge_radius = 3.0
  ├── current_occupant: UAV | None
  ├── is_occupied()
  ├── in_range(uav_pos)
  ├── try_occupy(uav)
  └── release(uav)
Nhóm 3: Core Systems
text

FILE: core/coverage_map.py

□ CLASS CoverageMap:
  ├── grid: np.zeros([100, 100])
  ├── reset()
  ├── mark_explored(uav_pos, fov_radius):
  │   └── Dùng get_circle_cells() mark = 1
  ├── get_coverage_rate():
  │   └── sum(grid) / total_cells
  ├── get_local_coverage(pos, radius=15)
  └── get_nearest_unexplored(pos)

---

FILE: core/map_generator.py

□ CLASS MapGenerator:
  □ generate() → dict:
    ├── N_VICTIMS = random [5, 20]
    ├── stations = place_stations()
    ├── obstacles = place_obstacles()
    ├── victims = spawn_victims(obstacles)
    └── return all entities

  □ place_stations():
    ├── 2 stations
    ├── Cách biên ≥ 5m
    └── Cách nhau ≥ 15m

  □ place_obstacles():
    ├── 6 Debris: random pos, width, height
    ├── 2 DangerZone: random pos, width, height
    ├── Không overlap nhau
    └── Cách station ≥ 5m

  □ spawn_victims(obstacles):
    ├── n_injured = round(N × ratio [0.4, 0.7])
    ├── Injured: 80% gần debris (≤ 5m)
    ├── Mobile: 40% gần debris, 60% random
    └── Tất cả: cách obstacle ≥ 2m, biên ≥ 2m

  □ get_uav_spawns(N_TOTAL):
    └── Offset từ stations, altitude = 5m

---

FILE: core/fleet_manager.py

□ CLASS FleetManager:
  ATTRIBUTES:
  ├── all_uavs: list[UAV]
  ├── stations: list[Station]
  ├── n_total, n_reserve
  ├── active_swaps: dict
  └── energy_tracker

  □ reset(n_total):
    ├── n_reserve = max(2, ceil(n_total × 0.2))
    ├── n_max_active = n_total - n_reserve
    └── n_min_active = min(2, n_max_active)

  □ count_by_tier():
    ├── hot = count(battery > 90, state=CHARGING)
    ├── warm = count(80 < battery ≤ 90, CHARGING)
    └── cold = count(60 < battery ≤ 80, CHARGING)

  □ compute_urgency(coverage_rate, time_remaining):
    └── 0.6×(1-coverage) + 0.4×(1-time/MAX)

  □ compute_target_active(urgency, n_ready):
    ├── n_deployable = max(0, n_ready - n_reserve)
    └── target = n_min + round(urgency × n_deployable)

  □ check_hot_swap(active_uavs):
    ├── For each uav: if battery ≤ 25%
    ├── Find replacement (pin cao nhất, hot tier)
    ├── If arrival_time < critical_time: deploy
    └── Handover khi distance < 20m

  □ staggered_deploy(n_needed):
    ├── Sort candidates by battery desc
    └── Deploy lệch battery: 80, 85, 90, 95, 100%

  □ get_fleet_incentives(n_active, target, n_ready):
    ├── deploy_incentive = +3.0 if n_active < target
    └── recall_pressure = -2.0 × (reserve - n_ready)

  □ step(coverage_rate, time_remaining):
    ├── Update energy tracking
    ├── Check hot swap triggers
    ├── Calculate urgency + target
    ├── Deploy if needed (hot first)
    └── Return incentives
Nhóm 4: Sensors
text

FILE: sensors/fov_sensor.py

□ CLASS FOVSensor:
  □ calculate_fov_radius(altitude):
    └── return altitude × FOV_TAN (=1.0)

  □ calculate_detection_prob(altitude):
    └── return 0.9 × exp(-0.05 × altitude)

  □ scan_victims(uav, victims):
    ├── fov_radius = calculate_fov_radius(uav.altitude)
    ├── Lọc: dist_2d ≤ fov_radius
    ├── Apply P_detect per victim
    ├── Sort by distance, lấy 5 gần nhất
    └── Return normalized [rel_x, rel_y, dist, urgency, found]

  □ scan_obstacles(uav, obstacles):
    ├── Lọc trong FOV
    ├── Sort, lấy 4 gần nhất
    └── Return [rel_x, rel_y]

  □ check_detected(uav, victim):
    ├── dist ≤ fov_radius?
    └── random() < P_detect?

---

FILE: sensors/comm_sensor.py

□ CLASS CommSensor:
  □ scan(ego_uav, all_active_uavs):
    ├── Lọc UAVs khác trong 30m
    ├── Sort by dist_3d
    ├── Lấy 3 gần nhất
    ├── Per UAV:
    │   ├── dist = dist_3d / 30
    │   ├── bearing = compute_bearing(...) / (2π)
    │   └── rel_alt = (other.z - ego.z) / 40
    └── Pad [1.0, 0, 0] nếu < 3

  □ get_n_in_range(ego_uav, all_uavs):
    └── count UAVs trong 30m
Nhóm 5: Observation Builder
text

FILE: observation/obs_builder.py

□ CLASS ObservationBuilder:

  □ build_actor_obs(uav, all_uavs, victims,
                    obstacles, stations,
                    coverage_map, step) → np.array(52):

    □ Part 1 - Self State (9 dims):
      [0] pos_x / 100
      [1] pos_y / 100
      [2] altitude / 40
      [3] vel_x / 5.0
      [4] vel_y / 5.0
      [5] vel_z / 2.0
      [6] battery / 100
      [7] is_charging {0,1}
      [8] fov_radius / 40

    □ Part 2 - Stations (8 dims):
      2 stations × [rel_x/100, rel_y/100,
                    dist/141, is_occupied]

    □ Part 3 - Local Teammates (9 dims):
      [0] n_in_comm_range / 10
      3 UAVs × [dist/30, bearing/2π, rel_alt/40]
      Pad [1.0, 0, 0] nếu thiếu

    □ Part 4 - Obstacles FOV (8 dims):
      4 obstacles × [rel_x/100, rel_y/100]
      Pad [1.0, 1.0] nếu thiếu

    □ Part 5 - Victims FOV (15 dims):
      5 victims × [rel_x/100, rel_y/100,
                   dist/fov_r, urgency/5, is_found]
      Pad [1,1,1,0,0] nếu thiếu

    □ Part 6 - Local Info (3 dims):
      [0] local_coverage_15m
      [1] local_coverage_30m
      [2] time_remaining / MAX_STEPS

    └── return np.concatenate(all_parts) → (52,)

  □ build_critic_obs(all_uavs, global_state,
                     all_actor_obs) → np.array:
    ├── Tất cả actor obs: N × 52
    ├── n_active / N_TOTAL
    ├── n_ready / N_TOTAL
    ├── global_coverage_rate
    ├── victims_found_total / 20
    ├── fleet_urgency
    └── hot_reserve_count / N_TOTAL

  □ VALIDATION:
    ├── Assert shape == (52,)
    ├── Assert all values ∈ [-1, 1] hoặc [0, 1]
    └── Assert no NaN/Inf
Nhóm 6: Reward
text

FILE: rewards/baseline_reward.py

□ CLASS BaselineReward:

  □ victim_found_reward(newly_found):
    └── sum(50 × v.urgency/5 for v in newly_found)

  □ coverage_delta_reward(prev, curr):
    └── 5.0 × (curr - prev)

  □ battery_penalty(uav):
    └── uav.get_battery_penalty()

  □ collision_penalty(uav, obstacles):
    ├── Obstacle collision: -50
    ├── DangerZone: -12/step
    └── return total penalty

  □ proximity_penalty(uav, all_uavs):
    ├── dist < 1m: -50/step
    ├── dist < 2m: -10/step
    ├── dist < 3m: -2/step
    └── return worst penalty found

  □ time_penalty():
    └── return -0.1

  □ terminal_bonus(coverage_rate, time_remaining):
    └── 200 × (coverage/0.9) × (time_rem/MAX)

  □ fleet_incentives(deploy_inc, recall_pres):
    └── deploy_inc + recall_pres

  □ compute(uav, all_uavs, victims, obstacles,
             coverage_map, fleet_manager,
             prev_coverage, newly_found,
             step, done) → float:
    ├── r = sum(R1 đến R7)
    ├── if done: r += terminal_bonus()
    └── return r

---

FILE: rewards/llm_reward.py (Phase 3)

□ CLASS LLMReward:
  □ aggregate_metrics(episode_buffer):
    └── Tính mean/std của tất cả metrics

  □ build_llm_context(metrics, env_info):
    ├── Environment description
    ├── Current metrics
    ├── N_victims avg, N_uavs
    └── Failure analysis

  □ query_llm(context) → str:
    └── Call GPT-4o API

  □ parse_and_validate(code_str) → bool:
    ├── Syntax check
    ├── Execute với dummy data
    └── Check output type/range

  □ update_reward_function(new_code):
    ├── If valid: replace current
    └── If invalid: keep previous
Nhóm 7: Environment
text

FILE: env/base_env.py

□ Abstract class BaseEnv(pettingzoo.ParallelEnv):
  □ reset() → abstract
  □ step(actions) → abstract
  □ observation_space → abstract
  □ action_space → abstract

---

FILE: env/pybullet/pybullet_bridge.py

□ CLASS PyBulletBridge:
  □ setup(gui=False):
    ├── p.connect(GUI hoặc DIRECT)
    ├── p.loadURDF("plane.urdf")
    └── Set gravity, timestep

  □ load_uav_body(pos) → body_id:
    └── p.createMultiBody(box shape)

  □ load_obstacle_body(obs) → body_id:
    └── p.createMultiBody(box shape)

  □ sync_positions(all_uavs):
    └── p.resetBasePositionAndOrientation per UAV

  □ step_simulation():
    └── p.stepSimulation()

  □ close():
    └── p.disconnect()

---

FILE: env/pybullet/sar_env_pybullet.py

□ CLASS SAREnvPyBullet(BaseEnv):

  □ reset() → dict[agent_id, obs]:
    ├── map_data = map_generator.generate()
    ├── Init all entities
    ├── fleet_manager.reset(N_TOTAL)
    ├── coverage_map.reset()
    ├── pybullet_bridge.setup()
    ├── Load all bodies vào PyBullet
    ├── current_step = 0
    └── return build_actor_obs() per active UAV

  □ step(actions) → obs, rewards, dones, infos:
    ├── [1] Apply actions to ACTIVE UAVs
    ├── [2] Update battery all UAVs
    ├── [3] Handle RETURNING UAVs (auto-navigate)
    ├── [4] Update victims (Mobile step)
    ├── [5] Check victim detection
    │       newly_found = []
    │       for uav in active: for victim in victims:
    │           if check_detected(): mark + append
    ├── [6] Update coverage map
    ├── [7] Fleet manager step
    │       → get incentives
    ├── [8] Compute rewards per UAV
    ├── [9] Build actor obs per UAV
    ├── [10] Check terminal:
    │        done = step ≥ MAX or coverage ≥ 0.9
    ├── [11] Sync PyBullet
    ├── [12] current_step += 1
    └── return obs, rewards, dones, infos

  □ Action masking:
    └── Inactive UAVs nhận zero action, zero reward
Nhóm 8: Curriculum
text

FILE: training/curriculum.py

□ CLASS CurriculumScheduler:

  □ get_phase(episode) → str:
    ├── episode < 500:  return "1a"
    ├── episode < 1500: return "1b"
    └── return "1c"

  □ get_config(phase) → dict:
    ├── "1a": N_VICTIMS = 10 (fixed)
    ├── "1b": N_VICTIMS = random [5, 15]
    └── "1c": N_VICTIMS = random [5, 20]

  NOTE: N_TOTAL luôn fixed (input từ user)
Nhóm 9: Utils & Visualization
text

FILE: utils/geometry.py

□ dist_2d(p1, p2) → float
□ dist_3d(p1, p2) → float
□ compute_bearing(from_pos, from_vel, to_pos) → float:
  ├── angle_to_target = atan2(dy, dx)
  ├── heading = atan2(vy, vx)
  └── return normalize(angle_to_target - heading)
□ compute_heading(vel) → float
□ normalize_angle(angle) → [-π, π]
□ check_los_2d(p1, p2, obstacles) → bool
□ get_circle_cells(center, radius, grid_size) → list
□ clip_position(pos, bounds) → pos

---

FILE: utils/logger.py

□ CLASS EpisodeLogger:
  ├── log_step(step, uav_id, obs, action, reward)
  ├── log_episode(total_reward, metrics_dict)
  └── save(filepath)

□ CLASS MetricsAggregator:
  ├── add_episode(metrics_dict)
  ├── get_stats(key, window=100) → mean, std
  └── save_csv(filepath)

□ CLASS RewardTracker:
  ├── track(component_name, value)
  ├── get_breakdown() → dict
  └── plot_components()

---

FILE: visualization/visualizer_2d.py

□ CLASS Visualizer2D:
  □ plot_map(env_state):
    ├── Map 100×100 background
    ├── Coverage heatmap overlay
    ├── Obstacles (circles + height color)
    ├── Stations (green squares)
    ├── Victims (red=injured, orange=mobile)
    └── UAVs (blue dots + FOV circles)

  □ plot_fleet_status(fleet_manager):
    ├── Pie chart: active/charging/returning
    └── Battery bars per UAV

  □ save_frame(filepath)
  □ make_video(frames, filepath)

---

FILE: visualization/comparison_plots.py

□ plot_learning_curves(results_dict):
  ├── X: Episodes, Y: Average Reward
  ├── Lines: MAPPO vs MASAC vs MATD3
  └── Shading: ± Std

□ plot_coverage_curves(results_dict):
  ├── X: Episodes, Y: Coverage Rate (%)
  └── Lines: 3 algorithms

□ plot_final_performance_bar(results_dict):
  ├── Grouped bars: Coverage + Victims Found
  ├── X: Algorithms
  └── Error bars: ± Std

□ plot_stability_boxplot(results_dict):
  ├── X: Algorithms
  └── Y: Episode Reward distribution

□ plot_radar_chart(results_dict):
  ├── Dimensions:
  │   ├── Coverage Rate
  │   ├── Victims Found
  │   ├── Safety (1 - collision_rate)
  │   ├── Efficiency (1 - battery_deaths)
  │   └── Convergence Speed
  └── Lines: 3 algorithms

□ plot_convergence_time(results_dict):
  ├── X: Algorithms
  └── Y: Episodes to stable performance
PHASE 2 - SO SÁNH THUẬT TOÁN
Checklist Phase 2
text

□ Implement MAPPO + CTDE
□ Implement MASAC + CTDE
□ Implement MATD3 + CTDE
□ Train 3 algorithms × 3000 episodes
□ Thu đủ metrics
□ Vẽ 6 biểu đồ so sánh
□ Chọn Winner
Algorithm 1: MAPPO + CTDE
text

FILE: training/algorithms/mappo/actor.py

□ CLASS MAPPOActor(nn.Module):
  ├── Input: 52 dims local obs
  ├── Architecture: MLP hoặc LSTM
  │   MLP: Linear(52→256) → ReLU → Linear(256→128)
  │        → ReLU → Linear(128→3) → Tanh
  │   LSTM: LSTM(52→256) → Linear(256→3) → Tanh
  ├── Output: [vx, vy, vz] ∈ [-1, 1]
  └── Shared weights giữa tất cả agents

FILE: training/algorithms/mappo/critic.py

□ CLASS MAPPOCritic(nn.Module):
  ├── Input: Global state (critic obs)
  ├── Architecture: MLP lớn hơn actor
  │   Linear(critic_dim→512) → ReLU
  │   → Linear(512→256) → ReLU
  │   → Linear(256→1)
  └── Output: V(s) scalar

FILE: training/algorithms/mappo/mappo_trainer.py

□ CLASS MAPPOTrainer:
  □ collect_rollouts(env, n_steps):
    ├── Run policy, collect transitions
    └── Store (obs, action, reward, done, value)

  □ compute_gae(rewards, values, dones):
    └── GAE advantage estimation (γ=0.99, λ=0.95)

  □ update(rollout_buffer):
    ├── For each minibatch:
    │   ├── Compute new log_probs
    │   ├── ratio = exp(new_log - old_log)
    │   ├── L_clip = min(ratio×A, clip(ratio,0.8,1.2)×A)
    │   ├── L_value = (V - V_target)²
    │   └── L_entropy = -entropy
    ├── Loss = -L_clip + 0.5×L_value - 0.01×L_entropy
    └── Backward + clip gradients + step

  □ HYPERPARAMETERS:
    ├── lr_actor = 3e-4
    ├── lr_critic = 1e-3
    ├── gamma = 0.99
    ├── gae_lambda = 0.95
    ├── clip_epsilon = 0.2
    ├── entropy_coef = 0.01
    ├── value_coef = 0.5
    └── n_epochs = 10
Algorithm 2: MASAC + CTDE
text

FILE: training/algorithms/masac/actor.py

□ CLASS MASACActor(nn.Module):
  ├── Input: 52 dims local obs
  ├── Output: mean + log_std
  └── Action = mean + std × noise (reparameterize)

FILE: training/algorithms/masac/critic.py

□ CLASS MASACCritic(nn.Module):
  ├── Twin Q-networks (Q1, Q2)
  ├── Input: Global state + all actions
  └── Output: Q-value (lấy min của 2)

FILE: training/algorithms/masac/masac_trainer.py

□ CLASS MASACTrainer:
  □ replay_buffer: size = 1,000,000

  □ update():
    ├── Sample batch từ buffer
    ├── Compute target Q:
    │   ├── next_action, log_prob = actor(next_obs)
    │   ├── target_Q = r + γ × (min(Q1,Q2) - α×log_prob)
    ├── Update critics: L = (Q - target_Q)²
    ├── Update actor: L = α×log_prob - min(Q1,Q2)
    ├── Update temperature α
    └── Soft update target networks (τ=0.005)

  □ HYPERPARAMETERS:
    ├── lr = 3e-4
    ├── gamma = 0.99
    ├── tau = 0.005
    ├── alpha_init = 0.2 (auto-tuned)
    └── batch_size = 256
Algorithm 3: MATD3 + CTDE
text

FILE: training/algorithms/matd3/actor.py

□ CLASS MATD3Actor(nn.Module):
  ├── Deterministic policy
  ├── Input: 52 dims local obs
  └── Output: [vx, vy, vz] (Tanh)

FILE: training/algorithms/matd3/critic.py

□ CLASS MATD3Critic(nn.Module):
  ├── Twin Q-networks
  ├── Input: Global state + all actions
  └── Output: Q-value

FILE: training/algorithms/matd3/matd3_trainer.py

□ CLASS MATD3Trainer:
  □ replay_buffer: size = 1,000,000

  □ update():
    ├── Sample batch
    ├── Target policy smoothing:
    │   noise = clip(N(0, 0.2), -0.5, 0.5)
    │   next_action = actor_target(obs) + noise
    ├── Compute target Q (min of twin)
    ├── Update critics
    ├── If step % 2 == 0: (delayed update)
    │   ├── Update actor
    │   └── Soft update target networks
    └── τ = 0.005

  □ HYPERPARAMETERS:
    ├── lr_actor = 3e-4
    ├── lr_critic = 3e-4
    ├── gamma = 0.99
    ├── tau = 0.005
    ├── policy_noise = 0.2
    ├── noise_clip = 0.5
    ├── policy_delay = 2
    └── batch_size = 256
Training & Evaluation Scripts
text

FILE: training/train_comparison.py

□ Setup:
  ├── env = SAREnvPyBullet(N_TOTAL=config.N_TOTAL)
  ├── curriculum = CurriculumScheduler()
  ├── algorithms = {MAPPO, MASAC, MATD3}
  └── results = {}

□ For each algorithm:
  ├── Reset algorithm
  ├── For episode in range(3000):
  │   ├── phase = curriculum.get_phase(episode)
  │   ├── env_config = curriculum.get_config(phase)
  │   ├── obs = env.reset(env_config)
  │   ├── Run episode
  │   ├── algorithm.update()
  │   ├── Log metrics
  │   └── Save checkpoint mỗi 100 eps
  └── results[algo_name] = metrics

□ Save all results → JSON

---

FILE: training/eval.py

□ evaluate_policy(algo, n_episodes=100):
  ├── Run deterministic policy
  ├── Collect all metrics
  └── Return aggregated stats

□ compare_algorithms(results_dict):
  ├── Print comparison table
  └── Call comparison_plots

□ METRICS TABLE:
  ┌──────────────────────────────────────────────┐
  │ Metric      │ MAPPO  │ MASAC  │ MATD3       │
  ├──────────────────────────────────────────────┤
  │ Coverage    │        │        │             │
  │ Victims     │        │        │             │
  │ Success Rate│        │        │             │
  │ Conv. Speed │        │        │             │
  │ Battery Dead│        │        │             │
  │ Collisions  │        │        │             │
  └──────────────────────────────────────────────┘
Metrics cần thu thập
text

PER EPISODE:
□ total_reward
□ coverage_rate (%)
□ victims_found_count
□ victims_found_rate (%)
□ battery_deaths_count
□ collision_count
□ episode_length
□ n_active_avg
□ fleet_utilization (active/total)
□ convergence_indicator

AGGREGATED (per 100 episodes):
□ mean ± std của tất cả metrics
□ success_rate (coverage ≥ 90%)
□ min/max episode reward
Biểu đồ cần vẽ (Phase 2)
text

□ BIỂU ĐỒ 1: Learning Curves
  X: Episodes (0-3000)
  Y: Average Reward (moving avg window=100)
  Lines: MAPPO(blue) vs MASAC(red) vs MATD3(green)
  Shading: ± Std

□ BIỂU ĐỒ 2: Coverage Rate Curves
  X: Episodes
  Y: Coverage Rate (%)
  Lines: 3 algorithms

□ BIỂU ĐỒ 3: Final Performance Bar Chart
  X: Algorithms
  Y: Coverage Rate / Victims Found (%)
  Grouped bars + Error bars ± Std

□ BIỂU ĐỒ 4: Box Plot - Stability
  X: Algorithms
  Y: Episode Reward
  Shows: median, IQR, outliers

□ BIỂU ĐỒ 5: Radar Chart
  Dimensions: Coverage, Victims, Safety,
              Efficiency, Convergence Speed
  Lines: 3 algorithms (normalized 0-1)

□ BIỂU ĐỒ 6: Convergence Time
  X: Algorithms
  Y: Episodes to reach stable performance
  Definition: First episode where avg reward
              stays within 5% for 100 eps
PHASE 3 - SO SÁNH REWARD
Checklist Phase 3
text

□ Lấy Winner từ Phase 2
□ Train Winner + Hand-crafted (baseline đã có)
□ Implement LLM reward pipeline
□ Train Winner + LLM reward
□ So sánh và vẽ biểu đồ
LLM Reward Pipeline
text

FILE: rewards/llm_reward.py

□ aggregate_metrics(episode_buffer, window=500):
  ├── avg_coverage_rate
  ├── avg_victims_found
  ├── battery_death_rate
  ├── collision_rate
  ├── avg_episode_length
  ├── avg_n_victims (environment context)
  ├── avg_n_uavs (environment context)
  └── high_urgency_rescue_rate

□ build_llm_context(metrics, env_description):
  ├── Task description (SAR scenario)
  ├── Environment info (map, UAVs, victims)
  ├── Current metrics (what's working/failing)
  ├── Reward function template
  └── Requirements for generated code

□ query_llm(context, model="gpt-4o") → str:
  └── Return generated reward code string

□ parse_and_validate(code_str) → bool:
  ├── Syntax check (compile)
  ├── Execute với dummy inputs
  ├── Check output is float
  └── Check no side effects

□ update_reward_function(new_code):
  ├── If valid: set as current reward
  ├── If invalid: keep previous + log error
  └── Log all changes

□ LLM UPDATE SCHEDULE:
  └── Every 500 episodes → update reward
So sánh Phase 3
text

FILE: training/train_llm.py

□ Setup:
  ├── Winner algorithm từ Phase 2
  ├── Same curriculum, same N_TOTAL
  └── LLM update every 500 episodes

□ EXP G: Winner + Hand-crafted
  └── Đã có từ Phase 2 (reuse results)

□ EXP H: Winner + LLM
  ├── Train 3000 episodes
  ├── LLM updates at: 500, 1000, 1500, 2000, 2500
  └── Log reward function evolution

□ Biểu đồ thêm:
  ├── Reward function evolution (LLM changes)
  ├── Before/after each LLM update
  └── Hand-crafted vs LLM final performance
PHASE 4 - ISAACLAB MIGRATION
Checklist Phase 4
text

□ Setup IsaacLab environment
□ Chuẩn bị USD assets
□ Viết IsaacLab env wrapper
□ Verify behavior vs PyBullet
□ Train với parallel envs
□ So sánh performance
IsaacLab Migration
text

GIỮ NGUYÊN (không thay đổi gì):
□ entities/
□ sensors/
□ rewards/
□ observation/
□ core/
□ training/algorithms/

CHỈ THAY ĐỔI:
□ env/isaaclab/

---

FILE: env/isaaclab/uav_cfg.py

□ UAV_ARTICULATION_CFG:
  ├── usd_path = "assets/quadrotor.usd"
  ├── spawn: pos offset từ stations
  └── init_state: battery=100%, pos, vel=0

FILE: env/isaaclab/scene_cfg.py

□ SCENE_CFG:
  ├── ground: AssetBaseCfg (plane)
  ├── uavs: ArticulationCfg
  ├── obstacles: RigidObjectCfg
  └── stations: VisualizationMarkersCfg

FILE: env/isaaclab/sar_env_isaac.py

□ CLASS SAREnvIsaac(DirectMARLEnv):

  □ reset():
    ├── scene.reset()
    ├── Gọi map_generator (giống PyBullet)
    ├── Set articulation states
    └── Return obs (giống PyBullet)

  □ step(actions):
    ├── [1-9] GIỐNG PyBullet step()
    │         (Reuse entities/sensors/rewards)
    ├── [10] IsaacLab physics:
    │   ├── Set articulation targets
    │   ├── scene.write_data_to_sim()
    │   ├── sim.step()
    │   └── scene.update()
    └── Return obs, rewards, dones, infos

  □ Parallel training:
    ├── num_envs = 64
    ├── Vectorized obs: [64, N, 52]
    ├── Batched actions: [64, N, 3]
    └── Action masking tensor

---

VERIFY STEPS:
□ Run 10 episodes IsaacLab
□ Compare trajectories vs PyBullet
□ Assert rewards within 5% difference
□ Assert obs values same range

PARALLEL TRAINING:
□ Start với num_envs = 4
□ Tăng dần: 4 → 16 → 64
□ Monitor GPU memory
□ Compare training speed vs PyBullet
CẤU TRÚC FILE HOÀN CHỈNH
text

sar_uav_swarm/
│
├── config.py
│
├── utils/
│   ├── geometry.py
│   └── logger.py
│
├── entities/
│   ├── __init__.py
│   ├── uav.py
│   ├── victim.py
│   ├── obstacle.py
│   └── charging_station.py
│
├── core/
│   ├── __init__.py
│   ├── coverage_map.py
│   ├── map_generator.py
│   └── fleet_manager.py
│
├── sensors/
│   ├── __init__.py
│   ├── fov_sensor.py
│   └── comm_sensor.py
│
├── observation/
│   ├── __init__.py
│   └── obs_builder.py
│
├── rewards/
│   ├── __init__.py
│   ├── baseline_reward.py
│   └── llm_reward.py
│
├── env/
│   ├── __init__.py
│   ├── base_env.py
│   ├── pybullet/
│   │   ├── __init__.py
│   │   ├── sar_env_pybullet.py
│   │   └── pybullet_bridge.py
│   └── isaaclab/
│       ├── __init__.py
│       ├── sar_env_isaac.py
│       ├── uav_cfg.py
│       └── scene_cfg.py
│
├── training/
│   ├── __init__.py
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── mappo/
│   │   │   ├── actor.py
│   │   │   ├── critic.py
│   │   │   └── mappo_trainer.py
│   │   ├── masac/
│   │   │   ├── actor.py
│   │   │   ├── critic.py
│   │   │   └── masac_trainer.py
│   │   └── matd3/
│   │       ├── actor.py
│   │       ├── critic.py
│   │       └── matd3_trainer.py
│   ├── curriculum.py
│   ├── train_comparison.py
│   ├── train_llm.py
│   └── eval.py
│
├── visualization/
│   ├── __init__.py
│   ├── visualizer_2d.py
│   └── comparison_plots.py
│
├── assets/
│   ├── quadrotor.urdf
│   ├── quadrotor.usd
│   ├── debris.usd
│   └── station.usd
│
├── results/
│   ├── phase2_algorithm_comparison/
│   ├── phase3_reward_comparison/
│   └── phase4_isaaclab/
│
└── tests/
    ├── test_entities.py
    ├── test_sensors.py
    ├── test_observation.py
    ├── test_reward.py
    └── test_env.py
THỨ TỰ THỰC HIỆN
text

PHASE 1 - XÂY DỰNG:

□ BƯỚC 1: Foundation
  └── config.py → geometry.py → logger.py

□ BƯỚC 2: Entities
  └── charging_station → obstacle → victim → uav

□ BƯỚC 3: Core Systems
  └── coverage_map → map_generator → fleet_manager

□ BƯỚC 4: Sensors
  └── fov_sensor → comm_sensor

□ BƯỚC 5: Observation
  └── obs_builder (52 dims actor + critic)

□ BƯỚC 6: Reward
  └── baseline_reward (7 components)

□ BƯỚC 7: Environment
  └── base_env → pybullet_bridge → sar_env_pybullet

□ BƯỚC 8: Curriculum
  └── curriculum.py (3 phases)

□ BƯỚC 9: Utils & Visualization
  └── visualizer_2d → comparison_plots

□ BƯỚC 10: Testing
  ├── Unit tests từng module
  ├── Integration test full episode
  ├── Sanity checks (obs shape, reward range)
  └── Visual debug với visualizer_2d

─────────────────────────────────────────

PHASE 2 - SO SÁNH THUẬT TOÁN:

□ BƯỚC 11: Implement Algorithms
  └── mappo → masac → matd3

□ BƯỚC 12: Training Infrastructure
  └── train_comparison.py → eval.py

□ BƯỚC 13: Run Experiments
  ├── MAPPO × 3000 episodes
  ├── MASAC × 3000 episodes
  └── MATD3 × 3000 episodes

□ BƯỚC 14: Analysis
  ├── Vẽ 6 biểu đồ
  ├── Fill metrics table
  └── Chọn WINNER

─────────────────────────────────────────

PHASE 3 - SO SÁNH REWARD:

□ BƯỚC 15: LLM Integration
  └── llm_reward.py

□ BƯỚC 16: Train với LLM
  └── train_llm.py

□ BƯỚC 17: Analysis
  ├── Hand-crafted vs LLM
  └── Vẽ biểu đồ so sánh

─────────────────────────────────────────

PHASE 4 - ISAACLAB:

□ BƯỚC 18: Setup IsaacLab
  └── Install + USD assets

□ BƯỚC 19: Port Environment
  └── sar_env_isaac.py

□ BƯỚC 20: Verify & Train
  ├── Verify vs PyBullet
  ├── Parallel training
  └── Final comparison
NGUYÊN TẮC QUAN TRỌNG
text

FRAMEWORK:
□ Dec-POMDP với CTDE xuyên suốt
□ Actor: 52 dims LOCAL ONLY
□ Critic: Global state (training only)
□ CTDE nhất quán tất cả experiments

THIẾT KẾ:
□ Soft constraints (không hard rules)
□ Fleet manager: rule-based (không RL)
□ N_TOTAL: input cố định (không random)
□ N_VICTIMS: random [5, 20] mỗi episode
□ Entities độc lập với simulator

TESTING:
□ Test từng module trước khi ghép
□ Không skip debug step
□ Sanity check observation: shape + range
□ Monitor battery_death_rate

COMPARISON:
□ Chỉ thay đổi 1 biến mỗi experiment
□ Phase 2: thay đổi algorithm
□ Phase 3: thay đổi reward
□ Phase 4: thay đổi simulator
□ Tất cả cùng N_TOTAL, curriculum, episodes
METRICS TRACKING TABLE
text

PHASE 2 - ALGORITHM COMPARISON:
┌──────────────────────────────────────────────────┐
│ Metric            │ MAPPO  │ MASAC  │ MATD3     │
├──────────────────────────────────────────────────┤
│ Coverage Rate (%) │        │        │           │
│ Victims Found (%) │        │        │           │
│ Success Rate (%)  │        │        │           │
│ Conv. Speed (eps) │        │        │           │
│ Battery Deaths    │        │        │           │
│ Collision Rate    │        │        │           │
│ Avg Episode Len   │        │        │           │
│ Reward Variance   │        │        │           │
└──────────────────────────────────────────────────┘

PHASE 3 - REWARD COMPARISON:
┌──────────────────────────────────────────────────┐
│ Metric            │ Hand-crafted │ LLM Reward   │
├──────────────────────────────────────────────────┤
│ Coverage Rate (%) │              │              │
│ Victims Found (%) │              │              │
│ Success Rate (%)  │              │              │
│ Conv. Speed (eps) │              │              │
│ Battery Deaths    │              │              │
└──────────────────────────────────────────────────┘

PHASE 4 - SIMULATOR COMPARISON:
┌──────────────────────────────────────────────────┐
│ Metric            │ PyBullet  │ IsaacLab        │
├──────────────────────────────────────────────────┤
│ Coverage Rate (%) │           │                 │
│ Training Speed    │           │                 │
│ Physics Accuracy  │           │                 │
└──────────────────────────────────────────────────┘