# 🚁 SAR UAV SWARM — PROJECT HANDOFF DOCUMENT v9.0

**Cập nhật:** Phase 1 hoàn thành 100% | Phase 2 chờ triển khai

---

## 🎯 MỤC TIÊU NGHIÊN CỨU

### Paper 1 — Algorithm Comparison

- **Câu hỏi:** MAPPO vs MASAC vs MATD3 — thuật toán nào tốt nhất cho SAR?
- **Phương pháp:** 3 algorithms × 5 seeds × 3 curriculum stages × 3000 episodes
- **Metrics:** Coverage rate, Victims found, Episode reward, Sample efficiency

### Paper 2 — LLM Reward vs Hand-Crafted

- **Câu hỏi:** LLM có generate reward function tốt hơn human expert không?
- **Baseline:** `BaselineReward v3.1` (hand-crafted, nghiên cứu kỹ)
- **LLM:** GPT-4/Claude generate reward code từ task description

### Task Definition

- **Agents:** 4 UAVs phối hợp tìm kiếm 10–36 victims (injured / mobile)
- **Môi trường:** Debris (vật cản cứng), Danger Zones (fire/radiation/smoke/gas/collapse)
- **Constraint:** Battery — UAV phải quay trạm sạc khi pin thấp
- **Curriculum:** EASY (150×150m) → MEDIUM (200×200m) → HARD (250×250m)

---

## 📊 TRẠNG THÁI TỔNG QUAN

| Phase   | Mô tả                              | Trạng thái         |
| ------- | ---------------------------------- | ------------------ |
| Phase 1 | Core Infrastructure (48 files)     | ✅ 100% hoàn thành |
| Phase 2 | RL Algorithms (MAPPO/MASAC/MATD3)  | ⬜ Chưa bắt đầu    |
| Phase 3 | LLM Reward Integration             | ⬜ Chưa bắt đầu    |
| Phase 4 | Advanced Backends (PyBullet/Isaac) | ⬜ Chưa bắt đầu    |

**Test status:** 26/26 tests PASS (100% core coverage)
**Actor obs vector:** 68 dimensions (n_stations=2)
**Critic obs vector:** 554 dimensions (8 × 68 + 10)

---

## 📁 CẤU TRÚC ĐẦY ĐỦ — TỪNG FILE & THÀNH PHẦN

---

### 📁 `config/` — Hệ thống cấu hình (8 files + `__init__`)

---

#### `config/__init__.py`

Export tất cả config classes ra ngoài. Import từ đây để dùng.

- **Export:** `AppConfig`, `EnvConfig`, `UAVConfig`, `SensorConfig`, `VictimConfig`, `ObstacleConfig`, `DangerZoneConfig`, `RewardConfig`, `ObsConfig`, `TrainConfig`, `StageConfig`, `STAGE_EASY`, `STAGE_MEDIUM`, `STAGE_HARD`, `CURRICULUM_STAGES`

---

#### `config/config.py` — `AppConfig` (Master Orchestrator)

**Vai trò:** Config tổng hợp duy nhất truyền vào toàn bộ hệ thống.

| Thuộc tính   | Kiểu               | Mô tả                |
| ------------ | ------------------ | -------------------- |
| `env`        | `EnvConfig`        | Map & physics        |
| `uav`        | `UAVConfig`        | Drone dynamics       |
| `sensor`     | `SensorConfig`     | FOV/Comm             |
| `victim`     | `VictimConfig`     | Victim spawning      |
| `obstacle`   | `ObstacleConfig`   | Debris params        |
| `danger`     | `DangerZoneConfig` | Danger zone configs  |
| `reward`     | `RewardConfig`     | Reward components    |
| `obs`        | `ObsConfig`        | Observation dims     |
| `train`      | `TrainConfig`      | RL training params   |
| `viz_mode`   | `str`              | "2d" / "3d" / "none" |
| `viz_3d_cfg` | `dict`             | 3D renderer config   |

| Method                      | Mô tả                                                           |
| --------------------------- | --------------------------------------------------------------- |
| `__post_init__()`           | Auto-sync `obs.n_stations = env.n_stations`, validate           |
| `apply_stage(stage)`        | Apply curriculum stage config in-place (single source of truth) |
| `map_diagonal` (property)   | `sqrt(2) × map_size`                                            |
| `grid_cell_size` (property) | `map_size / grid_size`                                          |
| `save(path)`                | Serialize sang JSON                                             |
| `load(path)`                | Restore từ JSON                                                 |

---

#### `config/env.py` — `EnvConfig`

**Vai trò:** Tất cả params liên quan map, thời gian, fleet.

| Thuộc tính                    | Giá trị mặc định | Mô tả                                 |
| ----------------------------- | ---------------- | ------------------------------------- |
| `map_size`                    | 100              | Kích thước map (m)                    |
| `grid_size`                   | 100              | Số ô lưới (luôn sync = map_size)      |
| `dt_seconds`                  | 1.0              | Timestep (s)                          |
| `max_steps`                   | 600              | Steps tối đa/episode                  |
| `n_uav`                       | 4                | Số UAV                                |
| `n_stations`                  | 2                | Số trạm sạc                           |
| `charge_radius_m`             | 3.0              | Bán kính sạc (m)                      |
| `station_capacity`            | 2                | UAVs tối đa/trạm                      |
| `min_station_spacing_m`       | 15.0             | Khoảng cách tối thiểu giữa các trạm   |
| `max_place_attempts`          | 500              | Số lần thử đặt object                 |
| `min_object_spacing_m`        | 2.5              | Spacing tối thiểu giữa objects        |
| `victim_clearance_m`          | 1.5              | Khoảng trống quanh victim             |
| `deterministic_eval`          | False            | Fixed seed khi eval                   |
| `eval_seed`                   | 42               | Seed cho eval mode                    |
| `placement_relax_threshold`   | 0.7              | Sau 70% attempts → relax spacing      |
| `placement_relaxed_spacing_m` | 1.5              | Spacing khi relaxed                   |
| `allow_partial_obstacles`     | True             | Skip thay vì crash khi không đặt được |

Properties backward compat: `dt`, `charge_radius`, `min_station_spacing`, v.v.

---

#### `config/uav.py` — `UAVConfig`

**Vai trò:** Vật lý UAV và battery model.

| Thuộc tính               | Giá trị mặc định | Mô tả                      |
| ------------------------ | ---------------- | -------------------------- |
| `z_min_m`                | 3.0              | Độ cao tối thiểu (m)       |
| `z_max_m`                | 40.0             | Độ cao tối đa (m)          |
| `max_speed_xy_mps`       | 5.0              | Tốc độ ngang tối đa (m/s)  |
| `max_speed_z_mps`        | 2.0              | Tốc độ dọc tối đa (m/s)    |
| `collision_radius_m`     | 0.5              | Bán kính va chạm (m)       |
| `drain_xy_pct_per_s`     | 0.10             | Drain ngang (%/s)          |
| `drain_z_up_pct_per_s`   | 0.15             | Drain leo cao (%/s)        |
| `drain_z_down_pct_per_s` | 0.03             | Drain hạ thấp (%/s)        |
| `drain_idle_pct_per_s`   | 0.05             | Drain hover (%/s)          |
| `charge_rate_pct_per_s`  | 1.5              | Tốc độ sạc (%/s)           |
| `battery_return_pct`     | 10.0             | Ngưỡng tự động về trạm (%) |
| `battery_ready_pct`      | 80.0             | Sẵn sàng xuất phát (%)     |
| `battery_dead_pct`       | 0.0              | Pin chết → DISABLED        |
| `battery_warning_pct`    | 20.0             | Cảnh báo thấp (%)          |
| `battery_critical_pct`   | 10.0             | Ngưỡng critical (%)        |
| `battery_emergency_pct`  | 5.0              | Ngưỡng emergency (%)       |
| `reserve_ratio`          | 0.2              | 20% swarm trong reserve    |
| `min_reserve`            | 2                | Tối thiểu 2 UAV reserve    |

Properties backward compat: `z_min`, `z_max`, `drain_xy_max`, `battery_dead`, v.v.

---

#### `config/sensor.py` — `SensorConfig`

**Vai trò:** Params sensor (FOV geometry + detection noise model).

| Thuộc tính          | Giá trị mặc định | Mô tả                   |
| ------------------- | ---------------- | ----------------------- |
| `comm_range_m`      | 30.0             | Tầm liên lạc (m)        |
| `hfov_deg`          | 90.0             | Góc FOV ngang (°)       |
| `p_detect_base`     | 0.95             | P_detect tại altitude=0 |
| `p_detect_decay`    | 0.04             | Decay theo altitude     |
| `enable_noise`      | True             | Bật noise model         |
| `motion_blur_coeff` | 0.06             | Penalty khi bay nhanh   |
| `base_miss_rate`    | 0.03             | Hardware miss rate      |

Properties: `fov_tan` = tan(hfov/2), `fov_radius_at_altitude` (closure), `comm_range`

---

#### `config/entity.py` — `VictimConfig`, `ObstacleConfig`, `DangerZoneConfig`

**`VictimConfig`:**
| Thuộc tính | Mặc định | Mô tả |
|-----------|---------|--------|
| `n_victims_min/max` | 5/20 | Số victims/episode |
| `injured_ratio_min/max` | 0.4/0.7 | Tỉ lệ injured |
| `injured_urgency_min/max` | 4.0/5.0 | Urgency injured |
| `mobile_urgency_min/max` | 1.0/3.0 | Urgency mobile |
| `mobile_speed_min/max_mps` | 0.2/0.4 | Tốc độ mobile (m/s) |
| `mobile_dir_change_steps` | 20 | Đổi hướng mỗi N steps |

**`ObstacleConfig`:**
| Thuộc tính | Mặc định | Mô tả |
|-----------|---------|--------|
| `n_debris` | 6 | Số debris |
| `debris_width_min/max_m` | 2.0/5.0 | Footprint diameter (m) |
| `debris_height_min/max_m` | 3.0/8.0 | Chiều cao 3D (m) |
| `n_danger_total` | 2 | Tổng số danger zones |

**`DangerZoneConfig`:**

- `heights`: Dict {type: max_height} — `gas=3`, `fire=15`, `smoke=25`, `collapse=10`, `radiation=inf`
- `penalties`: Dict {type: per_step_penalty} — `gas=-3`, `fire=-3`, `smoke=-1.5`, `collapse=-1`, `radiation=-5`
- `max_counts`: Dict {type: max_count} — tổng số mỗi loại
- `widths`: Dict {type: (min_diam, max_diam)} — kích thước (đường kính, không phải bán kính!)
- Method `validate()`: Kiểm tra consistency của 4 dicts
- Property `danger_types`: List các loại

---

#### `config/reward.py` — `RewardConfig` (v3.1 Research-Grade)

**Vai trò:** 14 reward components, rebalanced cho RL training.

| Component                      | Giá trị   | Mô tả                       |
| ------------------------------ | --------- | --------------------------- |
| `r_coverage_delta`             | +6.0      | Per 1% coverage tăng        |
| `r_victim_base`                | +50.0     | × urgency/5 khi tìm thấy    |
| `r_battery_20`                 | 0.0       | Penalty <20% (đã tắt)       |
| `r_battery_10`                 | -1.0      | Penalty <10%                |
| `r_battery_5`                  | -3.0      | Penalty <5%                 |
| `r_battery_dead`               | -100.0    | One-time khi chết pin       |
| `r_collision_obstacle`         | -30.0     | One-time khi va chạm debris |
| `r_proximity_1m`               | -10.0     | Per step khi 2 UAV < 1m     |
| `r_proximity_2m`               | -3.0      | Per step khi < 2m           |
| `r_proximity_3m`               | -0.5      | Per step khi < 3m           |
| `proximity_penalty_cap`        | -15.0     | Cap proximity/step          |
| `r_time_penalty`               | -0.05     | Per active UAV per step     |
| `r_terminal_base`              | +200.0    | Base terminal bonus         |
| `terminal_bonus_cap`           | +100.0    | Max terminal bonus          |
| `step_penalty_cap`             | -30.0     | Tổng penalty tối đa/step    |
| `step_reward_clip_min/max`     | -100/+100 | Clip mỗi step               |
| `enable_distance_shaping`      | True      | Delta-based shaping         |
| `distance_shaping_max_per_uav` | 1.0       | Cap shaping/UAV             |

---

#### `config/obs.py` — `ObsSchemaConfig`, `ObsConfig`

**`ObsSchemaConfig`** — Defines dims của từng slot:

- `SELF_FEATURES = 11`
- `STATION_FEATURES_PER = 4`
- `TEAMMATE_FEATURES_PER = 3`
- `OBSTACLE_FEATURES_PER = 3`
- `VICTIM_FEATURES_PER = 5`
- `COVERAGE_FEATURES = 3`
- `GLOBAL_FEATURES = 10`

**`ObsConfig`** — Runtime observation config:
| Thuộc tính | Mặc định | Mô tả |
|-----------|---------|--------|
| `n_obs_victims` | 5 | Tối đa victims trong obs |
| `n_obs_obstacles` | 4 | Tối đa obstacles trong obs |
| `n_tracked_teammates` | 3 | Tối đa teammates track |
| `local_cov_small` | 15 | Radius nhỏ coverage (m) |
| `local_cov_large` | 30 | Radius lớn coverage (m) |
| `max_uav` | 8 | Padding critic obs |
| `n_stations` | None | Auto-sync từ EnvConfig |

Properties (computed dims):

- `self_dim = 11`
- `station_dim = n_stations × 4` (= 8 với n_stations=2)
- `team_dim = 3 × 3 = 9`
- `obstacle_dim = 4 × 3 = 12`
- `victim_dim = 5 × 5 = 25`
- `coverage_dim = 3`
- **`actor_dim = 68`** (tổng với n_stations=2)
- `global_dim = 10`
- **`critic_dim = 554`** (8×68 + 10)
- Method `validate()`: Kiểm tra n_stations không None

---

#### `config/train.py` — `TrainConfig`

| Thuộc tính         | Mặc định              | Mô tả               |
| ------------------ | --------------------- | ------------------- |
| `n_seeds`          | 5                     | Số seeds để chạy    |
| `seeds`            | [42,123,456,789,1011] | Fixed seeds         |
| `confidence_level` | 0.95                  | Cho Wilcoxon/t-test |
| `total_episodes`   | 3000                  | Tổng episodes       |
| `eval_interval`    | 50                    | Eval mỗi N episodes |
| `save_interval`    | 100                   | Save checkpoint     |
| `log_window`       | 100                   | Rolling mean window |

---

#### `config/curriculum_config.py` — `StageConfig`, Stages

**Vai trò:** Định nghĩa 3 curriculum stages với difficulty progression.

**`StageConfig` fields:**
| Field | Mô tả |
|-------|--------|
| `name` | "easy" / "medium" / "hard" |
| `map_size` | Kích thước map (m) |
| `n_uav` | Số UAV (luôn = 4) |
| `n_victims_min/max` | Range victims |
| `n_debris` | Số debris |
| `n_danger_total` | Tổng danger zones |
| `station_capacity` | Capacity trạm sạc |
| `max_steps` | Max steps/episode |
| `min_episodes` | Tối thiểu episodes trước khi advance |
| `advance_coverage` | Ngưỡng coverage để advance |
| `advance_victims` | Ngưỡng victim rate để advance |

**Properties (computed, cho paper):**

- `map_area_m2` = map_size²
- `coverage_pressure_m2_per_uav` = area / n_uav (key difficulty metric)
- `victim_density_per_1000m2` ≈ 0.53 (constant across stages)
- `obstacle_density_per_1000m2` ≈ 0.35
- `steps_per_m2` (time budget per area)
- `describe()` → human-readable string

**3 stage instances:**
| Stage | Map | Pressure | Max Steps | Advance |
|-------|-----|----------|-----------|---------|
| `STAGE_EASY` | 150×150m | 5,625 m²/UAV | 300 | cov≥70%, vic≥80% |
| `STAGE_MEDIUM` | 200×200m | 10,000 m²/UAV | 350 | cov≥65%, vic≥75% |
| `STAGE_HARD` | 250×250m | 15,625 m²/UAV | 400 | cov≥60%, vic≥70% |

`CURRICULUM_STAGES = [STAGE_EASY, STAGE_MEDIUM, STAGE_HARD]`

`_verify_stages()` chạy khi import — validate density consistency.

---

### 📁 `utils/` — Tiện ích (2 files)

---

#### `utils/geometry.py`

**Vai trò:** 9 hàm geometry được vectorized với NumPy.

| Hàm                                                     | Signature         | Mô tả                              |
| ------------------------------------------------------- | ----------------- | ---------------------------------- |
| `dist_2d(pos1, pos2)`                                   | → float           | Khoảng cách XY                     |
| `dist_3d(pos1, pos2)`                                   | → float           | Khoảng cách XYZ                    |
| `normalize_angle(angle)`                                | → float           | Về [-π, π]                         |
| `compute_bearing(from_pos, from_vel, to_pos)`           | → float           | Góc tương đối từ hướng bay         |
| `check_los_2d(pos1, pos2, obstacles)`                   | → bool            | Line-of-sight, hỗ trợ Shapely      |
| `_line_intersects_circle(p1, p2, center, r)`            | → bool            | Helper (10× faster vectorized)     |
| `get_circle_cells(center, radius, grid_size, map_size)` | → np.ndarray(N,2) | FOV cells — **10× faster vs loop** |
| `get_relative_position(from_pos, to_pos)`               | → np.ndarray      | Vector relative [dx,dy,dz]         |
| `clip_position(pos, min_bounds, max_bounds)`            | → np.ndarray      | Boundary clamp                     |

`get_circle_cells_legacy()` — phiên bản loop cũ (giữ để benchmark).

---

#### `utils/logger.py`

**`EpisodeLogger`** — Log 1 episode:
| Method/Attr | Mô tả |
|-------------|--------|
| `log_step(rewards, coverage)` | Cập nhật reward sum và coverage max |
| `log_event(event_type)` | "collision_obstacle", "victim_found", "battery_death", "danger_zone", "hot_swap" |
| `set_total_victims(n)` | Đặt tổng số victims |
| `finalize()` | → Dict metrics (JSON-safe): reward, coverage_rate (%), victims, collisions, success |

**`TrainingLogger`** — Log toàn bộ training:
| Method | Mô tả |
|--------|--------|
| `log_episode(metrics)` | Cập nhật windows, check convergence, print nếu verbose |
| `get_stats(last_n)` | Stats dict (mean/std/success_rate/converged) |
| `save(filepath)` | Lưu JSON |
| `load(filepath)` | Khôi phục từ JSON |

`compare_training_runs(runs, labels)` — So sánh nhiều runs (cho Phase 2).

---

### 📁 `entities/` — Game Objects (4 files)

---

#### `entities/uav.py` — `UAV`, `UAVState`

**`UAVState` enum:** `ACTIVE`, `RETURNING`, `CHARGING`, `DEPLOYING`, `DISABLED`

**`UAV` class:**
| Thuộc tính instance | Mô tả |
|--------------------|--------|
| `id` | int ID |
| `pos` | np.ndarray [x,y,z] |
| `vel` | np.ndarray [vx,vy,vz] |
| `battery` | float [0,100] |
| `state` | UAVState |
| `target_station` | ChargingStation hoặc None |
| `steps_alive`, `distance_xy`, `distance_3d` | Tracking |
| `victims_found` | int |
| `battery_death` | bool (pin chết lần đầu) |

| Method                           | Mô tả                                            |
| -------------------------------- | ------------------------------------------------ |
| `battery_pct` (property)         | Alias của `battery`                              |
| `apply_action(action)`           | Nhận action [-1,1]³, scale thành vel, update pos |
| `auto_navigate(target_pos)`      | Bay tự động đến target (không overshoot)         |
| `update_battery(stations)`       | Drain hoặc charge tuỳ state                      |
| `_do_drain()`                    | Tính drain proportional theo velocity × dt       |
| `_do_charge(stations)`           | Charge qua target_station                        |
| `get_battery_penalty()`          | Legacy battery penalty (dùng reward fn thay thế) |
| `get_fov_radius()`               | altitude × fov_tan                               |
| `get_state_onehot()`             | np.ndarray(5,) one-hot                           |
| `set_state(new_state)`           | Chuyển state có validation                       |
| `needs_charging()`               | battery ≤ battery_return_pct                     |
| `is_ready_to_deploy()`           | battery ≥ battery_ready_pct                      |
| `find_nearest_station(stations)` | Tìm trạm gần nhất còn chỗ                        |
| `to_dict()`                      | JSON-safe dict                                   |

---

#### `entities/victim.py` — `BaseVictim`, `InjuredVictim`, `MobileVictim`

**`BaseVictim` (abstract):**
| Thuộc tính | Mô tả |
|-----------|--------|
| `id`, `pos`, `urgency` | Core info |
| `is_found`, `found_at_step`, `found_by_uav` | Detection tracking |

| Method                                                | Mô tả                                     |
| ----------------------------------------------------- | ----------------------------------------- |
| `step(obstacles)`                                     | Abstract — update physics                 |
| `update(step_count, obstacles)`                       | Alias của step() — cho logic_backend      |
| `is_detected_by(uav_pos, fov_r, obstacles, p_detect)` | Legacy detection                          |
| `mark_found(step, uav_id)`                            | Set is_found=True, gọi `_on_found()` hook |
| `get_reward_value()`                                  | r_victim_base × urgency/5                 |

**`InjuredVictim`:**

- Không di chuyển (`step()` = pass)
- `urgency` = [4.0, 5.0], `speed = 0.0`
- `_on_found()`: Không cần làm gì

**`MobileVictim`:**

- Random walk với `speed` [0.2, 0.4] m/s
- Đổi hướng mỗi `mobile_dir_change_steps=20` steps
- `_on_found()`: **Freeze** `speed=0.0` ngay lập tức
- `step(obstacles)`: Check `is_found` → boundary clip → obstacle check → update pos

---

#### `entities/charging_station.py` — `ChargingStation`

| Thuộc tính          | Mô tả                |
| ------------------- | -------------------- |
| `id`, `pos`         | ID và vị trí [x,y,0] |
| `capacity`          | UAVs tối đa (từ cfg) |
| `charge_radius`     | Bán kính sạc (m)     |
| `charge_rate`       | %/step               |
| `current_occupants` | List UAV đang sạc    |
| `occupant_ids`      | set() — O(1) lookup  |

| Method                  | Mô tả                                                  |
| ----------------------- | ------------------------------------------------------ |
| `is_full()`             | len(occupants) >= capacity                             |
| `is_available()`        | len < capacity                                         |
| `in_range(uav_pos)`     | dist_xy ≤ charge_radius AND z ≤ 0.5m                   |
| `try_occupy(uav)`       | Chiếm slot (bool)                                      |
| `release(uav)`          | Giải phóng slot (bool)                                 |
| `charge(uav)`           | Sạc 1 step: check range → try_occupy → battery += rate |
| `force_release_all()`   | Reset khi episode mới                                  |
| `get_occupancy_ratio()` | [0.0, 1.0]                                             |

---

#### `entities/obstacle.py` — `Debris`, `DangerZone`

**Cả hai hỗ trợ 3 shapes:** `"circle"`, `"rectangle"`, `"polygon"`

**`Debris`:**
| Thuộc tính | Mô tả |
|-----------|--------|
| `id`, `pos`, `height_3d` | Core |
| `shape` | circle/rectangle/polygon |
| `radius` / `width`+`height_2d`+`rotation` / `vertices` | Tuỳ shape |
| `polygon` | Shapely polygon object (nếu có) |
| `penalty` | Từ cfg.reward.r_collision_obstacle |

| Method                         | Mô tả                                |
| ------------------------------ | ------------------------------------ |
| `in_zone_2d(pos_2d)`           | XY containment check                 |
| `causes_collision(uav_pos)`    | in_zone_2d AND uav.z < height_3d     |
| `blocks_los(pos1, pos2)`       | LOS blocked?                         |
| `get_distance_to_edge(pos_2d)` | Khoảng cách đến cạnh                 |
| `_get_fallback_radius()`       | Bounding radius khi không có Shapely |

**`DangerZone`** (giống Debris về structure):

- Thêm: `danger_type`, `max_height`, `penalty`
- `is_inside(uav_pos)` thay `causes_collision`
- `blocks_los()`: Chỉ `fire` và `smoke` chặn LOS
- `get_sensor_modifier()` → float [0.4, 1.0] (smoke=0.4, fire=0.55, collapse=0.70, gas=0.85, radiation=0.95)
- `get_battery_modifier()` → 0.05 nếu fire, 0.0 otherwise

---

### 📁 `core/` — Hệ thống lõi (3 files)

---

#### `core/coverage_map.py` — `CoverageMap` v2.0

**Vai trò:** Tracking khu vực đã khám phá với temporal info.

| Thuộc tính   | Kiểu         | Mô tả                            |
| ------------ | ------------ | -------------------------------- |
| `grid`       | bool[GS,GS]  | Đã explore hay chưa              |
| `timestamps` | int32[GS,GS] | Step cuối cùng scan              |
| `first_scan` | int32[GS,GS] | Step đầu tiên scan (-1 nếu chưa) |
| `scan_count` | int32[GS,GS] | Số lần scan mỗi ô                |

| Method                                      | Mô tả                                                              |
| ------------------------------------------- | ------------------------------------------------------------------ |
| `reset()`                                   | Reset tất cả về 0/False/-1                                         |
| `mark_explored(uav_pos, fov_radius, step)`  | Vectorized: mark grid + update timestamps (không overwrite cũ hơn) |
| `get_coverage_rate()`                       | [0,1] toàn map                                                     |
| `get_coverage_percent()`                    | [0,100]                                                            |
| `get_local_coverage(pos, radius)`           | Coverage trong vùng bán kính                                       |
| `get_staleness(pos, radius, step)`          | Tuổi trung bình (unexplored = max_steps)                           |
| `get_staleness_normalized(...)`             | [0,1] normalize theo decay_threshold=200                           |
| `get_freshness(...)`                        | 1 - staleness_normalized                                           |
| `get_coverage_with_decay(step, decay=200)`  | Chỉ tính cells scan trong decay_threshold steps                    |
| `get_rescan_count(pos, radius)`             | Trung bình lần scan (float)                                        |
| `get_nearest_unexplored(pos, min_distance)` | O(N) scan — tìm ô chưa explore gần nhất                            |
| `get_nearest_stale(pos, step, threshold)`   | Tìm ô cũ nhất gần nhất                                             |
| `get_stats(step)`                           | Dict metrics cho logging                                           |
| `get_grid_snapshot()`                       | Export raw arrays (cho viz)                                        |

**Lưu ý complexity:** `get_nearest_*` = O(N), OK cho 100×100, bottleneck ở 500×500.

---

#### `core/map_generator.py` — `MapGenerator` v4.1

**Vai trò:** Sinh map procedurally với constraints chặt chẽ.

**Key fix v4.1:** Config `widths` = đường kính, KHÔNG phải bán kính. Generator dùng `radius = width / 2.0`.

| Method                                            | Mô tả                                                                     |
| ------------------------------------------------- | ------------------------------------------------------------------------- |
| `generate(n_victims_override, seed)`              | **Main method** — sinh toàn bộ map_data dict                              |
| `_place_stations(rng)`                            | Đặt stations với min_spacing constraint                                   |
| `_place_debris(stations, rng)`                    | Đặt debris (40% circle, 40% rect, 20% polygon) với progressive relaxation |
| `_place_danger_zones(existing, rng)`              | Đặt danger zones (50% circle, 50% rect)                                   |
| `_spawn_victims(n, obstacles, danger_zones, rng)` | Sinh victims với group spawn                                              |
| `_find_valid_victim_pos(...)`                     | Tìm vị trí hợp lệ cho victim                                              |
| `_spawn_group(n, type, ...)`                      | Sinh N victims cùng loại                                                  |
| `get_uav_spawns(stations, n_total, rng)`          | Sinh spawn positions quanh stations                                       |
| `get_map_statistics(map_data)`                    | Dict metrics (density, clustering, v.v.)                                  |

**Output `generate()` trả về:**

```python
{
  "stations": [...],      # List station dicts
  "debris": [...],        # List debris dicts
  "danger_zones": [...],  # List danger zone dicts
  "victims": [...],       # List victim dicts
  "uav_spawns": [...],    # List {"id": i, "pos": [x,y,z]}
  "seed": int,
  "n_victims": int
}
```

---

#### `core/fleet_manager.py` — `FleetManager` v2.0

**Vai trò:** Constraint enforcer (không phải rule engine). RL agent TỰ QUYẾT ĐỊNH.

| Thuộc tính                 | Mô tả                               |
| -------------------------- | ----------------------------------- |
| `n_total`, `n_reserve`     | Tổng và số reserve UAVs             |
| `all_uavs`, `stations`     | Lists                               |
| `_battery_death_penalized` | Set UAV IDs đã penalize             |
| `_uav_return_locks`        | Dict[uav_id, bool] hysteresis state |

| Method                                             | Mô tả                                                      |
| -------------------------------------------------- | ---------------------------------------------------------- |
| `reset(uavs, stations)`                            | Khởi tạo episode mới                                       |
| `get_deployable_uavs()`                            | UAVs CHARGING với battery ≥ ready, deploy từ top battery   |
| `get_best_deployable(prefer_station, min_battery)` | UAV tốt nhất để deploy                                     |
| `enforce_safety_constraints()`                     | **ENFORCE:** battery=0→DISABLED, <5%→RETURNING (emergency) |
| `suggest_deployments(target_active)`               | **SUGGEST** (RL có thể ignore)                             |
| `suggest_returns()`                                | Gợi ý về trạm                                              |
| `step()`                                           | Chạy enforce + suggest, trả về Dict                        |
| `get_mission_priority_hints()`                     | operational_ratio, reserve_health, station_pressure        |
| `get_spatial_awareness()`                          | active_positions, center_of_mass, spread_radius            |
| `count_by_state()`                                 | Dict {state: count}                                        |
| `get_battery_stats()`                              | mean/min/max/std + critical counts                         |
| `get_stats()`                                      | Full stats dict                                            |
| `get_fleet_incentives()`                           | Backward compat — luôn return 0.0                          |
| `is_episode_over()`                                | Tất cả UAVs disabled                                       |

---

### 📁 `sensors/` — Sensor Models (2 files)

---

#### `sensors/fov_sensor.py` — `FOVSensor`

**Vai trò:** FOV geometry + detection probability với noise model v2.

**Noise pipeline:**
`P_final = P_altitude × env_factor × (1 - motion_penalty) × victim_factor × (1 - base_miss_rate)`

| Method                                              | Mô tả                                                   |
| --------------------------------------------------- | ------------------------------------------------------- |
| `set_seed(seed)`                                    | Cho reproducible evaluation                             |
| `calculate_fov_radius(altitude)`                    | altitude × fov_tan                                      |
| `calculate_detection_prob(alt, speed, env, victim)` | Full noise pipeline                                     |
| `_get_env_factor(victim_pos, obstacles)`            | Kiểm tra victim trong DangerZone → modifier             |
| `_get_victim_factor(victim)`                        | InjuredVictim=1.15, MobileVictim≈0.85                   |
| `check_detected(uav, victim, obstacles)`            | **Full pipeline**: FOV → LOS → P_detect → sample        |
| `scan_victims(uav, victims, obstacles)`             | → np.ndarray(25,) obs vector (5 victims × 5 features)   |
| `scan_obstacles(uav, obstacles)`                    | → np.ndarray(12,) obs vector (4 obstacles × 3 features) |

**Victim features trong obs:** [rel_x, rel_y, dist, urgency/5, found(0/1)]
**Obstacle features trong obs:** [rel_x, rel_y, type_id(Debris=0/DangerZone=1)]

---

#### `sensors/comm_sensor.py` — `CommSensor`

**Vai trò:** V2V communication — teammates trong comm_range.

| Method                                      | Mô tả                                       |
| ------------------------------------------- | ------------------------------------------- |
| `scan(ego_uav, all_active_uavs)`            | → np.ndarray(9,) — 3 teammates × 3 features |
| `get_n_in_range(ego_uav, all_uavs)`         | Count teammates trong range                 |
| `get_teammates_in_range(ego_uav, all_uavs)` | Sorted list by distance                     |

**Teammate features:** [norm_dist, norm_bearing, norm_alt_diff]

---

### 📁 `observation/` — Observation Builder (1 file)

---

#### `observation/obs_builder.py` — `ObservationBuilder`, `ObsResult`

**Vai trò:** Build actor (68-dim) và critic (554-dim) observations.

**`ObsResult`:** Container với `actor_obs: Dict[int, np.ndarray]` và `critic_obs: np.ndarray`

**Actor obs layout (68 dims với n_stations=2):**
| Slice | Dims | Nội dung |
|-------|------|----------|
| [0:11] | 11 | Self: pos(3)/vel(3)/battery(1)/state_onehot(4) |
| [11:19] | 8 | Stations: 2 × [rel_x, rel_y, dist, occupancy] |
| [19:28] | 9 | Teammates: 3 × [dist, bearing, rel_alt] |
| [28:40] | 12 | Obstacles: 4 × [rel_x, rel_y, type_id] |
| [40:65] | 25 | Victims: 5 × [rel_x, rel_y, urgency, dist, found] |
| [65:68] | 3 | Coverage: [local_15m, local_30m, time_remaining] |

**Critic obs (554 dims):**

- [0:544] = 8 UAVs × 68 (padded với zeros)
- [544:554] = Global: n_active/n_charging/n_disabled/n_alive (×1/n), mean/std/min battery, global_coverage, victim_found_rate, time_remaining

| Method                                                               | Mô tả                                   |
| -------------------------------------------------------------------- | --------------------------------------- |
| `build_actor_obs(uav, all_uavs, stations, victims, obstacles, step)` | Build 68-dim obs cho 1 UAV              |
| `build_all(all_uavs, stations, victims, obstacles, step)`            | Build tất cả actor + critic trong 1 lần |
| Private `_write_*()`                                                 | Helpers điền từng slot                  |

---

### 📁 `rewards/` — Reward Functions (1/2 files)

---

#### `rewards/baseline_reward.py` — `BaselineReward` v3.1

**Vai trò:** Hand-crafted reward function, research-grade. Baseline cho Paper 2.

**Key fixes vs v3.0:**

- BUG-31: Penalty cap ADDITIVE (không scale distort components)
- BUG-32: Proximity cap scale theo swarm size
- BUG-33: Distance shaping DELTA-based với memory
- BUG-34: Terminal bonus không saturate
- BUG-35: Battery urgency shaping → distance-to-station incentive

| Method                                                                                                             | Mô tả                                                                      |
| ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| `reset()`                                                                                                          | Clear `_battery_death_penalized`, `_collision_penalized`, `_prev_min_dist` |
| `compute(uavs, victims, obstacles, coverage_map, fleet_manager, newly_found, prev_coverage, step, done, stations)` | **Global reward** — dict 14 components                                     |
| `compute_per_uav(uav, newly_found_by_uav, ...)`                                                                    | **Per-agent reward** (MAPPO/MASAC/MATD3)                                   |
| `_apply_penalty_cap(components, cap)`                                                                              | Additive adjustment nếu tổng penalty < cap                                 |
| `_terminal_bonus(coverage, victims, step)`                                                                         | 70% cov + 20% vic + 10% time                                               |
| `_delta_shaping_fleet(uavs, victims)`                                                                              | Fleet total shaping                                                        |
| `_delta_shaping_single(uav, victims, unfound)`                                                                     | Per-UAV: prev_dist - current_dist × weight                                 |
| `_battery_rewards(uavs, stations)`                                                                                 | Progressive penalty + urgency shaping                                      |
| `_collision_reward(uavs, obstacles)`                                                                               | One-time per UAV                                                           |
| `_danger_reward(uavs, obstacles)`                                                                                  | Per step inside zone                                                       |
| `get_component_names()`                                                                                            | List 14 component names                                                    |
| `summarize(reward_dict)`                                                                                           | Compact log string                                                         |

**Module-level functions (stateless, unit-testable):**

- `_coverage_delta_reward(prev, cur, weight)`
- `_victim_found_reward(newly_found, r_base)`
- `_battery_penalty_single(uav, reward_cfg, uav_cfg)`
- `_battery_urgency_shaping(uav, stations, map_size)`
- `_proximity_reward(active_uavs, ...)`
- `_proximity_reward_single(uav, active_uavs, ...)`

---

### 📁 `env/` — Environments (2 envs + 1 backend)

---

#### `env/base_env.py` — `SARBaseEnv`

**Vai trò:** Gymnasium interface cho SAR. Single-agent API (per-UAV actions dict).

| Thuộc tính            | Mô tả                           |
| --------------------- | ------------------------------- |
| `observation_space`   | Box(68,) float32                |
| `action_space`        | Box(3,) ∈ [-1,1] float32        |
| `backend`             | LogicBackend instance           |
| `_reward_fn`          | BaselineReward instance         |
| `_obs_builder`        | ObservationBuilder instance     |
| `_map_gen`            | MapGenerator instance           |
| `_step_count`         | Step hiện tại                   |
| `_prev_coverage`      | Coverage step trước (cho delta) |
| `_episode_reward_sum` | Tổng reward episode             |

| Method                      | Mô tả                                                   |
| --------------------------- | ------------------------------------------------------- |
| `reset(seed, options)`      | → (obs_dict, info) — generate map, reset backend/reward |
| `step(actions)`             | → (obs, rewards, done, truncated, info)                 |
| `render()`                  | → np.ndarray hoặc None                                  |
| `close()`                   | Cleanup renderer                                        |
| `n_agents` (property)       | Số UAV đang active                                      |
| `active_uav_ids` (property) | List UAV ACTIVE                                         |
| `coverage_rate` (property)  | Tỉ lệ coverage hiện tại                                 |
| `make(cls, ...)`            | Classmethod factory                                     |

**Step flow (BUG-ENV-06 đã fix):**

1. apply_actions → step_physics → step_world
2. **Check done/truncated TRƯỚC reward**
3. Compute rewards (dùng `is_terminal`)
4. Build observations
5. Return

**Done conditions:**

- `coverage ≥ 90%` → done="coverage"
- `all victims found` → done="victims"
- `all UAVs disabled` → done="disabled"
- `step ≥ max_steps` → truncated=True

---

#### `env/sar_pettingzoo_env.py` — `SARPettingZooEnv`

**Vai trò:** PettingZoo `ParallelEnv` wrapper. Convert int keys → str "uav_0", "uav_1", ...

| Method                     | Mô tả                                                                    |
| -------------------------- | ------------------------------------------------------------------------ |
| `reset(seed, options)`     | → (obs_dict[str], infos[str])                                            |
| `step(actions[str])`       | → (obs, rewards, terminations, truncations, infos) — tất cả keyed by str |
| `observation_space(agent)` | → Box(68,)                                                               |
| `action_space(agent)`      | → Box(3,)                                                                |
| `render()`, `close()`      | Delegate to base_env                                                     |
| `unwrapped`                | → SARBaseEnv                                                             |

Factory functions: `make_parallel_env(cfg, **kwargs)`, `make_aec_env(cfg, **kwargs)`

---

#### `env/backends/base_backend.py` — `BaseBackend` (ABC)

Abstract interface với 5 methods: `reset()`, `apply_actions()`, `step_physics()`, `step_world()`, `get_state()`

---

#### `env/backends/logic_backend.py` — `LogicBackend`

**Vai trò:** Pure Python physics backend. ~1000 steps/s.

| Method                   | Mô tả                                                               |
| ------------------------ | ------------------------------------------------------------------- |
| `reset(map_data)`        | Build entities từ map_data (dùng `uav_spawns` pre-generated)        |
| `apply_actions(actions)` | ACTIVE: apply velocity; RETURNING/DEPLOYING: auto_navigate          |
| `step_physics()`         | Battery drain/charge                                                |
| `step_world()`           | Fleet → victims → coverage → detection                              |
| `get_state()`            | Dict với uavs/victims/stations/obstacles/coverage_map/fleet_manager |
| Private `_build_*()`     | Build từng entity type từ dict                                      |

**UAV spawn:** Dùng `map_data["uav_spawns"]` (FIX-P10). Nếu empty → fallback spawn tại stations.

**⚠️ Known issue:** Tất cả UAVs spawn với state=ACTIVE, không có reserve pool ban đầu.

---

### 📁 `visualization/` — Renderers (3 files)

---

#### `visualization/renderer_factory.py`

- `create_renderer(cfg, render_mode, viz_mode)` → `Visualizer2D` hoặc `Visualizer3D`
- Fallback 3D→2D nếu import fail

---

#### `visualization/visualizer2d.py` — `Visualizer2D`

**Vai trò:** Matplotlib 2D renderer. ~50ms/frame (reuse figure).

| Method                                                               | Mô tả                                                                  |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `render(uavs, victims, obstacles, stations, cov_map, step, metrics)` | → np.ndarray hoặc None                                                 |
| `close()`                                                            | Cleanup figure                                                         |
| Private `_draw_*()`                                                  | Coverage, obstacles, stations, victims, UAVs, battery bars, info panel |

**Layout:** [3:1] — Map (trái) + Info panel (phải)

**Color scheme:** UAV=state-based (blue/orange/green/purple), Victim missing=orange X, Victim found=green circle, Debris=brown hatch, Danger=type-based bright colors

---

#### `visualization/visualizer3d.py` — `Visualizer3D`

**Vai trò:** Matplotlib 3D renderer. ~400ms/frame (new figure mỗi frame).

**Layout:** [3:1] — 3D scene (trái) + Dashboard panel (phải)

| Method                                                      | Mô tả                                                                                                     |
| ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `render(uavs, victims, obstacles, stations, cov_map, step)` | → np.ndarray                                                                                              |
| Private `_draw_*()`                                         | Debris (cylinder/box), Danger (cylinder capped <10m), Stations (box), UAVs (scatter + quiver), FOV (cone) |
| `_to_rgb(fig)`                                              | 4 fallback methods để export RGB                                                                          |

Geometry helpers: `_circle_xy`, `_cylinder_faces`, `_box_faces`, `_cone_faces`

---

### 📁 `training/` — Training Pipeline (2/6 files)

---

#### `training/curriculum.py` — `CurriculumManager`

**Vai trò:** Quản lý stage progression. Track metrics và decide khi nào advance.

**`StageStats` dataclass:** `stage_name`, `episodes_done`, lists của coverage/victims/reward → properties `avg_*` (last 50 episodes)

| Method                                   | Mô tả                                       |
| ---------------------------------------- | ------------------------------------------- |
| `current_stage` (property)               | StageConfig hiện tại                        |
| `is_final_stage` (property)              | Đang ở HARD?                                |
| `total_episodes` (property)              | Tổng episodes đã chạy                       |
| `update(coverage, victims_rate, reward)` | Cập nhật stats sau mỗi episode              |
| `should_advance()`                       | eps≥min AND cov≥threshold AND vic≥threshold |
| `advance()`                              | Tăng `_stage_idx`, log transition           |
| `apply_to_config(cfg)`                   | Gọi `cfg.apply_stage(current_stage)`        |
| `get_status()`                           | Dict đầy đủ trạng thái                      |
| `print_status()`                         | In formatted status ra console              |

---

#### `training/curriculum_trainer.py` — `CurriculumTrainer`

**Vai trò:** Training loop với random policy (placeholder cho Phase 2 RL).

| Method                                                | Mô tả                                                             |
| ----------------------------------------------------- | ----------------------------------------------------------------- |
| `train(total_episodes)`                               | Main loop: run episode → update curriculum → check advance → plot |
| `_build_env()`                                        | Tạo SARBaseEnv mới (rebuild khi stage advance)                    |
| `_run_episode(episode_num, render_frames)`            | Chạy 1 episode random policy                                      |
| `_sample_actions(n_uav)`                              | Sample random actions (support both Gymnasium & PettingZoo API)   |
| `_save_episode_visualization(frames, episode, stage)` | Lưu first+last frame PNG (GIF optional)                           |
| `_plot_training_curves(history, episode, final)`      | 4-panel plot: coverage/victims/reward/stage distribution          |
| `_save_gif(frames, path)`                             | Lưu GIF (cần Pillow, slow)                                        |
| `_print_summary(history)`                             | In final stats                                                    |

**History dict:** `episodes`, `coverages`, `victims`, `rewards`, `stages`, `steps`

---

### 📁 `examples/` (2 files)

- `test_pettingzoo.py` — Minimal PettingZoo API test (reset/step/close)
- `record_video.py` — MP4/GIF recording với argparse (--stage, --mode, --steps)

---

## 🔄 EXECUTION FLOW TÓM TẮT

```
reset() →
  MapGenerator.generate(seed)
    → place stations/debris/danger_zones/victims/uav_spawns
  LogicBackend.reset(map_data)
    → build entities → CoverageMap.reset() → FleetManager.reset()
  ObservationBuilder.build_all()
    → actor_obs[68] × 4 UAVs
  return (obs_dict, info)

step(actions) →
  apply_actions() → step_physics() → step_world()
    step_world: FleetManager.step() → victim.update() → coverage.mark() → FOVSensor.check_detected()
  CHECK done/truncated ← (TRƯỚC reward, BUG-ENV-06)
  BaselineReward.compute() → 14 components → clip → total
  ObservationBuilder.build_all()
  return (obs, rewards, done, truncated, info)
```

---

## ⚠️ KNOWN ISSUES

### Issue 1: UAV Spawn Tất Cả ACTIVE (Reserve Pool Empty)

- **Root cause:** `UAV.__init__` default `state=ACTIVE`, logic_backend không truyền state
- **Impact:** Reserve mechanism không hoạt động với random policy
- **Fix Phase 2:** Spawn `n_uav - min_reserve` ACTIVE + `min_reserve` CHARGING

### Issue 2: Reward Dương Với Random Policy

- **Lý do thiết kế:** Coverage delta (+330 avg) > time penalty (-60) → net dương
- **Không phải bug:** Phản ánh đúng task value, RL có clear gradient

### Issue 3: 3D Viz Chậm (~2-5 FPS)

- **Workaround:** Training với `viz_mode="none"`, demo với `viz_mode="3d"`

---

## 📈 BASELINE PERFORMANCE (Random Policy)

| Stage  | Coverage  | Victims Found | Reward     |
| ------ | --------- | ------------- | ---------- |
| EASY   | 55% ± 11% | 53% ± 19%     | +150 ± 200 |
| MEDIUM | 41% ± 9%  | 44% ± 17%     | +80 ± 180  |
| HARD   | 32% ± 8%  | 36% ± 15%     | +30 ± 160  |

**Target RL:**

- EASY: 82-88% coverage, 85-88% victims, +420-450 reward
- MEDIUM: 68-72% coverage
- HARD: 58-62% coverage

---

## 🎯 NEXT STEPS — PHASE 2

### Files cần tạo:

```
training/algorithms/
├── mappo/
│   ├── actor.py       # 68 → 256 → 256 → 3 (Gaussian)
│   ├── critic.py      # 554 → 512 → 256 → 1 (centralized)
│   ├── buffer.py      # Rollout buffer + GAE (λ=0.95)
│   └── trainer.py     # PPO clip ε=0.2, 10 epochs/rollout
├── masac/
│   ├── actor.py       # Off-policy, twin Q
│   ├── critic.py
│   ├── buffer.py      # Replay buffer 1M
│   └── trainer.py     # Entropy tuning, soft update τ=0.005
└── matd3/
    ├── actor.py       # Deterministic policy
    ├── critic.py
    ├── buffer.py
    └── trainer.py     # Delayed update, target smoothing

training/eval.py           # Evaluation protocol
training/train_comparison.py  # Multi-algorithm runner
visualization/comparison_plots.py  # Paper figures
rewards/llm_reward.py      # Phase 3
```

---

## 🛠️ DEVELOPMENT COMMANDS

```bash
# Test toàn bộ
pytest tests/ -v

# Random baseline (50 eps × 3 stages, parallel)
python test_random_parallel.py

# Chạy 1 episode xem visualization
python -c "
from config import AppConfig
from env import SARBaseEnv
env = SARBaseEnv(AppConfig(), viz_mode='2d', render_mode='human')
obs, _ = env.reset(seed=42)
for _ in range(100):
    actions = {i: env.action_space.sample() for i in range(4)}
    env.step(actions)
    env.render()
env.close()
"

# Curriculum training (random policy placeholder)
python -c "
from training import CurriculumTrainer
from config import AppConfig
trainer = CurriculumTrainer(AppConfig(), render_every=0)
trainer.train(100)
"
```

---

## 📐 KEY NUMBERS ĐỂ NHỚ

| Metric                | Giá trị                                                         |
| --------------------- | --------------------------------------------------------------- |
| Actor obs dim         | **68** (với n_stations=2)                                       |
| Critic obs dim        | **554** (8×68+10)                                               |
| Action space          | **3** dims ∈ [-1,1]                                             |
| UAV states            | **5**: ACTIVE/RETURNING/CHARGING/DEPLOYING/DISABLED             |
| Reward components     | **14**                                                          |
| Test pass             | **26/26**                                                       |
| Backend speed         | **~1000 steps/s**                                               |
| Episode time (random) | **~9s** (300 steps × 30ms)                                      |
| Difficulty metric     | `coverage_pressure = map_area / n_uav`                          |
| Victim density        | **~0.53/1000m²** (constant across stages — controlled variable) |

vậy bạn hayx tổng hợp những cái t đã làm ở project này từ đầu tới cuối ghi chi tiết rõ ràng từng folder có file nào file có hàm và thuộc tính nào có tác dụng gì một cách chi tiết (ko cần code) và ghi rõ ràng chi tiết để khi qua đoạn chat ms thì chỉ cần đưa cái đó cho nớ là nó sẽ hiểu đang làm gì và đang thực hiện cái gì biết đang thực heienj cái gì theo format này
