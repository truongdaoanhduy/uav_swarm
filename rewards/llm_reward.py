# rewards/llm_reward.py

import numpy as np
from entities.uav import UAVState


class LLMReward:
    """
    Wrap reward function do LLM sinh ra.
    Interface giống BaselineReward để env gọi được.
    
    VERSION: 18 factors (bổ sung urgency + exploration tracking)
    """

    def __init__(self, fn, cfg):
        """
        fn  : callable từ exec(llm_reward_generated.py)
        cfg : AppConfig (để lấy max_steps, map_size...)
        """
        self._fn  = fn
        self._cfg = cfg
        
        # ── Tracking state per episode ──────────────────────────────────
        self._dead_penalized = set()  # Tránh penalty battery death 2 lần
        self._prev_local_cov = {}     # Track local coverage change
        self._idle_steps     = {}     # Track consecutive no-progress steps

    # ══════════════════════════════════════════════════════════════════════════
    # RESET
    # ══════════════════════════════════════════════════════════════════════════
    
    def reset(self):
        """Reset per-episode tracking state."""
        self._dead_penalized.clear()
        self._prev_local_cov.clear()
        self._idle_steps.clear()

    # ══════════════════════════════════════════════════════════════════════════
    # COMPUTE PER UAV (Main Interface)
    # ══════════════════════════════════════════════════════════════════════════
    
    def compute_per_uav(
        self,
        uav,
        newly_found_by_uav,
        uavs,
        victims,
        obstacles,          # Không dùng nhưng giữ để interface khớp
        coverage_map,
        fleet_manager,      # Không dùng nhưng giữ để interface khớp
        prev_coverage,
        current_step,
        done,
        stations=None,
    ) -> dict:
        """
        Tính reward cho 1 UAV. Trả về dict có key 'total'.
        
        Flow:
            1. Check UAV state
            2. Build 18 factors
            3. Call LLM function
            4. Clip & return
        """
        # UAV disabled → không tính
        if uav.state == UAVState.DISABLED:
            return {"total": 0.0}

        # ── Step 1: Build factors ────────────────────────────────────────
        factors = self._build_factors(
            uav                = uav,
            newly_found_by_uav = newly_found_by_uav,
            uavs               = uavs,
            victims            = victims,
            coverage_map       = coverage_map,
            prev_coverage      = prev_coverage,
            current_step       = current_step,
            max_steps          = self._cfg.env.max_steps,
            stations           = stations,
        )

        # ── Step 2: Call LLM function ────────────────────────────────────
        try:
            reward = float(self._fn(factors))
            reward = float(np.clip(reward, -200.0, 200.0))
        except Exception as e:
            print(f"[LLMReward] UAV {uav.id} lỗi: {e}")
            reward = 0.0

        return {"total": reward}

    # ══════════════════════════════════════════════════════════════════════════
    # COMPUTE GLOBAL (for logging)
    # ══════════════════════════════════════════════════════════════════════════
    
    def compute(
        self,
        uavs,
        victims,
        obstacles,
        coverage_map,
        fleet_manager,
        newly_found,
        prev_coverage,
        current_step,
        done,
        stations=None,
    ) -> dict:
        """
        Global reward cho logging.
        base_env.step() gọi cái này để accumulate episode reward.
        """
        if not uavs:
            return {"total": 0.0}

        total     = 0.0
        n_counted = 0

        for uav in uavs:
            if uav.state == UAVState.DISABLED:
                continue

            # Filter victims found by this UAV
            newly_found_by_uav = [
                v for v in newly_found if v.found_by_uav == uav.id
            ]

            breakdown = self.compute_per_uav(
                uav                = uav,
                newly_found_by_uav = newly_found_by_uav,
                uavs               = uavs,
                victims            = victims,
                obstacles          = obstacles,
                coverage_map       = coverage_map,
                fleet_manager      = fleet_manager,
                prev_coverage      = prev_coverage,
                current_step       = current_step,
                done               = done,
                stations           = stations,
            )
            total += breakdown["total"]
            n_counted += 1

        mean_reward = total / max(n_counted, 1)

        return {
            "total":  mean_reward,
            "n_uavs": n_counted,
        }

    def summarize(self, breakdown: dict) -> str:
        """Log summary."""
        return f"llm_total={breakdown.get('total', 0):.2f}"

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD FACTORS - 18 FACTORS
    # ══════════════════════════════════════════════════════════════════════════
    
    def _build_factors(
        self,
        uav,
        newly_found_by_uav,
        uavs,
        victims,
        coverage_map,
        prev_coverage,
        current_step,
        max_steps,
        stations,
    ) -> list:
        """
        Chuyển tất cả objects từ env sang list 18 số.
        
        FACTORS (18 total):
        ───────────────────────────────────────────────────────────────
        [0]  battery_pct          : Current battery % (0-100)
        [1]  coverage_delta       : Global coverage increase this step (0-1)
        [2]  local_cov            : Coverage within 30m of UAV (0-1)
        [3]  n_found              : Number of victims found THIS STEP
        [4]  avg_urgency          : Average urgency of victims found (0-5)
        [5]  dist_unfound         : Distance to nearest unfound victim (m)
        [6]  dist_station         : Distance to nearest charging station (m)
        [7]  is_charging          : 1.0 if charging, else 0.0
        [8]  is_dead              : 1.0 if battery died THIS STEP, else 0.0
        [9]  n_nearby             : Number of teammates within 30m
        [10] time_ratio           : Episode progress (0=start, 1=end)
        [11] n_active             : Number of active UAVs
        ───────────────────────────────────────────────────────────────
        NEW FACTORS (urgency + exploration tracking):
        ───────────────────────────────────────────────────────────────
        [12] high_urgency_unfound : Count of urgency≥4 victims not found
        [13] dist_high_urgency    : Distance to nearest urgency≥4 victim (m)
        [14] local_cov_delta      : Coverage increase in 30m radius this step
        [15] idle_steps           : Consecutive steps without progress
        [16] high_urgency_ratio   : % of urgency≥4 victims found (0-1)
        [17] neighbor_coverage    : Avg coverage of 20m neighbors (0-1)
        ───────────────────────────────────────────────────────────────
        """
        
        # ══════════════════════════════════════════════════════════════════
        # ORIGINAL 12 FACTORS (unchanged)
        # ══════════════════════════════════════════════════════════════════
        
        # [0] battery_pct
        battery_pct = float(uav.battery_pct)

        # [1] coverage_delta
        cur_cov        = coverage_map.get_coverage_rate()
        coverage_delta = float(max(0.0, cur_cov - prev_coverage))

        # [2] local_cov
        local_cov = float(coverage_map.get_local_coverage(uav.pos, 30.0))

        # [3] n_found
        n_found = len(newly_found_by_uav)

        # [4] avg_urgency
        avg_urgency = (
            sum(v.urgency for v in newly_found_by_uav) / n_found
            if n_found > 0 else 0.0
        )

        # [5] dist_unfound
        unfound = [v for v in victims if not v.is_found]
        if unfound:
            dist_unfound = float(min(
                np.linalg.norm(uav.pos[:2] - v.pos[:2])
                for v in unfound
            ))
        else:
            dist_unfound = 353.0  # sqrt(500^2 + 500^2) ≈ 707, use 353 as half

        # [6] dist_station
        if stations:
            dist_station = float(min(
                np.linalg.norm(uav.pos[:2] - s.pos[:2])
                for s in stations
            ))
        else:
            dist_station = 353.0

        # [7] is_charging
        is_charging = 1.0 if uav.state == UAVState.CHARGING else 0.0

        # [8] is_dead (one-time)
        if uav.battery_death and uav.id not in self._dead_penalized:
            is_dead = 1.0
            self._dead_penalized.add(uav.id)
        else:
            is_dead = 0.0

        # [9] n_nearby
        active = [u for u in uavs if u.state != UAVState.DISABLED]
        n_nearby = float(sum(
            1 for u in active
            if u.id != uav.id
            and np.linalg.norm(uav.pos[:2] - u.pos[:2]) <= 30.0
        ))

        # [10] time_ratio
        time_ratio = float(current_step) / float(max(max_steps, 1))

        # [11] n_active
        n_active = float(len(active))

        # ══════════════════════════════════════════════════════════════════
        # NEW FACTORS (12-17)
        # ══════════════════════════════════════════════════════════════════
        
        # [12] high_urgency_unfound
        # Đếm số victim có urgency ≥ 4 chưa tìm thấy
        high_urgency_unfound = float(sum(
            1 for v in victims 
            if not v.is_found and v.urgency >= 4
        ))

        # [13] dist_high_urgency
        # Khoảng cách đến victim urgency ≥ 4 gần nhất
        high_urgency_victims = [
            v for v in victims 
            if not v.is_found and v.urgency >= 4
        ]
        if high_urgency_victims:
            dist_high_urgency = float(min(
                np.linalg.norm(uav.pos[:2] - v.pos[:2])
                for v in high_urgency_victims
            ))
        else:
            dist_high_urgency = 353.0  # Không còn victim urgency cao

        # [14] local_cov_delta
        # Coverage tăng trong vùng 30m (detect local exploration)
        prev_local = self._prev_local_cov.get(uav.id, local_cov)
        local_cov_delta = float(max(0.0, local_cov - prev_local))
        self._prev_local_cov[uav.id] = local_cov

        # [15] idle_steps
        # Số bước liên tiếp KHÔNG có tiến triển (no victim + no coverage)
        # CRITICAL: Đây là key factor để prevent hovering/wasting time
        has_progress = (n_found > 0) or (coverage_delta > 0.001)
        
        if not has_progress:
            self._idle_steps[uav.id] = self._idle_steps.get(uav.id, 0) + 1
        else:
            self._idle_steps[uav.id] = 0
        
        idle_steps = float(self._idle_steps.get(uav.id, 0))

        # [16] high_urgency_ratio
        # % victim urgency ≥ 4 đã tìm thấy (task progress metric)
        total_high_urgency = max(
            sum(1 for v in victims if v.urgency >= 4), 
            1  # Avoid division by zero
        )
        found_high_urgency = sum(
            1 for v in victims 
            if v.is_found and v.urgency >= 4
        )
        high_urgency_ratio = float(found_high_urgency / total_high_urgency)

        # [17] neighbor_coverage
        # Coverage trung bình vùng xung quanh (repulsion từ explored areas)
        neighbor_coverage = float(
            coverage_map.get_neighbor_coverage(uav.pos, radius=20.0)
        )

        # ══════════════════════════════════════════════════════════════════
        # RETURN 18 FACTORS
        # ══════════════════════════════════════════════════════════════════
        
        return [
            battery_pct,           # [0]
            coverage_delta,        # [1]
            local_cov,             # [2]
            float(n_found),        # [3]
            float(avg_urgency),    # [4]
            dist_unfound,          # [5]
            dist_station,          # [6]
            is_charging,           # [7]
            is_dead,               # [8]
            n_nearby,              # [9]
            time_ratio,            # [10]
            n_active,              # [11]
            # ── NEW ──
            high_urgency_unfound,  # [12]
            dist_high_urgency,     # [13]
            local_cov_delta,       # [14]
            idle_steps,            # [15]
            high_urgency_ratio,    # [16]
            neighbor_coverage,     # [17]
        ]


# ══════════════════════════════════════════════════════════════════════════════
# LOAD FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def load_llm_reward(filepath: str, cfg) -> "LLMReward":
    """
    Load LLMReward từ file code đã lưu.
    
    Usage:
        reward = load_llm_reward("llm_reward_generated.py", cfg)
        reward.compute_per_uav(uav, ...)
    """
    with open(filepath, "r") as f:
        code = f.read()

    ns = {"np": np}
    exec(code, ns)

    fn = ns.get("reward_func")
    if not callable(fn):
        raise RuntimeError(f"Không tìm thấy reward_func trong {filepath}")

    print(f"✅ Loaded LLM reward từ {filepath}")
    return LLMReward(fn=fn, cfg=cfg)