# rewards/llm_reward.py

import numpy as np
from entities.uav import UAVState


class LLMReward:
    """
    Wrap reward function do LLM sinh ra.
    Interface giống BaselineReward để env gọi được.
    """

    def __init__(self, fn, cfg):
        """
        fn  : callable từ exec(llm_reward_generated.py)
        cfg : AppConfig (để lấy max_steps, map_size...)
        """
        self._fn  = fn
        self._cfg = cfg
        self._dead_penalized = set()  # Tránh penalty battery death 2 lần

    # ── Gọi đầu mỗi episode ──────────────────────────────────
    def reset(self):
        self._dead_penalized.clear()

    # ── Interface chính - env gọi cái này ────────────────────
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
        """Tính reward cho 1 UAV. Trả về dict có key 'total'."""

        # UAV disabled → không tính
        if uav.state == UAVState.DISABLED:
            return {"total": 0.0}

        # Bước 1: Chuyển objects → list số
        factors = self._build_factors(
            uav, newly_found_by_uav, uavs,
            victims, coverage_map,
            prev_coverage, current_step,
            self._cfg.env.max_steps, stations
        )

        # Bước 2: Gọi fn LLM
        try:
            reward = float(self._fn(factors))
            reward = float(np.clip(reward, -200.0, 200.0))
        except Exception as e:
            print(f"[LLMReward] UAV {uav.id} lỗi: {e}")
            reward = 0.0

        return {"total": reward}

    # ── Chuyển objects → list số ─────────────────────────────
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
        Đây là bước quan trọng nhất.
        Chuyển tất cả objects từ env sang list 12 số
        để fn LLM hiểu được.
        """

        # factors[0]: battery hiện tại
        battery_pct = float(uav.battery_pct)

        # factors[1]: diện tích mới khám phá step này
        cur_cov = coverage_map.get_coverage_rate()
        coverage_delta = float(max(0.0, cur_cov - prev_coverage))

        # factors[2]: vùng 30m quanh UAV đã scan bao nhiêu
        local_cov = float(
            coverage_map.get_local_coverage(uav.pos, 30.0)
        )

        # factors[3]: số victim tìm thấy step này
        n_found = len(newly_found_by_uav)

        # factors[4]: urgency trung bình của victim vừa tìm
        avg_urgency = (
            sum(v.urgency for v in newly_found_by_uav) / n_found
            if n_found > 0 else 0.0
        )

        # factors[5]: khoảng cách đến victim chưa tìm gần nhất
        unfound = [v for v in victims if not v.is_found]
        if unfound:
            dist_unfound = float(min(
                np.linalg.norm(uav.pos[:2] - v.pos[:2])
                for v in unfound
            ))
        else:
            dist_unfound = 353.0  # Không còn victim

        # factors[6]: khoảng cách đến trạm sạc gần nhất
        if stations:
            dist_station = float(min(
                np.linalg.norm(uav.pos[:2] - s.pos[:2])
                for s in stations
            ))
        else:
            dist_station = 353.0

        # factors[7]: đang sạc không
        is_charging = 1.0 if uav.state == UAVState.CHARGING else 0.0

        # factors[8]: battery vừa chết step này (one-time)
        if uav.battery_death and uav.id not in self._dead_penalized:
            is_dead = 1.0
            self._dead_penalized.add(uav.id)
        else:
            is_dead = 0.0

        # factors[9]: số teammate trong 30m
        active = [u for u in uavs if u.state != UAVState.DISABLED]
        n_nearby = float(sum(
            1 for u in active
            if u.id != uav.id
            and np.linalg.norm(uav.pos[:2] - u.pos[:2]) <= 30.0
        ))

        # factors[10]: tiến độ episode (0=đầu, 1=cuối)
        time_ratio = float(current_step) / float(max(max_steps, 1))

        # factors[11]: số UAV còn hoạt động
        n_active = float(len(active))

        return [
            battery_pct,        # [0]
            coverage_delta,     # [1]
            local_cov,          # [2]
            float(n_found),     # [3]
            float(avg_urgency), # [4]
            dist_unfound,       # [5]
            dist_station,       # [6]
            is_charging,        # [7]
            is_dead,            # [8]
            n_nearby,           # [9]
            time_ratio,         # [10]
            n_active,           # [11]
        ]


# ── Load fn từ file đã lưu ───────────────────────────────────

def load_llm_reward(filepath: str, cfg) -> "LLMReward":
    """
    Load LLMReward từ file code đã lưu.
    
    Dùng:
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