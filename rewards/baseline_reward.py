# rewards/simple_baseline.py
"""
Baseline Reward - Minimal Version
Compatible với base_env.py interface.
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from config import AppConfig
    from entities.uav import UAV
    from entities.victim import BaseVictim

from entities.uav import UAVState


class SimpleBaselineReward:
    """
    Baseline reward tối giản - 5 core components.
    
    ✅ Compatible với base_env.py (cùng interface với BaselineReward)
    ✅ Dùng **kwargs để bỏ qua params không cần
    ✅ Có reset(), compute(), compute_per_uav(), summarize()
    """

    def __init__(self, cfg: "AppConfig") -> None:
        self.cfg = cfg

        # ── Lấy từ config (không hardcode) ──────────────────────
        self._r_coverage = cfg.reward.r_coverage_delta   # 30.0
        self._r_victim   = cfg.reward.r_victim_base      # 50.0
        self._r_bat_dead = cfg.reward.r_battery_dead     # -50.0
        self._r_time     = cfg.reward.r_time_penalty     # -0.2
        self._r_terminal = cfg.reward.terminal_bonus_cap # 200.0
        self._clip_min   = cfg.reward.step_reward_clip_min  # -30.0
        self._clip_max   = cfg.reward.step_reward_clip_max  # +200.0

        # One-time tracking
        self._dead_uavs: set = set()

    # ── Reset (BẮT BUỘC - base_env gọi mỗi episode) ─────────────
    def reset(self) -> None:
        self._dead_uavs.clear()

    # ── Per-UAV reward (BẮT BUỘC) ────────────────────────────────
    def compute_per_uav(
        self,
        uav:                "UAV",
        newly_found_by_uav: List["BaseVictim"],
        uavs:               List["UAV"],
        victims:            List["BaseVictim"],
        coverage_map,
        prev_coverage:      float,
        current_step:       int,
        done:               bool,
        # ✅ Params không dùng nhưng phải nhận để compatible
        obstacles:          Optional[List] = None,
        fleet_manager                      = None,
        stations:           Optional[List] = None,
    ) -> Dict[str, float]:
        """Per-UAV reward - compatible với base_env.py."""

        # Skip disabled
        if uav.state == UAVState.DISABLED:
            return {"total": 0.0}

        reward  = 0.0
        n_active = max(
            sum(1 for u in uavs if u.state != UAVState.DISABLED), 1
        )
        cur_cov = coverage_map.get_coverage_rate()

        # ── 1. Coverage (shared / n_active) ─────────────────────
        cov_delta = max(0.0, cur_cov - prev_coverage)
        reward += (cov_delta * self._r_coverage) / n_active

        # ── 2. Victim (individual) ───────────────────────────────
        if newly_found_by_uav:
            urgency_sum = sum(v.urgency / 5.0 for v in newly_found_by_uav)
            reward += self._r_victim * urgency_sum

        # ── 3. Battery penalty (progressive) ────────────────────
        bat = uav.battery_pct
        if bat <= 5.0:
            reward -= 8.0
        elif bat <= 10.0:
            reward -= 2.0
        elif bat <= 20.0:
            reward -= 0.5

        # ── 4. Battery death (one-time) ──────────────────────────
        if uav.battery_death and uav.id not in self._dead_uavs:
            reward += self._r_bat_dead
            self._dead_uavs.add(uav.id)

        # ── 5. Time penalty ──────────────────────────────────────
        reward += self._r_time

        # ── 6. Terminal bonus (shared / n_active) ────────────────
        if done:
            n_found = sum(1 for v in victims if v.is_found)
            n_total = max(len(victims), 1)

            bonus = self._r_terminal * (
                0.5 * cur_cov +
                0.5 * (n_found / n_total)
            )
            reward += bonus / n_active

        # ── Clip ─────────────────────────────────────────────────
        reward = float(np.clip(reward, self._clip_min, self._clip_max))

        return {"total": reward}

    # ── Global reward (BẮT BUỘC - base_env gọi để log) ──────────
    def compute(
        self,
        uavs:          List["UAV"],
        victims:       List["BaseVictim"],
        coverage_map,
        newly_found:   List["BaseVictim"],
        prev_coverage: float,
        current_step:  int,
        done:          bool,
        # Params không dùng nhưng phải nhận
        obstacles:     Optional[List] = None,
        fleet_manager                 = None,
        stations:      Optional[List] = None,
    ) -> Dict[str, float]:
        """Global reward - sum per-UAV rewards."""

        total    = 0.0
        n_counted = 0

        for uav in uavs:
            if uav.state == UAVState.DISABLED:
                continue

            newly_found_by_uav = [
                v for v in newly_found
                if v.found_by_uav == uav.id
            ]

            breakdown = self.compute_per_uav(
                uav                = uav,
                newly_found_by_uav = newly_found_by_uav,
                uavs               = uavs,
                victims            = victims,
                coverage_map       = coverage_map,
                prev_coverage      = prev_coverage,
                current_step       = current_step,
                done               = done,
                obstacles          = obstacles,
                fleet_manager      = fleet_manager,
                stations           = stations,
            )
            total     += breakdown["total"]
            n_counted += 1

        mean = total / max(n_counted, 1)
        return {"total": mean}

    # ── Utils (BẮT BUỘC - base_env gọi) ─────────────────────────
    def get_component_names(self) -> List[str]:
        return ["total"]

    def summarize(self, reward_dict: Dict[str, float]) -> str:
        return f"simple_reward={reward_dict.get('total', 0):+.2f}"