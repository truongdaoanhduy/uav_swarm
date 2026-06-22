"""
rewards/baseline_reward.py
Baseline Reward Function v4.0 - Compact

Key Changes:
  - Coverage delta: 30 (dominant)
  - Victim: urgency-scaled
  - Landing: 5 (small, anti-exploit)
  - Time penalty: -0.5/step
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np

from entities.uav import UAV, UAVState
from utils.geometry import dist_2d, dist_3d

if TYPE_CHECKING:
    from config import AppConfig
    from core.coverage_map import CoverageMap
    from entities.victim import BaseVictim

logger = logging.getLogger(__name__)


class BaselineReward:
    """Baseline Reward v4.0 - Anti-exploit design."""

    def __init__(self, cfg: "AppConfig") -> None:
        self.cfg = cfg
        
        # Reward weights từ config
        self._r_victim_base    = cfg.reward.r_victim_base
        self._r_coverage_delta = cfg.reward.r_coverage_delta
        self._r_battery_dead   = cfg.reward.r_battery_dead
        self._r_collision_obs  = cfg.reward.r_collision_obstacle
        self._r_time_penalty   = cfg.reward.r_time_penalty
        self._r_landing        = cfg.reward.r_landing_success
        
        # Caps
        self._clip_min = cfg.reward.step_reward_clip_min
        self._clip_max = cfg.reward.step_reward_clip_max
        
        # Tracking
        self._battery_death_penalized: Set[int] = set()
        self._collision_penalized:     Set[int] = set()
        self._landed_uavs:             Set[int] = set()
        
        logger.info(
            f"BaselineReward v4.0: coverage={self._r_coverage_delta}, "
            f"victim={self._r_victim_base}, landing={self._r_landing}"
        )

    def reset(self) -> None:
        """Reset per-episode tracking."""
        self._battery_death_penalized.clear()
        self._collision_penalized.clear()
        self._landed_uavs.clear()

    # ═════════════════════════════════════════════════════════════════════════
    # MAIN API
    # ═════════════════════════════════════════════════════════════════════════

    def compute_per_uav(
        self,
        uav,
        newly_found_by_uav,
        uavs,
        victims,
        obstacles,
        coverage_map,
        fleet_manager,
        prev_coverage,
        current_step,
        done,
        stations=None,
    ) -> Dict[str, float]:
        """Compute per-UAV reward (MAPPO)."""
        
        if uav.state == UAVState.DISABLED:
            return {"total": 0.0}
        
        active = [u for u in uavs if u.state != UAVState.DISABLED]
        n_active = max(len(active), 1)
        cur_cov = coverage_map.get_coverage_rate()
        
        reward = 0.0
        
        # 1. Coverage (shared)
        cov_delta = max(0.0, cur_cov - prev_coverage)
        reward += (cov_delta * self._r_coverage_delta) / n_active
        
        # 2. Victim (individual, urgency-scaled)
        if newly_found_by_uav:
            scale = 2.0 - cur_cov  # Early: 2.0, Late: 1.0
            urgency_sum = sum(v.urgency / 5.0 for v in newly_found_by_uav)
            reward += self._r_victim_base * scale * urgency_sum
        
        # 3. Battery death (one-time)
        if uav.battery_death and uav.id not in self._battery_death_penalized:
            reward += self._r_battery_dead
            self._battery_death_penalized.add(uav.id)
        
        # 4. Battery low penalty
        if uav.battery_pct <= 20 and uav.state == UAVState.ACTIVE:
            if uav.battery_pct <= 5:
                reward -= 8.0
            elif uav.battery_pct <= 10:
                reward -= 2.0
            else:
                reward -= 0.5
        
        # 5. Landing (small)
        if (uav.state == UAVState.CHARGING 
            and uav.id not in self._landed_uavs
            and uav.battery_pct < 30):
            reward += self._r_landing
            self._landed_uavs.add(uav.id)
        
        # 6. Collision (one-time)
        from entities.obstacle import Debris
        if uav.id not in self._collision_penalized:
            for obs in obstacles:
                if isinstance(obs, Debris) and obs.causes_collision(uav.pos):
                    reward += self._r_collision_obs
                    self._collision_penalized.add(uav.id)
                    break
        
        # 7. Proximity penalty
        for other in active:
            if other.id != uav.id:
                d = dist_3d(uav.pos, other.pos)
                if d <= 1.0:
                    reward -= 10.0
                elif d <= 2.0:
                    reward -= 5.0
                elif d <= 3.0:
                    reward -= 2.0
        
        # 8. Time penalty
        reward += self._r_time_penalty
        
        # 9. Terminal bonus (shared)
        if done:
            reward += self._terminal_bonus(cur_cov, victims, current_step, uavs) / n_active
        
        # Clip
        reward = float(np.clip(reward, self._clip_min, self._clip_max))
        
        return {"total": reward}

    def compute(self, uavs, victims, obstacles, coverage_map, fleet_manager,
                newly_found, prev_coverage, current_step, done, stations=None):
        """Compute global reward (logging only)."""
        
        total = 0.0
        n = 0
        
        for uav in uavs:
            if uav.state == UAVState.DISABLED:
                continue
            
            newly_found_by_uav = [v for v in newly_found if v.found_by_uav == uav.id]
            
            r = self.compute_per_uav(
                uav, newly_found_by_uav, uavs, victims, obstacles,
                coverage_map, fleet_manager, prev_coverage, current_step, done, stations
            )
            total += r["total"]
            n += 1
        
        return {"total": total / max(n, 1), "n_uavs": n}

    # ═════════════════════════════════════════════════════════════════════════
    # TERMINAL BONUS
    # ═════════════════════════════════════════════════════════════════════════

    def _terminal_bonus(self, coverage_rate, victims, current_step, uavs=None):
        """Terminal bonus: proportional to coverage + victims found."""
        
        n_total = max(len(victims), 1)
        n_found = sum(1 for v in victims if v.is_found)
        found_ratio = n_found / n_total
        time_ratio = current_step / max(self.cfg.env.max_steps, 1)
        
        bonus = 0.0
        bonus += 200 * 0.5 * coverage_rate       # 50% weight coverage
        bonus += 200 * 0.3 * found_ratio         # 30% weight victims
        bonus += 200 * 0.1 * (1.0 - time_ratio)  # 10% weight time
        
        # Battery survival
        if uavs:
            alive = [u for u in uavs if u.state != UAVState.DISABLED]
            if alive:
                mean_bat = np.mean([u.battery_pct for u in alive])
                bonus += 200 * 0.1 * (mean_bat / 100.0)
        
        return float(np.clip(bonus, 0.0, 200.0))

    # ═════════════════════════════════════════════════════════════════════════
    # UTILS
    # ═════════════════════════════════════════════════════════════════════════

    def get_component_names(self):
        return ["total"]

    def summarize(self, reward_dict):
        return f"total={reward_dict.get('total', 0):.2f}"

    def __repr__(self):
        return f"BaselineReward(v4.0, cov={self._r_coverage_delta}, vic={self._r_victim_base})"