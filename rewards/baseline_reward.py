"""
rewards/baseline_reward.py
Baseline Reward Function - v4.0 Anti-Exploit

BREAKING CHANGES vs v3.x:
    - Landing reward: 20+bonus → 5 (fixed, no bonus)
    - Coverage delta: 15 → 30 (dominant signal)
    - Approach weight: 0.5 → 0.05 (tiny nudge only)
    - Early bonus: REMOVED (exploit prevention)
    - All values từ config (không hardcode)

DESIGN PRINCIPLES v4.0:
    1. Coverage phải là signal CHÍNH (chiếm ~65% max reward)
    2. Landing là survival, không phải goal (~1% max reward)
    3. Proportional reward: 28% coverage → thấp, 80% → cao
    4. No exploit: không reward nào dễ farm hơn coverage/victim
    5. Sparse signals > Dense penalties (giữ từ v3)
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
    from core.fleet_manager import FleetManager
    from entities.victim import BaseVictim

logger = logging.getLogger(__name__)

# 4 UAVs → max 6 pairs
_BASELINE_PAIRS = 6.0


class BaselineReward:
    """
    Baseline Reward Function v4.0

    Anti-exploit redesign:
        - Landing reward nhỏ (5), không có bonus
        - Coverage reward lớn (30), dominant signal
        - Approach reward nhỏ (0.05), chỉ nudge
    """

    def __init__(self, cfg: "AppConfig") -> None:
        self.cfg = cfg

        # ── Cache reward params từ config ────────────────────────────────────
        self._r_victim_base    = cfg.reward.r_victim_base
        self._r_coverage_delta = cfg.reward.r_coverage_delta
        self._r_battery_dead   = cfg.reward.r_battery_dead
        self._r_collision_obs  = cfg.reward.r_collision_obstacle
        self._r_proximity_1m   = cfg.reward.r_proximity_1m
        self._r_proximity_2m   = cfg.reward.r_proximity_2m
        self._r_proximity_3m   = cfg.reward.r_proximity_3m
        self._r_time_penalty   = cfg.reward.r_time_penalty
        self._r_terminal_base  = cfg.reward.r_terminal_base

        # ── Caps & limits ────────────────────────────────────────────────────
        self._step_penalty_cap = cfg.reward.step_penalty_cap
        self._proximity_cap    = cfg.reward.proximity_penalty_cap
        self._terminal_cap     = cfg.reward.terminal_bonus_cap
        self._clip_min         = cfg.reward.step_reward_clip_min
        self._clip_max         = cfg.reward.step_reward_clip_max

        # ── Shaping ──────────────────────────────────────────────────────────
        self._enable_shaping = cfg.reward.enable_distance_shaping
        self._shaping_weight = 0.1
        self._shaping_max    = cfg.reward.distance_shaping_max_per_uav

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
        self._prev_min_dist:           Dict[int, float] = {}
        self._landed_uavs:             Set[int] = set()

        # ── Landing params - TỪ CONFIG (không hardcode) ──────────────────────
        # v4.0: Tất cả landing params đến từ RewardConfig
        self._r_landing_success = cfg.reward.r_landing_success  # 5.0
        self._approach_weight   = cfg.reward.r_approach_weight   # 0.05
        self._r_hover_penalty   = cfg.reward.r_hover_penalty     # -2.0
        self._landing_range     = cfg.env.charge_radius * 2.0    # 6.0m

        logger.info(
            "BaselineReward v4.0: "
            "coverage_delta=%.1f, victim_base=%.1f, "
            "landing=%.1f, approach=%.3f, "
            "clip=[%.0f, %.0f]",
            self._r_coverage_delta,
            self._r_victim_base,
            self._r_landing_success,
            self._approach_weight,
            self._clip_min,
            self._clip_max,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═════════════════════════════════════════════════════════════════════════

    def reset(self) -> None:
        """Reset per-episode state."""
        self._battery_death_penalized.clear()
        self._collision_penalized.clear()
        self._prev_min_dist.clear()
        self._landed_uavs.clear()
    def compute(
        self,
        uavs:          List[UAV],
        victims:       List["BaseVictim"],
        obstacles:     List,
        coverage_map:  "CoverageMap",
        fleet_manager: "FleetManager",
        newly_found:   List["BaseVictim"],
        prev_coverage: float,
        current_step:  int,
        done:          bool,
        stations:      Optional[List] = None,
    ) -> Dict[str, float]:
        """
        Compute global reward.
        Dùng cho logging, không dùng để train trực tiếp.
        """
        cur_coverage = coverage_map.get_coverage_rate()
        active_uavs  = [u for u in uavs if u.state != UAVState.DISABLED]
        n_active     = max(len(active_uavs), 1)

        components: Dict[str, float] = dict.fromkeys(
            self.get_component_names(), 0.0
        )

        # ── 1. Coverage (DOMINANT) ───────────────────────────────────────────
        components["coverage_delta"] = _coverage_delta_reward(
            prev_coverage, cur_coverage, self._r_coverage_delta
        )

        # ── 2. Victims ───────────────────────────────────────────────────────
        components["victim_found"] = _victim_found_reward(
            newly_found, self._r_victim_base, cur_coverage  # ← Pass coverage
        )

        # ── 3. Distance shaping ──────────────────────────────────────────────
        if self._enable_shaping:
            components["distance_shaping"] = self._delta_shaping_fleet(
                uavs, victims
            )

        # ── 4. Battery penalties ─────────────────────────────────────────────
        bat_pen, bat_dead = self._battery_rewards(uavs, stations)
        components["battery_penalty"] = bat_pen
        components["battery_dead"]    = bat_dead

        # ── 5. Collision ─────────────────────────────────────────────────────
        components["collision_obstacle"] = self._collision_reward(
            uavs, obstacles
        )

        # ── 6. Proximity ─────────────────────────────────────────────────────
        n_pairs = max(n_active * (n_active - 1) / 2, 1)
        scaled_prox_cap = self._proximity_cap * (n_pairs / _BASELINE_PAIRS)
        proximity_raw = _proximity_reward(
            active_uavs,
            self._PROX_1M, self._PROX_2M, self._PROX_3M,
            self._r_proximity_1m, self._r_proximity_2m, self._r_proximity_3m,
        )
        components["proximity"] = max(proximity_raw, scaled_prox_cap)

        # ── 7. Danger zone ───────────────────────────────────────────────────
        components["danger_zone"] = self._danger_reward(uavs, obstacles)

        # ── 8. Landing rewards (v4.0: nhỏ, không exploit) ───────────────────
        land_r, hover_p, approach_r = self._landing_rewards(uavs, stations)
        components["landing_reward"]  = land_r
        components["hover_penalty"]   = hover_p
        components["approach_reward"] = approach_r

        # ── 9. Deprecated ────────────────────────────────────────────────────
        components["fleet_incentive"] = 0.0

        # ── 10. Time penalty ─────────────────────────────────────────────────
        components["time_penalty"] = self._r_time_penalty * n_active

        # ── 11. Terminal bonus ───────────────────────────────────────────────
        if done:
            components["terminal"] = self._terminal_bonus(
                cur_coverage, victims, current_step, uavs=uavs
            )

        # ── 12. Penalty cap (additive) ───────────────────────────────────────
        components = self._apply_penalty_cap(components)

        # ── 13. Total + clip ─────────────────────────────────────────────────
        components = self._finalize(components)

        self._log_step_breakdown(components, current_step)

        return components

    def compute_per_uav(
        self,
        uav:                UAV,
        newly_found_by_uav: List["BaseVictim"],
        uavs:               List[UAV],
        victims:            List["BaseVictim"],
        obstacles:          List,
        coverage_map:       "CoverageMap",
        fleet_manager:      "FleetManager",
        prev_coverage:      float,
        current_step:       int,
        done:               bool,
        stations:           Optional[List] = None,
    ) -> Dict[str, float]:
        """
        Compute per-agent reward cho MAPPO.

        Per-agent design:
            coverage  → shared / n_active
            victim    → individual (chỉ agent tìm thấy)
            penalties → individual
            terminal  → shared / n_active
        """
        components: Dict[str, float] = dict.fromkeys(
            self.get_component_names(), 0.0
        )

        if uav.state == UAVState.DISABLED:
            components["raw_total"] = 0.0
            components["total"]     = 0.0
            return components

        active_uavs  = [u for u in uavs if u.state != UAVState.DISABLED]
        n_active     = max(len(active_uavs), 1)
        cur_coverage = coverage_map.get_coverage_rate()

        # ── 1. Coverage (shared) ─────────────────────────────────────────────
        components["coverage_delta"] = _coverage_delta_reward(
            prev_coverage, cur_coverage, self._r_coverage_delta
        ) / n_active

        # ── 2. Victim (individual) ───────────────────────────────────────────
        components["victim_found"] = _victim_found_reward(
            newly_found_by_uav, self._r_victim_base, cur_coverage  # ← Pass coverage
        )

        # ── 3. Distance shaping (individual) ────────────────────────────────
        if self._enable_shaping:
            components["distance_shaping"] = self._delta_shaping_single(
                uav, victims
            )

        # ── 4. Battery penalty (individual) ─────────────────────────────────
        components["battery_penalty"] = _battery_penalty_single(
            uav, self.cfg.reward, self.cfg.uav
        )
        components["battery_penalty"] += _battery_urgency_shaping(
            uav, stations, self._map_size
        )

        # ── 5. Battery dead (individual, one-time) ───────────────────────────
        if uav.battery_death and uav.id not in self._battery_death_penalized:
            components["battery_dead"] = self._r_battery_dead
            self._battery_death_penalized.add(uav.id)

        # ── 6. Collision (individual, one-time) ──────────────────────────────
        components["collision_obstacle"] = self._collision_reward(
            [uav], obstacles
        )

        # ── 7. Proximity (individual) ────────────────────────────────────────
        proximity_raw = _proximity_reward_single(
            uav, active_uavs,
            self._PROX_1M, self._PROX_2M, self._PROX_3M,
            self._r_proximity_1m, self._r_proximity_2m, self._r_proximity_3m,
        )
        per_uav_prox_cap = self._proximity_cap / n_active
        components["proximity"] = max(proximity_raw, per_uav_prox_cap)

        # ── 8. Danger zone (individual) ──────────────────────────────────────
        components["danger_zone"] = self._danger_reward([uav], obstacles)

        # ── 9. Landing rewards (individual, small) ───────────────────────────
        land_r, hover_p, approach_r = self._landing_rewards([uav], stations)
        components["landing_reward"]  = land_r
        components["hover_penalty"]   = hover_p
        components["approach_reward"] = approach_r

        # ── 10. Deprecated ───────────────────────────────────────────────────
        components["fleet_incentive"] = 0.0

        # ── 11. Time penalty (individual) ────────────────────────────────────
        components["time_penalty"] = self._r_time_penalty

        # ── 12. Terminal (shared) ────────────────────────────────────────────
        if done:
            components["terminal"] = self._terminal_bonus(
                cur_coverage, victims, current_step, uavs=uavs
            ) / n_active

        # ── 13. Penalty cap ──────────────────────────────────────────────────
        per_uav_cap = self._step_penalty_cap / n_active
        components  = self._apply_penalty_cap(components, cap=per_uav_cap)

        # ── 14. Total + clip ─────────────────────────────────────────────────
        components = self._finalize(components, label=f"uav_{uav.id}")

        return components

    # ═════════════════════════════════════════════════════════════════════════
    # LANDING REWARDS v4.0 - ANTI-EXPLOIT
    # ═════════════════════════════════════════════════════════════════════════

    def _landing_rewards(
        self,
        uavs:     List[UAV],
        stations: Optional[List],
    ) -> Tuple[float, float, float]:
        """
        Landing reward v4.0 - Anti-exploit design.

        CHANGES vs v3.x:
            - Tier 3 (landing): 20+bonus(50) → 5 (fixed, NO bonus)
            - Tier 1 (approach): weight 0.5 → 0.05 (tiny nudge)
            - Tier 2 (hover): -3.0 → -2.0

        RATIONALE:
            Landing là survival action, không phải goal.
            Reward nhỏ = agent land khi CẦN, không farm.
            
        MAGNITUDE CHECK:
            1 landing = +5 (vs 1 victim avg = +30)
            Ratio = 5/30 = 17% → Hợp lý, landing không dominant

        EXPLOIT PREVENTION:
            - No early bonus → Không incentivize land sớm
            - Small base (5) → Không worth farming
            - Approach tiny (0.05) → Không worth hovering
        """
        if not stations:
            return 0.0, 0.0, 0.0

        landing_total  = 0.0
        hover_total    = 0.0
        approach_total = 0.0

        max_dist = float(self._map_size) * 1.414

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

            # ── Tier 3: Landing success ────────────────────────────────────
            # v4.0: Fixed +5, NO early bonus
            # One-time per UAV per episode
            if (uav.state == UAVState.CHARGING
                    and uav.id not in self._landed_uavs):
                landing_total += self._r_landing_success  # +5.0 fixed
                self._landed_uavs.add(uav.id)

                logger.debug(
                    "UAV %d landing success: battery=%.1f%% → +%.1f",
                    uav.id, uav.battery_pct, self._r_landing_success
                )

            # ── Tier 1: Approach reward (TINY nudge) ──────────────────────
            # v4.0: approach_weight = 0.05 (giảm từ 0.5)
            # Max = 0.05 × 1.0 = +0.05 per step
            # → Không worth farming, chỉ là guidance signal
            if (uav.battery_pct <= 30.0
                    and uav.state in (UAVState.ACTIVE, UAVState.RETURNING)):
                norm_dist    = min(min_dist / max(max_dist, 1.0), 1.0)
                approach_rew = self._approach_weight * (1.0 - norm_dist)
                approach_total += approach_rew

            # ── Tier 2: Hover penalty ──────────────────────────────────────
            # Phạt nếu lơ lửng gần station + pin thấp + không land
            # v4.0: -2.0 (giảm từ -3.0)
            if (uav.state == UAVState.ACTIVE
                    and min_dist <= self._landing_range
                    and uav.battery_pct <= 30.0):
                hover_total += self._r_hover_penalty  # -2.0

        return landing_total, hover_total, approach_total

    # ═════════════════════════════════════════════════════════════════════════
    # PENALTY CAP (BUG-31 FIX)
    # ═════════════════════════════════════════════════════════════════════════

    def _apply_penalty_cap(
        self,
        components: Dict[str, float],
        cap:        Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Additive penalty cap (không distort components).
        Thêm adjustment nếu tổng penalty vượt cap.
        """
        if cap is None:
            cap = self._step_penalty_cap

        penalty_sum = sum(
            v for k, v in components.items()
            if v < 0
            and k not in ("raw_total", "total", "penalty_cap_adjustment")
        )

        if penalty_sum < cap:
            adjustment = cap - penalty_sum
            components["penalty_cap_adjustment"] = adjustment

        return components

    # ═════════════════════════════════════════════════════════════════════════
    # TERMINAL BONUS
    # ═════════════════════════════════════════════════════════════════════════

    def _terminal_bonus(
        self,
        coverage_rate: float,
        victims:       List,
        current_step:  int,
        uavs:          Optional[List] = None,
    ) -> float:
        """
        Terminal bonus v4.0.

        Proportional reward - không cần threshold để nhận:
            coverage 28% → coverage_bonus = 200 × 0.5 × 0.28 = +28
            coverage 80% → coverage_bonus = 200 × 0.5 × 0.80 = +80

        Agent với coverage thấp nhận reward thấp → Rõ ràng
        """
        n_total     = max(len(victims), 1)
        n_found     = sum(1 for v in victims if v.is_found)
        found_ratio = n_found / n_total
        time_ratio  = current_step / max(self._max_steps, 1)

        # Coverage: 50% weight
        coverage_bonus = self._terminal_cap * 0.50 * coverage_rate

        # Victims: 30% weight
        victim_bonus = self._terminal_cap * 0.30 * found_ratio

        # Time: 10% weight (hoàn thành sớm)
        time_bonus = self._terminal_cap * 0.10 * (1.0 - time_ratio)

        # Battery survival: 10% weight
        battery_bonus = 0.0
        if uavs is not None:
            alive = [u for u in uavs if u.state != UAVState.DISABLED]
            if alive:
                mean_bat      = np.mean([u.battery_pct for u in alive])
                battery_bonus = self._terminal_cap * 0.10 * (mean_bat / 100.0)

        raw = coverage_bonus + victim_bonus + time_bonus + battery_bonus
        return float(np.clip(raw, 0.0, self._terminal_cap))

    # ═════════════════════════════════════════════════════════════════════════
    # DELTA SHAPING (BUG-33 FIX)
    # ═════════════════════════════════════════════════════════════════════════

    def _delta_shaping_fleet(
        self,
        uavs:    List[UAV],
        victims: List["BaseVictim"],
    ) -> float:
        """Delta-based potential shaping (fleet total)."""
        unfound = [v for v in victims if not v.is_found]
        if not unfound:
            return 0.0

        total = 0.0
        for uav in uavs:
            total += self._delta_shaping_single(uav, victims, unfound)
        return total

    def _delta_shaping_single(
        self,
        uav:     UAV,
        victims: List["BaseVictim"],
        unfound: Optional[List["BaseVictim"]] = None,
    ) -> float:
        """
        Delta shaping per UAV.

        reward = (prev_dist - curr_dist) × weight
        Positive = approaching victim
        Negative = retreating

        Không thể farm: đứng yên → delta = 0
        """
        if uav.state not in (UAVState.ACTIVE, UAVState.DEPLOYING):
            return 0.0

        if unfound is None:
            unfound = [v for v in victims if not v.is_found]
        if not unfound:
            return 0.0

        current_min = min(dist_2d(uav.pos, v.pos) for v in unfound)
        prev_min    = self._prev_min_dist.get(uav.id, None)

        self._prev_min_dist[uav.id] = current_min

        if prev_min is None:
            return 0.0

        delta  = prev_min - current_min
        shaped = delta * self._shaping_weight

        return float(np.clip(shaped, -self._shaping_max, self._shaping_max))

    # ═════════════════════════════════════════════════════════════════════════
    # BATTERY REWARDS
    # ═════════════════════════════════════════════════════════════════════════

    def _battery_rewards(
        self,
        uavs:     List[UAV],
        stations: Optional[List] = None,
    ) -> Tuple[float, float]:
        """Battery penalty + dead penalty."""
        bat_penalty = 0.0
        bat_dead    = 0.0

        for uav in uavs:
            if uav.state != UAVState.DISABLED:
                bat_penalty += _battery_penalty_single(
                    uav, self.cfg.reward, self.cfg.uav
                )
                bat_penalty += _battery_urgency_shaping(
                    uav, stations, self._map_size
                )

            if (uav.battery_death
                    and uav.id not in self._battery_death_penalized):
                bat_dead += self._r_battery_dead
                self._battery_death_penalized.add(uav.id)

        return bat_penalty, bat_dead

    def _collision_reward(self, uavs: List[UAV], obstacles: List) -> float:
        """Collision penalty (one-time per UAV)."""
        from entities.obstacle import Debris
        penalty = 0.0
        for uav in uavs:
            if uav.state == UAVState.DISABLED:
                continue
            if uav.id in self._collision_penalized:
                continue
            for obs in obstacles:
                if isinstance(obs, Debris) and obs.causes_collision(uav.pos):
                    penalty += self._r_collision_obs
                    self._collision_penalized.add(uav.id)
                    break
        return penalty

    def _danger_reward(self, uavs: List[UAV], obstacles: List) -> float:
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
    # FINALIZE + LOGGING
    # ═════════════════════════════════════════════════════════════════════════

    def _finalize(
        self,
        components: Dict[str, float],
        label:      str = "global",
    ) -> Dict[str, float]:
        """Compute raw_total + apply clip."""
        raw_total = sum(
            v for k, v in components.items()
            if k not in ("raw_total", "total")
        )
        components["raw_total"] = raw_total
        components["total"]     = float(
            np.clip(raw_total, self._clip_min, self._clip_max)
        )
        _assert_no_nan_inf(components["total"], f"finalize.{label}")
        return components

    def _log_step_breakdown(
        self,
        components:   Dict[str, float],
        current_step: int,
    ) -> None:
        """
        Log reward breakdown mỗi 50 steps để debug.
        Giúp xác định component nào dominant.
        """
        if current_step % 50 != 0:
            return

        total = components.get("total", 0.0)
        parts = {
            k: v for k, v in components.items()
            if k not in ("raw_total", "total", "fleet_incentive",
                         "penalty_cap_adjustment")
            and abs(v) > 0.001
        }

        if not parts:
            return

        # Sort by absolute value
        sorted_parts = sorted(parts.items(), key=lambda x: abs(x[1]), reverse=True)

        breakdown = " | ".join(
            f"{k}={v:+.2f}" for k, v in sorted_parts[:6]
        )

        logger.debug(
            "[STEP %d] reward=%.2f | %s",
            current_step, total, breakdown
        )

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
            "penalty_cap_adjustment",
            "landing_reward",
            "hover_penalty",
            "approach_reward",
            "raw_total",
            "total",
        ]

    def summarize(self, reward_dict: Dict[str, float]) -> str:
        """Compact summary cho logging."""
        skip = {"raw_total", "total", "fleet_incentive",
                "penalty_cap_adjustment"}
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

    def __repr__(self) -> str:
        return (
            f"BaselineReward(v4.0, "
            f"coverage={self._r_coverage_delta}, "
            f"landing={self._r_landing_success}, "
            f"clip=[{self._clip_min:.0f}, {self._clip_max:.0f}])"
        )


# =============================================================================
# MODULE-LEVEL FUNCTIONS (stateless, unit-testable)
# =============================================================================

def _coverage_delta_reward(
    prev_coverage: float,
    cur_coverage:  float,
    weight:        float,
) -> float:
    """Coverage delta reward (chỉ positive, không penalty khi giảm)."""
    delta = max(0.0, cur_coverage - prev_coverage)
    return delta * weight


# baseline_reward.py - Tìm function _victim_found_reward()
# THAY THẾ toàn bộ:

def _victim_found_reward(
    newly_found:    List["BaseVictim"],
    r_victim_base:  float,
    coverage_rate:  float = 0.0,  # ← NEW: optional coverage context
) -> float:
    """
    Urgency-weighted victim discovery reward với dynamic scaling.
    
    '''Dynamic scaling rationale:
        Early (cov=0%):   scale=1.0 → focus exploration
        Mid   (cov=50%):  scale=1.5 → balanced
        Late  (cov=100%): scale=2.0 → victim priority
    
    
    """
    if not newly_found:
        return 0
    
    # FIX: Early coverage → higher victim reward (encourage exploration)
    scale = 2.0 - coverage_rate  # Early=2.0, Mid=1.5, Late=1.0
    
    total_urgency = sum(v.urgency / 5.0 for v in newly_found)
    return r_victim_base * scale * total_urgency


def _battery_penalty_single(
    uav:        UAV,
    reward_cfg,
    uav_cfg,
) -> float:
    """
    Progressive battery penalty per UAV.

    v4.0: Hard thresholds thực tế, KHÔNG dùng battery_emergency_pct.

    Penalty tăng dần khi pin GIẢM (đúng chiều):
        > 20%  →  0.0   không phạt
        ≤ 20%  → -0.5   warning
        ≤ 10%  → -2.0   moderate
        ≤  5%  → -8.0   severe

    NOTE: Không phạt ở 30-40% vì FleetManager đã
          force RETURNING ở battery_emergency_pct (40%).
          UAV ở trạng thái RETURNING không nhận penalty.
    """
    bat = uav.battery_pct

    if bat <= 5.0:
        return reward_cfg.r_battery_5             # -8.0
    if bat <= uav_cfg.battery_critical_pct:       # ≤ 10%
        return reward_cfg.r_battery_10            # -2.0
    if bat <= uav_cfg.battery_warning_pct:        # ≤ 20%
        return reward_cfg.r_battery_20            # -0.5
    return 0.0
    


def _battery_urgency_shaping(
    uav,
    stations,
    map_size: float,
) -> float:
    """
    Battery urgency shaping: penalty tỉ lệ distance × severity.

    Chỉ apply cho ACTIVE UAV với battery ≤ 30%.
    Không apply RETURNING/CHARGING/DISABLED.

    v4.0: Severity giảm để đảm bảo landing luôn rational hơn die.

    RATIONAL CHECK (worst realistic case: bat=18%, dist=150m):
        penalty = -0.6 × 1.5 × 2.0 = -1.8/step
        20 steps = -36.0
        vs die penalty = -50.0
        → Landing net = -36 + 5 = -31 > -50 ✅
    """
    if stations is None or not stations:
        return 0.0

    bat = uav.battery_pct

    if (bat > 30.0
            or uav.state in (
                UAVState.CHARGING,
                UAVState.RETURNING,
                UAVState.DISABLED,
            )):
        return 0.0

    if bat <= 5.0:
        severity = 6.0
    elif bat <= 10.0:
        severity = 3.0
    elif bat <= 20.0:
        severity = 1.5
    elif bat <= 30.0:
        severity = 0.5
    else:
        return 0.0

    min_dist        = min(dist_2d(uav.pos, s.pos) for s in stations)
    normalized_dist = min_dist / max(map_size, 1.0)
    penalty         = -normalized_dist * severity * 2.0

    return float(np.clip(penalty, -4.0, 0.0))


def _proximity_reward(
    active_uavs: List[UAV],
    thresh_1m:   float,
    thresh_2m:   float,
    thresh_3m:   float,
    r_1m:        float,
    r_2m:        float,
    r_3m:        float,
) -> float:
    """Pairwise proximity penalty (fleet total, uncapped)."""
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


def _proximity_reward_single(
    uav:         UAV,
    active_uavs: List[UAV],
    thresh_1m:   float,
    thresh_2m:   float,
    thresh_3m:   float,
    r_1m:        float,
    r_2m:        float,
    r_3m:        float,
) -> float:
    """Proximity penalty per UAV vs all others."""
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


def _assert_no_nan_inf(value: float, label: str) -> None:
    """Sanity check."""
    assert not np.isnan(value), f"NaN detected in {label}"
    assert not np.isinf(value), f"Inf detected in {label}"