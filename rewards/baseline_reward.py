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

# Baseline pairs để normalize proximity (4 UAVs → 6 pairs)
_BASELINE_PAIRS = 6.0


class BaselineReward:
    """
    Baseline Reward Function v3.1 - Research-grade.

    Multi-agent RL compatible (MAPPO / MASAC / MATD3).

    Key fixes vs v3.0:
        - Penalty cap: additive adjustment (không distort components)
        - Proximity: normalized theo swarm size
        - Shaping: delta-based với prev_pos memory
        - Terminal: không saturate với terminal_cap
    """

    def __init__(self, cfg: "AppConfig") -> None:
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
        self._battery_death_penalized.clear()
        self._collision_penalized.clear()
        self._prev_min_dist.clear()

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

    # ═════════════════════════════════════════════════════════════════════════
    # PRIVATE: PENALTY CAP (FIX BUG-31)
    # ═════════════════════════════════════════════════════════════════════════

    def _apply_penalty_cap(
        self,
        components: Dict[str, float],
        cap:        Optional[float] = None,
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

    def _terminal_bonus(
        self,
        coverage_rate: float,
        victims:       List["BaseVictim"],
        current_step:  int,
        uavs:          Optional[List[UAV]] = None,  # ← Thêm param
    ) -> float:
        """Terminal bonus với battery survival reward."""
        n_total     = max(len(victims), 1)
        n_found     = sum(1 for v in victims if v.is_found)
        found_ratio = n_found / n_total
        time_ratio  = current_step / max(self._max_steps, 1)

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

    def _delta_shaping_fleet(
        self,
        uavs:    List[UAV],
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

    def _delta_shaping_single(
        self,
        uav:     UAV,
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

    def _battery_rewards(
        self,
        uavs:     List[UAV],
        stations: Optional[List] = None,
    ) -> Tuple[float, float]:
        """Battery penalty + dead (one-time) + urgency shaping."""
        bat_penalty = 0.0
        bat_dead    = 0.0

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
    # PRIVATE: FINALIZE
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
        components["total"] = float(
            np.clip(raw_total, self._clip_min, self._clip_max)
        )
        _assert_no_nan_inf(components["total"], f"finalize.{label}")
        return components

    # ═════════════════════════════════════════════════════════════════════════
    # PRIVATE: LOGGING
    # ═════════════════════════════════════════════════════════════════════════

    def _log_extreme_reward(
        self,
        components:   Dict[str, float],
        current_step: int,
        coverage_map: "CoverageMap",
        uavs:         List[UAV],
        obstacles:    List,
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
            "penalty_cap_adjustment",  # ✅ NEW: BUG-31 fix
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

    def __repr__(self) -> str:
        return (
            f"BaselineReward(v3.1, "
            f"clip=[{self._clip_min:.0f}, {self._clip_max:.0f}], "
            f"penalty_cap={self._step_penalty_cap:.0f}, "
            f"shaping_weight={self._shaping_weight})"
        )


# =============================================================================
# MODULE-LEVEL FUNCTIONS (stateless, unit-testable)
# =============================================================================

def _coverage_delta_reward(
    prev_coverage: float,
    cur_coverage:  float,
    weight:        float,
) -> float:
    """Coverage delta reward (clamped to non-negative)."""
    delta = max(0.0, cur_coverage - prev_coverage)
    return delta * weight


def _victim_found_reward(
    newly_found:   List["BaseVictim"],
    r_victim_base: float,
) -> float:
    """Urgency-weighted victim discovery reward."""
    if not newly_found:
        return 0.0
    return sum(r_victim_base * (v.urgency / 5.0) for v in newly_found)


def _battery_penalty_single(
    uav:        UAV,
    reward_cfg: "RewardConfig",
    uav_cfg:    "UAVConfig",
) -> float:
    """
    Progressive battery penalty cho single UAV.

    Thresholds từ UAVConfig (không hardcode):
        20% → r_battery_20
        10% → r_battery_10
        5%  → r_battery_5
    """
    bat = uav.battery_pct
    if bat <= uav_cfg.battery_emergency_pct:
        return reward_cfg.r_battery_5
    if bat <= uav_cfg.battery_critical_pct:
        return reward_cfg.r_battery_10
    if bat <= uav_cfg.battery_warning_pct:
        return reward_cfg.r_battery_20
    return 0.0


# TRONG baseline_reward.py
# Thay hàm _battery_urgency_shaping():

def _battery_urgency_shaping(
    uav:      UAV,
    stations: Optional[List],
    map_size: float,
) -> float:
    """
    Battery urgency shaping v2.
    
    Khi battery thấp:
        - Penalize nặng nếu đang xa station VÀ không về
        - Reward nếu đang tiến về station (delta-based)
    """
    if stations is None or not stations:
        return 0.0

    bat = uav.battery_pct

    # Không áp dụng khi đủ pin hoặc đang sạc
    if bat > 20.0 or uav.state in (UAVState.CHARGING, UAVState.DISABLED):
        return 0.0

    # Severity tăng dần theo mức độ khẩn cấp
    if bat <= 5.0:
        severity = 1.0    # ← Tăng từ 0.3
    elif bat <= 10.0:
        severity = 0.5    # ← Tăng từ 0.15
    elif bat <= 15.0:
        severity = 0.2    # ← Tăng từ 0.05
    else:
        severity = 0.05   # 15-20% zone

    # Distance đến station gần nhất
    min_dist = min(dist_2d(uav.pos, s.pos) for s in stations)
    normalized_dist = min_dist / max(map_size, 1.0)

    # ✅ Penalty tỉ lệ thuận với khoảng cách × severity
    # Gần station → ít penalty, xa station → nhiều penalty
    penalty = -normalized_dist * severity * 3.0  # ← Scale up

    return float(np.clip(penalty, -2.0, 0.0))  # ← Cap để không dominate


def _proximity_reward(
    active_uavs: List[UAV],
    thresh_1m:   float,
    thresh_2m:   float,
    thresh_3m:   float,
    r_1m:        float,
    r_2m:        float,
    r_3m:        float,
) -> float:
    """
    Pairwise proximity penalty (fleet total, uncapped).
    
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


def _assert_no_nan_inf(value: float, label: str) -> None:
    """Sanity check: no NaN/Inf in reward."""
    assert not np.isnan(value), f"NaN detected in {label}"
    assert not np.isinf(value), f"Inf detected in {label}"