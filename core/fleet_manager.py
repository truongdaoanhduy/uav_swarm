from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from entities.uav import UAV, UAVState

"""
core/fleet_manager.py
Fleet Manager - RL-COMPATIBLE REDESIGN v2.0

PHILOSOPHY SHIFT:
    ❌ OLD: FleetManager = rule engine (auto-deploy, auto-return, incentives)
    ✅ NEW: FleetManager = constraint enforcer + state tracker
    
    → RL agent CONTROLS swarm behavior
    → FleetManager chỉ:
        - Enforce safety (battery dead, collision)
        - Track state
        - Provide info cho observation

FIXES APPLIED:
    ✅ FIX-01: Xóa sys.path hack
    ✅ FIX-02: get_best_deployable() không mutate list
    ✅ FIX-03: Deploy logic check battery_ready_threshold
    ✅ FIX-04: Batch deploy check station capacity
    ✅ FIX-05: Auto-return chỉ khi CRITICAL (hysteresis)
    ✅ FIX-06: XÓA incentive system (RL agent tự quyết định)
    ✅ FIX-07: Deploy xét station congestion
    ✅ FIX-08: Tách "suggest" vs "enforce" operations
    ✅ FIX-09: Add mission priority hints (cho RL observation)
    ✅ FIX-10: Add spatial awareness metrics

DESIGN PRINCIPLES:
    1. Constraint-only enforcement (safety critical)
    2. Rich state info cho RL observation
    3. Suggestions, NOT commands
    4. No hidden rules
"""

if TYPE_CHECKING:
    from config import AppConfig
    from entities.charging_station import ChargingStation

logger = logging.getLogger(__name__)


class FleetManager:
    """
    Fleet Manager - RL-compatible constraint enforcer.

    ROLES:
        [ENFORCE] Safety constraints (battery dead → disable)
        [TRACK]   State tracking cho observation
        [SUGGEST] Deployment suggestions (RL can ignore)
        [INFO]    Rich metrics cho decision making

    REMOVED (cho RL control):
        ❌ Auto-deploy (RL quyết định khi nào deploy)
        ❌ Auto-return non-critical (RL quyết định return strategy)
        ❌ Incentive hacks (RL học từ reward thật)

    Args:
        cfg: AppConfig object
    """

    def __init__(self, cfg: "AppConfig") -> None:
        self.cfg = cfg

        self.n_total:   int       = 0
        self.n_reserve: int       = 0
        self.all_uavs:  List[UAV] = []
        self.stations:  List["ChargingStation"] = []

        # Tracking (statistics only)
        self._enforced_disables:  int = 0  # Battery dead
        self._enforced_returns:   int = 0  # Critical safety returns
        self._suggested_deploys:  int = 0  # Suggestions given (not enforced)

        # Hysteresis state (prevent oscillation)
        self._uav_return_locks: Dict[int, bool] = {}  # uav_id → locked state

    # ─── Reset ────────────────────────────────────────────────────────────────

    def reset(
        self,
        all_uavs: List[UAV],
        stations: List["ChargingStation"],
    ) -> None:
        """Reset fleet manager cho episode mới."""
        self.all_uavs = all_uavs
        self.stations = stations
        self.n_total  = len(all_uavs)

        self.n_reserve = max(
            self.cfg.uav.min_reserve,
            int(np.ceil(self.n_total * self.cfg.uav.reserve_ratio)),
        )

        self._enforced_disables  = 0
        self._enforced_returns   = 0
        self._suggested_deploys  = 0
        self._uav_return_locks   = {}

        logger.debug(
            "FleetManager reset: n_total=%d, n_reserve=%d (constraint-only mode)",
            self.n_total, self.n_reserve,
        )

    # ─── Deployable UAVs (FIXED) ─────────────────────────────────────────────

    def get_deployable_uavs(self) -> List[UAV]:
        """
        ✅ FIX-P4: Deploy UAV pin cao nhất, reserve giữ phần còn lại.

        BEFORE (wrong logic):
            ready_pool sorted high→low
            return ready_pool[n_reserve:]  ← deploy UAV pin THẤP hơn, giữ pin CAO

        AFTER (correct):
            ready_pool sorted high→low
            deployable_count = len(ready_pool) - n_reserve
            return ready_pool[:deployable_count]  ← deploy UAV pin CAO nhất
        """
        ready_pool = [
            u for u in self.all_uavs
            if (u.state == UAVState.CHARGING and
                u.battery >= self.cfg.uav.battery_ready_threshold)
        ]

        if not ready_pool:
            return []

        # Sort high → low battery (không mutate, tạo list mới)
        ready_pool = sorted(ready_pool, key=lambda u: u.battery, reverse=True)

        # ✅ FIX-P4: Deploy từ top (pin cao), giữ phần cuối làm reserve
        deployable_count = max(0, len(ready_pool) - self.n_reserve)

        if deployable_count == 0:
            return []

        return ready_pool[:deployable_count]

    def get_best_deployable(
        self,
        prefer_station: Optional["ChargingStation"] = None,
        require_min_battery: float = 80.0,  # ✅ FIX-03
    ) -> Optional[UAV]:
        """
        ✅ FIX-02: Không mutate list (dùng max)
        ✅ FIX-03: Check min battery requirement
        ✅ FIX-07: Check station capacity

        Args:
            prefer_station:     Ưu tiên UAV gần station này
            require_min_battery: Minimum battery % (default 80%)

        Returns:
            UAV tốt nhất hoặc None
        """
        deployable = self.get_deployable_uavs()
        if not deployable:
            return None

        # Filter min battery
        deployable = [u for u in deployable if u.battery >= require_min_battery]
        if not deployable:
            return None

        # ✅ FIX-07: Check station capacity (nếu prefer_station specified)
        if prefer_station is not None and prefer_station.is_full:
            logger.debug(f"Station {prefer_station.id} full, cannot deploy from it")
            return None

        if prefer_station is None:
            # ✅ FIX-02: Dùng max (không mutate)
            return max(deployable, key=lambda u: u.battery)

        # Score theo battery + distance to station
        def score(uav: UAV) -> float:
            battery_score = uav.battery / 100.0  # [0,1]

            dist = np.linalg.norm(
                np.array(uav.pos[:2]) - np.array(prefer_station.pos[:2])
            )
            dist_score = 1.0 / (1.0 + dist / 10.0)  # [0,1], closer better

            return 0.7 * battery_score + 0.3 * dist_score

        # ✅ FIX-02: Dùng max
        return max(deployable, key=score)

    # ─── Safety Enforcement (CHỈ critical) ───────────────────────────────────

    def enforce_safety_constraints(self) -> Dict[str, int]:
        """
        ✅ FIX-05: CHỈ enforce critical safety (hysteresis)
        ✅ FIX-08: Tách enforce vs suggest

        ENFORCE (bắt buộc):
            - Battery dead → DISABLED
            - Battery emergency (<5%) + far from station → force RETURNING

        NOT enforce (RL quyết định):
            - Battery low (<20%) → RL agent tự quyết return
            - Deploy timing → RL agent control

        Returns:
            Dict với số lượng operations enforced
        """
        n_disabled = 0
        n_forced_return = 0

        for uav in self.all_uavs:
            # [1] ENFORCE: Battery dead → disable
            if uav.battery <= self.cfg.uav.battery_dead_threshold:
                if uav.state != UAVState.DISABLED:
                    uav.mark_disabled()
                    n_disabled += 1
                    self._enforced_disables += 1
                    logger.warning(f"UAV {uav.id} DISABLED (battery dead)")
                continue

            # [2] ENFORCE: Emergency return (hysteresis)
            # ✅ FIX-05: CHỈ khi <5% + chưa lock + ACTIVE
            if uav.state == UAVState.ACTIVE:
                emergency_threshold = self.cfg.uav.battery_penalty_emergency  # 5%
                resume_threshold    = self.cfg.uav.battery_return_threshold   # 10%

                # Check lock state (hysteresis)
                is_locked = self._uav_return_locks.get(uav.id, False)

                if not is_locked and uav.battery < emergency_threshold:
                    # LOCK: force return
                    target = uav.find_nearest_station(self.stations)
                    if target is not None:
                        uav.target_station = target
                        uav.set_state(UAVState.RETURNING)
                        self._uav_return_locks[uav.id] = True
                        n_forced_return += 1
                        self._enforced_returns += 1
                        logger.warning(
                            f"UAV {uav.id} FORCED RETURN (emergency battery={uav.battery:.1f}%)"
                        )

                # UNLOCK: battery recovered
                if is_locked and uav.battery > resume_threshold:
                    self._uav_return_locks[uav.id] = False

        return {
            "enforced_disables":      n_disabled,
            "enforced_returns":       n_forced_return,
            "total_enforced":         n_disabled + n_forced_return,
        }

    # ─── Suggestions (RL can ignore) ─────────────────────────────────────────

    def suggest_deployments(
        self,
        target_active: Optional[int] = None,
    ) -> List[UAV]:
        """
        ✅ FIX-04: Check station capacity
        ✅ FIX-08: Chỉ suggest, KHÔNG enforce

        Suggest UAVs nên deploy (RL agent quyết định cuối cùng).

        Args:
            target_active: Số UAV ACTIVE mong muốn (default: 30% swarm)

        Returns:
            List UAVs suggest deploy (sorted by priority)
        """
        if target_active is None:
            target_active = max(2, int(self.n_total * 0.3))

        n_active = sum(1 for u in self.all_uavs if u.state == UAVState.ACTIVE)
        need_deploy = max(0, target_active - n_active)

        if need_deploy == 0:
            return []

        deployable = self.get_deployable_uavs()
        if not deployable:
            return []

        # ✅ FIX-04: Check total station capacity
        total_station_capacity = sum(
            station.capacity - len(station.current_occupants)
            for station in self.stations
        )

        # Limit suggestions by available capacity
        max_can_deploy = min(need_deploy, total_station_capacity, len(deployable))

        suggestions = deployable[:max_can_deploy]
        self._suggested_deploys += len(suggestions)

        return suggestions

    def suggest_returns(self) -> List[UAV]:
        """
        Suggest UAVs nên return (non-critical, RL quyết định).

        Criteria:
            - Battery < return_threshold (10%)
            - Station có chỗ
            - Không locked (hysteresis)

        Returns:
            List UAVs suggest return
        """
        suggestions = []

        for uav in self.all_uavs:
            if uav.state != UAVState.ACTIVE:
                continue

            # Skip nếu đang locked (emergency return)
            if self._uav_return_locks.get(uav.id, False):
                continue

            # Suggest nếu battery thấp
            if uav.battery < self.cfg.uav.battery_return_threshold:
                target = uav.find_nearest_station(self.stations)
                if target is not None and not target.is_full:
                    suggestions.append(uav)

        return suggestions

    # ─── Rich State Info (cho RL observation) ────────────────────────────────

    def get_mission_priority_hints(self) -> Dict[str, float]:
        """
        ✅ FIX-09: Mission priority hints cho RL observation

        Hints (normalized [0,1]):
            - operational_ratio: % swarm đang ACTIVE
            - reserve_health:    % reserve đủ battery
            - station_pressure:  % stations bị congestion

        Usage:
            Thêm vào critic observation (global state)
        """
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
            "reserve_health":    float(reserve_health),
            "station_pressure":  float(station_pressure),
        }

    def get_spatial_awareness(self) -> Dict[str, np.ndarray]:
        """
        ✅ FIX-10: Spatial awareness cho RL coordination

        Returns:
            Dict với spatial metrics:
                - active_positions:   [N, 3] positions của ACTIVE UAVs
                - charging_positions: [M, 3] positions của CHARGING UAVs
                - center_of_mass:     [3,] center của swarm
                - spread_radius:      float dispersal metric
        """
        active_uavs = [u for u in self.all_uavs if u.state == UAVState.ACTIVE]
        charging_uavs = [u for u in self.all_uavs if u.state == UAVState.CHARGING]

        active_pos = np.array([u.pos for u in active_uavs]) if active_uavs else np.zeros((0, 3))
        charging_pos = np.array([u.pos for u in charging_uavs]) if charging_uavs else np.zeros((0, 3))

        # Center of mass
        all_active_pos = active_pos
        if len(all_active_pos) > 0:
            center = np.mean(all_active_pos, axis=0)
            spread = np.mean(np.linalg.norm(all_active_pos - center, axis=1))
        else:
            center = np.zeros(3)
            spread = 0.0

        return {
            "active_positions":   active_pos,
            "charging_positions": charging_pos,
            "center_of_mass":     center,
            "spread_radius":      float(spread),
        }

    # ─── Step ─────────────────────────────────────────────────────────────────

    def step(self) -> Dict[str, any]:
        """
        ✅ FIX-08: CHỈ enforce safety, RL control behavior

        Fleet manager step (constraint-only mode).

        Returns:
            Dict với:
                - enforced operations (disable, force return)
                - suggestions (deploy, return) — RL can ignore
                - state info (mission priority, spatial)
        """
        # [1] ENFORCE safety constraints
        safety = self.enforce_safety_constraints()

        # [2] SUGGEST operations (RL decides)
        deploy_suggestions = self.suggest_deployments()
        return_suggestions = self.suggest_returns()

        # [3] Compute state info
        priority_hints = self.get_mission_priority_hints()
        spatial = self.get_spatial_awareness()

        return {
            # Enforced (bắt buộc)
            "enforced": safety,

            # Suggestions (optional, cho RL)
            "suggestions": {
                "deploy": [u.id for u in deploy_suggestions],
                "return": [u.id for u in return_suggestions],
            },

            # State info (cho observation)
            "priority_hints": priority_hints,
            "spatial": {
                "center_of_mass": spatial["center_of_mass"].tolist(),
                "spread_radius":  spatial["spread_radius"],
                "n_active_positions": len(spatial["active_positions"]),
            },
        }

    # ─── Statistics ────────────────────────────────────────────────────────────

    def count_by_state(self) -> Dict[str, int]:
        counts = {s.value: 0 for s in UAVState}
        for u in self.all_uavs:
            counts[u.state.value] += 1
        return counts

    def get_battery_stats(self) -> Dict[str, float]:
        if not self.all_uavs:
            return {}

        batteries = [
            u.battery for u in self.all_uavs
            if u.state != UAVState.DISABLED
        ]

        if not batteries:
            return {
                "mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0,
                "critical_count": 0, "low_count": 0, "emergency_count": 0,
            }

        return {
            "mean":           float(np.mean(batteries)),
            "min":            float(np.min(batteries)),
            "max":            float(np.max(batteries)),
            "std":            float(np.std(batteries)),
            "critical_count": int(sum(
                1 for b in batteries
                if b <= self.cfg.uav.battery_return_threshold  # 10%
            )),
            "low_count": int(sum(
                1 for b in batteries
                if b <= self.cfg.uav.battery_penalty_low  # 20%
            )),
            "emergency_count": int(sum(
                1 for b in batteries
                if b <= self.cfg.uav.battery_penalty_emergency  # 5%
            )),
        }

    def get_stats(self) -> Dict:
        """Full stats cho logging/debugging."""
        state_counts = self.count_by_state()
        battery      = self.get_battery_stats()
        priority     = self.get_mission_priority_hints()
        spatial      = self.get_spatial_awareness()

        n_deployable = len(self.get_deployable_uavs())

        return {
            # Fleet composition
            "n_total":     int(self.n_total),
            "n_reserve":   int(self.n_reserve),
            "n_active":    int(state_counts.get(UAVState.ACTIVE.value,    0)),
            "n_returning": int(state_counts.get(UAVState.RETURNING.value, 0)),
            "n_charging":  int(state_counts.get(UAVState.CHARGING.value,  0)),
            "n_deploying": int(state_counts.get(UAVState.DEPLOYING.value, 0)),
            "n_disabled":  int(state_counts.get(UAVState.DISABLED.value,  0)),
            "n_deployable": int(n_deployable),

            # Battery health
            "battery": battery,

            # Mission priority
            "priority": priority,

            # Spatial
            "spatial": {
                "spread_radius": spatial["spread_radius"],
                "n_active_positions": len(spatial["active_positions"]),
            },

            # Operation counts
            "enforced_disables":  int(self._enforced_disables),
            "enforced_returns":   int(self._enforced_returns),
            "suggested_deploys":  int(self._suggested_deploys),
        }

    # ─── Backward Compatibility ──────────────────────────────────────────────

    def get_fleet_incentives(self) -> Dict[str, float]:
        """
        ✅ BACKWARD COMPATIBILITY: Return zero incentives.
        
        NOTE: Incentive system đã bị XÓA trong v2.0 redesign.
            RL agent tự học deploy/return strategy.
            Method này chỉ để backward compat, luôn return 0.
        
        Returns:
            Dict với all incentives = 0.0
        """
        return {
            "deploy": 0.0,
            "recall": 0.0,
            "total":  0.0,
        }


    def is_episode_over(self) -> bool:
        """Episode kết thúc khi tất cả UAVs disabled."""
        return all(u.state == UAVState.DISABLED for u in self.all_uavs)

    def __repr__(self) -> str:
        counts = self.count_by_state()
        priority = self.get_mission_priority_hints()
        return (
            f"FleetManager("
            f"total={self.n_total}, "
            f"active={counts.get(UAVState.ACTIVE.value, 0)}, "
            f"reserve={self.n_reserve}, "
            f"op_ratio={priority['operational_ratio']:.2f})"
        )