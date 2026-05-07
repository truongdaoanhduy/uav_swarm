from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from entities.uav import UAV, UAVState

if TYPE_CHECKING:
    from config import AppConfig
    from entities.charging_station import ChargingStation

# ✅ FIX: Tắt logger ra console, chỉ lưu file nếu cần
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # ← Không print ra console


class FleetManager:
    def __init__(self, cfg: "AppConfig") -> None:
        self.cfg = cfg
        self.n_total:   int       = 0
        self.n_reserve: int       = 0
        self.all_uavs:  List[UAV] = []
        self.stations:  List["ChargingStation"] = []
        self._enforced_disables:  int = 0
        self._enforced_returns:   int = 0
        self._suggested_deploys:  int = 0
        self._uav_return_locks: Dict[int, bool] = {}
        
        # ✅ FIX: Tracking cho episode summary (thay vì spam mỗi step)
        self._episode_forced_returns: int = 0
        self._episode_disables: int = 0
        self._cached_n_active: int = 0

    def reset(self, all_uavs, stations) -> None:
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
        
        # ✅ Reset episode counters
        self._episode_forced_returns = 0
        self._episode_disables = 0

    def get_deployable_uavs(self) -> List[UAV]:
        ready_pool = [
            u for u in self.all_uavs
            if (u.state == UAVState.CHARGING and
                u.battery >= self.cfg.uav.battery_ready_threshold)
        ]
        if not ready_pool:
            return []
        ready_pool = sorted(ready_pool, key=lambda u: u.battery, reverse=True)
        deployable_count = max(0, len(ready_pool) - self.n_reserve)
        if deployable_count == 0:
            return []
        return ready_pool[:deployable_count]

    def get_best_deployable(
        self,
        prefer_station=None,
        require_min_battery: float = 80.0,
    ):
        deployable = self.get_deployable_uavs()
        if not deployable:
            return None
        deployable = [u for u in deployable if u.battery >= require_min_battery]
        if not deployable:
            return None
        if prefer_station is not None and prefer_station.is_full:
            return None
        if prefer_station is None:
            return max(deployable, key=lambda u: u.battery)

        def score(uav):
            battery_score = uav.battery / 100.0
            dist = np.linalg.norm(
                np.array(uav.pos[:2]) - np.array(prefer_station.pos[:2])
            )
            dist_score = 1.0 / (1.0 + dist / 10.0)
            return 0.7 * battery_score + 0.3 * dist_score

        return max(deployable, key=score)

    # core/fleet_manager.py

    def enforce_safety_constraints(self) -> Dict[str, int]:
        """
        Enforce safety constraints.
        
        ✅ FIX PERF: Tính n_active 1 lần trước loop (O(n) thay vì O(n²))
        ✅ Update counter incremental thay vì recount mỗi iteration
        """
        n_disabled      = 0
        n_forced_return = 0
        n_auto_deploy   = 0

        # ✅ PERF FIX: Tính 1 lần
        n_active = sum(
            1 for u in self.all_uavs
            if u.state == UAVState.ACTIVE
        )
        
        self._cached_n_active = n_active

        for uav in self.all_uavs:

            # ── [1] Battery dead → DISABLED ──────────────────────────────────
            if uav.battery <= self.cfg.uav.battery_dead_threshold:
                if uav.state != UAVState.DISABLED:
                    was_active = (uav.state == UAVState.ACTIVE)

                    uav.state        = UAVState.DISABLED
                    uav.vel[:]       = 0.0
                    uav.battery_death = True

                    n_disabled              += 1
                    self._enforced_disables += 1
                    self._episode_disables  += 1

                    # ✅ Incremental update
                    if was_active:
                        n_active -= 1

                    logger.debug("UAV %d disabled (battery=0)", uav.id)
                continue

            # ── [2] Emergency return ──────────────────────────────────────────
            if uav.state == UAVState.ACTIVE:
                emergency_threshold = self.cfg.uav.battery_penalty_emergency
                resume_threshold    = self.cfg.uav.battery_return_threshold
                is_locked           = self._uav_return_locks.get(uav.id, False)

                if not is_locked and uav.battery < emergency_threshold:
                    target = uav.find_nearest_station(self.stations)
                    if target is not None:
                        uav.target_station = target
                        uav.set_state(UAVState.RETURNING)
                        self._uav_return_locks[uav.id] = True

                        n_forced_return              += 1
                        self._enforced_returns        += 1
                        self._episode_forced_returns  += 1

                        # ✅ Incremental update
                        n_active -= 1

                        logger.debug(
                            "UAV %d forced return (battery=%.1f%%)",
                            uav.id, uav.battery,
                        )

                if is_locked and uav.battery > resume_threshold:
                    self._uav_return_locks[uav.id] = False

            # ── [3] Auto-deploy khi CHARGING đủ pin ──────────────────────────
            if uav.state == UAVState.CHARGING:
                if uav.battery >= self.cfg.uav.battery_ready_threshold:
                    # ✅ PERF FIX: Dùng n_active đã tính, không recount
                    if n_active < (self.n_total - 1):
                        if uav.target_station is not None:
                            uav.target_station.release(uav)

                        uav.set_state(UAVState.ACTIVE)
                        uav.target_station = None
                        n_auto_deploy += 1

                        # ✅ Incremental update
                        n_active += 1

                        logger.debug(
                            "UAV %d auto-deployed (battery=%.1f%%, n_active=%d)",
                            uav.id, uav.battery, n_active,
                        )

        return {
            "enforced_disables": n_disabled,
            "enforced_returns":  n_forced_return,
            "auto_deploys":      n_auto_deploy,
            "total_enforced":    n_disabled + n_forced_return,
        }

    def suggest_deployments(self, target_active=None):
        if target_active is None:
            target_active = max(2, int(self.n_total * 0.3))
        n_active = self._cached_n_active 
        
        need_deploy = max(0, target_active - n_active)
        if need_deploy == 0:
            return []
        deployable = self.get_deployable_uavs()
        if not deployable:
            return []
        total_station_capacity = sum(
            station.capacity - len(station.current_occupants)
            for station in self.stations
        )
        max_can_deploy = min(need_deploy, total_station_capacity, len(deployable))
        suggestions = deployable[:max_can_deploy]
        self._suggested_deploys += len(suggestions)
        return suggestions

    def suggest_returns(self):
        suggestions = []
        for uav in self.all_uavs:
            if uav.state != UAVState.ACTIVE:
                continue
            if self._uav_return_locks.get(uav.id, False):
                continue
            if uav.battery < self.cfg.uav.battery_return_threshold:
                target = uav.find_nearest_station(self.stations)
                if target is not None and not target.is_full:
                    suggestions.append(uav)
        return suggestions

    def get_episode_summary(self) -> Dict[str, int]:
        """
        ✅ NEW: Trả về summary cho episode log
        Thay vì spam mỗi step, trainer gọi hàm này 1 lần khi episode done
        """
        return {
            "forced_returns": self._episode_forced_returns,
            "disables":       self._episode_disables,
        }

    def get_mission_priority_hints(self):
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

    def get_spatial_awareness(self):
        active_uavs   = [u for u in self.all_uavs if u.state == UAVState.ACTIVE]
        charging_uavs = [u for u in self.all_uavs if u.state == UAVState.CHARGING]
        active_pos   = np.array([u.pos for u in active_uavs])   if active_uavs   else np.zeros((0, 3))
        charging_pos = np.array([u.pos for u in charging_uavs]) if charging_uavs else np.zeros((0, 3))
        if len(active_pos) > 0:
            center = np.mean(active_pos, axis=0)
            spread = np.mean(np.linalg.norm(active_pos - center, axis=1))
        else:
            center = np.zeros(3)
            spread = 0.0
        return {
            "active_positions":   active_pos,
            "charging_positions": charging_pos,
            "center_of_mass":     center,
            "spread_radius":      float(spread),
        }

    def step(self):
        safety            = self.enforce_safety_constraints()
        deploy_suggestions = self.suggest_deployments()
        return_suggestions = self.suggest_returns()
        priority_hints    = self.get_mission_priority_hints()
        spatial           = self.get_spatial_awareness()
        return {
            "enforced": safety,
            "suggestions": {
                "deploy": [u.id for u in deploy_suggestions],
                "return": [u.id for u in return_suggestions],
            },
            "priority_hints": priority_hints,
            "spatial": {
                "center_of_mass":     spatial["center_of_mass"].tolist(),
                "spread_radius":      spatial["spread_radius"],
                "n_active_positions": len(spatial["active_positions"]),
            },
        }

    def count_by_state(self):
        counts = {s.value: 0 for s in UAVState}
        for u in self.all_uavs:
            counts[u.state.value] += 1
        return counts

    def get_battery_stats(self):
        if not self.all_uavs:
            return {}
        batteries = [u.battery for u in self.all_uavs if u.state != UAVState.DISABLED]
        if not batteries:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0,
                    "critical_count": 0, "low_count": 0, "emergency_count": 0}
        return {
            "mean": float(np.mean(batteries)),
            "min":  float(np.min(batteries)),
            "max":  float(np.max(batteries)),
            "std":  float(np.std(batteries)),
            "critical_count":  int(sum(1 for b in batteries if b <= self.cfg.uav.battery_return_threshold)),
            "low_count":       int(sum(1 for b in batteries if b <= self.cfg.uav.battery_penalty_low)),
            "emergency_count": int(sum(1 for b in batteries if b <= self.cfg.uav.battery_penalty_emergency)),
        }

    def get_stats(self):
        state_counts = self.count_by_state()
        battery      = self.get_battery_stats()
        priority     = self.get_mission_priority_hints()
        spatial      = self.get_spatial_awareness()
        n_deployable = len(self.get_deployable_uavs())
        return {
            "n_total":     int(self.n_total),
            "n_reserve":   int(self.n_reserve),
            "n_active":    int(state_counts.get(UAVState.ACTIVE.value,    0)),
            "n_returning": int(state_counts.get(UAVState.RETURNING.value, 0)),
            "n_charging":  int(state_counts.get(UAVState.CHARGING.value,  0)),
            "n_deploying": int(state_counts.get(UAVState.DEPLOYING.value, 0)),
            "n_disabled":  int(state_counts.get(UAVState.DISABLED.value,  0)),
            "n_deployable": int(n_deployable),
            "battery":     battery,
            "priority":    priority,
            "spatial":     {"spread_radius": spatial["spread_radius"],
                            "n_active_positions": len(spatial["active_positions"])},
            "enforced_disables": int(self._enforced_disables),
            "enforced_returns":  int(self._enforced_returns),
            "suggested_deploys": int(self._suggested_deploys),
        }

    def get_fleet_incentives(self):
        return {"deploy": 0.0, "recall": 0.0, "total": 0.0}

    def is_episode_over(self):
        return all(u.state == UAVState.DISABLED for u in self.all_uavs)

    def __repr__(self):
        counts   = self.count_by_state()
        priority = self.get_mission_priority_hints()
        return (
            f"FleetManager("
            f"total={self.n_total}, "
            f"active={counts.get(UAVState.ACTIVE.value, 0)}, "
            f"reserve={self.n_reserve}, "
            f"op_ratio={priority['operational_ratio']:.2f})"
        )