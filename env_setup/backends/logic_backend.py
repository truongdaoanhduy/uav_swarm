"""
Logic backend for SAR UAV environment.

Pure Python physics simulation (CPU only).
~1000 steps/second on modern hardware.

Design:
    - Deterministic given same seed
    - No hidden heuristics (RL controls behavior)
    - Clean separation: physics in step_physics, world in step_world
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

# ✅ FIX 4.1: REMOVED sys.path hack
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import AppConfig
from core.coverage_map import CoverageMap
from core.fleet_manager import FleetManager
from entities.charging_station import ChargingStation
from entities.obstacle import Debris, DangerZone
from entities.uav import UAV, UAVState
from entities.victim import InjuredVictim, MobileVictim
from sensors.fov_sensor import FOVSensor

from .base_backend import BaseBackend

logger = logging.getLogger(__name__)


class LogicBackend(BaseBackend):
    """
    Pure Python physics backend (~1000 steps/s).

    UAV spawn:
        ✅ FIX 4.2: Dùng map_data["uav_spawns"] trực tiếp (từ MapGenerator.generate())
                    KHÔNG gọi lại get_uav_spawns() → tránh re-generate với RNG khác
    
    Deterministic eval:
        ✅ FIX 4.3: Reset np.random seed khi cfg.env.deterministic_eval = True
    """

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg

        # Entities (populated in reset)
        self.uavs:      list[UAV]                       = []
        self.victims:   list[InjuredVictim | MobileVictim] = []
        self.stations:  list[ChargingStation]           = []
        self.obstacles: list[Debris | DangerZone]       = []

        # Sub-systems
        self._cov_map   = CoverageMap(cfg)
        self._fleet_mgr = FleetManager(cfg)
        self._fov_sensor = FOVSensor(cfg)

        # ✅ FIX 4.2: REMOVED self._map_gen (MapGenerator)
        # MapGenerator chỉ dùng để generate map (base_env làm điều đó)
        # Backend chỉ cần nhận map_data đã generate sẵn
        # Spawn positions đến từ map_data["uav_spawns"]

        self._step_count = 0

    # ══════════════════════════════════════════════════════════════════════
    # RESET
    # ══════════════════════════════════════════════════════════════════════

    def reset(self, map_data: dict[str, Any]) -> None:
        # ✅ FIX 4.3: Deterministic eval mode
        if self.cfg.env.deterministic_eval:
            np.random.seed(self.cfg.env.eval_seed)
            logger.debug(
                "LogicBackend: deterministic_eval=True, seed=%d",
                self.cfg.env.eval_seed,
            )

        # ✅ NEW: Set FOVSensor RNG seed cho reproducible detection
        if self.cfg.env.deterministic_eval:
            self._fov_sensor.set_seed(self.cfg.env.eval_seed)
        else:
            # Random seed mỗi episode → stochastic training
            self._fov_sensor.set_seed(
                int(np.random.randint(0, 2**31))
            )

        # Build entities từ map_data (giữ nguyên)
        self.stations  = self._build_stations(map_data)
        self.obstacles = self._build_obstacles(map_data)
        self.victims   = self._build_victims(map_data)
        self.uavs      = self._build_uavs(map_data)

        self._cov_map.reset()
        self._fleet_mgr.reset(self.uavs, self.stations)
        self._step_count = 0

        logger.debug(
            "LogicBackend reset: %d UAVs, %d victims, %d stations, %d obstacles",
            len(self.uavs), len(self.victims),
            len(self.stations), len(self.obstacles),
        )

    # ══════════════════════════════════════════════════════════════════════
    # APPLY ACTIONS
    # ══════════════════════════════════════════════════════════════════════

    def apply_actions(self, actions: dict[int, np.ndarray]) -> None:
        """Apply velocity commands to UAVs."""
        for uav in self.uavs:
            if uav.state == UAVState.ACTIVE:
                action = actions.get(uav.id, np.zeros(3, dtype=np.float32))
                action = np.clip(action, -1.0, 1.0).astype(np.float64)
                uav.apply_action(action)

            elif uav.state in (UAVState.RETURNING, UAVState.DEPLOYING):
                if uav.target_station is not None:
                    uav.auto_navigate(uav.target_station.pos)
                else:
                    # Fallback: navigate về station gần nhất
                    nearest = uav.find_nearest_station(self.stations)
                    if nearest is not None:
                        uav.auto_navigate(nearest.pos)

            # CHARGING / DISABLED: no movement

    # ══════════════════════════════════════════════════════════════════════
    # STEP PHYSICS
    # ══════════════════════════════════════════════════════════════════════

    def step_physics(self) -> None:
        """
        Update physics: battery drain/charge.

        Order:
            CHARGING UAV → charge via station
            Others       → drain (proportional to velocity)

        NOTE: Collision detection không disable UAV ở đây.
              Penalty chỉ tính trong reward function (BaselineReward).
        """
        for uav in self.uavs:
            if uav.state == UAVState.DISABLED:
                continue

            if uav.state == UAVState.CHARGING:
                # Tìm station UAV đang occupy
                charged = False
                for station in self.stations:
                    if station.has_uav(uav):
                        station.charge(uav)
                        charged = True
                        break

                if not charged:
                    # UAV ở state CHARGING nhưng không ở station
                    # → drain idle (consistency)
                    uav.update_battery(self.stations)
            else:
                uav.update_battery(self.stations)

    # ══════════════════════════════════════════════════════════════════════
    # STEP WORLD
    # ══════════════════════════════════════════════════════════════════════

    def step_world(self) -> None:
        self._step_count += 1

        # 1. Fleet
        self._fleet_mgr.step()

        # 2. Victim movement - truyền obstacles để obstacle-aware
        for v in self.victims:
            v.update(self._step_count, obstacles=self.obstacles)
            #                          ↑ FIX: truyền obstacles

        # 3. Coverage
        for uav in self.uavs:
            if uav.state == UAVState.DISABLED:
                continue
            fov_r = self._fov_sensor.calculate_fov_radius(uav.pos[2])
            self._cov_map.mark_explored(uav.pos, fov_r, self._step_count)

        # 4. Detection với noise
        for uav in self.uavs:
            if uav.state not in (UAVState.ACTIVE, UAVState.RETURNING):
                continue
            for victim in self.victims:
                if victim.is_found:
                    continue
                if self._fov_sensor.check_detected(uav, victim, self.obstacles):
                    victim.mark_found(self._step_count, uav.id)

    # ══════════════════════════════════════════════════════════════════════
    # GET STATE
    # ══════════════════════════════════════════════════════════════════════

    def get_state(self) -> dict[str, Any]:
        """Return current state của tất cả entities."""
        return {
            "uavs":          self.uavs,
            "victims":       self.victims,
            "stations":      self.stations,
            "obstacles":     self.obstacles,
            "coverage_map":  self._cov_map,
            "fleet_manager": self._fleet_mgr,
        }

    # ══════════════════════════════════════════════════════════════════════
    # BUILD ENTITIES (private helpers)
    # ══════════════════════════════════════════════════════════════════════

    def _build_stations(self, map_data: dict) -> list[ChargingStation]:
        """Build ChargingStation objects từ map_data."""
        stations = []
        for s in map_data["stations"]:
            station = ChargingStation(
                station_id = s["id"],
                pos        = s["pos"],
                cfg        = self.cfg,
            )
            stations.append(station)
        return stations

    def _build_obstacles(self, map_data: dict) -> list[Debris | DangerZone]:
        """
        Build Debris + DangerZone objects từ map_data.

        Supports multi-shape: circle | rectangle | polygon
        Unknown shapes are skipped with warning.
        """
        obstacles = []

        # ── Debris ───────────────────────────────────────────────────────
        for d_dict in map_data.get("debris", []):
            shape     = d_dict.get("shape", "circle")
            debris_id = d_dict["id"]
            pos       = d_dict["pos"]
            height_3d = d_dict.get("height_3d", 10.0)

            if shape == "circle":
                obj = Debris(
                    debris_id = debris_id, pos = pos,
                    height_3d = height_3d, cfg = self.cfg,
                    shape = "circle", radius = d_dict["radius"],
                )
            elif shape == "rectangle":
                obj = Debris(
                    debris_id = debris_id, pos = pos,
                    height_3d = height_3d, cfg = self.cfg,
                    shape = "rectangle",
                    width     = d_dict["width"],
                    height_2d = d_dict["height_2d"],
                    rotation  = d_dict["rotation"],
                )
            elif shape == "polygon":
                obj = Debris(
                    debris_id = debris_id, pos = pos,
                    height_3d = height_3d, cfg = self.cfg,
                    shape = "polygon", vertices = d_dict["vertices"],
                )
            else:
                logger.warning("Unknown debris shape '%s', skipping id=%d", shape, debris_id)
                continue

            obstacles.append(obj)

        # ── Danger Zones ─────────────────────────────────────────────────
        for z_dict in map_data.get("danger_zones", []):
            shape       = z_dict.get("shape", "circle")
            zone_id     = z_dict["id"]
            pos         = z_dict["pos"]
            danger_type = z_dict["danger_type"]

            if shape == "circle":
                obj = DangerZone(
                    zone_id     = zone_id, pos = pos,
                    danger_type = danger_type, cfg = self.cfg,
                    shape = "circle", radius = z_dict["radius"],
                )
            elif shape == "rectangle":
                obj = DangerZone(
                    zone_id     = zone_id, pos = pos,
                    danger_type = danger_type, cfg = self.cfg,
                    shape = "rectangle",
                    width     = z_dict["width"],
                    height_2d = z_dict["height_2d"],
                    rotation  = z_dict["rotation"],
                )
            else:
                logger.warning("Unknown zone shape '%s', skipping id=%d", shape, zone_id)
                continue

            obstacles.append(obj)

        return obstacles

    def _build_victims(self, map_data: dict) -> list[InjuredVictim | MobileVictim]:
        """Build Victim objects từ map_data."""
        victims = []
        for v in map_data["victims"]:
            victim_type = v.get("victim_type", "injured")
            cls         = MobileVictim if victim_type == "mobile" else InjuredVictim

            victim = cls(
                victim_id = v["id"],
                pos       = v["pos"],
                cfg       = self.cfg,
                urgency   = v.get("urgency", 3.0),
            )
            victims.append(victim)
        return victims

    def _build_uavs(self, map_data: dict) -> list[UAV]:
        """
        Build UAV objects từ map_data["uav_spawns"].

        ✅ FIX 4.2: Dùng pre-generated spawn positions từ MapGenerator.generate()
                    KHÔNG gọi lại get_uav_spawns() để tránh RNG drift.

        map_data["uav_spawns"] format:
            [{"id": 0, "pos": [x, y, z]}, ...]
        """
        spawns = map_data.get("uav_spawns", [])

        # Fallback nếu uav_spawns empty (backward compat)
        if not spawns:
            logger.warning(
                "map_data['uav_spawns'] is empty, "
                "falling back to station-based spawn. "
                "Check MapGenerator.generate() FIX-P10."
            )
            # Emergency fallback: spawn tại stations
            for i, station in enumerate(self.stations):
                if i >= self.cfg.env.n_uav:
                    break
                spawns.append({
                    "id":  i,
                    "pos": [station.pos[0], station.pos[1], self.cfg.uav.z_min],
                })

        uavs = []
        for s in spawns:
            uav = UAV(
                uav_id  = s["id"],
                pos     = s["pos"],
                cfg     = self.cfg,
                battery = 100.0,
            )
            uavs.append(uav)

        return uavs