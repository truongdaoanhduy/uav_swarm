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
import random  # ✅ THÊM import
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
        """Reset với deterministic seed từ map_data."""

        # ✅ FIX: Lấy episode seed từ map_data (do MapGenerator tạo)
        # map_data["seed"] = seed đã được base_env truyền vào generate()
        episode_seed = map_data.get("seed", None)

        if self.cfg.env.deterministic_eval:
            # Eval mode: seed cố định
            fixed_seed = self.cfg.env.eval_seed
            np.random.seed(fixed_seed)
            random.seed(fixed_seed)          # ✅ THÊM: python random
            self._fov_sensor.set_seed(fixed_seed)

        else:
            # Training mode: dùng episode_seed (từ map_data)
            if episode_seed is not None:
                # ✅ FIX: Set numpy + python random theo episode seed
                np.random.seed(episode_seed % (2**32))
                random.seed(episode_seed)
                self._fov_sensor.set_seed(episode_seed)
            else:
                # Fallback (không nên xảy ra)
                fallback = int(np.random.randint(0, 2**31))
                self._fov_sensor.set_seed(fallback)

        # Build entities (giữ nguyên)
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

    def _find_nearest_station_in_range(
        self,
        uav,
        range_m: float,
    ):
        """
        Tìm station gần nhất trong range_m.
        Returns None nếu không có station nào trong range.
        """
        best      = None
        best_dist = float("inf")

        for station in self.stations:
            dx   = float(uav.pos[0]) - float(station.pos[0])
            dy   = float(uav.pos[1]) - float(station.pos[1])
            dist = float(np.sqrt(dx * dx + dy * dy))

            if dist <= range_m and dist < best_dist:
                best      = station
                best_dist = dist

        return best

    # env_setup/backends/logic_backend.py

    def apply_actions(self, actions: dict[int, np.ndarray]) -> None:
        """
        Apply velocity commands + landing signal to UAVs.

        Action format: [vx, vy, vz, land]
            - [vx, vy, vz] ∈ [-1, 1]³  : movement
            - [land]       ∈ {0.0, 1.0} : landing intent

        Landing logic (ALL conditions must hold):
            1. land > 0.5           → agent muốn land
            2. state == ACTIVE      → chỉ active mới được land
            3. station available    → có station gần
            4. battery ≤ 60%        → ✅ FIX: chỉ land khi cần (tránh land ở 96%)
            
        Note:
            - land=1 nhưng battery > 60% → move bình thường (ignore land)
            - land=1 nhưng không có station → move bình thường
        """
        # ✅ FIX: Battery threshold để prevent random landing
        LANDING_BATTERY_THRESHOLD = 40.0  # %
        
        for uav in self.uavs:

            if uav.state == UAVState.ACTIVE:
                action = actions.get(uav.id, np.zeros(4, dtype=np.float32))
                action = np.clip(action, -1.0, 1.0).astype(np.float64)

                move_action = action[:3]    # [vx, vy, vz]
                land_signal = float(action[3])

                # ✅ Landing conditions
                wants_to_land = land_signal > 0.5

                if wants_to_land:
                    # Tìm station gần nhất
                    nearest = uav.find_nearest_station(self.stations)
                    
                    if nearest is not None:
                        # ✅ FIX: Chỉ land khi battery ≤ threshold
                        if uav.battery <= LANDING_BATTERY_THRESHOLD:
                            # Hạ cánh: navigate xuống station
                            target = np.array([
                                nearest.pos[0],
                                nearest.pos[1],
                                0.0,   # Ground level
                            ], dtype=np.float64)
                            uav.auto_navigate(target)
                            uav.target_station = nearest
                            uav.set_state(UAVState.RETURNING)
                            
                            logger.debug(
                                "UAV %d landing accepted: battery=%.1f%% ≤ %.1f%%",
                                uav.id, uav.battery, LANDING_BATTERY_THRESHOLD
                            )
                        else:
                            # Pin còn nhiều → ignore land signal
                            uav.apply_action(move_action)
                            
                            # Log 1 lần per episode để debug (không spam)
                            if not hasattr(uav, '_logged_high_battery_land'):
                                logger.debug(
                                    "UAV %d land rejected: battery=%.1f%% > %.1f%% "
                                    "(ignored, apply move)",
                                    uav.id, uav.battery, LANDING_BATTERY_THRESHOLD
                                )
                                uav._logged_high_battery_land = True
                    else:
                        # Không có station → move bình thường
                        uav.apply_action(move_action)
                else:
                    # Normal movement
                    uav.apply_action(move_action)
                    
            elif uav.state == UAVState.RETURNING:
                if uav.target_station is not None:
                    target = np.array([
                        uav.target_station.pos[0],
                        uav.target_station.pos[1],
                        0.0,
                    ], dtype=np.float64)
                    uav.auto_navigate(target)
                else:
                    # Fallback: tìm nearest nếu target bị mất
                    nearest = uav.find_nearest_station(self.stations)
                    if nearest is not None:
                        target = np.array([
                            nearest.pos[0],
                            nearest.pos[1],
                            0.0,
                        ], dtype=np.float64)
                        uav.auto_navigate(target)
                        uav.target_station = nearest

            elif uav.state == UAVState.DEPLOYING:
                # Simple: chuyển ACTIVE ngay
                uav.set_state(UAVState.ACTIVE)
                uav.target_station = None
                
            # CHARGING / DISABLED: no movement

    # ══════════════════════════════════════════════════════════════════════
    # STEP PHYSICS
    # ══════════════════════════════════════════════════════════════════════

    def step_physics(self) -> None:
        for uav in self.uavs:
            if uav.state == UAVState.DISABLED:
                continue

            # ✅ RETURNING → CHARGING transition
            if uav.state == UAVState.RETURNING:
                if uav.target_station is not None:
                    if uav.target_station.in_range(uav.pos):
                        if uav.target_station.try_occupy(uav):
                            uav.set_state(UAVState.CHARGING)
                            logger.debug(
                                f"UAV {uav.id} → CHARGING "
                                f"at station {uav.target_station.id}"
                            )
                # RETURNING vẫn drain
                uav.update_battery(self.stations)
                continue  # ← Skip phần dưới

            # ✅ CHARGING: charge via station
            if uav.state == UAVState.CHARGING:
                if uav.target_station is not None:
                    uav.target_station.charge(uav)
                else:
                    # Fallback: tìm station đang occupy
                    for station in self.stations:
                        if station.has_uav(uav):
                            station.charge(uav)
                            break
                continue  # ← Skip update_battery

            # ACTIVE / DEPLOYING: drain
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