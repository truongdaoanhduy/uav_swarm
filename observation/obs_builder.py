from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List

import numpy as np

from core.coverage_map import CoverageMap
from sensors.fov_sensor import FOVSensor
from sensors.comm_sensor import CommSensor
from utils.geometry import dist_2d

# ✅ REMOVED: sys.path.append hack

if TYPE_CHECKING:
    from entities.uav import UAV
    from entities.victim import BaseVictim
    from entities.charging_station import ChargingStation
    from config import AppConfig

logger = logging.getLogger(__name__)

"""
observation/obs_builder.py
Observation Builder - Dec-POMDP + CTDE
"""


class ObsResult:
    """Container: actor obs + critic obs."""
    __slots__ = ("actor_obs", "critic_obs")

    def __init__(
        self,
        actor_obs:  Dict[int, np.ndarray],
        critic_obs: np.ndarray,
    ) -> None:
        self.actor_obs  = actor_obs
        self.critic_obs = critic_obs


class ObservationBuilder:
    """
    Observation Builder cho SAR UAV Swarm.

    Actor obs (68 dims) - LOCAL ONLY (Dec-POMDP compliant)
    Critic obs (MAX_UAV×68 + 10) - GLOBAL STATE (CTDE)
    """

    def __init__(self, coverage_map: CoverageMap, cfg: "AppConfig") -> None:
        self.coverage_map = coverage_map
        self.cfg          = cfg

        self.fov_sensor  = FOVSensor(cfg)
        self.comm_sensor = CommSensor(cfg)

        # Dims từ cfg
        self._self_dim     = cfg.obs.self_dim
        self._station_dim  = cfg.obs.station_dim
        self._team_dim     = cfg.obs.team_dim
        self._obs_dim      = cfg.obs.obstacle_dim
        self._victim_dim   = cfg.obs.victim_dim
        self._coverage_dim = cfg.obs.coverage_dim
        self.actor_dim     = cfg.obs.actor_dim

        # Precompute slices
        dims = [
            self._self_dim,
            self._station_dim,
            self._team_dim,
            self._obs_dim,
            self._victim_dim,
            self._coverage_dim,
        ]
        self.slices: List[slice] = []
        start = 0
        for d in dims:
            self.slices.append(slice(start, start + d))
            start += d

        # Critic dims
        self._max_uav    = cfg.obs.max_uav
        self._global_dim = cfg.obs.global_dim
        self.critic_dim  = cfg.obs.critic_dim

        # Preallocate buffers
        self._actor_bufs: Dict[int, np.ndarray] = {}
        self._critic_buf  = np.zeros(self.critic_dim, dtype=np.float32)

        # Coverage radii
        self._cov_small = cfg.obs.local_cov_small
        self._cov_large = cfg.obs.local_cov_large

        # ✅ FIX 3.3: Debug flag cho sanity assert
        self._debug_obs = getattr(cfg.env, "debug_obs", False)

        logger.debug(
            "ObservationBuilder: actor_dim=%d, critic_dim=%d, max_uav=%d",
            self.actor_dim, self.critic_dim, self._max_uav,
        )

    def _get_actor_buf(self, uav_id: int) -> np.ndarray:
        if uav_id not in self._actor_bufs:
            self._actor_bufs[uav_id] = np.zeros(self.actor_dim, dtype=np.float32)
        return self._actor_bufs[uav_id]

    def _write_self(self, obs: np.ndarray, uav: "UAV") -> None:
        s = self.slices[0].start
        obs[s + 0] = uav.pos[0] / self.cfg.env.map_size
        obs[s + 1] = uav.pos[1] / self.cfg.env.map_size
        obs[s + 2] = uav.pos[2] / self.cfg.uav.z_max
        obs[s + 3] = uav.vel[0] / self.cfg.uav.max_speed_xy
        obs[s + 4] = uav.vel[1] / self.cfg.uav.max_speed_xy
        obs[s + 5] = uav.vel[2] / self.cfg.uav.max_speed_z
        obs[s + 6] = uav.battery / 100.0
        obs[s + 7: s + 11] = uav.get_state_onehot()[:4]

    def _write_stations(
        self,
        obs:      np.ndarray,
        uav:      "UAV",
        stations: List["ChargingStation"],
    ) -> None:
        s = self.slices[1]
        obs[s] = 0.0
        for i, st in enumerate(stations[:self.cfg.env.n_stations]):
            b          = s.start + i * 4
            obs[b + 0] = (st.pos[0] - uav.pos[0]) / self.cfg.env.map_size
            obs[b + 1] = (st.pos[1] - uav.pos[1]) / self.cfg.env.map_size
            obs[b + 2] = dist_2d(uav.pos, st.pos) / self.cfg.map_diagonal
            obs[b + 3] = st.get_occupancy_ratio()

    def _write_teammates(
        self, obs: np.ndarray, uav: "UAV", all_uavs: List["UAV"]
    ) -> None:
        obs[self.slices[2]] = self.comm_sensor.scan(uav, all_uavs)

    def _write_obstacles(
        self, obs: np.ndarray, uav: "UAV", obstacles: list
    ) -> None:
        obs[self.slices[3]] = self.fov_sensor.scan_obstacles(uav, obstacles)

    def _write_victims(
        self,
        obs:       np.ndarray,
        uav:       "UAV",
        victims:   List["BaseVictim"],
        obstacles: list,
    ) -> None:
        obs[self.slices[4]] = self.fov_sensor.scan_victims(uav, victims, obstacles)

    def _write_coverage(
        self, obs: np.ndarray, uav: "UAV", current_step: int
    ) -> None:
        s          = self.slices[5].start
        obs[s + 0] = self.coverage_map.get_local_coverage(uav.pos, self._cov_small)
        obs[s + 1] = self.coverage_map.get_local_coverage(uav.pos, self._cov_large)
        obs[s + 2] = (
            max(0, self.cfg.env.max_steps - current_step) / self.cfg.env.max_steps
        )

    def build_actor_obs(
        self,
        uav:          "UAV",
        all_uavs:     List["UAV"],
        stations:     List["ChargingStation"],
        victims:      List["BaseVictim"],
        obstacles:    list,
        current_step: int,
    ) -> np.ndarray:
        """Build actor obs cho 1 UAV."""
        obs      = self._get_actor_buf(uav.id)
        obs[:]   = 0.0

        self._write_self(obs, uav)
        self._write_stations(obs, uav, stations)
        self._write_teammates(obs, uav, all_uavs)
        self._write_obstacles(obs, uav, obstacles)
        self._write_victims(obs, uav, victims, obstacles)
        self._write_coverage(obs, uav, current_step)

        # ✅ FIX 3.3: Sanity check chỉ khi debug mode
        if self._debug_obs:
            assert obs.shape == (self.actor_dim,), \
                f"Shape error: {obs.shape} != ({self.actor_dim},)"
            assert not np.any(np.isnan(obs)), \
                f"NaN in actor obs UAV {uav.id}"
            assert not np.any(np.isinf(obs)), \
                f"Inf in actor obs UAV {uav.id}"

        return obs

    def build_all(
        self,
        all_uavs:     List["UAV"],
        stations:     List["ChargingStation"],
        victims:      List["BaseVictim"],
        obstacles:    list,
        current_step: int,
    ) -> ObsResult:
        """Build TẤT CẢ obs trong 1 lần."""
        from entities.uav import UAVState

        # ── 1. Actor obs ─────────────────────────────────────────────────────
        actor_obs: Dict[int, np.ndarray] = {}

        for uav in all_uavs:
            if uav.state == UAVState.DISABLED:
                buf      = self._get_actor_buf(uav.id)
                buf[:]   = 0.0
                actor_obs[uav.id] = buf
            else:
                actor_obs[uav.id] = self.build_actor_obs(
                    uav, all_uavs, stations, victims, obstacles, current_step
                )

        # ── 2. Critic: UAV part ──────────────────────────────────────────────
        critic   = self._critic_buf
        critic[:] = 0.0

        # ✅ FIX 3.2: Stable ordering → sorted by uav.id
        all_uavs_sorted = sorted(all_uavs, key=lambda u: u.id)

        for i in range(self._max_uav):
            start = i * self.actor_dim
            end   = start + self.actor_dim
            if i < len(all_uavs_sorted):
                uav = all_uavs_sorted[i]
                critic[start:end] = actor_obs[uav.id]
            # else: đã là 0

        # ── 3. Global info (10 dims) ─────────────────────────────────────────
        n  = max(len(all_uavs), 1)
        g  = self._max_uav * self.actor_dim

        bats_live  = [
            u.battery for u in all_uavs
            if u.state != UAVState.DISABLED
        ]
        n_active   = sum(1 for u in all_uavs if u.state == UAVState.ACTIVE)
        n_charging = sum(1 for u in all_uavs if u.state == UAVState.CHARGING)
        n_disabled = sum(1 for u in all_uavs if u.state == UAVState.DISABLED)
        n_alive    = n - n_disabled

        mean_bat = float(np.mean(bats_live)) / 100.0 if bats_live else 0.0
        std_bat  = float(np.std(bats_live))  / 100.0 if bats_live else 0.0
        min_bat  = float(np.min(bats_live))  / 100.0 if bats_live else 0.0

        global_cov    = self.coverage_map.get_coverage_rate()
        n_victims     = max(len(victims), 1)
        victims_found = sum(1 for v in victims if v.is_found)
        time_rem      = max(0, self.cfg.env.max_steps - current_step)

        critic[g + 0] = n_active   / n
        critic[g + 1] = n_charging / n
        critic[g + 2] = n_disabled / n
        critic[g + 3] = n_alive    / n
        critic[g + 4] = mean_bat
        critic[g + 5] = std_bat
        critic[g + 6] = min_bat
        critic[g + 7] = global_cov
        critic[g + 8] = victims_found / n_victims
        critic[g + 9] = time_rem / self.cfg.env.max_steps

        return ObsResult(actor_obs=actor_obs, critic_obs=critic)