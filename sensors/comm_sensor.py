from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from utils.geometry import dist_3d, compute_bearing

# ✅ REMOVED: sys.path.append hack

if TYPE_CHECKING:
    from entities.uav import UAV
    from config import AppConfig

logger = logging.getLogger(__name__)

"""
sensors/comm_sensor.py
Communication Sensor - V2V giữa các UAV trong COMM_RANGE
"""


class CommSensor:
    """V2V Communication Sensor."""

    def __init__(self, cfg: "AppConfig") -> None:
        self.cfg = cfg
        self._n_tracked    = cfg.obs.n_tracked_uavs
        self._dims_per_uav = 3
        self.obs_dim       = self._n_tracked * self._dims_per_uav
        self._comm_range   = float(cfg.sensor.comm_range)
        self._z_max        = float(cfg.uav.z_max)

    def scan(
        self,
        ego_uav:         "UAV",
        all_active_uavs: List["UAV"],
    ) -> np.ndarray:
        """Quét teammates gần nhất trong COMM_RANGE."""
        result  = np.zeros(self.obs_dim, dtype=np.float32)
        ego_pos = ego_uav.pos
        ego_vel = ego_uav.vel

        candidates: List[Tuple[float, "UAV"]] = []
        for uav in all_active_uavs:
            if uav.id == ego_uav.id:
                continue
            d = dist_3d(ego_pos, uav.pos)
            if d > self._comm_range:
                continue
            candidates.append((d, uav))

        if not candidates:
            return result

        candidates.sort(key=lambda x: x[0])

        for i, (d, teammate) in enumerate(candidates[:self._n_tracked]):
            base         = i * self._dims_per_uav
            norm_dist    = float(np.clip(d / self._comm_range, 0.0, 1.0))
            bearing_rad  = compute_bearing(ego_pos, ego_vel, teammate.pos)
            norm_bearing = float(
                np.clip((bearing_rad + np.pi) / (2.0 * np.pi), 0.0, 1.0)
            )
            rel_alt  = float(teammate.pos[2]) - float(ego_pos[2])
            norm_alt = float(np.clip(rel_alt / self._z_max, -1.0, 1.0))

            result[base + 0] = norm_dist
            result[base + 1] = norm_bearing
            result[base + 2] = norm_alt

        return result

    def get_n_in_range(self, ego_uav: "UAV", all_uavs: List["UAV"]) -> int:
        return sum(
            1 for uav in all_uavs
            if uav.id != ego_uav.id
            and dist_3d(ego_uav.pos, uav.pos) <= self._comm_range
        )

    def get_teammates_in_range(
        self, ego_uav: "UAV", all_uavs: List["UAV"]
    ) -> List["UAV"]:
        in_range = [
            (dist_3d(ego_uav.pos, uav.pos), uav)
            for uav in all_uavs
            if uav.id != ego_uav.id
            and dist_3d(ego_uav.pos, uav.pos) <= self._comm_range
        ]
        in_range.sort(key=lambda x: x[0])
        return [uav for _, uav in in_range]