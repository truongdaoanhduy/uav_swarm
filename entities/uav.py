"""
entities/uav.py
UAV entity cho SAR UAV Swarm - v2.1

Changes vs v2.0:
    BUG-36: Add battery_pct property (was missing, caused AttributeError)
    CLEAN:  Remove sys.path hack (dùng relative import ở caller)
    CLEAN:  Remove duplicate get_battery_penalty (delegate đến reward fn)
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from entities.charging_station import ChargingStation
    from config import AppConfig

logger = logging.getLogger(__name__)


class UAVState(Enum):
    """Trạng thái UAV."""
    ACTIVE    = "active"
    RETURNING = "returning"
    CHARGING  = "charging"
    DEPLOYING = "deploying"
    DISABLED  = "disabled"


class UAV:
    """
    UAV trong SAR Swarm.

    Battery convention:
        self.battery    → float ∈ [0.0, 100.0]  (percentage)
        self.battery_pct → same value, explicit name
        
        Dùng battery_pct trong reward functions để code tự-documenting.
    """

    STATE_ORDER = [
        UAVState.ACTIVE,
        UAVState.RETURNING,
        UAVState.CHARGING,
        UAVState.DEPLOYING,
        UAVState.DISABLED,
    ]

    def __init__(
        self,
        uav_id:  int,
        pos:     List[float],
        cfg:     "AppConfig",
        battery: float    = 100.0,
        state:   UAVState = UAVState.ACTIVE,
    ) -> None:
        self.id      = uav_id
        self.pos     = np.array(pos, dtype=np.float64)
        self.vel     = np.zeros(3, dtype=np.float64)
        self.battery = float(np.clip(battery, 0.0, 100.0))
        self.state   = state
        self.cfg     = cfg

        self.target_station: Optional["ChargingStation"] = None
        self.pybullet_body_id: Optional[int] = None

        # Per-episode tracking
        self.steps_alive:   int   = 0
        self.distance_xy:   float = 0.0
        self.distance_3d:   float = 0.0
        self.victims_found: int   = 0
        self.battery_death: bool  = False

    # ─── Battery Properties ───────────────────────────────────────────────────

    @property
    def battery_pct(self) -> float:
        """
        Battery percentage (0.0 → 100.0).

        ✅ FIX BUG-36: Add this property.
        
        Alias for self.battery with explicit semantic name.
        Used in reward functions for clarity:
            uav.battery_pct <= 5.0  (vs magic number check)
        """
        return self.battery

    @battery_pct.setter
    def battery_pct(self, value: float) -> None:
        """Allow setting via battery_pct for symmetry."""
        self.battery = float(np.clip(value, 0.0, 100.0))

    # ─── Action & Movement ────────────────────────────────────────────────────

    def apply_action(self, action: np.ndarray) -> None:
        """
        Áp dụng action từ policy (chỉ ACTIVE mới nhận).

        Args:
            action: [ax, ay, az] ∈ [-1, 1]³
        """
        if self.state != UAVState.ACTIVE:
            self.vel[:] = 0.0
            return

        action = np.clip(action, -1.0, 1.0)

        vx = action[0] * self.cfg.uav.max_speed_xy
        vy = action[1] * self.cfg.uav.max_speed_xy
        vz = action[2] * self.cfg.uav.max_speed_z

        # Cap diagonal speed XY
        speed_xy = np.sqrt(vx**2 + vy**2)
        if speed_xy > self.cfg.uav.max_speed_xy:
            scale = self.cfg.uav.max_speed_xy / speed_xy
            vx   *= scale
            vy   *= scale

        self.vel = np.array([vx, vy, vz], dtype=np.float64)

        prev_pos  = self.pos.copy()
        self.pos  = self.pos + self.vel * self.cfg.env.dt

        # Clip altitude + boundary
        self.pos[2] = np.clip(self.pos[2], self.cfg.uav.z_min, self.cfg.uav.z_max)
        self.pos[0] = np.clip(self.pos[0], 0.0, self.cfg.env.map_size)
        self.pos[1] = np.clip(self.pos[1], 0.0, self.cfg.env.map_size)

        # Track distance
        delta = self.pos - prev_pos
        self.distance_xy += float(np.sqrt(delta[0]**2 + delta[1]**2))
        self.distance_3d += float(np.linalg.norm(delta))

    def auto_navigate(self, target_pos: np.ndarray) -> None:
        """
        Tự động bay đến target (RETURNING / DEPLOYING).
        No overshoot: step = min(max_speed × dt, dist).

        Args:
            target_pos: [x, y, z]
        """
        target = np.array(target_pos, dtype=np.float64)
        diff   = target - self.pos
        dist   = np.linalg.norm(diff)

        if dist < 0.05:
            self.vel = np.zeros(3, dtype=np.float64)
            return

        direction = diff / dist

        vx = direction[0] * self.cfg.uav.max_speed_xy
        vy = direction[1] * self.cfg.uav.max_speed_xy
        vz = direction[2] * self.cfg.uav.max_speed_z

        # Cap diagonal XY
        speed_xy = np.sqrt(vx**2 + vy**2)
        if speed_xy > self.cfg.uav.max_speed_xy:
            scale = self.cfg.uav.max_speed_xy / speed_xy
            vx   *= scale
            vy   *= scale

        self.vel      = np.array([vx, vy, vz], dtype=np.float64)

        # No-overshoot step
        step      = self.vel * self.cfg.env.dt
        step_dist = np.linalg.norm(step)
        if step_dist > dist:
            step = direction * dist

        prev_pos  = self.pos.copy()
        self.pos  = self.pos + step

        # Boundary clip
        self.pos[0] = np.clip(self.pos[0], 0.0, self.cfg.env.map_size)
        self.pos[1] = np.clip(self.pos[1], 0.0, self.cfg.env.map_size)

        # Altitude clip theo state
        if self.state == UAVState.ACTIVE:
            self.pos[2] = np.clip(self.pos[2], self.cfg.uav.z_min, self.cfg.uav.z_max)
        elif self.state == UAVState.RETURNING:
            self.pos[2] = np.clip(self.pos[2], 0.0, self.cfg.uav.z_max)
        elif self.state == UAVState.CHARGING:
            self.pos[2] = np.clip(self.pos[2], 0.0, 0.5)
        elif self.state == UAVState.DEPLOYING:
            self.pos[2] = np.clip(self.pos[2], 0.0, self.cfg.uav.z_max)

        # Track distance
        delta = self.pos - prev_pos
        self.distance_xy += float(np.sqrt(delta[0]**2 + delta[1]**2))
        self.distance_3d += float(np.linalg.norm(delta))

    # ─── Battery ──────────────────────────────────────────────────────────────

    def update_battery(self, stations: List["ChargingStation"]) -> None:
        """
        Cập nhật pin theo state mỗi step.

        Drain rates (% per SECOND, × dt để ra per-step):
            ACTIVE:    drain_xy × speed_ratio + drain_z + drain_idle
            RETURNING: same as ACTIVE
            DEPLOYING: same as ACTIVE
            CHARGING:  +charge_rate via station
            DISABLED:  skip
        """
        if self.state == UAVState.DISABLED:
            return

        if self.state == UAVState.CHARGING:
            self._do_charge(stations)

        elif self.state in (UAVState.ACTIVE,
                            UAVState.RETURNING,
                            UAVState.DEPLOYING):
            self._do_drain()

        # Clamp
        self.battery = float(np.clip(self.battery, 0.0, 100.0))

        # Terminal: battery dead
        if self.battery <= 0.0 and not self.battery_death:
            self.battery_death = True
            self.state         = UAVState.DISABLED
            self.vel[:]        = 0.0
            logger.debug(f"UAV {self.id}: battery dead → DISABLED")

    def _do_charge(self, stations: List["ChargingStation"]) -> None:
        """Charge via target_station hoặc nearest in-range."""
        if self.target_station is not None:
            self.target_station.charge(self)
            return

        for station in stations:
            if station.in_range(self.pos):
                station.charge(self)
                self.target_station = station
                return

    def _do_drain(self) -> None:
        """
        Drain battery theo velocity.
        
        FIX BUG-30: Drain × dt_seconds (decoupled từ simulation step).
        """
        dt       = self.cfg.env.dt_seconds
        speed_xy = float(np.sqrt(self.vel[0]**2 + self.vel[1]**2))
        vz       = float(self.vel[2])
        vz_up    = max(0.0,  vz)
        vz_down  = max(0.0, -vz)

        max_xy  = self.cfg.uav.max_speed_xy_mps
        max_z   = self.cfg.uav.max_speed_z_mps

        # Proportional drain (0 speed → 0 motion drain, still idle drain)
        drain_xy   = self.cfg.uav.drain_xy_pct_per_s   * (speed_xy / max_xy if max_xy > 0 else 0.0)
        drain_up   = self.cfg.uav.drain_z_up_pct_per_s  * (vz_up   / max_z  if max_z  > 0 else 0.0)
        drain_down = self.cfg.uav.drain_z_down_pct_per_s * (vz_down / max_z  if max_z  > 0 else 0.0)
        drain_idle = self.cfg.uav.drain_idle_pct_per_s

        self.battery -= (drain_xy + drain_up + drain_down + drain_idle) * dt

    def get_battery_penalty(self) -> float:
        """
        Legacy method: progressive battery penalty.
        
        NOTE: Prefer _battery_penalty_single() trong baseline_reward.py
        vì nó nhận cfg explicitly → dễ test hơn.
        
        Kept for backward compatibility với code cũ.
        """
        cfg_r   = self.cfg.reward
        cfg_uav = self.cfg.uav

        if self.battery_pct <= cfg_uav.battery_emergency_pct:
            return cfg_r.r_battery_5
        if self.battery_pct <= cfg_uav.battery_critical_pct:
            return cfg_r.r_battery_10
        if self.battery_pct <= cfg_uav.battery_warning_pct:
            return cfg_r.r_battery_20
        return 0.0

    # ─── Sensor ───────────────────────────────────────────────────────────────

    def get_fov_radius(self) -> float:
        """FOV radius tại độ cao hiện tại (meters)."""
        return float(self.pos[2]) * self.cfg.sensor.fov_tan

    # ─── State helpers ────────────────────────────────────────────────────────

    def get_state_onehot(self) -> np.ndarray:
        """One-hot encoding: [ACTIVE, RETURNING, CHARGING, DEPLOYING, DISABLED]."""
        onehot = np.zeros(len(self.STATE_ORDER), dtype=np.float32)
        for i, s in enumerate(self.STATE_ORDER):
            if self.state == s:
                onehot[i] = 1.0
                break
        return onehot

    def set_state(self, new_state: UAVState) -> None:
        """Chuyển state với validation."""
        if self.state == UAVState.DISABLED:
            return  # terminal

        # CHARGING → ACTIVE: chỉ khi pin đủ
        if (self.state == UAVState.CHARGING and
                new_state == UAVState.ACTIVE and
                self.battery < self.cfg.uav.battery_ready):
            return

        self.state = new_state

    # ─── Convenience predicates ───────────────────────────────────────────────

    def is_active(self)        -> bool: return self.state == UAVState.ACTIVE
    def is_returning(self)     -> bool: return self.state == UAVState.RETURNING
    def is_charging(self)      -> bool: return self.state == UAVState.CHARGING
    def is_deploying(self)     -> bool: return self.state == UAVState.DEPLOYING
    def is_disabled(self)      -> bool: return self.state == UAVState.DISABLED
    def is_operational(self)   -> bool: return self.state != UAVState.DISABLED

    def needs_charging(self) -> bool:
        """Battery ≤ return threshold → cần về trạm."""
        return self.battery_pct <= self.cfg.uav.battery_return_pct

    def is_ready_to_deploy(self) -> bool:
        """Battery ≥ ready threshold → sẵn sàng nhiệm vụ."""
        return self.battery_pct >= self.cfg.uav.battery_ready_pct

    def find_nearest_station(
        self,
        stations: List["ChargingStation"],
    ) -> Optional["ChargingStation"]:
        """Tìm trạm gần nhất còn chỗ (fallback: gần nhất bất kể full)."""
        if not stations:
            return None

        def _dist(s):
            dx = self.pos[0] - s.pos[0]
            dy = self.pos[1] - s.pos[1]
            return float(np.sqrt(dx**2 + dy**2))

        sorted_s = sorted(stations, key=_dist)

        for s in sorted_s:
            if s.is_available():
                return s

        return sorted_s[0]  # fallback

    # ─── Info ─────────────────────────────────────────────────────────────────

    def get_speed_xy(self) -> float:
        return float(np.sqrt(self.vel[0]**2 + self.vel[1]**2))

    def get_speed_3d(self) -> float:
        return float(np.linalg.norm(self.vel))

    def to_dict(self) -> dict:
        return {
            "id":              int(self.id),
            "pos":             self.pos.tolist(),
            "vel":             self.vel.tolist(),
            "battery":         float(self.battery),
            "battery_pct":     float(self.battery_pct),
            "state":           self.state.value,
            "state_onehot":    self.get_state_onehot().tolist(),
            "fov_radius":      float(self.get_fov_radius()),
            "speed_xy":        float(self.get_speed_xy()),
            "steps_alive":     int(self.steps_alive),
            "distance_xy":     float(self.distance_xy),
            "distance_3d":     float(self.distance_3d),
            "victims_found":   int(self.victims_found),
            "battery_death":   bool(self.battery_death),
            "needs_charging":  bool(self.needs_charging()),
            "ready_to_deploy": bool(self.is_ready_to_deploy()),
            "target_station":  int(self.target_station.id)
                               if self.target_station else None,
        }

    def __repr__(self) -> str:
        return (
            f"UAV(id={self.id}, "
            f"state={self.state.value}, "
            f"bat={self.battery:.1f}%, "
            f"pos=[{self.pos[0]:.1f}, {self.pos[1]:.1f}, {self.pos[2]:.1f}])"
        )