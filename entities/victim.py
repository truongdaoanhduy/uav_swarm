from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from utils.geometry import dist_2d, check_los_2d

if TYPE_CHECKING:
    from config import AppConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# BASE CLASS
# ═══════════════════════════════════════════════════════════════════════

class BaseVictim(ABC):
    """
    Abstract base class cho tất cả victim types.

    FIXES:
        - update(step) alias cho step() → compatible với logic_backend
        - mark_found() → subclass override để freeze speed
    """

    def __init__(
        self,
        victim_id: int,
        pos:       List[float],
        urgency:   float,
        cfg:       "AppConfig",
    ) -> None:
        self.id      = victim_id
        self.pos     = np.array([pos[0], pos[1], 0.0], dtype=np.float64)
        self.urgency = float(np.clip(urgency, 1.0, 5.0))
        self.cfg     = cfg

        self.is_found:      bool          = False
        self.found_at_step: Optional[int] = None
        self.found_by_uav:  Optional[int] = None

    # ── Abstract ──────────────────────────────────────────────────────────────

    @abstractmethod
    def step(self, obstacles: List = None) -> None:
        """Update physics mỗi timestep."""
        pass

    @property
    @abstractmethod
    def victim_type(self) -> str:
        pass

    # ── Alias: update() = step() ──────────────────────────────────────────────
    # logic_backend gọi v.update(step_count)
    # victim.py định nghĩa step(obstacles)
    # → Cần bridge cả 2 interface

    def update(self, step_count: int = 0, obstacles: List = None) -> None:
        """
        Alias cho step() - compatible với logic_backend.

        logic_backend gọi: v.update(self._step_count)
        → obstacles=None (logic_backend không truyền obstacles vào update)
        → Nếu cần obstacle-aware movement, truyền từ backend

        Args:
            step_count: Step hiện tại (không dùng trực tiếp, để compatible)
            obstacles:  Optional obstacle list
        """
        self.step(obstacles)

    # ── Detection ─────────────────────────────────────────────────────────────

    def _calc_detection_prob(self, uav_z: float) -> float:
        """P(detect) theo altitude. Chỉ dùng khi KHÔNG có FOVSensor noise."""
        p = (
            self.cfg.sensor.p_detect_base
            * np.exp(-self.cfg.sensor.p_detect_decay * uav_z)
        )
        return float(np.clip(p, 0.0, 1.0))

    def is_detected_by(
        self,
        uav_pos:    np.ndarray,
        fov_radius: float,
        obstacles:  List = None,
        p_detect:   float = None,
    ) -> bool:
        """
        Legacy detection (không có noise model).
        FOVSensor.check_detected() là preferred method với noise đầy đủ.
        Giữ lại để backward compatibility.
        """
        d = dist_2d(uav_pos, self.pos)
        if d > fov_radius:
            return False

        if obstacles:
            if not check_los_2d(uav_pos, self.pos, obstacles):
                return False

        if p_detect is None:
            p_detect = self._calc_detection_prob(uav_pos[2])

        return bool(np.random.rand() < p_detect)

    def mark_found(self, step: int, uav_id: int) -> None:
        """
        Đánh dấu found (chỉ 1 lần).
        Subclass override để thêm behavior (freeze speed, etc.)
        """
        if not self.is_found:
            self.is_found      = True
            self.found_at_step = step
            self.found_by_uav  = uav_id
            self._on_found()   # ← Hook cho subclass

    def _on_found(self) -> None:
        """Hook gọi sau khi mark_found(). Override trong subclass."""
        pass

    # ── Info ──────────────────────────────────────────────────────────────────

    def get_reward_value(self) -> float:
        return self.cfg.reward.r_victim_base * (self.urgency / 5.0)

    def to_dict(self) -> dict:
        return {
            "id":            int(self.id),
            "type":          self.victim_type,
            "pos":           self.pos.tolist(),
            "urgency":       float(self.urgency),
            "is_found":      bool(self.is_found),
            "found_at_step": int(self.found_at_step) if self.found_at_step is not None else None,
            "found_by_uav":  int(self.found_by_uav)  if self.found_by_uav  is not None else None,
        }

    def __repr__(self) -> str:
        status = "FOUND" if self.is_found else "MISSING"
        return (
            f"{self.__class__.__name__}("
            f"id={self.id}, urgency={self.urgency:.1f}, "
            f"pos=({self.pos[0]:.1f},{self.pos[1]:.1f}), "
            f"status={status})"
        )


# ═══════════════════════════════════════════════════════════════════════
# INJURED VICTIM
# ═══════════════════════════════════════════════════════════════════════

class InjuredVictim(BaseVictim):
    """
    Nạn nhân bị thương nặng - KHÔNG DI CHUYỂN.
    Urgency cao [4, 5]. Detection factor = 1.15 (dễ thấy vì nằm im).
    """

    def __init__(
        self,
        victim_id: int,
        pos:       List[float],
        cfg:       "AppConfig",
        urgency:   float = None,
    ) -> None:
        if urgency is None:
            urgency = np.random.uniform(
                cfg.victim.injured_urgency_min,
                cfg.victim.injured_urgency_max,
            )
        super().__init__(victim_id, pos, urgency, cfg)

        # Speed=0 cho FOVSensor._get_victim_factor()
        self.speed = 0.0

    @property
    def victim_type(self) -> str:
        return "injured"

    def step(self, obstacles: List = None) -> None:
        """InjuredVictim không di chuyển."""
        pass

    def _on_found(self) -> None:
        """Đã found → không cần làm gì thêm (đã đứng im)."""
        pass


# ═══════════════════════════════════════════════════════════════════════
# MOBILE VICTIM
# ═══════════════════════════════════════════════════════════════════════

class MobileVictim(BaseVictim):
    """
    Nạn nhân còn di chuyển - RANDOM WALK.
    Urgency thấp [1, 3]. Detection factor tùy speed.

    FIXES:
        - _on_found(): freeze speed = 0.0 ngay lập tức
        - step(): check is_found trước khi move
        - update() nhận obstacles từ backend nếu có
    """

    _BOUNDARY_MARGIN: float = 2.0

    def __init__(
        self,
        victim_id: int,
        pos:       List[float],
        cfg:       "AppConfig",
        urgency:   float = None,
    ) -> None:
        if urgency is None:
            urgency = np.random.uniform(
                cfg.victim.mobile_urgency_min,
                cfg.victim.mobile_urgency_max,
            )
        super().__init__(victim_id, pos, urgency, cfg)

        self.speed      = float(np.random.uniform(
            cfg.victim.mobile_speed_min,
            cfg.victim.mobile_speed_max,
        ))
        self.direction  = float(np.random.uniform(0, 2 * np.pi))
        self.move_timer = 0

    @property
    def victim_type(self) -> str:
        return "mobile"

    def _on_found(self) -> None:
        """
        FIX BUG-04: Freeze ngay khi found.
        speed=0.0 → FOVSensor._get_victim_factor() trả về 1.0 (không penalty)
        """
        self.speed = 0.0
        logger.debug("MobileVictim %d frozen at step %d", self.id, self.found_at_step)

    def step(self, obstacles: List = None) -> None:
        """
        Random walk mỗi step.

        FIX: Check is_found/speed=0 TRƯỚC khi move.
        """
        # [1] Freeze nếu found hoặc speed=0
        if self.is_found or self.speed <= 0:
            return

        # [2] Timer đổi hướng
        self.move_timer += 1
        if self.move_timer >= self.cfg.victim.mobile_dir_change:
            self.direction  = float(np.random.uniform(0, 2 * np.pi))
            self.move_timer = 0

        # [3] Tính new_pos (dt từ config)
        dt = float(getattr(self.cfg.env, "dt_seconds",
                           getattr(self.cfg.env, "dt", 1.0)))
        dx = self.speed * np.cos(self.direction) * dt
        dy = self.speed * np.sin(self.direction) * dt

        new_pos    = self.pos.copy()
        new_pos[0] += dx
        new_pos[1] += dy

        # [4] Boundary clip
        ms     = float(self.cfg.env.map_size)
        margin = self._BOUNDARY_MARGIN

        clipped_x = float(np.clip(new_pos[0], margin, ms - margin))
        clipped_y = float(np.clip(new_pos[1], margin, ms - margin))

        if clipped_x != new_pos[0] or clipped_y != new_pos[1]:
            self.direction  = float(np.random.uniform(0, 2 * np.pi))
            self.move_timer = 0
            new_pos[0]      = clipped_x
            new_pos[1]      = clipped_y

        # [5] Obstacle check
        if obstacles and self._check_obstacle_block(new_pos, obstacles):
            self.direction  = float(np.random.uniform(0, 2 * np.pi))
            self.move_timer = 0
            return  # Giữ nguyên pos

        self.pos = new_pos

    def _check_obstacle_block(self, new_pos: np.ndarray, obstacles: List) -> bool:
        """Chỉ Debris chặn movement. DangerZone không chặn."""
        from entities.obstacle import Debris
        for obs in obstacles:
            if isinstance(obs, Debris) and obs.in_zone_2d(new_pos):
                return True
        return False

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "speed":      float(self.speed),
            "direction":  float(self.direction),
            "move_timer": int(self.move_timer),
        })
        return d