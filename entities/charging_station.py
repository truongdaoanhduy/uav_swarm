

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from config import AppConfig


logger = logging.getLogger(__name__)

"""
entities/charging_station.py
Trạm sạc cho UAV trong hệ thống SAR
"""

class ChargingStation:
    """
    Trạm sạc UAV với nhiều cổng sạc.

    Đặc điểm:
        - Capacity = 2 UAVs cùng lúc (từ cfg)
        - UAV PHẢI HẠ THẤP (z ≤ 0.5m) mới sạc được
        - Auto release khi UAV ra khỏi range hoặc pin đầy
        - Trạm KHÔNG BỊ HƯ (simplified)

    Args:
        station_id: ID định danh
        pos:        Vị trí trạm [x, y] hoặc [x, y, z]
        cfg:        AppConfig object

    Usage:
        station = ChargingStation(0, [10.0, 20.0], cfg)
        station.charge(uav)   # sạc pin
        station.release(uav)  # giải phóng slot
    """

    def __init__(
        self,
        station_id: int,
        pos: list,
        cfg: "AppConfig",
    ) -> None:
        self.id  = station_id
        self.pos = np.array([pos[0], pos[1], 0.0], dtype=np.float64)
        self.cfg = cfg

        # Lấy params từ cfg (không hardcode)
        self.capacity      = cfg.env.station_capacity  # 2
        self.charge_radius = cfg.env.charge_radius     # 3.0m
        self.charge_rate   = cfg.uav.charge_rate       # 1.5%/step

        # Occupant tracking
        self.current_occupants: List = []
        self.occupant_ids: set = set()  # O(1) lookup

        logger.debug(
            "ChargingStation %d khởi tạo tại (%.1f, %.1f), capacity=%d",
            self.id, self.pos[0], self.pos[1], self.capacity,
        )

    # ─── Query methods ────────────────────────────────────────────────────────

    def is_full(self) -> bool:
        """Trạm đã đầy chưa."""
        return len(self.current_occupants) >= self.capacity

    def is_occupied(self) -> bool:
        """Có ít nhất 1 UAV đang sạc."""
        return len(self.current_occupants) > 0

    def is_available(self) -> bool:
        """Trạm còn chỗ trống."""
        return len(self.current_occupants) < self.capacity

    def get_occupancy(self) -> int:
        """Số UAV đang sạc."""
        return len(self.current_occupants)

    def get_occupancy_ratio(self) -> float:
        """
        Tỉ lệ lấp đầy.

        Returns:
            float: [0.0, 1.0]
        """
        return len(self.current_occupants) / self.capacity

    def has_uav(self, uav) -> bool:
        """
        Kiểm tra UAV có đang sạc ở trạm này không.

        Args:
            uav: UAV object

        Returns:
            bool: True nếu UAV đang ở trạm
        """
        return uav.id in self.occupant_ids

    def in_range(self, uav_pos: np.ndarray) -> bool:
        """
        Kiểm tra UAV có trong vùng sạc không.

        Điều kiện:
            - Khoảng cách XY ≤ charge_radius
            - Độ cao z ≤ 0.5m (UAV phải hạ xuống gần mặt đất)

        Args:
            uav_pos: np.ndarray [x, y, z]

        Returns:
            bool: True nếu trong vùng VÀ đủ thấp
        """
        dx = float(uav_pos[0]) - self.pos[0]
        dy = float(uav_pos[1]) - self.pos[1]
        dist_xy = np.sqrt(dx**2 + dy**2)

        is_close_enough = dist_xy <= self.charge_radius
        is_low_enough   = float(uav_pos[2]) <= 0.5

        return is_close_enough and is_low_enough

    # ─── Action methods ───────────────────────────────────────────────────────

    def try_occupy(self, uav) -> bool:
        """
        UAV cố gắng chiếm 1 slot sạc.

        Args:
            uav: UAV object

        Returns:
            bool: True nếu thành công (kể cả đã occupy rồi)
                  False nếu trạm đầy
        """
        # UAV đã đang sạc ở đây → OK
        if uav.id in self.occupant_ids:
            return True

        if self.is_full():
            return False

        self.current_occupants.append(uav)
        self.occupant_ids.add(uav.id)
        return True

    def release(self, uav) -> bool:
        """
        UAV rời khỏi trạm sạc.

        Args:
            uav: UAV object

        Returns:
            bool: True nếu release thành công
                  False nếu UAV không ở đây
        """
        if uav.id not in self.occupant_ids:
            return False

        self.current_occupants = [
            u for u in self.current_occupants if u.id != uav.id
        ]
        self.occupant_ids.discard(uav.id)
        return True

    def charge(self, uav) -> float:
        """
        Sạc pin cho UAV 1 step.

        Logic:
            1. Out of range → auto release, return 0
            2. Pin đầy → auto release, return 0
            3. Occupy slot (nếu chưa)
            4. Sạc min(charge_rate, 100 - battery)

        Args:
            uav: UAV object cần sạc

        Returns:
            float: Lượng pin đã sạc thêm (%)
                   0.0 nếu không sạc được
        """
        # [1] Out of range → auto release (tránh ghost occupancy)
        if not self.in_range(uav.pos):
            self.release(uav)
            return 0.0

        # [2] Pin đầy → auto release
        if uav.battery >= 100.0:
            self.release(uav)
            return 0.0

        # [3] Thử chiếm slot
        if not self.try_occupy(uav):
            return 0.0

        # [4] Sạc pin
        charge_amount = min(self.charge_rate, 100.0 - uav.battery)
        uav.battery  += charge_amount

        # Auto release khi vừa đầy
        if uav.battery >= 100.0:
            uav.battery = 100.0
            self.release(uav)

        return charge_amount

    def force_release_all(self) -> None:
        """
        Giải phóng tất cả slots.
        Dùng khi reset episode.
        """
        self.current_occupants.clear()
        self.occupant_ids.clear()

    # ─── Info methods ─────────────────────────────────────────────────────────

    def get_occupant_ids(self) -> List[int]:
        """Danh sách ID của UAV đang sạc."""
        return list(self.occupant_ids)

    def to_dict(self) -> dict:
        """
        Chuyển thành dict JSON-safe.
        Dùng cho logging và visualization.
        """
        return {
            "id":              int(self.id),
            "pos":             self.pos.tolist(),
            "capacity":        int(self.capacity),
            "charge_radius":   float(self.charge_radius),
            "charge_rate":     float(self.charge_rate),
            "occupancy":       int(self.get_occupancy()),
            "occupancy_ratio": float(self.get_occupancy_ratio()),
            "occupant_ids":    self.get_occupant_ids(),
            "is_full":         bool(self.is_full()),
            "is_available":    bool(self.is_available()),
        }

    def __repr__(self) -> str:
        return (
            f"ChargingStation("
            f"id={self.id}, "
            f"pos=({self.pos[0]:.1f}, {self.pos[1]:.1f}), "
            f"slots={self.get_occupancy()}/{self.capacity})"
        )