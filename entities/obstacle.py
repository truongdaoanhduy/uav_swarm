from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple
import numpy as np

from utils.geometry import dist_2d, _line_intersects_circle


if TYPE_CHECKING:
    from config import AppConfig

logger = logging.getLogger(__name__)

# ─── Shapely import (lazy) ────────────────────────────────────────────────────
try:
    from shapely.geometry import Point, LineString, Polygon as ShapelyPolygon
    from shapely.geometry import box as shapely_box
    from shapely.ops import nearest_points
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    Point          = None
    LineString     = None
    ShapelyPolygon = None
    shapely_box    = None
    nearest_points = None

# Loại danger chắn LOS
_LOS_BLOCKING_TYPES = {"fire", "smoke"}


# ═══════════════════════════════════════════════════════════════════════════
# DEBRIS - Hỗ trợ nhiều hình dạng
# ═══════════════════════════════════════════════════════════════════════════
class Debris:
    """
    Mảnh vỡ tòa nhà sập - HỖ TRỢ NHIỀU HÌNH DẠNG.

    Hình dạng:
        - circle:    Hình tròn (center + radius)
        - rectangle: Hình chữ nhật (center + width + height + rotation)
        - polygon:   Hình đa giác bất kỳ (danh sách vertices)

    UAV va chạm khi:
        - Trong vùng XY của debris
        - uav.z < height_3d

    Args:
        debris_id: ID định danh
        pos:       Vị trí tâm [x, y] hoặc [x, y, z]
        height_3d: Độ cao debris (mét)
        cfg:       AppConfig object
        
        # Shape params (chọn 1 trong 3)
        shape:     "circle" | "rectangle" | "polygon"
        radius:    Bán kính (chỉ dùng với circle)
        width:     Chiều rộng (chỉ dùng với rectangle)
        height_2d: Chiều dài (chỉ dùng với rectangle)
        rotation:  Góc xoay theo độ (chỉ dùng với rectangle)
        vertices:  List vertices [[x1,y1], [x2,y2], ...] (chỉ dùng với polygon)
    """

    def __init__(
        self,
        debris_id: int,
        pos:       List[float],
        height_3d: float,
        cfg:       "AppConfig",
        # Shape params
        shape:     str = "circle",
        radius:    Optional[float] = None,
        width:     Optional[float] = None,
        height_2d: Optional[float] = None,
        rotation:  float = 0.0,
        vertices:  Optional[List[List[float]]] = None,
    ) -> None:
        self.id        = debris_id
        self.pos       = np.array([pos[0], pos[1], 0.0], dtype=np.float64)
        self.height_3d = float(height_3d)
        self.cfg       = cfg
        self.penalty   = cfg.reward.r_collision_obstacle  # -50.0

        self.shape     = shape
        self.radius    = None
        self.width     = None
        self.height_2d = None
        self.rotation  = rotation
        self.vertices  = None
        self.polygon   = None  # Shapely polygon object

        # ── Validate shape params ──
        if shape == "circle":
            if radius is None:
                raise ValueError("Circle debris requires 'radius' parameter")
            self.radius = float(radius)
            if SHAPELY_AVAILABLE:
                self.polygon = Point(self.pos[:2]).buffer(self.radius)

        elif shape == "rectangle":
            if width is None or height_2d is None:
                raise ValueError("Rectangle debris requires 'width' and 'height_2d' parameters")
            self.width     = float(width)
            self.height_2d = float(height_2d)
            self.rotation  = float(rotation)
            if SHAPELY_AVAILABLE:
                # Create rotated rectangle
                self.polygon = self._create_rotated_box()

        elif shape == "polygon":
            if vertices is None or len(vertices) < 3:
                raise ValueError("Polygon debris requires 'vertices' with >= 3 points")
            self.vertices = [np.array(v[:2]) for v in vertices]
            if SHAPELY_AVAILABLE:
                self.polygon = ShapelyPolygon(self.vertices)

        else:
            raise ValueError(f"Unknown debris shape: {shape}")

        logger.debug(
            "Debris %d (%s) at (%.1f, %.1f), h=%.1f",
            self.id, self.shape, self.pos[0], self.pos[1], self.height_3d,
        )

    # ─── Helper: Tạo rotated rectangle ───────────────────────────────────────

    def _create_rotated_box(self) -> "ShapelyPolygon":
        """Tạo hình chữ nhật xoay bằng Shapely."""
        if not SHAPELY_AVAILABLE:
            return None

        from shapely import affinity

        # Tạo box không xoay (centered at origin)
        box = shapely_box(
            -self.width / 2, -self.height_2d / 2,
            self.width / 2, self.height_2d / 2,
        )

        # Xoay quanh origin
        if self.rotation != 0:
            box = affinity.rotate(box, self.rotation, origin=(0, 0))

        # Dịch chuyển đến vị trí thực
        box = affinity.translate(box, self.pos[0], self.pos[1])

        return box

    # ─── Geometry checks ──────────────────────────────────────────────────────

    def in_zone_2d(self, pos_2d: np.ndarray) -> bool:
        """
        Kiểm tra vị trí 2D có trong vùng debris không.

        Args:
            pos_2d: [x, y] hoặc [x, y, z]

        Returns:
            bool: True nếu trong vùng
        """
        point = np.array(pos_2d[:2])

        # ── Circle: Dùng distance check (fast) ──
        if self.shape == "circle":
            return dist_2d(point, self.pos) <= self.radius

        # ── Rectangle/Polygon: Dùng Shapely ──
        # ✅ FIX 2.2: covers() thay vì contains() (include boundary)
        if SHAPELY_AVAILABLE and self.polygon is not None:
            return self.polygon.covers(Point(point))

        # ── Fallback: Sử dụng radius tương đương ──
        fallback_radius = self._get_fallback_radius()
        return dist_2d(point, self.pos) <= fallback_radius

    def causes_collision(self, uav_pos: np.ndarray) -> bool:
        """
        UAV có va chạm với debris không.

        Args:
            uav_pos: [x, y, z]

        Returns:
            bool: True nếu va chạm
        """
        if uav_pos[2] >= self.height_3d:
            return False
        return self.in_zone_2d(uav_pos)

    def blocks_los(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """
        Debris có chắn line-of-sight không.

        Args:
            pos1: [x, y, z] điểm bắt đầu
            pos2: [x, y, z] điểm kết thúc

        Returns:
            bool: True nếu bị chắn
        """
        # [1] 3D altitude check
        if min(pos1[2], pos2[2]) >= self.height_3d:
            return False

        p1 = np.array(pos1[:2])
        p2 = np.array(pos2[:2])

        # [2] XY intersection check
        if self.shape == "circle":
            return _line_intersects_circle(p1, p2, self.pos[:2], self.radius)

        # Rectangle/Polygon: Dùng Shapely
        if SHAPELY_AVAILABLE and self.polygon is not None:
            line = LineString([p1, p2])
            return line.intersects(self.polygon)

        # Fallback
        fallback_radius = self._get_fallback_radius()
        return _line_intersects_circle(p1, p2, self.pos[:2], fallback_radius)

    def get_distance_to_edge(self, pos_2d: np.ndarray) -> float:
        """
        Khoảng cách từ pos đến cạnh gần nhất của debris.

        Args:
            pos_2d: [x, y] hoặc [x, y, z]

        Returns:
            float: khoảng cách (0 nếu đang trong debris)
        """
        point = Point(pos_2d[:2])

        if self.shape == "circle":
            return max(0.0, dist_2d(pos_2d, self.pos) - self.radius)

        if SHAPELY_AVAILABLE and self.polygon is not None:
            if self.polygon.contains(point):
                return 0.0
            near_pt = nearest_points(self.polygon.boundary, point)[0]
            return point.distance(near_pt)

        # Fallback
        fallback_radius = self._get_fallback_radius()
        return max(0.0, dist_2d(pos_2d, self.pos) - fallback_radius)

    # ─── Helper: Fallback radius ──────────────────────────────────────────────

    def _get_fallback_radius(self) -> float:
        """Tính bán kính tương đương cho fallback (khi không có Shapely)."""
        if self.shape == "circle":
            return self.radius
        elif self.shape == "rectangle":
            # Bounding circle radius = sqrt(w²+h²)/2
            return np.sqrt(self.width**2 + self.height_2d**2) / 2
        elif self.shape == "polygon":
            # Max distance từ center đến vertices
            dists = [dist_2d(self.pos, v) for v in self.vertices]
            return max(dists) if dists else 1.0
        return 1.0

    # ─── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Chuyển thành dict JSON-safe."""
        data = {
            "id":        int(self.id),
            "type":      "debris",
            "pos":       self.pos.tolist(),
            "height_3d": float(self.height_3d),
            "penalty":   float(self.penalty),
            "shape":     self.shape,
        }

        if self.shape == "circle":
            data["radius"] = float(self.radius)
        elif self.shape == "rectangle":
            data["width"]     = float(self.width)
            data["height_2d"] = float(self.height_2d)
            data["rotation"]  = float(self.rotation)
        elif self.shape == "polygon":
            data["vertices"] = [v.tolist() for v in self.vertices]

        return data

    def __repr__(self) -> str:
        if self.shape == "circle":
            return (
                f"Debris(id={self.id}, circle, "
                f"pos=({self.pos[0]:.1f}, {self.pos[1]:.1f}), "
                f"r={self.radius:.1f}, h={self.height_3d:.1f})"
            )
        elif self.shape == "rectangle":
            return (
                f"Debris(id={self.id}, rect, "
                f"pos=({self.pos[0]:.1f}, {self.pos[1]:.1f}), "
                f"w={self.width:.1f}, h={self.height_2d:.1f}, "
                f"rot={self.rotation:.0f}°, h3d={self.height_3d:.1f})"
            )
        else:
            return (
                f"Debris(id={self.id}, polygon, "
                f"pos=({self.pos[0]:.1f}, {self.pos[1]:.1f}), "
                f"{len(self.vertices)} vertices, h={self.height_3d:.1f})"
            )


# ═══════════════════════════════════════════════════════════════════════════
# DANGERZONE - Giữ nguyên, chỉ thêm shape support (tương tự Debris)
# ═══════════════════════════════════════════════════════════════════════════
class DangerZone:
    """
    Vùng nguy hiểm - HỖ TRỢ NHIỀU HÌNH DẠNG.

    Shape params giống Debris.
    """

    def __init__(
        self,
        zone_id:     int,
        pos:         List[float],
        danger_type: str,
        cfg:         "AppConfig",
        # Shape params
        shape:       str = "circle",
        radius:      Optional[float] = None,
        width:       Optional[float] = None,
        height_2d:   Optional[float] = None,
        rotation:    float = 0.0,
        vertices:    Optional[List[List[float]]] = None,
    ) -> None:
        self.id          = zone_id
        self.pos         = np.array([pos[0], pos[1], 0.0], dtype=np.float64)
        self.danger_type = danger_type
        self.cfg         = cfg

        self.max_height = cfg.danger.heights.get(danger_type, 15.0)
        self.penalty    = cfg.danger.penalties.get(danger_type, -12.0)

        self.shape      = shape
        self.radius     = None
        self.width      = None
        self.height_2d  = None
        self.rotation   = rotation
        self.vertices   = None
        self.polygon    = None

        # ── Validate shape (giống Debris) ──
        if shape == "circle":
            if radius is None:
                raise ValueError("Circle zone requires 'radius'")
            self.radius = float(radius)
            if SHAPELY_AVAILABLE:
                self.polygon = Point(self.pos[:2]).buffer(self.radius)

        elif shape == "rectangle":
            if width is None or height_2d is None:
                raise ValueError("Rectangle zone requires 'width' and 'height_2d'")
            self.width     = float(width)
            self.height_2d = float(height_2d)
            self.rotation  = float(rotation)
            if SHAPELY_AVAILABLE:
                self.polygon = self._create_rotated_box()

        elif shape == "polygon":
            if vertices is None or len(vertices) < 3:
                raise ValueError("Polygon zone requires >= 3 vertices")
            self.vertices = [np.array(v[:2]) for v in vertices]
            if SHAPELY_AVAILABLE:
                self.polygon = ShapelyPolygon(self.vertices)

        else:
            raise ValueError(f"Unknown zone shape: {shape}")

    # ── Copy helper methods từ Debris ──
    def _create_rotated_box(self):
        """Tạo rotated rectangle (same as Debris)."""
        if not SHAPELY_AVAILABLE:
            return None
        from shapely import affinity
        box = shapely_box(
            -self.width / 2, -self.height_2d / 2,
            self.width / 2, self.height_2d / 2,
        )
        if self.rotation != 0:
            box = affinity.rotate(box, self.rotation, origin=(0, 0))
        box = affinity.translate(box, self.pos[0], self.pos[1])
        return box

    def _get_fallback_radius(self) -> float:
        """Fallback radius (same as Debris)."""
        if self.shape == "circle":
            return self.radius
        elif self.shape == "rectangle":
            return np.sqrt(self.width**2 + self.height_2d**2) / 2
        elif self.shape == "polygon":
            dists = [dist_2d(self.pos, v) for v in self.vertices]
            return max(dists) if dists else 1.0
        return 1.0

    # ── Geometry methods (giống Debris) ──
    def is_inside(self, uav_pos: np.ndarray) -> bool:
        """Check UAV trong zone không."""
        if uav_pos[2] >= self.max_height:
            return False

        point = np.array(uav_pos[:2])

        if self.shape == "circle":
            return dist_2d(point, self.pos) <= self.radius

        # ✅ FIX 2.2: covers() thay vì contains()
        if SHAPELY_AVAILABLE and self.polygon is not None:
            return self.polygon.covers(Point(point))

        fallback_radius = self._get_fallback_radius()
        return dist_2d(point, self.pos) <= fallback_radius
    
    def blocks_los(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """Check chắn LOS không."""
        if self.danger_type not in _LOS_BLOCKING_TYPES:
            return False

        if min(pos1[2], pos2[2]) >= self.max_height:
            return False

        p1 = np.array(pos1[:2])
        p2 = np.array(pos2[:2])

        if self.shape == "circle":
            return _line_intersects_circle(p1, p2, self.pos[:2], self.radius)

        if SHAPELY_AVAILABLE and self.polygon is not None:
            line = LineString([p1, p2])
            return line.intersects(self.polygon)

        fallback_radius = self._get_fallback_radius()
        return _line_intersects_circle(p1, p2, self.pos[:2], fallback_radius)

    # ── Sensor/Battery modifiers (giữ nguyên) ──
    def get_sensor_modifier(self) -> float:
        return 0.4 if self.danger_type == "smoke" else 1.0

    def get_battery_modifier(self) -> float:
        return 0.05 if self.danger_type == "fire" else 0.0

    # ── Serialization ──
    def to_dict(self) -> dict:
        data = {
            "id":          int(self.id),
            "type":        "danger_zone",
            "danger_type": self.danger_type,
            "pos":         self.pos.tolist(),
            "max_height":  None if np.isinf(self.max_height) else float(self.max_height),
            "penalty":     float(self.penalty),
            "shape":       self.shape,
        }

        if self.shape == "circle":
            data["radius"] = float(self.radius)
        elif self.shape == "rectangle":
            data["width"]     = float(self.width)
            data["height_2d"] = float(self.height_2d)
            data["rotation"]  = float(self.rotation)
        elif self.shape == "polygon":
            data["vertices"] = [v.tolist() for v in self.vertices]

        return data

    def __repr__(self) -> str:
        max_h = "inf" if np.isinf(self.max_height) else f"{self.max_height:.1f}"
        if self.shape == "circle":
            return (
                f"DangerZone(id={self.id}, {self.danger_type}, circle, "
                f"r={self.radius:.1f}, max_h={max_h})"
            )
        elif self.shape == "rectangle":
            return (
                f"DangerZone(id={self.id}, {self.danger_type}, rect, "
                f"w={self.width:.1f}×h={self.height_2d:.1f}, "
                f"rot={self.rotation:.0f}°, max_h={max_h})"
            )
        else:
            return (
                f"DangerZone(id={self.id}, {self.danger_type}, polygon, "
                f"{len(self.vertices)} vertices, max_h={max_h})"
            )