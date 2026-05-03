from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

"""
core/map_generator.py
Map Generator - v4.1 (FIXED)

FIXES v4.1 (so với v4.0):
    ✅ FIX-P1A: Danger circle radius = width/2 (was using width as radius → 2× too large)
    ✅ FIX-P1B: Debris circle radius = width/2 (same mismatch)
    ✅ FIX-P1C: Debris rectangle dùng đúng config w_range (was hardcoded 4-12m)
    ✅ FIX-P1D: Debris polygon avg_radius = width/2 (was 4-8m regardless of config)
    ✅ FIX-P1E: max_place_attempts tăng → config/env.py (500 thay vì 200)
    ✅ FIX-P10: generate() tự sinh uav_spawns (không trả empty list)
    ✅ FIX-P11: get_deployable_uavs() deploy top battery, reserve là phần còn lại
               (đã move sang fleet_manager.py - xem file đó)
    ✅ FIX-P12: Xóa sys.path hack (đã xóa)
"""

if TYPE_CHECKING:
    from config import AppConfig

logger = logging.getLogger(__name__)

# ─── Shapely import (lazy) ────────────────────────────────────────────────────
try:
    from shapely.geometry import Point, Polygon as ShapelyPolygon
    from shapely.geometry import box as shapely_box
    from shapely import affinity
    from shapely.ops import unary_union
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    Point          = None
    ShapelyPolygon = None
    shapely_box    = None
    affinity       = None
    unary_union    = None
    logger.warning("Shapely not available, using fallback geometry (less accurate)")


def _sq_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Squared distance."""
    d = a - b
    return float(d @ d)


class MapGenerator:
    """
    Map Generator v4.1 - Fixed size mismatch + self-contained map data.

    KEY FIX (v4.1):
        Config field "widths" → đường kính (diameter), KHÔNG phải radius.
        Generator phải dùng radius = width / 2.0

        BEFORE (WRONG):
            radius = rng.uniform(4.0, 8.0)   # disaster: diameter = 8-16m
        AFTER (CORRECT):
            width  = rng.uniform(w_min, w_max)
            radius = width / 2.0              # diameter = w_min to w_max ✅
    """

    # ── Constants ─────────────────────────────────────────────────────────────
    _VICTIM_BOUNDARY_MARGIN:   float = 2.0
    _OBSTACLE_MIN_STATION:     float = 3.0
    # _MIN_OBSTACLE_SPACING:     float = 2.5
    _ROTATION_ANGLES:          list  = [0.0, 90.0, 180.0, 270.0]
    # _SPACING_RELAX_THRESHOLD:  float = 0.7   # Relax sau 70% attempts
    # _SPACING_RELAXED:          float = 1.5   # Relaxed spacing (mét)

    def __init__(self, cfg: "AppConfig") -> None:
        self.cfg = cfg
        self._shapely_cache: Dict[int, "ShapelyPolygon"] = {}

    # ═════════════════════════════════════════════════════════════════════════
    # STATIONS
    # ═════════════════════════════════════════════════════════════════════════

    def _place_stations(self, rng: np.random.Generator) -> List[Dict]:
        """Đặt charging stations với spacing validation."""
        cfg          = self.cfg
        map_size     = cfg.env.map_size
        n_stations   = cfg.env.n_stations
        min_spacing  = cfg.env.min_station_spacing
        min_boundary = cfg.env.station_min_boundary_dist
        max_attempts = cfg.env.max_place_attempts
        station_cap  = cfg.env.station_capacity

        stations:   List[Dict]         = []
        pos_arrays: List[np.ndarray]   = []

        for i in range(n_stations):
            placed = False

            for _ in range(max_attempts):
                x   = float(rng.uniform(min_boundary, map_size - min_boundary))
                y   = float(rng.uniform(min_boundary, map_size - min_boundary))
                pos = np.array([x, y], dtype=np.float64)

                min_sq = min_spacing ** 2
                valid  = all(_sq_dist(pos, p) >= min_sq for p in pos_arrays)

                if valid:
                    stations.append({"id": i, "pos": [x, y], "capacity": station_cap})
                    pos_arrays.append(pos)
                    placed = True
                    break

            if not placed:
                # Fallback: 4 góc map
                m = map_size * 0.1
                fallbacks = [
                    [m, m],
                    [map_size - m, map_size - m],
                    [m, map_size - m],
                    [map_size - m, m],
                ]
                pos_list = fallbacks[i % len(fallbacks)]
                logger.warning("Station %d: forced placement at %s", i, pos_list)
                stations.append({"id": i, "pos": pos_list, "capacity": station_cap})
                pos_arrays.append(np.array(pos_list, dtype=np.float64))

        return stations

    # ═════════════════════════════════════════════════════════════════════════
    # DEBRIS — FIX P1B, P1C, P1D
    # ═════════════════════════════════════════════════════════════════════════

    def _place_debris(
        self,
        stations: List[Dict],
        rng:      np.random.Generator,
    ) -> List[Dict]:
        """
        Đặt debris với progressive spacing relaxation.

        ✅ FIX-P1B: Circle radius = width / 2.0
        ✅ FIX-P1C: Rectangle dùng đúng config w_range
        ✅ FIX-P1D: Polygon avg_radius = width / 2.0
        """
        cfg          = self.cfg
        map_size     = cfg.env.map_size
        n_debris     = cfg.obstacle.n_debris
        max_attempts = cfg.env.max_place_attempts

        # Config "width" = footprint diameter
        w_range = (cfg.obstacle.debris_width_min, cfg.obstacle.debris_width_max)
        h_range = (cfg.obstacle.debris_height_min, cfg.obstacle.debris_height_max)

        station_pos = [np.array(s["pos"], dtype=np.float64) for s in stations]
        debris_list: List[Dict] = []

        for i in range(n_debris):
            placed = False

            for attempt in range(max_attempts):
                # ── Random shape (weighted 40/40/20) ──────────────────────
                shape_roll = rng.random()
                if shape_roll < 0.4:
                    shape = "circle"
                elif shape_roll < 0.8:
                    shape = "rectangle"
                else:
                    shape = "polygon"

                # ── Random position ───────────────────────────────────────
                x   = float(rng.uniform(5, map_size - 5))
                y   = float(rng.uniform(5, map_size - 5))
                pos = np.array([x, y], dtype=np.float64)

                height_3d = float(rng.uniform(*h_range))

                # ── Shape params ──────────────────────────────────────────
                debris_dict: Dict = {
                    "id":        i,
                    "pos":       [x, y],
                    "height_3d": height_3d,
                    "type":      "debris",
                    "shape":     shape,
                }

                if shape == "circle":
                    # ✅ FIX-P1B: width = diameter → radius = width/2
                    width  = float(rng.uniform(*w_range))
                    radius = width / 2.0
                    debris_dict["radius"] = radius

                elif shape == "rectangle":
                    # ✅ FIX-P1C: dùng đúng config w_range (không hardcode 4-12m)
                    width     = float(rng.uniform(*w_range))
                    height_2d = float(rng.uniform(*w_range))
                    rotation  = float(rng.choice(self._ROTATION_ANGLES))
                    debris_dict["width"]     = width
                    debris_dict["height_2d"] = height_2d
                    debris_dict["rotation"]  = rotation

                elif shape == "polygon":
                    # ✅ FIX-P1D: avg_radius = width/2 (không hardcode 4-8m)
                    poly_width = float(rng.uniform(*w_range))
                    avg_radius = poly_width / 2.0
                    vertices   = self._generate_random_convex_polygon(
                        center       = pos,
                        avg_radius   = avg_radius,
                        irregularity = 0.3,
                        spikiness    = 0.2,
                        n_vertices   = int(rng.integers(4, 7)),
                        rng          = rng,
                    )
                    debris_dict["vertices"] = vertices

                # ── Validation ───────────────────────────────────────────
                if not self._check_station_clearance(debris_dict, station_pos):
                    continue

                # ✅ FIX 3.1: Dùng cfg thay vì constant
                relax_threshold = int(max_attempts * cfg.env.placement_relax_threshold)
                use_relaxed     = attempt >= relax_threshold

                if not self._check_obstacle_spacing(
                    debris_dict, debris_list, relaxed=use_relaxed
                ):
                    continue

                # ── Success ──────────────────────────────────────────────
                debris_list.append(debris_dict)
                placed = True
                if use_relaxed:
                    logger.debug("Debris %d placed with relaxed spacing (attempt %d)", i, attempt)
                break

            if not placed:
                logger.warning("Debris %d: skipped (no valid position after %d attempts)", i, max_attempts)

        return debris_list

    # ═════════════════════════════════════════════════════════════════════════
    # DANGER ZONES — FIX P1A
    # ═════════════════════════════════════════════════════════════════════════

    def _place_danger_zones(
        self,
        existing_objects: List[Dict],
        rng:              np.random.Generator,
    ) -> List[Dict]:
        """
        Đặt danger zones với progressive spacing relaxation.

        ✅ FIX-P1A: Circle radius = width / 2.0
                    Config "widths" = diameter range, KHÔNG phải radius range.

        BEFORE (wrong): radius = rng.uniform(15, 25)  → radiation diameter 30-50m (!!!)
        AFTER (correct): width = rng.uniform(15, 25)
                         radius = width / 2.0          → diameter 15-25m ✅
        """
        cfg          = self.cfg
        map_size     = cfg.env.map_size
        n_danger     = cfg.obstacle.n_danger_total
        max_attempts = cfg.env.max_place_attempts

        danger_types  = list(cfg.danger.heights.keys())
        max_counts    = cfg.danger.max_counts
        danger_widths = cfg.danger.widths   # ← đây là DIAMETER range

        danger_zones: List[Dict]    = []
        type_counts:  Dict[str,int] = {t: 0 for t in danger_types}

        for i in range(n_danger):
            placed = False

            for attempt in range(max_attempts):
                # ── Available types ───────────────────────────────────────
                available = [
                    t for t in danger_types
                    if type_counts[t] < max_counts.get(t, 99)
                ]
                if not available:
                    logger.debug("All danger type counts exhausted at zone %d", i)
                    break

                dtype = available[int(rng.integers(0, len(available)))]

                # ── Random shape (50/50) ──────────────────────────────────
                shape = "circle" if rng.random() < 0.5 else "rectangle"

                # ── Random position ───────────────────────────────────────
                x   = float(rng.uniform(10, map_size - 10))
                y   = float(rng.uniform(10, map_size - 10))

                zone_dict: Dict = {
                    "id":          i,
                    "pos":         [x, y],
                    "danger_type": dtype,
                    "type":        "danger_zone",
                    "shape":       shape,
                }

                w_min, w_max = danger_widths[dtype]   # ← DIAMETER range

                if shape == "circle":
                    # ✅ FIX-P1A: width = diameter → radius = width/2
                    width  = float(rng.uniform(w_min, w_max))
                    radius = width / 2.0
                    zone_dict["radius"] = radius

                elif shape == "rectangle":
                    # Rectangle: width/height_2d là footprint → dùng trực tiếp
                    width     = float(rng.uniform(w_min, w_max))
                    height_2d = float(rng.uniform(w_min, w_max))
                    rotation  = float(rng.choice(self._ROTATION_ANGLES))
                    zone_dict["width"]     = width
                    zone_dict["height_2d"] = height_2d
                    zone_dict["rotation"]  = rotation

                # ── Validation ───────────────────────────────────────────
                # ✅ FIX 3.1: Dùng cfg.env thay vì self._SPACING_RELAX_THRESHOLD
                relax_threshold = int(max_attempts * cfg.env.placement_relax_threshold)
                use_relaxed     = attempt >= relax_threshold

                if not self._check_obstacle_spacing(
                    zone_dict,
                    existing_objects + danger_zones,
                    relaxed=use_relaxed,
                ):
                    continue

                # ── Success ──────────────────────────────────────────────
                danger_zones.append(zone_dict)
                type_counts[dtype] += 1
                placed = True
                if use_relaxed:
                    logger.debug(
                        "DangerZone %d (%s) placed with relaxed spacing (attempt %d)",
                        i, dtype, attempt
                    )
                break
            if self.cfg.env.warn_on_skipped_objects:
                if not placed:
                    logger.warning(
                        "Danger zone %d: skipped after %d attempts", i, max_attempts
                    )

        return danger_zones

    # ═════════════════════════════════════════════════════════════════════════
    # POLYGON GENERATION
    # ═════════════════════════════════════════════════════════════════════════

    def _generate_random_convex_polygon(
        self,
        center:       np.ndarray,
        avg_radius:   float,
        irregularity: float,
        spikiness:    float,
        n_vertices:   int,
        rng:          np.random.Generator,
    ) -> List[List[float]]:
        """Generate random convex polygon."""
        irregularity = float(np.clip(irregularity, 0.0, 1.0))
        spikiness    = float(np.clip(spikiness,    0.0, 1.0))

        angle_steps = 2 * np.pi / n_vertices
        lower_bound = (1 - irregularity) * angle_steps
        upper_bound = (1 + irregularity) * angle_steps

        angles = []
        cumsum = 0.0
        for _ in range(n_vertices):
            angle   = float(rng.uniform(lower_bound, upper_bound))
            cumsum += angle
            angles.append(cumsum)

        # Normalize → [0, 2π]
        if cumsum > 0:
            angles = [a * 2 * np.pi / cumsum for a in angles]

        vertices = []
        for angle in angles:
            radius_noise = float(rng.uniform(0, spikiness))
            r            = avg_radius * (1.0 - radius_noise)
            x            = center[0] + r * np.cos(angle)
            y            = center[1] + r * np.sin(angle)
            vertices.append([float(x), float(y)])

        return vertices

    # ═════════════════════════════════════════════════════════════════════════
    # VALIDATION HELPERS
    # ═════════════════════════════════════════════════════════════════════════

    def _check_station_clearance(
        self,
        obstacle:    Dict,
        station_pos: List[np.ndarray],
    ) -> bool:
        """Obstacle đủ xa stations không."""
        min_dist   = self._OBSTACLE_MIN_STATION
        obs_pos    = np.array(obstacle["pos"][:2], dtype=np.float64)
        obs_radius = self._get_bounding_radius(obstacle)

        for st_pos in station_pos:
            if _sq_dist(obs_pos, st_pos) < (min_dist + obs_radius) ** 2:
                return False
        return True

    def _check_obstacle_spacing(
        self,
        new_obstacle:       Dict,
        existing_obstacles: List[Dict],
        relaxed:            bool = False,
    ) -> bool:
        """Check spacing giữa obstacles."""
        # ✅ FIX 3.1+3.2: Dùng config spacing
        min_spacing = (
            self.cfg.env.placement_relaxed_spacing_m if relaxed
            else self.cfg.env.min_object_spacing_m
        )

        if SHAPELY_AVAILABLE:
            new_poly = self._get_or_create_polygon(new_obstacle)
            if new_poly is None:
                return self._check_spacing_fallback(
                    new_obstacle, existing_obstacles, min_spacing
                )
            for obs in existing_obstacles:
                obs_poly = self._get_or_create_polygon(obs)
                if obs_poly is None:
                    continue
                if new_poly.distance(obs_poly) < min_spacing:
                    return False
            return True

        return self._check_spacing_fallback(
            new_obstacle, existing_obstacles, min_spacing
        )

    def _check_spacing_fallback(
        self,
        new_obstacle:       Dict,
        existing_obstacles: List[Dict],
        min_spacing:        float,
    ) -> bool:
        """Fallback spacing check dùng bounding circles."""
        new_pos = np.array(new_obstacle["pos"][:2], dtype=np.float64)
        new_r   = self._get_bounding_radius(new_obstacle)

        for obs in existing_obstacles:
            obs_pos = np.array(obs["pos"][:2], dtype=np.float64)
            obs_r   = self._get_bounding_radius(obs)
            if _sq_dist(new_pos, obs_pos) < (new_r + obs_r + min_spacing) ** 2:
                return False
        return True

    # ═════════════════════════════════════════════════════════════════════════
    # GEOMETRY HELPERS
    # ═════════════════════════════════════════════════════════════════════════

    def _get_bounding_radius(self, obstacle: Dict) -> float:
        """Bounding radius cho obstacle dict."""
        shape = obstacle.get("shape", "circle")

        if shape == "circle":
            return float(obstacle.get("radius", 3.0))

        elif shape == "rectangle":
            w = obstacle.get("width",     5.0)
            h = obstacle.get("height_2d", 5.0)
            return float(np.sqrt(w**2 + h**2) / 2.0)

        elif shape == "polygon":
            vertices = obstacle.get("vertices", [])
            if not vertices:
                return 3.0
            pos   = np.array(obstacle["pos"][:2], dtype=np.float64)
            dists = [np.linalg.norm(np.array(v[:2]) - pos) for v in vertices]
            return float(max(dists)) if dists else 3.0

        return 3.0

    def _get_or_create_polygon(self, obstacle: Dict) -> Optional["ShapelyPolygon"]:
        """Cache Shapely polygons."""
        if not SHAPELY_AVAILABLE:
            return None

        obs_id = id(obstacle)
        if obs_id in self._shapely_cache:
            return self._shapely_cache[obs_id]

        poly = self._create_shapely_polygon(obstacle)
        if poly is not None:
            self._shapely_cache[obs_id] = poly
        return poly

    def _create_shapely_polygon(self, obstacle: Dict) -> Optional["ShapelyPolygon"]:
        """Create Shapely polygon từ obstacle dict."""
        if not SHAPELY_AVAILABLE:
            return None

        shape = obstacle.get("shape", "circle")
        pos   = obstacle["pos"][:2]

        try:
            if shape == "circle":
                radius = obstacle.get("radius", 3.0)
                return Point(pos).buffer(radius)

            elif shape == "rectangle":
                width     = obstacle.get("width",     5.0)
                height_2d = obstacle.get("height_2d", 5.0)
                rotation  = obstacle.get("rotation",  0.0)
                box       = shapely_box(
                    -width / 2, -height_2d / 2,
                     width / 2,  height_2d / 2,
                )
                if rotation != 0:
                    box = affinity.rotate(box, rotation, origin=(0, 0))
                return affinity.translate(box, pos[0], pos[1])

            elif shape == "polygon":
                vertices = obstacle.get("vertices", [])
                if len(vertices) < 3:
                    return None
                return ShapelyPolygon(vertices)

        except Exception as e:
            logger.warning("Failed to create Shapely polygon: %s", e)
            return None

        return None

    # ═════════════════════════════════════════════════════════════════════════
    # VICTIMS
    # ═════════════════════════════════════════════════════════════════════════

    def _find_valid_victim_pos(
        self,
        obs_pos:          List[np.ndarray],
        obs_r:            List[float],
        danger_zones:     List[Dict],
        existing_victims: List[Dict],
        debris_list:      List[Dict],
        near_distance:    float,
        rng:              np.random.Generator,
        max_attempts:     int = 100,
    ) -> Optional[List[float]]:
        """Find valid victim position (tránh obstacles + danger zones)."""
        cfg              = self.cfg
        victim_clearance = cfg.env.victim_clearance
        victim_min_dist  = cfg.env.victim_min_dist
        boundary_margin  = self._VICTIM_BOUNDARY_MARGIN
        map_size         = cfg.env.map_size

        victim_pos_cache = [np.array(v["pos"], dtype=np.float64) for v in existing_victims]
        victim_min_sq    = victim_min_dist ** 2

        danger_pos = [np.array(dz["pos"][:2], dtype=np.float64) for dz in danger_zones]
        danger_r   = [self._get_bounding_radius(dz)              for dz in danger_zones]

        lo = boundary_margin
        hi = map_size - boundary_margin

        for _ in range(max_attempts):
            if debris_list:
                debris = debris_list[int(rng.integers(0, len(debris_list)))]
                d_pos  = np.array(debris["pos"], dtype=np.float64)
                d_r    = self._get_bounding_radius(debris)
                min_d  = d_r + victim_clearance
                max_d  = near_distance

                if min_d >= max_d:
                    x = float(rng.uniform(lo, hi))
                    y = float(rng.uniform(lo, hi))
                else:
                    angle = float(rng.uniform(0, 2 * np.pi))
                    dist  = float(rng.uniform(min_d, max_d))
                    x     = d_pos[0] + dist * np.cos(angle)
                    y     = d_pos[1] + dist * np.sin(angle)
            else:
                x = float(rng.uniform(lo, hi))
                y = float(rng.uniform(lo, hi))

            if not (lo <= x <= hi and lo <= y <= hi):
                continue

            pos = np.array([x, y], dtype=np.float64)

            # Check obstacles
            if not all(
                _sq_dist(pos, op) >= (r + victim_clearance) ** 2
                for op, r in zip(obs_pos, obs_r)
            ):
                continue

            # Check danger zones
            if not all(
                _sq_dist(pos, dp) >= (dr + victim_clearance * 2) ** 2
                for dp, dr in zip(danger_pos, danger_r)
            ):
                continue

            # Check other victims
            if not all(
                _sq_dist(pos, vp) >= victim_min_sq
                for vp in victim_pos_cache
            ):
                continue

            return [float(x), float(y)]

        return None

    def _spawn_group(
        self,
        n:             int,
        victim_type:   str,
        urgency_range: Tuple[float, float],
        near_ratio:    float,
        debris_list:   List[Dict],
        obs_pos:       List[np.ndarray],
        obs_r:         List[float],
        danger_zones:  List[Dict],
        victims:       List[Dict],
        rng:           np.random.Generator,
    ) -> None:
        """Spawn group of victims."""
        cfg    = self.cfg
        n_near = int(n * near_ratio)
        n_rand = n - n_near

        def _try_spawn(use_near: bool) -> None:
            pos = self._find_valid_victim_pos(
                obs_pos=obs_pos, obs_r=obs_r,
                danger_zones=danger_zones,
                existing_victims=victims,
                debris_list=debris_list if use_near else [],
                near_distance=cfg.env.victim_near_dist,
                rng=rng,
            )
            if pos is None and use_near:
                # Fallback: random placement
                pos = self._find_valid_victim_pos(
                    obs_pos=obs_pos, obs_r=obs_r,
                    danger_zones=danger_zones,
                    existing_victims=victims,
                    debris_list=[],
                    near_distance=cfg.env.victim_near_dist,
                    rng=rng,
                )
            if pos is not None:
                victims.append({
                    "id":          len(victims),
                    "pos":         pos,
                    "urgency":     float(rng.uniform(*urgency_range)),
                    "victim_type": victim_type,
                })

        for _ in range(n_near): _try_spawn(use_near=True)
        for _ in range(n_rand): _try_spawn(use_near=False)

    def _spawn_victims(
        self,
        n_victims:    int,
        obstacles:    List[Dict],
        danger_zones: List[Dict],
        rng:          np.random.Generator,
    ) -> List[Dict]:
        """Spawn all victims."""
        cfg = self.cfg

        injured_ratio = float(rng.uniform(
            cfg.victim.injured_ratio_min, cfg.victim.injured_ratio_max
        ))
        n_injured = int(np.round(n_victims * injured_ratio))
        n_mobile  = n_victims - n_injured

        obs_pos     = [np.array(o["pos"], dtype=np.float64) for o in obstacles]
        obs_r       = [self._get_bounding_radius(o)          for o in obstacles]
        debris_list = [o for o in obstacles if o.get("type") == "debris"]

        victims: List[Dict] = []

        self._spawn_group(
            n=n_injured, victim_type="injured",
            urgency_range=(cfg.victim.injured_urgency_min, cfg.victim.injured_urgency_max),
            near_ratio=0.8,
            debris_list=debris_list, obs_pos=obs_pos, obs_r=obs_r,
            danger_zones=danger_zones, victims=victims, rng=rng,
        )
        self._spawn_group(
            n=n_mobile, victim_type="mobile",
            urgency_range=(cfg.victim.mobile_urgency_min, cfg.victim.mobile_urgency_max),
            near_ratio=0.4,
            debris_list=debris_list, obs_pos=obs_pos, obs_r=obs_r,
            danger_zones=danger_zones, victims=victims, rng=rng,
        )

        return victims

    # ═════════════════════════════════════════════════════════════════════════
    # MAIN GENERATE — FIX P10: tự sinh uav_spawns
    # ═════════════════════════════════════════════════════════════════════════

    def generate(
        self,
        n_victims_override: Optional[int] = None,
        seed:               Optional[int] = None,
    ) -> Dict:
        """
        Generate complete map data.

        ✅ FIX-P10: uav_spawns được sinh trong generate() (self-contained)
                    Không còn return empty list.

        Returns:
            dict: {stations, debris, danger_zones, victims, uav_spawns, seed, n_victims}
        """
        cfg = self.cfg
        rng = np.random.default_rng(seed)

        # Clear cache mỗi lần generate
        self._shapely_cache.clear()

        stations     = self._place_stations(rng)
        debris       = self._place_debris(stations, rng)
        danger_zones = self._place_danger_zones(debris + stations, rng)

        n_victims = (
            int(rng.integers(cfg.victim.n_victims_min, cfg.victim.n_victims_max + 1))
            if n_victims_override is None
            else int(n_victims_override)
        )

        victims = self._spawn_victims(
            n_victims=n_victims,
            obstacles=debris + danger_zones,
            danger_zones=danger_zones,
            rng=rng,
        )

        # ✅ FIX-P10: Sinh uav_spawns ngay trong generate()
        uav_spawns = self.get_uav_spawns(
            stations=stations,
            n_total=cfg.env.n_uav,
            rng=rng,
        )

        # Log placement stats
        logger.info(
            "Map generated (seed=%s): stations=%d, debris=%d/%d, "
            "danger=%d/%d, victims=%d/%d, spawns=%d",
            seed,
            len(stations),
            len(debris),     cfg.obstacle.n_debris,
            len(danger_zones), cfg.obstacle.n_danger_total,
            len(victims),    n_victims,
            len(uav_spawns),
        )

        return {
            "stations":     stations,
            "debris":       debris,
            "danger_zones": danger_zones,
            "victims":      victims,
            "uav_spawns":   uav_spawns,   # ✅ FIX-P10: không còn empty
            "seed":         seed,
            "n_victims":    len(victims),
        }

    # ═════════════════════════════════════════════════════════════════════════
    # UAV SPAWNS
    # ═════════════════════════════════════════════════════════════════════════

    def get_uav_spawns(
        self,
        stations: List[Dict],
        n_total:  int,
        rng:      Optional[np.random.Generator] = None,
    ) -> List[Dict]:
        """Generate UAV spawn positions quanh stations."""
        cfg = self.cfg

        if rng is None:
            rng = np.random.default_rng()

        spawns        = []
        n_stations    = len(stations)
        n_per_station = n_total // n_stations
        remainder     = n_total % n_stations
        uav_id        = 0
        spawn_radius  = cfg.env.uav_spawn_radius
        map_size      = cfg.env.map_size
        z_min         = cfg.uav.z_min

        for i, station in enumerate(stations):
            n_spawn     = n_per_station + (1 if i < remainder else 0)
            station_pos = np.array(station["pos"], dtype=np.float64)

            for _ in range(n_spawn):
                offset_x = float(rng.uniform(-spawn_radius, spawn_radius))
                offset_y = float(rng.uniform(-spawn_radius, spawn_radius))

                x = float(np.clip(station_pos[0] + offset_x, 0.0, map_size))
                y = float(np.clip(station_pos[1] + offset_y, 0.0, map_size))
                z = float(z_min)

                spawns.append({"id": uav_id, "pos": [x, y, z]})
                uav_id += 1

        return spawns

    # ═════════════════════════════════════════════════════════════════════════
    # MAP STATISTICS
    # ═════════════════════════════════════════════════════════════════════════

    def get_map_statistics(self, map_data: Dict) -> Dict:
        """Rich map statistics cho curriculum + analysis."""
        cfg      = self.cfg
        map_area = cfg.env.map_size ** 2

        n_obstacles = len(map_data["debris"]) + len(map_data["danger_zones"])
        n_victims   = len(map_data["victims"])

        obstacle_density = n_obstacles / (map_area / 1000.0)
        victim_density   = n_victims   / (map_area / 1000.0)

        # Danger zone coverage area
        if SHAPELY_AVAILABLE:
            danger_polygons = [
                self._get_or_create_polygon(dz)
                for dz in map_data["danger_zones"]
            ]
            danger_polygons = [p for p in danger_polygons if p is not None]
            danger_area = unary_union(danger_polygons).area if danger_polygons else 0.0
        else:
            danger_area = sum(
                np.pi * self._get_bounding_radius(dz) ** 2
                for dz in map_data["danger_zones"]
            )

        danger_coverage = (danger_area / map_area) * 100.0

        avg_urgency = (
            float(np.mean([v["urgency"] for v in map_data["victims"]]))
            if n_victims > 0 else 0.0
        )

        # Spatial clustering
        if n_victims >= 2:
            positions          = np.array([v["pos"] for v in map_data["victims"]])
            center             = positions.mean(axis=0)
            avg_dist_to_center = np.mean(np.linalg.norm(positions - center, axis=1))
            clustering         = float(
                np.clip(1.0 - avg_dist_to_center / (cfg.env.map_size / 2), 0.0, 1.0)
            )
        else:
            clustering = 0.0

        return {
            "map_area_m2":         float(map_area),
            "n_obstacles":         int(n_obstacles),
            "n_victims":           int(n_victims),
            "obstacle_density":    float(obstacle_density),
            "victim_density":      float(victim_density),
            "danger_coverage_pct": float(danger_coverage),
            "avg_victim_urgency":  float(avg_urgency),
            "spatial_clustering":  clustering,
        }