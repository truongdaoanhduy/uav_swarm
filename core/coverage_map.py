from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from utils.geometry import get_circle_cells

"""
core/coverage_map.py
Coverage Map - Temporal-aware cho SAR UAV Swarm (FIXED v2.0)

FIXES applied (theo review):
    ✅ FIX-01: Xóa hẳn sys.path hack
    ✅ FIX-02: Timestamps không overwrite → preserve temporal gradient
    ✅ FIX-03: get_staleness() return max_steps nếu unexplored (không phải 0)
    ✅ FIX-04: normalize staleness theo decay_threshold (không phải max_steps)
    ✅ FIX-05: get_rescan_count return float (không làm tròn int)
    ✅ FIX-06: grid dtype → bool (semantic hơn int8)
    ✅ FIX-07: Comment rõ O(N) queries → future optimization needed
    ✅ FIX-08: Add first_scan tracking (optional, cho advanced analysis)

Chức năng:
    - Tracking explored areas (binary grid)
    - Timestamps cho re-scan logic (temporal info)
    - Scan count cho hotspot analysis
    - Staleness computation cho POMDP observation

Complexity notes:
    - mark_explored: O(cells) - vectorized, fast
    - get_nearest_*: O(N) full scan - OK for 100x100, bottleneck at 500x500
      → TODO Phase 4: Implement frontier set / KD-tree
"""

if TYPE_CHECKING:
    from config import AppConfig

logger = logging.getLogger(__name__)


class CoverageMap:
    """
    Coverage map với temporal information (FIXED v2.0).

    Grid structure:
        grid:        [grid_size, grid_size] bool   - đã explore hay chưa
        timestamps:  [grid_size, grid_size] int32  - step cuối cùng scan
        first_scan:  [grid_size, grid_size] int32  - step đầu tiên scan (NEW)
        scan_count:  [grid_size, grid_size] int32  - số lần scan mỗi ô

    Dùng cho:
        - Reward: coverage delta
        - Observation: local coverage + staleness
        - Analysis: re-scan patterns, hotspot detection

    Args:
        cfg: AppConfig object

    Usage:
        cov_map = CoverageMap(cfg)
        cov_map.mark_explored(uav.pos, fov_r, step)
        rate = cov_map.get_coverage_rate()
        staleness = cov_map.get_staleness(pos, radius, step)
    """

    def __init__(self, cfg: "AppConfig") -> None:
        self.cfg = cfg

        # Map dimensions từ cfg
        self.grid_size   = cfg.env.grid_size          # 100
        self.map_size    = cfg.env.map_size           # 100.0m
        self.total_cells = self.grid_size * self.grid_size  # 10000

        # Main grids (khởi tạo lần đầu, reset() khi mới episode)
        self.grid       = np.zeros((self.grid_size, self.grid_size), dtype=bool)  # ✅ FIX-06
        self.timestamps = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.first_scan = np.full((self.grid_size, self.grid_size), -1, dtype=np.int32)  # ✅ FIX-08
        self.scan_count = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

    # ─── Reset ────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset tất cả về 0 (dùng khi bắt đầu episode mới)."""
        self.grid[:]       = False  # ✅ FIX-06
        self.timestamps[:] = 0
        self.first_scan[:] = -1     # ✅ FIX-08
        self.scan_count[:] = 0

    # ─── Update (mỗi step) ───────────────────────────────────────────────────

    def mark_explored(
        self,
        uav_pos:      np.ndarray,
        fov_radius:   float,
        current_step: int,
    ) -> None:
        """
        Đánh dấu vùng FOV đã explore.
        
        ✅ FIX-02: Timestamps CHỈ update nếu scan mới hơn (preserve gradient)
        ✅ FIX-08: Track first_scan cho temporal analysis
        """
        cells = get_circle_cells(
            center=uav_pos[:2],
            radius=fov_radius,
            grid_size=self.grid_size,
            map_size=self.map_size,
        )

        if len(cells) == 0:
            return

        rows = cells[:, 0]
        cols = cells[:, 1]

        # ✅ FIX-02: CHỈ update timestamp nếu scan MỚI HƠN (không overwrite)
        newer_mask = self.timestamps[rows, cols] < current_step
        newer_rows = rows[newer_mask]
        newer_cols = cols[newer_mask]

        # Update timestamps (CHỈ cells mới hơn)
        if len(newer_rows) > 0:
            self.timestamps[newer_rows, newer_cols] = current_step

        # ✅ FIX-08: Track first scan (CHỈ khi chưa từng scan)
        first_time_mask = self.first_scan[rows, cols] == -1
        first_rows = rows[first_time_mask]
        first_cols = cols[first_time_mask]
        
        if len(first_rows) > 0:
            self.first_scan[first_rows, first_cols] = current_step

        # Update grid & scan count (tất cả cells)
        self.grid[rows, cols] = True  # ✅ FIX-06
        self.scan_count[rows, cols] += 1

    # ─── Query - Coverage ────────────────────────────────────────────────────

    def get_coverage_rate(self) -> float:
        """
        Coverage rate toàn map.

        Returns:
            float: [0, 1] (0.0 = chưa quét gì, 1.0 = quét hết)
        """
        return float(np.sum(self.grid) / self.total_cells)

    def get_coverage_percent(self) -> float:
        """
        Coverage percent.

        Returns:
            float: [0, 100]
        """
        return self.get_coverage_rate() * 100.0

    def get_local_coverage(
        self,
        pos:    np.ndarray,
        radius: float = 15.0,
    ) -> float:
        """
        Coverage trong vùng bán kính radius quanh pos.

        Args:
            pos:    [x, y, ...] center position
            radius: Bán kính (mét)

        Returns:
            float: Local coverage [0, 1]
        """
        cells = get_circle_cells(
            center=pos[:2],
            radius=radius,
            grid_size=self.grid_size,
            map_size=self.map_size,
        )

        if len(cells) == 0:
            return 0.0

        explored = np.sum(self.grid[cells[:, 0], cells[:, 1]])
        return float(explored / len(cells))

    def get_staleness(
        self,
        pos:          np.ndarray,
        radius:       float,
        current_step: int,
    ) -> float:
        """
        Staleness trung bình.

        ✅ FIX-P3: Unexplored cells contribute max_steps (không bỏ qua).
                   BEFORE: chỉ mean explored → underestimate khi vùng chưa quét
                   AFTER:  unexplored = max_steps → đúng semantic
        """
        cells = get_circle_cells(
            center=pos[:2],
            radius=radius,
            grid_size=self.grid_size,
            map_size=self.map_size,
        )

        if len(cells) == 0:
            return 0.0

        rows          = cells[:, 0]
        cols          = cells[:, 1]
        explored_mask = self.grid[rows, cols]

        # ✅ FIX-P3: Init tất cả = max_steps, override explored
        ages = np.full(len(cells), float(self.cfg.env.max_steps), dtype=np.float32)

        if np.any(explored_mask):
            explored_times      = self.timestamps[rows, cols][explored_mask]
            ages[explored_mask] = (current_step - explored_times).astype(np.float32)

        return float(np.mean(ages))

    def get_staleness_normalized(
        self,
        pos:          np.ndarray,
        radius:       float,
        current_step: int,
        decay_threshold: int = 200,  # ✅ FIX-04: normalize theo decay_threshold
    ) -> float:
        """
        Staleness normalized về [0, 1].

        0.0 = mới quét (fresh)
        1.0 = cũ lắm (stale)

        ✅ FIX-04: Normalize theo decay_threshold (không phải max_steps)
                   → Semantically đúng hơn

        Args:
            pos:             [x, y, ...]
            radius:          Bán kính (mét)
            current_step:    Step hiện tại
            decay_threshold: Ngưỡng coi là "stale" (default 200 steps)

        Returns:
            float: Staleness normalized [0, 1]
        """
        staleness = self.get_staleness(pos, radius, current_step)
        return float(min(staleness / decay_threshold, 1.0))  # ✅ FIX-04

    def get_freshness(
        self,
        pos:          np.ndarray,
        radius:       float,
        current_step: int,
        decay_threshold: int = 200,  # ✅ FIX-04
    ) -> float:
        """
        Freshness = 1 - staleness_normalized.

        1.0 = mới quét
        0.0 = cũ lắm

        Args:
            pos:             [x, y, ...]
            radius:          Bán kính (mét)
            current_step:    Step hiện tại
            decay_threshold: Ngưỡng decay (default 200)

        Returns:
            float: Freshness [0, 1]
        """
        return 1.0 - self.get_staleness_normalized(pos, radius, current_step, decay_threshold)

    # ─── Query - Advanced ────────────────────────────────────────────────────

    def get_coverage_with_decay(
        self,
        current_step:    int,
        decay_threshold: int = 200,
    ) -> float:
        """
        Coverage với decay: vùng quét lâu sẽ không tính.

        Ví dụ: Vùng quét cách đây > 200 steps → coi như chưa quét
        (vì mobile victims có thể đã di chuyển).

        Args:
            current_step:    Step hiện tại
            decay_threshold: Ngưỡng cũ (steps)

        Returns:
            float: Fresh coverage rate [0, 1]
        """
        ages = current_step - self.timestamps
        fresh_mask = self.grid & (ages < decay_threshold)  # ✅ FIX-06: bool mask
        return float(np.sum(fresh_mask) / self.total_cells)

    def get_rescan_count(
        self,
        pos:    np.ndarray,
        radius: float,
    ) -> float:  # ✅ FIX-05: return float (không làm tròn)
        """
        Số lần trung bình đã scan vùng quanh pos.

        ✅ FIX-05: Return float (không int) để preserve precision cho RL reward

        Dùng để:
            - Detect hotspots (scan nhiều lần)
            - Reward re-scan behavior

        Args:
            pos:    [x, y, ...]
            radius: Bán kính (mét)

        Returns:
            float: Scan count trung bình (preserve precision)
        """
        cells = get_circle_cells(
            center=pos[:2],
            radius=radius,
            grid_size=self.grid_size,
            map_size=self.map_size,
        )

        if len(cells) == 0:
            return 0.0

        counts = self.scan_count[cells[:, 0], cells[:, 1]]
        return float(np.mean(counts))  # ✅ FIX-05

    # ─── Navigation Helper ───────────────────────────────────────────────────

    def get_nearest_unexplored(
        self,
        pos:          np.ndarray,
        min_distance: float = 0.0,
    ) -> Optional[np.ndarray]:
        """
        Tìm ô chưa explore gần nhất.

        ⚠️ COMPLEXITY: O(N) full grid scan
        ✅ FIX-07: OK cho 100×100 (10K cells)
        ❌ TODO Phase 4: Bottleneck khi 500×500 (250K cells)
                        → Implement frontier set / KD-tree

        Args:
            pos:          [x, y, ...]
            min_distance: Khoảng cách tối thiểu (mét) - tránh ô quá gần

        Returns:
            np.ndarray [x, y] world coords hoặc None nếu đã explore hết
        """
        unexplored = np.argwhere(~self.grid)  # ✅ FIX-06: bool negation

        if len(unexplored) == 0:
            return None

        # Convert grid coords → world coords (center of cell)
        world_coords = (unexplored + 0.5) * (self.map_size / self.grid_size)

        # Tính distances
        dists = np.linalg.norm(world_coords - pos[:2], axis=1)

        # Lọc min_distance
        valid_mask = dists >= min_distance
        if not np.any(valid_mask):
            # Không có ô nào thỏa min_distance → lấy gần nhất
            nearest_idx = np.argmin(dists)
            return world_coords[nearest_idx]
        else:
            # Có ô thỏa → lấy gần nhất trong valid set
            valid_dists  = dists[valid_mask]
            valid_coords = world_coords[valid_mask]
            nearest_local_idx = np.argmin(valid_dists)
            return valid_coords[nearest_local_idx]

    def get_nearest_stale(
        self,
        pos:                 np.ndarray,
        current_step:        int,
        staleness_threshold: int = 200,
    ) -> Optional[np.ndarray]:
        """
        Tìm ô cũ nhất (stale) gần pos.

        ⚠️ COMPLEXITY: O(N) full grid scan (same issue với get_nearest_unexplored)

        Dùng cho re-scan behavior.

        Args:
            pos:                 [x, y, ...]
            current_step:        Step hiện tại
            staleness_threshold: Ngưỡng cũ (steps)

        Returns:
            np.ndarray [x, y] world coords hoặc None
        """
        ages = current_step - self.timestamps
        stale_mask = self.grid & (ages >= staleness_threshold)  # ✅ FIX-06

        stale_cells = np.argwhere(stale_mask)

        if len(stale_cells) == 0:
            return None

        # Convert → world coords
        world_coords = (stale_cells + 0.5) * (self.map_size / self.grid_size)

        # Tìm gần nhất
        dists = np.linalg.norm(world_coords - pos[:2], axis=1)
        nearest_idx = np.argmin(dists)

        return world_coords[nearest_idx]

    # ─── Analysis ─────────────────────────────────────────────────────────────

    def get_stats(self, current_step: int) -> dict:
        """
        Statistics cho logging/analysis.

        Args:
            current_step: Step hiện tại

        Returns:
            Dict với các metrics
        """
        explored_mask = self.grid

        if not np.any(explored_mask):
            return {
                "coverage_rate":     0.0,
                "coverage_percent":  0.0,
                "total_explored":    0,
                "avg_scan_count":    0.0,
                "avg_age":           0.0,
                "max_age":           0,
                "fresh_coverage_200": 0.0,
            }

        ages = current_step - self.timestamps[explored_mask]

        return {
            "coverage_rate":     float(self.get_coverage_rate()),
            "coverage_percent":  float(self.get_coverage_percent()),
            "total_explored":    int(np.sum(explored_mask)),
            "avg_scan_count":    float(np.mean(self.scan_count[explored_mask])),
            "avg_age":           float(np.mean(ages)),
            "max_age":           int(np.max(ages)),
            "fresh_coverage_200": float(self.get_coverage_with_decay(current_step, 200)),
        }

    # ─── Serialization ───────────────────────────────────────────────────────

    def to_dict(self, current_step: int) -> dict:
        """
        Export state (JSON-safe).

        NOTE: grid/timestamps/scan_count quá lớn (100×100)
              → chỉ export stats, không export raw arrays.
        """
        return {
            "grid_size":   int(self.grid_size),
            "map_size":    float(self.map_size),
            "total_cells": int(self.total_cells),
            "stats":       self.get_stats(current_step),
        }

    def get_grid_snapshot(self) -> dict:
        """
        Export raw grids (dùng cho visualization).

        ⚠️ WARNING: Large data (~30KB cho 100×100)
        ✅ FIX-07: CHỈ dùng khi cần (visualization/debugging)
                   KHÔNG log mỗi step (memory/disk chết)
        """
        return {
            "grid":       self.grid.astype(np.int8).tolist(),  # bool → int8 cho JSON
            "timestamps": self.timestamps.tolist(),
            "first_scan": self.first_scan.tolist(),  # ✅ FIX-08
            "scan_count": self.scan_count.tolist(),
        }

    def __repr__(self) -> str:
        cov = self.get_coverage_percent()
        return f"CoverageMap(coverage={cov:.1f}%, cells={self.total_cells})"