"""
visualization/visualizer2d.py - REDESIGNED v2
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np

try:
    from shapely.geometry import Point, Polygon as ShapelyPolygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

if TYPE_CHECKING:
    from config import AppConfig
    from core.coverage_map import CoverageMap
    from entities.uav import UAV
    from entities.victim import BaseVictim
    from entities.charging_station import ChargingStation

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════
# REDESIGNED COLOR PALETTE - Phân biệt rõ ràng
# ═══════════════════════════════════════════════════════════════

# UAV colors - mỗi state 1 màu hoàn toàn khác nhau
UAV_COLORS = {
    "ACTIVE":    "#2196F3",   # Blue
    "RETURNING": "#FF9800",   # Orange
    "CHARGING":  "#4CAF50",   # Green
    "DEPLOYING": "#9C27B0",   # Purple
    "DISABLED":  "#607D8B",   # Blue Grey
}

# Entities - màu tương phản cao
COVERAGE_COLOR   = "#FFF9C4"   # Vàng nhạt (explored)
STATION_COLOR    = "#1565C0"   # Xanh dương đậm
VICTIM_MISSING   = "#FF5722"   # Deep Orange (X marker)
VICTIM_FOUND     = "#00C853"   # Green A700 (checkmark)
DEBRIS_COLOR     = "#795548"   # Brown
DEBRIS_EDGE      = "#3E2723"   # Dark Brown

DANGER_COLORS = {
    "fire":      "#FF3D00",   # Red Orange
    "radiation": "#AA00FF",   # Deep Purple
    "flood":     "#0288D1",   # Light Blue
    "smoke":     "#78909C",   # Blue Grey
    "gas":       "#9CCC65",   # Light Green
    "collapse":  "#FF8F00",   # Amber
}

# ═══════════════════════════════════════════════════════════════
class Visualizer2D:
    """2D Visualizer - REDESIGNED cho clarity"""

    def __init__(self, cfg: AppConfig, render_mode: str = "human") -> None:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import Circle, Rectangle
        from matplotlib.gridspec import GridSpec

        self._plt      = plt
        self._mpatches = mpatches
        self._Circle   = Circle
        self._Rectangle = Rectangle
        self._GridSpec = GridSpec

        self.cfg         = cfg
        self.render_mode = render_mode
        self.map_size    = cfg.env.map_size
        self._fig        = None
        self._ax_map     = None
        self._ax_info    = None
        self._initialized = False

    # ── Public API ───────────────────────────────────────────────────────────

    def render(self, uavs, victims, obstacles, stations,
               cov_map, step, metrics=None):
        if not self._initialized:
            self._init_figure()

        self._ax_map.cla()
        self._ax_info.cla()
        self._setup_map_axes()
        self._ax_info.axis("off")

        # Thứ tự vẽ quan trọng (zorder)
        self._draw_coverage(cov_map)          # zorder 0-1
        self._draw_obstacles(obstacles)        # zorder 2-4
        self._draw_stations(stations)          # zorder 5-6
        self._draw_victims(victims)            # zorder 7-8
        self._draw_uavs(uavs)                 # zorder 9-12
        self._draw_map_title(step, cov_map)
        self._draw_info_panel(uavs, victims, cov_map, step, metrics)

        self._fig.canvas.draw()

        if self.render_mode == "rgb_array":
            return self._to_rgb_array()
        else:
            self._plt.pause(0.001)
            return None

    def close(self):
        if self._fig is not None:
            self._plt.close(self._fig)
            self._fig = None
            self._initialized = False

    # ── Init ────────────────────────────────────────────────────────────────

    def _init_figure(self):
        if self.render_mode == "human":
            self._plt.ion()

        self._fig = self._plt.figure(
            figsize=(16, 9),
            facecolor="#ECEFF1",  # Light blue grey background
        )
        gs = self._GridSpec(
            1, 2, figure=self._fig,
            width_ratios=[3, 1],
            wspace=0.02,
        )
        self._ax_map  = self._fig.add_subplot(gs[0, 0])
        self._ax_info = self._fig.add_subplot(gs[0, 1])
        self._initialized = True

    def _setup_map_axes(self):
        self._ax_map.set_facecolor("#F5F5F5")  # Light grey map bg
        self._ax_map.set_xlim(0, self.map_size)
        self._ax_map.set_ylim(0, self.map_size)
        self._ax_map.set_aspect("equal")
        self._ax_map.tick_params(labelsize=9)
        self._ax_map.set_xlabel("X (meters)", fontsize=10)
        self._ax_map.set_ylabel("Y (meters)", fontsize=10)
        self._ax_map.grid(True, alpha=0.2, color="#B0BEC5",
                          linewidth=0.5, linestyle="--")
        self._ax_map.set_axisbelow(True)

    # ── COVERAGE - Vàng nhạt rõ ràng ────────────────────────────────────────

    def _draw_coverage(self, cov_map) -> None:
        """
        Vẽ coverage map.
        
        FIX: Bỏ transpose vì grid[y, x] đã đúng format cho imshow.
        """
        grid = cov_map.grid.astype(float)

        # ═══ DEBUG: Verify grid range ═══
        # import logging
        # logger = logging.getLogger(__name__)
        # explored_count = int(grid.sum())
        # if explored_count > 0:
        #     explored_idx = np.argwhere(grid > 0)
        #     y_range = (explored_idx[:, 0].min(), explored_idx[:, 0].max())
        #     x_range = (explored_idx[:, 1].min(), explored_idx[:, 1].max())
        #     logger.info(
        #         f"[VIZ_DEBUG] Coverage grid: {explored_count} cells, "
        #         f"y=[{y_range[0]}, {y_range[1]}], x=[{x_range[0]}, {x_range[1]}]"
        #     )

        # Custom colormap: grey → yellow
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "coverage",
            ["#F5F5F5",   # Unexplored
            "#FFF176",   # Partially explored
            "#FFEE58"],  # Fully explored
        )

        # ═══ FIX: BỎ .T ═══
        self._ax_map.imshow(
            grid,  # ← CHANGED: Không transpose
            origin="lower",
            extent=[0, self.map_size, 0, self.map_size],
            cmap=cmap,
            vmin=0.0, vmax=1.0,
            alpha=0.70,
            zorder=1,
            interpolation="nearest",
        )

        # Coverage legend
        coverage_pct = cov_map.get_coverage_rate() * 100
        self._ax_map.text(
            self.map_size - 1, 1,
            f"Explored: {coverage_pct:.1f}%",
            color="#F57F17",
            fontsize=9, fontweight="bold",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3",
                    facecolor="white", alpha=0.9,
                    edgecolor="#F9A825"),
            zorder=20,
        )

    # ── OBSTACLES ────────────────────────────────────────────────────────────

    def _draw_obstacles(self, obstacles) -> None:
        from entities.obstacle import Debris, DangerZone
        for obs in obstacles:
            if isinstance(obs, Debris):
                self._draw_debris(obs)
            elif isinstance(obs, DangerZone):
                self._draw_danger_zone(obs)

    def _draw_debris(self, debris) -> None:
        """Brown/dark với hatch pattern → rõ là obstacle cứng"""
        x, y  = float(debris.pos[0]), float(debris.pos[1])
        color = DEBRIS_COLOR
        edge  = DEBRIS_EDGE

        patch = self._get_shape_patch(
            debris, x, y,
            facecolor=color,
            edgecolor=edge,
            alpha=0.85,
            linewidth=2.0,
            hatch="///",    # ← Hatch pattern để nhận ra ngay
            zorder=3,
        )
        if patch:
            self._ax_map.add_patch(patch)

        # Height label
        self._ax_map.text(
            x, y,
            f"{debris.height_3d:.0f}m",
            color="white", fontsize=6.5,
            ha="center", va="center",
            fontweight="bold", zorder=4,
        )

    def _draw_danger_zone(self, zone) -> None:
        """Màu rực rõ theo type, dashed border, fill nhạt"""
        x, y  = float(zone.pos[0]), float(zone.pos[1])
        color = DANGER_COLORS.get(zone.danger_type, "#FF5722")

        # Outer zone (fill nhạt)
        patch = self._get_shape_patch(
            zone, x, y,
            facecolor=color,
            edgecolor=color,
            alpha=0.15,
            linewidth=2.5,
            linestyle="--",
            zorder=2,
        )
        if patch:
            self._ax_map.add_patch(patch)

        # Solid border
        border = self._get_shape_patch(
            zone, x, y,
            facecolor="none",
            edgecolor=color,
            alpha=0.8,
            linewidth=2.5,
            linestyle="--",
            zorder=3,
        )
        if border:
            self._ax_map.add_patch(border)

        # Center dot
        self._ax_map.add_patch(self._Circle(
            (x, y), 1.8,
            facecolor=color, alpha=0.95,
            linewidth=1.5, edgecolor="white",
            zorder=4,
        ))

        # Type label ngắn gọn
        type_labels = {
            "fire": "FIRE", "radiation": "RAD",
            "flood": "FLD", "smoke": "SMK",
            "gas": "GAS", "collapse": "COL",
        }
        label = type_labels.get(zone.danger_type, zone.danger_type[:3].upper())
        r = zone._get_fallback_radius()

        self._ax_map.text(
            x, y + r + 1.0,
            label,
            color=color, fontsize=7.5, fontweight="bold",
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2",
                      facecolor="white", alpha=0.9,
                      edgecolor=color, linewidth=1.5),
            zorder=10,
        )

    def _get_shape_patch(self, obs, x, y, **patch_kwargs):
        """Helper tạo patch đúng shape."""
        shape = getattr(obs, "shape", "circle")

        if shape == "circle":
            return self._Circle(
                (x, y), obs.radius,
                **patch_kwargs,
            )

        elif shape in ("rectangle", "polygon"):
            if SHAPELY_AVAILABLE and obs.polygon is not None:
                from matplotlib.patches import Polygon as MPLPolygon
                coords = list(obs.polygon.exterior.coords)
                return MPLPolygon(coords, **patch_kwargs)
            else:
                # Fallback circle
                return self._Circle(
                    (x, y), obs._get_fallback_radius(),
                    **patch_kwargs,
                )
        return None

    # ── STATIONS - Xanh đậm dễ nhận ─────────────────────────────────────────

    def _draw_stations(self, stations) -> None:
        size = 5.0  # Lớn hơn để dễ thấy

        for i, st in enumerate(stations):
            x, y = float(st.pos[0]), float(st.pos[1])

            # Charge radius (dotted circle)
            self._ax_map.add_patch(self._Circle(
                (x, y), self.cfg.env.charge_radius,
                fill=False,
                edgecolor=STATION_COLOR,
                alpha=0.40,
                linestyle=":",
                linewidth=1.5,
                zorder=5,
            ))

            # Station body - xanh đậm
            self._ax_map.add_patch(self._Rectangle(
                (x - size/2, y - size/2), size, size,
                facecolor=STATION_COLOR,
                alpha=0.95,
                linewidth=2.0,
                edgecolor="white",
                zorder=6,
            ))

            # Label
            n_occ = len(st.current_occupants)
            cap   = self.cfg.env.station_capacity
            self._ax_map.text(
                x, y,
                f"S{i}\n{n_occ}/{cap}",
                color="white", fontsize=6.5,
                ha="center", va="center",
                fontweight="bold", zorder=7,
            )

    # ── VICTIMS - Rõ ràng không nhầm ─────────────────────────────────────────

    def _draw_victims(self, victims) -> None:
        for i, v in enumerate(victims):
            x, y = float(v.pos[0]), float(v.pos[1])

            if v.is_found:
                # Found: Green circle với checkmark lớn
                self._ax_map.add_patch(self._Circle(
                    (x, y), 2.5,
                    facecolor=VICTIM_FOUND,
                    alpha=0.95,
                    linewidth=2.0,
                    edgecolor="#004D40",
                    zorder=8,
                ))
                self._ax_map.text(
                    x, y, "V",
                    color="white", fontsize=7,
                    ha="center", va="center",
                    fontweight="bold", zorder=9,
                )
            else:
                # Missing: Orange X lớn + urgency badge
                self._ax_map.plot(
                    x, y,
                    marker="X",
                    markersize=11,
                    color=VICTIM_MISSING,
                    markeredgecolor="#BF360C",
                    markeredgewidth=1.5,
                    zorder=8,
                )
                # Urgency badge nhỏ bên trên
                self._ax_map.text(
                    x, y + 3.5,
                    f"V{i}({v.urgency:.1f})",
                    color="#BF360C", fontsize=6.5,
                    ha="center", va="bottom",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", alpha=0.85,
                              edgecolor=VICTIM_MISSING,
                              linewidth=1.0),
                    zorder=9,
                )

    # ── UAVs ────────────────────────────────────────────────────────────────

    def _draw_uavs(self, uavs) -> None:
        from entities.uav import UAVState
        from sensors.fov_sensor import FOVSensor

        fov_sensor = FOVSensor(self.cfg)

        for uav in uavs:
            x   = float(uav.pos[0])
            y   = float(uav.pos[1])
            z   = float(uav.pos[2])
            state_name = uav.state.name
            color      = UAV_COLORS.get(state_name, "#FFFFFF")

            if uav.state == UAVState.DISABLED:
                self._ax_map.plot(x, y, "x",
                    markersize=12, color="#607D8B",
                    markeredgewidth=3, zorder=10)
                self._ax_map.text(x, y - 3,
                    f"UAV{uav.id}\nDEAD",
                    color="#607D8B", fontsize=6,
                    ha="center", va="top", zorder=11)
                continue

            # FOV circle - rõ hơn (alpha cao hơn)
            fov_r = fov_sensor.calculate_fov_radius(z)
            self._ax_map.add_patch(self._Circle(
                (x, y), fov_r,
                facecolor=color,
                alpha=0.08,
                linestyle="--",
                linewidth=1.0,
                edgecolor=color,
                zorder=9,
            ))

            # UAV body - trắng với viền màu → dễ thấy trên mọi background
            body_r = 2.5
            self._ax_map.add_patch(self._Circle(
                (x, y), body_r,
                facecolor="white",
                alpha=0.95,
                linewidth=3.0,
                edgecolor=color,
                zorder=10,
            ))

            # ID với màu state
            self._ax_map.text(
                x, y, str(uav.id),
                color=color, fontsize=8,
                ha="center", va="center",
                fontweight="bold", zorder=11,
            )

            # Battery bar
            self._draw_battery_bar(x, y, uav.battery, color)

            # State label rõ ràng
            state_short = {
                "ACTIVE": "ACT", "RETURNING": "RET",
                "CHARGING": "CHG", "DEPLOYING": "DEP",
            }.get(state_name, state_name[:3])

            self._ax_map.text(
                x, y - body_r - 2.0,
                f"z={z:.0f}m|{state_short}",
                color="#212121", fontsize=6,
                ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="white", alpha=0.9,
                          edgecolor=color, linewidth=1.5),
                zorder=11,
            )

            # Velocity arrow
            vx, vy = float(uav.vel[0]), float(uav.vel[1])
            speed = np.sqrt(vx**2 + vy**2)
            if speed > 0.1:
                scale = min(speed / self.cfg.uav.max_speed_xy * 7.0, 7.0)
                self._ax_map.annotate(
                    "",
                    xy=(x + vx/speed*scale, y + vy/speed*scale),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="->",
                                   color=color, lw=2.5),
                    zorder=11,
                )

    def _draw_battery_bar(self, x, y, battery, color):
        bar_w = 5.5
        bar_h = 0.9
        bar_y = y + 3.2

        # Background
        self._ax_map.add_patch(self._Rectangle(
            (x - bar_w/2, bar_y), bar_w, bar_h,
            facecolor="#E0E0E0", alpha=1.0,
            linewidth=0.8, edgecolor="#9E9E9E",
            zorder=11,
        ))

        # Fill
        pct = np.clip(battery / 100.0, 0.0, 1.0)
        fill_color = (
            "#4CAF50" if pct > 0.5 else
            "#FF9800" if pct > 0.2 else
            "#F44336"
        )
        if pct > 0:
            self._ax_map.add_patch(self._Rectangle(
                (x - bar_w/2, bar_y), bar_w*pct, bar_h,
                facecolor=fill_color, alpha=0.95,
                linewidth=0, zorder=12,
            ))

        self._ax_map.text(
            x, bar_y + bar_h/2,
            f"{battery:.0f}%",
            color="black", fontsize=5.5,
            ha="center", va="center",
            fontweight="bold", zorder=13,
        )

    # ── MAP TITLE ────────────────────────────────────────────────────────────

    def _draw_map_title(self, step, cov_map) -> None:
        cov = cov_map.get_coverage_rate() * 100
        self._ax_map.set_title(
            f"SAR UAV Swarm  |  Step {step}/{self.cfg.env.max_steps}"
            f"  |  Coverage {cov:.1f}%",
            color="#1A237E", fontsize=11,
            fontweight="bold", pad=8,
        )

    # ── INFO PANEL ───────────────────────────────────────────────────────────

    def _draw_info_panel(self, uavs, victims, cov_map, step, metrics):
        ax = self._ax_info
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor("#FAFAFA")

        # Title
        ax.text(0.5, 0.98, "Mission Status",
                fontsize=11, fontweight="bold",
                color="#1A237E", ha="center", va="top",
                transform=ax.transAxes)

        coverage    = cov_map.get_coverage_rate() * 100
        n_found     = sum(1 for v in victims if v.is_found)
        n_total     = len(victims)
        avg_battery = np.mean([u.battery for u in uavs]) if uavs else 0

        y = 0.90

        def section(title, y):
            ax.text(0.05, y, title,
                    fontsize=8.5, fontweight="bold",
                    color="#37474F", transform=ax.transAxes)
            ax.plot([0.05, 0.95], [y - 0.012, y - 0.012],
                    color="#CFD8DC", linewidth=1.0,
                    transform=ax.transAxes, clip_on=False)
            return y - 0.045

        def row(label, value, y, color="#212121"):
            ax.text(0.08, y, label, fontsize=7.5,
                    color="#546E7A", transform=ax.transAxes)
            ax.text(0.92, y, value, fontsize=7.5,
                    color=color, fontweight="bold",
                    ha="right", transform=ax.transAxes)
            return y - 0.038

        # MISSION section
        y = section("MISSION", y)
        y = row("Step", f"{step}/{self.cfg.env.max_steps}", y)
        y = row("Coverage",
                f"{coverage:.1f}%",
                y,
                "#2E7D32" if coverage > 60 else "#E65100")
        y = row("Victims",
                f"{n_found}/{n_total}",
                y,
                "#2E7D32" if n_found == n_total else "#C62828")
        y = row("Avg Battery",
                f"{avg_battery:.0f}%",
                y,
                "#2E7D32" if avg_battery > 50 else
                "#E65100" if avg_battery > 20 else "#C62828")

        y -= 0.02

        # UAV STATUS section
        y = section("UAV STATUS", y)
        for uav in uavs:
            sname = uav.state.name
            color = UAV_COLORS.get(sname, "#666")
            bat   = uav.battery
            alt   = float(uav.pos[2])

            # Color indicator
            ax.add_patch(self._Rectangle(
                (0.06, y - 0.013), 0.035, 0.026,
                facecolor=color, alpha=0.9,
                linewidth=1, edgecolor="white",
                transform=ax.transAxes, zorder=5,
            ))

            bat_color = ("#2E7D32" if bat > 50 else
                         "#E65100" if bat > 20 else "#C62828")

            ax.text(0.12, y, f"UAV {uav.id}",
                    fontsize=7.5, fontweight="bold",
                    color="#212121", transform=ax.transAxes, va="center")
            ax.text(0.92, y,
                    f"{bat:.0f}% | z={alt:.0f}m | {sname[:3]}",
                    fontsize=7, color=bat_color,
                    fontweight="bold", ha="right",
                    transform=ax.transAxes, va="center")
            y -= 0.052

        y -= 0.01

        # LEGEND section - màu thực tế khớp với map
        y = section("LEGEND", y)
        legend_items = [
            # (color, label, marker_type)
            (UAV_COLORS["ACTIVE"],    "UAV Active",         "rect_white"),
            (UAV_COLORS["RETURNING"], "UAV Returning",      "rect_white"),
            (UAV_COLORS["CHARGING"],  "UAV Charging",       "rect_white"),
            (VICTIM_MISSING,          "Victim (missing)",   "x"),
            (VICTIM_FOUND,            "Victim (found)",     "circle"),
            (DEBRIS_COLOR,            "Debris/Building",    "rect_hatch"),
            (STATION_COLOR,           "Charging Station",   "rect"),
            ("#FFEE58",               "Explored area",      "rect"),
        ]

        for color, label, mtype in legend_items:
            if y < 0.04:
                break

            if mtype == "rect_white":
                # UAV style: trắng với viền màu
                ax.add_patch(self._Circle(
                    (0.08, y), 0.015,
                    facecolor="white", alpha=0.95,
                    linewidth=2.0, edgecolor=color,
                    transform=ax.transAxes, zorder=5,
                ))
            elif mtype == "x":
                ax.plot([0.08], [y], "X",
                        markersize=8, color=color,
                        markeredgecolor="#BF360C",
                        markeredgewidth=1.0,
                        transform=ax.transAxes,
                        clip_on=False)
            elif mtype == "circle":
                ax.add_patch(self._Circle(
                    (0.08, y), 0.015,
                    facecolor=color, alpha=0.95,
                    linewidth=1.5, edgecolor="#004D40",
                    transform=ax.transAxes, zorder=5,
                ))
            else:
                # rect / rect_hatch
                hatch = "///" if mtype == "rect_hatch" else ""
                ax.add_patch(self._Rectangle(
                    (0.055, y - 0.012), 0.05, 0.024,
                    facecolor=color, alpha=0.9,
                    linewidth=0.8, edgecolor="#444",
                    hatch=hatch,
                    transform=ax.transAxes, zorder=5,
                ))

            ax.text(0.15, y, label,
                    fontsize=7, color="#37474F",
                    transform=ax.transAxes, va="center")
            y -= 0.038

        # Border
        for sp in ['top', 'bottom', 'left', 'right']:
            ax.spines[sp].set_visible(True)
            ax.spines[sp].set_color("#CFD8DC")
            ax.spines[sp].set_linewidth(1.5)

    # ── RGB Array ────────────────────────────────────────────────────────────

    def _to_rgb_array(self):
        self._fig.canvas.draw()
        try:
            buf = np.asarray(self._fig.canvas.buffer_rgba())
            return buf[:, :, :3].copy()
        except AttributeError:
            buf = self._fig.canvas.tostring_rgb()
            w, h = self._fig.canvas.get_width_height()
            return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy()