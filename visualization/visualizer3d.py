"""
visualization/visualizer3d.py
3D Visualizer - Fixed v2.
Fixes:
    - Danger zone cylinder quá to che map → giới hạn height + opacity
    - Coverage ground rõ hơn
    - UAV markers lớn hơn, không bị che
    - Layout 2-panel: 3D (trái) + Dashboard (phải)
    - Mỗi frame = figure mới → animation đúng
"""
from __future__ import annotations

import io
import warnings
from typing import Optional

import numpy as np

_mpl_available = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    _mpl_available = True
except ImportError:
    warnings.warn("Matplotlib not available")

# ─────────────────────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────────────────────
_UAV_COLORS = {
    "ACTIVE":    "#2196F3",
    "RETURNING": "#FF9800",
    "CHARGING":  "#4CAF50",
    "DEPLOYING": "#9C27B0",
    "DISABLED":  "#9E9E9E",
}

_DANGER_COLORS = {
    "fire":      ("#FF5722", 0.18),
    "radiation": ("#E91E63", 0.15),
    "smoke":     ("#607D8B", 0.20),
    "gas":       ("#F9A825", 0.18),   # FIX: opacity thấp hơn
    "collapse":  ("#795548", 0.20),
}

_BG_DARK    = "#1A1A2E"
_PANEL_BG   = "#0D1B2A"
_TEXT_DIM   = "#B0BEC5"
_TEXT_BRIGHT= "#E3F2FD"
_ACCENT     = "#2196F3"

# ─────────────────────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────────────────────

def _circle_xy(cx, cy, r, n=32):
    t = np.linspace(0, 2 * np.pi, n)
    return cx + r * np.cos(t), cy + r * np.sin(t)


def _cylinder_faces(cx, cy, r, z0, z1, n=20):
    """Chỉ side faces (không cap) để giảm visual noise."""
    t  = np.linspace(0, 2 * np.pi, n, endpoint=True)
    xs = cx + r * np.cos(t)
    ys = cy + r * np.sin(t)
    faces = []
    for i in range(n - 1):
        faces.append([
            [xs[i],   ys[i],   z0],
            [xs[i+1], ys[i+1], z0],
            [xs[i+1], ys[i+1], z1],
            [xs[i],   ys[i],   z1],
        ])
    return faces


def _box_faces(cx, cy, w, d, z0, z1):
    hw, hd = w / 2, d / 2
    x0, x1 = cx - hw, cx + hw
    y0, y1 = cy - hd, cy + hd
    b = [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]]
    t = [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]]
    return [
        b, t,
        [b[0],b[1],t[1],t[0]],
        [b[1],b[2],t[2],t[1]],
        [b[2],b[3],t[3],t[2]],
        [b[3],b[0],t[0],t[3]],
    ]


def _cone_faces(apex, r_base, n=16):
    """FOV cone từ UAV xuống ground."""
    cx, cy = apex[0], apex[1]
    z_top  = float(apex[2])
    t  = np.linspace(0, 2 * np.pi, n, endpoint=True)
    xs = cx + r_base * np.cos(t)
    ys = cy + r_base * np.sin(t)
    faces = []
    for i in range(n - 1):
        faces.append([
            [cx, cy, z_top],
            [xs[i],   ys[i],   0.0],
            [xs[i+1], ys[i+1], 0.0],
        ])
    return faces


# ─────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────

class Visualizer3D:
    """
    3D Matplotlib Visualizer.
    Mỗi render() = figure mới → animation đúng.
    Layout: [3D Scene | Dashboard]
    """

    def __init__(self, cfg, render_mode: str = "rgb_array"):
        if not _mpl_available:
            raise ImportError("Matplotlib required")
        self.cfg         = cfg
        self.render_mode = render_mode
        self._map_size   = float(cfg.env.map_size)
        self._z_max      = float(cfg.uav.z_max_m)

    # ── Public ────────────────────────────────────────────────

    def render(self, uavs, victims, obstacles, stations, cov_map, step) -> Optional[np.ndarray]:
        fig = self._make_figure()
        ax3, ax_dash = self._make_axes(fig)
        try:
            self._draw_scene(ax3, uavs, victims, obstacles, stations, cov_map, step)
            self._draw_dashboard(ax_dash, uavs, victims, cov_map, step, stations)
            fig.tight_layout(pad=0.5)
            frame = self._to_rgb(fig)
        finally:
            plt.close(fig)

        if self.render_mode == "human":
            self._show_frame(frame)
            return None
        return frame

    def reset_scene(self):
        pass

    def close(self):
        plt.close("all")

    # ── Figure setup ──────────────────────────────────────────

    def _make_figure(self):
        return plt.figure(figsize=(16, 9), dpi=100, facecolor=_BG_DARK)

    def _make_axes(self, fig):
        gs   = gridspec.GridSpec(1, 2, figure=fig,
                                 width_ratios=[3, 1], wspace=0.03)
        ax3  = fig.add_subplot(gs[0, 0], projection="3d")
        axd  = fig.add_subplot(gs[0, 1])

        # Style 3D
        ms = self._map_size
        ax3.set_facecolor("#16213E")
        for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
            pane.fill = True
        ax3.xaxis.pane.set_facecolor("#0F3460")
        ax3.yaxis.pane.set_facecolor("#0F3460")
        ax3.zaxis.pane.set_facecolor("#16213E")
        for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
            axis._axinfo["grid"]["color"] = "#FFFFFF18"
        ax3.set_xlabel("X (m)", color="#90CAF9", fontsize=8, labelpad=4)
        ax3.set_ylabel("Y (m)", color="#90CAF9", fontsize=8, labelpad=4)
        ax3.set_zlabel("Z (m)", color="#90CAF9", fontsize=8, labelpad=4)
        ax3.tick_params(colors="#90CAF9", labelsize=6)
        ax3.set_xlim(0, ms); ax3.set_ylim(0, ms)
        ax3.set_zlim(0, self._z_max + 5)
        ax3.view_init(elev=35, azim=-55)

        # Style dashboard
        axd.set_facecolor(_PANEL_BG)
        axd.axis("off")
        for sp in axd.spines.values():
            sp.set_edgecolor(_ACCENT)
            sp.set_linewidth(0.8)

        return ax3, axd

    # ── Scene ─────────────────────────────────────────────────

    def _draw_scene(self, ax, uavs, victims, obstacles, stations, cov_map, step):
        self._scene_boundary(ax)
        self._scene_coverage(ax, cov_map)
        self._scene_obstacles(ax, obstacles)
        self._scene_stations(ax, stations)
        self._scene_victims(ax, victims)
        self._scene_uavs(ax, uavs)
        self._scene_fov(ax, uavs)
        self._scene_title(ax, step, cov_map, victims)

    def _scene_boundary(self, ax):
        ms = self._map_size
        xs = [0, ms, ms, 0,  0]
        ys = [0,  0, ms, ms, 0]
        ax.plot(xs, ys, [0]*5, color=_ACCENT, lw=1.0, alpha=0.5)

    def _scene_coverage(self, ax, cov_map):
        """Coverage = màu ground cells."""
        ms   = self._map_size
        grid = cov_map.grid          # bool [GS, GS]
        gs   = grid.shape[0]
        cell = ms / gs

        explored = np.argwhere(grid)
        if len(explored) == 0:
            return

        # Batch: tất cả cells 1 lần
        verts = []
        for row, col in explored:
            x0, y0 = col * cell, row * cell
            x1, y1 = x0 + cell, y0 + cell
            verts.append([[x0,y0,0],[x1,y0,0],[x1,y1,0],[x0,y1,0]])

        rate  = len(explored) / (gs * gs)
        alpha = 0.10 + 0.25 * rate

        poly = Poly3DCollection(verts, facecolor="#FFF9C4",
                                edgecolor="none", alpha=alpha, zorder=0)
        ax.add_collection3d(poly)

    def _scene_obstacles(self, ax, obstacles):
        from entities.obstacle import Debris, DangerZone
        for obs in obstacles:
            if isinstance(obs, DangerZone):
                self._draw_danger(ax, obs)
            elif isinstance(obs, Debris):
                self._draw_debris(ax, obs)

    def _draw_debris(self, ax, d):
        cx, cy  = float(d.pos[0]), float(d.pos[1])
        h       = float(getattr(d, "height_3d", None) or 5.0)
        shape   = getattr(d, "shape", "circle")

        if shape == "circle":
            r     = float(getattr(d, "radius", None) or 2.5)
            verts = _cylinder_faces(cx, cy, r, 0, h, n=14)
        elif shape == "rectangle":
            w = float(getattr(d, "width",     None) or 4.0)
            dd= float(getattr(d, "height_2d", None) or w)
            verts = _box_faces(cx, cy, w, dd, 0, h)
        else:
            r     = float(d._get_fallback_radius() if hasattr(d,"_get_fallback_radius") else 2.5)
            verts = _cylinder_faces(cx, cy, r, 0, h, n=14)

        ax.add_collection3d(Poly3DCollection(
            verts, facecolor="#8D6E63",
            edgecolor="#5D4037", lw=0.4, alpha=0.85))

        # Height label
        ax.text(cx, cy, h + 0.5, f"{h:.0f}m",
                color="#FFCCBC", fontsize=5, ha="center")

    def _draw_danger(self, ax, z):
        cx, cy  = float(z.pos[0]), float(z.pos[1])
        dtype   = getattr(z, "danger_type", "fire")
        col, alpha = _DANGER_COLORS.get(dtype, ("#FF5722", 0.18))

        r = getattr(z, "radius", None)
        if r is None:
            r = z._get_fallback_radius() if hasattr(z,"_get_fallback_radius") else 5.0
        r = float(r)

        # FIX: Giới hạn height hiển thị = 10m (không che map)
        z_top = min(float(getattr(z, "max_altitude", self._z_max)), 10.0)

        verts = _cylinder_faces(cx, cy, r, 0, z_top, n=18)
        ax.add_collection3d(Poly3DCollection(
            verts, facecolor=col, edgecolor=col, lw=0.2, alpha=alpha))

        # Ground circle rõ hơn
        xs, ys = _circle_xy(cx, cy, r)
        ax.plot(xs, ys, np.zeros_like(xs), color=col, lw=1.5, alpha=0.8)

        # Label ngắn
        ax.text(cx, cy, z_top + 0.5, dtype[:3].upper(),
                color=col, fontsize=6, ha="center", fontweight="bold")

    def _scene_stations(self, ax, stations):
        for st in stations:
            cx, cy = float(st.pos[0]), float(st.pos[1])

            # Platform
            verts = _box_faces(cx, cy, 5.0, 5.0, 0, 1.0)
            ax.add_collection3d(Poly3DCollection(
                verts, facecolor="#1565C0",
                edgecolor="#90CAF9", lw=0.8, alpha=0.95))

            # Charge radius on ground
            r  = float(getattr(st, "charge_radius_m",
                               getattr(self.cfg.env, "charge_radius_m", 3.0)))
            xs, ys = _circle_xy(cx, cy, r)
            ax.plot(xs, ys, np.zeros_like(xs),
                    color="#4FC3F7", lw=1.0, ls="--", alpha=0.7)

            sid = getattr(st, "id", getattr(st, "station_id", "?"))
            ax.text(cx, cy, 2.0, f"S{sid}",
                    color="white", fontsize=7, ha="center", fontweight="bold")

    def _scene_victims(self, ax, victims):
        fx, fy, fz = [], [], []
        ux, uy, uz = [], [], []
        urgencies  = []

        for v in victims:
            p = v.pos
            if v.is_found:
                fx.append(p[0]); fy.append(p[1]); fz.append(0.8)
            else:
                ux.append(p[0]); uy.append(p[1]); uz.append(0.8)
                urgencies.append(float(getattr(v, "urgency", 3.0)))

        if fx:
            ax.scatter(fx, fy, fz, c="#00E676", s=100, marker="D",
                       edgecolors="white", lw=0.8, depthshade=True, zorder=15)
        if ux:
            sz = [40 + 14 * u for u in urgencies]
            ax.scatter(ux, uy, uz, c="#FF6F00", s=sz, marker="*",
                       edgecolors="#FFCC02", lw=0.4, depthshade=True, zorder=15)

    def _scene_uavs(self, ax, uavs):
        """UAV = sphere lớn + drop line + velocity arrow."""
        by_state: dict[str, list] = {}
        for u in uavs:
            sn = u.state.name if hasattr(u.state, "name") else str(u.state)
            by_state.setdefault(sn, []).append(u)

        for sn, group in by_state.items():
            col = _UAV_COLORS.get(sn, "#FFFFFF")
            xs  = [u.pos[0] for u in group]
            ys  = [u.pos[1] for u in group]
            zs  = [u.pos[2] for u in group]

            # Body sphere
            ax.scatter(xs, ys, zs, c=col, s=200, marker="o",
                       edgecolors="white", lw=1.5,
                       depthshade=False,   # False = luôn visible
                       zorder=25)

            for u in group:
                x, y, z = float(u.pos[0]), float(u.pos[1]), float(u.pos[2])

                # ID label
                ax.text(x, y, z + 2.5, f"U{u.id}",
                        color=col, fontsize=7,
                        ha="center", fontweight="bold", zorder=26)

                # Battery % nhỏ
                ax.text(x, y, z - 2.0,
                        f"{u.battery_pct:.0f}%",
                        color=col, fontsize=5.5,
                        ha="center", alpha=0.9, zorder=26)

                # Drop line
                if sn not in ("DISABLED", "CHARGING"):
                    ax.plot([x,x],[y,y],[0,z],
                            color=col, lw=0.5, ls=":", alpha=0.45)

                # Velocity arrow
                vel   = getattr(u, "vel", np.zeros(3))
                speed = float(np.linalg.norm(vel))
                if speed > 0.2:
                    sc = min(speed / 5.0, 1.0) * 8.0
                    ax.quiver(x, y, z,
                              vel[0]/speed*sc, vel[1]/speed*sc, vel[2]/speed*sc,
                              color=col, lw=1.5, alpha=0.9,
                              arrow_length_ratio=0.35)

    def _scene_fov(self, ax, uavs):
        for u in uavs:
            sn = u.state.name if hasattr(u.state, "name") else "ACTIVE"
            if sn in ("DISABLED", "CHARGING"):
                continue
            fov_r = (u.get_fov_radius() if hasattr(u, "get_fov_radius")
                     else float(u.pos[2]) * 0.577)
            if fov_r < 0.5:
                continue
            col   = _UAV_COLORS.get(sn, "#FFFFFF")
            faces = _cone_faces(u.pos, fov_r, n=14)
            ax.add_collection3d(Poly3DCollection(
                faces, facecolor=col, edgecolor="none", alpha=0.04))

            xs, ys = _circle_xy(float(u.pos[0]), float(u.pos[1]), fov_r)
            ax.plot(xs, ys, np.zeros_like(xs),
                    color=col, lw=0.6, ls="-.", alpha=0.5)

    def _scene_title(self, ax, step, cov_map, victims):
        cov     = cov_map.get_coverage_percent()
        n_found = sum(1 for v in victims if v.is_found)
        n_total = len(victims)
        ax.set_title(
            f"SAR UAV Swarm  │  Step {step:4d}  │  "
            f"Coverage {cov:.1f}%  │  Victims {n_found}/{n_total}",
            color=_TEXT_BRIGHT, fontsize=10, pad=6, fontweight="bold")

        # State legend
        handles = [
            mpatches.Patch(facecolor=c, edgecolor="white",
                           lw=0.5, label=s)
            for s, c in _UAV_COLORS.items()
        ]
        ax.legend(handles=handles, loc="upper left",
                  fontsize=6, framealpha=0.25,
                  facecolor=_PANEL_BG, edgecolor=_ACCENT,
                  labelcolor=_TEXT_BRIGHT)

    # ── Dashboard (right panel) ───────────────────────────────

    def _draw_dashboard(self, ax, uavs, victims, cov_map, step, stations):
        """
        Dashboard dọc: Mission Metrics + Fleet + Battery bars.
        Chia thành nhiều subplot nhỏ bên trong axes.
        """
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_facecolor(_PANEL_BG)
        ax.axis("off")

        ms        = self._map_size
        max_steps = self.cfg.env.max_steps
        cov_rate  = cov_map.get_coverage_rate()
        n_found   = sum(1 for v in victims if v.is_found)
        n_total   = len(victims)
        time_left = max(0, max_steps - step) / max_steps
        avg_bat   = np.mean([u.battery_pct for u in uavs]) if uavs else 0.0

        n_active   = sum(1 for u in uavs if u.state.name == "ACTIVE")
        n_charging = sum(1 for u in uavs if u.state.name == "CHARGING")
        n_return   = sum(1 for u in uavs if u.state.name == "RETURNING")
        n_disabled = sum(1 for u in uavs if u.state.name == "DISABLED")

        T = ax.transAxes

        def title(text, y):
            ax.text(0.5, y, text, ha="center", color="#90CAF9",
                    fontsize=8, fontweight="bold", transform=T)

        def divider(y):
            ax.plot([0.05, 0.95], [y, y], color="#1E3A5F",
                    lw=0.8, transform=T, clip_on=False)

        def metric(label, value, y, vc="#E3F2FD"):
            ax.text(0.06, y, label, ha="left", color=_TEXT_DIM,
                    fontsize=7.5, transform=T, va="center")
            ax.text(0.94, y, value, ha="right", color=vc,
                    fontsize=7.5, fontweight="bold", transform=T, va="center")

        def progress_bar(y, val, h=0.028, fill="#2196F3", label=""):
            # bg
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.06, y), 0.88, h,
                boxstyle="round,pad=0.003",
                facecolor="#1E3A5F", edgecolor="#2196F3",
                lw=0.5, transform=T))
            # fill
            fw = max(0.005, val * 0.88)
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.06, y), fw, h,
                boxstyle="round,pad=0.003",
                facecolor=fill, edgecolor="none",
                transform=T))
            # label
            ax.text(0.5, y + h / 2, label,
                    ha="center", va="center",
                    color="white", fontsize=6, fontweight="bold",
                    transform=T)

        # ── Section: MISSION ──────────────────────────────────
        y = 0.96
        title("MISSION STATUS", y);  y -= 0.04
        divider(y);                   y -= 0.035

        cov_col  = "#00E676" if cov_rate > 0.6 else ("#FF9800" if cov_rate > 0.3 else "#F44336")
        vic_col  = "#00E676" if n_found == n_total else ("#FF9800" if n_found > 0 else "#F44336")
        bat_col  = "#00E676" if avg_bat > 50 else ("#FF9800" if avg_bat > 20 else "#F44336")
        time_col = "#00E676" if time_left > 0.5 else ("#FF9800" if time_left > 0.2 else "#F44336")

        metric("Coverage",   f"{cov_rate*100:.1f}%",   y, cov_col);  y -= 0.042
        metric("Victims",    f"{n_found}/{n_total}",    y, vic_col);  y -= 0.042
        metric("Time Left",  f"{time_left*100:.0f}%",  y, time_col); y -= 0.042
        metric("Avg Battery",f"{avg_bat:.0f}%",         y, bat_col);  y -= 0.042
        metric("Step",       f"{step}/{max_steps}",     y);           y -= 0.042

        # Progress bars
        y -= 0.005
        progress_bar(y, cov_rate, fill=cov_col,
                     label=f"Coverage {cov_rate*100:.0f}%"); y -= 0.040
        progress_bar(y, n_found/max(n_total,1), fill=vic_col,
                     label=f"Victims {n_found}/{n_total}");  y -= 0.040
        progress_bar(y, avg_bat/100, fill=bat_col,
                     label=f"Battery {avg_bat:.0f}%");       y -= 0.040

        divider(y); y -= 0.035

        # ── Section: FLEET ────────────────────────────────────
        title("FLEET STATUS", y); y -= 0.04
        divider(y);                y -= 0.035

        fleet_rows = [
            ("Active",    n_active,   _UAV_COLORS["ACTIVE"]),
            ("Charging",  n_charging, _UAV_COLORS["CHARGING"]),
            ("Returning", n_return,   _UAV_COLORS["RETURNING"]),
            ("Disabled",  n_disabled, _UAV_COLORS["DISABLED"]),
        ]
        for label, val, col in fleet_rows:
            # Color dot
            ax.add_patch(mpatches.Circle(
                (0.09, y), 0.012,
                facecolor=col, edgecolor="white",
                lw=0.5, transform=T))
            ax.text(0.14, y, label, ha="left", color=_TEXT_DIM,
                    fontsize=7.5, transform=T, va="center")
            ax.text(0.94, y, str(val), ha="right", color=col,
                    fontsize=8, fontweight="bold", transform=T, va="center")
            y -= 0.042

        divider(y); y -= 0.035

        # ── Section: BATTERY per UAV ──────────────────────────
        title("BATTERY", y); y -= 0.04
        divider(y);           y -= 0.03

        bar_h   = 0.026
        n_uavs  = len(uavs)
        spacing = min(0.042, (y - 0.02) / max(n_uavs, 1))

        for u in uavs:
            bat  = u.battery_pct / 100.0
            sn   = u.state.name if hasattr(u.state, "name") else "ACTIVE"
            sc   = _UAV_COLORS.get(sn, "#FFFFFF")
            bc   = "#4CAF50" if bat > 0.5 else ("#FF9800" if bat > 0.2 else "#F44336")

            # UAV label
            ax.text(0.06, y, f"U{u.id}",
                    ha="left", color=sc,
                    fontsize=7, fontweight="bold",
                    transform=T, va="center")

            # Bar background
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.15, y - bar_h/2), 0.72, bar_h,
                boxstyle="round,pad=0.002",
                facecolor="#1E3A5F", edgecolor="#334155",
                lw=0.4, transform=T))

            # Bar fill
            fw = max(0.005, bat * 0.72)
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.15, y - bar_h/2), fw, bar_h,
                boxstyle="round,pad=0.002",
                facecolor=bc, edgecolor="none",
                transform=T))

            # Percent text
            ax.text(0.94, y, f"{u.battery_pct:.0f}%",
                    ha="right", color=bc,
                    fontsize=6.5, transform=T, va="center")

            y -= spacing

        # ── Map info ──────────────────────────────────────────
        ax.text(0.5, 0.01,
                f"Map {int(ms)}×{int(ms)}m  │  {n_uavs} UAVs",
                ha="center", color="#546E7A",
                fontsize=6, transform=T)

    # ── Output ────────────────────────────────────────────────

    def _to_rgb(self, fig) -> np.ndarray:
        fig.canvas.draw()
        # Method 1
        try:
            buf = fig.canvas.buffer_rgba()
            arr = np.asarray(buf).copy()
            return arr[:, :, :3]
        except (AttributeError, ValueError):
            pass
        # Method 2: BytesIO PNG (most compatible)
        try:
            from PIL import Image
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches=None)
            buf.seek(0)
            return np.array(Image.open(buf).convert("RGB"))
        except Exception:
            pass
        # Method 3
        try:
            buf = fig.canvas.tostring_rgb()
            w   = int(fig.get_figwidth()  * fig.dpi)
            h   = int(fig.get_figheight() * fig.dpi)
            return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy()
        except Exception:
            pass
        # Method 4: ARGB
        buf = fig.canvas.tostring_argb()
        w   = int(fig.get_figwidth()  * fig.dpi)
        h   = int(fig.get_figheight() * fig.dpi)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        return arr[:, :, 1:].copy()

    def _show_frame(self, frame: np.ndarray):
        try:
            import cv2
            cv2.imshow("SAR 3D", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        except ImportError:
            fig = plt.figure("SAR 3D Live", figsize=(16, 9))
            fig.clear()
            ax = fig.add_subplot(111)
            ax.imshow(frame); ax.axis("off")
            plt.pause(0.001)