#!/usr/bin/env python3
"""
📊 Plot So Sánh - Style Paper
Usage:
    python plot_compare.py /path/to/run1 /path/to/run2 --save-dir ./plots
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List

# ── Colors ───────────────────────────────────────────────────────────────────
RUN_COLORS = {
    "masac_reward_base_new_1": "#E53935",  # Red
    "masac_reward_base_new_2": "#1E88E5",  # Blue
    "masac_reward_llm_new_1": "#43A047",  # Red
    "masac_reward_llm_new_2": "#FB8C00",  # Blue
    "masac_reward_llm_3.6_new_1": "#BF2EAE",  # Red
    "masac_reward_llm_3.6_new_2": "#EAFB00",  # Blue
    "masac_reward_llm_3.6_elsiver_new_1": "#1CD5F6",  # Red
    "masac_reward_llm_3.6_elsiver_new_2": "#B0FB00",  # Blue
}

PALETTE = [
    "#1E88E5",  # Blue
    "#E53935",  # Red  
    "#43A047",  # Green
    "#FB8C00",  # Orange
    "#BF2EAE",  # Purple
    "#EAFB00",  # Cyan
    "#1CD5F6"
    "#B0FB00"
]

LABEL_MAP = {
    "masac_reward_base_new_1": "MASAC v1 (Old Reward)",
    "masac_reward_base_new_2": "MASAC v2 (New Reward)",
    "masac_reward_llm_new_1": "MASAC v1 (Old Reward)",
    "masac_reward_llm_new_2": "MASAC v2 (New Reward)",
}

MARKERS = ['o', 's', '^', 'v', 'D', 'p']


def _get_color(run_name: str, idx: int) -> str:
    return RUN_COLORS.get(run_name, PALETTE[idx % len(PALETTE)])


def _get_label(run_name: str) -> str:
    return LABEL_MAP.get(run_name, run_name)


def _smooth(data: np.ndarray, window: int) -> np.ndarray:
    """Moving average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def _plot_metric(
    runs_data:    Dict[str, Dict],
    metric_key:   str,
    ylabel:       str,
    title:        str,
    save_path:    str,
    smooth_window: int   = 20,      # Smooth vừa phải
    n_markers:     int   = 15,      # Số markers trên đường
    target_line:   float = None,
    target_label:  str   = None,
    ylim:          tuple = None,
):
    """Plot clean - smooth + marker thưa, giống style paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Style theo paper ──────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':      'DejaVu Sans',
        'font.size':        12,
        'axes.linewidth':   1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.visible': False,
        'ytick.minor.visible': False,
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (run_name, metrics) in enumerate(runs_data.items()):
        data = metrics.get(metric_key, [])
        if not data:
            print(f"  ⚠️  [{run_name}] no data for '{metric_key}'")
            continue

        raw    = np.array(data, dtype=float)
        color  = _get_color(run_name, idx)
        label  = _get_label(run_name)
        marker = MARKERS[idx % len(MARKERS)]

        # ── Smooth ────────────────────────────────────────────────────────
        if len(raw) >= smooth_window:
            smoothed = _smooth(raw, smooth_window)
            # x axis bắt đầu từ smooth_window//2 để center
            x_smooth = np.arange(smooth_window - 1, len(raw))
        else:
            smoothed = raw
            x_smooth = np.arange(len(raw))

        # ── Chọn vị trí markers thưa ──────────────────────────────────────
        n_pts     = len(smoothed)
        mark_step = max(1, n_pts // n_markers)
        mark_idx  = np.arange(0, n_pts, mark_step)

        # ── Plot đường smooth ─────────────────────────────────────────────
        ax.plot(
            x_smooth, smoothed,
            color=color,
            linewidth=2.2,
            alpha=0.95,
            zorder=3,
        )

        # ── Plot markers thưa (riêng biệt để đẹp hơn) ────────────────────
        ax.plot(
            x_smooth[mark_idx], smoothed[mark_idx],
            color=color,
            marker=marker,
            markersize=7,
            linewidth=0,          # Không vẽ đường, chỉ markers
            markerfacecolor=color,
            markeredgecolor='white',
            markeredgewidth=1.2,
            label=label,
            zorder=4,
        )

    # ── Target line ───────────────────────────────────────────────────────
    if target_line is not None:
        ax.axhline(
            target_line,
            color='gray',
            linestyle='--',
            linewidth=1.5,
            alpha=0.6,
            label=target_label or f"Target {target_line}",
            zorder=1,
        )

    # ── Labels & Style ────────────────────────────────────────────────────
    ax.set_xlabel("Episode",  fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel,     fontsize=13, fontweight='bold')
    ax.set_title(title,       fontsize=15, fontweight='bold', pad=15)

    # Legend - box có border như paper
    ax.legend(
        fontsize=11,
        loc='best',
        framealpha=0.95,
        edgecolor='black',
        fancybox=False,
        borderpad=0.8,
    )

    # Grid dashed nhẹ như paper
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.4, color='gray')
    ax.set_axisbelow(True)  # Grid ở dưới đường vẽ

    # Giới hạn trục
    if ylim:
        ax.set_ylim(*ylim)

    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ Saved: {save_path}")


def plot_summary_table(runs_data: Dict[str, Dict], save_path: str):
    """Summary table đẹp."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 3 + len(runs_data)))
    ax.axis("off")

    rows = []
    cols = ["Run", "Episodes", "Reward (mean)", "Reward (last 50)", "Best Reward"]

    for run_name, metrics in runs_data.items():
        ep_r = metrics.get("ep_rewards", [])
        n    = len(ep_r)

        if ep_r:
            r_all  = float(np.mean(ep_r))
            r_last = float(np.mean(ep_r[-50:])) if n >= 50 else float(np.mean(ep_r))
            r_best = float(np.max(ep_r))
        else:
            r_all = r_last = r_best = 0

        rows.append([
            _get_label(run_name),
            str(n),
            f"{r_all:.1f}",
            f"{r_last:.1f}",
            f"{r_best:.1f}",
        ])

    if rows:
        tbl = ax.table(
            cellText=rows, colLabels=cols,
            cellLoc="center", loc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1.3, 2.8)

        # Header
        for j in range(len(cols)):
            tbl[0, j].set_facecolor("#37474F")
            tbl[0, j].set_text_props(color="white", fontweight="bold")

        # Row color
        for i, (run_name, _) in enumerate(runs_data.items()):
            c = _get_color(run_name, i)
            for j in range(len(cols)):
                tbl[i + 1, j].set_facecolor(c + "25")

        ax.set_title("Summary Statistics", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
        plt.close(fig)
        print(f"  ✅ Saved: {save_path}")


def plot_all(
    runs_data:     Dict[str, Dict],
    save_dir:      str = "./plots",
    smooth_window: int = 20,
    n_markers:     int = 15,
):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"\n📊 Saving plots → {save_path}\n")

    _plot_metric(
        runs_data, "ep_rewards",
        ylabel="Episode Reward",
        title="Episode Reward Comparison",
        save_path=str(save_path / "01_reward.png"),
        smooth_window=smooth_window,
        n_markers=n_markers,
    )

    _plot_metric(
        runs_data, "ep_coverage",
        ylabel="Coverage Rate (%)",
        title="Coverage Rate Comparison",
        save_path=str(save_path / "02_coverage.png"),
        smooth_window=smooth_window,
        n_markers=n_markers,
        target_line=80,
        target_label="Target 80%",
        ylim=(0, 100),
    )

    _plot_metric(
        runs_data, "ep_victims",
        ylabel="Victims Found Rate (%)",
        title="Victims Found Rate Comparison",
        save_path=str(save_path / "03_victims.png"),
        smooth_window=smooth_window,
        n_markers=n_markers,
        target_line=80,
        target_label="Target 80%",
        ylim=(0, 100),
    )

    _plot_metric(
        runs_data, "ep_lengths",
        ylabel="Episode Length (Steps)",
        title="Episode Length Comparison",
        save_path=str(save_path / "04_length.png"),
        smooth_window=smooth_window,
        n_markers=n_markers,
    )

    plot_summary_table(
        runs_data,
        save_path=str(save_path / "05_summary.png"),
    )

    print(f"\n✅ Done! All plots in: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="*", default=[])
    p.add_argument("--save-dir",      type=str, default="./plots")
    p.add_argument("--smooth-window", type=int, default=20,
                   help="Smoothing window (default=20, tăng nếu muốn mượt hơn)")
    p.add_argument("--n-markers",     type=int, default=15,
                   help="Số markers trên mỗi đường (default=15)")
    return p.parse_args()


def main():
    args = parse_args()
    runs_data: Dict[str, Dict] = {}

    if not args.paths:
        print("❌ Cần path! Ví dụ:")
        print("   python plot_compare.py /path/run1 /path/run2")
        return

    print("📂 Loading...")
    for ps in args.paths:
        path = Path(ps)

        if path.suffix in [".pt", ".pth"]:
            mf, rn = path.parent / "metrics.json", path.parent.name
        elif path.name == "metrics.json":
            mf, rn = path, path.parent.name
        elif path.is_dir():
            mf, rn = path / "metrics.json", path.name
        else:
            print(f"  ⚠️  Skip: {path}")
            continue

        if not mf.exists():
            print(f"  ❌ Not found: {mf}")
            continue

        with open(mf) as f:
            runs_data[rn] = json.load(f)
        print(f"  ✅ {rn} ({len(runs_data[rn].get('ep_rewards', []))} eps)")

    if not runs_data:
        print("❌ No data!")
        return

    plot_all(
        runs_data     = runs_data,
        save_dir      = args.save_dir,
        smooth_window = args.smooth_window,
        n_markers     = args.n_markers,
    )


if __name__ == "__main__":
    main()