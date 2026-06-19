#!/usr/bin/env python3
"""
📊 Plot So Sánh MAPPO vs MASAC vs MATD3
Download metrics từ HuggingFace → plot → save

Usage:
    python plot_compare.py
    python plot_compare.py --runs mappo_s42 masac_s42 matd3_s42
    python plot_compare.py --local-dir ./downloaded_metrics
    python plot_compare.py --save comparison.png
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from hf_upload import HFDownloader


# ── Màu cố định cho từng thuật toán ──────────────────────────────────────────
ALGO_COLORS = {
    "mappo":  "#2196F3",   # Blue
    "masac":  "#F44336",   # Red
    "matd3":  "#4CAF50",   # Green
}

ALGO_LABELS = {
    "mappo":  "MAPPO",
    "masac":  "MASAC",
    "matd3":  "MATD3",
}


def _detect_algo(run_name: str) -> str:
    rn = run_name.lower()
    for algo in ["mappo", "masac", "matd3"]:
        if algo in rn:
            return algo
    return "unknown"


def _smooth(data: List[float], window: int) -> np.ndarray:
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode="valid")


def _plot_metric(
    ax,
    runs_data:   Dict[str, Dict],
    metric_key:  str,
    ylabel:      str,
    title:       str,
    window:      int  = 50,
    target_line: float = None,
    target_label: str = None,
    ylim:        tuple = None,
):
    """Plot 1 metric với mean ± std cho nhiều seeds cùng algo."""

    # Group theo algo
    algo_data: Dict[str, List[List[float]]] = {}

    for run_name, metrics in runs_data.items():
        algo = _detect_algo(run_name)
        data = metrics.get(metric_key, [])
        if not data:
            continue
        if algo not in algo_data:
            algo_data[algo] = []
        algo_data[algo].append(data)

    for algo, all_seeds in sorted(algo_data.items()):
        color = ALGO_COLORS.get(algo, "gray")
        label = ALGO_LABELS.get(algo, algo.upper())

        # Align lengths (min length across seeds)
        min_len = min(len(d) for d in all_seeds)
        arr     = np.array([d[:min_len] for d in all_seeds])  # [n_seeds, T]

        episodes = np.arange(1, min_len + 1)

        if len(all_seeds) == 1:
            # Single seed: plot raw + smoothed
            raw = arr[0]
            ax.plot(episodes, raw, alpha=0.15, color=color, linewidth=0.5)

            if min_len >= window:
                sm  = _smooth(raw, window)
                sx  = np.arange(window, min_len + 1)
                ax.plot(sx, sm, color=color, linewidth=2.0, label=label)
            else:
                ax.plot(episodes, raw, color=color, linewidth=2.0, label=label)

        else:
            # Multi-seed: plot mean ± std
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)

            if min_len >= window:
                sm_mean = _smooth(mean, window)
                sm_std  = _smooth(std,  window)
                sx      = np.arange(window, min_len + 1)
                ax.plot(sx, sm_mean, color=color, linewidth=2.0, label=f"{label} (n={len(all_seeds)})")
                ax.fill_between(sx, sm_mean - sm_std, sm_mean + sm_std, alpha=0.2, color=color)
            else:
                ax.plot(episodes, mean, color=color, linewidth=2.0, label=f"{label} (n={len(all_seeds)})")
                ax.fill_between(episodes, mean - std, mean + std, alpha=0.2, color=color)

    if target_line is not None:
        ax.axhline(
            target_line,
            color     = "orange",
            linestyle = "--",
            linewidth = 1.0,
            alpha     = 0.8,
            label     = target_label or f"Target {target_line}",
        )

    ax.set_xlabel("Episode",  fontsize=11)
    ax.set_ylabel(ylabel,     fontsize=11)
    ax.set_title(title,       fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    if ylim:
        ax.set_ylim(*ylim)


def plot_comparison(
    runs_data:  Dict[str, Dict],
    save_path:  str = None,
    window:     int = 50,
    title:      str = "MAPPO vs MASAC vs MATD3 — HARD Stage Comparison",
):
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)

    # Row 1
    _plot_metric(
        axes[0, 0], runs_data,
        metric_key  = "ep_rewards",
        ylabel      = "Episode Reward",
        title       = "Episode Reward",
        window      = window,
        target_line = 0,
    )

    _plot_metric(
        axes[0, 1], runs_data,
        metric_key   = "ep_coverage",
        ylabel       = "Coverage (%)",
        title        = "Coverage Rate",
        window       = window,
        target_line  = 70,
        target_label = "Target 70%",
        ylim         = (0, 100),
    )

    _plot_metric(
        axes[0, 2], runs_data,
        metric_key   = "ep_victims",
        ylabel       = "Victims Found (%)",
        title        = "Victims Found Rate",
        window       = window,
        target_line  = 80,
        target_label = "Target 80%",
        ylim         = (0, 100),
    )

    # Row 2
    _plot_metric(
        axes[1, 0], runs_data,
        metric_key = "ep_lengths",
        ylabel     = "Steps",
        title      = "Episode Length",
        window     = window,
    )

    # Sample efficiency: reward vs steps (nếu có total_steps)
    ax_eff = axes[1, 1]
    has_steps_data = False

    for run_name, metrics in runs_data.items():
        algo     = _detect_algo(run_name)
        color    = ALGO_COLORS.get(algo, "gray")
        label    = ALGO_LABELS.get(algo, algo.upper())
        rewards  = metrics.get("ep_rewards", [])
        n_steps  = metrics.get("total_steps", None)

        if rewards and n_steps:
            has_steps_data = True
            # Approximate: steps per episode = total_steps / n_episodes
            n_eps        = len(rewards)
            steps_approx = np.linspace(0, n_steps, n_eps)

            sm_r = _smooth(rewards, window) if len(rewards) >= window else np.array(rewards)
            sx   = steps_approx[window - 1:] if len(rewards) >= window else steps_approx

            ax_eff.plot(sx, sm_r, color=color, linewidth=2.0, label=label)

    if has_steps_data:
        ax_eff.set_xlabel("Environment Steps", fontsize=11)
        ax_eff.set_ylabel("Reward",            fontsize=11)
        ax_eff.set_title("⚡ Sample Efficiency", fontsize=12, fontweight="bold")
        ax_eff.legend(fontsize=9)
        ax_eff.grid(alpha=0.3)
    else:
        ax_eff.set_title("⚡ Sample Efficiency\n(N/A — no step data)", fontsize=12)
        ax_eff.axis("off")

    # Summary table
    ax_tbl = axes[1, 2]
    ax_tbl.axis("off")

    rows  = []
    cols  = ["Algorithm", "Runs", "Reward", "Coverage", "Victims"]

    algo_summary: Dict[str, Dict] = {}
    for run_name, metrics in runs_data.items():
        algo = _detect_algo(run_name)
        if algo not in algo_summary:
            algo_summary[algo] = {"rewards": [], "coverage": [], "victims": []}

        ep_r = metrics.get("ep_rewards", [])
        ep_c = metrics.get("ep_coverage", [])
        ep_v = metrics.get("ep_victims", [])

        if ep_r:
            algo_summary[algo]["rewards"].append(float(np.mean(ep_r[-100:])))
        if ep_c:
            algo_summary[algo]["coverage"].append(float(np.mean(ep_c[-100:])))
        if ep_v:
            algo_summary[algo]["victims"].append(float(np.mean(ep_v[-100:])))

    for algo in ["mappo", "masac", "matd3"]:
        if algo not in algo_summary:
            continue
        s     = algo_summary[algo]
        n     = len(s["rewards"])
        r_mu  = np.mean(s["rewards"])  if s["rewards"]  else 0
        c_mu  = np.mean(s["coverage"]) if s["coverage"] else 0
        v_mu  = np.mean(s["victims"])  if s["victims"]  else 0

        rows.append([
            ALGO_LABELS.get(algo, algo.upper()),
            str(n),
            f"{r_mu:+.1f}",
            f"{c_mu:.1f}%",
            f"{v_mu:.1f}%",
        ])

    if rows:
        tbl = ax_tbl.table(
            cellText     = rows,
            colLabels    = cols,
            cellLoc      = "center",
            loc          = "center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.2, 2.0)

        # Header color
        for j in range(len(cols)):
            tbl[0, j].set_facecolor("#37474F")
            tbl[0, j].set_text_props(color="white", fontweight="bold")

        # Row colors per algo
        algo_list = [r[0].lower() for r in rows]
        for i, algo_label in enumerate(algo_list):
            algo_key = next((k for k, v in ALGO_LABELS.items() if v == rows[i][0]), None)
            if algo_key:
                c = ALGO_COLORS.get(algo_key, "#ECEFF1")
                for j in range(len(cols)):
                    tbl[i + 1, j].set_facecolor(c + "40")  # 25% opacity hex

        ax_tbl.set_title("📋 Summary (last 100 eps)", fontsize=12, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Comparison plot saved: {save_path}")
    else:
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Plot comparison từ HuggingFace")
    parser.add_argument(
        "--runs", nargs="*", default=None,
        help="Danh sách run_names (mặc định: tất cả runs trên HF)",
    )
    parser.add_argument(
        "--local-dir", type=str, default="./hf_downloads",
        help="Thư mục lưu file download",
    )
    parser.add_argument(
        "--save", type=str, default="comparison.png",
        help="Lưu plot ra file (None = show)",
    )
    parser.add_argument(
        "--window", type=int, default=50,
        help="Smoothing window",
    )
    parser.add_argument(
        "--from-local", action="store_true",
        help="Đọc metrics từ local (không download lại)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dl   = HFDownloader()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # ── Load metrics ──────────────────────────────────────────────────────
    runs_data: Dict[str, Dict] = {}

    if args.from_local:
        # Đọc từ local JSON files đã download trước
        for jf in local_dir.rglob("metrics.json"):
            run_name = jf.parent.name
            with open(jf) as f:
                runs_data[run_name] = json.load(f)
            print(f"📂 Loaded local: {run_name}")
    else:
        # Download từ HF
        if args.runs:
            for run in args.runs:
                try:
                    metrics          = dl.download_metrics(run, str(local_dir))
                    runs_data[run]   = metrics
                    print(f"✅ {run}")
                except Exception as e:
                    print(f"⚠️  {run}: {e}")
        else:
            runs_data = dl.download_all_metrics(str(local_dir))

    if not runs_data:
        print("❌ Không có data để plot!")
        return

    print(f"\n📊 Plotting {len(runs_data)} runs...")
    for rn in runs_data:
        algo = _detect_algo(rn)
        n    = len(runs_data[rn].get("ep_rewards", []))
        print(f"   {rn:30s} [{algo:6s}] — {n} episodes")

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_comparison(
        runs_data = runs_data,
        save_path = args.save,
        window    = args.window,
    )

    # Upload comparison plot lên HF
    # if args.save and Path(args.save).exists():
    #     try:
    #         from hf_upload import HFUploader
    #         uploader = HFUploader()
    #         uploader._get_api().upload_file(
    #             path_or_fileobj = args.save,
    #             path_in_repo    = f"comparisons/{Path(args.save).name}",
    #             repo_id         = HF_REPO_ID,
    #             repo_type       = "dataset",
    #         )
    #         print(f"📤 Plot uploaded → comparisons/{Path(args.save).name}")
    #     except Exception as e:
    #         print(f"⚠️  Upload skipped: {e}")

if __name__ == "__main__":
    main()