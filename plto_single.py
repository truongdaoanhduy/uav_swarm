#!/usr/bin/env python3
"""
📊 Plot So Sánh MAPPO vs MASAC vs MATD3
Download metrics từ HuggingFace → plot → save

Usage:
    python plot_compare.py
    python plot_compare.py --runs mappo_s42 masac_s42 matd3_s42
    python plot_compare.py --local-dir ./downloaded_metrics
    python plot_compare.py --save-dir ./plots
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


def _plot_single_metric(
    runs_data:   Dict[str, Dict],
    metric_key:  str,
    ylabel:      str,
    title:       str,
    save_path:   str,
    window:      int  = 50,
    target_line: float = None,
    target_label: str = None,
    ylim:        tuple = None,
):
    """Plot 1 metric riêng lẻ và lưu file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

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
                ax.plot(sx, sm, color=color, linewidth=2.5, label=label)
            else:
                ax.plot(episodes, raw, color=color, linewidth=2.5, label=label)

        else:
            # Multi-seed: plot mean ± std
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)

            if min_len >= window:
                sm_mean = _smooth(mean, window)
                sm_std  = _smooth(std,  window)
                sx      = np.arange(window, min_len + 1)
                ax.plot(sx, sm_mean, color=color, linewidth=2.5, label=f"{label} (n={len(all_seeds)})")
                ax.fill_between(sx, sm_mean - sm_std, sm_mean + sm_std, alpha=0.2, color=color)
            else:
                ax.plot(episodes, mean, color=color, linewidth=2.5, label=f"{label} (n={len(all_seeds)})")
                ax.fill_between(episodes, mean - std, mean + std, alpha=0.2, color=color)

    if target_line is not None:
        ax.axhline(
            target_line,
            color     = "orange",
            linestyle = "--",
            linewidth = 1.5,
            alpha     = 0.8,
            label     = target_label or f"Target {target_line}",
        )

    ax.set_xlabel("Episode",  fontsize=13)
    ax.set_ylabel(ylabel,     fontsize=13)
    ax.set_title(title,       fontsize=15, fontweight="bold", pad=20)
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--")

    if ylim:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {save_path}")


def plot_sample_efficiency(runs_data: Dict[str, Dict], save_path: str, window: int = 50):
    """Plot Sample Efficiency: reward vs steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    has_data = False

    for run_name, metrics in runs_data.items():
        algo     = _detect_algo(run_name)
        color    = ALGO_COLORS.get(algo, "gray")
        label    = ALGO_LABELS.get(algo, algo.upper())
        rewards  = metrics.get("ep_rewards", [])
        n_steps  = metrics.get("total_steps", None)

        if rewards and n_steps:
            has_data = True
            n_eps        = len(rewards)
            steps_approx = np.linspace(0, n_steps, n_eps)

            sm_r = _smooth(rewards, window) if len(rewards) >= window else np.array(rewards)
            sx   = steps_approx[window - 1:] if len(rewards) >= window else steps_approx

            ax.plot(sx, sm_r, color=color, linewidth=2.5, label=label)

    if has_data:
        ax.set_xlabel("Environment Steps", fontsize=13)
        ax.set_ylabel("Reward",            fontsize=13)
        ax.set_title("⚡ Sample Efficiency", fontsize=15, fontweight="bold", pad=20)
        ax.legend(fontsize=11, loc="best", framealpha=0.9)
        ax.grid(alpha=0.3, linestyle="--")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Saved: {save_path}")
    else:
        plt.close(fig)
        print(f"⚠️  Skipped sample_efficiency (no step data)")


def plot_summary_table(runs_data: Dict[str, Dict], save_path: str):
    """Plot Summary Table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

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
            f"{r_mu:.1f}",
            f"{c_mu:.1f}%",
            f"{v_mu:.1f}%",
        ])

    if rows:
        tbl = ax.table(
            cellText     = rows,
            colLabels    = cols,
            cellLoc      = "center",
            loc          = "center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1.3, 2.5)

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
                    tbl[i + 1, j].set_facecolor(c + "40")  # 25% opacity

        ax.set_title("📋 Summary (last 100 episodes)", 
                     fontsize=15, fontweight="bold", pad=20)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Saved: {save_path}")
    else:
        plt.close(fig)
        print(f"⚠️  Skipped summary table (no data)")


def plot_all_metrics(
    runs_data:  Dict[str, Dict],
    save_dir:   str = "./plots",
    window:     int = 50,
):
    """Plot tất cả metrics ra từng file riêng."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"\n📊 Plotting metrics to: {save_path}")

    # 1. Episode Reward
    _plot_single_metric(
        runs_data   = runs_data,
        metric_key  = "ep_rewards",
        ylabel      = "Episode Reward",
        title       = " Episode Reward Comparison",
        save_path   = str(save_path / "01_episode_reward.png"),
        window      = window,
        target_line = 0,
        target_label = "Zero Baseline",
    )

    # 2. Coverage
    _plot_single_metric(
        runs_data    = runs_data,
        metric_key   = "ep_coverage",
        ylabel       = "Coverage (%)",
        title        = "Coverage Rate Comparison",
        save_path    = str(save_path / "02_coverage.png"),
        window       = window,
        target_line  = 70,
        target_label = "Target 70%",
        ylim         = (0, 100),
    )

    # 3. Victims Found
    _plot_single_metric(
        runs_data    = runs_data,
        metric_key   = "ep_victims",
        ylabel       = "Victims Found (%)",
        title        = " Victims Found Rate Comparison",
        save_path    = str(save_path / "03_victims_found.png"),
        window       = window,
        target_line  = 80,
        target_label = "Target 80%",
        ylim         = (0, 100),
    )

    # 4. Episode Length
    _plot_single_metric(
        runs_data  = runs_data,
        metric_key = "ep_lengths",
        ylabel     = "Steps",
        title      = " Episode Length Comparison",
        save_path  = str(save_path / "04_episode_length.png"),
        window     = window,
    )

    # 5. Sample Efficiency
    plot_sample_efficiency(
        runs_data = runs_data,
        save_path = str(save_path / "05_sample_efficiency.png"),
        window    = window,
    )

    # 6. Summary Table
    plot_summary_table(
        runs_data = runs_data,
        save_path = str(save_path / "06_summary_table.png"),
    )

    print(f"\n✅ All plots saved to: {save_path}")


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
        "--save-dir", type=str, default="./plots",
        help="Thư mục lưu plots",
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


def parse_args():
    parser = argparse.ArgumentParser(description="Plot comparison từ HuggingFace")
    
    # THÊM: hỗ trợ positional checkpoint paths
    parser.add_argument(
        "checkpoints", nargs="*", default=[],
        help="Đường dẫn checkpoint files (tự động tìm metrics.json trong cùng thư mục)"
    )
    
    parser.add_argument(
        "--runs", nargs="*", default=None,
        help="Danh sách run_names (mặc định: tất cả runs trên HF)",
    )
    parser.add_argument(
        "--local-dir", type=str, default="./hf_downloads",
        help="Thư mục lưu file download",
    )
    parser.add_argument(
        "--save-dir", type=str, default="./plots",
        help="Thư mục lưu plots",
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
    
    # ── Load metrics ──────────────────────────────────────────────────────
    runs_data: Dict[str, Dict] = {}

    # THÊM: Xử lý checkpoint paths
    if args.checkpoints:
        print("📂 Loading from checkpoint directories...")
        for ckpt_path in args.checkpoints:
            ckpt_path = Path(ckpt_path)
            
            # Tìm metrics.json trong cùng thư mục
            metrics_file = ckpt_path.parent / "metrics.json"
            
            if not metrics_file.exists():
                print(f"⚠️  Không tìm thấy metrics.json: {metrics_file}")
                continue
            
            run_name = ckpt_path.parent.name
            with open(metrics_file) as f:
                runs_data[run_name] = json.load(f)
            print(f"✅ Loaded: {run_name} ({metrics_file})")
    
    elif args.from_local:
        # Đọc từ local JSON files đã download trước
        local_dir = Path(args.local_dir)
        for jf in local_dir.rglob("metrics.json"):
            run_name = jf.parent.name
            with open(jf) as f:
                runs_data[run_name] = json.load(f)
            print(f"📂 Loaded local: {run_name}")
    
    else:
        # Download từ HF
        dl = HFDownloader()
        if args.runs:
            for run in args.runs:
                try:
                    metrics        = dl.download_metrics(run, args.local_dir)
                    runs_data[run] = metrics
                    print(f"✅ {run}")
                except Exception as e:
                    print(f"⚠️  {run}: {e}")
        else:
            runs_data = dl.download_all_metrics(args.local_dir)

    if not runs_data:
        print("❌ Không có data để plot!")
        return

    print(f"\n📊 Found {len(runs_data)} runs:")
    for rn in runs_data:
        algo = _detect_algo(rn)
        n    = len(runs_data[rn].get("ep_rewards", []))
        print(f"   {rn:30s} [{algo:6s}] — {n} episodes")

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_all_metrics(
        runs_data = runs_data,
        save_dir  = args.save_dir,
        window    = args.window,
    )

if __name__ == "__main__":
    main()