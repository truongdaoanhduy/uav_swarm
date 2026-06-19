#!/usr/bin/env python3
"""
📊 Plot So Sánh các runs (mỗi run = 1 đường riêng biệt)

Usage:
    python plto_single.py run1/metrics.json run2/metrics.json --save-dir ./plots
    python plto_single.py masac_llm/metrics.json masac_s42/metrics.json --save-dir ./plots_llm
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List


# ── Palette: mỗi run 1 màu khác nhau ─────────────────────────────────────────
PALETTE = [
    "#2196F3",  # Blue
    "#F44336",  # Red
    "#4CAF50",  # Green
    "#FF9800",  # Orange
    "#9C27B0",  # Purple
    "#00BCD4",  # Cyan
    "#795548",  # Brown
    "#E91E63",  # Pink
]


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
    runs_data:    Dict[str, Dict],
    metric_key:   str,
    ylabel:       str,
    title:        str,
    save_path:    str,
    window:       int   = 50,
    target_line:  float = None,
    target_label: str   = None,
    ylim:         tuple = None,
):
    """Plot 1 metric - MỖI RUN LÀ 1 ĐƯỜNG RIÊNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (run_name, metrics) in enumerate(runs_data.items()):
        data = metrics.get(metric_key, [])
        if not data:
            continue

        color = PALETTE[idx % len(PALETTE)]
        label = run_name  # ← Dùng run_name để phân biệt rõ ràng!

        min_len  = len(data)
        episodes = np.arange(1, min_len + 1)
        raw      = np.array(data)

        # Raw mờ phía sau
        ax.plot(episodes, raw, alpha=0.12, color=color, linewidth=0.5)

        # Đường smoothed chính
        if min_len >= window:
            sm = _smooth(raw, window)
            sx = np.arange(window, min_len + 1)
            ax.plot(sx, sm, color=color, linewidth=2.5, label=label)
        else:
            ax.plot(episodes, raw, color=color, linewidth=2.5, label=label)

    if target_line is not None:
        ax.axhline(
            target_line, color="gray", linestyle="--",
            linewidth=1.5, alpha=0.7,
            label=target_label or f"Target {target_line}",
        )

    ax.set_xlabel("Episode", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--")

    if ylim:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {save_path}")


def plot_sample_efficiency(runs_data: Dict[str, Dict], save_path: str, window: int = 50):
    """Plot Sample Efficiency - mỗi run 1 đường."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    has_data = False

    for idx, (run_name, metrics) in enumerate(runs_data.items()):
        color   = PALETTE[idx % len(PALETTE)]
        rewards = metrics.get("ep_rewards", [])
        n_steps = metrics.get("total_steps", None)

        if rewards and n_steps:
            has_data = True
            n_eps        = len(rewards)
            steps_approx = np.linspace(0, n_steps, n_eps)

            sm_r = _smooth(rewards, window) if len(rewards) >= window else np.array(rewards)
            sx   = steps_approx[window - 1:] if len(rewards) >= window else steps_approx

            ax.plot(sx, sm_r, color=color, linewidth=2.5, label=run_name)

    if has_data:
        ax.set_xlabel("Environment Steps", fontsize=13)
        ax.set_ylabel("Reward", fontsize=13)
        ax.set_title("Sample Efficiency", fontsize=15, fontweight="bold", pad=20)
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
    """Plot Summary Table - mỗi run 1 hàng."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    rows = []
    cols = ["Run", "Episodes", "Reward", "Coverage", "Victims"]

    for run_name, metrics in runs_data.items():
        ep_r = metrics.get("ep_rewards", [])
        ep_c = metrics.get("ep_coverage", [])
        ep_v = metrics.get("ep_victims", [])

        n    = len(ep_r)
        r_mu = float(np.mean(ep_r[-100:])) if ep_r else 0
        c_mu = float(np.mean(ep_c[-100:])) if ep_c else 0
        v_mu = float(np.mean(ep_v[-100:])) if ep_v else 0

        rows.append([
            run_name, str(n),
            f"{r_mu:.1f}", f"{c_mu:.1f}%", f"{v_mu:.1f}%",
        ])

    if rows:
        tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1.3, 2.5)

        # Header
        for j in range(len(cols)):
            tbl[0, j].set_facecolor("#37474F")
            tbl[0, j].set_text_props(color="white", fontweight="bold")

        # Mỗi hàng 1 màu theo palette
        for i, row in enumerate(rows):
            c = PALETTE[i % len(PALETTE)]
            for j in range(len(cols)):
                tbl[i + 1, j].set_facecolor(c + "30")  # nhạt

        ax.set_title("Summary (last 100 episodes)", fontsize=15, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Saved: {save_path}")
    else:
        plt.close(fig)
        print(f"⚠️  Skipped summary table (no data)")


def plot_all_metrics(runs_data: Dict[str, Dict], save_dir: str = "./plots", window: int = 50):
    """Plot tất cả metrics."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"\n📊 Plotting metrics to: {save_path}")

    _plot_single_metric(
        runs_data, "ep_rewards", "Episode Reward", "Episode Reward Comparison",
        str(save_path / "01_episode_reward.png"), window,
        target_line=0, target_label="Zero Baseline",
    )
    _plot_single_metric(
        runs_data, "ep_coverage", "Coverage (%)", "Coverage Rate Comparison",
        str(save_path / "02_coverage.png"), window,
        target_line=70, target_label="Target 70%", ylim=(0, 100),
    )
    _plot_single_metric(
        runs_data, "ep_victims", "Victims Found (%)", "Victims Found Rate Comparison",
        str(save_path / "03_victims_found.png"), window,
        target_line=80, target_label="Target 80%", ylim=(0, 100),
    )
    _plot_single_metric(
        runs_data, "ep_lengths", "Steps", "Episode Length Comparison",
        str(save_path / "04_episode_length.png"), window,
    )
    plot_sample_efficiency(runs_data, str(save_path / "05_sample_efficiency.png"), window)
    plot_summary_table(runs_data, str(save_path / "06_summary_table.png"))

    print(f"\n✅ All plots saved to: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Plot comparison (mỗi run 1 đường)")
    parser.add_argument(
        "paths", nargs="*", default=[],
        help="Đường dẫn metrics.json / checkpoint (.pt) / thư mục",
    )
    parser.add_argument("--runs", nargs="*", default=None, help="run_names trên HF")
    parser.add_argument("--local-dir", type=str, default="./hf_downloads")
    parser.add_argument("--save-dir", type=str, default="./plots")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--from-local", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    runs_data: Dict[str, Dict] = {}

    if args.paths:
        print("📂 Loading from paths...")
        for path_str in args.paths:
            path = Path(path_str)

            if path.suffix == ".pt":
                metrics_file = path.parent / "metrics.json"
                run_name = path.parent.name
            elif path.name == "metrics.json":
                metrics_file = path
                run_name = path.parent.name
            elif path.is_dir():
                metrics_file = path / "metrics.json"
                run_name = path.name
            else:
                print(f"⚠️  Bỏ qua: {path}")
                continue

            if not metrics_file.exists():
                print(f"❌ Không tìm thấy: {metrics_file}")
                continue

            with open(metrics_file) as f:
                runs_data[run_name] = json.load(f)
            print(f"✅ Loaded: {run_name} ({metrics_file})")

    elif args.from_local:
        print("📂 Loading from local...")
        for jf in Path(args.local_dir).rglob("metrics.json"):
            run_name = jf.parent.name
            with open(jf) as f:
                runs_data[run_name] = json.load(f)
            print(f"✅ Loaded: {run_name}")

    else:
        print("📥 Downloading from HuggingFace...")
        from hf_upload import HFDownloader
        dl = HFDownloader()
        if args.runs:
            for run in args.runs:
                try:
                    runs_data[run] = dl.download_metrics(run, args.local_dir)
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
        n = len(runs_data[rn].get("ep_rewards", []))
        print(f"   {rn:30s} [{algo:6s}] — {n} episodes")

    plot_all_metrics(runs_data=runs_data, save_dir=args.save_dir, window=args.window)


if __name__ == "__main__":
    main()