#!/usr/bin/env python3
"""
📊 Plot Eval Results từ HuggingFace
Download eval results từ HF → plot comparison → save

Usage:
    python plot_eval_results.py
    python plot_eval_results.py --stage extreme
    python plot_eval_results.py --stage hard --save-dir ./eval_plots
    python plot_eval_results.py --from-local --local-dir ./eval_downloads
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from huggingface_hub import hf_hub_download
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Config ────────────────────────────────────────────────────────────────────
HF_REPO_ID = "duy95/sar-uav-eval-results"

ALGO_COLORS = {
    "mappo": "#2196F3",   # Blue
    "masac": "#F44336",   # Red
    "matd3": "#4CAF50",   # Green
}

ALGO_LABELS = {
    "mappo": "MAPPO",
    "masac": "MASAC",
    "matd3": "MATD3",
}


# ── Download Functions ────────────────────────────────────────────────────────
def download_eval_results(stage: str, local_dir: str = "./eval_downloads") -> Dict[str, Dict]:
    """Download tất cả eval results cho 1 stage."""
    results = {}
    
    for algo in ["mappo", "masac", "matd3"]:
        try:
            file_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=f"eval_results/{stage}/{algo}_results.json",
                local_dir=local_dir,
            )
            
            with open(file_path) as f:
                results[algo] = json.load(f)
            
            print(f"✅ Downloaded: {algo} ({stage})")
            
        except Exception as e:
            print(f"⚠️  Failed to download {algo}: {e}")
    
    return results


def load_local_results(stage: str, local_dir: str) -> Dict[str, Dict]:
    """Load từ local files."""
    results = {}
    base_path = Path(local_dir) / HF_REPO_ID / f"eval_results/{stage}"
    
    for algo in ["mappo", "masac", "matd3"]:
        file_path = base_path / f"{algo}_results.json"
        
        if file_path.exists():
            with open(file_path) as f:
                results[algo] = json.load(f)
            print(f"📂 Loaded: {algo} ({stage})")
        else:
            print(f"⚠️  Not found: {file_path}")
    
    return results


# ── Plotting Functions ────────────────────────────────────────────────────────
def plot_episode_trajectories(results: Dict[str, Dict], save_path: str, stage: str):
    """Plot reward trajectories cho từng episode."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ["rewards", "coverages", "victim_rates"]
    titles = [" Episode Rewards", " Coverage Rate", " Victim Found Rate"]
    ylabels = ["Reward", "Coverage (%)", "Victim Rate (%)"]
    
    for idx, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        ax = axes[idx]
        
        for algo in ["mappo", "masac", "matd3"]:
            if algo not in results:
                continue
            
            data = results[algo].get(metric, [])
            if not data:
                continue
            
            # Convert to percentage if needed
            if metric in ["coverages", "victim_rates"]:
                data = [x * 100 for x in data]
            
            episodes = list(range(1, len(data) + 1))
            color = ALGO_COLORS[algo]
            label = ALGO_LABELS[algo]
            
            # Plot line + markers
            ax.plot(episodes, data, 
                   color=color, linewidth=2, 
                   marker='o', markersize=4, alpha=0.7,
                   label=label)
            
            # Mean line
            mean_val = np.mean(data)
            ax.axhline(mean_val, color=color, linestyle='--', 
                      linewidth=1, alpha=0.5)
        
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{title}\n({stage.upper()} Stage)", 
                    fontsize=12, fontweight="bold")
        ax.legend(fontsize=10, loc="best")
        ax.grid(alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_box_comparison(results: Dict[str, Dict], save_path: str, stage: str):
    """Box plot comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ["rewards", "coverages", "victim_rates"]
    titles = ["Reward Distribution", "Coverage Distribution", "Victim Rate Distribution"]
    ylabels = ["Reward", "Coverage (%)", "Victim Rate (%)"]
    
    for idx, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        ax = axes[idx]
        
        data_list = []
        labels = []
        colors = []
        
        for algo in ["mappo", "masac", "matd3"]:
            if algo not in results:
                continue
            
            data = results[algo].get(metric, [])
            if not data:
                continue
            
            # Convert to percentage
            if metric in ["coverages", "victim_rates"]:
                data = [x * 100 for x in data]
            
            data_list.append(data)
            labels.append(ALGO_LABELS[algo])
            colors.append(ALGO_COLORS[algo])
        
        # Box plot
        bp = ax.boxplot(data_list, labels=labels, patch_artist=True,
                       widths=0.6, showmeans=True)
        
        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Color other elements
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)
        
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{title}\n({stage.upper()} Stage)", 
                    fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3, axis='y', linestyle="--")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_summary_table(results: Dict[str, Dict], save_path: str, stage: str):
    """Summary statistics table."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    
    rows = []
    cols = ["Algorithm", "Episodes", "Reward", "Coverage", "Victims", "Success Rate"]
    
    for algo in ["mappo", "masac", "matd3"]:
        if algo not in results:
            continue
        
        data = results[algo]
        
        rows.append([
            ALGO_LABELS[algo],
            str(data.get("n_episodes", 0)),
            f"{data.get('reward_mean', 0):.1f} ± {data.get('reward_std', 0):.1f}",
            f"{data.get('coverage_mean', 0) * 100:.1f}% ± {data.get('coverage_std', 0) * 100:.1f}%",
            f"{data.get('victim_rate_mean', 0) * 100:.1f}% ± {data.get('victim_rate_std', 0) * 100:.1f}%",
            f"{data.get('success_rate', 0) * 100:.1f}%",
        ])
    
    if rows:
        tbl = ax.table(cellText=rows, colLabels=cols,
                      cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.2, 2.5)
        
        # Header style
        for j in range(len(cols)):
            tbl[0, j].set_facecolor("#37474F")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        
        # Row colors
        for i, row in enumerate(rows):
            algo_key = next((k for k, v in ALGO_LABELS.items() if v == row[0]), None)
            if algo_key:
                color = ALGO_COLORS[algo_key]
                for j in range(len(cols)):
                    tbl[i + 1, j].set_facecolor(color + "40")
        
        ax.set_title(f" Evaluation Summary - {stage.upper()} Stage",
                    fontsize=15, fontweight="bold", pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_success_rate_bar(results: Dict[str, Dict], save_path: str, stage: str):
    """Bar chart for success rates."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    algos = []
    success_rates = []
    colors = []
    
    for algo in ["mappo", "masac", "matd3"]:
        if algo not in results:
            continue
        
        algos.append(ALGO_LABELS[algo])
        success_rates.append(results[algo].get("success_rate", 0) * 100)
        colors.append(ALGO_COLORS[algo])
    
    bars = ax.bar(algos, success_rates, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title(f" Success Rate Comparison\n({stage.upper()} Stage)",
                fontsize=14, fontweight="bold", pad=20)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, axis='y', linestyle="--")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Plot eval results từ HuggingFace")
    parser.add_argument(
        "--stage", type=str, default="extreme",
        choices=["easy", "medium", "hard", "extreme"],
        help="Stage to plot",
    )
    parser.add_argument(
        "--local-dir", type=str, default="./eval_downloads",
        help="Local download directory",
    )
    parser.add_argument(
        "--save-dir", type=str, default="./eval_plots",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--from-local", action="store_true",
        help="Load from local files (skip download)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    save_dir = Path(args.save_dir) / args.stage
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n📊 Loading eval results for {args.stage.upper()} stage...")
    
    if args.from_local:
        results = load_local_results(args.stage, args.local_dir)
    else:
        results = download_eval_results(args.stage, args.local_dir)
    
    if not results:
        print("❌ No data to plot!")
        return
    
    print(f"\n📈 Plotting {len(results)} algorithms...")
    
    # Generate plots
    plot_episode_trajectories(
        results, 
        str(save_dir / "01_trajectories.png"),
        args.stage
    )
    
    plot_box_comparison(
        results,
        str(save_dir / "02_distributions.png"),
        args.stage
    )
    
    plot_summary_table(
        results,
        str(save_dir / "03_summary.png"),
        args.stage
    )
    
    plot_success_rate_bar(
        results,
        str(save_dir / "04_success_rate.png"),
        args.stage
    )
    
    print(f"\n✅ All plots saved to: {save_dir}")


if __name__ == "__main__":
    main()