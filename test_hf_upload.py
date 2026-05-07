#!/usr/bin/env python3
"""
test_hf_flow.py
Test flow: Train 10 eps → Upload HF → Download → Plot

Usage:
    python test_hf_flow.py               # Train 10 eps + upload + download + plot
    python test_hf_flow.py --skip-train  # Chỉ download + plot (đã upload rồi)
    python test_hf_flow.py --only-plot   # Chỉ plot từ local
"""

import argparse
import os
import time
import json
import numpy as np
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes",   type=int, default=10,
                   help="Số episodes train thử")
    p.add_argument("--run-name",     type=str, default=None,
                   help="Tên run (auto nếu None)")
    p.add_argument("--skip-train",   action="store_true",
                   help="Bỏ qua train, chỉ download + plot")
    p.add_argument("--only-plot",    action="store_true",
                   help="Chỉ plot từ local (không download)")
    p.add_argument("--local-dir",    type=str, default="./hf_downloads",
                   help="Thư mục lưu download")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--n-envs",       type=int, default=1)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: TRAIN + UPLOAD
# ══════════════════════════════════════════════════════════════════════════════

def step1_train_and_upload(args) -> str:
    """
    Train N episodes với MAPPO.
    Upload checkpoint + metrics lên HF sau mỗi episode.
    Trả về run_name.
    """
    import torch
    from config import AppConfig, STAGE_HARD
    from hf_upload import HFUploader, HF_REPO_ID

    run_name = args.run_name or f"test_mappo_s{args.seed}_{time.strftime('%H%M%S')}"

    print(f"\n{'='*60}")
    print(f"STEP 1: TRAIN {args.n_episodes} EPISODES")
    print(f"{'='*60}")
    print(f"  run_name : {run_name}")
    print(f"  n_envs   : {args.n_envs}")
    print(f"  seed     : {args.seed}")

    # ── Config ────────────────────────────────────────────────────────────
    cfg = AppConfig()
    cfg.apply_stage(STAGE_HARD)
    cfg.env.n_uav = 4

    # Rollout ngắn cho test
    cfg.train.mappo_rollout_length = 128
    cfg.train.mappo_batch_size     = 64
    cfg.train.mappo_n_epochs       = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Trainer ───────────────────────────────────────────────────────────
    from training.algorithms.mappo.trainer import MAPPOTrainer

    trainer = MAPPOTrainer(
        config   = cfg,
        device   = device,
        run_name = run_name,
        n_envs   = args.n_envs,
    )

    # ── HF Uploader ───────────────────────────────────────────────────────
    uploader = HFUploader()

    # ── Train với upload sớm ──────────────────────────────────────────────
    print(f"\n🚀 Training {args.n_episodes} episodes...\n")

    trainer.train(
        total_episodes         = args.n_episodes,
        curriculum_manager     = None,
        seed                   = args.seed,
        log_every_n_eps        = 5,       # Log mỗi 5 eps
        viz_every_n_eps        = 999999,  # Không viz
        checkpoint_every_n_eps = 5,       # Checkpoint mỗi 5 eps
    )

    # ── Upload final lên HF ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"📤 Uploading to HuggingFace...")
    print(f"{'='*60}")

    ckpt_path  = trainer.checkpoint_dir / "checkpoint_final.pt"
    plot_path  = trainer.output_dir / "training_curves.png"

    # Plot curves
    trainer.plot_training_curves(save_path=str(plot_path))

    # Build full metrics
    metrics = {
        "algo":           "mappo",
        "run_name":       run_name,
        "seed":           args.seed,
        "total_episodes": trainer.total_episodes_done,
        "total_steps":    trainer.total_steps,
        "ep_rewards":     list(trainer.ep_rewards),
        "ep_coverage":    list(trainer.ep_coverage),
        "ep_victims":     list(trainer.ep_victims),
        "ep_lengths":     list(trainer.ep_lengths),
        "mean_reward":    float(np.mean(trainer.ep_rewards))  if trainer.ep_rewards  else 0,
        "mean_coverage":  float(np.mean(trainer.ep_coverage)) if trainer.ep_coverage else 0,
        "mean_victims":   float(np.mean(trainer.ep_victims))  if trainer.ep_victims  else 0,
    }

    uploader.upload_final(
        run_name        = run_name,
        checkpoint_path = ckpt_path,
        metrics         = metrics,
        plot_path       = plot_path,
    )

    print(f"\n✅ STEP 1 DONE!")
    print(f"   run_name : {run_name}")
    print(f"   HF URL   : https://huggingface.co/datasets/duy95/sar-uav-results/tree/main/{run_name}")

    return run_name


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def step2_download(run_name: str, local_dir: str) -> dict:
    """
    Download metrics.json + checkpoint từ HF về local.
    Trả về metrics dict.
    """
    from hf_upload import HFDownloader

    print(f"\n{'='*60}")
    print(f"STEP 2: DOWNLOAD FROM HUGGINGFACE")
    print(f"{'='*60}")
    print(f"  run_name  : {run_name}")
    print(f"  local_dir : {local_dir}")

    dl = HFDownloader()

    # List tất cả runs
    print(f"\n📋 All runs on HF:")
    runs = dl.list_runs()
    for r in runs:
        marker = " ← target" if r == run_name else ""
        print(f"   {r}{marker}")

    # Download metrics
    print(f"\n📥 Downloading metrics...")
    metrics = dl.download_metrics(run_name, local_dir)

    # Download checkpoint
    print(f"\n📥 Downloading checkpoint...")
    ckpt_local = dl.download_checkpoint(
        run_name  = run_name,
        filename  = "checkpoint_final.pt",
        local_dir = local_dir,
    )

    print(f"\n✅ STEP 2 DONE!")
    print(f"   metrics.json  : {Path(local_dir) / run_name / 'metrics.json'}")
    print(f"   checkpoint.pt : {ckpt_local}")

    # Summary metrics
    print(f"\n📊 Metrics summary:")
    print(f"   algo          : {metrics.get('algo', 'N/A')}")
    print(f"   total_episodes: {metrics.get('total_episodes', 0)}")
    print(f"   mean_reward   : {metrics.get('mean_reward', 0):+.2f}")
    print(f"   mean_coverage : {metrics.get('mean_coverage', 0):.1f}%")
    print(f"   uploaded_at   : {metrics.get('uploaded_at', 'N/A')}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: PLOT
# ══════════════════════════════════════════════════════════════════════════════

def step3_plot(metrics_dict: dict, save_dir: str = "."):
    """
    Plot training curves từ metrics dict.
    Lưu ra PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"\n{'='*60}")
    print(f"STEP 3: PLOT TRAINING CURVES")
    print(f"{'='*60}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for run_name, metrics in metrics_dict.items():
        algo     = metrics.get("algo", "unknown")
        ep_r     = metrics.get("ep_rewards",  [])
        ep_c     = metrics.get("ep_coverage", [])
        ep_v     = metrics.get("ep_victims",  [])
        ep_l     = metrics.get("ep_lengths",  [])
        n        = len(ep_r)

        if n == 0:
            print(f"  ⚠️  {run_name}: no data")
            continue

        episodes = list(range(1, n + 1))
        window   = max(2, min(5, n // 2))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Training Curves — {run_name} ({algo.upper()})",
            fontsize=14, fontweight="bold"
        )

        def _plot(ax, data, color, ylabel, title, target=None):
            ax.plot(episodes, data, alpha=0.3, color=color, linewidth=0.8)
            if len(data) >= window:
                sm = np.convolve(data, np.ones(window)/window, mode="valid")
                sx = list(range(window, len(data)+1))
                ax.plot(sx, sm, color=color, linewidth=2.5,
                        label=f"MA({window})")
            if target is not None:
                ax.axhline(target, color="orange", linestyle="--",
                           linewidth=1.0, alpha=0.8,
                           label=f"Target {target}")
            ax.set_xlabel("Episode", fontsize=10)
            ax.set_ylabel(ylabel,    fontsize=10)
            ax.set_title(title,      fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_facecolor("#F9F9F9")

        _plot(axes[0,0], ep_r, "#2196F3", "Reward",       "Episode Reward",     target=0)
        _plot(axes[0,1], ep_c, "#4CAF50", "Coverage (%)", "Coverage Rate",      target=70)
        _plot(axes[1,0], ep_v, "#FF9800", "Victims (%)",  "Victims Found Rate", target=80)
        _plot(axes[1,1], ep_l, "#9C27B0", "Steps",        "Episode Length")

        # Stats box
        stats_text = (
            f"Episodes : {n}\n"
            f"Reward   : {np.mean(ep_r):+.1f} ± {np.std(ep_r):.1f}\n"
            f"Coverage : {np.mean(ep_c):.1f}%\n"
            f"Victims  : {np.mean(ep_v):.1f}%\n"
            f"Uploaded : {metrics.get('uploaded_at', 'N/A')}"
        )
        axes[1,1].text(
            0.98, 0.97, stats_text,
            transform   = axes[1,1].transAxes,
            fontsize    = 8,
            va          = "top", ha = "right",
            bbox        = dict(boxstyle="round,pad=0.4",
                               facecolor="white", alpha=0.9,
                               edgecolor="#9C27B0"),
        )

        plt.tight_layout()

        out = save_dir / f"{run_name}_curves.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✅ Plot saved: {out.resolve()}")

    print(f"\n✅ STEP 3 DONE!")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args      = parse_args()
    local_dir = args.local_dir
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# HuggingFace Flow Test")
    print(f"# Train → Upload → Download → Plot")
    print(f"{'#'*60}")

    # ── Step 1: Train + Upload ────────────────────────────────────────────
    if args.only_plot:
        # Đọc từ local JSON
        metrics_dict = {}
        for jf in Path(local_dir).rglob("metrics.json"):
            run_name = jf.parent.name
            with open(jf) as f:
                metrics_dict[run_name] = json.load(f)
            print(f"📂 Loaded local: {run_name}")
        run_name = list(metrics_dict.keys())[0] if metrics_dict else "unknown"

    elif args.skip_train:
        # Chỉ download + plot
        from hf_upload import HFDownloader
        dl   = HFDownloader()
        runs = dl.list_runs()
        print(f"\n📋 Runs on HF: {runs}")

        if not runs:
            print("❌ Không có run nào trên HF!")
            return

        # Lấy run gần nhất hoặc theo tên
        run_name = args.run_name or runs[-1]
        print(f"👉 Using run: {run_name}")

        metrics      = step2_download(run_name, local_dir)
        metrics_dict = {run_name: metrics}

    else:
        # Full flow: train → upload → download → plot
        run_name     = step1_train_and_upload(args)
        metrics      = step2_download(run_name, local_dir)
        metrics_dict = {run_name: metrics}

    # ── Step 3: Plot ──────────────────────────────────────────────────────
    step3_plot(metrics_dict, save_dir=local_dir)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"# DONE!")
    print(f"{'#'*60}")
    print(f"  HF URL   : https://huggingface.co/datasets/duy95/sar-uav-results")
    print(f"  Local    : {Path(local_dir).resolve()}")
    print(f"  Plot     : {Path(local_dir).resolve()}/{run_name}_curves.png")
    print(f"")
    print(f"  Xem plot:")
    print(f"    → {Path(local_dir).resolve()}/{run_name}_curves.png")
    print(f"")
    print(f"  Download checkpoint:")
    print(f"    python test_hf_flow.py --skip-train --run-name {run_name}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()