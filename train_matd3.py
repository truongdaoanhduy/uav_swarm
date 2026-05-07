#!/usr/bin/env python3
"""
⚡ MATD3 Training — HARD Stage
Kaggle optimized: no viz during training.
Hyperparams đọc từ config.train.matd3_*.
"""

import argparse
import os
import random
import numpy as np
import torch

from config import AppConfig, STAGE_HARD
from training.algorithms.matd3.trainer import MATD3Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="MATD3 Training — HARD Stage")

    # ── Infra (không phải hyperparams) ────────────────────────────────────
    parser.add_argument("--total-episodes",      type=int,  default=3000)
    parser.add_argument("--seed",                type=int,  default=42)
    parser.add_argument("--device",              type=str,  default="auto")
    parser.add_argument("--run-name",            type=str,  default=None)
    parser.add_argument("--n-envs",              type=int,  default=1)
    parser.add_argument("--max-steps",           type=int,  default=None)
    parser.add_argument("--log-interval",        type=int,  default=50)
    parser.add_argument("--checkpoint-interval", type=int,  default=100)
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token (hoặc set env var HF_TOKEN)")
    parser.add_argument("--hf-upload",           action="store_true")
    parser.add_argument("--hf-upload-every",     type=int,  default=100)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # ── Config (hyperparams đọc từ TrainConfig.matd3_*) ───────────────────
    cfg = AppConfig()
    cfg.apply_stage(STAGE_HARD)
    cfg.env.n_uav = 4

    if args.max_steps:
        cfg.env.max_steps = args.max_steps

    HF_TOKEN = args.hf_token or os.getenv("HF_TOKEN")
    HF_REPO  = "duy95/sar-uav-results"

    run_name = args.run_name or f"matd3_s{args.seed}"

    # ── Print config ──────────────────────────────────────────────────────
    tr = cfg.train
    print(f"\n{'='*70}")
    print(f"⚡ MATD3 TRAINING — HARD STAGE (Kaggle mode)")
    print(f"{'='*70}")
    print(f"ENVIRONMENT:")
    print(f"  Map          : {cfg.env.map_size}×{cfg.env.map_size}m")
    print(f"  Max Steps    : {cfg.env.max_steps}")
    print(f"  n_envs       : {args.n_envs}")
    print(f"  seed         : {args.seed}")
    print(f"  device       : {device}")
    print(f"  run_name     : {run_name}")
    print(f"HYPERPARAMS (from TrainConfig):")
    print(f"  buffer       : {tr.matd3_buffer_capacity:,}")
    print(f"  batch        : {tr.matd3_batch_size}")
    print(f"  lr_actor     : {tr.matd3_lr_actor}")
    print(f"  lr_critic    : {tr.matd3_lr_critic}")
    print(f"  gamma/tau    : {tr.matd3_gamma}/{tr.matd3_tau}")
    print(f"  policy_delay : {tr.matd3_policy_delay}")
    print(f"  explore_noise: {tr.matd3_explore_noise}")
    print(f"  target_noise : {tr.matd3_target_noise}")
    print(f"  noise_clip   : {tr.matd3_noise_clip}")
    print(f"  warmup       : {tr.matd3_warmup_steps:,} steps")
    print(f"  actor_hidden : {tr.matd3_actor_hidden}")
    print(f"  critic_hidden: {tr.matd3_critic_hidden}")
    print(f"LOGGING:")
    print(f"  log          : {args.log_interval} eps")
    print(f"  checkpoint   : {args.checkpoint_interval} eps")
    print(f"  hf_upload    : {args.hf_upload}")
    if args.hf_upload:
        print(f"  hf_repo      : {HF_REPO}")
        print(f"  upload_every : {args.hf_upload_every} eps")
    print(f"{'='*70}\n")

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = MATD3Trainer(
        config          = cfg,
        device          = device,
        run_name        = run_name,
        n_envs          = args.n_envs,
        hf_token        = HF_TOKEN if args.hf_upload else None,
        hf_repo         = HF_REPO  if args.hf_upload else None,
        hf_upload_every = args.hf_upload_every,
    )

    trainer.train(
        total_episodes         = args.total_episodes,
        curriculum_manager     = None,
        seed                   = args.seed,
        log_every_n_eps        = args.log_interval,
        checkpoint_every_n_eps = args.checkpoint_interval,
    )

    print(f"\n{'='*70}")
    print(f"✅ MATD3 TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Episodes  : {trainer.total_episodes_done:,}")
    print(f"  Updates   : {trainer.update_count:,}")
    print(f"  Steps     : {trainer.total_steps:,}")
    if trainer.ep_rewards:
        print(f"  Reward    : {np.mean(trainer.ep_rewards):+.2f} ± {np.std(trainer.ep_rewards):.2f}")
        print(f"  Coverage  : {np.mean(trainer.ep_coverage):.1f}%")
        print(f"  Victims   : {np.mean(trainer.ep_victims):.1f}%")
    print(f"  Checkpoint: {trainer.checkpoint_dir}/checkpoint_final.pt")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()