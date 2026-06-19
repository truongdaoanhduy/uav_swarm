#!/usr/bin/env python3
"""
🔥 MASAC Training — HARD Stage
Supports Baseline v4.0 and LLM-generated rewards
"""

import argparse
import os, random, numpy as np, torch

from config import AppConfig, STAGE_HARD
from training.algorithms.masac.trainer import MASACTrainer

# train_masac.py

import argparse
import os
import random
import numpy as np
import torch

from config import AppConfig, STAGE_HARD
from training.algorithms.masac.trainer import MASACTrainer


def set_seed(seed: int) -> None:
    """
    ✅ FIX: Set seed đầy đủ và đúng thứ tự.
    
    Thứ tự quan trọng:
        1. CUBLAS env var TRƯỚC khi import torch (hoặc ít nhất trước cuda ops)
        2. Python random
        3. NumPy  
        4. PyTorch CPU
        5. PyTorch CUDA
        6. CUDNN flags
        7. use_deterministic_algorithms CUỐI CÙNG
    """
    # ✅ 1. Env vars TRƯỚC
    os.environ["PYTHONHASHSEED"]          = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # ← ":16:8" có thể thiếu bộ nhớ

    # ✅ 2. Python random
    random.seed(seed)

    # ✅ 3. NumPy
    np.random.seed(seed)

    # ✅ 4. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ✅ 5. CUDNN - benchmark=False BẮT BUỘC cho reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False  # ← True sẽ chọn algo khác nhau mỗi lần

    # ✅ 6. Deterministic algorithms - warn_only thay vì try/except
    # warn_only=True: warning thay vì crash khi gặp non-deterministic op
    # Không dùng try/except vì nó che giấu lỗi thật
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"[Seed] ✅ seed={seed} set toàn diện")
    print(f"[Seed]    PYTHONHASHSEED={os.environ['PYTHONHASHSEED']}")
    print(f"[Seed]    CUBLAS={os.environ['CUBLAS_WORKSPACE_CONFIG']}")
    print(f"[Seed]    cudnn.benchmark=False, deterministic=True")



def parse_args():
    parser = argparse.ArgumentParser(description="MASAC Training")
    parser.add_argument("--total-episodes",      type=int,   default=3000)
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--device",              type=str,   default="auto")
    parser.add_argument("--run-name",            type=str,   default=None)
    parser.add_argument("--n-envs",              type=int,   default=1)
    parser.add_argument("--max-steps",           type=int,   default=None)
    parser.add_argument("--log-interval",        type=int,   default=50)
    parser.add_argument("--checkpoint-interval", type=int,   default=100)
    parser.add_argument("--hf-token",            type=str,   default=None)
    parser.add_argument("--hf-upload",           action="store_true")
    parser.add_argument("--hf-upload-every",     type=int,   default=100)
    parser.add_argument(
        "--llm-reward", type=str, default=None,
        help="Path to LLM reward file (e.g., llm_reward_generated.py)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # ✅ FIX: set_seed TRƯỚC MỌI THỨ KHÁC
    set_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # ✅ FIX: Ghi seed vào config để tất cả components dùng chung
    cfg = AppConfig()
    cfg.apply_stage(STAGE_HARD)
    cfg.env.n_uav       = 4
    cfg.env.global_seed = args.seed  # ← base_env dùng để tính episode seeds
    cfg.env.eval_seed   = args.seed  # ← eval mode dùng

    if args.max_steps:
        cfg.env.max_steps = args.max_steps

    run_name        = args.run_name or f"masac_s{args.seed}"
    HF_TOKEN        = args.hf_token or os.getenv("HF_TOKEN")
    HF_REPO         = "duy95/sar-uav-results"
    llm_reward_path = args.llm_reward

    if llm_reward_path:
        print(f"\n{'='*70}")
        print(f"🤖 LLM REWARD: {llm_reward_path}")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'='*70}")
        print(f"📊 BASELINE REWARD v4.0")
        print(f"{'='*70}\n")

    # ── Print config ─────────────────────────────────────────────────────────
    tr = cfg.train
    print(f"{'='*70}")
    print(f"🔥 MASAC TRAINING — HARD STAGE")
    print(f"   Reward: {'LLM' if llm_reward_path else 'Baseline v4.0'}")
    print(f"{'='*70}")
    print(f"ENVIRONMENT:")
    print(f"  Map       : {cfg.env.map_size}×{cfg.env.map_size}m")
    print(f"  Max Steps : {cfg.env.max_steps}")
    print(f"  n_envs    : {args.n_envs}")
    print(f"  seed      : {args.seed}")
    print(f"  device    : {device}")
    print(f"  run_name  : {run_name}")
    print(f"")
    print(f"HYPERPARAMS (from TrainConfig):")
    print(f"  buffer    : {tr.masac_buffer_capacity:,}")
    print(f"  batch     : {tr.masac_batch_size}")
    print(f"  lr_actor  : {tr.masac_lr_actor}")
    print(f"  lr_critic : {tr.masac_lr_critic}")
    print(f"  lr_alpha  : {tr.masac_lr_alpha}")
    print(f"  gamma/tau : {tr.masac_gamma}/{tr.masac_tau}")
    print(f"  auto_α    : {tr.masac_auto_alpha} | α_init: {tr.masac_alpha_init}")
    print(f"  warmup    : {tr.masac_warmup_steps:,} steps")
    print(f"  actor_h   : {tr.masac_actor_hidden}")
    print(f"  critic_h  : {tr.masac_critic_hidden}")
    print(f"")
    print(f"LOGGING:")
    print(f"  log       : {args.log_interval} eps")
    print(f"  checkpoint: {args.checkpoint_interval} eps")
    print(f"  hf_upload : {args.hf_upload}")
    if args.hf_upload:
        print(f"  hf_upload_every: {args.hf_upload_every} eps")
    print(f"{'='*70}\n")

    # ── Create Trainer ───────────────────────────────────────────────────────
    # ✅ Tạo 1 lần, truyền llm_reward_path
    trainer = MASACTrainer(
        config          = cfg,
        device          = device,
        run_name        = run_name,
        n_envs          = args.n_envs,
        llm_reward_path = llm_reward_path,
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

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"✅ MASAC TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Episodes  : {trainer.total_episodes_done:,}")
    print(f"  Updates   : {trainer.update_count:,}")
    print(f"  Steps     : {trainer.total_steps:,}")
    if trainer.ep_rewards:
        print(f"  Reward    : {np.mean(trainer.ep_rewards):+.2f} ± {np.std(trainer.ep_rewards):.2f}")
        print(f"  Coverage  : {np.mean(trainer.ep_coverage):.1f}%")
        print(f"  Victims   : {np.mean(trainer.ep_victims):.1f}%")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()