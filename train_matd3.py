#!/usr/bin/env python3
"""
🎯 MATD3 Training — HARD Stage
Supports Baseline v4.0 and LLM-generated rewards
"""

import argparse
import os, random, numpy as np, torch

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


def apply_matd3_noise_config(
    cfg: AppConfig,
    explore_noise: float | None = None,
    target_noise: float | None = None,
    noise_clip: float | None = None,
) -> None:
    """Adjust MATD3 exploration/target smoothing noise from CLI or experiments."""
    if explore_noise is not None:
        cfg.train.matd3_explore_noise = max(0.0, float(explore_noise))
    if target_noise is not None:
        cfg.train.matd3_target_noise = max(0.0, float(target_noise))
    if noise_clip is not None:
        cfg.train.matd3_noise_clip = max(0.0, float(noise_clip))


def parse_args():
    parser = argparse.ArgumentParser(description="MATD3 Training")

    parser.add_argument("--total-episodes",      type=int,   default=3000)
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--device",              type=str,   default="auto")
    parser.add_argument("--run-name",            type=str,   default=None)
    parser.add_argument("--n-envs",              type=int,   default=1)
    parser.add_argument("--max-steps",           type=int,   default=None)
    parser.add_argument("--n-victims-min",       type=int,   default=None)
    parser.add_argument("--n-victims-max",       type=int,   default=None)
    parser.add_argument("--coverage-threshold",  type=float, default=None)
    parser.add_argument("--explore-noise",       type=float, default=None)
    parser.add_argument("--target-noise",        type=float, default=None)
    parser.add_argument("--noise-clip",          type=float, default=None)
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
    set_seed(args.seed)

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # ── Config ───────────────────────────────────────────────────────────────
    cfg = AppConfig()
    cfg.apply_stage(STAGE_HARD)
    cfg.env.n_uav = 4

    if args.max_steps:
        cfg.env.max_steps = args.max_steps
    if args.n_victims_min is not None:
        cfg.victim.n_victims_min = args.n_victims_min
    if args.n_victims_max is not None:
        cfg.victim.n_victims_max = args.n_victims_max
    if cfg.victim.n_victims_min > cfg.victim.n_victims_max:
        raise ValueError("--n-victims-min must be <= --n-victims-max")
    if args.coverage_threshold is not None:
        cfg.env.done_coverage_threshold = float(np.clip(args.coverage_threshold, 0.0, 1.0))
    apply_matd3_noise_config(
        cfg,
        explore_noise=args.explore_noise,
        target_noise=args.target_noise,
        noise_clip=args.noise_clip,
    )

    # ── Run name + HF ────────────────────────────────────────────────────────
    # ✅ Định nghĩa TRƯỚC khi dùng
    run_name = args.run_name or f"matd3_s{args.seed}"
    HF_TOKEN = args.hf_token or os.getenv("HF_TOKEN")
    HF_REPO  = "duy95/sar-uav-results"

    # ── LLM Reward ───────────────────────────────────────────────────────────
    # ✅ Chỉ giữ PATH — object load bên trong worker (safe với spawn)
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
    print(f"🎯 MATD3 TRAINING — HARD STAGE")
    print(f"   Reward: {'LLM' if llm_reward_path else 'Baseline v4.0'}")
    print(f"{'='*70}")
    print(f"ENVIRONMENT:")
    print(f"  Map       : {cfg.env.map_size}×{cfg.env.map_size}m")
    print(f"  Max Steps : {cfg.env.max_steps}")
    print(f"  Victims   : {cfg.victim.n_victims_min}-{cfg.victim.n_victims_max}")
    print(f"  Done Cov  : {cfg.env.done_coverage_threshold:.0%}")
    print(f"  n_envs    : {args.n_envs}")
    print(f"  seed      : {args.seed}")
    print(f"  device    : {device}")
    print(f"  run_name  : {run_name}")
    print(f"")
    print(f"HYPERPARAMS (from TrainConfig):")
    print(f"  buffer    : {tr.matd3_buffer_capacity:,}")
    print(f"  batch     : {tr.matd3_batch_size}")
    print(f"  lr_actor  : {tr.matd3_lr_actor}")
    print(f"  lr_critic : {tr.matd3_lr_critic}")
    print(f"  gamma/tau : {tr.matd3_gamma}/{tr.matd3_tau}")
    print(f"  policy_delay : {tr.matd3_policy_delay}")
    print(f"  explore_noise: {tr.matd3_explore_noise}")
    print(f"  target_noise : {tr.matd3_target_noise}")
    print(f"  noise_clip   : {tr.matd3_noise_clip}")
    print(f"  warmup    : {tr.matd3_warmup_steps:,} steps")
    print(f"  actor_h   : {tr.matd3_actor_hidden}")
    print(f"  critic_h  : {tr.matd3_critic_hidden}")
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
    trainer = MATD3Trainer(
        config          = cfg,
        device          = device,
        run_name        = run_name,
        n_envs          = args.n_envs,
        llm_reward_path = llm_reward_path,                      # ← PATH
        hf_token        = HF_TOKEN if args.hf_upload else None,
        hf_repo         = HF_REPO  if args.hf_upload else None,
        hf_upload_every = args.hf_upload_every,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    trainer.train(
        total_episodes         = args.total_episodes,
        curriculum_manager     = None,
        seed                   = args.seed,
        log_every_n_eps        = args.log_interval,
        checkpoint_every_n_eps = args.checkpoint_interval,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
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
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
