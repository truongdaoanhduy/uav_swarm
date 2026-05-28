#!/usr/bin/env python3
"""
🚁 MAPPO Training — HARD Stage Focus
Supports Baseline v4.0 and LLM-generated rewards
"""

import argparse
import torch
from dataclasses import dataclass
import os, random, numpy as np

from config import AppConfig, STAGE_HARD
from training.algorithms.mappo.trainer import MAPPOTrainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


@dataclass
class AutoConfig:
    rollout_length:  int
    buffer_capacity: int
    batch_size:      int
    safety_margin:   float


def auto_compute_config(
    max_steps, n_envs, n_uav,
    batch_size_hint=None, safety_factor=1.5
):
    min_rollout = int(max_steps * safety_factor)
    rollout_length = ((min_rollout + 63) // 64) * 64
    buffer_capacity = rollout_length * n_envs
    total_samples = buffer_capacity * n_uav

    if batch_size_hint is None:
        batch_size = max(256, buffer_capacity // 16)
        batch_size = ((batch_size + 255) // 256) * 256
    else:
        batch_size = batch_size_hint

    batch_size = min(batch_size, total_samples // 2)

    return AutoConfig(
        rollout_length=rollout_length,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        safety_margin=buffer_capacity / (max_steps * n_envs),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="MAPPO Training")
    
    parser.add_argument("--total-episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--map-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--safety-factor", type=float, default=1.5)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--lr-actor", type=float, default=None)
    parser.add_argument("--lr-critic", type=float, default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--hf-upload", action="store_true")
    parser.add_argument("--hf-upload-every", type=int, default=100)
    
    # ← LLM reward support
    parser.add_argument("--llm-reward", type=str, default=None,
                        help="Path to LLM reward file (e.g., llm_reward_generated.py)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device

    cfg = AppConfig()
    cfg.apply_stage(STAGE_HARD)
    cfg.env.n_uav = 4

    if args.max_steps:
        cfg.env.max_steps = args.max_steps
    if args.map_size:
        cfg.env.map_size = args.map_size
        cfg.env.grid_size = args.map_size

    auto_cfg = auto_compute_config(
        cfg.env.max_steps, args.n_envs, cfg.env.n_uav,
        args.batch_size, args.safety_factor
    )

    cfg.train.mappo_rollout_length = auto_cfg.rollout_length
    cfg.train.mappo_batch_size = auto_cfg.batch_size

    if args.n_epochs:
        cfg.train.mappo_n_epochs = args.n_epochs
    if args.lr_actor:
        cfg.train.mappo_lr_actor = args.lr_actor
    if args.lr_critic:
        cfg.train.mappo_lr_critic = args.lr_critic

    # ─────────────────────────────────────────────────────────
    # ← LOAD LLM REWARD
    # ─────────────────────────────────────────────────────────
    llm_reward = None
    if args.llm_reward:
        from rewards.llm_reward import load_llm_reward
        llm_reward = load_llm_reward(args.llm_reward, cfg)
        print(f"\n{'='*70}")
        print(f"🤖 LLM REWARD LOADED")
        print(f"   File: {args.llm_reward}")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'='*70}")
        print(f"📊 BASELINE REWARD v4.0")
        print(f"{'='*70}\n")

    # ─────────────────────────────────────────────────────────
    # ← PRINT CONFIG
    # ─────────────────────────────────────────────────────────
    avg_ep_len = cfg.env.max_steps * 0.85
    steps_per_update = auto_cfg.rollout_length * args.n_envs
    eps_per_update = steps_per_update / avg_ep_len
    est_updates = max(1, int(args.total_episodes / eps_per_update))

    print(f"{'='*70}")
    print(f"🚁 MAPPO TRAINING — HARD STAGE")
    print(f"   Reward: {'LLM' if llm_reward else 'Baseline v4.0'}")
    print(f"{'='*70}")
    print(f"ENVIRONMENT:")
    print(f"  Map Size:            {cfg.env.map_size}×{cfg.env.map_size}m")
    print(f"  Max Steps:           {cfg.env.max_steps}")
    print(f"  n_envs:              {args.n_envs}")
    print(f"  n_uav:               {cfg.env.n_uav}")
    print(f"  Victims:             {cfg.victim.n_victims_min}-{cfg.victim.n_victims_max}")
    print(f"  Debris:              {cfg.obstacle.n_debris}")
    print(f"")
    print(f"AUTO-CONFIG:")
    print(f"  Rollout Length:      {auto_cfg.rollout_length:,}")
    print(f"  Buffer Capacity:     {auto_cfg.buffer_capacity:,}")
    print(f"  Batch Size:          {auto_cfg.batch_size}")
    print(f"  Safety Margin:       {auto_cfg.safety_margin:.2f}×")
    print(f"")
    print(f"TRAINING:")
    print(f"  Target Episodes:     {args.total_episodes:,}")
    print(f"  Est. Updates:        ~{est_updates:,}")
    print(f"  Episodes/Update:     ~{eps_per_update:.1f}")
    print(f"  n_epochs:            {cfg.train.mappo_n_epochs}")
    print(f"  LR (actor/critic):   {cfg.train.mappo_lr_actor}/{cfg.train.mappo_lr_critic}")
    print(f"  Device:              {device}")
    print(f"")
    print(f"LOGGING:")
    print(f"  Log interval:        {args.log_interval} episodes")
    print(f"  Checkpoint interval: {args.checkpoint_interval} episodes")
    print(f"  HF Upload:           {args.hf_upload}")
    if args.hf_upload:
        print(f"  HF Upload every:     {args.hf_upload_every} eps")
    print(f"{'='*70}\n")

    assert auto_cfg.rollout_length >= cfg.env.max_steps, \
        "Rollout too short for episode length"
    assert auto_cfg.buffer_capacity >= cfg.env.max_steps * args.n_envs, \
        "Buffer too small"

    # ─────────────────────────────────────────────────────────
    # ← CREATE TRAINER
    # ─────────────────────────────────────────────────────────
    HF_TOKEN = args.hf_token or os.getenv("HF_TOKEN")
    HF_REPO = "duy95/sar-uav-results"
    run_name = args.run_name or f"mappo_s{args.seed}"

    trainer = MAPPOTrainer(
        config=cfg,
        device=device,
        run_name=run_name,
        n_envs=args.n_envs,
        hf_token=HF_TOKEN if args.hf_upload else None,
        hf_repo=HF_REPO if args.hf_upload else None,
        hf_upload_every=args.hf_upload_every,
    )

    # ─────────────────────────────────────────────────────────
    # ← PATCH LLM REWARD
    # ─────────────────────────────────────────────────────────
    if llm_reward:
        import training.algorithms.mappo.trainer as tm
        _orig = tm._EnvWrapper.__init__
        
        def patched(self, config, n_envs, seed):
            _orig(self, config, n_envs, seed)
            
            if hasattr(self, '_env') and not self._is_vec:
                if hasattr(self._env, '_base_env'):
                    self._env._base_env.baseline_reward = llm_reward
                    print(f"\n{'='*70}")
                    print(f"✅ LLM REWARD INJECTED INTO ENV")
                    print(f"{'='*70}\n")
                else:
                    print(f"\n⚠️  Cannot inject: no _base_env attribute\n")
            elif self._is_vec:
                print(f"\n⚠️  Vectorized env - LLM reward not supported yet\n")
        
        tm._EnvWrapper.__init__ = patched

    # ─────────────────────────────────────────────────────────
    # ← TRAIN
    # ─────────────────────────────────────────────────────────
    trainer.train(
        total_episodes=args.total_episodes,
        curriculum_manager=None,
        seed=args.seed,
        log_every_n_eps=args.log_interval,
        checkpoint_every_n_eps=args.checkpoint_interval,
    )

    # ─────────────────────────────────────────────────────────
    # ← SUMMARY
    # ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Episodes:  {trainer.total_episodes_done:,}")
    print(f"  Steps:     {trainer.total_steps:,}")
    print(f"  Updates:   {trainer.update_count:,}")
    if trainer.ep_rewards:
        print(f"  Reward:    {np.mean(trainer.ep_rewards):.2f} ± {np.std(trainer.ep_rewards):.2f}")
        print(f"  Coverage:  {np.mean(trainer.ep_coverage):.1f}%")
        print(f"  Victims:   {np.mean(trainer.ep_victims):.1f}%")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()