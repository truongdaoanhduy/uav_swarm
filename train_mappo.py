#!/usr/bin/env python3
"""
🚁 MAPPO Training — HARD Stage Focus
Full auto-config system for production training
"""

import argparse
import torch
from dataclasses import dataclass

from config import AppConfig, STAGE_HARD
from training.algorithms.mappo.trainer import MAPPOTrainer


@dataclass
class AutoConfig:
    """Auto-computed training configuration."""
    rollout_length: int
    buffer_capacity: int
    batch_size: int
    safety_margin: float


def auto_compute_config(max_steps: int, n_envs: int, n_uav: int, 
                        batch_size_hint: int = None, safety_factor: float = 1.5):
    """
    Auto-compute optimal training hyperparameters.
    
    GUARANTEES:
        rollout_length ≥ max_steps × safety_factor
        buffer_capacity = rollout_length × n_envs
    """
    # Rollout must be >= episode length × safety factor
    min_rollout = int(max_steps * safety_factor)
    rollout_length = ((min_rollout + 63) // 64) * 64  # Align to 64
    
    # Buffer capacity
    buffer_capacity = rollout_length * n_envs
    
    # Batch size
    if batch_size_hint is None:
        batch_size = max(64, buffer_capacity // 8)
        batch_size = ((batch_size + 63) // 64) * 64
    else:
        min_batch = 64
        max_batch = buffer_capacity // 2
        batch_size = max(min_batch, min(batch_size_hint, max_batch))
    
    safety = buffer_capacity / max(max_steps * n_envs, 1)
    
    return AutoConfig(
        rollout_length=rollout_length,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        safety_margin=safety,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="MAPPO Training — HARD Stage")
    
    # Core params
    parser.add_argument("--total-episodes", type=int, default=3000,
                        help="Total training episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Training device")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom run name (auto-generated if None)")
    
    # Environment
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of parallel environments")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Episode timeout (default: 400 from HARD stage)")
    parser.add_argument("--map-size", type=int, default=None,
                        help="Map size (default: 250m from HARD stage)")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (auto-computed if None)")
    parser.add_argument("--safety-factor", type=float, default=1.5,
                        help="Rollout safety margin")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="PPO epochs per update")
    parser.add_argument("--lr-actor", type=float, default=None,
                        help="Actor learning rate")
    parser.add_argument("--lr-critic", type=float, default=None,
                        help="Critic learning rate")
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Detailed log every N episodes")
    parser.add_argument("--viz-interval", type=int, default=None,
                        help="2D viz every N episodes (default: 5 × log-interval)")
    parser.add_argument("--checkpoint-interval", type=int, default=20,
                        help="Save checkpoint every N episodes")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    
    # Config - HARD stage only
    cfg = AppConfig()
    cfg.viz_mode = "2d"  # ✅ Enable 2D visualization
    cfg.apply_stage(STAGE_HARD)
    
    # Apply overrides
    cfg.env.n_uav = 4  # ✅ Fixed to 4 UAVs (không dùng args.n_uav)
    if args.max_steps:
        cfg.env.max_steps = args.max_steps
    if args.map_size:
        cfg.env.map_size = args.map_size
        cfg.env.grid_size = args.map_size
    
    # Auto-compute training config
    auto_cfg = auto_compute_config(
        max_steps=cfg.env.max_steps,
        n_envs=args.n_envs,
        n_uav=cfg.env.n_uav,
        batch_size_hint=args.batch_size,
        safety_factor=args.safety_factor,
    )
    
    # Apply auto-computed values
    cfg.train.mappo_rollout_length = auto_cfg.rollout_length
    cfg.train.mappo_batch_size = auto_cfg.batch_size
    
    # Apply optional overrides
    if args.n_epochs:
        cfg.train.mappo_n_epochs = args.n_epochs
    if args.lr_actor:
        cfg.train.mappo_lr_actor = args.lr_actor
    if args.lr_critic:
        cfg.train.mappo_lr_critic = args.lr_critic
    
    # ✅ Viz interval: 5× log interval
    viz_interval = args.viz_interval if args.viz_interval else (args.log_interval * 5)
    
    # Estimate training
    avg_ep_len = cfg.env.max_steps * 0.85
    steps_per_update = auto_cfg.rollout_length * args.n_envs
    eps_per_update = steps_per_update / avg_ep_len
    est_updates = max(1, int(args.total_episodes / eps_per_update))
    
    # Print config
    print(f"\n{'='*70}")
    print(f"🚁 MAPPO TRAINING — HARD STAGE")
    print(f"{'='*70}")
    print(f"ENVIRONMENT:")
    print(f"  Map Size:            {cfg.env.map_size}×{cfg.env.map_size}m")
    print(f"  Max Steps:           {cfg.env.max_steps}")
    print(f"  n_envs:              {args.n_envs}")
    print(f"  n_uav:               {cfg.env.n_uav}")  # Always 4
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
    print(f"  Viz interval:        {viz_interval} episodes (5× log)")
    print(f"  Checkpoint interval: {args.checkpoint_interval} episodes")
    print(f"{'='*70}\n")
    
    # Runtime assertions
    assert auto_cfg.rollout_length >= cfg.env.max_steps, \
        "Rollout too short for episode length"
    assert auto_cfg.buffer_capacity >= cfg.env.max_steps * args.n_envs, \
        "Buffer too small"
    
    # Create trainer
    trainer = MAPPOTrainer(
        config=cfg,
        device=device,
        run_name=args.run_name,
        n_envs=args.n_envs,
    )
    
    # Train
    print(f"🚀 Starting training...\n")
    trainer.train(
        total_episodes=args.total_episodes,
        curriculum_manager=None,  # No curriculum
        seed=args.seed,
        log_every_n_eps=args.log_interval,
        viz_every_n_eps=viz_interval,
        checkpoint_every_n_eps=args.checkpoint_interval,
    )
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Episodes:            {trainer.total_episodes_done:,}")
    print(f"  Steps:               {trainer.total_steps:,}")
    print(f"  Updates:             {trainer.update_count:,}")
    print(f"  Final Checkpoint:    {trainer.checkpoint_dir}/checkpoint_final.pt")
    print(f"")
    print(f"📊 Final Metrics:")
    if trainer.ep_rewards:
        import numpy as np
        print(f"  Reward:              {np.mean(trainer.ep_rewards):.2f} ± {np.std(trainer.ep_rewards):.2f}")
        print(f"  Coverage:            {np.mean(trainer.ep_coverage):.1f}%")
        print(f"  Victims Found:       {np.mean(trainer.ep_victims):.1f}%")
    print(f"")
    print(f"🎬 Visualize results:")
    print(f"  python visualize_policy.py --checkpoint {trainer.checkpoint_dir}/checkpoint_final.pt")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()