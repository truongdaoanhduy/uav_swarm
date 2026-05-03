#!/usr/bin/env python3
"""
🚁 MAPPO Training Entry Point
Curriculum-based training: EASY → MEDIUM → HARD

Usage:
    python train_mappo.py
    python train_mappo.py --seed 42 --total-updates 500
    python train_mappo.py --no-curriculum --stage easy
    python train_mappo.py --device cpu --run-name my_exp
"""

import argparse
import torch

from config import AppConfig, STAGE_EASY, STAGE_MEDIUM, STAGE_HARD
from training.curriculum import CurriculumManager
from training.algorithms.mappo.trainer import MAPPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="MAPPO Training for SAR UAV Swarm")
    
    # Training params
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--total-updates", type=int, default=500,
                        help="Total PPO updates (default: 500)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device (default: auto)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom run name (default: auto-generate)")
    
    # Curriculum
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum, dùng fixed stage")
    parser.add_argument("--stage", type=str, default="easy",
                        choices=["easy", "medium", "hard"],
                        help="Stage khi --no-curriculum (default: easy)")
    
    # Config overrides
    parser.add_argument("--rollout-length", type=int, default=None,
                        help="Rollout length (default: config value 2048)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Minibatch size (default: config value 256)")
    parser.add_argument("--lr-actor", type=float, default=None,
                        help="Actor learning rate (default: 3e-4)")
    parser.add_argument("--lr-critic", type=float, default=None,
                        help="Critic learning rate (default: 1e-3)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("\n" + "="*60)
    print("🚁 SAR UAV SWARM — MAPPO TRAINING")
    print("="*60)
    print(f"  Seed:          {args.seed}")
    print(f"  Total Updates: {args.total_updates}")
    print(f"  Device:        {device}")
    print(f"  Curriculum:    {'Disabled → ' + args.stage.upper() if args.no_curriculum else 'EASY → MEDIUM → HARD'}")
    print("="*60 + "\n")
    
    # ============ Load Config ============
    cfg = AppConfig()
    cfg.viz_mode = "none"  # Disable real-time viz khi training
    
    # Apply overrides nếu có
    if args.rollout_length:
        cfg.train.mappo_rollout_length = args.rollout_length
    if args.batch_size:
        cfg.train.mappo_batch_size = args.batch_size
    if args.lr_actor:
        cfg.train.mappo_lr_actor = args.lr_actor
    if args.lr_critic:
        cfg.train.mappo_lr_critic = args.lr_critic
    
    # ============ Setup Curriculum ============
    curriculum_manager = None
    
    if not args.no_curriculum:
        # Curriculum mode: EASY → MEDIUM → HARD
        curriculum_manager = CurriculumManager([STAGE_EASY, STAGE_MEDIUM, STAGE_HARD])
        curriculum_manager.apply_to_config(cfg)
        print(f"📚 Curriculum enabled: starting at EASY stage")
    else:
        # Fixed stage mode
        stage_map = {
            "easy": STAGE_EASY,
            "medium": STAGE_MEDIUM,
            "hard": STAGE_HARD
        }
        cfg.apply_stage(stage_map[args.stage])
        print(f"📌 Fixed stage: {args.stage.upper()} ({stage_map[args.stage].map_size}m map)")
    
    # ============ Create Trainer ============
    trainer = MAPPOTrainer(
        config=cfg,
        device=device,
        run_name=args.run_name
    )
    
    # ============ Train ============
    trainer.train(
        total_updates=args.total_updates,
        curriculum_manager=curriculum_manager,
        seed=args.seed
    )
    
    # ============ Done ============
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"  Checkpoints: {trainer.checkpoint_dir}")
    print(f"  Viz:         {trainer.viz_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()