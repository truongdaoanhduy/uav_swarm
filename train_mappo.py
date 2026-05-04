#!/usr/bin/env python3
"""
🚁 MAPPO Training Entry Point
Auto-balance rollout_length theo n_envs (luôn bật mặc định)
"""

import argparse
import torch

from config import AppConfig, STAGE_EASY, STAGE_MEDIUM, STAGE_HARD
from training.curriculum import CurriculumManager
from training.algorithms.mappo.trainer import MAPPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="MAPPO Training for SAR UAV Swarm")

    parser.add_argument("--total-episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--stage", type=str, default="easy",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--map-size", type=int, default=None)
    parser.add_argument("--n-victims", type=int, default=None)
    parser.add_argument("--n-debris", type=int, default=None)
    parser.add_argument("--rollout-length", type=int, default=None,
                        help="Base rollout length cho n_envs=1 (default: 2048)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--lr-actor", type=float, default=None)
    parser.add_argument("--lr-critic", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--clip-epsilon", type=float, default=None)
    parser.add_argument("--entropy-coeff", type=float, default=None)
    parser.add_argument("--no-balance", action="store_true",
                        help="Tắt auto-balance (không khuyến nghị)")

    return parser.parse_args()


def calculate_balanced_rollout(
    base_rollout: int,
    n_envs: int,
    batch_size: int,
    max_episode_steps: int,
) -> dict:
    """
    Returns dict với rollout và metadata.
    """
    N_AGENTS = 4

    # Core
    rollout_per_env = base_rollout // n_envs

    # Constraints
    min_for_episode = int(max_episode_steps * 1.2)
    min_for_batch   = (batch_size + N_AGENTS - 1) // N_AGENTS
    absolute_min    = 128

    min_rollout = max(min_for_episode, min_for_batch, absolute_min)
    adjusted    = max(rollout_per_env, min_rollout)

    # Round to 64
    adjusted = ((adjusted + 63) // 64) * 64

    # Ideal (không có constraint)
    ideal = ((rollout_per_env + 63) // 64) * 64

    return {
        'rollout':     adjusted,
        'ideal':       ideal,
        'was_clamped': adjusted > ideal,
        'clamp_reason': 'min_episode' if adjusted == min_for_episode else
                        'min_batch'   if adjusted == min_for_batch   else
                        'absolute'    if adjusted == absolute_min     else
                        'none'
    }


def estimate_updates(
    total_episodes: int,
    rollout: int,
    n_envs: int,
    max_episode_steps: int,
) -> dict:
    """
    Ước tính số updates thực tế.
    
    Với early stop: không phải mọi rollout đều collect đủ eps.
    Ước tính conservative hơn.
    """
    avg_ep_length = float(max_episode_steps) * 1.02

    # Steps mỗi rollout (thực tế, có early stop)
    # Worst case: mỗi rollout chỉ collect (rollout × n_envs) steps
    # Nhưng episode early stop → thực tế ít hơn
    total_steps_per_update = rollout * n_envs

    # Episodes per update: chia total steps cho avg episode length
    eps_per_update = total_steps_per_update / avg_ep_length

    # Updates thực tế: có thể cao hơn ước tính vì early stop
    est_updates = max(1, int(total_episodes / eps_per_update))

    return {
        'eps_per_update': eps_per_update,
        'est_updates':    est_updates,
    }



def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Config
    cfg = AppConfig()
    cfg.viz_mode = "none"

    # Curriculum
    curriculum_manager = None
    if not args.no_curriculum:
        curriculum_manager = CurriculumManager([STAGE_HARD])
        curriculum_manager.apply_to_config(cfg)
        print(f"📚 Curriculum enabled: EASY → MEDIUM → HARD")
    else:
        stage_map = {"easy": STAGE_EASY, "medium": STAGE_MEDIUM, "hard": STAGE_HARD}
        cfg.apply_stage(stage_map[args.stage])
        print(f"📌 Fixed stage: {args.stage.upper()}")

    # ── Environment overrides ───────────────────────────────────
    if args.max_steps:
        cfg.env.max_steps = args.max_steps
        print(f"  ⚙️  max_steps      = {args.max_steps}")
    if args.map_size:
        cfg.env.map_size = args.map_size
        cfg.env.grid_size = args.map_size
        print(f"  ⚙️  map_size       = {args.map_size}m")
    if args.n_victims:
        cfg.victim.n_victims_min = args.n_victims
        cfg.victim.n_victims_max = args.n_victims
        print(f"  ⚙️  n_victims      = {args.n_victims}")
    if args.n_debris:
        cfg.obstacle.n_debris = args.n_debris
        print(f"  ⚙️  n_debris       = {args.n_debris}")

    # ── MAPPO overrides ─────────────────────────────────────────
    if args.batch_size:
        cfg.train.mappo_batch_size = args.batch_size
        print(f"  ⚙️  batch_size     = {args.batch_size}")
    if args.n_epochs:
        cfg.train.mappo_n_epochs = args.n_epochs
        print(f"  ⚙️  n_epochs       = {args.n_epochs}")
    if args.lr_actor:
        cfg.train.mappo_lr_actor = args.lr_actor
        print(f"  ⚙️  lr_actor       = {args.lr_actor}")
    if args.lr_critic:
        cfg.train.mappo_lr_critic = args.lr_critic
        print(f"  ⚙️  lr_critic      = {args.lr_critic}")
    if args.gamma:
        cfg.train.mappo_gamma = args.gamma
        print(f"  ⚙️  gamma          = {args.gamma}")
    if args.gae_lambda:
        cfg.train.mappo_gae_lambda = args.gae_lambda
        print(f"  ⚙️  gae_lambda     = {args.gae_lambda}")
    if args.clip_epsilon:
        cfg.train.mappo_clip_epsilon = args.clip_epsilon
        print(f"  ⚙️  clip_epsilon   = {args.clip_epsilon}")
    if args.entropy_coeff:
        cfg.train.mappo_entropy_coeff = args.entropy_coeff
        print(f"  ⚙️  entropy_coeff  = {args.entropy_coeff}")

    # ── Rollout length: tính 1 lần duy nhất ────────────────────
    base_rollout = args.rollout_length or cfg.train.mappo_rollout_length
    need_balance = (not args.no_balance) and (args.n_envs > 1)

    if need_balance:
        result = calculate_balanced_rollout(
            base_rollout=base_rollout,
            n_envs=args.n_envs,
            batch_size=cfg.train.mappo_batch_size,
            max_episode_steps=cfg.env.max_steps,
        )
        final_rollout = result['rollout']
        balance_status = "ON (auto)"
    else:
        result = {'rollout': base_rollout, 'ideal': base_rollout,
                  'was_clamped': False, 'clamp_reason': 'none'}
        final_rollout = base_rollout
        balance_status = "OFF" if args.no_balance else "N/A (n_envs=1)"

    cfg.train.mappo_rollout_length = final_rollout

    # ── Estimate updates ────────────────────────────────────────
    est = estimate_updates(
        total_episodes=args.total_episodes,
        rollout=final_rollout,
        n_envs=args.n_envs,
        max_episode_steps=cfg.env.max_steps,
    )

    # ── Print summary ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"🚁 SAR UAV SWARM — MAPPO TRAINING")
    print(f"{'='*60}")
    print(f"  Seed:              {args.seed}")
    print(f"  Device:            {device}")
    print(f"  n_envs:            {args.n_envs}")
    print(f"  Auto-balance:      {balance_status}")
    print(f"")
    print(f"  Target episodes:   {args.total_episodes}")
    print(f"  Est. updates:      ~{est['est_updates']}")
    print(f"  Eps/update:        ~{est['eps_per_update']:.1f}")
    print(f"")
    print(f"  Rollout (base):    {base_rollout}")
    print(f"  Rollout (final):   {final_rollout}")

    if need_balance:
        status = "✅" if final_rollout >= cfg.env.max_steps else "❌"
        print(f"  Rollout ≥ max_steps: {final_rollout} ≥ {cfg.env.max_steps} {status}")
        print(f"  Total/step:        {final_rollout * args.n_envs}")

    print(f"  Batch size:        {cfg.train.mappo_batch_size}")
    print(f"  n_epochs:          {cfg.train.mappo_n_epochs}")
    print(f"  Map size:          {cfg.env.map_size}m")
    print(f"  Max steps/ep:      {cfg.env.max_steps}")
    print(f"  LR actor/critic:   {cfg.train.mappo_lr_actor}/{cfg.train.mappo_lr_critic}")
    print(f"{'='*60}")

    # ── Warnings (chỉ khi thực sự cần) ─────────────────────────
    warnings = []

    if result['was_clamped'] and need_balance:
        # ✅ Chỉ warn nếu updates giảm ĐÁNG KỂ (< 70% expected)
        ideal_eps_per_update = (result['ideal'] * args.n_envs) / (cfg.env.max_steps * 1.02)
        ideal_updates = max(1, int(args.total_episodes / ideal_eps_per_update))
        
        if est['est_updates'] < ideal_updates * 0.7:
            warnings.append(
                f"  ⚠️  Rollout adjusted: {result['ideal']} → {final_rollout}\n"
                f"      (rollout/env={result['ideal']} < max_steps={cfg.env.max_steps})\n"
                f"      Updates: ~{est['est_updates']} vs ideal ~{ideal_updates}\n"
                f"      Fix: --rollout-length {base_rollout * 2} hoặc giảm --n-envs"
            )

    if est['est_updates'] < 5:
        warnings.append(
            f"  ⚠️  Chỉ ~{est['est_updates']} updates - quá ít!\n"
            f"      Fix: Tăng --total-episodes lên {args.total_episodes * 3}"
        )

    if warnings:
        print()
        for w in warnings:
            print(w)

    print()

    # ── Create & Run ────────────────────────────────────────────
    trainer = MAPPOTrainer(
        config=cfg,
        device=device,
        run_name=args.run_name,
        n_envs=args.n_envs
    )

    trainer.train(
        total_episodes=args.total_episodes,
        curriculum_manager=curriculum_manager,
        seed=args.seed
    )

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"  Total episodes:  {trainer.total_episodes_done}")
    print(f"  Total steps:     {trainer.total_steps_collected}")
    print(f"  Checkpoints:     {trainer.checkpoint_dir}")
    print(f"  Viz:             {trainer.viz_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()