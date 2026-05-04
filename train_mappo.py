#!/usr/bin/env python3
"""
🚁 MAPPO Training Entry Point
Hiện tại: Chỉ train HARD stage (no curriculum)
"""

import argparse
import torch

from config import AppConfig, STAGE_EASY, STAGE_MEDIUM, STAGE_HARD
from training.curriculum import CurriculumManager
from training.algorithms.mappo.trainer import MAPPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="MAPPO Training for SAR UAV Swarm")
    parser.add_argument("--total-episodes",  type=int,   default=3000)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--device",          type=str,   default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--run-name",        type=str,   default=None)
    parser.add_argument("--n-envs",          type=int,   default=1)

    # ✅ Stage control - default hard, no curriculum
    parser.add_argument("--stage",           type=str,   default="hard",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--curriculum",      action="store_true",
                        help="Bật curriculum EASY→MEDIUM→HARD (mặc định TẮT)")

    # Environment overrides
    parser.add_argument("--max-steps",       type=int,   default=None)
    parser.add_argument("--map-size",        type=int,   default=None)
    parser.add_argument("--n-victims",       type=int,   default=None)
    parser.add_argument("--n-debris",        type=int,   default=None)
    parser.add_argument(
        "--force-print",
        action="store_true",
        help="Kaggle/Colab: bypass tqdm, dùng print+flush"
    )
    # Training hyperparams
    parser.add_argument("--rollout-length",  type=int,   default=None)
    parser.add_argument("--batch-size",      type=int,   default=None)
    parser.add_argument("--n-epochs",        type=int,   default=None)
    parser.add_argument("--lr-actor",        type=float, default=None)
    parser.add_argument("--lr-critic",       type=float, default=None)
    parser.add_argument("--gamma",           type=float, default=None)
    parser.add_argument("--gae-lambda",      type=float, default=None)
    parser.add_argument("--clip-epsilon",    type=float, default=None)
    parser.add_argument("--entropy-coeff",   type=float, default=None)
    parser.add_argument("--no-balance",      action="store_true")

    return parser.parse_args()


def calculate_balanced_rollout(base_rollout, n_envs, batch_size, max_episode_steps):
    N_AGENTS        = 4
    rollout_per_env = base_rollout // n_envs
    min_for_episode = int(max_episode_steps * 1.2)
    min_for_batch   = (batch_size + N_AGENTS - 1) // N_AGENTS
    absolute_min    = 128
    min_rollout     = max(min_for_episode, min_for_batch, absolute_min)
    adjusted        = max(rollout_per_env, min_rollout)
    adjusted        = ((adjusted + 63) // 64) * 64
    ideal           = ((rollout_per_env + 63) // 64) * 64
    return {
        'rollout':     adjusted,
        'ideal':       ideal,
        'was_clamped': adjusted > ideal,
    }


def estimate_updates(total_episodes, rollout, n_envs, max_episode_steps):
    avg_ep_length       = float(max_episode_steps) * 1.02
    total_steps_per_upd = rollout * n_envs
    eps_per_update      = total_steps_per_upd / avg_ep_length
    est_updates         = max(1, int(total_episodes / eps_per_update))
    return {'eps_per_update': eps_per_update, 'est_updates': est_updates}


def main():
    args = parse_args()

    # ── Device ──────────────────────────────────────────────────
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device

    # ── Config ──────────────────────────────────────────────────
    cfg          = AppConfig()
    cfg.viz_mode = "none"

    # ── Stage / Curriculum setup ─────────────────────────────────
    curriculum_manager = None

    STAGE_MAP = {
        "easy":   STAGE_EASY,
        "medium": STAGE_MEDIUM,
        "hard":   STAGE_HARD,
    }

    if args.curriculum:
        # ── Curriculum mode: EASY → MEDIUM → HARD ───────────────
        curriculum_manager = CurriculumManager(
            [STAGE_EASY, STAGE_MEDIUM, STAGE_HARD]
        )
        curriculum_manager.apply_to_config(cfg)
        stage = curriculum_manager.current_stage

        print(f"📚 Mode: CURRICULUM (EASY → MEDIUM → HARD)")
        print(
            f"✅ Start stage: [{stage.name.upper()}] "
            f"map={stage.map_size}×{stage.map_size}m | "
            f"victims={stage.n_victims_min}-{stage.n_victims_max} | "
            f"steps={stage.max_steps}"
        )

    else:
        # ── Single stage mode (default: HARD) ───────────────────
        selected = STAGE_MAP[args.stage]
        cfg.apply_stage(selected)

        print(f"📌 Mode: SINGLE STAGE [{args.stage.upper()}] (no curriculum)")
        print(
            f"✅ Stage config: "
            f"map={selected.map_size}×{selected.map_size}m "
            f"({selected.map_size**2:,}m²) | "
            f"UAVs={cfg.env.n_uav} | "
            f"victims={selected.n_victims_min}-{selected.n_victims_max} | "
            f"debris={selected.n_debris} | "
            f"danger={selected.n_danger_total} | "
            f"steps={selected.max_steps}"
        )

    # ── Environment overrides ────────────────────────────────────
    overrides = []
    if args.max_steps:
        cfg.env.max_steps        = args.max_steps
        overrides.append(f"max_steps={args.max_steps}")
    if args.map_size:
        cfg.env.map_size         = args.map_size
        cfg.env.grid_size        = args.map_size
        overrides.append(f"map_size={args.map_size}m")
    if args.n_victims:
        cfg.victim.n_victims_min = args.n_victims
        cfg.victim.n_victims_max = args.n_victims
        overrides.append(f"n_victims={args.n_victims}")
    if args.n_debris:
        cfg.obstacle.n_debris    = args.n_debris
        overrides.append(f"n_debris={args.n_debris}")
    if overrides:
        print(f"  ⚙️  Overrides: {' | '.join(overrides)}")

    # ── MAPPO hyperparams ────────────────────────────────────────
    if args.batch_size:    cfg.train.mappo_batch_size    = args.batch_size
    if args.n_epochs:      cfg.train.mappo_n_epochs      = args.n_epochs
    if args.lr_actor:      cfg.train.mappo_lr_actor      = args.lr_actor
    if args.lr_critic:     cfg.train.mappo_lr_critic     = args.lr_critic
    if args.gamma:         cfg.train.mappo_gamma         = args.gamma
    if args.gae_lambda:    cfg.train.mappo_gae_lambda    = args.gae_lambda
    if args.clip_epsilon:  cfg.train.mappo_clip_epsilon  = args.clip_epsilon
    if args.entropy_coeff: cfg.train.mappo_entropy_coeff = args.entropy_coeff

    # ── Rollout length ───────────────────────────────────────────
    base_rollout = args.rollout_length or cfg.train.mappo_rollout_length
    need_balance = (not args.no_balance) and (args.n_envs > 1)

    if need_balance:
        result        = calculate_balanced_rollout(
            base_rollout, args.n_envs,
            cfg.train.mappo_batch_size, cfg.env.max_steps
        )
        final_rollout = result['rollout']
        balance_label = "ON (auto)"
    else:
        result        = {'rollout': base_rollout, 'ideal': base_rollout,
                         'was_clamped': False}
        final_rollout = base_rollout
        balance_label = "OFF" if args.no_balance else "N/A (n_envs=1)"

    cfg.train.mappo_rollout_length = final_rollout

    # ── Estimate updates ─────────────────────────────────────────
    est = estimate_updates(
        args.total_episodes, final_rollout,
        args.n_envs, cfg.env.max_steps
    )

    # ── Print training config ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"🚁 SAR UAV SWARM — MAPPO TRAINING")
    print(f"{'='*60}")
    print(f"  Seed:              {args.seed}")
    print(f"  Device:            {device}")
    print(f"  n_envs:            {args.n_envs}")
    print(f"  Auto-balance:      {balance_label}")
    print(f"")
    print(f"  Target episodes:   {args.total_episodes}")
    print(f"  Est. updates:      ~{est['est_updates']}")
    print(f"  Eps/update:        ~{est['eps_per_update']:.1f}")
    print(f"")
    print(f"  Rollout (base):    {base_rollout}")
    print(f"  Rollout (final):   {final_rollout}")
    if need_balance:
        ok = "✅" if final_rollout >= cfg.env.max_steps else "⚠️"
        print(f"  Rollout≥max_steps: {final_rollout}≥{cfg.env.max_steps} {ok}")
        print(f"  Total steps/upd:   {final_rollout * args.n_envs:,}")
    print(f"  Batch size:        {cfg.train.mappo_batch_size}")
    print(f"  n_epochs:          {cfg.train.mappo_n_epochs}")
    print(f"  Map size:          {cfg.env.map_size}m")
    print(f"  Max steps/ep:      {cfg.env.max_steps}")
    print(f"  LR actor/critic:   {cfg.train.mappo_lr_actor}/{cfg.train.mappo_lr_critic}")
    print(f"{'='*60}\n")

    # ── Create & Run ─────────────────────────────────────────────
    trainer = MAPPOTrainer(
        config   = cfg,
        device   = device,
        run_name = args.run_name,
        n_envs   = args.n_envs
    )

    trainer.train(
    total_episodes     = args.total_episodes,
    curriculum_manager = curriculum_manager,
    seed               = args.seed,
    log_every_ep       = 10,
    force_print        = args.force_print,
)

    print(f"\n{'='*60}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total episodes:  {trainer.total_episodes_done}")
    print(f"  Total steps:     {trainer.total_steps_collected:,}")
    print(f"  Checkpoints:     {trainer.checkpoint_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()