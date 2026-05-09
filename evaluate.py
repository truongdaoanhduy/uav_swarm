# evaluate.py
"""
Evaluation script - Train on HARD, Eval on HARD/EXTREME/TRANSFER.

Usage examples:
    # 1. Eval 3 algo trên HARD (nơi đã train)
    python evaluate.py \\
        --mappo results/mappo/run_s42/checkpoints/checkpoint_final.pt \\
        --masac results/masac/run_s42/checkpoints/checkpoint_final.pt \\
        --matd3 results/matd3/run_s42/checkpoints/checkpoint_final.pt \\
        --stage hard --n-episodes 100

    # 2. Zero-shot eval trên EXTREME (khó hơn, chưa train)
    python evaluate.py \\
        --mappo results/mappo/run_s42/checkpoints/checkpoint_final.pt \\
        --masac results/masac/run_s42/checkpoints/checkpoint_final.pt \\
        --matd3 results/matd3/run_s42/checkpoints/checkpoint_final.pt \\
        --stage extreme --n-episodes 50

    # 3. Multi-seed eval (Paper 1 - 5 seeds × 3 algos × 100 eps)
    python evaluate.py \\
        --mappo-dir results/mappo/ \\
        --masac-dir results/masac/ \\
        --matd3-dir results/matd3/ \\
        --stage hard --n-episodes 100 --multi-seed

    # 4. Kaggle notebook (gọi từ Python)
    from evaluate import quick_eval
    results = quick_eval(
        checkpoints={
            "mappo": "/kaggle/working/results/mappo/run_s42/checkpoints/checkpoint_final.pt",
            "masac": "/kaggle/working/results/masac/run_s42/checkpoints/checkpoint_final.pt",
            "matd3": "/kaggle/working/results/matd3/run_s42/checkpoints/checkpoint_final.pt",
        },
        stage="extreme",
        n_episodes=50,
    )
"""

from __future__ import annotations

import argparse
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import torch

from config import AppConfig
from config.curriculum_config import STAGE_HARD, STAGE_EXTREME, STAGE_TRANSFER
from env_setup.sar_pettingzoo_env import SARPettingZooEnv

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

STAGE_MAP = {
    "hard":     STAGE_HARD,
    "extreme":  STAGE_EXTREME,
    "transfer": STAGE_TRANSFER,
}

ALGO_COLORS = {
    "mappo": "#2196F3",
    "masac": "#4CAF50",
    "matd3": "#FF9800",
}

def _normalize_coverage(value: float) -> float:
    """
    Normalize coverage về [0, 1] range.
    
    Handles cả [0, 1] và [0, 100] input.
    """
    if value > 1.0:
        # Assume percentage [0, 100]
        return min(value / 100.0, 1.0)
    else:
        # Already [0, 1]
        return min(value, 1.0)  # Clip to 1.0 just in case


def _normalize_rate(value: float) -> float:
    """Normalize victim/success rate về [0, 1]."""
    if value > 1.0:
        return min(value / 100.0, 1.0)
    else:
        return min(value, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EpisodeResult:
    """Kết quả 1 episode eval."""
    episode_id:      int
    seed:            int
    algo:            str
    stage:           str
    # Primary metrics
    total_reward:    float
    coverage_rate:   float   # [0, 1]
    victims_found:   int
    total_victims:   int
    episode_length:  int
    # Secondary metrics
    n_landings:      int
    battery_deaths:  int
    collision_count: int
    # Computed
    success:         bool    # coverage >= 0.9
    victim_rate:     float   # found / total
    # Wall time
    wall_time_s:     float


@dataclass
class AlgoResult:
    """Tổng hợp kết quả N episodes của 1 algo."""
    algo:             str
    stage:            str
    n_episodes:       int
    checkpoint:       str
    # Statistics
    reward_mean:      float
    reward_std:       float
    reward_median:    float
    coverage_mean:    float
    coverage_std:     float
    victim_rate_mean: float
    victim_rate_std:  float
    success_rate:     float
    length_mean:      float
    # Raw arrays (cho Wilcoxon)
    rewards:          List[float] = field(default_factory=list)
    coverages:        List[float] = field(default_factory=list)
    victim_rates:     List[float] = field(default_factory=list)
    successes:        List[bool]  = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_eval_config(stage: str, n_uav: int = 4) -> AppConfig:
    """
    Tạo AppConfig cho eval stage.

    actor_dim = 80 (thực tế từ code):
        11 (self) + 8 (stations 2×4) + 9 (teammates 3×3)
        + 24 (obstacles 8×3) + 25 (victims 5×5) + 3 (coverage) = 80

    critic_dim = 650:
        8 (max_uav) × 80 (actor_dim) + 10 (global) = 650

    Chỉ thay đổi: map_size, max_steps, n_victims, n_debris, n_danger_total
    Network weights KHÔNG thay đổi → load checkpoint bình thường.
    """
    if stage not in STAGE_MAP:
        raise ValueError(
            f"Unknown stage '{stage}'. Valid: {list(STAGE_MAP.keys())}"
        )

    cfg = AppConfig()
    cfg.apply_stage(STAGE_MAP[stage])
    cfg.env.n_uav = n_uav
    cfg.env.deterministic_eval = False

    actual_actor_dim  = cfg.obs.actor_dim
    actual_critic_dim = cfg.obs.critic_dim

    print(f"\n📐 Eval Config:")
    print(f"   Stage      : {STAGE_MAP[stage].describe()}")
    print(f"   actor_dim  : {actual_actor_dim}   (80 = 11+8+9+24+25+3)")
    print(f"   critic_dim : {actual_critic_dim}  (650 = 8×80+10)")
    print(f"   map_size   : {cfg.env.map_size}m")
    print(f"   max_steps  : {cfg.env.max_steps}")
    print(f"   n_victims  : {cfg.victim.n_victims_min}-{cfg.victim.n_victims_max}")
    print(f"   n_debris   : {cfg.obstacle.n_debris}")
    print(f"   n_danger   : {cfg.obstacle.n_danger_total}")
    print(f"   n_stations : {cfg.env.n_stations}  (không đổi)")

    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# ACTOR LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_actor(
    checkpoint_path: str,
    algo:            str,
    config:          AppConfig,
    device:          str = "cpu",
) -> torch.nn.Module:
    """
    Load actor network từ checkpoint.
    
    Hỗ trợ tất cả 3 algo: mappo, masac, matd3.
    Tự động detect key trong checkpoint dict.
    
    Args:
        checkpoint_path: Path đến .pt file
        algo:           "mappo" | "masac" | "matd3"
        config:         AppConfig (phải match với training config)
        device:         "cpu" | "cuda"
    
    Returns:
        Actor network ở eval mode (no grad)
    """
    algo = algo.lower().strip()
    tr   = config.train

    # ── Build network ─────────────────────────────────────────────────────────
    if algo == "mappo":
        from training.algorithms.mappo.actor import ActorNetwork
        actor = ActorNetwork(
            obs_dim        = config.obs.actor_dim,
            action_dim     = 4,
            hidden_dims    = tr.mappo_actor_hidden,
            activation     = tr.mappo_activation,
            use_layer_norm = tr.mappo_use_layer_norm,
            log_std_init   = -0.5,
        )

    elif algo == "masac":
        from training.algorithms.masac.actor import SACActorNetwork
        actor = SACActorNetwork(
            obs_dim     = config.obs.actor_dim,
            action_dim  = 4,
            hidden_dims = tr.masac_actor_hidden,
        )

    elif algo == "matd3":
        from training.algorithms.matd3.actor import TD3ActorNetwork
        actor = TD3ActorNetwork(
            obs_dim     = config.obs.actor_dim,
            action_dim  = 4,
            hidden_dims = tr.matd3_actor_hidden,
        )

    else:
        raise ValueError(
            f"Unknown algo '{algo}'. Valid: mappo | masac | matd3"
        )

    # ── Load weights ──────────────────────────────────────────────────────────
    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # Tìm key chứa actor state dict
    # MAPPOTrainer lưu với key "actor_state_dict"
    actor_state = None
    for key in ["actor_state_dict", "actor", "model", "state_dict"]:
        if key in ckpt:
            actor_state = ckpt[key]
            print(f"   Loaded from key: '{key}'")
            break

    if actor_state is None:
        # Thử load trực tiếp (nếu file chỉ chứa state_dict)
        if isinstance(ckpt, dict) and "weight" in str(list(ckpt.keys())):
            actor_state = ckpt
            print(f"   Loaded directly (raw state_dict)")
        else:
            raise KeyError(
                f"Không tìm thấy actor state trong checkpoint.\n"
                f"Available keys: {list(ckpt.keys())}"
            )

    # strict=False: cho phép thiếu keys (ví dụ critic không cần)
    missing, unexpected = actor.load_state_dict(actor_state, strict=False)
    if missing:
        print(f"   ⚠️  Missing keys: {missing[:3]}{'...' if len(missing)>3 else ''}")

    actor.to(device)
    actor.eval()

    # In thông tin checkpoint
    ep = ckpt.get("total_episodes_done", ckpt.get("episode", "?"))
    print(f"   ✅ {algo.upper()} loaded | episode={ep}")

    return actor


def get_deterministic_action(
    actor:  torch.nn.Module,
    obs:    np.ndarray,
    algo:   str,
    device: str = "cpu",
) -> np.ndarray:
    """
    Lấy deterministic action cho eval (no exploration noise).
    
    Returns:
        action: ndarray(4,) — [vx, vy, vz, land]
    """
    with torch.no_grad():
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)

        if algo == "mappo":
            # deterministic=True → dùng mean, không sample
            action, _ = actor.get_action(obs_t, deterministic=True)

        elif algo == "masac":
            # deterministic=True → tanh(mean), không rsample
            action, _ = actor.get_action(obs_t, deterministic=True)

        elif algo == "matd3":
            # explore_noise=0.0 → pure deterministic
            action, _ = actor.get_action(
                obs_t,
                explore_noise=0.0,
                noise_clip=0.0,
                deterministic=True,
            )

    return action.squeeze(0).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# EPISODE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_single_episode(
    env:        SARPettingZooEnv,
    actor:      torch.nn.Module,
    algo:       str,
    episode_id: int,
    seed:       int,
    stage:      str,
    device:     str = "cpu",
    max_steps:  int = None,
) -> EpisodeResult:
    """
    Chạy 1 episode eval với deterministic policy.
    
    FIX: Normalize coverage về [0, 1] nếu nhận được [0, 100].
    """
    t_start = time.time()
    ep_seed = (seed + episode_id * 7919) % (2 ** 31)
    obs_dict, info = env.reset(seed=ep_seed)

    total_reward = 0.0
    step_count   = 0
    done         = False

    n_landings      = 0
    battery_deaths  = 0
    collision_count = 0

    _max_steps = max_steps or env.cfg.env.max_steps

    while not done and step_count < _max_steps:
        actions = {}
        for agent_id, obs in obs_dict.items():
            action = get_deterministic_action(actor, obs, algo, device)
            actions[agent_id] = action

        obs_dict, rewards, terminations, truncations, infos = env.step(actions)

        if rewards:
            step_reward   = np.mean(list(rewards.values()))
            total_reward += step_reward

        step_count += 1
        done = any(terminations.values()) or any(truncations.values())

    wall_time = time.time() - t_start

    # ── Extract final metrics (FIX: normalize coverage) ───────────────────────
    ep_metrics = {}
    last_info  = {}

    if infos:
        first_agent = next(iter(infos))
        last_info   = infos[first_agent]
        ep_metrics  = last_info.get("episode", {})

    # ✅ FIX: Normalize coverage
    coverage_raw = float(
        ep_metrics.get("coverage_rate",
        last_info.get("coverage_rate",
        last_info.get("coverage", 0.0)))
    )
    
    # Detect nếu là percentage [0, 100] → normalize về [0, 1]
    if coverage_raw > 1.0:
        coverage = coverage_raw / 100.0
    else:
        coverage = coverage_raw

    # Victims
    victims_found = int(
        ep_metrics.get("victims_found",
        last_info.get("victims_found", 0))
    )
    total_victims = max(int(
        ep_metrics.get("total_victims",
        ep_metrics.get("victims_total",
        last_info.get("victims_total", 1)))
    ), 1)
    
    # Landings
    n_landings = int(ep_metrics.get("total_landings", 0))
    battery_deaths = int(ep_metrics.get("battery_deaths", 0))
    collision_count = int(ep_metrics.get("collision_obstacle", 0))

    victim_rate = victims_found / total_victims

    # ✅ FIX: success check với coverage normalized [0, 1]
    success = coverage >= 0.9  # 90%

    return EpisodeResult(
        episode_id      = episode_id,
        seed            = ep_seed,
        algo            = algo,
        stage           = stage,
        total_reward    = total_reward,
        coverage_rate   = coverage,  # ← [0, 1]
        victims_found   = victims_found,
        total_victims   = total_victims,
        episode_length  = step_count,
        n_landings      = n_landings,
        battery_deaths  = battery_deaths,
        collision_count = collision_count,
        success         = success,  # ← Đúng với coverage [0, 1]
        victim_rate     = victim_rate,
        wall_time_s     = wall_time,
    )

# ══════════════════════════════════════════════════════════════════════════════
# MULTI-EPISODE EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════
# evaluate.py - THAY THẾ evaluate_algo()

def evaluate_algo(
    algo:             str,
    checkpoint_path:  str,
    config:           AppConfig,
    stage:            str,
    n_episodes:       int = 100,
    base_seed:        int = 9999,
    device:           str = "cpu",
    verbose:          bool = True,
    n_envs:           int = 1,  # ← THÊM param
) -> AlgoResult:
    """
    Eval 1 algo trên N episodes.
    
    NEW: Hỗ trợ n_envs > 1 để chạy song song.
    
    Args:
        n_envs: Số envs chạy song song (1=sequential, 4=4× faster)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {algo.upper()} on {stage.upper()}")
        print(f"  Checkpoint: {Path(checkpoint_path).name}")
        print(f"  Episodes  : {n_episodes}")
        print(f"  Parallel  : {n_envs} envs")  # ← THÊM
        print(f"{'='*60}")

    # Load actor
    actor = load_actor(checkpoint_path, algo, config, device)

    # ──────────────────────────────────────────────────────────────────────────
    # BRANCH: Sequential vs Parallel
    # ──────────────────────────────────────────────────────────────────────────
    if n_envs == 1:
        # Original sequential code
        results = _evaluate_sequential(
            env_config=config,
            actor=actor,
            algo=algo,
            n_episodes=n_episodes,
            base_seed=base_seed,
            stage=stage,
            device=device,
            verbose=verbose,
        )
    else:
        # NEW: Parallel eval
        results = _evaluate_parallel(
            env_config=config,
            actor=actor,
            algo=algo,
            n_episodes=n_episodes,
            base_seed=base_seed,
            stage=stage,
            device=device,
            n_envs=n_envs,
            verbose=verbose,
        )

    # ── Aggregate (common) ────────────────────────────────────────────────────
    rewards      = [r.total_reward for r in results]
    coverages    = [r.coverage_rate for r in results]
    victim_rates = [r.victim_rate for r in results]
    successes    = [r.success for r in results]
    lengths      = [r.episode_length for r in results]

    algo_result = AlgoResult(
        algo             = algo,
        stage            = stage,
        n_episodes       = n_episodes,
        checkpoint       = str(checkpoint_path),
        reward_mean      = float(np.mean(rewards)),
        reward_std       = float(np.std(rewards)),
        reward_median    = float(np.median(rewards)),
        coverage_mean    = float(np.mean(coverages)),
        coverage_std     = float(np.std(coverages)),
        victim_rate_mean = float(np.mean(victim_rates)),
        victim_rate_std  = float(np.std(victim_rates)),
        success_rate     = float(np.mean(successes)),
        length_mean      = float(np.mean(lengths)),
        rewards          = [float(x) for x in rewards],
        coverages        = [float(x) for x in coverages],
        victim_rates     = [float(x) for x in victim_rates],
        successes        = [bool(x) for x in successes],
    )

    if verbose:
        _print_algo_summary(algo_result)

    return algo_result


# ══════════════════════════════════════════════════════════════════════════════
# SEQUENTIAL EVAL (original code)
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_sequential(
    env_config: AppConfig,
    actor:      torch.nn.Module,
    algo:       str,
    n_episodes: int,
    base_seed:  int,
    stage:      str,
    device:     str,
    verbose:    bool,
) -> List[EpisodeResult]:
    """Sequential eval (1 episode at a time)."""
    from tqdm import tqdm
    
    env = SARPettingZooEnv(env_config, render_mode=None)
    results: List[EpisodeResult] = []
    
    pbar = tqdm(
        total=n_episodes,
        desc=f"  {algo.upper()} eval",
        unit="ep",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )
    
    for ep_id in range(n_episodes):
        result = run_single_episode(
            env        = env,
            actor      = actor,
            algo       = algo,
            episode_id = ep_id,
            seed       = base_seed,
            stage      = stage,
            device     = device,
        )
        results.append(result)
        
        # Update progress
        recent = results[-min(10, len(results)):]
        pbar.set_postfix(ordered_dict={
            "rew":  f"{np.mean([r.total_reward for r in recent]):+6.1f}",
            "cov":  f"{np.mean([r.coverage_rate for r in recent]) * 100:.1f}%",
            "vic":  f"{np.mean([r.victim_rate for r in recent]) * 100:.1f}%",
            "suc":  f"{np.mean([r.success for r in recent]) * 100:.0f}%",
        })
        pbar.update(1)
    
    pbar.close()
    env.close()
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PARALLEL EVAL (NEW - 4× faster)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# PARALLEL EVAL (FIXED - tracking bug)
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_parallel(
    env_config: AppConfig,
    actor:      torch.nn.Module,
    algo:       str,
    n_episodes: int,
    base_seed:  int,
    stage:      str,
    device:     str,
    n_envs:     int,
    verbose:    bool,
) -> List[EpisodeResult]:
    """
    Parallel eval với VectorizedEnv.
    
    FIXED: Episode tracking logic.
    """
    from tqdm import tqdm
    from env_setup.vec_env import VectorizedEnv
    
    # Create vectorized env
    vec_env = VectorizedEnv(env_config, n_envs=n_envs, start_seed=base_seed)
    
    results: List[EpisodeResult] = []
    
    # ✅ FIX: Simple counter-based tracking
    next_episode_id = 0  # Episode ID to assign when env finishes
    
    # Track per-env state
    env_rewards     = np.zeros(n_envs, dtype=np.float32)
    env_step_counts = np.zeros(n_envs, dtype=np.int32)
    env_start_times = [time.time()] * n_envs
    env_episode_ids = [-1] * n_envs  # Current episode ID per env (-1 = not started)
    
    pbar = tqdm(
        total=n_episodes,
        desc=f"  {algo.upper()} eval",
        unit="ep",
        ncols=110,
    )
    
    # Reset all envs
    obs_batch, global_obs_batch = vec_env.reset()
    
    # ✅ Assign initial episode IDs
    for i in range(min(n_envs, n_episodes)):
        env_episode_ids[i] = next_episode_id
        next_episode_id += 1
    
    while len(results) < n_episodes:
        # ── Get actions for all envs ──────────────────────────────────────────
        actions_batch = []
        
        for env_idx in range(n_envs):
            # Skip envs that finished all episodes
            if env_episode_ids[env_idx] == -1:
                # Dummy action (env won't be used)
                actions_batch.append(
                    np.zeros((env_config.env.n_uav, 4), dtype=np.float32)
                )
                continue
            
            env_obs = obs_batch[env_idx]  # [n_agents, 68]
            env_actions = {}
            
            for agent_idx in range(env_config.env.n_uav):
                agent_obs = env_obs[agent_idx]
                action = get_deterministic_action(actor, agent_obs, algo, device)
                env_actions[agent_idx] = action
            
            # Convert dict → array [n_agents, 4]
            actions_arr = np.array([
                env_actions[i] for i in range(env_config.env.n_uav)
            ])
            actions_batch.append(actions_arr)
        
        actions_batch = np.array(actions_batch)  # [n_envs, n_agents, 4]
        
        # ── Step all envs ─────────────────────────────────────────────────────
        obs_batch, global_obs_batch, rewards_batch, dones, infos = vec_env.step(
            actions_batch
        )
        
        # Accumulate
        env_rewards += rewards_batch.sum(axis=1)
        env_step_counts += 1
        
        # ── Check done ────────────────────────────────────────────────────────
        for env_idx in range(n_envs):
            # Skip inactive envs
            if env_episode_ids[env_idx] == -1:
                continue
            
            if not dones[env_idx]:
                continue
            
            # ✅ Episode done for env_idx
            ep_id = env_episode_ids[env_idx]
            
            # Extract metrics
            info = infos[env_idx]
            ep_metrics = info.get("uav_0", {}).get("episode", {})
            last_info  = info.get("uav_0", {})
            
            coverage_raw = float(
                ep_metrics.get("coverage_rate",
                last_info.get("coverage_rate",
                last_info.get("coverage", 0.0)))
            )
            coverage = _normalize_coverage(coverage_raw)
            
            victims_found = int(
                ep_metrics.get("victims_found",
                last_info.get("victims_found", 0))
            )
            total_victims = max(int(
                ep_metrics.get("total_victims",
                ep_metrics.get("victims_total",
                last_info.get("victims_total", 1)))
            ), 1)
            
            n_landings = int(ep_metrics.get("total_landings", 0))
            battery_deaths = int(ep_metrics.get("battery_deaths", 0))
            collision_count = int(ep_metrics.get("collision_obstacle", 0))
            
            victim_rate = victims_found / total_victims
            success = coverage >= 0.9
            
            wall_time = time.time() - env_start_times[env_idx]
            
            # ✅ Create result
            result = EpisodeResult(
                episode_id      = ep_id,
                seed            = base_seed + ep_id * 7919,
                algo            = algo,
                stage           = stage,
                total_reward    = float(env_rewards[env_idx]),
                coverage_rate   = coverage,
                victims_found   = victims_found,
                total_victims   = total_victims,
                episode_length  = int(env_step_counts[env_idx]),
                n_landings      = n_landings,
                battery_deaths  = battery_deaths,
                collision_count = collision_count,
                success         = success,
                victim_rate     = victim_rate,
                wall_time_s     = wall_time,
            )
            results.append(result)
            
            # Update progress
            recent = results[-min(10, len(results)):]
            pbar.set_postfix({
                "rew":  f"{np.mean([r.total_reward for r in recent]):+6.1f}",
                "cov":  f"{np.mean([r.coverage_rate for r in recent]) * 100:.1f}%",
                "vic":  f"{np.mean([r.victim_rate for r in recent]) * 100:.1f}%",
                "suc":  f"{np.mean([r.success for r in recent]) * 100:.0f}%",
            })
            pbar.update(1)
            
            # ✅ Assign next episode or mark inactive
            if next_episode_id < n_episodes:
                env_episode_ids[env_idx] = next_episode_id
                next_episode_id += 1
                # Reset tracking for new episode
                env_rewards[env_idx] = 0.0
                env_step_counts[env_idx] = 0
                env_start_times[env_idx] = time.time()
                # obs_batch[env_idx] already reset by vec_env
            else:
                # No more episodes to run
                env_episode_ids[env_idx] = -1
        
        # ✅ Early exit if all envs inactive
        if all(eid == -1 for eid in env_episode_ids):
            break
    
    pbar.close()
    vec_env.close()
    
    return results
# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def compare_algos(
    results:     Dict[str, AlgoResult],
    output_path: str = "results/eval/comparison.json",
    verbose:     bool = True,
) -> Dict:
    """
    So sánh thống kê giữa các algo.
    Dùng Wilcoxon signed-rank test (Paper 1).
    
    Args:
        results: Dict[algo_name → AlgoResult]
    
    Returns:
        Dict với p-values, effect sizes, rankings
    """
    # ── Check scipy ───────────────────────────────────────────────────────────
    try:
        from scipy import stats as scipy_stats
        HAS_SCIPY = True
    except ImportError:
        print("  ⚠️  scipy not found. pip install scipy")
        HAS_SCIPY = False

    algos = list(results.keys())

    comparison = {
        "stage":     next(iter(results.values())).stage if results else "?",
        "n_algos":   len(algos),
        "algos":     algos,
        "summary":   {},
        "pairwise":  {},
        "rankings":  {},
    }

    # ── Summary stats ─────────────────────────────────────────────────────────
    for algo, r in results.items():
        comparison["summary"][algo] = {
            "reward":      {"mean": r.reward_mean, "std": r.reward_std,
                           "median": r.reward_median},
            "coverage":    {"mean": r.coverage_mean, "std": r.coverage_std},
            "victim_rate": {"mean": r.victim_rate_mean,
                           "std": r.victim_rate_std},
            "success_rate": r.success_rate,
            "n_episodes":   r.n_episodes,
            "checkpoint":   r.checkpoint,
        }

    # ── Pairwise Wilcoxon ─────────────────────────────────────────────────────
    metric_keys = {
        "rewards":      "Episode Reward",
        "coverages":    "Coverage Rate",
        "victim_rates": "Victim Found Rate",
    }

    for i, a1 in enumerate(algos):
        for a2 in algos[i + 1:]:
            pair_key = f"{a1}_vs_{a2}"
            comparison["pairwise"][pair_key] = {}

            for mkey, mname in metric_keys.items():
                data1 = results[a1].__dict__[mkey]
                data2 = results[a2].__dict__[mkey]
                n1, n2 = len(data1), len(data2)

                test_entry = {
                    "metric": mname,
                    "n1": n1, "n2": n2,
                    "mean1": float(np.mean(data1)),
                    "mean2": float(np.mean(data2)),
                    "diff_mean": float(np.mean(data1) - np.mean(data2)),
                    "winner": a1 if np.mean(data1) > np.mean(data2) else a2,
                }

                if HAS_SCIPY:
                    if n1 == n2:
                        # Wilcoxon signed-rank (paired)
                        stat, pval = scipy_stats.wilcoxon(
                            data1, data2, alternative="two-sided"
                        )
                        test_name = "wilcoxon"
                    else:
                        # Mann-Whitney U (unpaired)
                        stat, pval = scipy_stats.mannwhitneyu(
                            data1, data2, alternative="two-sided"
                        )
                        test_name = "mannwhitneyu"

                    # Effect size r = |Z| / sqrt(N)
                    z_approx  = scipy_stats.norm.ppf(max(pval / 2, 1e-15))
                    effect_r  = abs(z_approx) / np.sqrt(max(n1, n2))

                    # Cohen's d
                    pooled_std = np.sqrt(
                        (np.std(data1) ** 2 + np.std(data2) ** 2) / 2
                    )
                    cohen_d = (
                        (np.mean(data1) - np.mean(data2))
                        / max(pooled_std, 1e-8)
                    )

                    test_entry.update({
                        "test":        test_name,
                        "stat":        float(stat),
                        "p_value":     float(pval),
                        "significant": bool(pval < 0.05),
                        "effect_r":    float(effect_r),
                        "cohens_d":    float(cohen_d),
                    })
                else:
                    test_entry["note"] = "scipy not available"

                comparison["pairwise"][pair_key][mkey] = test_entry

    # ── Rankings ──────────────────────────────────────────────────────────────
    for mkey in ["rewards", "coverages", "victim_rates"]:
        comparison["rankings"][mkey] = sorted(
            algos,
            key=lambda a: float(np.mean(results[a].__dict__[mkey])),
            reverse=True,
        )

    # ── Print table ───────────────────────────────────────────────────────────
    if verbose:
        _print_comparison_table(results, comparison, metric_keys)

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2, default=_json_safe)
    print(f"\n  💾 Statistical results → {output_path}")

    return comparison


def _print_comparison_table(results, comparison, metric_keys):
    algos = list(results.keys())

    print(f"\n{'═'*68}")
    print(f"  STATISTICAL COMPARISON  —  {comparison['stage'].upper()}")
    print(f"{'═'*68}")

    # Header
    hdr = f"  {'Metric':<22}"
    for a in algos:
        hdr += f"  {a.upper():>14}"
    print(hdr)
    print(f"  {'-'*22}" + f"  {'-'*14}" * len(algos))

    # Rows
    rows = [
        ("Reward (mean±std)",   "reward_mean",      "reward_std",      ".1f"),
        ("Coverage (mean±std)", "coverage_mean",     "coverage_std",    ".4f"),
        ("VictimRate(mean±std)","victim_rate_mean",  "victim_rate_std", ".4f"),
        ("Success Rate",        "success_rate",      None,              ".1%"),
        ("Episode Length",      "length_mean",       None,              ".1f"),
    ]

    for label, mean_key, std_key, fmt in rows:
        row = f"  {label:<22}"
        for algo in algos:
            r   = results[algo]
            val = getattr(r, mean_key)
            if std_key:
                std = getattr(r, std_key)
                row += f"  {val:>8{fmt}}±{std:>{fmt}}"
            else:
                row += f"  {val:>14{fmt}}"
        print(row)

    # Pairwise p-values
    print(f"\n  {'─'*30} Wilcoxon p-values {'─'*18}")
    for pair_key, pair_data in comparison.get("pairwise", {}).items():
        a1, _, a2 = pair_key.partition("_vs_")
        print(f"\n  {a1.upper()} vs {a2.upper()}:")
        for mkey, test in pair_data.items():
            if "p_value" in test:
                sig  = "✅ SIG" if test["significant"] else "     "
                win  = test.get("winner", "?")
                d    = test.get("cohens_d", 0)
                p    = test["p_value"]
                print(
                    f"    {sig}  {test['metric']:<20}: "
                    f"p={p:.4f}  d={d:+.3f}  winner={win}"
                )

    print(f"\n  Rankings (best → worst):")
    for mkey, ranked in comparison.get("rankings", {}).items():
        print(f"    {mkey:<15}: {' > '.join(ranked)}")

    print(f"{'═'*68}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-SEED (Paper 1)
# ══════════════════════════════════════════════════════════════════════════════

def find_checkpoints(algo_dir: str) -> List[str]:
    """
    Tìm checkpoint_final.pt trong các subdirectories.
    
    Expected structure:
        algo_dir/
        ├── run_s42/checkpoints/checkpoint_final.pt
        ├── run_s123/checkpoints/checkpoint_final.pt
        └── ...
    """
    base = Path(algo_dir)

    # Tìm checkpoint_final.pt trước
    finals = sorted(base.rglob("checkpoint_final.pt"))
    if finals:
        return [str(p) for p in finals]

    # Fallback: tìm checkpoint ep lớn nhất trong mỗi subdir
    checkpoints = sorted(base.rglob("checkpoint_ep*.pt"))
    if checkpoints:
        # Nhóm theo thư mục, lấy ep lớn nhất mỗi thư mục
        by_dir: Dict[str, List[Path]] = {}
        for p in checkpoints:
            key = str(p.parent)
            by_dir.setdefault(key, []).append(p)

        result = []
        for ckpts in by_dir.values():
            result.append(str(sorted(ckpts)[-1]))
        return sorted(result)

    return []


def evaluate_multi_seed(
    algo_checkpoints: Dict[str, List[str]],
    config:           AppConfig,
    stage:            str,
    n_eps_per_seed:   int = 50,
    base_seed:        int = 9999,
    device:           str = "cpu",
    output_dir:       str = "results/eval_multiseed",
) -> Dict:
    """
    Eval nhiều seeds cho Paper 1.
    
    Args:
        algo_checkpoints: {
            "mappo": [path_seed42, path_seed123, ...],
            "masac": [...],
            "matd3": [...],
        }
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Eval seeds khác với training seeds để tránh data leakage
    eval_seeds = [10001, 20002, 30003, 40004, 50005]

    all_results: Dict[str, List[AlgoResult]] = {}

    for algo, ckpt_list in algo_checkpoints.items():
        print(f"\n{'#'*60}")
        print(f"# Multi-seed: {algo.upper()} ({len(ckpt_list)} seeds)")
        print(f"{'#'*60}")

        all_results[algo] = []

        for seed_idx, ckpt_path in enumerate(ckpt_list):
            eval_seed = eval_seeds[seed_idx % len(eval_seeds)]

            print(f"\n  Seed {seed_idx+1}/{len(ckpt_list)}: "
                  f"{Path(ckpt_path).parent.parent.name}")

            r = evaluate_algo(
                algo            = algo,
                checkpoint_path = ckpt_path,
                config          = config,
                stage           = stage,
                n_episodes      = n_eps_per_seed,
                base_seed       = eval_seed,
                device          = device,
                verbose         = True,
            )
            all_results[algo].append(r)

            # Save per-seed
            seed_file = out_dir / f"{algo}_seed{seed_idx}.json"
            with open(seed_file, "w") as f:
                json.dump(asdict(r), f, indent=2, default=_json_safe)

    # ── Aggregate across seeds ────────────────────────────────────────────────
    aggregated: Dict[str, AlgoResult] = {}

    for algo, seed_results in all_results.items():
        # Pool tất cả episodes
        all_rewards      = []
        all_coverages    = []
        all_victim_rates = []
        all_successes    = []

        for sr in seed_results:
            all_rewards.extend(sr.rewards)
            all_coverages.extend(sr.coverages)
            all_victim_rates.extend(sr.victim_rates)
            all_successes.extend(sr.successes)

        # std across seed means (not within-seed std)
        seed_means_rew = [np.mean(sr.rewards) for sr in seed_results]
        seed_means_cov = [np.mean(sr.coverages) for sr in seed_results]
        seed_means_vic = [np.mean(sr.victim_rates) for sr in seed_results]

        aggregated[algo] = AlgoResult(
            algo             = algo,
            stage            = stage,
            n_episodes       = len(all_rewards),
            checkpoint       = f"multi-seed ({len(seed_results)} seeds)",
            reward_mean      = float(np.mean(all_rewards)),
            reward_std       = float(np.std(seed_means_rew)),  # across seeds
            reward_median    = float(np.median(all_rewards)),
            coverage_mean    = float(np.mean(all_coverages)),
            coverage_std     = float(np.std(seed_means_cov)),
            victim_rate_mean = float(np.mean(all_victim_rates)),
            victim_rate_std  = float(np.std(seed_means_vic)),
            success_rate     = float(np.mean(all_successes)),
            length_mean      = float(np.mean(
                [sr.length_mean for sr in seed_results]
            )),
            rewards      = [float(x) for x in all_rewards],
            coverages    = [float(x) for x in all_coverages],
            victim_rates = [float(x) for x in all_victim_rates],
            successes    = [bool(x) for x in all_successes],
        )

    # ── Statistical comparison ────────────────────────────────────────────────
    stats = compare_algos(
        aggregated,
        output_path=str(out_dir / "statistical_comparison.json"),
    )

    return {
        "aggregated": {a: asdict(r) for a, r in aggregated.items()},
        "per_seed":   {
            a: [asdict(r) for r in rs]
            for a, rs in all_results.items()
        },
        "statistics": stats,
    }


# ══════════════════════════════════════════════════════════════════════════════
# QUICK EVAL (Kaggle notebook shortcut)
# ══════════════════════════════════════════════════════════════════════════════

def quick_eval(
    checkpoints: Dict[str, str],
    stage:       str = "extreme",
    n_episodes:  int = 50,
    base_seed:   int = 9999,
    device:      str = "cpu",
    output_dir:  str = "results/eval",
    n_uav:       int = 4,
    n_envs:      int = 1
) -> Dict[str, AlgoResult]:
    """
    Shortcut function cho Kaggle notebook.
    
    Usage:
        from evaluate import quick_eval
        
        results = quick_eval(
            checkpoints={
                "mappo": "/kaggle/working/results/mappo/.../checkpoint_final.pt",
                "masac": "/kaggle/working/results/masac/.../checkpoint_final.pt",
                "matd3": "/kaggle/working/results/matd3/.../checkpoint_final.pt",
            },
            stage="extreme",   # hoặc "hard", "transfer"
            n_episodes=50,
        )
        
        # Access results
        print(f"MAPPO coverage: {results['mappo'].coverage_mean:.3f}")
        print(f"MASAC success:  {results['masac'].success_rate:.1%}")
    
    Returns:
        Dict[algo → AlgoResult]
    """
    print(f"\n🚀 Quick Eval: {stage.upper()} stage | {n_episodes} episodes")
    print(f"   Algos: {list(checkpoints.keys())}")

    cfg  = build_eval_config(stage, n_uav)
    algo_results: Dict[str, AlgoResult] = {}

    for algo, ckpt_path in checkpoints.items():
        if not Path(ckpt_path).exists():
            print(f"  ⚠️  {algo}: checkpoint not found: {ckpt_path}")
            continue

        r = evaluate_algo(
            algo            = algo,
            checkpoint_path = ckpt_path,
            config          = cfg,
            stage           = stage,
            n_episodes      = n_episodes,
            base_seed       = base_seed,
            device          = device,
            n_envs          = n_envs,
            verbose         = True,
        )
        algo_results[algo] = r

    # Compare nếu có >= 2 algos
    if len(algo_results) >= 2:
        out = str(Path(output_dir) / f"comparison_{stage}.json")
        compare_algos(algo_results, output_path=out)

    return algo_results


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _print_algo_summary(result):
    """
    In tóm tắt kết quả eval 1 algo.
    
    Args:
        result: AlgoResult dataclass
    """
    print(f"\n{'─'*65}")
    print(f"  📊 {result.algo.upper()} Results ({result.stage.upper()})")
    print(f"{'─'*65}")
    print(f"  Episodes     : {result.n_episodes:>6,}")
    print(f"  Checkpoint   : {Path(result.checkpoint).name}")
    print(f"\n  Performance:")
    print(f"    Reward       : {result.reward_mean:>8.1f} ± {result.reward_std:>6.1f}  (median: {result.reward_median:>7.1f})")
    
    # ✅ FIX: Handle coverage ở cả [0,1] và [0,100] range
    cov_mean = result.coverage_mean
    cov_std  = result.coverage_std
    if cov_mean <= 1.0:  # Đã normalized [0, 1]
        cov_mean *= 100
        cov_std  *= 100
    
    vic_mean = result.victim_rate_mean
    vic_std  = result.victim_rate_std
    if vic_mean <= 1.0:
        vic_mean *= 100
        vic_std  *= 100
    
    suc_rate = result.success_rate
    if suc_rate <= 1.0:
        suc_rate *= 100
    
    print(f"    Coverage     : {cov_mean:>7.1f}% ± {cov_std:>5.1f}%")
    print(f"    Victim Rate  : {vic_mean:>7.1f}% ± {vic_std:>5.1f}%")
    print(f"    Success Rate : {suc_rate:>7.1f}%")
    print(f"    Episode Len  : {result.length_mean:>7.1f} steps")
    
    # Distribution stats
    if result.rewards:
        print(f"\n  Distribution (n={len(result.rewards)}):")
        print(f"    Reward   : min={min(result.rewards):>7.1f}  max={max(result.rewards):>7.1f}")
        
        cov_vals = [x * 100 if x <= 1.0 else x for x in result.coverages]
        print(f"    Coverage : min={min(cov_vals):>6.1f}%  max={max(cov_vals):>6.1f}%")
        
        vic_vals = [x * 100 if x <= 1.0 else x for x in result.victim_rates]
        print(f"    Victims  : min={min(vic_vals):>6.1f}%  max={max(vic_vals):>6.1f}%")
        
        n_success = sum(result.successes)
        print(f"    Success  : {n_success}/{len(result.successes)} episodes")
    
    print(f"{'─'*65}\n")

def _json_safe(obj):
    """JSON serializer cho numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, bool):
        return bool(obj)
    return str(obj)

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Eval MAPPO/MASAC/MATD3 on HARD/EXTREME stage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Single checkpoint
    p.add_argument("--mappo", type=str, default=None,
                   help="Path đến MAPPO checkpoint_final.pt")
    p.add_argument("--masac", type=str, default=None,
                   help="Path đến MASAC checkpoint_final.pt")
    p.add_argument("--matd3", type=str, default=None,
                   help="Path đến MATD3 checkpoint_final.pt")

    # Multi-seed dirs
    p.add_argument("--mappo-dir", type=str, default=None,
                   help="Dir chứa MAPPO multi-seed checkpoints")
    p.add_argument("--masac-dir", type=str, default=None,
                   help="Dir chứa MASAC multi-seed checkpoints")
    p.add_argument("--matd3-dir", type=str, default=None,
                   help="Dir chứa MATD3 multi-seed checkpoints")

    # Stage
    p.add_argument("--stage", type=str, default="hard",
                   choices=["hard", "extreme", "transfer"],
                   help="Eval stage (default: hard)")

    # Eval params
    p.add_argument("--n-episodes", type=int, default=100)
    p.add_argument("--n-eps-per-seed", type=int, default=50)
    p.add_argument("--seed", type=int, default=9999)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-uav", type=int, default=4)
    p.add_argument("--n-envs", type=int, default=1,  # ← THÊM DÒNG NÀY
                   help="Number of parallel envs (1=sequential, 4=4x faster)")
    
    # Output
    p.add_argument("--output-dir", type=str, default="results/eval")

    # Mode
    p.add_argument("--multi-seed", action="store_true",
                   help="Multi-seed eval mode")

    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'═'*60}")
    print(f"  SAR UAV SWARM — EVALUATION")
    print(f"{'═'*60}")
    print(f"  Stage     : {args.stage}")
    print(f"  Episodes  : {args.n_episodes}")
    print(f"  Parallel  : {args.n_envs} envs")  # ← THÊM
    print(f"  Device    : {args.device}")
    print(f"  Seed      : {args.seed}")
    print(f"  Multi-seed: {args.multi_seed}")

    cfg = build_eval_config(args.stage, args.n_uav)

    # ── Multi-seed mode ───────────────────────────────────────────────────────
    if args.multi_seed:
        algo_checkpoints = {}
        for algo, algo_dir in [
            ("mappo", args.mappo_dir),
            ("masac", args.masac_dir),
            ("matd3", args.matd3_dir),
        ]:
            if algo_dir is None:
                continue
            ckpts = find_checkpoints(algo_dir)
            if not ckpts:
                print(f"  ⚠️  No checkpoints in: {algo_dir}")
                continue
            algo_checkpoints[algo] = ckpts
            print(f"  {algo.upper()}: {len(ckpts)} checkpoints found")

        if not algo_checkpoints:
            print("  ❌ No checkpoints!")
            return

        evaluate_multi_seed(
            algo_checkpoints = algo_checkpoints,
            config           = cfg,
            stage            = args.stage,
            n_eps_per_seed   = args.n_eps_per_seed,
            base_seed        = args.seed,
            device           = args.device,
            output_dir       = args.output_dir,
        )

    # ── Single-seed mode ──────────────────────────────────────────────────────
    else:
        checkpoints = {}
        for algo, ckpt_path in [
            ("mappo", args.mappo),
            ("masac", args.masac),
            ("matd3", args.matd3),
        ]:
            if ckpt_path is None:
                continue
            if not Path(ckpt_path).exists():
                print(f"  ⚠️  Not found: {ckpt_path}")
                continue
            checkpoints[algo] = ckpt_path

        if not checkpoints:
            print("  ❌ No checkpoints!")
            return

        # ✅ FIX: CHỈ GỌI 1 LẦN!
        algo_results = {}
        for algo, ckpt_path in checkpoints.items():
            r = evaluate_algo(
                algo            = algo,
                checkpoint_path = ckpt_path,
                config          = cfg,
                stage           = args.stage,
                n_episodes      = args.n_episodes,
                base_seed       = args.seed,
                device          = args.device,
                n_envs          = args.n_envs,  # ← Dùng n_envs từ args
                verbose         = True,
            )
            algo_results[algo] = r
        
        # ✅ FIX: Compare với output_path
        if len(algo_results) >= 2:
            out = str(Path(args.output_dir) / f"comparison_{args.stage}.json")
            compare_algos(algo_results, output_path=out)

        # ❌ XÓA: Không gọi quick_eval() nữa!

    print(f"\n{'═'*60}")
    print(f"  ✅ Done! Results in: {args.output_dir}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()