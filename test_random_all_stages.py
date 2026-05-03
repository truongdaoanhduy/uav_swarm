"""
test_random_parallel.py
Random Policy Baseline Test - OPTIMIZED VERSION

FEATURES:
- Parallel processing với joblib (8 cores)
- NumPy vectorization optimization
- Better progress tracking với ETA
- Error handling cho failed episodes
- Auto-save results to JSON
- Clean console output
- Summary statistics

PERFORMANCE:
- ~6-7s per episode (with NumPy optimization)
- ~6 minutes per 50 episodes (parallel)
- ~18 minutes total (3 stages)

USAGE:
    python test_random_parallel.py
"""

import os
os.environ['MPLBACKEND'] = 'Agg'  # Non-interactive backend

import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import time
import json
from datetime import datetime
from joblib import Parallel, delayed

# Disable all logging except CRITICAL errors
logging.basicConfig(level=logging.CRITICAL)

from config import AppConfig
from config.curriculum_config import STAGE_EASY, STAGE_MEDIUM, STAGE_HARD
from env import SARBaseEnv


# ══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def run_single_episode(stage_config, episode_seed: int) -> Dict:
    """
    Chạy 1 episode với random policy.
    
    Args:
        stage_config: StageConfig object
        episode_seed: Random seed cho episode
    
    Returns:
        Dict với metrics hoặc error info
    """
    try:
        # Create environment
        cfg = AppConfig()
        # ✅ FIX: Dùng apply_stage() thay vì gán thủ công
        cfg.apply_stage(stage_config)
        
        env = SARBaseEnv(cfg)
        
        # Reset environment
        obs, info = env.reset(seed=episode_seed)
        
        # Run episode
        total_reward = 0.0
        done = False
        step = 0
        
        while not done and step < cfg.env.max_steps:
            # Random actions
            actions = {i: env.action_space.sample() for i in range(cfg.env.n_uav)}
            
            # Step
            obs, rewards_dict, dones, truncs, infos = env.step(actions)
            
            # ✅ FIX: Dùng MEAN thay vì SUM (team reward, không nhân hệ số swarm)
            global_r = infos.get("rewards_breakdown", {}).get("total", 0.0)
            total_reward += float(global_r)
            
            # Check done
            if isinstance(dones, dict):
                done = all(dones.values()) or all(truncs.values())
            else:
                done = dones or truncs
            
            step += 1
        
        # Get final metrics
        state = env.unwrapped.backend.get_state()
        cov_map = state["coverage_map"]
        victims = state["victims"]
        
        coverage = cov_map.get_coverage_rate()
        n_found = sum(1 for v in victims if v.is_found)
        n_total = len(victims)
        victim_rate = n_found / n_total if n_total > 0 else 0.0
        
        # Cleanup
        env.close()
        
        return {
            "success": True,
            "coverage": coverage,
            "victim_rate": victim_rate,
            "reward": total_reward,
            "steps": step,
            "n_found": n_found,
            "n_total": n_total,
        }
    
    except Exception as e:
        # Handle errors gracefully
        return {
            "success": False,
            "error": str(e),
            "coverage": 0.0,
            "victim_rate": 0.0,
            "reward": -1000.0,
            "steps": 0,
            "n_found": 0,
            "n_total": 0,
        }


def format_time(seconds: float) -> str:
    """Format seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


# ══════════════════════════════════════════════════════════════════════════
# MAIN TEST FUNCTION
# ══════════════════════════════════════════════════════════════════════════

def test_random_policy_parallel(
    stage_config,
    n_episodes: int = 50,
    n_jobs: int = -1,
    verbose: bool = True,
) -> Dict:
    """
    Test random policy với parallel processing.
    
    Args:
        stage_config: StageConfig object
        n_episodes: Số episodes
        n_jobs: Số parallel jobs (-1 = all cores)
        verbose: Print progress hay không
    
    Returns:
        Dict với aggregated metrics
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"  Testing {stage_config.name.upper()} Stage")
        print(f"{'='*70}")
        print(f"  Config: {stage_config.describe()}")
        print(f"  Episodes: {n_episodes}")
        print(f"  Parallel jobs: {n_jobs} (-1 = all cores)")
        print(f"{'='*70}\n")
    
    start_time = time.time()
    
    # Run parallel episodes
    if verbose:
        print(f"  Running {n_episodes} episodes in parallel...")
    
    results = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
        delayed(run_single_episode)(stage_config, seed) 
        for seed in range(n_episodes)
    )
    
    elapsed = time.time() - start_time
    
    # Filter successful episodes
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    if len(failed) > 0 and verbose:
        print(f"\n  ⚠️  WARNING: {len(failed)}/{n_episodes} episodes failed!")
        for i, f in enumerate(failed[:3]):  # Show first 3 errors
            print(f"    Error {i+1}: {f['error'][:80]}...")
    
    # Extract metrics from successful episodes
    if len(successful) == 0:
        raise RuntimeError("All episodes failed! Check environment setup.")
    
    coverages = [r["coverage"] for r in successful]
    victims_rates = [r["victim_rate"] for r in successful]
    rewards = [r["reward"] for r in successful]
    steps_list = [r["steps"] for r in successful]
    n_found_list = [r["n_found"] for r in successful]
    n_total_list = [r["n_total"] for r in successful]
    
    # Compute statistics
    summary = {
        "stage": stage_config.name,
        "n_episodes": n_episodes,
        "n_successful": len(successful),
        "n_failed": len(failed),
        
        # Coverage
        "coverage_mean": float(np.mean(coverages)),
        "coverage_std": float(np.std(coverages)),
        "coverage_min": float(np.min(coverages)),
        "coverage_max": float(np.max(coverages)),
        
        # Victims
        "victims_mean": float(np.mean(victims_rates)),
        "victims_std": float(np.std(victims_rates)),
        "victims_min": float(np.min(victims_rates)),
        "victims_max": float(np.max(victims_rates)),
        
        # Reward
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        
        # Steps
        "steps_mean": float(np.mean(steps_list)),
        "steps_std": float(np.std(steps_list)),
        
        # Absolute counts
        "found_mean": float(np.mean(n_found_list)),
        "total_mean": float(np.mean(n_total_list)),
        
        # Performance
        "elapsed_time": float(elapsed),
        "time_per_episode": float(elapsed / n_episodes),
        
        # Raw data (for plotting)
        "coverages": coverages,
        "victims_rates": victims_rates,
        "rewards": rewards,
        "steps": steps_list,
    }
    
    # Print results
    if verbose:
        print(f"\n{'─'*70}")
        print(f"  RESULTS - {stage_config.name.upper()}")
        print(f"{'─'*70}")
        print(f"  Success rate:  {len(successful)}/{n_episodes} episodes")
        print(f"  Coverage:      {summary['coverage_mean']*100:5.1f}% ± {summary['coverage_std']*100:4.1f}%")
        print(f"                 (min={summary['coverage_min']*100:.1f}%, max={summary['coverage_max']*100:.1f}%)")
        print(f"  Victims found: {summary['victims_mean']*100:5.1f}% ± {summary['victims_std']*100:4.1f}%")
        print(f"                 ({summary['found_mean']:.1f}/{summary['total_mean']:.1f} avg)")
        print(f"  Reward:        {summary['reward_mean']:7.1f} ± {summary['reward_std']:6.1f}")
        print(f"                 (min={summary['reward_min']:.1f}, max={summary['reward_max']:.1f})")
        print(f"  Steps:         {summary['steps_mean']:.0f} ± {summary['steps_std']:.0f}")
        print(f"  Time:          {format_time(elapsed)} ({summary['time_per_episode']:.2f}s/ep)")
        print(f"{'─'*70}\n")
    
    return summary


# ══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════

def plot_comparison(results_list: List[Dict], save_path: str = "results/random_policy_parallel.png") -> None:
    """
    Plot comparison của 3 stages.
    
    Args:
        results_list: List of summary dicts
        save_path: Output path
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Random Policy Baseline - All Curriculum Stages", 
                 fontsize=14, fontweight="bold")
    
    stages = [r["stage"] for r in results_list]
    colors = ["#AED6F1", "#A9DFBF", "#F9E79F"]  # EASY, MEDIUM, HARD
    
    # ── Coverage Rate ──
    ax = axes[0, 0]
    for i, r in enumerate(results_list):
        episodes = range(len(r["coverages"]))
        ax.plot(episodes, np.array(r["coverages"]) * 100, 
                label=r["stage"].upper(),
                color=colors[i], alpha=0.6, linewidth=1.5)
        
        # Add mean line
        ax.axhline(y=r["coverage_mean"]*100, 
                   color=colors[i], linestyle="--", linewidth=1.0, alpha=0.8)
    
    ax.set_title("Coverage Rate", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Coverage (%)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)
    
    # ── Victim Found Rate ──
    ax = axes[0, 1]
    for i, r in enumerate(results_list):
        episodes = range(len(r["victims_rates"]))
        ax.plot(episodes, np.array(r["victims_rates"]) * 100,
                label=r["stage"].upper(),
                color=colors[i], alpha=0.6, linewidth=1.5)
        
        # Add mean line
        ax.axhline(y=r["victims_mean"]*100,
                   color=colors[i], linestyle="--", linewidth=1.0, alpha=0.8)
    
    ax.set_title("Victim Found Rate", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Found Rate (%)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)
    
    # ── Episode Reward ──
    ax = axes[1, 0]
    for i, r in enumerate(results_list):
        episodes = range(len(r["rewards"]))
        ax.plot(episodes, r["rewards"],
                label=r["stage"].upper(),
                color=colors[i], alpha=0.6, linewidth=1.5)
        
        # Add mean line
        ax.axhline(y=r["reward_mean"],
                   color=colors[i], linestyle="--", linewidth=1.0, alpha=0.8)
    
    ax.set_title("Episode Reward", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    
    # ── Bar Comparison ──
    ax = axes[1, 1]
    x = np.arange(len(stages))
    width = 0.35
    
    cov_means = [r["coverage_mean"] * 100 for r in results_list]
    vic_means = [r["victims_mean"] * 100 for r in results_list]
    
    bars1 = ax.bar(x - width/2, cov_means, width, label="Coverage",
                   color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    bars2 = ax.bar(x + width/2, vic_means, width, label="Victims Found",
                   color=colors, alpha=0.6, edgecolor="black", linewidth=1.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_title("Average Performance", fontweight="bold")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in stages])
    ax.legend(loc="best")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved comparison plot: {save_path}")
    
    plt.close(fig)


def save_results_json(results_list: List[Dict], save_path: str = "results/random_baseline_results.json") -> None:
    """
    Save results to JSON file.
    
    Args:
        results_list: List of summary dicts
        save_path: Output path
    """
    # Remove raw data arrays (too large for JSON)
    clean_results = []
    for r in results_list:
        clean = {k: v for k, v in r.items() 
                 if k not in ["coverages", "victims_rates", "rewards", "steps"]}
        clean_results.append(clean)
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "test_config": {
            "n_episodes_per_stage": results_list[0]["n_episodes"],
            "policy": "random_uniform",
            "parallel": True,
        },
        "results": clean_results,
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✅ Saved results JSON: {save_path}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Print header
    print("\n" + "="*70)
    print("  RANDOM POLICY BASELINE - PARALLEL PROCESSING")
    print("="*70)
    
    # System info
    import multiprocessing
    n_cores = multiprocessing.cpu_count()
    print(f"\n  System: {n_cores} CPU cores detected")
    print(f"  NumPy optimization: ENABLED")
    print(f"  Parallel backend: joblib")
    print(f"  Using: ALL cores (n_jobs=-1)\n")
    
    # Test settings
    N_EPISODES = 50
    N_JOBS = -1  # Use all cores
    
    # Test all stages
    results = []
    total_start = time.time()
    
    for i, stage in enumerate([STAGE_EASY, STAGE_MEDIUM, STAGE_HARD]):
        print(f"\n{'▶'*35}")
        print(f"  STAGE {i+1}/3: {stage.name.upper()}")
        print(f"{'▶'*35}")
        
        result = test_random_policy_parallel(
            stage_config=stage,
            n_episodes=N_EPISODES,
            n_jobs=N_JOBS,
            verbose=True,
        )
        results.append(result)
    
    total_elapsed = time.time() - total_start
    
    # Generate plots
    print(f"\n{'='*70}")
    print("  GENERATING PLOTS")
    print(f"{'='*70}\n")
    
    plot_comparison(results)
    
    # Save JSON
    save_results_json(results)
    
    # Final summary
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Stage':<10} | {'Coverage':<18} | {'Victims':<18} | {'Reward':<15}")
    print("-"*70)
    
    for r in results:
        print(f"{r['stage'].upper():<10} | "
              f"{r['coverage_mean']*100:5.1f}% ± {r['coverage_std']*100:4.1f}% "
              f"[{r['coverage_min']*100:4.1f}-{r['coverage_max']*100:4.1f}] | "
              f"{r['victims_mean']*100:5.1f}% ± {r['victims_std']*100:4.1f}% "
              f"[{r['victims_min']*100:4.1f}-{r['victims_max']*100:4.1f}] | "
              f"{r['reward_mean']:7.1f} ± {r['reward_std']:5.1f}")
    
    print("="*70)
    print(f"\n  Total episodes:  {N_EPISODES * 3}")
    print(f"  Total time:      {format_time(total_elapsed)}")
    print(f"  Average time:    {total_elapsed/(N_EPISODES*3):.2f}s per episode")
    print(f"  Throughput:      {(N_EPISODES*3)/total_elapsed:.2f} episodes/second")
    
    # Analysis
    print(f"\n{'='*70}")
    print("  ANALYSIS")
    print(f"{'='*70}\n")
    
    # Check difficulty progression
    cov_easy = results[0]["coverage_mean"]
    cov_med = results[1]["coverage_mean"]
    cov_hard = results[2]["coverage_mean"]
    
    if cov_easy > cov_med > cov_hard:
        print("  ✅ Difficulty progression: CORRECT (EASY > MEDIUM > HARD)")
    else:
        print("  ⚠️  Difficulty progression: INCORRECT!")
        print(f"     EASY={cov_easy*100:.1f}% > MEDIUM={cov_med*100:.1f}% > HARD={cov_hard*100:.1f}%")
    
    # Gap for RL
    print(f"\n  Random baseline performance:")
    print(f"    EASY:   {cov_easy*100:5.1f}% coverage")
    print(f"    MEDIUM: {cov_med*100:5.1f}% coverage")
    print(f"    HARD:   {cov_hard*100:5.1f}% coverage")
    print(f"\n  Target RL performance (estimate):")
    print(f"    EASY:   85-90% coverage  (gap: +{85-cov_easy*100:.0f}%)")
    print(f"    MEDIUM: 75-80% coverage  (gap: +{75-cov_med*100:.0f}%)")
    print(f"    HARD:   65-70% coverage  (gap: +{65-cov_hard*100:.0f}%)")
    
    print(f"\n  → RL has clear opportunity to improve! ✅")
    
    print(f"\n{'='*70}\n")
    
    print("✅ Baseline test complete!")
    print("\nNext steps:")
    print("  1. Review results in: results/random_policy_parallel.png")
    print("  2. Check JSON data: results/random_baseline_results.json")
    print("  3. Document findings in: docs/baseline_results.md")
    print("  4. Proceed to Phase 2: RL algorithm implementation\n")