"""
analysis/reward_analyzer.py
Tool để analyze reward distribution và identify outliers
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


def analyze_reward_distribution(
    episode_rewards: List[float],
    stage_name: str = "UNKNOWN",
    save_path: str = None,
) -> Dict:
    """
    Analyze reward distribution và identify outliers.
    
    Args:
        episode_rewards: List of episode rewards
        stage_name: Stage name for labeling
        save_path: Path to save plot (optional)
    
    Returns:
        Dict với statistics + outlier info
    """
    rewards = np.array(episode_rewards)
    
    # Basic stats
    mean = np.mean(rewards)
    std = np.std(rewards)
    median = np.median(rewards)
    q1, q3 = np.percentile(rewards, [25, 75])
    iqr = q3 - q1
    
    # Outlier detection (IQR method)
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers_low = rewards[rewards < lower_bound]
    outliers_high = rewards[rewards > upper_bound]
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(rewards, bins=30, alpha=0.7, edgecolor='black')
    axes[0].axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.1f}')
    axes[0].axvline(median, color='green', linestyle='--', label=f'Median: {median:.1f}')
    axes[0].axvline(lower_bound, color='orange', linestyle=':', label=f'Lower bound: {lower_bound:.1f}')
    axes[0].axvline(upper_bound, color='orange', linestyle=':', label=f'Upper bound: {upper_bound:.1f}')
    axes[0].set_xlabel('Episode Reward')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{stage_name} - Reward Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot
    axes[1].boxplot(rewards, vert=True)
    axes[1].set_ylabel('Episode Reward')
    axes[1].set_title(f'{stage_name} - Box Plot')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved reward analysis plot: {save_path}")
    
    plt.close()
    
    # Return statistics
    return {
        "mean": float(mean),
        "std": float(std),
        "median": float(median),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "n_outliers_low": len(outliers_low),
        "n_outliers_high": len(outliers_high),
        "outliers_low": outliers_low.tolist() if len(outliers_low) > 0 else [],
        "outliers_high": outliers_high.tolist() if len(outliers_high) > 0 else [],
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
    }


# Example usage in test script
if __name__ == "__main__":
    # Mock data
    import json
    
    # Load from baseline results
    with open("results/random_baseline_results.json", "r") as f:
        data = json.load(f)
    
    for stage_name, stage_data in data.items():
        if stage_name in ["EASY", "MEDIUM", "HARD"]:
            rewards = stage_data["rewards"]
            
            stats = analyze_reward_distribution(
                rewards,
                stage_name=stage_name,
                save_path=f"results/reward_analysis_{stage_name}.png"
            )
            
            print(f"\n{'='*60}")
            print(f"  {stage_name} - Reward Analysis")
            print(f"{'='*60}")
            print(f"  Mean:   {stats['mean']:>8.1f}")
            print(f"  Median: {stats['median']:>8.1f}")
            print(f"  Std:    {stats['std']:>8.1f}")
            print(f"  Min:    {stats['min']:>8.1f}")
            print(f"  Max:    {stats['max']:>8.1f}")
            print(f"  IQR:    {stats['iqr']:>8.1f}")
            print(f"  Outliers (low):  {stats['n_outliers_low']}")
            print(f"  Outliers (high): {stats['n_outliers_high']}")
            
            if stats['n_outliers_low'] > 0:
                print(f"\n  ⚠️  Low outliers: {stats['outliers_low'][:3]}...")