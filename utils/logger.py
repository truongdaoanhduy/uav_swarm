import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import time
import json


class EpisodeLogger:
    """
    Logger cho một episode - CHỈ LƯU DATA, KHÔNG IN
    v2.0: Thêm reward breakdown accumulation
    """

    def __init__(self, episode_id: int, seed: Optional[int] = None):
        self.episode_id   = episode_id
        self.seed         = seed
        self.start_time   = time.time()

        # Metrics chính
        self.total_reward  = 0.0
        self.coverage_rate = 0.0
        self.victims_found = 0
        self.total_victims = 0
        self.episode_length = 0

        self.collision_events: List[Dict] = []
        self.events: Dict[str, int] = {}

        # Safety metrics
        self.collision_obstacle  = 0
        self.collision_uav       = 0
        self.collision_proximity = 0
        self.battery_deaths      = 0
        self.danger_zone_entries = 0
        self.hot_swaps           = 0

        # Landing tracking
        self.landing_events:    List[Dict]     = []
        self.total_landings:    int            = 0
        self.total_charge_time: int            = 0
        self.per_uav_landings:  Dict[int, int] = {}

        # ✅ NEW v2.0: Reward breakdown accumulation
        # Key = component name, Value = sum over episode
        self._reward_breakdown_accum: Dict[str, float] = {}

    def log_step(
        self,
        rewards:   Dict,
        coverage:  float,
        breakdown: Optional[Dict[str, float]] = None,  # ✅ NEW param
    ) -> None:
        """
        Log một step.

        Args:
            rewards:   Dict {agent_id: reward} — per-agent rewards
            coverage:  Coverage rate [0, 1]
            breakdown: Global reward breakdown dict từ BaselineReward.compute()
                       Keys: coverage_delta, victim_found, landing_reward, ...
                       Nếu None → không accumulate breakdown
        """
        self.total_reward  += sum(rewards.values())
        self.coverage_rate  = max(self.coverage_rate, coverage)
        self.episode_length += 1

        # ✅ Accumulate breakdown
        if breakdown:
            skip_keys = {"raw_total", "total", "fleet_incentive"}
            for key, val in breakdown.items():
                if key in skip_keys:
                    continue
                if not isinstance(val, (int, float)):
                    continue
                self._reward_breakdown_accum[key] = (
                    self._reward_breakdown_accum.get(key, 0.0) + float(val)
                )

    def log_event(self, event_type: str, **kwargs):
        if event_type == 'collision_obstacle':
            self.collision_obstacle += 1
        elif event_type == 'collision_uav':
            self.collision_uav += 1
        elif event_type == 'collision_proximity':
            self.collision_proximity += 1
        elif event_type == 'victim_found':
            self.victims_found += 1
        elif event_type == 'battery_death':
            self.battery_deaths += 1
        elif event_type == 'danger_zone':
            self.danger_zone_entries += 1
        elif event_type == 'hot_swap':
            self.hot_swaps += 1

    def set_total_victims(self, n: int):
        self.total_victims = n

    def log_landing(
        self,
        uav_id:         int,
        step:           int,
        battery_before: float,
        battery_after:  float,
    ):
        self.landing_events.append({
            "uav_id":        uav_id,
            "step":          step,
            "battery_before": battery_before,
            "battery_after":  battery_after,
            "charge_amount":  battery_after - battery_before,
        })
        self.total_landings += 1
        self.per_uav_landings[uav_id] = (
            self.per_uav_landings.get(uav_id, 0) + 1
        )

    def log_charging_step(self, uav_id: int):
        self.total_charge_time += 1

    def log_collision(self, uav_id: int, step: int, obstacle_info: dict):
        self.collision_events.append({
            "step":          step,
            "uav_id":        uav_id,
            "obstacle_id":   obstacle_info.get("id"),
            "obstacle_type": obstacle_info.get("type"),
            "position":      obstacle_info.get("pos"),
            "height":        obstacle_info.get("height"),
        })

    def finalize(self) -> Dict[str, Any]:
        """
        Hoàn tất episode → trả về metrics đầy đủ.

        v2.0: Thêm rewards_breakdown (accumulated) vào output.
        """
        duration = time.time() - self.start_time

        coverage_ratio  = float(self.coverage_rate)
        coverage_percent = coverage_ratio * 100.0

        victim_found_rate = (
            self.victims_found / max(1, self.total_victims) * 100
        )
        total_collisions = (
            self.collision_obstacle
            + self.collision_uav
            + self.collision_proximity
        )

        # ✅ Tính % của từng breakdown component
        # Dùng abs(total_reward) để tránh sign issue
        total_abs = abs(self.total_reward) if abs(self.total_reward) > 1.0 else 1.0
        breakdown_pct: Dict[str, float] = {}
        for key, val in self._reward_breakdown_accum.items():
            breakdown_pct[f"{key}_pct"] = val / total_abs * 100.0

        metrics = {
            # Episode info
            "episode_id":     int(self.episode_id),
            "seed":           int(self.seed) if self.seed is not None else None,
            "duration":       float(duration),
            "episode_length": int(self.episode_length),

            # Performance
            "total_reward":        float(self.total_reward),
            "avg_reward_per_step": float(
                self.total_reward / max(1, self.episode_length)
            ),
            "coverage_rate":       float(coverage_percent),
            "victims_found":       int(self.victims_found),
            "total_victims":       int(self.total_victims),
            "victims_found_rate":  float(victim_found_rate),

            # Safety
            "collision_obstacle":  int(self.collision_obstacle),
            "collision_uav":       int(self.collision_uav),
            "collision_proximity": int(self.collision_proximity),
            "total_collisions":    int(total_collisions),
            "battery_deaths":      int(self.battery_deaths),
            "danger_zone_entries": int(self.danger_zone_entries),
            "hot_swaps":           int(self.hot_swaps),

            # Landing
            "total_landings":        int(self.total_landings),
            "total_charge_time":     int(self.total_charge_time),
            "avg_charge_per_landing": float(
                self.total_charge_time / max(self.total_landings, 1)
            ),
            "landings_per_uav": dict(self.per_uav_landings),

            # Success
            "success": bool(coverage_ratio >= 0.9),

            # ✅ NEW v2.0: Accumulated reward breakdown
            # Absolute values (sum over episode)
            "rewards_breakdown": dict(self._reward_breakdown_accum),
            # Percentage relative to |total_reward|
            "rewards_breakdown_pct": breakdown_pct,
        }

        return metrics


class TrainingLogger:
    """Logger CHÍNH cho training — Research Grade"""

    def __init__(self, verbose: int = 1, window_size: int = 100):
        self.verbose     = verbose
        self.window_size = window_size

        self.all_metrics: List[Dict] = []

        self.recent_rewards         = deque(maxlen=window_size)
        self.recent_coverage        = deque(maxlen=window_size)
        self.recent_success         = deque(maxlen=window_size)
        self.recent_episode_lengths = deque(maxlen=window_size)

        self.converged              = False
        self.convergence_episode    = None
        self.convergence_std_threshold = 0.05

    def log_episode(self, metrics: Dict[str, Any]):
        self.all_metrics.append(metrics)

        self.recent_rewards.append(metrics["total_reward"])
        self.recent_coverage.append(metrics["coverage_rate"])
        self.recent_success.append(1 if metrics["success"] else 0)
        self.recent_episode_lengths.append(metrics["episode_length"])

        ep_id = metrics["episode_id"]

        if not self.converged and len(self.recent_rewards) == self.window_size:
            self._check_convergence(ep_id)

        if self.verbose >= 1:
            self._print_episode_line(metrics)

        if self.verbose >= 2 and (ep_id + 1) % 100 == 0:
            self._print_summary(last_n=100)

    def _print_episode_line(self, metrics: Dict[str, Any]):
        success_icon = "✅" if metrics["success"] else "❌"
        conv_icon    = "🎯" if self.converged else ""

        print(
            f"Ep {metrics['episode_id']:4d} | "
            f"R: {metrics['total_reward']:6.1f} | "
            f"Cov: {metrics['coverage_rate']:5.1f}% | "
            f"Vic: {metrics['victims_found']:2d}/{metrics['total_victims']} | "
            f"Len: {metrics['episode_length']:3d} | "
            f"{success_icon}{conv_icon}"
        )

    def _check_convergence(self, episode: int):
        if len(self.recent_rewards) < self.window_size:
            return

        mean_reward  = np.mean(self.recent_rewards)
        std_reward   = np.std(self.recent_rewards)
        success_rate = np.mean(self.recent_success)

        if abs(mean_reward) > 10.0:
            relative_std = std_reward / abs(mean_reward)
            threshold    = self.convergence_std_threshold
        else:
            relative_std = std_reward / 10.0
            threshold    = 0.5

        if relative_std < threshold and success_rate > 0.5:
            self.converged            = True
            self.convergence_episode  = episode

            if self.verbose >= 1:
                print(f"\n🎯 CONVERGENCE DETECTED at episode {episode}")
                print(f"   Mean reward: {mean_reward:.2f}")
                print(
                    f"   Std: {std_reward:.2f} "
                    f"(relative: {relative_std*100:.1f}%)"
                )
                print(f"   Success rate: {success_rate*100:.1f}%\n")

    def _print_summary(self, last_n: int = 100):
        if not self.all_metrics:
            return

        recent     = self.all_metrics[-last_n:]
        rewards    = [e["total_reward"]    for e in recent]
        coverage   = [e["coverage_rate"]   for e in recent]
        success    = [1 if e["success"] else 0 for e in recent]
        lengths    = [e["episode_length"]  for e in recent]
        collisions = [e["total_collisions"] for e in recent]

        print(f"\n{'='*70}")
        print(f"SUMMARY - LAST {last_n} EPISODES:")
        print(f"{'='*70}")
        print(f"Reward      : {np.mean(rewards):6.1f} ± {np.std(rewards):5.1f}")
        print(f"Coverage    : {np.mean(coverage):5.1f}% ± {np.std(coverage):4.1f}%")
        print(f"Success Rate: {np.mean(success)*100:5.1f}%")
        print(f"Avg Length  : {np.mean(lengths):5.1f} steps")
        print(f"Collisions  : {np.mean(collisions):5.2f} ± {np.std(collisions):4.2f}")
        print(f"{'='*70}\n")

    def get_stats(self, last_n: Optional[int] = None) -> Dict[str, float]:
        if not self.all_metrics:
            return {}

        episodes = self.all_metrics if last_n is None else self.all_metrics[-last_n:]

        rewards  = [e["total_reward"]   for e in episodes]
        coverage = [e["coverage_rate"]  for e in episodes]
        success  = [1 if e["success"] else 0 for e in episodes]
        lengths  = [e["episode_length"] for e in episodes]

        return {
            "n_episodes":           len(episodes),
            "reward_mean":          float(np.mean(rewards)),
            "reward_std":           float(np.std(rewards)),
            "coverage_mean":        float(np.mean(coverage)),
            "coverage_std":         float(np.std(coverage)),
            "success_rate":         float(np.mean(success) * 100),
            "avg_episode_length":   float(np.mean(lengths)),
            "converged":            bool(self.converged),
            "convergence_episode":  (
                int(self.convergence_episode)
                if self.convergence_episode is not None else None
            ),
        }

    def get_overall_stats(self) -> Dict[str, float]:
        return self.get_stats(last_n=None)

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.all_metrics, f, indent=2)
        if self.verbose >= 1:
            print(f"✅ Saved {len(self.all_metrics)} episodes to {filepath}")

    def load(self, filepath: str):
        with open(filepath, "r") as f:
            self.all_metrics = json.load(f)

        recent = self.all_metrics[-self.window_size:]
        for ep in recent:
            self.recent_rewards.append(ep["total_reward"])
            self.recent_coverage.append(ep["coverage_rate"])
            self.recent_success.append(1 if ep["success"] else 0)
            self.recent_episode_lengths.append(ep["episode_length"])

        if self.verbose >= 1:
            print(f"✅ Loaded {len(self.all_metrics)} episodes from {filepath}")


def compare_training_runs(
    runs:   List[TrainingLogger],
    labels: List[str],
):
    print(f"\n{'='*80}")
    print("TRAINING COMPARISON - FINAL RESULTS")
    print(f"{'='*80}")
    print(
        f"{'Algorithm':<15} | {'Reward':<15} | "
        f"{'Coverage':<15} | {'Success':<10} | Converged"
    )
    print(f"{'-'*80}")

    for run, label in zip(runs, labels):
        stats     = run.get_overall_stats()
        conv_text = (
            f"Ep {stats['convergence_episode']}"
            if stats["convergence_episode"] is not None else "No"
        )
        print(
            f"{label:<15} | "
            f"{stats['reward_mean']:6.1f} ± {stats['reward_std']:5.1f} | "
            f"{stats['coverage_mean']:5.1f}% ± {stats['coverage_std']:4.1f}% | "
            f"{stats['success_rate']:5.1f}% | "
            f"{conv_text}"
        )

    print(f"{'='*80}\n")