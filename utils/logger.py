import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import time
import json

"""
Hệ thống logging cho SAR UAV Swarm - Research Grade
Đã fix: coverage units, total_victims missing, convergence logic
"""


class EpisodeLogger:
    """
    Logger cho một episode - CHỈ LƯU DATA, KHÔNG IN
    Research-ready với đầy đủ metrics cần thiết
    """
    
    def __init__(self, episode_id: int, seed: Optional[int] = None):
        """
        Tham số:
            episode_id: ID của episode
            seed: Random seed (để track khi chạy nhiều seeds)
        """
        self.episode_id = episode_id
        self.seed = seed
        self.start_time = time.time()
        
        # Metrics chính (4 metrics quan trọng nhất)
        self.total_reward = 0.0
        self.coverage_rate = 0.0  # Lưu dưới dạng [0, 1], sẽ convert sang % khi finalize
        self.victims_found = 0
        self.total_victims = 0
        
        # Episode length (QUAN TRỌNG cho research)
        self.episode_length = 0
        
        # Safety metrics (phân loại chi tiết)
        self.collision_obstacle = 0  # Va chạm debris
        self.collision_uav = 0        # Va chạm UAV khác
        self.collision_proximity = 0  # Proximity warning
        self.battery_deaths = 0
        self.danger_zone_entries = 0
        
        # Fleet metrics
        self.hot_swaps = 0
        
    def log_step(self, rewards: Dict[str, float], coverage: float):
        """
        Log step - CHỈ CẬP NHẬT DATA
        
        Tham số:
            rewards: Dict {agent_id: reward}
            coverage: Coverage rate hiện tại [0, 1]
        """
        self.total_reward += sum(rewards.values())
        
        # Coverage: lưu giá trị MAX (tránh noise), giữ [0,1]
        self.coverage_rate = max(self.coverage_rate, coverage)
        
        # Tăng episode length
        self.episode_length += 1
    
    def log_event(self, event_type: str, **kwargs):
        """
        Log event với phân loại chi tiết
        
        Tham số:
            event_type: Loại event
            **kwargs: Thông tin bổ sung
        """
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
        """Đặt tổng số victims trong episode"""
        self.total_victims = n
    
    def finalize(self) -> Dict[str, Any]:
        """
        Hoàn tất episode, trả về metrics
        ĐẢM BẢO tất cả values là Python native types (không phải numpy)
        
        FIX:
        - Tách rõ coverage_ratio [0,1] và coverage_percent [0,100]
        - Success dùng đúng ngưỡng 0.9 với coverage_ratio
        - Thêm total_victims vào metrics
        
        Trả về:
            Dict chứa tất cả metrics (JSON-safe)
        """
        duration = time.time() - self.start_time
        
        # Tách rõ coverage units (FIX LỖI 1)
        coverage_ratio = float(self.coverage_rate)  # [0, 1]
        coverage_percent = coverage_ratio * 100.0   # [0, 100]
        
        # Tính toán metrics
        victim_found_rate = (self.victims_found / max(1, self.total_victims)) * 100
        total_collisions = (self.collision_obstacle + 
                           self.collision_uav + 
                           self.collision_proximity)
        
        # ĐẢM BẢO tất cả là float/int, KHÔNG phải numpy types
        metrics = {
            # Episode info
            'episode_id': int(self.episode_id),
            'seed': int(self.seed) if self.seed is not None else None,
            'duration': float(duration),
            'episode_length': int(self.episode_length),
            
            # Performance metrics (CAST FLOAT)
            'total_reward': float(self.total_reward),
            'avg_reward_per_step': float(self.total_reward / max(1, self.episode_length)),
            'coverage_rate': float(coverage_percent),  # Store as percent [0, 100]
            'victims_found': int(self.victims_found),
            'total_victims': int(self.total_victims),  # FIX LỖI 2: Thêm field này
            'victims_found_rate': float(victim_found_rate),
            
            # Safety metrics (phân loại chi tiết)
            'collision_obstacle': int(self.collision_obstacle),
            'collision_uav': int(self.collision_uav),
            'collision_proximity': int(self.collision_proximity),
            'total_collisions': int(total_collisions),
            'battery_deaths': int(self.battery_deaths),
            'danger_zone_entries': int(self.danger_zone_entries),
            
            # Fleet metrics
            'hot_swaps': int(self.hot_swaps),
            
            # Success criteria (FIX LỖI 1: Dùng đúng coverage_ratio >= 0.9)
            'success': bool(coverage_ratio >= 0.9),
        }
        
        return metrics


class TrainingLogger:
    """
    Logger CHÍNH cho training - Research Grade
    Hỗ trợ multi-seed, convergence tracking, và phân tích chi tiết
    """
    
    def __init__(self, verbose: int = 1, window_size: int = 100):
        """
        Tham số:
            verbose: 
                0 = Im lặng (chỉ lưu file)
                1 = Cơ bản (mỗi episode in 1 dòng)
                2 = Chi tiết (mỗi 100 episodes in summary)
            window_size: Kích thước window cho moving average
        """
        self.verbose = verbose
        self.window_size = window_size
        
        # Lưu trữ TẤT CẢ episodes
        self.all_metrics = []
        
        # Moving windows cho monitoring
        self.recent_rewards = deque(maxlen=window_size)
        self.recent_coverage = deque(maxlen=window_size)
        self.recent_success = deque(maxlen=window_size)
        self.recent_episode_lengths = deque(maxlen=window_size)
        
        # Convergence tracking (FIX LỖI 3: Dùng relative threshold)
        self.converged = False
        self.convergence_episode = None
        self.convergence_std_threshold = 0.05  # 5% của mean reward
        
    def log_episode(self, metrics: Dict[str, Any]):
        """
        Log episode - TỰ ĐỘNG QUYẾT ĐỊNH IN GÌ
        
        Tham số:
            metrics: Dict từ EpisodeLogger.finalize()
        """
        self.all_metrics.append(metrics)
        
        # Update moving windows
        self.recent_rewards.append(metrics['total_reward'])
        self.recent_coverage.append(metrics['coverage_rate'])
        self.recent_success.append(1 if metrics['success'] else 0)
        self.recent_episode_lengths.append(metrics['episode_length'])
        
        ep_id = metrics['episode_id']
        
        # Check convergence (chỉ sau 100 episodes đầu)
        if not self.converged and len(self.recent_rewards) == self.window_size:
            self._check_convergence(ep_id)
        
        # LEVEL 1: Mỗi episode - 1 DÒNG
        if self.verbose >= 1:
            self._print_episode_line(metrics)
        
        # LEVEL 2: Mỗi 100 episodes - SUMMARY
        if self.verbose >= 2 and (ep_id + 1) % 100 == 0:
            self._print_summary(last_n=100)
    
    def _print_episode_line(self, metrics: Dict[str, Any]):
        """In một dòng cho episode (compact)"""
        success_icon = "✅" if metrics['success'] else "❌"
        
        # Thêm icon convergence nếu đã converge
        conv_icon = "🎯" if self.converged else ""
        
        print(f"Ep {metrics['episode_id']:4d} | "
              f"R: {metrics['total_reward']:6.1f} | "
              f"Cov: {metrics['coverage_rate']:5.1f}% | "
              f"Vic: {metrics['victims_found']:2d}/{metrics['total_victims']} | "  # FIX: Không cần .get()
              f"Len: {metrics['episode_length']:3d} | "
              f"{success_icon}{conv_icon}")
    
    def _check_convergence(self, episode: int):
        """Check convergence với threshold phù hợp cho reward âm."""
        if len(self.recent_rewards) < self.window_size:
            return
        
        mean_reward = np.mean(self.recent_rewards)
        std_reward = np.std(self.recent_rewards)
        success_rate = np.mean(self.recent_success)
        
        # FIX: Dùng absolute std nếu mean gần 0 hoặc âm
        if abs(mean_reward) > 10.0:
            relative_std = std_reward / abs(mean_reward)
            threshold = self.convergence_std_threshold  # 5%
        else:
            # Mean reward gần 0 → dùng absolute threshold
            relative_std = std_reward / 10.0  # normalize by reasonable scale
            threshold = 0.5  # absolute threshold
        
        if relative_std < threshold and success_rate > 0.5:
            self.converged = True
            self.convergence_episode = episode
            
            if self.verbose >= 1:
                print(f"\n🎯 CONVERGENCE DETECTED at episode {episode}")
                print(f"   Mean reward: {mean_reward:.2f}")
                print(f"   Std: {std_reward:.2f} (relative: {relative_std*100:.1f}%)")
                print(f"   Success rate: {success_rate*100:.1f}%\n")
    
    def _print_summary(self, last_n: int = 100):
        """In summary ngắn gọn"""
        if not self.all_metrics:
            return
        
        recent = self.all_metrics[-last_n:]
        
        # Tính stats
        rewards = [e['total_reward'] for e in recent]
        coverage = [e['coverage_rate'] for e in recent]
        success = [1 if e['success'] else 0 for e in recent]
        lengths = [e['episode_length'] for e in recent]
        collisions = [e['total_collisions'] for e in recent]
        
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
        """
        Lấy stats - KHÔNG IN
        
        Tham số:
            last_n: Số episodes gần nhất (None = tất cả)
        
        Trả về:
            Dict chứa statistics
        """
        if not self.all_metrics:
            return {}
        
        if last_n is None:
            episodes = self.all_metrics
        else:
            episodes = self.all_metrics[-last_n:]
        
        rewards = [e['total_reward'] for e in episodes]
        coverage = [e['coverage_rate'] for e in episodes]
        success = [1 if e['success'] else 0 for e in episodes]
        lengths = [e['episode_length'] for e in episodes]
        
        return {
            'n_episodes': len(episodes),
            'reward_mean': float(np.mean(rewards)),
            'reward_std': float(np.std(rewards)),
            'coverage_mean': float(np.mean(coverage)),
            'coverage_std': float(np.std(coverage)),
            'success_rate': float(np.mean(success) * 100),
            'avg_episode_length': float(np.mean(lengths)),
            'converged': bool(self.converged),
            # FIX LỖI 4: Dùng is not None thay vì truthy check
            'convergence_episode': int(self.convergence_episode) if self.convergence_episode is not None else None,
        }
    
    def get_overall_stats(self) -> Dict[str, float]:
        """
        Lấy overall stats cho TOÀN BỘ training
        (Dùng cho final report)
        """
        return self.get_stats(last_n=None)
    
    def save(self, filepath: str):
        """
        Lưu ra file JSON (JSON-safe)
        
        Tham số:
            filepath: Đường dẫn file
        """
        with open(filepath, 'w') as f:
            json.dump(self.all_metrics, f, indent=2)
        
        if self.verbose >= 1:
            print(f"✅ Saved {len(self.all_metrics)} episodes to {filepath}")
    
    def load(self, filepath: str):
        """
        Load từ file JSON
        
        Tham số:
            filepath: Đường dẫn file
        """
        with open(filepath, 'r') as f:
            self.all_metrics = json.load(f)
        
        # Rebuild moving windows từ episodes gần nhất
        recent = self.all_metrics[-self.window_size:]
        for ep in recent:
            self.recent_rewards.append(ep['total_reward'])
            self.recent_coverage.append(ep['coverage_rate'])
            self.recent_success.append(1 if ep['success'] else 0)
            self.recent_episode_lengths.append(ep['episode_length'])
        
        if self.verbose >= 1:
            print(f"✅ Loaded {len(self.all_metrics)} episodes from {filepath}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compare_training_runs(runs: List[TrainingLogger], labels: List[str]):
    """
    So sánh nhiều training runs (cho Phase 2 & 3)
    
    Tham số:
        runs: Danh sách TrainingLogger objects
        labels: Tên của từng run (vd: ["MAPPO", "MASAC", "MATD3"])
    """
    print(f"\n{'='*80}")
    print(f"TRAINING COMPARISON - FINAL RESULTS")
    print(f"{'='*80}")
    print(f"{'Algorithm':<15} | {'Reward':<15} | {'Coverage':<15} | {'Success':<10} | {'Converged'}")
    print(f"{'-'*80}")
    
    for run, label in zip(runs, labels):
        stats = run.get_overall_stats()
        
        # FIX: Dùng is not None
        conv_text = f"Ep {stats['convergence_episode']}" if stats['convergence_episode'] is not None else "No"
        
        print(f"{label:<15} | "
              f"{stats['reward_mean']:6.1f} ± {stats['reward_std']:5.1f} | "
              f"{stats['coverage_mean']:5.1f}% ± {stats['coverage_std']:4.1f}% | "
              f"{stats['success_rate']:5.1f}% | "
              f"{conv_text}")
    
    print(f"{'='*80}\n")
