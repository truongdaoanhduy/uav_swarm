from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

"""
training/curriculum.py
Curriculum Manager - Quản lý việc chuyển stage trong training.

FIXED:
    - apply_to_config(): Gán victim fields vào cfg.victim (không phải cfg.env)
    - apply_to_config(): Sync grid_size = map_size
    - apply_to_config(): Xóa reward_scale (không còn trong StageConfig)
"""


@dataclass
class StageStats:
    """Thống kê cho 1 stage."""
    stage_name:    str
    episodes_done: int  = 0
    coverage_list: list = field(default_factory=list)
    victims_list:  list = field(default_factory=list)
    reward_list:   list = field(default_factory=list)

    @property
    def avg_coverage(self) -> float:
        if not self.coverage_list:
            return 0.0
        recent = self.coverage_list[-50:]
        return float(np.mean(recent))

    @property
    def avg_victims(self) -> float:
        if not self.victims_list:
            return 0.0
        recent = self.victims_list[-50:]
        return float(np.mean(recent))

    @property
    def avg_reward(self) -> float:
        if not self.reward_list:
            return 0.0
        recent = self.reward_list[-50:]
        return float(np.mean(recent))


class CurriculumManager:
    """
    Quản lý curriculum learning.

    Args:
        stages: List[StageConfig] theo thứ tự dễ → khó
    """

    def __init__(self, stages) -> None:
        self.stages = stages
        self._stage_idx = 0
        self._stats: List[StageStats] = [
            StageStats(stage_name=s.name) for s in stages
        ]

    # ─── Properties ──────────────────────────────────────────────────────────

    @property
    def current_stage(self):
        return self.stages[self._stage_idx]

    @property
    def current_stats(self) -> StageStats:
        return self._stats[self._stage_idx]

    @property
    def stage_idx(self) -> int:
        return self._stage_idx

    @property
    def is_final_stage(self) -> bool:
        return self._stage_idx >= len(self.stages) - 1

    @property
    def total_episodes(self) -> int:
        return sum(s.episodes_done for s in self._stats)

    # ─── Update ──────────────────────────────────────────────────────────────

    def update(self, coverage: float, victims_rate: float, reward: float = 0.0) -> None:
        """Cập nhật metrics sau mỗi episode."""
        stats = self.current_stats
        stats.episodes_done += 1
        stats.coverage_list.append(float(coverage))
        stats.victims_list.append(float(victims_rate))
        stats.reward_list.append(float(reward))

    # ─── Advance Check ───────────────────────────────────────────────────────

    def should_advance(self) -> bool:
        """
        Kiểm tra có nên advance stage không.

        Điều kiện (TẤT CẢ phải đúng):
            1. Chưa phải final stage
            2. Đã đủ min_episodes
            3. avg_coverage >= advance_coverage
            4. avg_victims >= advance_victims
        """
        if self.is_final_stage:
            return False

        stage = self.current_stage
        stats = self.current_stats

        if stats.episodes_done < stage.min_episodes:
            return False

        if stats.avg_coverage < stage.advance_coverage:
            return False

        if stats.avg_victims < stage.advance_victims:
            return False

        return True

    def advance(self) -> None:
        """Chuyển sang stage tiếp theo."""
        if self.is_final_stage:
            logger.warning("Already at final stage, cannot advance")
            return

        old_stage = self.current_stage
        old_stats = self.current_stats

        self._stage_idx += 1
        new_stage = self.current_stage

        logger.info(
            "CURRICULUM ADVANCE: %s -> %s",
            old_stage.name.upper(),
            new_stage.name.upper(),
        )
        logger.info(
            "Stage '%s' done: %d eps | cov=%.1f%% | vic=%.1f%% | rew=%.2f",
            old_stage.name,
            old_stats.episodes_done,
            old_stats.avg_coverage * 100,
            old_stats.avg_victims * 100,
            old_stats.avg_reward,
        )
        logger.info("New stage: %s", new_stage.describe())

    # ─── Apply Config ─────────────────────────────────────────────────────────

    def apply_to_config(self, cfg) -> None:
        """
        Apply stage config vào AppConfig (in-place).

        UPDATED: Dùng cfg.apply_stage() thay vì gán thủ công.
        Single point of truth cho stage → config mapping.

        Args:
            cfg: AppConfig object
        """
        # ── Delegate to AppConfig.apply_stage() ──────────────────────────────
        cfg.apply_stage(self.current_stage)

        logger.info("Applied stage config: %s", self.current_stage.describe())
        print(f"✅ Applied stage config: {self.current_stage.describe()}")

    # ─── Status ──────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Trả về status dict."""
        stage = self.current_stage
        stats = self.current_stats

        return {
            "stage":             stage.name,
            "stage_idx":         self._stage_idx,
            "total_stages":      len(self.stages),
            "episodes_in_stage": stats.episodes_done,
            "total_episodes":    self.total_episodes,
            "avg_coverage":      stats.avg_coverage,
            "avg_victims":       stats.avg_victims,
            "avg_reward":        stats.avg_reward,
            "advance_coverage":  stage.advance_coverage,
            "advance_victims":   stage.advance_victims,
            "min_episodes":      stage.min_episodes,
            "ready_to_advance":  self.should_advance(),
            "is_final":          self.is_final_stage,
            # Thêm để dễ debug
            "map_size":          stage.map_size,
            "coverage_pressure": stage.coverage_pressure_m2_per_uav,
        }

    def print_status(self) -> None:
        """In status ra console."""
        status = self.get_status()
        stage  = self.current_stage

        print("\n" + "=" * 60)
        print(f"  CURRICULUM STATUS - {stage.name.upper()}")
        print("=" * 60)
        print(f"  Stage:    {status['stage_idx']+1}/{status['total_stages']}")
        print(f"  Map:      {status['map_size']}×{status['map_size']}m "
              f"(pressure={status['coverage_pressure']:,.0f}m²/UAV)")
        print(f"  Episodes: {status['episodes_in_stage']:,} "
              f"/ {status['min_episodes']:,} (min)")
        print(f"  Total:    {status['total_episodes']:,}")
        print("-" * 60)
        print(f"  Coverage: {status['avg_coverage']*100:5.1f}% "
              f"/ {status['advance_coverage']*100:.0f}% target")
        print(f"  Victims:  {status['avg_victims']*100:5.1f}% "
              f"/ {status['advance_victims']*100:.0f}% target")
        print(f"  Reward:   {status['avg_reward']:8.2f}")
        print("-" * 60)

        if status["ready_to_advance"]:
            print("  >> READY TO ADVANCE -> next stage!")
        elif status["is_final"]:
            print("  >> FINAL STAGE - Training complete!")
        else:
            if status['avg_coverage'] < status['advance_coverage']:
                delta = (status['advance_coverage'] - status['avg_coverage']) * 100
                print(f"  >> Coverage need: +{delta:.1f}%")
            if status['avg_victims'] < status['advance_victims']:
                delta = (status['advance_victims'] - status['avg_victims']) * 100
                print(f"  >> Victims need:  +{delta:.1f}%")
            if status['episodes_in_stage'] < status['min_episodes']:
                remaining = status['min_episodes'] - status['episodes_in_stage']
                print(f"  >> Episodes need: +{remaining:,}")

        print("=" * 60 + "\n")