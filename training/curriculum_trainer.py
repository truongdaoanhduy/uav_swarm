from __future__ import annotations

import logging
import inspect
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class CurriculumTrainer:
    """Training loop với Curriculum Learning + Visualization."""

    def __init__(self, cfg, render_every: int = 100,save_gif: bool = False) -> None:
        from config.curriculum_config import CURRICULUM_STAGES
        from training.curriculum import CurriculumManager

        self.cfg          = cfg
        self.curriculum   = CurriculumManager(stages=CURRICULUM_STAGES)
        self.env          = None
        self._env_type    = None
        self.save_gif     = save_gif
        self.render_every = render_every  # ← Render mỗi N episodes

        # Tạo thư mục results
        os.makedirs("results/frames",   exist_ok=True)
        os.makedirs("results/episodes", exist_ok=True)
        os.makedirs("results/curves",   exist_ok=True)

        # Apply easy stage
        self.curriculum.apply_to_config(cfg)

    # ─── Env Management ──────────────────────────────────────────────────────

    def _build_env(self):
        from env_setup import SARBaseEnv
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.env = SARBaseEnv(self.cfg)
        self._env_type = None
        return self.env

    # ─── Env Type + Action Sampling (giữ nguyên) ─────────────────────────────

    def _detect_env_type(self) -> str:
        if self._env_type is not None:
            return self._env_type
        action_space = self.env.action_space
        if callable(action_space) and (
            inspect.ismethod(action_space) or
            inspect.isfunction(action_space)
        ):
            self._env_type = "pettingzoo"
        elif hasattr(action_space, 'sample'):
            self._env_type = "gymnasium"
        else:
            self._env_type = "gymnasium"
        return self._env_type

    def _sample_actions(self, n_uav: int) -> dict:
        env_type = self._detect_env_type()
        if env_type == "pettingzoo":
            try:
                return {i: self.env.action_space(f"uav_{i}").sample()
                        for i in range(n_uav)}
            except Exception:
                pass
        else:
            try:
                return {i: self.env.action_space.sample()
                        for i in range(n_uav)}
            except Exception:
                pass
        return {i: np.random.uniform(-1, 1, 3).astype(np.float32)
                for i in range(n_uav)}

    # ─── Main Training Loop ──────────────────────────────────────────────────

    def train(self, total_episodes: int = 1000) -> Dict[str, Any]:
        """Main training loop với visualization."""
        history = {
            "episodes": [], "coverages": [], "victims": [],
            "rewards":  [], "stages":    [], "steps":   [],
        }

        self._build_env()

        print(f"\n{'='*60}")
        print(f"  Curriculum Training | {total_episodes} episodes")
        if self.render_every > 0:
            print(f"  Render every {self.render_every} episodes")
        else:
            print(f"  Render disabled (render_every=0)")
        print(f"  Stage 1: {self.curriculum.current_stage.describe()}")
        print(f"{'='*60}\n")

        for episode in range(total_episodes):

            # ═══ FIX: Check render_every > 0 trước khi modulo ═══
            if self.render_every > 0:
                should_render = (episode % self.render_every == 0)
            else:
                should_render = False  # ← Tắt render hoàn toàn

            # ── Run 1 episode ──
            ep_metrics = self._run_episode(
                episode_num   = episode,
                render_frames = should_render,
            )

            # ── Update curriculum ──
            self.curriculum.update(
                coverage           = ep_metrics["coverage"],
                victims_found_rate = ep_metrics["victims_rate"],
                episode_reward     = ep_metrics["reward"],
            )

            # ── Ghi history ──
            history["episodes"].append(episode)
            history["coverages"].append(ep_metrics["coverage"])
            history["victims"].append(ep_metrics["victims_rate"])
            history["rewards"].append(ep_metrics["reward"])
            history["stages"].append(self.curriculum.stage_idx)
            history["steps"].append(ep_metrics["steps"])

            # ── Log mỗi 50 episodes ──
            if (episode + 1) % 50 == 0:
                self.curriculum.print_status()

            # ── Plot training curves mỗi 100 episodes ──
            if (episode + 1) % 100 == 0:
                self._plot_training_curves(history, episode)

            # ── Check Advance ──
            if self.curriculum.should_advance():
                self.curriculum.advance()
                self.curriculum.apply_to_config(self.cfg)
                self._build_env()
                print(f"\n{'*'*60}")
                print(f"  ADVANCED → {self.curriculum.current_stage.name.upper()}")
                print(f"  {self.curriculum.current_stage.describe()}")
                print(f"{'*'*60}\n")

        # ── Final plots ──
        self._plot_training_curves(history, total_episodes - 1, final=True)
        self.curriculum.print_status()
        self._print_summary(history)

        if self.env is not None:
            self.env.close()

        return history

    # ─── Episode Runner ──────────────────────────────────────────────────────

    def _run_episode(
        self,
        episode_num:   int,
        render_frames: bool = False,
    ) -> Dict[str, Any]:
        """Chạy 1 episode."""
        obs, info = self.env.reset(seed=episode_num)
        total_reward = 0.0
        done         = False
        step         = 0
        n_uav        = self.cfg.env.n_uav
        frames       = []

        # Init visualizer nếu cần render
        if render_frames:
            from visualization.visualizer2d import Visualizer2D
            viz = Visualizer2D(self.cfg, render_mode="rgb_array")

        while not done and step < self.cfg.env.max_steps:
            actions = self._sample_actions(n_uav)
            obs, rewards, dones, truncs, infos = self.env.step(actions)

            # ... reward tracking ...

            step += 1

            # ═══ OPTIMIZE: Giảm frequency ═══
            # Chỉ capture 10 frames/episode thay vì 30
            capture_interval = self.cfg.env.max_steps // 10  # 300 // 10 = 30
            
            if render_frames and step % capture_interval == 0:
                state = self.env.unwrapped.backend.get_state()
                frame = viz.render(
                    uavs      = state["uavs"],
                    victims   = state["victims"],
                    obstacles = state["obstacles"],
                    stations  = state["stations"],
                    cov_map   = state["coverage_map"],
                    step      = step,
                )
                frames.append(frame)

        # Lưu frames
        if render_frames and frames:
            self._save_episode_visualization(
                frames     = frames,
                episode    = episode_num,
                stage_name = self.curriculum.current_stage.name,
            )
        
        # ── Metrics từ backend ──
        try:
            state    = self.env.unwrapped.backend.get_state()
            cov_map  = state["coverage_map"]
            victims  = state["victims"]
            coverage = float(cov_map.get_coverage_rate())
            n_found  = sum(1 for v in victims if v.is_found)
            n_total  = len(victims)
            vic_rate = n_found / n_total if n_total > 0 else 0.0
        except Exception as e:
            logger.warning("Backend metrics failed: %s", e)
            coverage = 0.0
            vic_rate = 0.0

        return {
            "coverage":     coverage,
            "victims_rate": vic_rate,
            "reward":       total_reward,
            "steps":        step,
        }

    # ─── Visualization Helpers ────────────────────────────────────────────────

    def _save_episode_visualization(
        self,
        frames:     List[np.ndarray],
        episode:    int,
        stage_name: str,
    ) -> None:
        """
        Lưu visualization của 1 episode.
        
        OPTIMIZED: Chỉ lưu first+last PNG, bỏ GIF (quá chậm).
        """
        stage = stage_name.upper()

        # ── 1. Lưu first + last frame (NHANH) ──
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(
            f"Episode {episode} | Stage: {stage}",
            fontsize=14, fontweight="bold"
        )

        axes[0].imshow(frames[0])
        axes[0].set_title("Start", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(frames[-1])
        axes[1].set_title("End", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        plt.tight_layout()
        save_path = f"results/episodes/ep{episode:04d}_{stage}.png"
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved episode viz: %s", save_path)

        # ── 2. GIF (OPTIONAL - CHẬM) ──
        if self.save_gif and len(frames) >= 5:
            self._save_gif(frames, f"results/episodes/ep{episode:04d}_{stage}.gif")
        elif not self.save_gif:
            logger.debug("GIF disabled (save_gif=False), skipping")
            
    def _save_gif(self, frames: List[np.ndarray], path: str) -> None:
        """Lưu frames thành GIF animation."""
        try:
            from PIL import Image

            pil_frames = [Image.fromarray(f) for f in frames]
            pil_frames[0].save(
                path,
                save_all   = True,
                append_images = pil_frames[1:],
                duration   = 400,   # ms per frame
                loop       = 0,     # 0 = loop forever
            )
            logger.info("Saved GIF: %s (%d frames)", path, len(frames))

        except ImportError:
            logger.warning("PIL not installed, skipping GIF. pip install Pillow")
        except Exception as e:
            logger.warning("GIF save failed: %s", e)

    def _plot_training_curves(
        self,
        history: Dict[str, Any],
        episode: int,
        final:   bool = False,
    ) -> None:
        """
        Plot training curves.

        Tạo 1 figure với 4 subplots:
            [1] Coverage per episode
            [2] Victims found rate
            [3] Episode reward
            [4] Stage progression
        """
        if len(history["episodes"]) < 2:
            return

        episodes = np.array(history["episodes"])
        stages   = np.array(history["stages"])
        covs     = np.array(history["coverages"])
        vics     = np.array(history["victims"])
        rews     = np.array(history["rewards"])

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        title = "FINAL" if final else f"Episode {episode}"
        fig.suptitle(
            f"Curriculum Training Curves — {title}",
            fontsize=13, fontweight="bold"
        )

        STAGE_COLORS = ["#AED6F1", "#A9DFBF", "#F9E79F"]
        STAGE_NAMES  = {0: "EASY", 1: "MEDIUM", 2: "HARD"}

        def add_bg(ax):
            for sid, color in enumerate(STAGE_COLORS):
                mask = np.where(stages == sid)[0]
                if len(mask) == 0:
                    continue
                ax.axvspan(mask[0], mask[-1], alpha=0.15,
                           color=color, label=STAGE_NAMES[sid])
            prev = stages[0]
            for i, s in enumerate(stages):
                if s != prev:
                    ax.axvline(x=i, color="red",
                               linestyle="--", linewidth=1.5, alpha=0.6)
                    prev = s

        def rolling(arr, w=20):
            if len(arr) < w:
                return range(len(arr)), arr
            rm = np.convolve(arr, np.ones(w)/w, mode="valid")
            return range(w-1, len(arr)), rm

        # ── [1] Coverage ──
        ax = axes[0, 0]
        add_bg(ax)
        ax.plot(episodes, covs*100, alpha=0.25, color="#2980B9", linewidth=0.8)
        x_rm, y_rm = rolling(covs*100)
        ax.plot(x_rm, y_rm, color="#1A5276", linewidth=2.5)
        ax.axhline(y=60, color="#E74C3C", linestyle="--",
                   linewidth=1.5, label="EASY threshold (60%)")
        ax.axhline(y=55, color="#E67E22", linestyle=":",
                   linewidth=1.5, label="MED threshold (55%)")
        ax.set_title("Coverage Rate", fontweight="bold")
        ax.set_ylabel("Coverage (%)")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # ── [2] Victims ──
        ax = axes[0, 1]
        add_bg(ax)
        ax.plot(episodes, vics*100, alpha=0.25, color="#27AE60", linewidth=0.8)
        x_rm, y_rm = rolling(vics*100)
        ax.plot(x_rm, y_rm, color="#1E8449", linewidth=2.5)
        ax.axhline(y=75, color="#E74C3C", linestyle="--",
                   linewidth=1.5, label="EASY threshold (75%)")
        ax.axhline(y=70, color="#E67E22", linestyle=":",
                   linewidth=1.5, label="MED threshold (70%)")
        ax.set_title("Victim Found Rate", fontweight="bold")
        ax.set_ylabel("Found Rate (%)")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # ── [3] Reward ──
        ax = axes[1, 0]
        add_bg(ax)
        ax.plot(episodes, rews, alpha=0.25, color="#E67E22", linewidth=0.8)
        x_rm, y_rm = rolling(rews)
        ax.plot(x_rm, y_rm, color="#D35400", linewidth=2.5)
        ax.set_title("Episode Reward", fontweight="bold")
        ax.set_ylabel("Total Reward")
        ax.grid(alpha=0.3)

        # ── [4] Stage bar chart ──
        ax = axes[1, 1]
        counts = [int((stages == i).sum()) for i in range(3)]
        bars = ax.bar(
            [STAGE_NAMES[i] for i in range(3)],
            counts,
            color=STAGE_COLORS,
            edgecolor="gray",
            linewidth=1.5,
            width=0.5,
        )
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f"{count} eps",
                    ha="center", va="bottom",
                    fontweight="bold", fontsize=10,
                )
        ax.set_title("Episodes per Stage", fontweight="bold")
        ax.set_ylabel("Episodes")
        ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()

        suffix = "final" if final else f"ep{episode:04d}"
        save_path = f"results/curves/training_{suffix}.png"
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved training curves: %s", save_path)

    # ─── Summary ─────────────────────────────────────────────────────────────

    def _print_summary(self, history: Dict[str, Any]) -> None:
        print(f"\n{'='*60}")
        print("  CURRICULUM TRAINING COMPLETE")
        print(f"{'='*60}")
        for i, stage in enumerate(self.curriculum.stages):
            stats = self.curriculum._stats[i]
            if stats.episodes_done == 0:
                continue
            print(f"\n  Stage {i+1}: {stage.name.upper()}")
            print(f"    Episodes: {stats.episodes_done:,}")
            print(f"    Coverage: {stats.avg_coverage*100:.1f}%")
            print(f"    Victims:  {stats.avg_victims*100:.1f}%")
            print(f"    Reward:   {stats.avg_reward:.2f}")
        print(f"\n  Total: {self.curriculum.total_episodes:,} episodes")
        print(f"  Results saved to results/")
        print(f"{'='*60}\n")