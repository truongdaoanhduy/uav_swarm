"""
training/algorithms/mappo/trainer.py
MAPPO Trainer với tqdm + episode tracking + viz support

FIXES:
    ✅ FIX-T1: Checkpoint/Viz/Log dùng >= thay vì % (không miss trigger)
    ✅ FIX-T2: Print absolute path khi save checkpoint
    ✅ FIX-T3: _next_trigger tracking cho tất cả intervals
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from collections import deque


# ✅ Auto-detect tqdm
def _get_tqdm():
    try:
        shell = get_ipython().__class__.__name__
        if shell in ('ZMQInteractiveShell', 'Shell'):
            try:
                import ipywidgets
                from tqdm.notebook import tqdm as _tqdm
                return _tqdm
            except ImportError:
                from tqdm import tqdm as _tqdm
                return _tqdm
        else:
            from tqdm import tqdm as _tqdm
            return _tqdm
    except NameError:
        from tqdm import tqdm as _tqdm
        return _tqdm


tqdm = _get_tqdm()

from config import AppConfig
from env_setup.sar_pettingzoo_env import SARPettingZooEnv
from env_setup.vec_env import VectorizedEnv
from training.algorithms.mappo.actor import ActorNetwork
from training.algorithms.mappo.critic import CriticNetwork
from training.algorithms.mappo.buffer import RolloutBuffer


# ══════════════════════════════════════════════════════════════════════════════
# ENV WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class _EnvWrapper:
    """Unified env interface (single hoặc vectorized)."""

    def __init__(self, config: AppConfig, n_envs: int, seed: int):
        self.n_envs   = n_envs
        self.n_agents = config.env.n_uav
        self.obs_dim  = config.obs.actor_dim
        self._config  = config
        self._seed    = seed

        if n_envs == 1:
            self._env    = SARPettingZooEnv(config, render_mode=None)
            self._env.reset(seed=seed)
            self._is_vec = False
        else:
            self._env    = VectorizedEnv(config, n_envs=n_envs, start_seed=seed)
            self._is_vec = True

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns: obs[n_envs, n_agents, obs_dim], global[n_envs, global_obs_dim]"""
        if self._is_vec:
            return self._env.reset()

        obs_d, info = self._env.reset(seed=self._seed)
        obs = np.array(
            [obs_d[f"uav_{i}"] for i in range(self.n_agents)],
            dtype=np.float32,
        )[None]
        g = info["uav_0"]["global_obs"][None]
        return obs, g

    def step(self, actions_batch: np.ndarray):
        """actions_batch: [n_envs, n_agents, 3]"""
        if self._is_vec:
            return self._env.step(actions_batch)

        act_dict = {f"uav_{i}": actions_batch[0][i] for i in range(self.n_agents)}
        obs_d, rew_d, term_d, trunc_d, info = self._env.step(act_dict)

        obs = np.array(
            [obs_d.get(f"uav_{i}", np.zeros(self.obs_dim, np.float32))
             for i in range(self.n_agents)],
            dtype=np.float32,
        )[None]
        g    = info["uav_0"]["global_obs"][None]
        rews = np.array(
            [rew_d.get(f"uav_{i}", 0.0) for i in range(self.n_agents)],
            dtype=np.float32,
        )[None]
        done = any(term_d.values()) or any(trunc_d.values())
        return obs, g, rews, [done], [info]

    def render(self):
        if not self._is_vec and hasattr(self._env, "render"):
            return self._env.render()
        return None

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# MAPPO TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class MAPPOTrainer:
    """
    MAPPO Trainer.

    FIXES:
        ✅ FIX-T1: Dùng >= thay vì % để trigger checkpoint/viz/log
                   → Không bao giờ miss dù episodes_per_update lớn
        ✅ FIX-T2: Print absolute path khi save
        ✅ FIX-T3: _next_*_ep tracking cho mỗi interval
    """

    def __init__(
        self,
        config:   AppConfig,
        device:   str = "auto",
        run_name: str = None,
        n_envs:   int = 1,
    ):
        self.config   = config
        self.n_envs   = n_envs
        self.run_name = run_name or f"mappo_{int(time.time())}"

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.n_agents       = config.env.n_uav
        self.obs_dim        = config.obs.actor_dim
        self.global_obs_dim = config.obs.critic_dim

        tr = config.train
        self.rollout_length = tr.mappo_rollout_length
        self.n_epochs       = tr.mappo_n_epochs
        self.batch_size     = tr.mappo_batch_size
        self.clip_epsilon   = tr.mappo_clip_epsilon
        self.gamma          = tr.mappo_gamma
        self.gae_lambda     = tr.mappo_gae_lambda
        self.max_grad_norm  = tr.mappo_max_grad_norm
        self.entropy_coeff  = tr.mappo_entropy_coeff

        # ── Networks ─────────────────────────────────────────────────────────
        self.actor = ActorNetwork(
            obs_dim       = self.obs_dim,
            action_dim    = 3,
            hidden_dims   = tr.mappo_actor_hidden,
            activation    = tr.mappo_activation,
            use_layer_norm= tr.mappo_use_layer_norm,
        ).to(self.device)

        self.critic = CriticNetwork(
            global_obs_dim = self.global_obs_dim,
            hidden_dims    = tr.mappo_critic_hidden,
            activation     = tr.mappo_activation,
            use_layer_norm = tr.mappo_use_layer_norm,
        ).to(self.device)

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=tr.mappo_lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=tr.mappo_lr_critic)

        # ── Buffer ───────────────────────────────────────────────────────────
        buffer_capacity = self.rollout_length * n_envs
        self.buffer = RolloutBuffer(
            rollout_length = buffer_capacity,
            n_agents       = self.n_agents,
            obs_dim        = self.obs_dim,
            global_obs_dim = self.global_obs_dim,
            action_dim     = 3,
            gamma          = self.gamma,
            gae_lambda     = self.gae_lambda,
        )

        # ── Stats ────────────────────────────────────────────────────────────
        self.ep_rewards  = deque(maxlen=100)
        self.ep_lengths  = deque(maxlen=100)
        self.ep_coverage = deque(maxlen=100)
        self.ep_victims  = deque(maxlen=100)

        self.total_episodes_done = 0
        self.total_steps         = 0
        self.update_count        = 0

        # ✅ FIX-T3: Next trigger tracking (khởi tạo trong train())
        self._next_log_ep        = 0
        self._next_viz_ep        = 0
        self._next_checkpoint_ep = 0

        # ── Dirs ─────────────────────────────────────────────────────────────
        is_kaggle = os.path.exists("/kaggle/working")
        base_dir  = Path("/kaggle/working/results") if is_kaggle else Path("results")

        self.output_dir     = base_dir / "mappo" / self.run_name
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.viz_dir        = self.output_dir / "viz"

        for d in [self.checkpoint_dir, self.viz_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._print_init()

    # ══════════════════════════════════════════════════════════════════════════
    # TRAIN
    # ══════════════════════════════════════════════════════════════════════════

    def train(
        self,
        total_episodes:         int,
        curriculum_manager                = None,
        seed:                   int       = 42,
        log_every_n_eps:        int       = 10,
        viz_every_n_eps:        int       = 50,
        checkpoint_every_n_eps: int       = 100,
    ):
        """
        Main training loop.

        ✅ FIX-T1: Checkpoint/Viz/Log dùng >= thay vì %
                   → Đảm bảo luôn trigger dù episodes_per_update lớn bất kỳ
        """
        start_time = time.time()
        env        = _EnvWrapper(self.config, self.n_envs, seed)

        # ✅ FIX-T3: Init next trigger points
        self._next_log_ep        = log_every_n_eps
        self._next_viz_ep        = viz_every_n_eps        if viz_every_n_eps > 0  else 10**9
        self._next_checkpoint_ep = checkpoint_every_n_eps

        # ── Print plan ───────────────────────────────────────────────────────
        print(f"\n🚀 MAPPO Training")
        print(f"  target episodes  : {total_episodes:,}")
        print(f"  n_envs           : {self.n_envs}")
        print(f"  max_steps/ep     : {self.config.env.max_steps}")
        print(f"  log every        : {log_every_n_eps} eps")
        print(f"  viz every        : {viz_every_n_eps} eps  → first at ep {self._next_viz_ep}")
        print(f"  checkpoint every : {checkpoint_every_n_eps} eps → first at ep {self._next_checkpoint_ep}")
        print(f"  output dir       : {self.output_dir.resolve()}\n")

        pbar = tqdm(
            total      = total_episodes,
            desc       = "🚁 Training",
            unit       = "ep",
            dynamic_ncols = True,
            bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} ep "
                         "[{elapsed}<{remaining}] {postfix}",
        )

        last_rollout: Dict = {}
        last_train:   Dict = {}

        while self.total_episodes_done < total_episodes:

            # ── Collect + Update ──────────────────────────────────────────────
            last_rollout = self._rollout(env, pbar, total_episodes)
            last_train   = self._update()
            self.update_count += 1

            ep = self.total_episodes_done  # Shortcut

            # ✅ FIX-T1: LOG — trigger khi ep >= ngưỡng tiếp theo
            if ep >= self._next_log_ep:
                elapsed = time.time() - start_time
                fps     = self.total_steps / max(elapsed, 1e-6)
                self._log_detail(
                    pbar, last_rollout, last_train,
                    elapsed, fps, curriculum_manager,
                )
                # Advance đến ngưỡng kế tiếp (xử lý trường hợp nhảy nhiều)
                while self._next_log_ep <= ep:
                    self._next_log_ep += log_every_n_eps

            # ✅ FIX-T1: VIZ — trigger khi ep >= ngưỡng tiếp theo
            if viz_every_n_eps > 0 and ep >= self._next_viz_ep:
                self._save_viz(env, ep)
                while self._next_viz_ep <= ep:
                    self._next_viz_ep += viz_every_n_eps

            # ✅ FIX-T1: CHECKPOINT — trigger khi ep >= ngưỡng tiếp theo
            if ep >= self._next_checkpoint_ep:
                self.save_checkpoint(ep, curriculum_manager)
                while self._next_checkpoint_ep <= ep:
                    self._next_checkpoint_ep += checkpoint_every_n_eps

            # ── Curriculum ────────────────────────────────────────────────────
            if curriculum_manager and self.ep_rewards:
                curriculum_manager.update(
                    coverage     = last_rollout.get("mean_coverage", 0) / 100,
                    victims_rate = last_rollout.get("mean_victims",  0) / 100,
                    reward       = last_rollout.get("mean_ep_reward", 0),
                )
                if curriculum_manager.should_advance():
                    old = curriculum_manager.current_stage.name
                    curriculum_manager.advance()
                    new = curriculum_manager.current_stage.name
                    pbar.write(f"\n🎓 CURRICULUM: {old.upper()} → {new.upper()}\n")
                    curriculum_manager.apply_to_config(self.config)
                    env.close()
                    env = _EnvWrapper(self.config, self.n_envs, seed)

        pbar.close()
        env.close()

        # ── Final checkpoint ──────────────────────────────────────────────────
        self.save_checkpoint(self.total_episodes_done, curriculum_manager, tag="final")
        self._print_final(time.time() - start_time, last_rollout)

        # ── Kaggle hint ───────────────────────────────────────────────────────
        if os.path.exists("/kaggle/working"):
            print(f"\n📥 KAGGLE DOWNLOAD:")
            print(f"   {self.checkpoint_dir}/checkpoint_final.pt")
            print(f"   {self.output_dir}/training_curves.png")

    # ══════════════════════════════════════════════════════════════════════════
    # ROLLOUT
    # ══════════════════════════════════════════════════════════════════════════

    def _rollout(
        self,
        env:          _EnvWrapper,
        pbar:         tqdm,
        max_episodes: int,
    ) -> Dict:

        obs_batch, g_batch = env.reset()
        ep_rew   = np.zeros(self.n_envs, dtype=np.float32)
        ep_len   = np.zeros(self.n_envs, dtype=np.int32)
        last_g   = g_batch.copy()
        last_dones = [False] * self.n_envs

        for _ in range(self.rollout_length):
            if self.total_episodes_done >= max_episodes:
                break

            # ── Inference ─────────────────────────────────────────────────────
            n        = self.n_envs
            obs_flat = obs_batch.reshape(n * self.n_agents, self.obs_dim)
            obs_t    = torch.FloatTensor(obs_flat).to(self.device)
            g_t      = torch.FloatTensor(g_batch).to(self.device)

            with torch.no_grad():
                act_t, lp_t = self.actor.get_action(obs_t)
                val_t       = self.critic.get_value(g_t)

            act_batch = act_t.cpu().numpy().reshape(n, self.n_agents, 3)
            lp_batch  = lp_t.cpu().numpy().reshape(n, self.n_agents)
            val_batch = np.repeat(
                val_t.cpu().numpy()[:, None], self.n_agents, axis=1
            )

            # ── Step ──────────────────────────────────────────────────────────
            next_obs, next_g, rews, dones, infos = env.step(act_batch)

            # ── Store + Track ─────────────────────────────────────────────────
            for ei in range(n):
                self.buffer.add(
                    obs        = obs_batch[ei],
                    global_obs = g_batch[ei],
                    actions    = act_batch[ei],
                    rewards    = rews[ei],
                    values     = val_batch[ei],
                    log_probs  = lp_batch[ei],
                    done       = dones[ei],
                )
                self.total_steps += 1
                ep_rew[ei] += float(rews[ei][0])
                ep_len[ei] += 1

                if dones[ei]:
                    info_ei = infos[ei] if infos[ei] else {}
                    u0      = info_ei.get("uav_0", {})

                    cov  = float(u0.get("coverage_rate",  0.0)) * 100
                    vf   = int(u0.get("victims_found",    0))
                    vt   = max(int(u0.get("victims_total", 1)), 1)

                    done_reason  = u0.get("done_reason", "unknown")
                    success      = u0.get("success",     False)

                    bstats       = u0.get("battery_stats", {})
                    battery_mean = bstats.get("mean", 0.0)
                    battery_min  = bstats.get("min",  0.0)

                    self.ep_rewards.append(float(ep_rew[ei]))
                    self.ep_lengths.append(int(ep_len[ei]))
                    self.ep_coverage.append(cov)
                    self.ep_victims.append(vf / vt * 100)
                    self.total_episodes_done += 1

                    # ── tqdm postfix ──────────────────────────────────────────
                    status = "✓" if success else "✗"
                    pbar.update(1)

                    reason_label = {
                        "disabled":  "🔋dead",
                        "truncated": "⏱timeout",
                        "coverage":  "✅cov",
                        "victims":   "✅vic",
                        None:        "?",
                    }.get(done_reason, done_reason[:6] if done_reason else "?")

                    pbar.set_postfix(ordered_dict={
                        "rew":  f"{float(ep_rew[ei]):+.0f}",
                        "cov":  f"{cov:.0f}%",
                        "vic":  f"{vf}/{vt}",
                        "step": f"{ep_len[ei]}/{self.config.env.max_steps}",
                        "bat":  f"{battery_mean:.0f}%",
                        "end":  f"{status}{reason_label}",
                    })

                    # ── Per-10-episode log ────────────────────────────────────
                    if self.total_episodes_done % 10 == 0:
                        pbar.write(
                            f"[EP {self.total_episodes_done:5d}] "
                            f"{status} {str(done_reason):8s} | "
                            f"step={ep_len[ei]:4d}/{self.config.env.max_steps} | "
                            f"bat={battery_mean:5.1f}% (min={battery_min:5.1f}%) | "
                            f"rew={ep_rew[ei]:+8.1f} | "
                            f"cov={cov:5.1f}% | "
                            f"vic={vf:2d}/{vt:2d}"
                        )

                    ep_rew[ei] = 0.0
                    ep_len[ei] = 0

                    if self.total_episodes_done >= max_episodes:
                        break

            obs_batch  = next_obs
            g_batch    = next_g
            last_g     = next_g.copy()
            last_dones = dones

        # ── GAE bootstrap ─────────────────────────────────────────────────────
        with torch.no_grad():
            last_vals = self.critic.get_value(
                torch.FloatTensor(last_g).to(self.device)
            ).cpu().numpy()

        bootstrap = np.full(
            self.n_agents,
            float(np.mean(last_vals)),
            dtype=np.float32,
        )
        self.buffer.compute_gae(bootstrap, last_done=all(last_dones))

        return {
            "mean_ep_reward": float(np.mean(self.ep_rewards)) if self.ep_rewards else 0.0,
            "mean_ep_length": float(np.mean(self.ep_lengths)) if self.ep_lengths else 0.0,
            "mean_coverage":  float(np.mean(self.ep_coverage)) if self.ep_coverage else 0.0,
            "mean_victims":   float(np.mean(self.ep_victims))  if self.ep_victims  else 0.0,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # UPDATE
    # ══════════════════════════════════════════════════════════════════════════

    def _update(self) -> Dict:
        actor_losses, critic_losses, entropies, clip_fracs = [], [], [], []

        for _ in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                obs     = torch.FloatTensor(batch["obs"]).to(self.device)
                g_obs   = torch.FloatTensor(batch["global_obs"]).to(self.device)
                actions = torch.FloatTensor(batch["actions"]).to(self.device)
                old_lp  = torch.FloatTensor(batch["old_log_probs"]).to(self.device)
                adv     = torch.FloatTensor(batch["advantages"]).to(self.device)
                returns = torch.FloatTensor(batch["returns"]).to(self.device)

                # Actor loss
                lp, entropy = self.actor.evaluate_actions(obs, actions)
                ratio = torch.exp(lp - old_lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(
                    ratio,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon,
                ) * adv
                a_loss = (
                    -torch.min(surr1, surr2).mean()
                    - self.entropy_coeff * entropy.mean()
                )

                self.actor_opt.zero_grad()
                a_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()

                # Critic loss
                values = self.critic.get_value(g_obs)
                c_loss = nn.functional.mse_loss(values, returns)

                self.critic_opt.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()

                actor_losses.append(a_loss.item())
                critic_losses.append(c_loss.item())
                entropies.append(entropy.mean().item())
                with torch.no_grad():
                    clip_fracs.append(
                        ((ratio - 1.0).abs() > self.clip_epsilon)
                        .float().mean().item()
                    )

        self.buffer.clear()

        return {
            "actor_loss":     float(np.mean(actor_losses)),
            "critic_loss":    float(np.mean(critic_losses)),
            "entropy":        float(np.mean(entropies)),
            "clip_fraction":  float(np.mean(clip_fracs)),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # LOGGING & VIZ
    # ══════════════════════════════════════════════════════════════════════════

    def _log_detail(
        self,
        pbar,
        rollout:            Dict,
        train:              Dict,
        elapsed:            float,
        fps:                float,
        curriculum_manager,
    ):
        lines = [
            f"\n{'─'*65}",
            f"📊 Episode {self.total_episodes_done:,} | Update {self.update_count:,}",
            f"{'─'*65}",
            f"  Task:",
            f"    reward   : {rollout['mean_ep_reward']:+10.2f}",
            f"    coverage : {rollout['mean_coverage']:8.2f}%",
            f"    victims  : {rollout['mean_victims']:8.2f}%",
            f"    ep_len   : {rollout['mean_ep_length']:8.1f} steps",
            f"  Train:",
            f"    a_loss   : {train['actor_loss']:10.4f}",
            f"    c_loss   : {train['critic_loss']:10.4f}",
            f"    entropy  : {train['entropy']:10.4f}",
            f"    clip_frac: {train['clip_fraction']:10.3f}",
            f"  Perf:",
            f"    fps      : {fps:10.1f}",
            f"    elapsed  : {elapsed/60:8.1f} min",
            f"  Next triggers:",
            f"    log      : ep {self._next_log_ep:,}",
            f"    viz      : ep {self._next_viz_ep:,}",
            f"    ckpt     : ep {self._next_checkpoint_ep:,}",
        ]

        if curriculum_manager:
            stage = curriculum_manager.current_stage
            lines += [
                f"  Curriculum:",
                f"    stage    : {stage.name.upper()}",
                f"    map_size : {stage.map_size}m",
            ]

        lines.append(f"{'─'*65}\n")
        for line in lines:
            pbar.write(line)

    def _save_viz(self, env: _EnvWrapper, episode: int):
        """Lưu 2D viz snapshot (chỉ single env)."""
        if env.n_envs > 1:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            frame = env.render()
            if frame is None:
                return

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(frame)
            ax.axis("off")
            ax.set_title(f"Episode {episode:,}", fontsize=14)

            path = self.viz_dir / f"ep_{episode:06d}.png"
            fig.savefig(path, bbox_inches="tight", dpi=100)
            plt.close(fig)

        except Exception:
            pass

    def plot_training_curves(self, save_path: str = None):
        """Vẽ 4-panel training curves."""
        try:
            import matplotlib
            if save_path:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if not self.ep_rewards:
                print("⚠️  No training data to plot")
                return

            episodes = list(range(1, len(self.ep_rewards) + 1))
            rewards  = list(self.ep_rewards)
            coverage = list(self.ep_coverage)
            victims  = list(self.ep_victims)
            lengths  = list(self.ep_lengths)

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(
                f"MAPPO Training Curves — {self.run_name}",
                fontsize=16, fontweight="bold",
            )

            window = max(5, min(20, len(rewards) // 5))

            def _plot(ax, data, color, smooth_color, ylabel, title,
                      target=None, target_label=None, ylim=None):
                ax.plot(episodes, data, alpha=0.3, color=color, linewidth=0.5)
                if len(data) >= window:
                    smoothed = np.convolve(
                        data, np.ones(window) / window, mode="valid"
                    )
                    sx = list(range(window, len(data) + 1))
                    ax.plot(sx, smoothed, color=smooth_color,
                            linewidth=2, label=f"MA({window})")
                if target is not None:
                    ax.axhline(target, color="orange", linestyle="--",
                               linewidth=0.8, alpha=0.6, label=target_label)
                ax.set_xlabel("Episode", fontsize=11)
                ax.set_ylabel(ylabel,    fontsize=11)
                ax.set_title(title,      fontsize=12, fontweight="bold")
                if ylim:
                    ax.set_ylim(*ylim)
                ax.grid(alpha=0.3)
                ax.legend()

            _plot(axes[0,0], rewards,  "steelblue", "darkblue",
                  "Reward",        "Episode Reward",     target=0)
            _plot(axes[0,1], coverage, "green",     "darkgreen",
                  "Coverage (%)",  "Coverage Rate",
                  target=70, target_label="Target 70%",  ylim=(0,100))
            _plot(axes[1,0], victims,  "purple",    "indigo",
                  "Victims (%)",   "Victims Found Rate",
                  target=80, target_label="Target 80%",  ylim=(0,100))
            _plot(axes[1,1], lengths,  "coral",     "red",
                  "Steps",         "Episode Length",
                  target=self.config.env.max_steps,
                  target_label=f"Max ({self.config.env.max_steps})")

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"📊 Training curves saved: {save_path}")
            else:
                plt.show()

        except Exception as e:
            print(f"⚠️  Failed to plot: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # CHECKPOINT
    # ══════════════════════════════════════════════════════════════════════════

    def save_checkpoint(
        self,
        episode:            int,
        curriculum_manager              = None,
        tag:                str = None,
    ):
        """
        Lưu checkpoint.

        ✅ FIX-T2: Print absolute path để dễ tìm file
        """
        name = (
            f"checkpoint_{tag}.pt"
            if tag
            else f"checkpoint_ep{episode:06d}.pt"
        )
        path = self.checkpoint_dir / name

        torch.save(
            {
                "episode":                      episode,
                "update":                       self.update_count,
                "total_episodes_done":          self.total_episodes_done,
                "actor_state_dict":             self.actor.state_dict(),
                "critic_state_dict":            self.critic.state_dict(),
                "actor_optimizer_state_dict":   self.actor_opt.state_dict(),
                "critic_optimizer_state_dict":  self.critic_opt.state_dict(),
                "ep_rewards":                   list(self.ep_rewards),
                "ep_coverage":                  list(self.ep_coverage),
                "ep_victims":                   list(self.ep_victims),
                "curriculum_stage": (
                    curriculum_manager.stage_idx
                    if curriculum_manager else 0
                ),
            },
            path,
        )

        # ✅ FIX-T2: Print absolute path
        print(f"\n💾 Checkpoint saved:")
        print(f"   {path.resolve()}")
        print(f"   episode={episode:,} | update={self.update_count:,} | "
              f"reward={np.mean(self.ep_rewards):+.1f} | "
              f"cov={np.mean(self.ep_coverage):.1f}%\n"
              if self.ep_rewards else "\n")

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint và restore state."""
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.actor_opt.load_state_dict(ckpt["actor_optimizer_state_dict"])
        self.critic_opt.load_state_dict(ckpt["critic_optimizer_state_dict"])
        self.total_episodes_done = ckpt.get("total_episodes_done", 0)
        self.update_count        = ckpt.get("update", 0)

        print(f"✅ Checkpoint loaded: {path}")
        print(f"   episode={self.total_episodes_done:,} | "
              f"update={self.update_count:,}")

        return ckpt.get("episode", 0)

    # ══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _print_init(self):
        print(f"\n{'='*65}")
        print(f"🚁 MAPPO Trainer")
        print(f"{'='*65}")
        print(f"  device     : {self.device}")
        print(f"  run_name   : {self.run_name}")
        print(f"  n_envs     : {self.n_envs}")
        print(f"  n_agents   : {self.n_agents}")
        print(f"  obs_dim    : {self.obs_dim}")
        print(f"  global_dim : {self.global_obs_dim}")
        print(f"  actor      : {sum(p.numel() for p in self.actor.parameters()):,} params")
        print(f"  critic     : {sum(p.numel() for p in self.critic.parameters()):,} params")
        print(f"  buffer     : {self.rollout_length * self.n_envs:,} steps")
        print(f"  output     : {self.output_dir.resolve()}")
        print(f"{'='*65}\n")

    def _print_final(self, elapsed: float, metrics: Dict):
        print(f"\n{'='*65}")
        print(f"✅ Training Complete!")
        print(f"{'='*65}")
        print(f"  episodes : {self.total_episodes_done:,}")
        print(f"  updates  : {self.update_count:,}")
        print(f"  steps    : {self.total_steps:,}")
        print(f"  time     : {elapsed/60:.1f} min  ({elapsed/3600:.2f} h)")

        if metrics:
            print(f"\n📊 Final Performance (last 100 eps):")
            print(f"  reward   : {metrics.get('mean_ep_reward',  0):+.2f}")
            print(f"  coverage : {metrics.get('mean_coverage',   0):.2f}%")
            print(f"  victims  : {metrics.get('mean_victims',    0):.2f}%")
            print(f"  ep_len   : {metrics.get('mean_ep_length',  0):.1f} steps")

        print(f"\n💾 Saved:")
        final_path = self.checkpoint_dir / "checkpoint_final.pt"
        print(f"  checkpoint : {final_path.resolve()}")

        plot_path = self.output_dir / "training_curves.png"
        self.plot_training_curves(save_path=str(plot_path))
        print(f"  plot       : {plot_path.resolve()}")
        print(f"{'='*65}\n")