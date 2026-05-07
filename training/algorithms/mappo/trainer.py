"""
training/algorithms/mappo/trainer.py
MAPPO Trainer — Kaggle optimized (no viz/plot during training)

FIXES:
    ✅ FIX-T1: Checkpoint/Log dùng >= thay vì % (không miss trigger)
    ✅ FIX-T2: Print absolute path khi save checkpoint
    ✅ FIX-T3: _next_trigger tracking cho tất cả intervals
    ✅ FIX-P0-3: Re-compute log_prob SAU khi clip action
    ✅ FIX-P1: Terminal obs không đưa vào buffer bước tiếp
    ✅ FIX-P2: rews_team.reshape() thay vì squeeze()
    ✅ FIX-K1: Xóa viz/plot — chỉ lưu checkpoint + metrics lên HF
    ✅ FIX-K2: _all_* full history cho metrics.json
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from hf_upload import HFUploader


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
            self._is_vec = False
        else:
            self._env    = VectorizedEnv(config, n_envs=n_envs, start_seed=seed)
            self._is_vec = True

        self._current_obs:    np.ndarray | None = None
        self._current_global: np.ndarray | None = None
        self._needs_reset:    bool = True

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        if self._is_vec:
            obs, g = self._env.reset()
        else:
            obs_d, info = self._env.reset(seed=self._seed)
            obs = np.array(
                [obs_d[f"uav_{i}"] for i in range(self.n_agents)],
                dtype=np.float32,
            )[None]
            g = info["uav_0"]["global_obs"][None]

        self._current_obs    = obs
        self._current_global = g
        self._needs_reset    = False
        return obs, g

    def get_current_obs(self) -> tuple[np.ndarray, np.ndarray]:
        if self._needs_reset or self._current_obs is None:
            return self.reset()
        return self._current_obs, self._current_global

    def step(self, actions_batch: np.ndarray):
        """
        actions_batch: [n_envs, n_agents, 4]
        ✅ FIX-P1: Khi done=True với n_envs=1, reset ngay và cache obs mới
        """
        if self._is_vec:
            obs, g, rews, dones, infos = self._env.step(actions_batch)
            self._current_obs    = obs
            self._current_global = g
            return obs, g, rews, dones, infos

        act_dict = {
            f"uav_{i}": actions_batch[0][i]
            for i in range(self.n_agents)
        }
        obs_d, rew_d, term_d, trunc_d, info = self._env.step(act_dict)

        done = any(term_d.values()) or any(trunc_d.values())

        obs_terminal = np.array(
            [
                obs_d.get(f"uav_{i}", np.zeros(self.obs_dim, np.float32))
                for i in range(self.n_agents)
            ],
            dtype=np.float32,
        )[None]

        g_terminal = info["uav_0"]["global_obs"][None]

        rews = np.array(
            [rew_d.get(f"uav_{i}", 0.0) for i in range(self.n_agents)],
            dtype=np.float32,
        )[None]

        dones = [done]
        infos = [info]

        if done:
            new_seed    = int(np.random.randint(0, 2**31))
            new_obs_d, new_info = self._env.reset(seed=new_seed)

            new_obs = np.array(
                [
                    new_obs_d.get(f"uav_{i}", np.zeros(self.obs_dim, np.float32))
                    for i in range(self.n_agents)
                ],
                dtype=np.float32,
            )[None]
            new_g = new_info["uav_0"]["global_obs"][None]

            self._current_obs    = new_obs
            self._current_global = new_g
            return new_obs, new_g, rews, dones, infos
        else:
            self._current_obs    = obs_terminal
            self._current_global = g_terminal
            return obs_terminal, g_terminal, rews, dones, infos

    def reset_hard(self) -> tuple[np.ndarray, np.ndarray]:
        self._needs_reset = True
        return self.reset()

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# MAPPO TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class MAPPOTrainer:
    """MAPPO Trainer — Kaggle optimized (no viz/plot during training)."""

    def __init__(
        self,
        config:          AppConfig,
        device:          str = "auto",
        run_name:        str = None,
        n_envs:          int = 1,
        hf_token:        str = None,
        hf_repo:         str = None,
        hf_upload_every: int = 500,
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
            obs_dim        = self.obs_dim,
            action_dim     = 4,
            hidden_dims    = tr.mappo_actor_hidden,
            activation     = tr.mappo_activation,
            use_layer_norm = tr.mappo_use_layer_norm,
            log_std_init   = -0.5,
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
        self.buffer = RolloutBuffer(
            rollout_length = self.rollout_length,
            n_envs         = n_envs,
            n_agents       = self.n_agents,
            obs_dim        = self.obs_dim,
            global_obs_dim = self.global_obs_dim,
            action_dim     = 4,
            gamma          = self.gamma,
            gae_lambda     = self.gae_lambda,
        )

        # ── Stats (rolling window) ────────────────────────────────────────────
        self.ep_rewards  = deque(maxlen=100)
        self.ep_lengths  = deque(maxlen=100)
        self.ep_coverage = deque(maxlen=100)
        self.ep_victims  = deque(maxlen=100)

        # ── Full history để upload HF ─────────────────────────────────────────
        self._all_rewards  = []
        self._all_coverage = []
        self._all_victims  = []
        self._all_lengths  = []

        self._persist_ep_len = np.zeros(n_envs, dtype=np.int32)
        self._persist_ep_rew = np.zeros(n_envs, dtype=np.float32)

        self.total_episodes_done = 0
        self.total_steps         = 0
        self.update_count        = 0

        self._next_log_ep        = 0
        self._next_checkpoint_ep = 0

        # ── HF Uploader ───────────────────────────────────────────────────────
        self.hf_uploader     = None
        self.hf_upload_every = hf_upload_every
        self._hf_run_name    = self.run_name

        if hf_token and hf_repo:
            self.hf_uploader = HFUploader(token=hf_token, repo_id=hf_repo)

        # ── Dirs (chỉ checkpoint, không có viz) ──────────────────────────────
        is_kaggle = os.path.exists("/kaggle/working")
        base_dir  = Path("/kaggle/working/results") if is_kaggle else Path("results")

        self.output_dir     = base_dir / "mappo" / self.run_name
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
        checkpoint_every_n_eps: int       = 100,
    ):
        start_time = time.time()
        env        = _EnvWrapper(self.config, self.n_envs, seed)

        self._next_log_ep        = log_every_n_eps
        self._next_checkpoint_ep = checkpoint_every_n_eps

        print(f"\n🚀 MAPPO Training (Kaggle mode — no viz)")
        print(f"  target episodes  : {total_episodes:,}")
        print(f"  n_envs           : {self.n_envs}")
        print(f"  max_steps/ep     : {self.config.env.max_steps}")
        print(f"  log every        : {log_every_n_eps} eps")
        print(f"  checkpoint every : {checkpoint_every_n_eps} eps")
        print(f"  hf_upload        : {self.hf_uploader is not None}")
        print(f"  output dir       : {self.output_dir.resolve()}\n")

        pbar = tqdm(
            total         = total_episodes,
            desc          = "🚁 Training",
            unit          = "ep",
            dynamic_ncols = True,
            bar_format    = (
                "{l_bar}{bar}| {n_fmt}/{total_fmt} ep "
                "[{elapsed}<{remaining}] {postfix}"
            ),
        )

        last_rollout: dict = {}
        last_train:   dict = {}

        env.reset()

        while self.total_episodes_done < total_episodes:

            last_rollout = self._rollout(env, pbar, total_episodes)
            last_train   = self._update()
            self.update_count += 1

            ep = self.total_episodes_done

            if ep >= self._next_log_ep:
                elapsed = time.time() - start_time
                fps     = self.total_steps / max(elapsed, 1e-6)
                self._log_detail(
                    pbar, last_rollout, last_train,
                    elapsed, fps, curriculum_manager,
                )
                while self._next_log_ep <= ep:
                    self._next_log_ep += log_every_n_eps

            if ep >= self._next_checkpoint_ep:
                self.save_checkpoint(ep, curriculum_manager)
                while self._next_checkpoint_ep <= ep:
                    self._next_checkpoint_ep += checkpoint_every_n_eps

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
                    env.reset()

        pbar.close()
        env.close()

        self.save_checkpoint(
            self.total_episodes_done, curriculum_manager, tag="final"
        )
        self._print_final(time.time() - start_time, last_rollout)

    # ══════════════════════════════════════════════════════════════════════════
    # ROLLOUT
    # ══════════════════════════════════════════════════════════════════════════

    def _rollout(self, env: _EnvWrapper, pbar, max_episodes: int) -> Dict:
        """
        Collect rollout data.
        ✅ FIX-P0-3: Re-compute log_prob SAU khi clip action
        ✅ FIX-P2:   rews_team.reshape(n_envs) thay vì squeeze()
        """
        obs_batch, g_batch = env.get_current_obs()

        last_g     = g_batch.copy()
        last_dones = np.zeros(self.n_envs, dtype=np.float32)

        for _ in range(self.rollout_length):
            if self.total_episodes_done >= max_episodes:
                break

            n = self.n_envs

            obs_flat = obs_batch.reshape(n * self.n_agents, self.obs_dim)
            obs_t    = torch.FloatTensor(obs_flat).to(self.device)
            g_t      = torch.FloatTensor(g_batch).to(self.device)

            with torch.no_grad():
                act_t, _ = self.actor.get_action(obs_t)
                val_t    = self.critic.get_value(g_t)

            act_batch = act_t.cpu().numpy().reshape(n, self.n_agents, -1)
            act_batch = np.clip(act_batch, -1.0, 1.0)

            # ✅ FIX-P0-3: Re-compute log_prob với clipped action
            act_clipped_t = torch.FloatTensor(
                act_batch.reshape(n * self.n_agents, -1)
            ).to(self.device)

            with torch.no_grad():
                lp_t, _ = self.actor.evaluate_actions(obs_t, act_clipped_t)

            lp_batch  = lp_t.cpu().numpy().reshape(n, self.n_agents)
            val_np    = val_t.cpu().numpy()
            val_batch = np.repeat(val_np[:, None], self.n_agents, axis=1)

            next_obs, next_g, rews, dones, infos = env.step(act_batch)

            rews_team   = rews.sum(axis=1, keepdims=True)
            rews_shared = np.repeat(rews_team, self.n_agents, axis=1)
            dones_arr   = np.array(dones, dtype=np.float32)

            self.buffer.add(
                obs        = obs_batch,
                global_obs = g_batch,
                actions    = act_batch,
                rewards    = rews_shared,
                values     = val_batch,
                log_probs  = lp_batch,
                dones      = dones_arr,
            )
            self.total_steps += self.n_envs

            # ✅ FIX-P2: reshape thay vì squeeze
            self._persist_ep_rew += rews_team.reshape(self.n_envs)
            self._persist_ep_len += 1

            for ei in range(n):
                if dones[ei]:
                    info_ei    = infos[ei] if infos[ei] else {}
                    u0         = info_ei.get("uav_0", {})
                    ep_metrics = u0.get("episode", {})

                    if ep_metrics:
                        cov = float(ep_metrics.get("coverage_rate", 0.0))
                        vf  = int(ep_metrics.get("victims_found", 0))
                        vt  = max(int(ep_metrics.get("total_victims", 1)), 1)
                    else:
                        cov = float(u0.get("coverage_rate", 0.0)) * 100
                        vf  = int(u0.get("victims_found", 0))
                        vt  = max(int(u0.get("victims_total", 1)), 1)

                    done_reason    = (ep_metrics.get("done_reason") or u0.get("done_reason", "unknown"))
                    success        = ep_metrics.get("success") or u0.get("success", False)
                    bstats         = u0.get("battery_stats", {})
                    battery_mean   = bstats.get("mean", 0.0)
                    total_landings = int(ep_metrics.get("total_landings", 0))

                    actual_ep_len = int(self._persist_ep_len[ei])
                    actual_ep_rew = float(self._persist_ep_rew[ei])

                    # Rolling window
                    self.ep_rewards.append(actual_ep_rew)
                    self.ep_lengths.append(actual_ep_len)
                    self.ep_coverage.append(cov)
                    self.ep_victims.append(vf / vt * 100)

                    # ✅ Full history
                    self._all_rewards.append(actual_ep_rew)
                    self._all_lengths.append(actual_ep_len)
                    self._all_coverage.append(cov)
                    self._all_victims.append(vf / vt * 100)

                    self.total_episodes_done += 1

                    status = "✓" if success else "✗"
                    pbar.update(1)

                    reason_label = {
                        "disabled:battery_death": "🔋AllDead",
                        "disabled:other":         "🔋Mixed",
                        "disabled":               "🔋dead",
                        "truncated":              "⏱Timeout",
                        "coverage":               "✅Cov",
                        "victims":                "✅Vic",
                        None:                     "?Unknown",
                    }.get(done_reason, done_reason[:8] if done_reason else "?")

                    pbar.set_postfix(ordered_dict={
                        "rew":  f"{actual_ep_rew:+.0f}",
                        "cov":  f"{cov:.0f}%",
                        "vic":  f"{vf}/{vt}",
                        "step": f"{actual_ep_len}/{self.config.env.max_steps}",
                        "bat":  f"{battery_mean:.0f}%",
                        "land": f"{total_landings}×",
                        "end":  f"{status}{reason_label}",
                    })

                    self._persist_ep_len[ei] = 0
                    self._persist_ep_rew[ei] = 0.0

                    if self.total_episodes_done >= max_episodes:
                        break

            obs_batch  = next_obs
            g_batch    = next_g
            last_g     = next_g.copy()
            last_dones = dones_arr

        # GAE bootstrap
        with torch.no_grad():
            last_vals = self.critic.get_value(
                torch.FloatTensor(last_g).to(self.device)
            ).cpu().numpy()

        self.buffer.compute_gae(last_values=last_vals, last_dones=last_dones)

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

                lp, entropy = self.actor.evaluate_actions(obs, actions)
                ratio = torch.exp(lp - old_lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv
                a_loss = (
                    -torch.min(surr1, surr2).mean()
                    - self.entropy_coeff * entropy.mean()
                )

                self.actor_opt.zero_grad()
                a_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()

                values = self.critic.get_value(g_obs)
                c_loss = nn.functional.mse_loss(values, returns) / self.n_agents

                self.critic_opt.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()

                actor_losses.append(a_loss.item())
                critic_losses.append(c_loss.item())
                entropies.append(entropy.mean().item())
                with torch.no_grad():
                    clip_fracs.append(
                        ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    )

        self.buffer.clear()

        return {
            "actor_loss":    float(np.mean(actor_losses)),
            "critic_loss":   float(np.mean(critic_losses)),
            "entropy":       float(np.mean(entropies)),
            "clip_fraction": float(np.mean(clip_fracs)),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # LOGGING (text only)
    # ══════════════════════════════════════════════════════════════════════════

    def _log_detail(self, pbar, rollout, train, elapsed, fps, curriculum_manager):
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

    # ══════════════════════════════════════════════════════════════════════════
    # CHECKPOINT
    # ══════════════════════════════════════════════════════════════════════════

    def save_checkpoint(self, episode, curriculum_manager=None, tag=None):
        name = (
            f"checkpoint_{tag}.pt"
            if tag
            else f"checkpoint_ep{episode:06d}.pt"
        )
        path = self.checkpoint_dir / name

        torch.save(
            {
                "episode":                     episode,
                "update":                      self.update_count,
                "total_episodes_done":         self.total_episodes_done,
                "actor_state_dict":            self.actor.state_dict(),
                "critic_state_dict":           self.critic.state_dict(),
                "actor_optimizer_state_dict":  self.actor_opt.state_dict(),
                "critic_optimizer_state_dict": self.critic_opt.state_dict(),
                # ✅ Full history
                "ep_rewards":                  self._all_rewards,
                "ep_coverage":                 self._all_coverage,
                "ep_victims":                  self._all_victims,
                "ep_lengths":                  self._all_lengths,
                "curriculum_stage": (
                    curriculum_manager.stage_idx
                    if curriculum_manager else 0
                ),
            },
            path,
        )

        print(f"\n💾 Checkpoint saved: {path.resolve()}")

        # ── Upload HF ─────────────────────────────────────────────────────────
        if self.hf_uploader:
            metrics = {
                "algo":           "mappo",
                "run_name":       self.run_name,
                "total_episodes": self.total_episodes_done,
                "total_steps":    self.total_steps,
                "mean_reward":    float(np.mean(self.ep_rewards))  if self.ep_rewards  else 0.0,
                "mean_coverage":  float(np.mean(self.ep_coverage)) if self.ep_coverage else 0.0,
                "mean_victims":   float(np.mean(self.ep_victims))  if self.ep_victims  else 0.0,
                # ✅ Full history cho plot_compare.py
                "ep_rewards":     self._all_rewards,
                "ep_coverage":    self._all_coverage,
                "ep_victims":     self._all_victims,
                "ep_lengths":     self._all_lengths,
            }

            if tag == "final":
                # ✅ Không plot, chỉ upload checkpoint + metrics
                self.hf_uploader.upload_final(
                    run_name        = self._hf_run_name,
                    checkpoint_path = path,
                    metrics         = metrics,
                    plot_path       = None,
                )
                self._cleanup_local()

            elif episode % self.hf_upload_every == 0:
                self.hf_uploader.upload_checkpoint(
                    checkpoint_path = path,
                    run_name        = self._hf_run_name,
                    episode         = episode,
                    metrics         = metrics,
                )
                try:
                    path.unlink()
                except Exception:
                    pass

        if self.ep_rewards:
            print(
                f"   episode={episode:,} | reward={np.mean(self.ep_rewards):+.1f} | "
                f"cov={np.mean(self.ep_coverage):.1f}%\n"
            )

    def _cleanup_local(self):
        if not self.hf_uploader:
            return
        import shutil
        try:
            shutil.rmtree(str(self.output_dir))
            print(f"🗑️  Local files cleaned: {self.output_dir}")
        except Exception as e:
            print(f"⚠️  Cleanup failed: {e}")

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.actor_opt.load_state_dict(ckpt["actor_optimizer_state_dict"])
        self.critic_opt.load_state_dict(ckpt["critic_optimizer_state_dict"])
        self.total_episodes_done = ckpt.get("total_episodes_done", 0)
        self.update_count        = ckpt.get("update", 0)
        # Restore history
        self._all_rewards  = ckpt.get("ep_rewards",  [])
        self._all_coverage = ckpt.get("ep_coverage", [])
        self._all_victims  = ckpt.get("ep_victims",  [])
        self._all_lengths  = ckpt.get("ep_lengths",  [])
        print(f"✅ Checkpoint loaded: {path}")
        print(f"   episode={self.total_episodes_done:,} | update={self.update_count:,}")
        return ckpt.get("episode", 0)

    # ══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _print_init(self):
        print(f"\n{'='*65}")
        print(f"🚁 MAPPO Trainer — Kaggle Mode")
        print(f"{'='*65}")
        print(f"  device     : {self.device}")
        print(f"  run_name   : {self.run_name}")
        print(f"  n_envs     : {self.n_envs}")
        print(f"  n_agents   : {self.n_agents}")
        print(f"  obs_dim    : {self.obs_dim}")
        print(f"  global_dim : {self.global_obs_dim}")
        print(f"  actor      : {sum(p.numel() for p in self.actor.parameters()):,} params")
        print(f"  critic     : {sum(p.numel() for p in self.critic.parameters()):,} params")
        print(
            f"  buffer     : {self.rollout_length}T × {self.n_envs}E × {self.n_agents}A "
            f"= {self.rollout_length * self.n_envs * self.n_agents:,} transitions"
        )
        print(f"  hf_upload  : {self.hf_uploader is not None}")
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

        if self.hf_uploader:
            print(f"\n🤗 HuggingFace:")
            print(f"  https://huggingface.co/datasets/duy95/sar-uav-results")
            print(f"  tree/main/{self._hf_run_name}")
            print(f"\n📥 Download checkpoint:")
            print(f"  python plot_compare.py --skip-train --run-name {self._hf_run_name}")
        else:
            final_path = self.checkpoint_dir / "checkpoint_final.pt"
            print(f"\n💾 Saved locally:")
            print(f"  checkpoint : {final_path.resolve()}")

        print(f"{'='*65}\n")