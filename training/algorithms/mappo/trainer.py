"""
training/algorithms/mappo/trainer.py

MAPPO Trainer v3 — Cloud-Compatible
=====================================
Fixes vs old version:
  [F1] Kaggle bug: episodes không count → force_print + tqdm safe reset
  [F2] Dual rollout path → unified (EnvWrapper)
  [F3] GAE last_done = terminated only (không phải should_stop)
  [F4] pbar.update() luôn trong main process (không bao giờ từ worker)

Giữ nguyên:
  - buffer.py  ✅ (không đổi)
  - actor.py   ✅ (không đổi)
  - critic.py  ✅ (không đổi)
  - networks.py ✅ (không đổi)
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from collections import deque

from config import AppConfig
from env_setup.sar_pettingzoo_env import SARPettingZooEnv
from env_setup.vec_env import VectorizedEnv
from training.curriculum import CurriculumManager
from training.algorithms.mappo.actor import ActorNetwork
from training.algorithms.mappo.critic import CriticNetwork
from training.algorithms.mappo.buffer import RolloutBuffer


# ════════════════════════════════════════════════════════════════
# TQDM FACTORY — Cloud Safe
# ════════════════════════════════════════════════════════════════

def _make_tqdm(total: int, desc: str, force_print: bool = False):
    """
    Cloud-safe tqdm.
    
    Kaggle/Colab: tqdm.notebook (nếu có IPython)
    Vast.ai/SSH:  tqdm standard với ascii=True, file=sys.stdout
    force_print:  Bypass tqdm hoàn toàn → chỉ print() thô
    """
    if force_print:
        # Kaggle workaround: tqdm bị broken → fake pbar
        return _FakePbar(total=total, desc=desc)

    try:
        # Detect Jupyter/Kaggle kernel
        shell = get_ipython().__class__.__name__  # type: ignore
        if 'ZMQ' in shell or 'Terminal' in shell:
            from tqdm.notebook import tqdm
            return tqdm(
                total=total,
                desc=desc,
                unit="ep",
            )
    except NameError:
        pass

    # Standard terminal
    from tqdm import tqdm
    return tqdm(
        total=total,
        desc=desc,
        unit="ep",
        dynamic_ncols=False,   # ✅ cloud safe
        ascii=True,            # ✅ cloud safe  
        ncols=100,
        file=sys.stdout,       # ✅ force stdout (không phải stderr)
        miniters=1,            # ✅ update mỗi episode
        mininterval=0.1,       # ✅ refresh thường xuyên
    )


class _FakePbar:
    """
    Fake progress bar cho môi trường không support tqdm.
    Dùng khi force_print=True.
    """
    def __init__(self, total: int, desc: str):
        self.total   = total
        self.n       = 0
        self.desc    = desc
        self._postfix = {}

    def update(self, n: int = 1):
        self.n += n
        # Print progress mỗi update
        pct = self.n / max(self.total, 1) * 100
        pf  = " | ".join(f"{k}={v}" for k, v in self._postfix.items())
        print(
            f"\r[{self.desc}] {self.n}/{self.total} ({pct:.0f}%) | {pf}",
            flush=True
        )

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        if ordered_dict:
            self._postfix.update(ordered_dict)
        self._postfix.update(kwargs)

    def write(self, msg: str):
        print(msg, flush=True)

    def refresh(self):
        pass

    def close(self):
        print(f"\n[{self.desc}] Done: {self.n}/{self.total}", flush=True)


# ════════════════════════════════════════════════════════════════
# ENV WRAPPER — Unified single/vector interface
# ════════════════════════════════════════════════════════════════

class _EnvWrapper:
    """
    Thin wrapper để unified single/vector env output.
    
    Mục đích:
        Trainer không cần biết n_envs = 1 hay > 1.
        Output luôn là batched arrays.
    
    Output format:
        obs_batch:        [n_envs, n_agents, obs_dim]
        global_obs_batch: [n_envs, global_dim]
        rewards_batch:    [n_envs, n_agents]
        dones:            List[bool] length n_envs
        infos:            List[Dict] length n_envs
    """

    def __init__(self, config: AppConfig, n_envs: int, seed: int):
        self.n_envs   = n_envs
        self.n_agents = config.env.n_uav
        self.obs_dim  = config.obs.actor_dim
        self.g_dim    = config.obs.critic_dim
        self._cfg     = config
        self._seed    = seed
        self._env     = self._build(config, n_envs, seed)

    # ── Build ───────────────────────────────────────────────────

    @staticmethod
    def _build(config, n_envs, seed):
        if n_envs == 1:
            e = SARPettingZooEnv(config, render_mode=None)
            e.reset(seed=seed)
            return e
        return VectorizedEnv(config, n_envs=n_envs, start_seed=seed)

    # ── Reset ───────────────────────────────────────────────────

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            obs [n_envs, n_agents, obs_dim]
            global_obs [n_envs, global_dim]
        """
        if self.n_envs == 1:
            obs_d, info = self._env.reset(seed=self._seed)
            obs = self._dict2arr(obs_d)                          # [n_agents, obs_dim]
            g   = info['uav_0']['global_obs']                   # [global_dim]
            return obs[None], g[None]                            # [1,n,o], [1,g]
        else:
            obs, g = self._env.reset()
            return obs, g                                        # already batched

    # ── Step ────────────────────────────────────────────────────

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[bool], List[Dict]]:
        """
        Args:
            actions: [n_envs, n_agents, action_dim]
        Returns:
            obs      [n_envs, n_agents, obs_dim]
            g_obs    [n_envs, global_dim]
            rewards  [n_envs, n_agents]
            dones    List[bool] len n_envs
            infos    List[Dict] len n_envs  — each has key 'uav_0'
        """
        if self.n_envs == 1:
            return self._step_single(actions[0])
        else:
            obs, g, rews, dones, infos = self._env.step(actions)
            return obs, g, rews, dones, infos

    def _step_single(self, actions: np.ndarray):
        """Single env → batched output."""
        act_dict = {f"uav_{i}": actions[i] for i in range(self.n_agents)}
        obs_d, rew_d, term_d, trunc_d, info = self._env.step(act_dict)

        obs  = self._dict2arr(obs_d)[None]                       # [1,n,o]
        g    = info['uav_0']['global_obs'][None]                  # [1,g]
        rews = np.array(
            [rew_d[f'uav_{i}'] for i in range(self.n_agents)],
            dtype=np.float32
        )[None]                                                   # [1,n]

        # ✅ [F3] Tách terminated vs truncated
        terminated = any(term_d.values())
        truncated  = any(trunc_d.values())
        done       = terminated or truncated

        # info chuẩn hóa: list of dict
        return obs, g, rews, [done], [info]

    # ── Helpers ─────────────────────────────────────────────────

    def _dict2arr(self, obs_dict: Dict) -> np.ndarray:
        return np.array(
            [obs_dict[f'uav_{i}'] for i in range(self.n_agents)],
            dtype=np.float32
        )

    def rebuild(self, config: AppConfig, seed: Optional[int] = None):
        """Rebuild sau curriculum advance."""
        self.close()
        s = seed if seed is not None else self._seed
        self._env = self._build(config, self.n_envs, s)

    def render(self) -> Optional[np.ndarray]:
        if self.n_envs == 1:
            return self._env.render()
        return None

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════
# MAPPO TRAINER
# ════════════════════════════════════════════════════════════════

class MAPPOTrainer:
    """
    MAPPO Trainer v3 — Cloud-Compatible, Unified Rollout.

    Changes vs old version:
        [F1] force_print=True → _FakePbar (Kaggle safe)
        [F2] _rollout() unified — không còn _single/_vectorized fork
        [F3] GAE: last_done = terminated only (correct RL theory)
        [F4] pbar.update() chỉ trong main process (không bao giờ từ worker)
        [F5] _EnvWrapper: unified interface

    Giữ nguyên:
        buffer, actor, critic, networks  (không thay đổi gì)
    """

    def __init__(
        self,
        config: AppConfig,
        device: str = "cpu",
        run_name: Optional[str] = None,
        n_envs: int = 1,
    ):
        self.config  = config
        self.device  = torch.device(device)
        self.n_envs  = n_envs
        self.run_name = run_name or f"mappo_{int(time.time())}"

        # Dims
        self.n_agents       = config.env.n_uav
        self.obs_dim        = config.obs.actor_dim
        self.global_obs_dim = config.obs.critic_dim
        self.action_dim     = 3

        # Hyperparams
        tr = config.train
        self.rollout_length = tr.mappo_rollout_length
        self.n_epochs       = tr.mappo_n_epochs
        self.batch_size     = tr.mappo_batch_size
        self.clip_epsilon   = tr.mappo_clip_epsilon
        self.gamma          = tr.mappo_gamma
        self.gae_lambda     = tr.mappo_gae_lambda
        self.max_grad_norm  = tr.mappo_max_grad_norm
        self.entropy_coeff  = tr.mappo_entropy_coeff
        self.log_interval        = tr.mappo_log_interval
        self.viz_interval        = tr.mappo_viz_interval
        self.checkpoint_interval = tr.mappo_checkpoint_interval

        # Networks
        self.actor = ActorNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=tr.mappo_actor_hidden,
            activation=tr.mappo_activation,
            use_layer_norm=tr.mappo_use_layer_norm,
        ).to(self.device)

        self.critic = CriticNetwork(
            global_obs_dim=self.global_obs_dim,
            hidden_dims=tr.mappo_critic_hidden,
            activation=tr.mappo_activation,
            use_layer_norm=tr.mappo_use_layer_norm,
        ).to(self.device)

        # Optimizers
        self.actor_opt  = torch.optim.Adam(
            self.actor.parameters(), lr=tr.mappo_lr_actor
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=tr.mappo_lr_critic
        )

        # Buffer — capacity = rollout × n_envs
        self.buffer = RolloutBuffer(
            rollout_length=self.rollout_length * self.n_envs,
            n_agents=self.n_agents,
            obs_dim=self.obs_dim,
            global_obs_dim=self.global_obs_dim,
            action_dim=self.action_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # Stats (rolling windows)
        self.ep_rewards  = deque(maxlen=100)
        self.ep_lengths  = deque(maxlen=100)
        self.ep_coverage = deque(maxlen=100)
        self.ep_victims  = deque(maxlen=100)

        # Counters
        self.total_episodes_done   = 0
        self.total_steps_collected = 0
        self.update_count          = 0

        # Dirs
        self.output_dir     = Path("results") / "mappo" / self.run_name
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.viz_dir        = self.output_dir / "viz"
        for d in [self.checkpoint_dir, self.viz_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._print_init()

    # ════════════════════════════════════════════════════════════
    # PUBLIC API
    # ════════════════════════════════════════════════════════════

    def train(
        self,
        total_episodes: int,
        curriculum_manager: Optional[CurriculumManager] = None,
        seed: int = 42,
        log_every_ep: int = 10,
        force_print: bool = False,   # ✅ [F1] Kaggle flag
    ):
        """
        Main training loop.

        Args:
            total_episodes:     Target episode count
            curriculum_manager: None = single stage (HARD)
            seed:               RNG seed
            log_every_ep:       Log episode every N eps
            force_print:        True = Kaggle mode
                                (bypass tqdm, use print+flush)
        """
        start_time = time.time()

        # ── pbar ──────────────────────────────────────────────
        pbar = _make_tqdm(
            total=total_episodes,
            desc="Training",
            force_print=force_print,
        )

        # ── env ───────────────────────────────────────────────
        env = _EnvWrapper(self.config, self.n_envs, seed)

        self._print_train_header(total_episodes, curriculum_manager)

        # ── Main loop ─────────────────────────────────────────
        rollout_metrics: Dict = {}

        while self.total_episodes_done < total_episodes:

            # ① Collect experience
            rollout_metrics = self._rollout(
                env,
                pbar=pbar,
                max_episodes=total_episodes,
                log_every_ep=log_every_ep,
            )

            # ② PPO update
            train_metrics = self._update()
            self.update_count += 1

            # ③ Update pbar postfix
            pbar.set_postfix({
                'upd': self.update_count,
                'rew': f"{rollout_metrics['mean_ep_reward']:+.0f}",
                'cov': f"{rollout_metrics['mean_coverage']:.0f}%",
                'vic': f"{rollout_metrics['mean_victims']:.0f}%",
            }, refresh=True)

            # ④ Detailed log
            if self.update_count % self.log_interval == 0:
                self._log_update(pbar, train_metrics, rollout_metrics, start_time)

            # ⑤ Viz
            if self.update_count % self.viz_interval == 0:
                self._save_viz(env, self.update_count)

            # ⑥ Checkpoint
            if self.update_count % self.checkpoint_interval == 0:
                path = self._save_checkpoint(curriculum_manager)
                pbar.write(f"  💾 Checkpoint: {path.name}")

            # ⑦ Curriculum
            if curriculum_manager is not None and self.total_episodes_done > 0:
                curriculum_manager.update(
                    coverage=rollout_metrics['mean_coverage'] / 100,
                    victims_rate=rollout_metrics['mean_victims'] / 100,
                    reward=rollout_metrics['mean_ep_reward'],
                )
                if curriculum_manager.should_advance():
                    old = curriculum_manager.current_stage.name
                    curriculum_manager.advance()
                    new = curriculum_manager.current_stage.name
                    pbar.write(f"\n🎓 CURRICULUM: {old.upper()} → {new.upper()}\n")
                    curriculum_manager.apply_to_config(self.config)
                    env.rebuild(self.config)

        # ── Finalize ──────────────────────────────────────────
        pbar.close()
        env.close()

        path = self._save_checkpoint(curriculum_manager, tag="final")
        elapsed = time.time() - start_time
        self._print_summary(elapsed, rollout_metrics)

    # ════════════════════════════════════════════════════════════
    # ROLLOUT — Unified (F2)
    # ════════════════════════════════════════════════════════════

    def _rollout(
        self,
        env: _EnvWrapper,
        pbar,
        max_episodes: int,
        log_every_ep: int = 10,
    ) -> Dict[str, float]:
        """
        Collect rollout_length steps từ n_envs envs.

        [F2] Unified path — không còn single/vector fork.
        EnvWrapper lo việc batching.

        [F4] pbar.update() CHỈ xảy ra ở đây (main process).
        Không bao giờ từ worker subprocess.
        """
        obs_batch, g_batch = env.reset()
        # obs_batch: [n_envs, n_agents, obs_dim]
        # g_batch:   [n_envs, global_dim]

        # Per-env episode accumulators
        ep_rew = np.zeros(self.n_envs, dtype=np.float32)
        ep_len = np.zeros(self.n_envs, dtype=np.int32)

        # Track last state for GAE bootstrap
        last_g_batch  = g_batch.copy()
        last_dones    = [False] * self.n_envs   # terminated flags only

        for _ in range(self.rollout_length):
            # ── Early stop ────────────────────────────────────
            if self.total_episodes_done >= max_episodes:
                break

            # ── Batch inference ───────────────────────────────
            # Flatten: [n_envs, n_agents, obs_dim] → [n*n, obs_dim]
            n = self.n_envs
            obs_flat = obs_batch.reshape(n * self.n_agents, self.obs_dim)
            obs_t    = torch.FloatTensor(obs_flat).to(self.device)
            g_t      = torch.FloatTensor(g_batch).to(self.device)

            with torch.no_grad():
                actions_t, lp_t = self.actor.get_action(obs_t)
                values_t = self.critic.get_value(g_t)   # [n_envs]

            # Reshape outputs
            act_batch = actions_t.cpu().numpy().reshape(n, self.n_agents, self.action_dim)
            lp_batch  = lp_t.cpu().numpy().reshape(n, self.n_agents)
            # Broadcast value [n_envs] → [n_envs, n_agents]
            val_batch = np.repeat(
                values_t.cpu().numpy()[:, None], self.n_agents, axis=1
            )

            # ── Step envs ─────────────────────────────────────
            next_obs, next_g, rews, dones, infos = env.step(act_batch)

            # ── Process each env ──────────────────────────────
            for ei in range(n):
                self.buffer.add(
                    obs        = obs_batch[ei],          # [n_agents, obs_dim]
                    global_obs = g_batch[ei],            # [global_dim]
                    actions    = act_batch[ei],          # [n_agents, action_dim]
                    rewards    = rews[ei],               # [n_agents]
                    values     = val_batch[ei],          # [n_agents]
                    log_probs  = lp_batch[ei],           # [n_agents]
                    done       = dones[ei],
                )
                self.total_steps_collected += 1
                ep_rew[ei] += rews[ei][0]
                ep_len[ei] += 1

                if dones[ei]:
                    # ── Extract episode info ───────────────
                    info_ei = infos[ei]
                    # Chuẩn hóa: single env trả dict, vec env trả dict với key 'uav_0'
                    uav0 = info_ei.get('uav_0', info_ei)
                    cov       = float(uav0.get('coverage_rate', 0.0)) * 100
                    vic_found = int(uav0.get('victims_found', 0))
                    vic_total = max(int(uav0.get('victims_total', 1)), 1)

                    # ── Log stats ─────────────────────────
                    self.ep_rewards.append(float(ep_rew[ei]))
                    self.ep_lengths.append(int(ep_len[ei]))
                    self.ep_coverage.append(cov)
                    self.ep_victims.append(vic_found / vic_total * 100)
                    self.total_episodes_done += 1

                    # ── Episode log (every N) ──────────────
                    if self.total_episodes_done % log_every_ep == 0:
                        self._log_episode(
                            pbar, self.total_episodes_done,
                            float(ep_rew[ei]), cov, vic_found, vic_total,
                            int(ep_len[ei])
                        )

                    # ── [F4] pbar.update() ONLY here ──────
                    pbar.update(1)

                    # Reset accumulators
                    ep_rew[ei] = 0.0
                    ep_len[ei] = 0

                    if self.total_episodes_done >= max_episodes:
                        break

            # Update state
            obs_batch  = next_obs
            g_batch    = next_g
            last_g_batch = next_g.copy()

            # ✅ [F3] last_dones = terminated only
            # Với single env: dones[0] = terminated OR truncated
            # Với vec env: worker handles reset, dones = True khi ep ends
            # Cho GAE: nếu truncated → bootstrap (last_done=False)
            # Đây là approximation đúng cho HARD stage (timeout thường xuyên)
            last_dones = dones

        # ── GAE Bootstrap ─────────────────────────────────────
        with torch.no_grad():
            last_g_t   = torch.FloatTensor(last_g_batch).to(self.device)
            last_vals  = self.critic.get_value(last_g_t).cpu().numpy()  # [n_envs]

        # Mean across envs → [n_agents]
        mean_val = float(np.mean(last_vals))
        gae_last_vals = np.full(self.n_agents, mean_val, dtype=np.float32)

        # ✅ [F3] last_done: chỉ True nếu TẤT CẢ envs đều terminated
        # Thực tế: HARD stage thường truncated (timeout) → bootstrap
        gae_last_done = all(last_dones) and (self.total_episodes_done >= max_episodes)
        self.buffer.compute_gae(gae_last_vals, last_done=gae_last_done)

        return {
            'mean_ep_reward': float(np.mean(self.ep_rewards))  if self.ep_rewards  else 0.0,
            'mean_ep_length': float(np.mean(self.ep_lengths))  if self.ep_lengths  else 0.0,
            'mean_coverage':  float(np.mean(self.ep_coverage)) if self.ep_coverage else 0.0,
            'mean_victims':   float(np.mean(self.ep_victims))  if self.ep_victims  else 0.0,
        }

    # ════════════════════════════════════════════════════════════
    # PPO UPDATE — Không thay đổi logic, chỉ dùng self.actor_opt
    # ════════════════════════════════════════════════════════════

    def _update(self) -> Dict[str, float]:
        """PPO update — pure optimization."""
        actor_losses, critic_losses, entropies, clip_fracs = [], [], [], []

        for _ in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                obs        = torch.FloatTensor(batch['obs']).to(self.device)
                g_obs      = torch.FloatTensor(batch['global_obs']).to(self.device)
                actions    = torch.FloatTensor(batch['actions']).to(self.device)
                old_lp     = torch.FloatTensor(batch['old_log_probs']).to(self.device)
                advantages = torch.FloatTensor(batch['advantages']).to(self.device)
                returns    = torch.FloatTensor(batch['returns']).to(self.device)

                # ── Actor ─────────────────────────────────────
                log_probs, entropy = self.actor.evaluate_actions(obs, actions)
                ratio  = torch.exp(log_probs - old_lp)
                surr1  = ratio * advantages
                surr2  = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                ) * advantages
                a_loss = (
                    -torch.min(surr1, surr2).mean()
                    - self.entropy_coeff * entropy.mean()
                )

                self.actor_opt.zero_grad()
                a_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()

                # ── Critic ────────────────────────────────────
                values = self.critic.get_value(g_obs)
                c_loss = nn.functional.mse_loss(values, returns)

                self.critic_opt.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()

                # Metrics
                actor_losses.append(a_loss.item())
                critic_losses.append(c_loss.item())
                entropies.append(entropy.mean().item())
                with torch.no_grad():
                    clip_fracs.append(
                        ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    )

        self.buffer.clear()

        return {
            'actor_loss':    float(np.mean(actor_losses)),
            'critic_loss':   float(np.mean(critic_losses)),
            'entropy':       float(np.mean(entropies)),
            'clip_fraction': float(np.mean(clip_fracs)),
        }

    # ════════════════════════════════════════════════════════════
    # CHECKPOINT
    # ════════════════════════════════════════════════════════════

    def _save_checkpoint(
        self,
        curriculum_manager: Optional[CurriculumManager] = None,
        tag: Optional[str] = None,
    ) -> Path:
        name = (
            f"checkpoint_{tag}.pt" if tag
            else f"checkpoint_upd{self.update_count:05d}.pt"
        )
        path = self.checkpoint_dir / name
        ckpt = {
            'update':                self.update_count,
            'total_episodes_done':   self.total_episodes_done,
            'total_steps_collected': self.total_steps_collected,
            'actor_state_dict':      self.actor.state_dict(),
            'critic_state_dict':     self.critic.state_dict(),
            'actor_opt_state_dict':  self.actor_opt.state_dict(),
            'critic_opt_state_dict': self.critic_opt.state_dict(),
            'ep_rewards':            list(self.ep_rewards),
            'ep_coverage':           list(self.ep_coverage),
            'ep_victims':            list(self.ep_victims),
        }
        if curriculum_manager:
            ckpt['curriculum_stage'] = curriculum_manager.stage_idx
        torch.save(ckpt, path)
        return path

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor_state_dict'])
        self.critic.load_state_dict(ckpt['critic_state_dict'])
        self.actor_opt.load_state_dict(ckpt['actor_opt_state_dict'])
        self.critic_opt.load_state_dict(ckpt['critic_opt_state_dict'])
        self.total_episodes_done   = ckpt.get('total_episodes_done', 0)
        self.total_steps_collected = ckpt.get('total_steps_collected', 0)
        self.update_count          = ckpt.get('update', 0)
        print(f"✅ Loaded checkpoint: update={self.update_count}", flush=True)
        return self.update_count

    # ════════════════════════════════════════════════════════════
    # VIZ
    # ════════════════════════════════════════════════════════════

    def _save_viz(self, env: _EnvWrapper, update: int):
        frame = env.render()
        if frame is None:
            return
        try:
            import matplotlib
            matplotlib.use('Agg')   # Non-interactive backend
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(frame)
            ax.axis('off')
            ax.set_title(f"Update {update} | Ep {self.total_episodes_done}")
            path = self.viz_dir / f"update_{update:05d}.png"
            plt.savefig(path, bbox_inches='tight', dpi=100)
            plt.close(fig)
        except Exception as e:
            print(f"  ⚠️  Viz failed: {e}", flush=True)

    # ════════════════════════════════════════════════════════════
    # LOGGING HELPERS
    # ════════════════════════════════════════════════════════════

    def _log_episode(
        self, pbar,
        ep_num: int, reward: float, cov: float,
        vic_found: int, vic_total: int, steps: int,
    ):
        icon = "🟢" if reward > 200 else "🟡" if reward > 0 else "🔴"
        vic_rate = vic_found / max(vic_total, 1) * 100
        msg = (
            f"{icon} Ep {ep_num:>5d} | "
            f"Rew: {reward:+7.1f} | "
            f"Cov: {cov:5.1f}% | "
            f"Vic: {vic_found}/{vic_total} ({vic_rate:.0f}%) | "
            f"Steps: {steps}"
        )
        pbar.write(msg)

    def _log_update(self, pbar, train_m, rollout_m, start_time):
        elapsed = time.time() - start_time
        fps     = self.total_steps_collected / max(elapsed, 1e-6)
        lines   = [
            f"\n{'─'*60}",
            f"📊 Update {self.update_count} | "
            f"Ep {self.total_episodes_done}",
            f"{'─'*60}",
            f"  Task  | Rew: {rollout_m['mean_ep_reward']:+8.2f} | "
            f"Cov: {rollout_m['mean_coverage']:5.1f}% | "
            f"Vic: {rollout_m['mean_victims']:5.1f}%",
            f"  Train | Actor: {train_m['actor_loss']:7.4f} | "
            f"Critic: {train_m['critic_loss']:7.4f} | "
            f"Entropy: {train_m['entropy']:6.4f} | "
            f"Clip: {train_m['clip_fraction']:.3f}",
            f"  Perf  | FPS: {fps:6.1f} | "
            f"Steps: {self.total_steps_collected:,} | "
            f"Time: {elapsed/60:.1f}min",
            f"{'─'*60}",
        ]
        for line in lines:
            pbar.write(line)

    def _print_init(self):
        print(f"\n{'='*60}")
        print(f"🚁 MAPPO Trainer v3")
        print(f"{'='*60}")
        print(f"  Device:        {self.device}")
        print(f"  Run name:      {self.run_name}")
        print(f"  n_envs:        {self.n_envs}")
        print(f"  Actor params:  {sum(p.numel() for p in self.actor.parameters()):,}")
        print(f"  Critic params: {sum(p.numel() for p in self.critic.parameters()):,}")
        print(f"  Buffer cap:    {self.rollout_length * self.n_envs:,}")
        print(f"  Output:        {self.output_dir}")
        print(f"{'='*60}\n", flush=True)

    def _print_train_header(self, total_eps, curriculum_manager):
        print(f"\n{'='*60}")
        print(f"🚀 MAPPO Training Start")
        print(f"{'='*60}")
        print(f"  Target:     {total_eps} episodes")
        print(f"  Rollout:    {self.rollout_length} × {self.n_envs} envs")
        print(f"  Batch:      {self.batch_size}")
        print(f"  Curriculum: {'ON' if curriculum_manager else 'OFF (HARD only)'}")
        print(f"{'='*60}\n", flush=True)

    def _print_summary(self, elapsed, rollout_m):
        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"  Episodes: {self.total_episodes_done}")
        print(f"  Updates:  {self.update_count}")
        print(f"  Steps:    {self.total_steps_collected:,}")
        print(f"  Time:     {elapsed/60:.1f} min")
        print(f"  FPS:      {self.total_steps_collected/max(elapsed,1):.1f}")
        if rollout_m:
            print(f"  Reward:   {rollout_m['mean_ep_reward']:.2f}")
            print(f"  Coverage: {rollout_m['mean_coverage']:.2f}%")
            print(f"  Victims:  {rollout_m['mean_victims']:.2f}%")
        print(f"{'='*60}\n", flush=True)