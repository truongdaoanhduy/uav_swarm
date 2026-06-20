"""
MASAC Trainer — Kaggle optimized (no viz/plot during training)
Chỉ lưu checkpoint + metrics, upload HF.
Hyperparams đọc từ config.train (TrainConfig).
"""

import os
import time
from pathlib import Path
from typing import Dict, Tuple
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
from .actor import SACActorNetwork
from .twin_critic import TwinCriticNetwork
from .replay_buffer import ReplayBuffer


# ══════════════════════════════════════════════════════════════════════════════
# ENV WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class _EnvWrapper:
    """Unified env interface — same as MAPPO."""

    def __init__(self, config: AppConfig, n_envs: int, seed: int,llm_reward_path: str = None,   # ← THÊM
):
        self.n_envs   = n_envs
        self.n_agents = config.env.n_uav
        self.obs_dim  = config.obs.actor_dim
        self._config  = config
        self._seed    = seed
        self._single_episode_count = 0

        # ✅ Inject LLM reward cho single env
        if n_envs == 1:
            self._env    = SARPettingZooEnv(config, render_mode=None)
            self._is_vec = False

            # ✅ Inject LLM reward cho single env
            if llm_reward_path is not None:
                try:
                    import sys, os
                    sys.path.insert(0, os.getcwd())
                    from rewards.llm_reward import load_llm_reward
                    llm_rw = load_llm_reward(llm_reward_path, config)
                    self._env._base_env.baseline_reward = llm_rw
                    print(f"\n✅ LLM reward injected (MASAC single env)\n")
                except Exception as e:
                    print(f"\n⚠️  LLM inject failed: {e}\n")
        else:
            self._env = VectorizedEnv(
                config,
                n_envs          = n_envs,
                start_seed      = seed,
                llm_reward_path = llm_reward_path,   # ← TRUYỀN
            )
            self._is_vec = True


        self._current_obs:    np.ndarray | None = None
        self._current_global: np.ndarray | None = None
        self._needs_reset:    bool = True

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._is_vec:
            obs, g = self._env.reset()
        else:
            obs_d, info = self._env.reset(seed=self._episode_seed())
            self._single_episode_count += 1
            obs = np.array(
                [obs_d[f"uav_{i}"] for i in range(self.n_agents)],
                dtype=np.float32,
            )[None]
            g = info["uav_0"]["global_obs"][None]

        self._current_obs    = obs
        self._current_global = g
        self._needs_reset    = False
        return obs, g

    def get_current_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._needs_reset or self._current_obs is None:
            return self.reset()
        return self._current_obs, self._current_global

    def step(self, actions_batch: np.ndarray):
        """actions_batch: [n_envs, n_agents, 4]"""
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

        obs_next = np.array(
            [obs_d.get(f"uav_{i}", np.zeros(self.obs_dim, np.float32))
             for i in range(self.n_agents)],
            dtype=np.float32,
        )[None]
        g_next = info["uav_0"]["global_obs"][None]
        rews   = np.array(
            [rew_d.get(f"uav_{i}", 0.0) for i in range(self.n_agents)],
            dtype=np.float32,
        )[None]

        if done:
            new_seed    = self._episode_seed()
            self._single_episode_count += 1
            new_obs_d, new_info = self._env.reset(seed=new_seed)
            new_obs = np.array(
                [new_obs_d.get(f"uav_{i}", np.zeros(self.obs_dim, np.float32))
                 for i in range(self.n_agents)],
                dtype=np.float32,
            )[None]
            new_g                = new_info["uav_0"]["global_obs"][None]
            self._current_obs    = new_obs
            self._current_global = new_g
        else:
            self._current_obs    = obs_next
            self._current_global = g_next

        return obs_next, g_next, rews, [done], [info]

    def _episode_seed(self) -> int:
        return (self._seed + self._single_episode_count * 10_000) % (2**31)

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# MASAC TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class MASACTrainer:
    """Multi-Agent Soft Actor-Critic Trainer.
    Kaggle optimized — no viz/plot during training.
    Hyperparams đọc từ config.train.masac_*.
    """

    def __init__(
        self,
        config:          AppConfig,
        device:          str = "auto",
        run_name:        str = None,
        n_envs:          int = 1,
        hf_token:        str = None,
        hf_repo:         str = None,
        hf_upload_every: int = 500,
        llm_reward_path: str = None,   # ← THÊM

    ):
        # ── 1. Lưu config + basic attrs TRƯỚC ────────────────────────────
        self.config   = config
        self.n_envs   = n_envs
        self.run_name = run_name or f"masac_{int(time.time())}"

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # ── 2. Dims từ config ─────────────────────────────────────────────
        self.n_agents       = config.env.n_uav
        self.obs_dim        = config.obs.actor_dim
        self.global_obs_dim = config.obs.critic_dim
        self.action_dim     = 4

        # ── 3. Hyperparams từ config.train ────────────────────────────────
        tr = config.train

        self.buffer_capacity  = tr.masac_buffer_capacity
        self.batch_size       = tr.masac_batch_size
        self.gamma            = tr.masac_gamma
        self.tau              = tr.masac_tau
        self.update_every     = tr.masac_update_every
        self.updates_per_step = tr.masac_updates_per_step
        self.warmup_steps     = tr.masac_warmup_steps

        # ── 4. Networks ───────────────────────────────────────────────────
        self.actor = SACActorNetwork(
            obs_dim     = self.obs_dim,
            action_dim  = self.action_dim,
            hidden_dims = tr.masac_actor_hidden,
        ).to(self.device)

        self.critic = TwinCriticNetwork(
            global_obs_dim = self.global_obs_dim,
            action_dim     = self.action_dim,
            hidden_dims    = tr.masac_critic_hidden,
        ).to(self.device)

        self.critic_target = TwinCriticNetwork(
            global_obs_dim = self.global_obs_dim,
            action_dim     = self.action_dim,
            hidden_dims    = tr.masac_critic_hidden,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self._llm_reward_path = llm_reward_path  

        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ── 5. Entropy temperature α ──────────────────────────────────────
        self.auto_alpha     = tr.masac_auto_alpha
        self.target_entropy = -float(self.action_dim - 1)  # = -3.0

        if self.auto_alpha:
            self.log_alpha = torch.tensor(
                np.log(tr.masac_alpha_init), dtype=torch.float32,
                requires_grad=True, device=self.device,
            )
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=tr.masac_lr_alpha)
            self.alpha     = self.log_alpha.exp().item()
        else:
            self.log_alpha = None
            self.alpha_opt = None
            self.alpha     = tr.masac_alpha_init

        # ── 6. Optimizers ─────────────────────────────────────────────────
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=tr.masac_lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=tr.masac_lr_critic)

        # ── 7. Replay Buffer ──────────────────────────────────────────────
        self.buffer = ReplayBuffer(
            capacity       = self.buffer_capacity,
            n_agents       = self.n_agents,
            obs_dim        = self.obs_dim,
            global_obs_dim = self.global_obs_dim,
            action_dim     = self.action_dim,
        )

        # ── 8. Stats ──────────────────────────────────────────────────────
        self.ep_rewards  = deque(maxlen=100)
        self.ep_lengths  = deque(maxlen=100)
        self.ep_coverage = deque(maxlen=100)
        self.ep_victims  = deque(maxlen=100)

        # Full history để upload HF
        self._all_rewards  = []
        self._all_coverage = []
        self._all_victims  = []
        self._all_lengths  = []

        self._persist_ep_rew = np.zeros(n_envs, dtype=np.float32)
        self._persist_ep_len = np.zeros(n_envs, dtype=np.int32)

        self.total_episodes_done = 0
        self.total_steps         = 0
        self.update_count        = 0
        self._np_rng             = np.random.default_rng(0)

        self._next_log_ep        = 0
        self._next_checkpoint_ep = 0

        # ── 9. HF Uploader ────────────────────────────────────────────────
        self.hf_uploader     = None
        self.hf_upload_every = hf_upload_every
        self._hf_run_name    = self.run_name

        if hf_token and hf_repo:
            self.hf_uploader = HFUploader(token=hf_token, repo_id=hf_repo)

        # ── 10. Dirs ──────────────────────────────────────────────────────
        is_kaggle = os.path.exists("/kaggle/working")
        base_dir  = Path("/kaggle/working/results") if is_kaggle else Path("results")

        self.output_dir     = base_dir / "masac" / self.run_name
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
        log_every_n_eps:        int       = 50,
        checkpoint_every_n_eps: int       = 100,
    ):
        start_time = time.time()
        self._np_rng = np.random.default_rng(seed + 17)
        self.buffer.set_seed(seed + 31)
        env = _EnvWrapper(
                self.config,
                self.n_envs,
                seed,
                llm_reward_path=self._llm_reward_path   # ← THÊM
            )
        self._next_log_ep        = log_every_n_eps
        self._next_checkpoint_ep = checkpoint_every_n_eps

        print(f"\n🚀 MASAC Training (Kaggle mode — no viz)")
        print(f"  target episodes : {total_episodes:,}")
        print(f"  n_envs          : {self.n_envs}")
        print(f"  buffer capacity : {self.buffer_capacity:,}")
        print(f"  batch_size      : {self.batch_size}")
        print(f"  warmup_steps    : {self.warmup_steps}")
        print(f"  auto_alpha      : {self.auto_alpha}")
        print(f"  target_entropy  : {self.target_entropy:.2f}")
        print(f"  hf_upload       : {self.hf_uploader is not None}")
        print(f"  output dir      : {self.output_dir.resolve()}\n")

        pbar = tqdm(
            total         = total_episodes,
            desc          = "🔥 MASAC",
            unit          = "ep",
            dynamic_ncols = True,
        )

        env.reset()
        obs_batch, g_batch = env.get_current_obs()
        last_metrics = {}

        while self.total_episodes_done < total_episodes:

            n        = self.n_envs
            obs_flat = obs_batch.reshape(n * self.n_agents, self.obs_dim)
            obs_t    = torch.FloatTensor(obs_flat).to(self.device)

            with torch.no_grad():
                if self.total_steps < self.warmup_steps:
                    act_np       = self._np_rng.uniform(-1, 1, (n * self.n_agents, self.action_dim))
                    act_np[:, 3] = (act_np[:, 3] > 0.5).astype(np.float32)
                else:
                    act_t, _ = self.actor.get_action(obs_t, deterministic=False)
                    act_np   = act_t.cpu().numpy()

            act_batch = act_np.reshape(n, self.n_agents, self.action_dim)
            act_batch = np.clip(act_batch, -1.0, 1.0)

            next_obs, next_g, rews, dones, infos = env.step(act_batch)

            self.buffer.add_batch(
                obs         = obs_batch,
                global_obs  = g_batch,
                actions     = act_batch,
                rewards     = rews,
                next_obs    = next_obs,
                next_global = next_g,
                dones       = np.array(dones, dtype=np.float32),
            )

            self.total_steps     += n
            self._persist_ep_rew += rews.sum(axis=1)
            self._persist_ep_len += 1

            for ei in range(n):
                if dones[ei]:
                    info_ei    = infos[ei] if infos[ei] else {}
                    u0         = info_ei.get("uav_0", {})
                    ep_metrics = u0.get("episode", {})

                    cov    = float(
                        ep_metrics.get("coverage_rate", 0.0) or
                        u0.get("coverage_rate", 0.0) * 100
                    )
                    vf     = int(ep_metrics.get("victims_found", 0) or u0.get("victims_found", 0))
                    vt     = max(int(ep_metrics.get("total_victims", 1) or u0.get("victims_total", 1)), 1)
                    ep_rew = float(self._persist_ep_rew[ei])
                    ep_len = int(self._persist_ep_len[ei])

                    # Rolling window
                    self.ep_rewards.append(ep_rew)
                    self.ep_lengths.append(ep_len)
                    self.ep_coverage.append(cov)
                    self.ep_victims.append(vf / vt * 100)

                    # Full history
                    self._all_rewards.append(ep_rew)
                    self._all_lengths.append(ep_len)
                    self._all_coverage.append(cov)
                    self._all_victims.append(vf / vt * 100)

                    self.total_episodes_done += 1
                    pbar.update(1)
                    pbar.set_postfix(ordered_dict={
                        "rew": f"{ep_rew:+.0f}",
                        "cov": f"{cov:.0f}%",
                        "vic": f"{vf}/{vt}",
                        "buf": f"{len(self.buffer):,}",
                        "α":   f"{self.alpha:.3f}",
                    })

                    self._persist_ep_rew[ei] = 0.0
                    self._persist_ep_len[ei] = 0

            obs_batch = env._current_obs
            g_batch   = env._current_global

            # Update
            if (self.buffer.is_ready(self.batch_size) and
                self.total_steps >= self.warmup_steps and
                self.total_steps % self.update_every == 0):
                for _ in range(self.updates_per_step):
                    last_metrics = self._update()
                    self.update_count += 1

            ep = self.total_episodes_done

            if ep >= self._next_log_ep and self.ep_rewards:
                elapsed = time.time() - start_time
                self._log_detail(pbar, last_metrics, elapsed)
                while self._next_log_ep <= ep:
                    self._next_log_ep += log_every_n_eps

            if ep >= self._next_checkpoint_ep:
                self.save_checkpoint(ep)
                while self._next_checkpoint_ep <= ep:
                    self._next_checkpoint_ep += checkpoint_every_n_eps

        pbar.close()
        env.close()
        self.save_checkpoint(self.total_episodes_done, tag="final")
        self._print_final(time.time() - start_time)

    # ══════════════════════════════════════════════════════════════════════════
    # SAC UPDATE
    # ══════════════════════════════════════════════════════════════════════════

    def _update(self) -> Dict:
        batch = self.buffer.sample(self.batch_size)

        obs      = torch.FloatTensor(batch["obs"]).to(self.device)
        g_obs    = torch.FloatTensor(batch["global_obs"]).to(self.device)
        actions  = torch.FloatTensor(batch["actions"]).to(self.device)
        rewards  = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_obs = torch.FloatTensor(batch["next_obs"]).to(self.device)
        next_g   = torch.FloatTensor(batch["next_global"]).to(self.device)
        dones    = torch.FloatTensor(batch["dones"]).to(self.device)

        B, A          = obs.shape[0], self.n_agents
        obs_flat      = obs.reshape(B * A, self.obs_dim)
        next_obs_flat = next_obs.reshape(B * A, self.obs_dim)
        team_reward   = rewards.mean(dim=-1)

        # ── 1. TD target ──────────────────────────────────────────────────
        with torch.no_grad():
            next_act_flat, next_lp = self.actor.get_action(next_obs_flat)
            next_act_mean = next_act_flat.reshape(B, A, self.action_dim).mean(dim=1)
            next_lp_mean  = next_lp.reshape(B, A).mean(dim=1)

            q1_next, q2_next = self.critic_target(next_g, next_act_mean)
            q_next = torch.min(q1_next, q2_next).squeeze(-1)

            alpha  = self.log_alpha.exp() if self.auto_alpha else self.alpha
            target = (
                team_reward
                + self.gamma * (1.0 - dones) * (q_next - alpha * next_lp_mean)
            )

        # ── 2. Update critics ─────────────────────────────────────────────
        act_mean         = actions.mean(dim=1)
        q1_loss, q2_loss = self.critic.compute_loss(g_obs, act_mean, target)
        critic_loss      = q1_loss + q2_loss

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ── 3. Update actor ───────────────────────────────────────────────
        act_flat_new, lp_new = self.actor.get_action(obs_flat)
        act_new_mean = act_flat_new.reshape(B, A, self.action_dim).mean(dim=1)
        lp_new_mean  = lp_new.reshape(B, A).mean(dim=1)
        q_new        = self.critic.min_q(g_obs, act_new_mean).squeeze(-1)

        alpha_val  = self.log_alpha.exp() if self.auto_alpha else self.alpha
        actor_loss = (alpha_val * lp_new_mean - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # ── 4. Update α ───────────────────────────────────────────────────
        alpha_loss = 0.0
        if self.auto_alpha:
            alpha_loss_t = -(
                self.log_alpha * (lp_new_mean + self.target_entropy).detach()
            ).mean()
            self.alpha_opt.zero_grad()
            alpha_loss_t.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss_t.item()

        # ── 5. Soft update target ─────────────────────────────────────────
        self._soft_update(self.critic, self.critic_target, self.tau)

        return {
            "actor_loss":  actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "q1_loss":     q1_loss.item(),
            "q2_loss":     q2_loss.item(),
            "alpha_loss":  alpha_loss,
            "alpha":       self.alpha,
            "entropy":    -lp_new_mean.mean().item(),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module, tau: float):
        with torch.no_grad():
            for sp, tp in zip(source.parameters(), target.parameters()):
                tp.data.mul_(1.0 - tau)
                tp.data.add_(tau * sp.data)

    # ══════════════════════════════════════════════════════════════════════════
    # CHECKPOINT
    # ══════════════════════════════════════════════════════════════════════════

    def save_checkpoint(self, episode: int, curriculum_manager=None, tag: str = None):
        name = f"checkpoint_{tag}.pt" if tag else f"checkpoint_ep{episode:06d}.pt"
        path = self.checkpoint_dir / name

        ckpt = {
            "episode":                     episode,
            "update":                      self.update_count,
            "total_episodes_done":         self.total_episodes_done,
            "actor_state_dict":            self.actor.state_dict(),
            "critic_state_dict":           self.critic.state_dict(),
            "critic_target_state_dict":    self.critic_target.state_dict(),
            "actor_optimizer_state_dict":  self.actor_opt.state_dict(),
            "critic_optimizer_state_dict": self.critic_opt.state_dict(),
            # Full history
            "ep_rewards":                  self._all_rewards,
            "ep_coverage":                 self._all_coverage,
            "ep_victims":                  self._all_victims,
            "ep_lengths":                  self._all_lengths,
        }
        if self.auto_alpha:
            ckpt["log_alpha"]             = self.log_alpha.item()
            ckpt["alpha_optimizer_state"] = self.alpha_opt.state_dict()

        torch.save(ckpt, path)
        print(f"\n💾 MASAC Checkpoint: {path.resolve()}")

        # ── Upload HF ─────────────────────────────────────────────────────
        if self.hf_uploader:
            metrics = {
                "algo":           "masac",
                "run_name":       self.run_name,
                "total_episodes": self.total_episodes_done,
                "total_steps":    self.total_steps,
                "mean_reward":    float(np.mean(self.ep_rewards))  if self.ep_rewards  else 0.0,
                "mean_coverage":  float(np.mean(self.ep_coverage)) if self.ep_coverage else 0.0,
                "mean_victims":   float(np.mean(self.ep_victims))  if self.ep_victims  else 0.0,
                "alpha":          self.alpha,
                "ep_rewards":     self._all_rewards,
                "ep_coverage":    self._all_coverage,
                "ep_victims":     self._all_victims,
                "ep_lengths":     self._all_lengths,
            }

            if tag == "final":
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
                f"   ep={episode:,} | upd={self.update_count:,} | "
                f"rew={np.mean(self.ep_rewards):+.1f} | "
                f"cov={np.mean(self.ep_coverage):.1f}%\n"
            )

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.critic_target.load_state_dict(ckpt["critic_target_state_dict"])
        self.actor_opt.load_state_dict(ckpt["actor_optimizer_state_dict"])
        self.critic_opt.load_state_dict(ckpt["critic_optimizer_state_dict"])
        if self.auto_alpha and "log_alpha" in ckpt:
            with torch.no_grad():
                self.log_alpha.fill_(ckpt["log_alpha"])
            self.alpha = self.log_alpha.exp().item()
            self.alpha_opt.load_state_dict(ckpt["alpha_optimizer_state"])
        self.total_episodes_done = ckpt.get("total_episodes_done", 0)
        self._all_rewards  = ckpt.get("ep_rewards",  [])
        self._all_coverage = ckpt.get("ep_coverage", [])
        self._all_victims  = ckpt.get("ep_victims",  [])
        self._all_lengths  = ckpt.get("ep_lengths",  [])
        print(f"✅ MASAC loaded: {path} | ep={self.total_episodes_done:,}")
        return ckpt.get("episode", 0)

    def _cleanup_local(self):
        if not self.hf_uploader:
            return
        import shutil
        try:
            shutil.rmtree(str(self.output_dir))
            print(f"🗑️  Local cleaned: {self.output_dir}")
        except Exception as e:
            print(f"⚠️  Cleanup failed: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # LOGGING
    # ══════════════════════════════════════════════════════════════════════════

    def _log_detail(self, pbar, metrics: Dict, elapsed: float):
        lines = [
            f"\n{'─'*65}",
            f"🔥 MASAC | ep={self.total_episodes_done:,} | upd={self.update_count:,} | {elapsed/60:.1f}min",
            f"{'─'*65}",
            f"  Task:",
            f"    reward   : {np.mean(self.ep_rewards) if self.ep_rewards else 0:+10.2f}",
            f"    coverage : {np.mean(self.ep_coverage) if self.ep_coverage else 0:8.2f}%",
            f"    victims  : {np.mean(self.ep_victims) if self.ep_victims else 0:8.2f}%",
            f"    ep_len   : {np.mean(self.ep_lengths) if self.ep_lengths else 0:8.1f} steps",
        ]
        if metrics:
            lines += [
                f"  SAC:",
                f"    a_loss   : {metrics.get('actor_loss',  0):10.4f}",
                f"    c_loss   : {metrics.get('critic_loss', 0):10.4f}",
                f"    entropy  : {metrics.get('entropy',     0):10.4f}",
                f"    alpha    : {metrics.get('alpha', self.alpha):10.4f}",
                f"  Buffer   : {len(self.buffer):>10,}",
                f"  Steps    : {self.total_steps:>10,}",
            ]
        lines.append(f"{'─'*65}\n")
        for line in lines:
            pbar.write(line)

    def _print_init(self):
        print(f"\n{'='*65}")
        print(f"🔥 MASAC Trainer — Kaggle Mode")
        print(f"{'='*65}")
        print(f"  device       : {self.device}")
        print(f"  n_agents     : {self.n_agents}")
        print(f"  obs_dim      : {self.obs_dim}")
        print(f"  global_dim   : {self.global_obs_dim}")
        print(f"  buffer       : {self.buffer_capacity:,}")
        print(f"  batch_size   : {self.batch_size}")
        print(f"  gamma/tau    : {self.gamma}/{self.tau}")
        print(f"  auto_alpha   : {self.auto_alpha}")
        print(f"  actor_hidden : {self.config.train.masac_actor_hidden}")
        print(f"  critic_hidden: {self.config.train.masac_critic_hidden}")
        print(f"  hf_upload    : {self.hf_uploader is not None}")
        print(f"  actor params : {sum(p.numel() for p in self.actor.parameters()):,}")
        print(f"  critic params: {sum(p.numel() for p in self.critic.parameters()):,}")
        print(f"{'='*65}\n")

    def _print_final(self, elapsed: float):
        print(f"\n{'='*65}")
        print(f"✅ MASAC Complete | {elapsed/60:.1f} min ({elapsed/3600:.2f}h)")
        print(f"  episodes : {self.total_episodes_done:,}")
        print(f"  updates  : {self.update_count:,}")
        print(f"  steps    : {self.total_steps:,}")
        if self.ep_rewards:
            print(f"  reward   : {np.mean(self.ep_rewards):+.2f}")
            print(f"  coverage : {np.mean(self.ep_coverage):.2f}%")
        if self.hf_uploader:
            print(f"\n🤗 https://huggingface.co/datasets/duy95/sar-uav-results/tree/main/{self._hf_run_name}")
        else:
            final_path = self.checkpoint_dir / "checkpoint_final.pt"
            print(f"\n💾 Saved locally: {final_path.resolve()}")
        print(f"{'='*65}\n")
