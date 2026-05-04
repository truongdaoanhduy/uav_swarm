"""
training/algorithms/mappo/trainer.py
MAPPO Trainer với tqdm + episode tracking + viz support
"""

import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from collections import deque
# Đầu file, thay import tqdm

# ✅ Auto-detect tqdm (terminal vs notebook)
def _get_tqdm():
    """Return đúng tqdm class tùy theo môi trường."""
    try:
        shell = get_ipython().__class__.__name__
        if shell in ('ZMQInteractiveShell', 'Shell'):
            try:
                import ipywidgets  # noqa
                from tqdm.notebook import tqdm as _tqdm
                return _tqdm
            except ImportError:
                # ipywidgets chưa install → dùng tqdm thường
                from tqdm import tqdm as _tqdm
                return _tqdm
        else:
            from tqdm import tqdm as _tqdm
            return _tqdm
    except NameError:
        # Đang chạy từ terminal (python script)
        from tqdm import tqdm as _tqdm
        return _tqdm


tqdm = _get_tqdm()

from config import AppConfig
from env_setup.sar_pettingzoo_env import SARPettingZooEnv
from env_setup.vec_env import VectorizedEnv
from training.algorithms.mappo.actor import ActorNetwork
from training.algorithms.mappo.critic import CriticNetwork
from training.algorithms.mappo.buffer import RolloutBuffer


class _EnvWrapper:
    """Unified env interface (single hoặc vectorized)."""
    
    def __init__(self, config: AppConfig, n_envs: int, seed: int):
        self.n_envs   = n_envs
        self.n_agents = config.env.n_uav
        self.obs_dim  = config.obs.actor_dim
        self._config  = config
        self._seed    = seed
        
        if n_envs == 1:
            self._env = SARPettingZooEnv(config, render_mode=None)
            self._env.reset(seed=seed)
            self._is_vec = False
        else:
            self._env = VectorizedEnv(config, n_envs=n_envs, start_seed=seed)
            self._is_vec = True
    
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns: obs_batch [n_envs, n_agents, obs_dim], global_batch [n_envs, global_obs_dim]"""
        if self._is_vec:
            return self._env.reset()
        
        obs_d, info = self._env.reset(seed=self._seed)
        obs = np.array([obs_d[f"uav_{i}"] for i in range(self.n_agents)], dtype=np.float32)[None]
        g   = info["uav_0"]["global_obs"][None]
        return obs, g
    
    def step(self, actions_batch: np.ndarray):
        """Args: actions_batch [n_envs, n_agents, 3]"""
        if self._is_vec:
            return self._env.step(actions_batch)
        
        act_dict = {f"uav_{i}": actions_batch[0][i] for i in range(self.n_agents)}
        obs_d, rew_d, term_d, trunc_d, info = self._env.step(act_dict)
        
        obs  = np.array([obs_d.get(f"uav_{i}", np.zeros(self.obs_dim, np.float32)) for i in range(self.n_agents)], dtype=np.float32)[None]
        g    = info["uav_0"]["global_obs"][None]
        rews = np.array([rew_d.get(f"uav_{i}", 0.0) for i in range(self.n_agents)], dtype=np.float32)[None]
        done = any(term_d.values()) or any(trunc_d.values())
        return obs, g, rews, [done], [info]
    
    def render(self):
        """Chỉ single env support."""
        if not self._is_vec and hasattr(self._env, "render"):
            return self._env.render()
        return None
    
    def close(self):
        try:
            self._env.close()
        except:
            pass


class MAPPOTrainer:
    """MAPPO Trainer với tqdm progress bar theo episodes."""
    
    def __init__(self, config: AppConfig, device: str = "auto", run_name: str = None, n_envs: int = 1):
        self.config = config
        self.n_envs = n_envs
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
        
        # Networks
        self.actor = ActorNetwork(
            obs_dim=self.obs_dim, action_dim=3,
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
        
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=tr.mappo_lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=tr.mappo_lr_critic)
        
        # Buffer
        buffer_capacity = self.rollout_length * n_envs
        self.buffer = RolloutBuffer(
            rollout_length=buffer_capacity,
            n_agents=self.n_agents,
            obs_dim=self.obs_dim,
            global_obs_dim=self.global_obs_dim,
            action_dim=3,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        
        # Stats
        self.ep_rewards  = deque(maxlen=100)
        self.ep_lengths  = deque(maxlen=100)
        self.ep_coverage = deque(maxlen=100)
        self.ep_victims  = deque(maxlen=100)
        
        self.total_episodes_done = 0
        self.total_steps = 0
        self.update_count = 0
        import os
        # Dirs
        is_kaggle = os.path.exists("/kaggle/working")
    
        if is_kaggle:
            # Kaggle: lưu vào /kaggle/working (có thể download)
            base_dir = Path("/kaggle/working/results")
        else:
            # Local: lưu vào ./results
            base_dir = Path("results")
        self.output_dir = base_dir / "mappo" / self.run_name
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.viz_dir = self.output_dir / "viz"
        
        for d in [self.checkpoint_dir, self.viz_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self._print_init()
    
    # ══════════════════════════════════════════════════════════════════
    # TRAIN
    # ══════════════════════════════════════════════════════════════════
    
    def train(
        self,
        total_episodes: int,
        curriculum_manager = None,
        seed: int = 42,
        log_every_n_eps: int = 10,
        viz_every_n_eps: int = 50,
        checkpoint_every_n_eps: int = 100,
    ):
        """Main training loop với tqdm."""
        start_time = time.time()
        env = _EnvWrapper(self.config, self.n_envs, seed)
        
        pbar = tqdm(
            total=total_episodes,
            desc="🚁 Training",
            unit="ep",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} ep [{elapsed}<{remaining}] {postfix}",
        )
        
        print(f"\n🚀 MAPPO Training")
        print(f"  target episodes : {total_episodes}")
        print(f"  n_envs          : {self.n_envs}")
        print(f"  max_steps/ep    : {self.config.env.max_steps}")
        print(f"  log every       : {log_every_n_eps} eps")
        print(f"  viz every       : {viz_every_n_eps} eps")
        print(f"  checkpoint every: {checkpoint_every_n_eps} eps\n")
        
        last_rollout: Dict = {}
        last_train: Dict = {}
        
        while self.total_episodes_done < total_episodes:
            # Rollout + Update
            last_rollout = self._rollout(env, pbar, total_episodes)
            last_train   = self._update()
            self.update_count += 1
            
            # Detailed log
            if self.total_episodes_done % log_every_n_eps == 0:
                elapsed = time.time() - start_time
                fps = self.total_steps / max(elapsed, 1e-6)
                self._log_detail(pbar, last_rollout, last_train, elapsed, fps, curriculum_manager)
            
            # Viz
            if viz_every_n_eps > 0 and self.total_episodes_done % viz_every_n_eps == 0:
                self._save_viz(env, self.total_episodes_done)
            
            # Checkpoint
            if self.total_episodes_done % checkpoint_every_n_eps == 0:
                self.save_checkpoint(self.total_episodes_done, curriculum_manager)
            
            # Curriculum
            if curriculum_manager and self.ep_rewards:
                curriculum_manager.update(
                    coverage=last_rollout.get("mean_coverage", 0) / 100,
                    victims_rate=last_rollout.get("mean_victims", 0) / 100,
                    reward=last_rollout.get("mean_ep_reward", 0),
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
        
        self.save_checkpoint(self.total_episodes_done, curriculum_manager, tag="final")
        self._print_final(time.time() - start_time, last_rollout)
        import os
        if os.path.exists("/kaggle/working"):
            print(f"\n📥 KAGGLE DOWNLOAD:")
            print(f"   Files saved to: /kaggle/working/results/")
            print(f"   Right sidebar → Output → Download all files")
            print(f"")
            print(f"   Or download individual files:")
            print(f"   - Final checkpoint: {self.checkpoint_dir}/checkpoint_final.pt")
            print(f"   - Training plot:    {self.output_dir}/training_curves.png")
    # ══════════════════════════════════════════════════════════════════
    # ROLLOUT
    # ══════════════════════════════════════════════════════════════════
    
    def _rollout(self, env: _EnvWrapper, pbar: tqdm, max_episodes: int) -> Dict:
        obs_batch, g_batch = env.reset()
        ep_rew = np.zeros(self.n_envs, dtype=np.float32)
        ep_len = np.zeros(self.n_envs, dtype=np.int32)
        last_g = g_batch.copy()
        last_dones = [False] * self.n_envs
        
        for _ in range(self.rollout_length):
            if self.total_episodes_done >= max_episodes:
                break
            
            # Inference
            n = self.n_envs
            obs_flat = obs_batch.reshape(n * self.n_agents, self.obs_dim)
            obs_t = torch.FloatTensor(obs_flat).to(self.device)
            g_t   = torch.FloatTensor(g_batch).to(self.device)
            
            with torch.no_grad():
                act_t, lp_t = self.actor.get_action(obs_t)
                val_t = self.critic.get_value(g_t)
            
            act_batch = act_t.cpu().numpy().reshape(n, self.n_agents, 3)
            lp_batch  = lp_t.cpu().numpy().reshape(n, self.n_agents)
            val_batch = np.repeat(val_t.cpu().numpy()[:, None], self.n_agents, axis=1)
            
            # Step
            next_obs, next_g, rews, dones, infos = env.step(act_batch)
            
            # Store
            for ei in range(n):
                self.buffer.add(
                    obs=obs_batch[ei], global_obs=g_batch[ei], actions=act_batch[ei],
                    rewards=rews[ei], values=val_batch[ei], log_probs=lp_batch[ei], done=dones[ei]
                )
                self.total_steps += 1
                ep_rew[ei] += float(rews[ei][0])
                ep_len[ei] += 1
                
                if dones[ei]:
                    info_ei = infos[ei] if infos[ei] else {}
                    u0 = info_ei.get("uav_0", {})
                    cov = float(u0.get("coverage_rate", 0.0)) * 100
                    vf  = int(u0.get("victims_found", 0))
                    vt  = max(int(u0.get("victims_total", 1)), 1)
                    
                    # Done reason & success
                    done_reason = u0.get("done_reason", "unknown")
                    success = u0.get("success", False)
                    
                    # Battery stats (nếu có)
                    battery_mean = 0.0
                    battery_min = 0.0
                    if "battery_stats" in u0:
                        bstats = u0["battery_stats"]
                        battery_mean = bstats.get("mean", 0.0)
                        battery_min = bstats.get("min", 0.0)
                    
                    self.ep_rewards.append(float(ep_rew[ei]))
                    self.ep_lengths.append(int(ep_len[ei]))
                    self.ep_coverage.append(cov)
                    self.ep_victims.append(vf / vt * 100)
                    self.total_episodes_done += 1
                    
                    
                    # Update tqdm
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
                        "rew":    f"{float(ep_rew[ei]):+.0f}",
                        "cov":    f"{cov:.0f}%",
                        "vic":    f"{vf}/{vt}",
                        "step":   f"{ep_len[ei]}/{self.config.env.max_steps}",
                        "bat":    f"{battery_mean:.0f}%",
                        "end":    f"{status}{reason_label}",
                    })
                    
                    # Log every 10 episodes
                    if self.total_episodes_done % 10 == 0:
                        pbar.write(
                            f"[EP {self.total_episodes_done:4d}] "
                            f"{status} {done_reason:8s} | "
                            f"step={ep_len[ei]:4d}/{self.config.env.max_steps} | "
                            f"bat={battery_mean:5.1f}% (min={battery_min:5.1f}%) | "
                            f"rew={ep_rew[ei]:+7.1f} | "
                            f"cov={cov:5.1f}% | vic={vf:2d}/{vt:2d}"
                        )
                    
                    ep_rew[ei] = 0.0
                    ep_len[ei] = 0
                    
                    if self.total_episodes_done >= max_episodes:
                        break
            
            obs_batch = next_obs
            g_batch   = next_g
            last_g    = next_g.copy()
            last_dones = dones
        
        # GAE
        with torch.no_grad():
            last_vals = self.critic.get_value(torch.FloatTensor(last_g).to(self.device)).cpu().numpy()
        
        bootstrap = np.full(self.n_agents, float(np.mean(last_vals)), dtype=np.float32)
        self.buffer.compute_gae(bootstrap, last_done=all(last_dones))
        
        return {
            "mean_ep_reward": float(np.mean(self.ep_rewards)) if self.ep_rewards else 0.0,
            "mean_ep_length": float(np.mean(self.ep_lengths)) if self.ep_lengths else 0.0,
            "mean_coverage":  float(np.mean(self.ep_coverage)) if self.ep_coverage else 0.0,
            "mean_victims":   float(np.mean(self.ep_victims))  if self.ep_victims else 0.0,
        }
    
    # ══════════════════════════════════════════════════════════════════
    # UPDATE
    # ══════════════════════════════════════════════════════════════════
    
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
                
                # Actor
                lp, entropy = self.actor.evaluate_actions(obs, actions)
                ratio = torch.exp(lp - old_lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv
                a_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy.mean()
                
                self.actor_opt.zero_grad()
                a_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_opt.step()
                
                # Critic
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
                    clip_fracs.append(((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item())
        
        self.buffer.clear()
        
        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
            "clip_fraction": float(np.mean(clip_fracs)),
        }
    
        # Thêm vào cuối class MAPPOTrainer, trước _print_init():

    def plot_training_curves(self, save_path: str = None):
        """
        Vẽ 4 đồ thị training curves.
        
        Args:
            save_path: Path để lưu figure (nếu None → show interactive)
        """
        try:
            import matplotlib
            if save_path:
                matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            if not self.ep_rewards:
                print("⚠️  No training data to plot")
                return
            
            # Convert deque → list
            episodes = list(range(1, len(self.ep_rewards) + 1))
            rewards  = list(self.ep_rewards)
            coverage = list(self.ep_coverage)
            victims  = list(self.ep_victims)
            lengths  = list(self.ep_lengths)
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"MAPPO Training Curves — {self.run_name}", fontsize=16, fontweight="bold")
            
            # ── 1. Episode Reward ─────────────────────────────────────
            ax = axes[0, 0]
            ax.plot(episodes, rewards, alpha=0.3, color="steelblue", linewidth=0.5)
            
            # Smooth với rolling mean
            window = min(20, len(rewards) // 5)
            if len(rewards) >= window:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                smooth_x = list(range(window, len(rewards) + 1))
                ax.plot(smooth_x, smoothed, color="darkblue", linewidth=2, label=f"MA({window})")
            
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_xlabel("Episode", fontsize=11)
            ax.set_ylabel("Reward", fontsize=11)
            ax.set_title("Episode Reward", fontsize=12, fontweight="bold")
            ax.grid(alpha=0.3)
            ax.legend()
            
            # ── 2. Coverage Rate ──────────────────────────────────────
            ax = axes[0, 1]
            ax.plot(episodes, coverage, alpha=0.3, color="green", linewidth=0.5)
            
            if len(coverage) >= window:
                smoothed = np.convolve(coverage, np.ones(window)/window, mode='valid')
                smooth_x = list(range(window, len(coverage) + 1))
                ax.plot(smooth_x, smoothed, color="darkgreen", linewidth=2, label=f"MA({window})")
            
            ax.axhline(70, color='orange', linestyle='--', linewidth=0.8, alpha=0.5, label="Target 70%")
            ax.set_xlabel("Episode", fontsize=11)
            ax.set_ylabel("Coverage (%)", fontsize=11)
            ax.set_title("Coverage Rate", fontsize=12, fontweight="bold")
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.3)
            ax.legend()
            
            # ── 3. Victims Found Rate ─────────────────────────────────
            ax = axes[1, 0]
            ax.plot(episodes, victims, alpha=0.3, color="purple", linewidth=0.5)
            
            if len(victims) >= window:
                smoothed = np.convolve(victims, np.ones(window)/window, mode='valid')
                smooth_x = list(range(window, len(victims) + 1))
                ax.plot(smooth_x, smoothed, color="indigo", linewidth=2, label=f"MA({window})")
            
            ax.axhline(80, color='orange', linestyle='--', linewidth=0.8, alpha=0.5, label="Target 80%")
            ax.set_xlabel("Episode", fontsize=11)
            ax.set_ylabel("Victims Found (%)", fontsize=11)
            ax.set_title("Victims Found Rate", fontsize=12, fontweight="bold")
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.3)
            ax.legend()
            
            # ── 4. Episode Length ─────────────────────────────────────
            ax = axes[1, 1]
            ax.plot(episodes, lengths, alpha=0.3, color="coral", linewidth=0.5)
            
            if len(lengths) >= window:
                smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
                smooth_x = list(range(window, len(lengths) + 1))
                ax.plot(smooth_x, smoothed, color="red", linewidth=2, label=f"MA({window})")
            
            ax.axhline(self.config.env.max_steps, color='gray', linestyle='--', 
                    linewidth=0.8, alpha=0.5, label=f"Max ({self.config.env.max_steps})")
            ax.set_xlabel("Episode", fontsize=11)
            ax.set_ylabel("Steps", fontsize=11)
            ax.set_title("Episode Length", fontsize=12, fontweight="bold")
            ax.grid(alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            # Save hoặc show
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"📊 Training curves saved: {save_path}")
            else:
                plt.show()
        
        except Exception as e:
            print(f"⚠️  Failed to plot training curves: {e}")

    # ══════════════════════════════════════════════════════════════════
    # LOGGING & VIZ
    # ══════════════════════════════════════════════════════════════════
    
    def _log_detail(self, pbar, rollout, train, elapsed, fps, curriculum_manager):
        lines = [
            f"\n{'─'*60}",
            f"📊 Episode {self.total_episodes_done} | Update {self.update_count}",
            f"{'─'*60}",
            f"  Task:",
            f"    reward   : {rollout['mean_ep_reward']:+8.2f}",
            f"    coverage : {rollout['mean_coverage']:7.2f}%",
            f"    victims  : {rollout['mean_victims']:7.2f}%",
            f"    ep_len   : {rollout['mean_ep_length']:7.1f}",
            f"  Train:",
            f"    a_loss   : {train['actor_loss']:.4f}",
            f"    c_loss   : {train['critic_loss']:.4f}",
            f"    entropy  : {train['entropy']:.4f}",
            f"    clip_frac: {train['clip_fraction']:.3f}",
            f"  Perf:",
            f"    fps      : {fps:.1f}",
            f"    elapsed  : {elapsed/60:.1f} min",
        ]
        
        if curriculum_manager:
            stage = curriculum_manager.current_stage
            lines += [
                f"  Curriculum:",
                f"    stage    : {stage.name.upper()}",
                f"    map_size : {stage.map_size}m",
            ]
        
        lines.append(f"{'─'*60}\n")
        for line in lines:
            pbar.write(line)
    
    def _save_viz(self, env: _EnvWrapper, episode: int):
        """Lưu 2D viz (chỉ single env)."""
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
            ax.set_title(f"Episode {episode}", fontsize=14)
            
            path = self.viz_dir / f"ep_{episode:06d}.png"
            fig.savefig(path, bbox_inches="tight", dpi=100)
            plt.close(fig)
        except:
            pass
    
    # ══════════════════════════════════════════════════════════════════
    # CHECKPOINT
    # ══════════════════════════════════════════════════════════════════
    
    def save_checkpoint(self, episode: int, curriculum_manager=None, tag: str = None):
        name = f"checkpoint_{tag}.pt" if tag else f"checkpoint_ep{episode:06d}.pt"
        path = self.checkpoint_dir / name
        torch.save({
            "episode": episode,
            "update": self.update_count,
            "total_episodes_done": self.total_episodes_done,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_opt.state_dict(),
            "critic_optimizer_state_dict": self.critic_opt.state_dict(),
            "ep_rewards": list(self.ep_rewards),
            "ep_coverage": list(self.ep_coverage),
            "ep_victims": list(self.ep_victims),
            "curriculum_stage": curriculum_manager.stage_idx if curriculum_manager else 0,
        }, path)
    
    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.actor_opt.load_state_dict(ckpt["actor_optimizer_state_dict"])
        self.critic_opt.load_state_dict(ckpt["critic_optimizer_state_dict"])
        self.total_episodes_done = ckpt.get("total_episodes_done", 0)
        self.update_count = ckpt.get("update", 0)
        return ckpt.get("episode", 0)
    
    # ══════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════
    
    def _print_init(self):
        print(f"\n{'='*60}")
        print(f"🚁 MAPPO Trainer")
        print(f"{'='*60}")
        print(f"  device   : {self.device}")
        print(f"  run_name : {self.run_name}")
        print(f"  n_envs   : {self.n_envs}")
        print(f"  n_agents : {self.n_agents}")
        print(f"  actor    : {sum(p.numel() for p in self.actor.parameters()):,} params")
        print(f"  critic   : {sum(p.numel() for p in self.critic.parameters()):,} params")
        print(f"  buffer   : {self.rollout_length * self.n_envs:,} steps")
        print(f"  output   : {self.output_dir}")
        print(f"{'='*60}\n")
    
    def _print_final(self, elapsed: float, metrics: Dict):
        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"{'='*60}")
        print(f"  episodes : {self.total_episodes_done:,}")
        print(f"  updates  : {self.update_count:,}")
        print(f"  steps    : {self.total_steps:,}")
        print(f"  time     : {elapsed/60:.1f} min")
        
        if metrics:
            print(f"\n📊 Final Performance (last 100 episodes):")
            print(f"  reward   : {metrics.get('mean_ep_reward', 0):+.2f}")
            print(f"  coverage : {metrics.get('mean_coverage', 0):.2f}%")
            print(f"  victims  : {metrics.get('mean_victims', 0):.2f}%")
            print(f"  ep_len   : {metrics.get('mean_ep_length', 0):.1f}")
        
        print(f"\n💾 Saved:")
        print(f"  checkpoint: {self.checkpoint_dir}/checkpoint_final.pt")
        
        # Plot training curves
        plot_path = self.output_dir / "training_curves.png"
        self.plot_training_curves(save_path=str(plot_path))
        print(f"  plot:       {plot_path}")
        
        print(f"{'='*60}\n")