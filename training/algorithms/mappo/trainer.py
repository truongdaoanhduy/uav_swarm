"""
MAPPO Trainer - Main training loop với curriculum support
Fixed: Viz với VectorizedEnv, memory leak, accurate FPS
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from collections import deque

from config import AppConfig
from env_setup.sar_pettingzoo_env import SARPettingZooEnv
from training.curriculum import CurriculumManager
from training.algorithms.mappo.actor import ActorNetwork
from training.algorithms.mappo.critic import CriticNetwork
from training.algorithms.mappo.buffer import RolloutBuffer
from env_setup.vec_env import VectorizedEnv


class MAPPOTrainer:
    """
    MAPPO trainer với curriculum learning và comprehensive logging.
    
    Design:
        - Centralized training (critic sees global obs)
        - Decentralized execution (actor sees local obs)
        - Shared reward (cooperative multi-agent)
        - Curriculum progression (easy → medium → hard)
    """
    
    def __init__(
        self,
        config: AppConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        run_name: Optional[str] = None,
        n_envs: int = 1
    ):
        self.config = config
        self.device = torch.device(device)
        self.run_name = run_name or f"mappo_{int(time.time())}"
        
        # Extract hyperparams
        self.n_agents = config.env.n_uav
        self.obs_dim = config.obs.actor_dim
        self.global_obs_dim = config.obs.critic_dim
        self.action_dim = 3
        
        self.rollout_length = config.train.mappo_rollout_length
        self.n_epochs = config.train.mappo_n_epochs
        self.batch_size = config.train.mappo_batch_size
        self.clip_epsilon = config.train.mappo_clip_epsilon
        self.gamma = config.train.mappo_gamma
        self.gae_lambda = config.train.mappo_gae_lambda
        self.max_grad_norm = config.train.mappo_max_grad_norm
        self.entropy_coeff = config.train.mappo_entropy_coeff
        self.n_envs = n_envs
        
        # Networks
        self.actor = ActorNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=config.train.mappo_actor_hidden,
            activation=config.train.mappo_activation,
            use_layer_norm=config.train.mappo_use_layer_norm
        ).to(self.device)
        
        self.critic = CriticNetwork(
            global_obs_dim=self.global_obs_dim,
            hidden_dims=config.train.mappo_critic_hidden,
            activation=config.train.mappo_activation,
            use_layer_norm=config.train.mappo_use_layer_norm
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.train.mappo_lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.train.mappo_lr_critic
        )
        
        # Rollout buffer
        self.buffer = RolloutBuffer(
            rollout_length=self.rollout_length * self.n_envs,
            n_agents=self.n_agents,
            obs_dim=self.obs_dim,
            global_obs_dim=self.global_obs_dim,
            action_dim=self.action_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        # Logging
        self.log_interval = config.train.mappo_log_interval
        self.viz_interval = config.train.mappo_viz_interval
        self.checkpoint_interval = config.train.mappo_checkpoint_interval
        
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_coverage = deque(maxlen=100)
        self.episode_victims = deque(maxlen=100)
        self.total_episodes_done = 0
        self.total_steps_collected = 0  # ✅ Track actual steps
        
        # Directories
        self.output_dir = Path("results") / "mappo" / self.run_name
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.viz_dir = self.output_dir / "viz"
        self.plots_dir = self.output_dir / "plots"
        
        for d in [self.checkpoint_dir, self.viz_dir, self.plots_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"🚁 MAPPO Trainer Initialized")
        print(f"{'='*60}")
        print(f"  Device: {self.device}")
        print(f"  Run name: {self.run_name}")
        print(f"  Actor params: {sum(p.numel() for p in self.actor.parameters()):,}")
        print(f"  Critic params: {sum(p.numel() for p in self.critic.parameters()):,}")
        print(f"  Output dir: {self.output_dir}")
        print(f"{'='*60}\n")
    
    def select_action(
        self,
        obs_dict: Dict[str, np.ndarray],
        deterministic: bool = False
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Select actions cho tất cả agents."""
        agent_ids = sorted(obs_dict.keys())
        obs_list = [obs_dict[aid] for aid in agent_ids]
        obs_batch = torch.FloatTensor(np.array(obs_list)).to(self.device)
        
        with torch.no_grad():
            actions, log_probs = self.actor.get_action(obs_batch, deterministic=deterministic)
        
        actions_np = actions.cpu().numpy()
        log_probs_np = log_probs.cpu().numpy()
        
        actions_dict = {aid: actions_np[i] for i, aid in enumerate(agent_ids)}
        
        return actions_dict, actions_np, log_probs_np
    
    def get_values(self, global_obs: np.ndarray) -> np.ndarray:
        """Get value estimates từ critic."""
        global_obs_tensor = torch.FloatTensor(global_obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.critic.get_value(global_obs_tensor).cpu().item()
        
        return np.full(self.n_agents, value, dtype=np.float32)
    
    def train(
        self,
        total_episodes: int,
        curriculum_manager=None,
        seed: int = 42
    ):
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                from tqdm.notebook import tqdm
            else:
                from tqdm import tqdm
        except ImportError:
            from tqdm import tqdm

        env = self._create_env(seed)
        update_count = 0
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"🚀 Starting MAPPO Training")
        print(f"{'='*60}")
        print(f"  Target episodes:   {total_episodes}")
        print(f"  Rollout length:    {self.rollout_length}")
        print(f"  Batch size:        {self.batch_size}")
        print(f"  n_envs:            {self.n_envs}")
        print(f"  Curriculum:        {'Enabled' if curriculum_manager else 'Disabled'}")
        print(f"{'='*60}\n")

        pbar = tqdm(
            total=total_episodes,
            desc="🚁 SAR",
            unit="ep",
            dynamic_ncols=True,
        )

        pbar.set_postfix(ordered_dict={
            'upd': '0',
            'rew': 'collecting...',
            'cov': '...',
            'vic': '...',
            'a_loss': '...',
            'c_loss': '...',
        })

        while self.total_episodes_done < total_episodes:
            rollout_metrics = self.rollout(
                env, 
                pbar=pbar,
                max_episodes=total_episodes
            )
            
            train_metrics = self.update()
            update_count += 1

            pbar.set_postfix(ordered_dict={
                'upd':    str(update_count),
                'rew':    f"{rollout_metrics['mean_ep_reward']:+.1f}",
                'cov':    f"{rollout_metrics['mean_coverage']:.1f}%",
                'vic':    f"{rollout_metrics['mean_victims']:.1f}%",
                'a_loss': f"{train_metrics['actor_loss']:.3f}",
                'c_loss': f"{train_metrics['critic_loss']:.1f}",
            })
            pbar.refresh()

            # Detailed log
            if update_count % self.log_interval == 0:
                elapsed = time.time() - start_time
                # ✅ FPS chính xác dựa trên actual steps
                fps = self.total_steps_collected / max(elapsed, 1e-6)

                pbar.write(f"\n{'─'*60}")
                pbar.write(
                    f"📊 Update {update_count} | "
                    f"Episodes {self.total_episodes_done}/{total_episodes}"
                )
                pbar.write(f"{'─'*60}")
                pbar.write(
                    f"  Task   | Reward: {rollout_metrics['mean_ep_reward']:+8.2f} | "
                    f"Coverage: {rollout_metrics['mean_coverage']:5.1f}% | "
                    f"Victims: {rollout_metrics['mean_victims']:5.1f}%"
                )
                pbar.write(
                    f"  Train  | Actor: {train_metrics['actor_loss']:7.4f} | "
                    f"Critic: {train_metrics['critic_loss']:7.4f} | "
                    f"Entropy: {train_metrics['entropy']:6.4f} | "
                    f"Clip: {train_metrics['clip_fraction']:.3f}"
                )
                pbar.write(
                    f"  Perf   | FPS: {fps:6.1f} | "
                    f"Steps: {self.total_steps_collected} | "
                    f"Time: {elapsed/60:.1f}min"
                )
                
                # ✅ Buffer stats
                buf_stats = self.buffer.get_stats()
                if buf_stats:
                    pbar.write(
                        f"  Buffer | Fill: {buf_stats['buffer_fill']:.1%} | "
                        f"Reward: {buf_stats['mean_reward']:+.2f} | "
                        f"Value: {buf_stats['mean_value']:+.2f}"
                    )
                
                if curriculum_manager:
                    s = curriculum_manager.current_stage
                    pbar.write(f"  Stage  | {s.name.upper()} ({s.map_size}m)")
                pbar.write(f"{'─'*60}\n")

            # Viz & Checkpoint
            if update_count % self.viz_interval == 0:
                self._save_visualization(env, update_count)

            if update_count % self.checkpoint_interval == 0:
                self.save_checkpoint(update_count, curriculum_manager)

            # Curriculum
            if curriculum_manager and self.total_episodes_done > 0:
                curriculum_manager.update(
                    coverage=rollout_metrics['mean_coverage'] / 100,
                    victims_rate=rollout_metrics['mean_victims'] / 100,
                    reward=rollout_metrics['mean_ep_reward']
                )
                if curriculum_manager.should_advance():
                    old = curriculum_manager.current_stage.name
                    curriculum_manager.advance()
                    new = curriculum_manager.current_stage.name
                    pbar.write(f"\n🎓 CURRICULUM: {old.upper()} → {new.upper()}\n")
                    
                    # ✅ Close old env before creating new
                    if hasattr(env, 'close'):
                        env.close()
                    
                    curriculum_manager.apply_to_config(self.config)
                    env = self._create_env(seed)

        pbar.close()
        
        # ✅ Final checkpoint
        self.save_checkpoint(update_count, curriculum_manager)
        
        # ✅ Close env
        if hasattr(env, 'close'):
            env.close()

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"  Total updates:   {update_count}")
        print(f"  Total episodes:  {self.total_episodes_done}")
        print(f"  Total steps:     {self.total_steps_collected}")
        print(f"  Time:            {elapsed/60:.1f} min")
        print(f"  FPS (avg):       {self.total_steps_collected/elapsed:.1f}")
        if rollout_metrics:
            print(f"  Final reward:    {rollout_metrics['mean_ep_reward']:.2f}")
            print(f"  Final coverage:  {rollout_metrics['mean_coverage']:.2f}%")
            print(f"  Final victims:   {rollout_metrics['mean_victims']:.2f}%")
        print(f"{'='*60}\n")

    def rollout(self, env, pbar=None, max_episodes=None) -> Dict[str, float]:
        """Dispatch rollout."""
        if self.n_envs == 1:
            return self._rollout_single(env, pbar=pbar, max_episodes=max_episodes)
        else:
            return self._rollout_vectorized(env, pbar=pbar, max_episodes=max_episodes)

    def _rollout_single(self, env, pbar=None, max_episodes=None) -> Dict[str, float]:
        """Rollout với single env."""
        obs_dict, infos = env.reset()
        global_obs = infos['uav_0']['global_obs']

        episode_reward = 0.0
        episode_length = 0
        steps_collected = 0
        last_values = None
        last_done = False

        while steps_collected < self.rollout_length:
            if max_episodes is not None and self.total_episodes_done >= max_episodes:
                break

            actions_dict, actions_np, log_probs_np = self.select_action(
                obs_dict, deterministic=False
            )
            values_np = self.get_values(global_obs)

            next_obs_dict, rewards_dict, terms_dict, truncs_dict, next_infos = (
                env.step(actions_dict)
            )
            next_global_obs = next_infos['uav_0']['global_obs']

            agent_ids = sorted(rewards_dict.keys())
            rewards_np = np.array(
                [rewards_dict[aid] for aid in agent_ids], dtype=np.float32
            )
            done = any(terms_dict.values()) or any(truncs_dict.values())

            obs_np = np.array(
                [obs_dict[aid] for aid in agent_ids], dtype=np.float32
            )
            self.buffer.add(
                obs=obs_np,
                global_obs=global_obs,
                actions=actions_np,
                rewards=rewards_np,
                values=values_np,
                log_probs=log_probs_np,
                done=done
            )

            obs_dict = next_obs_dict
            global_obs = next_global_obs
            episode_reward += rewards_np[0]
            episode_length += 1
            steps_collected += 1
            self.total_steps_collected += 1  # ✅ Track steps

            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                cov = next_infos['uav_0']['coverage_rate'] * 100
                vic = (
                    next_infos['uav_0']['victims_found']
                    / max(next_infos['uav_0']['victims_total'], 1) * 100
                )
                self.episode_coverage.append(cov)
                self.episode_victims.append(vic)
                self.total_episodes_done += 1

                if pbar is not None:
                    pbar.update(1)

                if max_episodes is not None and self.total_episodes_done >= max_episodes:
                    last_values = self.get_values(global_obs)
                    last_done = done
                    break

                obs_dict, infos = env.reset()
                global_obs = infos['uav_0']['global_obs']
                episode_reward = 0.0
                episode_length = 0

            if steps_collected == self.rollout_length:
                last_values = self.get_values(global_obs)
                last_done = done

        if last_values is None:
            last_values = self.get_values(global_obs)
            last_done = False
        
        self.buffer.compute_gae(last_values, last_done)

        return {
            'mean_ep_reward': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            'mean_ep_length': float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0,
            'mean_coverage':  float(np.mean(self.episode_coverage)) if self.episode_coverage else 0.0,
            'mean_victims':   float(np.mean(self.episode_victims)) if self.episode_victims else 0.0,
        }

    def _rollout_vectorized(self, env, pbar=None, max_episodes=None) -> Dict[str, float]:
        """Rollout với vectorized envs."""
        obs_batch, global_obs_batch = env.reset()

        episode_reward_buffer = np.zeros(self.n_envs, dtype=np.float32)
        episode_length_buffer = np.zeros(self.n_envs, dtype=np.int32)

        n_iterations = self.rollout_length
        should_stop = False

        for step_idx in range(n_iterations):
            if should_stop:
                break

            total_agents = self.n_envs * self.n_agents
            obs_flat = obs_batch.reshape(total_agents, self.obs_dim)

            obs_tensor = torch.FloatTensor(obs_flat).to(self.device)
            with torch.no_grad():
                actions_tensor, log_probs_tensor = self.actor.get_action(
                    obs_tensor, deterministic=False
                )

            actions_batch = actions_tensor.cpu().numpy().reshape(
                self.n_envs, self.n_agents, self.action_dim
            )
            log_probs_batch = log_probs_tensor.cpu().numpy().reshape(
                self.n_envs, self.n_agents
            )

            global_obs_tensor = torch.FloatTensor(global_obs_batch).to(self.device)
            with torch.no_grad():
                values_per_env = self.critic.get_value(
                    global_obs_tensor
                ).cpu().numpy()

            values_batch = np.stack(
                [np.full(self.n_agents, v, dtype=np.float32) for v in values_per_env],
                axis=0
            )

            (next_obs_batch,
             next_global_obs_batch,
             rewards_batch,
             dones,
             infos) = env.step(actions_batch)

            for env_idx in range(self.n_envs):
                if max_episodes is not None and self.total_episodes_done >= max_episodes:
                    should_stop = True
                    break

                self.buffer.add(
                    obs=obs_batch[env_idx],
                    global_obs=global_obs_batch[env_idx],
                    actions=actions_batch[env_idx],
                    rewards=rewards_batch[env_idx],
                    values=values_batch[env_idx],
                    log_probs=log_probs_batch[env_idx],
                    done=dones[env_idx]
                )
                
                # ✅ Track steps per env
                self.total_steps_collected += 1

                episode_reward_buffer[env_idx] += rewards_batch[env_idx][0]
                episode_length_buffer[env_idx] += 1

                if dones[env_idx]:
                    self.episode_rewards.append(float(episode_reward_buffer[env_idx]))
                    self.episode_lengths.append(int(episode_length_buffer[env_idx]))

                    info_e = infos[env_idx]
                    if 'uav_0' in info_e:
                        cov = info_e['uav_0']['coverage_rate'] * 100
                        vic = (
                            info_e['uav_0']['victims_found']
                            / max(info_e['uav_0']['victims_total'], 1) * 100
                        )
                        self.episode_coverage.append(cov)
                        self.episode_victims.append(vic)

                    self.total_episodes_done += 1

                    if pbar is not None:
                        pbar.update(1)

                    if max_episodes is not None and self.total_episodes_done >= max_episodes:
                        should_stop = True
                        break

                    episode_reward_buffer[env_idx] = 0.0
                    episode_length_buffer[env_idx] = 0

            if should_stop:
                break

            obs_batch = next_obs_batch
            global_obs_batch = next_global_obs_batch

        # Bootstrap GAE
        last_global_obs_tensor = torch.FloatTensor(global_obs_batch).to(self.device)
        with torch.no_grad():
            last_values_per_env = self.critic.get_value(
                last_global_obs_tensor
            ).cpu().numpy()

        mean_last_value = float(np.mean(last_values_per_env))
        last_values_for_gae = np.full(self.n_agents, mean_last_value, dtype=np.float32)

        self.buffer.compute_gae(last_values_for_gae, last_done=False)

        return {
            'mean_ep_reward': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            'mean_ep_length': float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0,
            'mean_coverage':  float(np.mean(self.episode_coverage)) if self.episode_coverage else 0.0,
            'mean_victims':   float(np.mean(self.episode_victims)) if self.episode_victims else 0.0,
        }
    
    def update(self) -> Dict[str, float]:
        """PPO update."""
        actor_losses = []
        critic_losses = []
        entropies = []
        clip_fractions = []
        
        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                obs = torch.FloatTensor(batch['obs']).to(self.device)
                global_obs = torch.FloatTensor(batch['global_obs']).to(self.device)
                actions = torch.FloatTensor(batch['actions']).to(self.device)
                old_log_probs = torch.FloatTensor(batch['old_log_probs']).to(self.device)
                advantages = torch.FloatTensor(batch['advantages']).to(self.device)
                returns = torch.FloatTensor(batch['returns']).to(self.device)
                
                # Actor
                log_probs, entropy = self.actor.evaluate_actions(obs, actions)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy.mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Critic
                values = self.critic.get_value(global_obs)
                critic_loss = nn.functional.mse_loss(values, returns)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                # Metrics
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())
                
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    clip_fractions.append(clip_frac)
        
        self.buffer.clear()
        
        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropies),
            'clip_fraction': np.mean(clip_fractions)
        }
    
    def _create_env(self, seed: int):
        """Tạo env."""
        if self.n_envs == 1:
            env = SARPettingZooEnv(self.config, render_mode=None)
            env.reset(seed=seed)
            return env
        else:
            return VectorizedEnv(self.config, n_envs=self.n_envs, start_seed=seed)
    
    def _save_visualization(self, env, update: int):
        """
        Render và lưu snapshot.
        
        ✅ Chỉ hoạt động với single env (SARPettingZooEnv).
        """
        # ✅ Check env type
        if not isinstance(env, SARPettingZooEnv):
            # VectorizedEnv không có render()
            return
        
        try:
            frame = env.render()
            if frame is not None:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(frame)
                ax.axis('off')
                ax.set_title(f"Update {update}", fontsize=16)
                
                save_path = self.viz_dir / f"update_{update:05d}.png"
                plt.savefig(save_path, bbox_inches='tight', dpi=100)
                plt.close(fig)
                
                print(f"  💾 Saved viz: {save_path.name}")
        except Exception as e:
            print(f"  ⚠️  Viz save failed: {e}")
    
    def save_checkpoint(self, update: int, curriculum_manager: Optional[CurriculumManager] = None):
        """Lưu checkpoint."""
        checkpoint = {
            'update': update,
            'total_episodes_done': self.total_episodes_done,
            'total_steps_collected': self.total_steps_collected,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config,
            'episode_rewards': list(self.episode_rewards),
            'episode_coverage': list(self.episode_coverage),
            'episode_victims': list(self.episode_victims)
        }
        
        if curriculum_manager:
            checkpoint['curriculum_stage'] = curriculum_manager.stage_idx
        
        save_path = self.checkpoint_dir / f"checkpoint_update_{update:05d}.pt"
        torch.save(checkpoint, save_path)
        print(f"  💾 Saved checkpoint: {save_path.name}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.total_episodes_done = checkpoint.get('total_episodes_done', 0)
        self.total_steps_collected = checkpoint.get('total_steps_collected', 0)
        
        print(f"✅ Loaded checkpoint from update {checkpoint['update']}")
        return checkpoint['update']