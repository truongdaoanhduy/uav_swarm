"""
MAPPO Trainer - Main training loop với curriculum support
Console log mỗi 10 episodes, 2D viz mỗi 100 episodes, checkpoint mỗi 100 episodes.
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
        run_name: Optional[str] = None
    ):
        """
        Args:
            config: Master config
            device: torch device
            run_name: Tên run (auto-gen nếu None)
        """
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
        
        # Networks (shared actor weights across agents)
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
            rollout_length=self.rollout_length,
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
        """
        Select actions cho tất cả agents.
        
        Args:
            obs_dict: {"uav_0": obs[68], "uav_1": ...}
            deterministic: Greedy mode (eval)
        
        Returns:
            actions_dict: {"uav_0": action[3], ...}
            actions_array: [n_agents, 3]
            log_probs_array: [n_agents]
        """
        # Stack observations
        agent_ids = sorted(obs_dict.keys())
        obs_list = [obs_dict[aid] for aid in agent_ids]
        obs_batch = torch.FloatTensor(np.array(obs_list)).to(self.device)  # [n_agents, 68]
        
        # Forward pass
        with torch.no_grad():
            actions, log_probs = self.actor.get_action(obs_batch, deterministic=deterministic)
        
        # Convert to numpy
        actions_np = actions.cpu().numpy()
        log_probs_np = log_probs.cpu().numpy()
        
        # Build dict
        actions_dict = {aid: actions_np[i] for i, aid in enumerate(agent_ids)}
        
        return actions_dict, actions_np, log_probs_np
    
    def get_values(self, global_obs: np.ndarray) -> np.ndarray:
        """
        Get value estimates từ critic.
        
        Args:
            global_obs: [global_obs_dim=554]
        
        Returns:
            values: [n_agents] (broadcast same value)
        """
        global_obs_tensor = torch.FloatTensor(global_obs).unsqueeze(0).to(self.device)  # [1, 554]
        
        with torch.no_grad():
            value = self.critic.get_value(global_obs_tensor).cpu().item()
        
        # MAPPO: tất cả agents share same value (centralized critic)
        return np.full(self.n_agents, value, dtype=np.float32)
    
    def rollout(self, env: SARPettingZooEnv) -> Dict[str, float]:
        """
        Collect rollout_length steps of experience.
        
        Returns:
            metrics: Dict với episode statistics
        """
        obs_dict, infos = env.reset()
        global_obs = infos['uav_0']['global_obs']
        
        episode_reward = 0.0
        episode_length = 0
        steps_collected = 0
        
        last_values = None
        last_done = False
        
        while steps_collected < self.rollout_length:
            # Select actions
            actions_dict, actions_np, log_probs_np = self.select_action(obs_dict, deterministic=False)
            
            # Get values
            values_np = self.get_values(global_obs)
            
            # Step env
            next_obs_dict, rewards_dict, terms_dict, truncs_dict, next_infos = env.step(actions_dict)
            next_global_obs = next_infos['uav_0']['global_obs']
            
            # Extract rewards (shared reward - all agents same value)
            agent_ids = sorted(rewards_dict.keys())
            rewards_np = np.array([rewards_dict[aid] for aid in agent_ids], dtype=np.float32)
            
            done = any(terms_dict.values()) or any(truncs_dict.values())
            
            # Store transition
            obs_list = [obs_dict[aid] for aid in agent_ids]
            obs_np = np.array(obs_list, dtype=np.float32)
            
            self.buffer.add(
                obs=obs_np,
                global_obs=global_obs,
                actions=actions_np,
                rewards=rewards_np,
                values=values_np,
                log_probs=log_probs_np,
                done=done
            )
            
            # Update state
            obs_dict = next_obs_dict
            global_obs = next_global_obs
            episode_reward += rewards_np[0]  # shared reward
            episode_length += 1
            steps_collected += 1
            
            # Episode termination
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_coverage.append(next_infos['uav_0']['coverage_rate'] * 100)
                self.episode_victims.append(
                    next_infos['uav_0']['victims_found'] / next_infos['uav_0']['victims_total'] * 100
                )
                
                # Reset
                obs_dict, infos = env.reset()
                global_obs = infos['uav_0']['global_obs']
                episode_reward = 0.0
                episode_length = 0
            
            # Store last state for GAE bootstrap
            if steps_collected == self.rollout_length:
                last_values = self.get_values(global_obs)
                last_done = done
        
        # Compute GAE
        self.buffer.compute_gae(last_values, last_done)
        
        return {
            'mean_ep_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'mean_ep_length': np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            'mean_coverage': np.mean(self.episode_coverage) if self.episode_coverage else 0.0,
            'mean_victims': np.mean(self.episode_victims) if self.episode_victims else 0.0
        }
    
    def update(self) -> Dict[str, float]:
        """
        PPO update với n_epochs epochs và minibatch sampling.
        
        Returns:
            train_metrics: Dict với loss values
        """
        actor_losses = []
        critic_losses = []
        entropies = []
        clip_fractions = []
        
        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                # Convert to tensors
                obs = torch.FloatTensor(batch['obs']).to(self.device)
                global_obs = torch.FloatTensor(batch['global_obs']).to(self.device)
                actions = torch.FloatTensor(batch['actions']).to(self.device)
                old_log_probs = torch.FloatTensor(batch['old_log_probs']).to(self.device)
                advantages = torch.FloatTensor(batch['advantages']).to(self.device)
                returns = torch.FloatTensor(batch['returns']).to(self.device)
                
                # ============ Actor Update ============
                log_probs, entropy = self.actor.evaluate_actions(obs, actions)
                
                # PPO clipped loss
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy.mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # ============ Critic Update ============
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
                
                # Clip fraction (indicator of too large updates)
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    clip_fractions.append(clip_frac)
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropies),
            'clip_fraction': np.mean(clip_fractions)
        }
    
    def train(
        self,
        total_updates: int,
        curriculum_manager: Optional[CurriculumManager] = None,
        seed: int = 42
    ):
        from tqdm import tqdm

        env = self._create_env(seed)
        update_count = 0
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"🚀 Starting MAPPO Training")
        print(f"{'='*60}")
        print(f"  Total updates:  {total_updates}")
        print(f"  Rollout length: {self.rollout_length}")
        print(f"  Batch size:     {self.batch_size}")
        print(f"  Curriculum:     {'Enabled' if curriculum_manager else 'Disabled'}")
        print(f"{'='*60}\n")

        # ── TQDM progress bar ──────────────────────────────────────
        pbar = tqdm(
            total=total_updates,
            desc="🚁 Training",
            unit="update",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
        )

        rollout_metrics = {}
        train_metrics = {}

        while update_count < total_updates:
            # Rollout + Update
            rollout_metrics = self.rollout(env)
            train_metrics   = self.update()

            update_count += 1
            episode_count = len(self.episode_rewards)

            # ── Update tqdm postfix (real-time) ───────────────────
            pbar.set_postfix(ordered_dict={
                'ep':      episode_count,
                'rew':     f"{rollout_metrics['mean_ep_reward']:+.1f}",
                'cov':     f"{rollout_metrics['mean_coverage']:.1f}%",
                'vic':     f"{rollout_metrics['mean_victims']:.1f}%",
                'a_loss':  f"{train_metrics['actor_loss']:.3f}",
                'c_loss':  f"{train_metrics['critic_loss']:.1f}",
            })
            pbar.update(1)

            # ── Detailed log mỗi log_interval updates ─────────────
            if update_count % self.log_interval == 0:
                elapsed = time.time() - start_time
                fps = (update_count * self.rollout_length) / elapsed

                pbar.write(f"\n{'─'*60}")
                pbar.write(f"📊 Update {update_count}/{total_updates} | Episodes: {episode_count}")
                pbar.write(f"{'─'*60}")
                pbar.write(f"  Task Metrics:")
                pbar.write(f"    Reward:    {rollout_metrics['mean_ep_reward']:8.2f}")
                pbar.write(f"    Coverage:  {rollout_metrics['mean_coverage']:7.2f}%")
                pbar.write(f"    Victims:   {rollout_metrics['mean_victims']:7.2f}%")
                pbar.write(f"    Ep Length: {rollout_metrics['mean_ep_length']:7.1f}")
                pbar.write(f"  Training Metrics:")
                pbar.write(f"    Actor Loss:  {train_metrics['actor_loss']:.4f}")
                pbar.write(f"    Critic Loss: {train_metrics['critic_loss']:.4f}")
                pbar.write(f"    Entropy:     {train_metrics['entropy']:.4f}")
                pbar.write(f"    Clip Frac:   {train_metrics['clip_fraction']:.3f}")
                pbar.write(f"  Performance:")
                pbar.write(f"    FPS:       {fps:.1f}")
                pbar.write(f"    Time:      {elapsed/60:.1f} min")

                if curriculum_manager:
                    stage = curriculum_manager.current_stage
                    pbar.write(f"  Curriculum:")
                    pbar.write(f"    Stage:     {stage.name.upper()}")
                    pbar.write(f"    Map Size:  {stage.map_size}m")
                pbar.write(f"{'─'*60}\n")

            # ── Visualization ──────────────────────────────────────
            if update_count % self.viz_interval == 0:
                self._save_visualization(env, update_count)

            # ── Checkpoint ────────────────────────────────────────
            if update_count % self.checkpoint_interval == 0:
                self.save_checkpoint(update_count, curriculum_manager)

            # ── Curriculum Advancement ────────────────────────────
            if curriculum_manager and episode_count > 0:
                curriculum_manager.update(
                    coverage=rollout_metrics['mean_coverage'] / 100,
                    victims_rate=rollout_metrics['mean_victims'] / 100,
                    reward=rollout_metrics['mean_ep_reward']
                )

                if curriculum_manager.should_advance():
                    old_stage = curriculum_manager.current_stage.name
                    curriculum_manager.advance()
                    new_stage = curriculum_manager.current_stage.name

                    pbar.write(f"\n🎓 CURRICULUM ADVANCE: {old_stage.upper()} → {new_stage.upper()}\n")

                    curriculum_manager.apply_to_config(self.config)
                    env = self._create_env(seed)

        pbar.close()

        # Final save
        self.save_checkpoint(update_count, curriculum_manager)

        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"{'='*60}")
        print(f"  Total updates:  {update_count}")
        print(f"  Total episodes: {episode_count}")
        print(f"  Total time:     {(time.time()-start_time)/60:.1f} min")
        if rollout_metrics:
            print(f"  Final reward:   {rollout_metrics['mean_ep_reward']:.2f}")
            print(f"  Final coverage: {rollout_metrics['mean_coverage']:.2f}%")
            print(f"  Final victims:  {rollout_metrics['mean_victims']:.2f}%")
        print(f"{'='*60}\n")
    
    def _create_env(self, seed: int) -> SARPettingZooEnv:
        """Tạo environment từ config."""
        env = SARPettingZooEnv(self.config, render_mode=None)
        env.reset(seed=seed)
        return env
    
    def _save_visualization(self, env: SARPettingZooEnv, update: int):
        """Render và lưu 2D snapshot."""
        try:
            # Render frame
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
        
        print(f"✅ Loaded checkpoint from update {checkpoint['update']}")
        return checkpoint['update']