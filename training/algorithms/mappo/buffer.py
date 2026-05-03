"""
MAPPO Rollout Buffer with GAE (Generalized Advantage Estimation)
Lưu trữ experience và tính advantages cho PPO update.
"""

import numpy as np
from typing import List, Tuple, Iterator, Dict


class RolloutBuffer:
    """
    Rollout buffer cho MAPPO với GAE computation.
    
    Storage layout:
        - observations: [rollout_length, n_agents, obs_dim]
        - global_obs: [rollout_length, global_obs_dim]
        - actions: [rollout_length, n_agents, action_dim]
        - rewards: [rollout_length, n_agents]
        - values: [rollout_length, n_agents]
        - log_probs: [rollout_length, n_agents]
        - dones: [rollout_length]  # shared termination
        - advantages: [rollout_length, n_agents]  # computed
        - returns: [rollout_length, n_agents]  # computed
    """
    
    def __init__(
        self,
        rollout_length: int,
        n_agents: int,
        obs_dim: int,
        global_obs_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """
        Args:
            rollout_length: Steps per update (e.g., 2048)
            n_agents: Số UAVs (4)
            obs_dim: Actor obs dim (68)
            global_obs_dim: Critic obs dim (554)
            action_dim: Action dim (3)
            gamma: Discount factor
            gae_lambda: GAE lambda
        """
        self.rollout_length = rollout_length
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.global_obs_dim = global_obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Storage arrays (preallocate)
        self.observations = np.zeros((rollout_length, n_agents, obs_dim), dtype=np.float32)
        self.global_obs = np.zeros((rollout_length, global_obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_length, n_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((rollout_length, n_agents), dtype=np.float32)
        self.values = np.zeros((rollout_length, n_agents), dtype=np.float32)
        self.log_probs = np.zeros((rollout_length, n_agents), dtype=np.float32)
        self.dones = np.zeros(rollout_length, dtype=np.float32)
        
        # Computed arrays
        self.advantages = np.zeros((rollout_length, n_agents), dtype=np.float32)
        self.returns = np.zeros((rollout_length, n_agents), dtype=np.float32)
        
        # Buffer pointer
        self.ptr = 0
        self.full = False
    
    def add(
        self,
        obs: np.ndarray,          # [n_agents, obs_dim]
        global_obs: np.ndarray,   # [global_obs_dim]
        actions: np.ndarray,      # [n_agents, action_dim]
        rewards: np.ndarray,      # [n_agents]
        values: np.ndarray,       # [n_agents]
        log_probs: np.ndarray,    # [n_agents]
        done: bool
    ):
        """
        Thêm 1 transition vào buffer.
        
        Args:
            obs: Local observations từ env (dict → stacked array)
            global_obs: Global observation cho critic
            actions: Actions đã thực hiện
            rewards: Rewards nhận được (shared reward)
            values: Value estimates từ critic
            log_probs: Log probabilities của actions
            done: Episode done flag
        """
        assert self.ptr < self.rollout_length, "Buffer overflow! Call compute_gae() and clear()"
        
        self.observations[self.ptr] = obs
        self.global_obs[self.ptr] = global_obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.values[self.ptr] = values
        self.log_probs[self.ptr] = log_probs
        self.dones[self.ptr] = float(done)
        
        self.ptr += 1
        if self.ptr == self.rollout_length:
            self.full = True
    
    def compute_gae(
        self,
        last_values: np.ndarray,  # [n_agents]
        last_done: bool
    ):
        """
        Tính GAE advantages và returns.
        
        GAE formula:
            δ_t = r_t + γ V(s_{t+1}) (1 - done_{t+1}) - V(s_t)
            A_t = δ_t + (γλ) δ_{t+1} + (γλ)² δ_{t+2} + ...
            R_t = A_t + V(s_t)
        
        Args:
            last_values: Value estimate của state cuối (bootstrap)
            last_done: Done flag của step cuối
        """
        assert self.full, "Buffer chưa đầy, không thể compute GAE"
        
        # Compute advantages per agent
        for agent_idx in range(self.n_agents):
            last_gae_lam = 0.0
            
            # Backward iteration (T-1 → 0)
            for t in reversed(range(self.rollout_length)):
                if t == self.rollout_length - 1:
                    next_non_terminal = 1.0 - float(last_done)
                    next_value = last_values[agent_idx]
                else:
                    next_non_terminal = 1.0 - self.dones[t + 1]
                    next_value = self.values[t + 1, agent_idx]
                
                # TD residual: δ_t = r_t + γ V(s_{t+1}) (1-done) - V(s_t)
                delta = (
                    self.rewards[t, agent_idx] +
                    self.gamma * next_value * next_non_terminal -
                    self.values[t, agent_idx]
                )
                
                # GAE: A_t = δ_t + (γλ)(1-done) A_{t+1}
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                self.advantages[t, agent_idx] = last_gae_lam
        
        # Returns = Advantages + Values (TD(λ) targets)
        self.returns = self.advantages + self.values
        
        # Normalize advantages (across all agents and steps)
        adv_flat = self.advantages.reshape(-1)
        adv_mean = adv_flat.mean()
        adv_std = adv_flat.std()
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
    
    def get_batches(self, batch_size: int) -> Iterator[Dict[str, np.ndarray]]:
        """
        Generator minibatches cho PPO update.
        
        Yields:
            Dict với keys: obs, global_obs, actions, old_log_probs, advantages, returns
            Mỗi tensor shape: [batch_size, ...]
        """
        assert self.full, "Buffer chưa đầy"
        
        # Flatten to [rollout_length * n_agents, ...]
        total_samples = self.rollout_length * self.n_agents
        
        obs_flat = self.observations.reshape(total_samples, self.obs_dim)
        actions_flat = self.actions.reshape(total_samples, self.action_dim)
        log_probs_flat = self.log_probs.reshape(total_samples)
        advantages_flat = self.advantages.reshape(total_samples)
        returns_flat = self.returns.reshape(total_samples)
        
        # Repeat global_obs for each agent
        # [rollout_length, global_obs_dim] → [rollout_length, n_agents, global_obs_dim] → [total_samples, global_obs_dim]
        global_obs_repeated = np.repeat(self.global_obs[:, None, :], self.n_agents, axis=1)
        global_obs_flat = global_obs_repeated.reshape(total_samples, self.global_obs_dim)
        
        # Random permutation
        indices = np.random.permutation(total_samples)
        
        # Yield batches
        start_idx = 0
        while start_idx < total_samples:
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            yield {
                'obs': obs_flat[batch_indices],
                'global_obs': global_obs_flat[batch_indices],
                'actions': actions_flat[batch_indices],
                'old_log_probs': log_probs_flat[batch_indices],
                'advantages': advantages_flat[batch_indices],
                'returns': returns_flat[batch_indices]
            }
            
            start_idx += batch_size
    
    def clear(self):
        """Reset buffer pointer (giữ arrays cho reuse)."""
        self.ptr = 0
        self.full = False
    
    def get_stats(self) -> Dict[str, float]:
        """Lấy buffer statistics."""
        if not self.full:
            return {}
        
        return {
            'mean_reward': self.rewards.mean(),
            'mean_value': self.values.mean(),
            'mean_advantage': self.advantages.mean(),
            'std_advantage': self.advantages.std(),
            'mean_return': self.returns.mean(),
        }