"""
Off-Policy Replay Buffer cho MASAC/MATD3.

Stores transitions: (obs, global_obs, action, reward, next_obs, next_global, done)
Capacity: 500K–1M transitions

Key differences from RolloutBuffer (on-policy):
    - Large capacity (1M vs ~2K)
    - Random sampling (not sequential)
    - No GAE computation
    - Persistent across updates (không clear)
"""

import numpy as np
from typing import Dict, Optional, Tuple


class ReplayBuffer:
    """
    Uniform replay buffer cho off-policy algorithms.

    Shape convention:
        obs:         [N, n_agents, obs_dim]        — per-agent local obs
        global_obs:  [N, global_obs_dim]           — centralized obs
        actions:     [N, n_agents, action_dim]     — per-agent actions
        rewards:     [N, n_agents]                 — per-agent rewards
        next_obs:    [N, n_agents, obs_dim]
        next_global: [N, global_obs_dim]
        dones:       [N]                           — episode termination
    """

    def __init__(
        self,
        capacity:       int,
        n_agents:       int,
        obs_dim:        int,
        global_obs_dim: int,
        action_dim:     int,
    ):
        self.capacity       = capacity
        self.n_agents       = n_agents
        self.obs_dim        = obs_dim
        self.global_obs_dim = global_obs_dim
        self.action_dim     = action_dim

        self.ptr  = 0      # Write pointer
        self.size = 0      # Current number of valid transitions

        # ── Storage ─────────────────────────────────────────────────────
        self.obs         = np.zeros((capacity, n_agents, obs_dim),    dtype=np.float32)
        self.global_obs  = np.zeros((capacity, global_obs_dim),       dtype=np.float32)
        self.actions     = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.rewards     = np.zeros((capacity, n_agents),             dtype=np.float32)
        self.next_obs    = np.zeros((capacity, n_agents, obs_dim),    dtype=np.float32)
        self.next_global = np.zeros((capacity, global_obs_dim),       dtype=np.float32)
        self.dones       = np.zeros((capacity,),                       dtype=np.float32)

    def add(
        self,
        obs:         np.ndarray,   # [n_agents, obs_dim]
        global_obs:  np.ndarray,   # [global_obs_dim]
        actions:     np.ndarray,   # [n_agents, action_dim]
        rewards:     np.ndarray,   # [n_agents]
        next_obs:    np.ndarray,   # [n_agents, obs_dim]
        next_global: np.ndarray,   # [global_obs_dim]
        done:        float,        # scalar
    ):
        """Add single transition."""
        self.obs[self.ptr]         = obs
        self.global_obs[self.ptr]  = global_obs
        self.actions[self.ptr]     = actions
        self.rewards[self.ptr]     = rewards
        self.next_obs[self.ptr]    = next_obs
        self.next_global[self.ptr] = next_global
        self.dones[self.ptr]       = done

        # Circular buffer
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(
        self,
        obs:         np.ndarray,   # [E, n_agents, obs_dim]
        global_obs:  np.ndarray,   # [E, global_obs_dim]
        actions:     np.ndarray,   # [E, n_agents, action_dim]
        rewards:     np.ndarray,   # [E, n_agents]
        next_obs:    np.ndarray,   # [E, n_agents, obs_dim]
        next_global: np.ndarray,   # [E, global_obs_dim]
        dones:       np.ndarray,   # [E]
    ):
        """Add batch of transitions (từ vectorized env)."""
        n = len(obs)
        for i in range(n):
            self.add(
                obs[i], global_obs[i], actions[i], rewards[i],
                next_obs[i], next_global[i], float(dones[i]),
            )

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Uniform random sample.

        Returns dict với keys:
            obs, global_obs, actions, rewards,
            next_obs, next_global, dones
        All shapes: [batch_size, ...]
        """
        assert self.size >= batch_size, (
            f"Buffer has {self.size} < batch_size={batch_size}"
        )
        idx = np.random.randint(0, self.size, size=batch_size)

        return {
            "obs":         self.obs[idx],          # [B, A, obs_dim]
            "global_obs":  self.global_obs[idx],   # [B, global_obs_dim]
            "actions":     self.actions[idx],       # [B, A, action_dim]
            "rewards":     self.rewards[idx],       # [B, A]
            "next_obs":    self.next_obs[idx],      # [B, A, obs_dim]
            "next_global": self.next_global[idx],   # [B, global_obs_dim]
            "dones":       self.dones[idx],         # [B]
        }

    def __len__(self) -> int:
        return self.size

    def is_ready(self, min_size: int) -> bool:
        return self.size >= min_size

    def get_stats(self) -> Dict[str, float]:
        return {
            "size":          self.size,
            "capacity":      self.capacity,
            "fill_ratio":    self.size / self.capacity,
            "ptr":           self.ptr,
        }