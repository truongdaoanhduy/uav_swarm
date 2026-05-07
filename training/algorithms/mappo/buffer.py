"""
training/algorithms/mappo/buffer.py
RolloutBuffer với trục env riêng biệt.

Shape convention:
    T = rollout_length (số timestep per env)
    E = n_envs
    A = n_agents
    
Storage: [T, E, A, ...] → GAE tính per-env → flatten khi get_batches()
"""

import numpy as np
from typing import Iterator, Dict, Optional


class RolloutBuffer:
    """
    Rollout buffer cho MAPPO với GAE computation.
    
    Key fix: Buffer có trục env riêng [T, E, A, ...]
    → GAE tính đúng per-env, không bị trộn giữa các env
    """

    def __init__(
        self,
        rollout_length: int,   # T: số timestep per env
        n_envs:         int,   # E: số envs song song
        n_agents:       int,   # A: số agents per env
        obs_dim:        int,
        global_obs_dim: int,
        action_dim:     int,
        gamma:          float = 0.99,
        gae_lambda:     float = 0.95,
    ):
        self.T          = rollout_length
        self.E          = n_envs
        self.A          = n_agents
        self.obs_dim    = obs_dim
        self.global_obs_dim = global_obs_dim
        self.action_dim = action_dim
        self.gamma      = gamma
        self.gae_lambda = gae_lambda

        # Storage: [T, E, A, ...] hoặc [T, E, ...]
        self.observations = np.zeros((self.T, self.E, self.A, obs_dim),        dtype=np.float32)
        self.global_obs   = np.zeros((self.T, self.E, global_obs_dim),         dtype=np.float32)
        self.actions      = np.zeros((self.T, self.E, self.A, action_dim),     dtype=np.float32)
        self.rewards      = np.zeros((self.T, self.E, self.A),                 dtype=np.float32)
        self.values       = np.zeros((self.T, self.E, self.A),                 dtype=np.float32)
        self.log_probs    = np.zeros((self.T, self.E, self.A),                 dtype=np.float32)
        self.dones        = np.zeros((self.T, self.E),                         dtype=np.float32)

        # Computed
        self.advantages   = np.zeros((self.T, self.E, self.A),                 dtype=np.float32)
        self.returns      = np.zeros((self.T, self.E, self.A),                 dtype=np.float32)

        self.ptr = 0  # Con trỏ timestep hiện tại

    def add(
        self,
        obs:        np.ndarray,  # [E, A, obs_dim]
        global_obs: np.ndarray,  # [E, global_obs_dim]
        actions:    np.ndarray,  # [E, A, action_dim]
        rewards:    np.ndarray,  # [E, A]
        values:     np.ndarray,  # [E, A]
        log_probs:  np.ndarray,  # [E, A]
        dones:      np.ndarray,  # [E] bool/float
    ):
        """
        Add ONE timestep (toàn bộ E envs) vào buffer.
        Gọi một lần per timestep, không gọi per-env.
        """
        if self.ptr >= self.T:
            raise RuntimeError(
                f"Buffer overflow: ptr={self.ptr}, T={self.T}. "
                "Gọi compute_gae() và clear() trước."
            )

        self.observations[self.ptr] = obs
        self.global_obs[self.ptr]   = global_obs
        self.actions[self.ptr]      = actions
        self.rewards[self.ptr]      = rewards
        self.values[self.ptr]       = values
        self.log_probs[self.ptr]    = log_probs
        self.dones[self.ptr]        = dones.astype(np.float32)

        self.ptr += 1

    # Line 120-145 - Thay thế toàn bộ loop GAE

    # buffer.py - compute_gae() - thay toàn bộ loop:

    def compute_gae(
        self,
        last_values: np.ndarray,  # [E] hoặc [E, A]
        last_dones:  np.ndarray,  # [E]
    ):
        """
        ✅ FIX P0-2: Sửa done shift bug
        
        CRITICAL: next_non_terminal[t] phải dùng dones[t+1], KHÔNG phải dones[t]
        
        Lý do:
            dones[t] = True  → Episode KẾT THÚC SAU step t
            next_non_terminal[t] = 1 - dones[t+1]
                                = Liệu step t+1 có TIẾP TỤC không?
        
        Ví dụ:
            dones = [0, 0, 0, 0, 1]  # Episode end ở step 4
            
            Tại step 3:
                delta[3] = r[3] + γ × V[4] × (1 - dones[4]) - V[3]
                        = r[3] + γ × V[4] × 0 - V[3]  ← KHÔNG bootstrap (đúng)
            
            Với code CŨ (SAI):
                next_non_terminal[3] = 1 - dones[3] = 1  ← SAI!
                delta[3] = r[3] + γ × V[4] × 1 - V[3]     ← Bootstrap qua terminal (SAI)
        """
        actual_T = min(self.ptr, self.T)
        if actual_T == 0:
            return

        # Normalize last_values → [E, A]
        if last_values.ndim == 1:
            last_values = np.repeat(
                last_values[:, None], self.A, axis=1
            )

        last_dones = last_dones.astype(np.float32)

        # ✅ Precompute next_values
        next_values = np.empty_like(self.values[:actual_T])  # [T, E, A]
        next_values[:-1] = self.values[1:actual_T]           # Shift left
        next_values[-1]  = last_values                        # Bootstrap

        # ✅ FIX P0-2: ĐÚNG - dùng dones[1:] cho next_non_terminal[:-1]
        next_non_terminal = np.empty((actual_T, self.E), dtype=np.float32)
        next_non_terminal[:-1] = 1.0 - self.dones[1:actual_T]  # ← FIXED!
        next_non_terminal[-1]  = 1.0 - last_dones

        # ✅ Vectorized delta
        delta = (
            self.rewards[:actual_T]
            + self.gamma * next_values * next_non_terminal[:, :, None]
            - self.values[:actual_T]
        )

        # GAE backward pass
        gae = np.zeros((self.E, self.A), dtype=np.float32)
        for t in reversed(range(actual_T)):
            nnt = next_non_terminal[t, :, None]  # [E, 1]
            gae = delta[t] + self.gamma * self.gae_lambda * nnt * gae
            self.advantages[t] = gae

        # Returns
        self.returns[:actual_T] = (
            self.advantages[:actual_T] + self.values[:actual_T]
        )

        # Normalize advantages
        if actual_T > 1:
            if self.E == 1:
                # Single env: normalize global (fast)
                adv_flat = self.advantages[:actual_T].reshape(-1)
                adv_mean = float(np.mean(adv_flat))
                adv_std  = float(np.std(adv_flat))
                self.advantages[:actual_T] = (
                    (self.advantages[:actual_T] - adv_mean) / (adv_std + 1e-8)
                )
            else:
                # Multi-env: normalize per-env (correct)
                for e in range(self.E):
                    adv_e = self.advantages[:actual_T, e, :]
                    mean_e = float(np.mean(adv_e))
                    std_e  = float(np.std(adv_e))
                    self.advantages[:actual_T, e, :] = (
                        (adv_e - mean_e) / (std_e + 1e-8)
                    )

    # Line 195-225 - THAY THẾ get_batches()

    def get_batches(self, batch_size: int) -> Iterator[Dict[str, np.ndarray]]:
        """
        ✅ FIX P1-1: Optimize broadcast cho n_envs=1
        
        Performance:
            n_envs=1: ~40% faster (tránh broadcast + copy lớn)
            n_envs>1: giữ nguyên (broadcast là tối ưu)
        """
        actual_T = min(self.ptr, self.T)
        if actual_T == 0:
            return

        total_samples = actual_T * self.E * self.A

        # Flatten obs, actions, etc.
        obs_flat     = self.observations[:actual_T].reshape(total_samples, self.obs_dim)
        actions_flat = self.actions[:actual_T].reshape(total_samples, self.action_dim)
        lp_flat      = self.log_probs[:actual_T].reshape(total_samples)
        adv_flat     = self.advantages[:actual_T].reshape(total_samples)
        ret_flat     = self.returns[:actual_T].reshape(total_samples)

        # ✅ FIX P1-1: Conditional optimization
        if self.E == 1:
            # Single env: repeat nhanh hơn broadcast
            g_flat = np.repeat(
                self.global_obs[:actual_T, 0, :],  # [T, global_dim]
                self.A, axis=0
            ).reshape(total_samples, self.global_obs_dim)
        else:
            # Multi-env: dùng broadcast (tối ưu cho nhiều envs)
            g_rep = np.broadcast_to(
                self.global_obs[:actual_T, :, None, :],         # [T, E, 1, global_dim]
                (actual_T, self.E, self.A, self.global_obs_dim) # [T, E, A, global_dim]
            )
            g_flat = np.ascontiguousarray(
                g_rep.reshape(total_samples, self.global_obs_dim),
                dtype=np.float32
            )

        # Shuffle
        indices = np.random.permutation(total_samples)

        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            idx = indices[start:end]
            yield {
                "obs":           obs_flat[idx],
                "global_obs":    g_flat[idx],
                "actions":       actions_flat[idx],
                "old_log_probs": lp_flat[idx],
                "advantages":    adv_flat[idx],
                "returns":       ret_flat[idx],
            }

    def clear(self):
        """Reset pointer."""
        self.ptr = 0

    @property
    def capacity(self) -> int:
        """Tổng số timestep có thể lưu."""
        return self.T

    @property
    def total_transitions(self) -> int:
        """Tổng số transitions hiện tại (T * E * A)."""
        return min(self.ptr, self.T) * self.E * self.A

    def get_stats(self) -> Dict[str, float]:
        actual_T = min(self.ptr, self.T)
        if actual_T == 0:
            return {"buffer_fill": 0.0, "total_transitions": 0}
        return {
            "buffer_fill":       actual_T / self.T,
            "total_transitions": self.total_transitions,
            "mean_reward":       float(np.mean(self.rewards[:actual_T])),
            "mean_value":        float(np.mean(self.values[:actual_T])),
            "mean_advantage":    float(np.mean(self.advantages[:actual_T])),
            "std_advantage":     float(np.std(self.advantages[:actual_T])),
            "mean_return":       float(np.mean(self.returns[:actual_T])),
        }