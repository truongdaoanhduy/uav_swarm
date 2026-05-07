"""
MATD3 Twin Q-Critic (Centralized)

Same architecture as MASAC twin critic.
Used for TD3's clipped double Q-learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np


def _build_q_net(input_dim: int, hidden_dims: List[int]) -> nn.Sequential:
    dims   = [input_dim] + list(hidden_dims) + [1]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    nn.init.xavier_uniform_(net[-1].weight, gain=0.01)
    nn.init.zeros_(net[-1].bias)
    return net


class TD3TwinCriticNetwork(nn.Module):
    """
    Twin Q-networks cho MATD3.

    Key TD3 differences:
        - Use min(Q1, Q2) for TARGET computation only
        - Update actor using Q1 only (not min)
        - Delayed actor updates (every d steps)
    """

    def __init__(
        self,
        global_obs_dim: int,
        action_dim:     int   = 4,
        hidden_dims:    tuple = (400, 300),
    ):
        super().__init__()
        sa_dim  = global_obs_dim + action_dim
        self.q1 = _build_q_net(sa_dim, list(hidden_dims))
        self.q2 = _build_q_net(sa_dim, list(hidden_dims))

    def forward(
        self,
        global_obs: torch.Tensor,
        actions:    torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([global_obs, actions], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_value(self, global_obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Q1 only — dùng cho actor update."""
        sa = torch.cat([global_obs, actions], dim=-1)
        return self.q1(sa)

    def min_q(self, global_obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """min(Q1, Q2) — dùng cho TARGET computation."""
        q1, q2 = self.forward(global_obs, actions)
        return torch.min(q1, q2)

    def compute_loss(
        self,
        global_obs: torch.Tensor,
        actions:    torch.Tensor,
        targets:    torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q1, q2    = self.forward(global_obs, actions)
        targets_u = targets.unsqueeze(-1)
        return F.mse_loss(q1, targets_u), F.mse_loss(q2, targets_u)