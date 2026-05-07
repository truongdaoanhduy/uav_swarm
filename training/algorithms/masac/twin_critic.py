"""
MASAC Twin Q-Critic (Centralized)

Architecture:
    [global_obs (554), action (4)] → Q1, Q2

Twin critics:
    - Train Q1, Q2 independently
    - Use min(Q1, Q2) for actor update (conservative)
    - Reduces overestimation bias (Fujimoto et al., 2018)

Centralized:
    - Takes global_obs (554-dim) instead of local obs
    - Sees all agents' states + actions (CTDE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np


def _build_q_net(input_dim: int, hidden_dims: List[int]) -> nn.Sequential:
    """Build single Q network."""
    dims   = [input_dim] + list(hidden_dims) + [1]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)

    # Init output layer small
    nn.init.xavier_uniform_(net[-1].weight, gain=0.01)
    nn.init.zeros_(net[-1].bias)
    return net


class TwinCriticNetwork(nn.Module):
    """
    Twin Q-network cho MASAC.

    Input: concat(global_obs, action) — centralized
    Output: Q1(s,a), Q2(s,a)

    Training:
        target = r + γ × min(Q1', Q2') × (1 - done)
        loss   = MSE(Q1, target) + MSE(Q2, target)

    Actor update uses:
        Q = min(Q1, Q2)
    """

    def __init__(
        self,
        global_obs_dim: int,
        action_dim:     int   = 4,
        hidden_dims:    tuple = (400, 300),
    ):
        super().__init__()
        self.global_obs_dim = global_obs_dim
        self.action_dim     = action_dim

        sa_dim = global_obs_dim + action_dim  # 554 + 4 = 558

        self.q1 = _build_q_net(sa_dim, list(hidden_dims))
        self.q2 = _build_q_net(sa_dim, list(hidden_dims))

    def forward(
        self,
        global_obs: torch.Tensor,  # [batch, 554]
        actions:    torch.Tensor,  # [batch, 4]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            q1: [batch, 1]
            q2: [batch, 1]
        """
        sa = torch.cat([global_obs, actions], dim=-1)   # [batch, 558]
        return self.q1(sa), self.q2(sa)

    def q1_value(
        self,
        global_obs: torch.Tensor,
        actions:    torch.Tensor,
    ) -> torch.Tensor:
        """Q1 only — dùng cho actor update."""
        sa = torch.cat([global_obs, actions], dim=-1)
        return self.q1(sa)

    def min_q(
        self,
        global_obs: torch.Tensor,
        actions:    torch.Tensor,
    ) -> torch.Tensor:
        """min(Q1, Q2) — conservative estimate."""
        q1, q2 = self.forward(global_obs, actions)
        return torch.min(q1, q2)

    def compute_loss(
        self,
        global_obs:  torch.Tensor,   # [batch, 554]
        actions:     torch.Tensor,   # [batch, 4]
        targets:     torch.Tensor,   # [batch]  — TD targets
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q1 + Q2 losses.

        Returns:
            q1_loss: MSE(Q1, target)
            q2_loss: MSE(Q2, target)
        """
        q1, q2 = self.forward(global_obs, actions)
        targets = targets.unsqueeze(-1)   # [batch, 1]
        q1_loss = F.mse_loss(q1, targets)
        q2_loss = F.mse_loss(q2, targets)
        return q1_loss, q2_loss