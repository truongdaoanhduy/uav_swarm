"""
MASAC Actor — Squashed Gaussian Policy

Architecture:
    obs (68-dim) → MLP → mean[4], log_std[4]
    [vx, vy, vz] ~ TanhNormal (squashed Gaussian) ∈ [-1,1]³
    [land]       ~ Bernoulli(sigmoid(logit))       ∈ {0,1}

Key differences from MAPPO actor:
    - Squashed Gaussian (tanh) for movement → action ∈ [-1,1]
    - log_prob corrected for tanh squashing
    - Reparameterization trick (rsample) for gradient flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np


def _mlp(dims: List[int], activation=nn.ReLU, output_activation=nn.Identity):
    """Build MLP từ list dims."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
        else:
            layers.append(output_activation())
    return nn.Sequential(*layers)


class SACActorNetwork(nn.Module):
    """
    SAC Actor với Squashed Gaussian + Bernoulli hybrid action.

    Action space:
        [vx, vy, vz] ~ TanhNormal   — continuous, ∈ [-1,1]
        [land]       ~ Bernoulli    — discrete,   ∈ {0,1}

    SAC-specific:
        - Entropy maximization objective
        - Reparameterization for gradient
        - log_prob correction: tanh squashing
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        obs_dim:        int   = 68,
        action_dim:     int   = 4,
        hidden_dims:    tuple = (256, 256),
        activation:     str   = "relu",
        log_std_init:   float = 0.0,
    ):
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.move_dim   = action_dim - 1  # = 3

        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}[activation]

        # ── Shared backbone ──────────────────────────────────────────────
        dims = [obs_dim] + list(hidden_dims)
        backbone_layers = []
        for i in range(len(dims) - 1):
            backbone_layers.append(nn.Linear(dims[i], dims[i + 1]))
            backbone_layers.append(act_fn())
        self.backbone = nn.Sequential(*backbone_layers)

        feat_dim = hidden_dims[-1]  # 256

        # ── Movement head: mean + log_std ────────────────────────────────
        self.move_mean_head    = nn.Linear(feat_dim, self.move_dim)
        self.move_log_std_head = nn.Linear(feat_dim, self.move_dim)

        # ── Landing head ─────────────────────────────────────────────────
        self.land_head = nn.Linear(feat_dim, 1)

        # Init
        for head in [self.move_mean_head, self.move_log_std_head, self.land_head]:
            nn.init.xavier_uniform_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

        # Land bias → P(land) ≈ 0.12 initially
        nn.init.constant_(self.land_head.bias, -2.0)

    def forward(self, obs: torch.Tensor):
        """
        Returns:
            move_mean:     [batch, 3]
            move_log_std:  [batch, 3]  clipped
            land_logit:    [batch, 1]
        """
        feat         = self.backbone(obs)
        move_mean    = self.move_mean_head(feat)
        move_log_std = self.move_log_std_head(feat)
        move_log_std = torch.clamp(move_log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        land_logit   = self.land_head(feat)
        return move_mean, move_log_std, land_logit

    def get_action(
        self,
        obs:           torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action.

        Returns:
            action:   [batch, 4]  — [vx, vy, vz, land] ∈ [-1,1]³ × {0,1}
            log_prob: [batch]     — corrected for tanh squashing
        """
        move_mean, move_log_std, land_logit = self.forward(obs)
        move_std = move_log_std.exp()

        dist = torch.distributions.Normal(move_mean, move_std)

        if deterministic:
            # Greedy: mean → tanh squash
            u           = move_mean
            move_action = torch.tanh(u)
            # Log prob deterministic = 0 (no randomness)
            move_log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            # Sample from Normal, then squash
            u             = dist.rsample()                          # [batch, 3]
            move_action   = torch.tanh(u)                          # ∈ [-1, 1]

            # ✅ SAC log_prob correction for tanh squashing:
            # log π(a|s) = log N(u|μ,σ) - Σ log(1 - tanh²(u))
            # = log N(u|μ,σ) - Σ log(1 - a²)  (numerically stable)
            move_log_prob = dist.log_prob(u).sum(dim=-1)            # [batch]
            # Jacobian correction
            squash_corr   = torch.log(1 - move_action.pow(2) + 1e-6).sum(dim=-1)
            move_log_prob = move_log_prob - squash_corr             # [batch]

        # Landing: Bernoulli
        land_dist     = torch.distributions.Bernoulli(logits=land_logit)
        if deterministic:
            land_action = (land_logit > 0.0).float()
        else:
            land_action = land_dist.sample()                        # [batch, 1]

        land_log_prob = land_dist.log_prob(land_action).squeeze(-1) # [batch]
        log_prob      = move_log_prob + land_log_prob               # [batch]

        action = torch.cat([move_action, land_action], dim=-1)      # [batch, 4]
        return action, log_prob

    def get_log_prob(
        self,
        obs:     torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log_prob + entropy của given actions.
        Dùng trong SAC actor update.

        Args:
            obs:     [batch, 68]
            actions: [batch, 4]  — stored actions (already squashed)

        Returns:
            log_prob: [batch]
            entropy:  [batch]  = -log_prob
        """
        move_mean, move_log_std, land_logit = self.forward(obs)
        move_std = move_log_std.exp()

        # Un-squash movement (atanh)
        move_actions = actions[:, :3]   # [batch, 3], already in [-1,1]
        land_actions = actions[:, 3:]   # [batch, 1]

        # Clamp để tránh atanh(±1) = ±∞
        move_actions_clamped = move_actions.clamp(-1 + 1e-6, 1 - 1e-6)
        u = torch.atanh(move_actions_clamped)   # [batch, 3]

        dist = torch.distributions.Normal(move_mean, move_std)
        move_log_prob = dist.log_prob(u).sum(dim=-1)
        squash_corr   = torch.log(1 - move_actions_clamped.pow(2) + 1e-6).sum(dim=-1)
        move_log_prob = move_log_prob - squash_corr

        land_dist     = torch.distributions.Bernoulli(logits=land_logit)
        land_log_prob = land_dist.log_prob(land_actions).squeeze(-1)

        log_prob = move_log_prob + land_log_prob  # [batch]
        entropy  = -log_prob                       # [batch]

        return log_prob, entropy

    def get_land_prob(self, obs: torch.Tensor) -> torch.Tensor:
        """[batch] ∈ [0,1]"""
        _, _, land_logit = self.forward(obs)
        return torch.sigmoid(land_logit).squeeze(-1)