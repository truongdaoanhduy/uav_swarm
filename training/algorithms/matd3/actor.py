"""
MATD3 Actor — Deterministic Policy

Architecture:
    obs (68-dim) → MLP → tanh → action[4]

Key differences from SAC/MAPPO:
    - DETERMINISTIC (no sampling during execution)
    - tanh output squashing: action ∈ [-1,1]
    - Exploration via Gaussian noise (not entropy)
    - Target actor: smoothed with clipped noise (TD3 trick)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


class TD3ActorNetwork(nn.Module):
    """
    Deterministic Actor for TD3.

    Policy: a = μ(s) = tanh(MLP(s))

    Exploration:
        Training:   a + N(0, σ_explore)
        Evaluation: a (no noise)

    Target smoothing:
        a_target = tanh(μ(s') + clip(N(0, σ_smooth), -c, c))
        Prevents Q overfit to narrow action peaks
    """

    def __init__(
        self,
        obs_dim:     int   = 68,
        action_dim:  int   = 4,
        hidden_dims: tuple = (256, 256),
        activation:  str   = "relu",
    ):
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.move_dim   = action_dim - 1   # = 3

        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}[activation]

        # ── Backbone ────────────────────────────────────────────────────
        dims = [obs_dim] + list(hidden_dims)
        backbone_layers = []
        for i in range(len(dims) - 1):
            backbone_layers.append(nn.Linear(dims[i], dims[i + 1]))
            backbone_layers.append(act_fn())
        self.backbone = nn.Sequential(*backbone_layers)

        feat_dim = hidden_dims[-1]

        # ── Movement head: deterministic ───────────────────────────────
        self.move_head = nn.Linear(feat_dim, self.move_dim)
        nn.init.xavier_uniform_(self.move_head.weight, gain=0.01)
        nn.init.zeros_(self.move_head.bias)

        # ── Landing head ──────────────────────────────────────────────
        self.land_head = nn.Linear(feat_dim, 1)
        nn.init.xavier_uniform_(self.land_head.weight, gain=0.01)
        nn.init.constant_(self.land_head.bias, -2.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Deterministic forward pass.

        Args:
            obs: [batch, 68]

        Returns:
            action: [batch, 4] — [tanh(vx,vy,vz), sigmoid(land)]
        """
        feat        = self.backbone(obs)
        move_action = torch.tanh(self.move_head(feat))   # [batch, 3] ∈ [-1,1]
        land_logit  = self.land_head(feat)               # [batch, 1]
        land_action = torch.sigmoid(land_logit)          # [batch, 1] ∈ [0,1]
        return torch.cat([move_action, land_action], dim=-1)  # [batch, 4]

    def get_action(
        self,
        obs:              torch.Tensor,
        explore_noise:    float = 0.1,       # σ for exploration noise
        noise_clip:       float = 0.5,       # Clip range for noise
        deterministic:    bool  = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action with optional exploration noise.

        Training:   action + N(0, explore_noise), clipped
        Evaluation: action (no noise)

        Returns:
            action:   [batch, 4]
            log_prob: [batch]  — zeros (deterministic, no log_prob)
        """
        action = self.forward(obs)   # [batch, 4]

        if not deterministic and explore_noise > 0:
            # Gaussian noise for exploration
            noise = torch.randn_like(action[:, :3]) * explore_noise
            noise = torch.clamp(noise, -noise_clip, noise_clip)
            action = action.clone()
            action[:, :3] = torch.clamp(
                action[:, :3] + noise, -1.0, 1.0
            )
            # Binarize land with noise
            land_noise    = torch.randn_like(action[:, 3:]) * explore_noise * 0.3
            action[:, 3:] = (action[:, 3:] + land_noise > 0.5).float()
        else:
            # Binarize land (deterministic threshold)
            action = action.clone()
            action[:, 3:] = (action[:, 3:] > 0.5).float()

        # No log_prob for deterministic policy
        log_prob = torch.zeros(action.shape[0], device=obs.device)
        return action, log_prob

    def get_target_action(
        self,
        obs:            torch.Tensor,
        target_noise:   float = 0.2,    # σ for target policy smoothing
        noise_clip:     float = 0.5,    # TD3: clip(N(0,σ), -c, c)
    ) -> torch.Tensor:
        """
        Target policy with smoothing noise (TD3 trick).

        Prevents Q-function overfit to narrow action peaks.

        a' = clip(μ(s') + clip(N(0, target_noise), -c, c), -1, 1)

        Returns:
            action: [batch, 4]
        """
        with torch.no_grad():
            action = self.forward(obs)   # [batch, 4]

            noise = torch.randn_like(action[:, :3]) * target_noise
            noise = torch.clamp(noise, -noise_clip, noise_clip)

            smooth_action = action.clone()
            smooth_action[:, :3] = torch.clamp(
                action[:, :3] + noise, -1.0, 1.0
            )
            # Keep land head output continuous for smoothing
            # (binarize at inference only)
            smooth_action[:, 3:] = torch.clamp(
                action[:, 3:] + torch.randn_like(action[:, 3:]) * target_noise * 0.3,
                0.0, 1.0
            )
        return smooth_action

    def get_land_prob(self, obs: torch.Tensor) -> torch.Tensor:
        """[batch] ∈ [0,1]"""
        feat = self.backbone(obs)
        return torch.sigmoid(self.land_head(feat)).squeeze(-1)