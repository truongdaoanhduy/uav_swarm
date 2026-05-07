"""
training/algorithms/mappo/actor.py

Actor network for MAPPO — Hybrid Action Space v2.0

Action space:
    [vx, vy, vz]  — Continuous ∈ [-1, 1]³  (movement)
    [land]        — Discrete  ∈ {0, 1}      (landing signal)

Architecture:
    obs (68-dim)
        → Shared Backbone MLP (68 → 256 → 256)
        → Movement Head → mean[3], log_std[3] (Gaussian)
        → Landing Head  → logit[1]            (Bernoulli)
"""

import torch
import torch.nn as nn
from typing import Tuple, List

from .networks import MLP


class ActorNetwork(nn.Module):
    """
    Actor network with hybrid continuous + discrete action space.

    Output: action[4] = [vx, vy, vz, land]
        - [vx, vy, vz] ~ Normal(mean, std)   — movement
        - [land]       ~ Bernoulli(sigmoid(logit)) — land signal
    """

    def __init__(
        self,
        obs_dim:        int   = 68,
        action_dim:     int   = 4,       # ✅ 3 movement + 1 land
        hidden_dims:    tuple = (256, 256),
        activation:     str   = "tanh",
        use_layer_norm: bool  = False,
        log_std_init:   float = 0.0,
    ):
        super().__init__()

        self.obs_dim    = obs_dim
        self.action_dim = action_dim     # = 4
        self.move_dim   = action_dim - 1 # = 3

        # ── Shared backbone ──────────────────────────────────────────────────
        # Cả movement và landing share feature extractor
        # Output: feature vector [batch, hidden_dims[-1]]
        self.backbone = MLP(
            input_dim         = obs_dim,
            hidden_dims       = list(hidden_dims),
            output_dim        = hidden_dims[-1],
            activation        = activation,
            use_layer_norm    = use_layer_norm,
            output_activation = activation,   # ✅ Activate backbone output
        )

        feat_dim = hidden_dims[-1]  # = 256

        # ── Movement head (continuous Gaussian) ──────────────────────────────
        self.movement_head = nn.Linear(feat_dim, self.move_dim)
        nn.init.orthogonal_(self.movement_head.weight, gain=0.01)
        nn.init.zeros_(self.movement_head.bias)

        # Learnable log_std (state-independent, same as v1.0)
        self.log_std = nn.Parameter(
            torch.ones(self.move_dim) * log_std_init
        )

        # ── Landing head (discrete Bernoulli) ────────────────────────────────
        self.land_head = nn.Linear(feat_dim, 1)
        nn.init.orthogonal_(self.land_head.weight, gain=0.01)
        # ✅ Bias = -2.0 → P(land) ≈ 0.12 ban đầu
        # Tránh random landing làm noise training
        nn.init.constant_(self.land_head.bias, -1.0)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, obs: torch.Tensor):
        """
        Forward pass.

        Args:
            obs: [batch, 68]

        Returns:
            move_mean:  [batch, 3]
            move_std:   [batch, 3]
            land_logit: [batch, 1]
        """
        features   = self.backbone(obs)                          # [batch, 256]
        move_mean  = self.movement_head(features)                # [batch, 3]
        move_std   = self.log_std.exp().expand_as(move_mean)     # [batch, 3]
        land_logit = self.land_head(features)                    # [batch, 1]
        return move_mean, move_std, land_logit

    # ── Action sampling (ROLLOUT) ─────────────────────────────────────────────

    # Line 115-145 - Thay thế get_action()

    def get_action(
        self,
        obs:           torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action từ policy.
        
        ✅ P0 FIX: Bỏ clamp để log_prob chính xác
        Backend sẽ clip action trong apply_action()
        
        Args:
            obs:           [batch, 68]
            deterministic: True → greedy, False → sample
        
        Returns:
            action:   [batch, 4]  = [vx, vy, vz, land]
            log_prob: [batch]
        """
        move_mean, move_std, land_logit = self.forward(obs)

        # ── Movement ──────────────────────────────────────────────────────────
        move_dist = torch.distributions.Normal(move_mean, move_std)

        if deterministic:
            move_action = move_mean
        else:
            move_action = move_dist.rsample()

        # ✅ P0 FIX: BỎ CLAMP
        # Lý do: clamp làm sai log_prob → PPO ratio bias
        # Backend (logic_backend.py line 128) đã có clip:
        #   action = np.clip(action, -1.0, 1.0)
        # → Actor không cần clamp

        # DELETED: move_action = torch.clamp(move_action, -1.0, 1.0)

        # ── Landing ───────────────────────────────────────────────────────────
        land_dist = torch.distributions.Bernoulli(logits=land_logit)

        if deterministic:
            land_action = (land_logit > 0.0).float()
        else:
            land_action = land_dist.sample()

        # ── Log probability ───────────────────────────────────────────────────
        # ✅ Log prob ĐÚNG vì action chưa bị clamp
        move_log_prob = move_dist.log_prob(move_action).sum(dim=-1)   # [batch]
        land_log_prob = land_dist.log_prob(land_action).squeeze(-1)   # [batch]
        log_prob      = move_log_prob + land_log_prob

        # ── Concatenate → action[4] ───────────────────────────────────────────
        action = torch.cat([move_action, land_action], dim=-1)   # [batch, 4]

        return action, log_prob

    # ── Evaluate actions (PPO UPDATE) ─────────────────────────────────────────

    def evaluate_actions(
        self,
        obs:     torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log_prob + entropy của stored actions.
        Dùng trong PPO update để compute ratio.

        Args:
            obs:     [batch, 68]
            actions: [batch, 4]   — stored actions từ buffer

        Returns:
            log_prob: [batch]
            entropy:  [batch]
        """
        move_mean, move_std, land_logit = self.forward(obs)

        # Split stored actions
        move_actions = actions[:, :3]    # [batch, 3]
        land_actions = actions[:, 3:]    # [batch, 1]

        # Movement
        move_dist     = torch.distributions.Normal(move_mean, move_std)
        move_log_prob = move_dist.log_prob(move_actions).sum(dim=-1)   # [batch]
        move_entropy  = move_dist.entropy().sum(dim=-1)                 # [batch]

        # Landing
        land_dist     = torch.distributions.Bernoulli(logits=land_logit)
        land_log_prob = land_dist.log_prob(land_actions).squeeze(-1)    # [batch]
        land_entropy  = land_dist.entropy().squeeze(-1)                 # [batch]

        log_prob = move_log_prob + land_log_prob   # [batch]
        entropy  = move_entropy  + land_entropy    # [batch]

        return log_prob, entropy

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_land_prob(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Xác suất land tại state obs (dùng cho logging/viz).

        Returns: [batch] ∈ [0, 1]
        """
        _, _, land_logit = self.forward(obs)
        return torch.sigmoid(land_logit).squeeze(-1)

    def get_log_std(self) -> torch.Tensor:
        return self.log_std.detach()

    def set_log_std(self, value: float):
        with torch.no_grad():
            self.log_std.fill_(value)


# ═════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════

def test_actor_output_range(
    actor: ActorNetwork, 
    n_samples: int = 1000
) -> dict:
    """
    Test actor output statistics.
    
    Useful for debugging initialization.
    
    Args:
        actor: ActorNetwork instance
        n_samples: Number of samples to test
    
    Returns:
        Dict with statistics
    
    """
    actor.eval()
    
    with torch.no_grad():
        obs = torch.randn(n_samples, actor.obs_dim)
        mean, std = actor(obs)
        actions, _ = actor.get_action(obs)
    
    return {
        'mean_min': mean.min(dim=0).values.tolist(),
        'mean_max': mean.max(dim=0).values.tolist(),
        'mean_avg': mean.mean(dim=0).tolist(),
        'std_value': std[0].tolist(),  # Same for all samples
        'action_min': actions.min(dim=0).values.tolist(),
        'action_max': actions.max(dim=0).values.tolist(),
    }