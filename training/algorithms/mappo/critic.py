"""
training/algorithms/mappo/critic.py

Critic network for MAPPO (Centralized value function).

Architecture:
    global_obs (554-dim) → MLP → value (1-dim scalar)

Key Design:
    - Centralized training (uses global state)
    - Shared across all agents (single critic)
    - Outputs state value V(s), not Q(s,a)
    
CTDE Paradigm:
    - Actor: Decentralized (68-dim local obs)
    - Critic: Centralized (554-dim global obs)
    
    Benefits:
    1. Critic learns faster (sees full state)
    2. Better credit assignment (knows what all agents doing)
    3. Actor still executable with local obs only
"""

import torch
import torch.nn as nn
from typing import List

from .networks import MLP


class CriticNetwork(nn.Module):
    """
    Critic network for centralized value function.
    
    What is V(s)?
    ─────────────
    State value function: Expected return from state s
    
    V(s) = E[R_t | s_t = s]
         = E[r_t + γr_{t+1} + γ²r_{t+2} + ... | s_t = s]
    
    Critic learns to predict total future reward.
    Used to compute advantages: A = r + γV(s') - V(s)
    
    Why Centralized?
    ────────────────
    Uses global observation (554-dim):
    - All 8 UAVs observations (padded)
    - Global features (fleet health, coverage, etc.)
    
    Advantages:
    1. More accurate value estimates (sees full picture)
    2. Better credit assignment (knows team coordination)
    3. Faster learning (richer signal)
    
    Execution still decentralized:
    - Actor uses local obs only (68-dim)
    - Critic only used during training
    - Deploy: only need actor network
    
    Why Shared Critic?
    ──────────────────
    Single critic for all agents (not per-agent critics).
    
    Benefits:
    - Parameter efficiency (1 network vs 4)
    - Better generalization (learns team value)
    - Consistent value estimates
    
    Alternative (per-agent critics):
    - Each agent has own critic
    - More parameters, may overfit
    - Not standard in MAPPO
    """
    
    def __init__(
        self,
        global_obs_dim: int,
        hidden_dims: List[int] = [512, 256],
        activation: str = 'tanh',
        use_layer_norm: bool = False
    ):
        """
        Initialize Critic network.
        
        Args:
            global_obs_dim: Global observation dimension
                           - MAPPO: 554 (8 UAVs × 68 + 10 global)
            
        """
        super().__init__()
        
        self.global_obs_dim = global_obs_dim
        
        # ═════════════════════════════════════════════════════
        # VALUE NETWORK
        # ═════════════════════════════════════════════════════
        self.value_net = MLP(
            input_dim=global_obs_dim,
            hidden_dims=hidden_dims,
            output_dim=1,  # Single scalar output
            activation=activation,
            use_layer_norm=use_layer_norm,
            output_activation='none'  # No activation for value
        )
    
    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: global_obs → value.
        
        Args:
            global_obs: Global observation tensor
                       Shape: [batch_size, global_obs_dim]
        
        Returns:
            value: State value
                  Shape: [batch_size, 1]
                  Example: [256, 1]
                  
                  Note: Output is [batch, 1] not [batch]
                  Use get_value() for [batch] shape
        
        """
        return self.value_net(global_obs)
    
    def get_value(self, global_obs: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate (squeeze output).
        
        Convenience method that returns [batch] instead of [batch, 1].
        
        Args:
            global_obs: Global observation tensor [batch, global_obs_dim]
        
        Returns:
            value: State value [batch]
                  Squeezed from [batch, 1] → [batch]
        
        Example:
            >>> critic = CriticNetwork(554)
            >>> global_obs = torch.randn(256, 554)
            >>> value = critic.get_value(global_obs)
            >>> print(value.shape)
            torch.Size([256])
            
        Usage in Training:
            >>> # During rollout
            >>> values = critic.get_value(global_obs_batch)  # [2048]
            >>> 
            >>> # Compute GAE
            >>> advantages = rewards + gamma * values[1:] - values[:-1]
        """
        return self.forward(global_obs).squeeze(-1)  # [batch, 1] → [batch]
    
    def compute_loss(
        self, 
        global_obs: torch.Tensor, 
        returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute value loss (MSE between V(s) and returns).
        
        PPO value loss: L_V = MSE(V(s), returns)
        
        Returns (targets) are computed from GAE:
            returns = advantages + values
                    = (rewards + γV(s') - V(s)) + V(s)
                    = rewards + γV(s')
        
        Args:
            global_obs: Global observation [batch, global_obs_dim]
            returns: Target returns [batch]
                    Computed from GAE in buffer
        
        Returns:
            loss: MSE loss (scalar)
        
        """
        values = self.get_value(global_obs)  # [batch]
        
        # MSE loss
        loss = nn.functional.mse_loss(values, returns)
        
        return loss
    
    def compute_value_metrics(
        self, 
        global_obs: torch.Tensor, 
        returns: torch.Tensor
    ) -> dict:
        """
        Compute value function metrics for monitoring.
        
        Metrics:
        1. Value loss (MSE)
        2. Explained variance: How well V(s) predicts returns
           - 1.0: Perfect prediction
           - 0.0: No better than predicting mean
           - <0.0: Worse than mean
        
        Args:
            global_obs: Global observation [batch, global_obs_dim]
            returns: Target returns [batch]
        
        Returns:
            Dict with metrics:
            - 'value_loss': MSE loss
            - 'explained_variance': 1 - Var(returns - V) / Var(returns)
            - 'value_mean': Mean predicted value
            - 'value_std': Std of predicted values
            - 'returns_mean': Mean target returns
            - 'returns_std': Std of target returns
        """
        with torch.no_grad():
            values = self.get_value(global_obs)
            
            # Value loss
            value_loss = nn.functional.mse_loss(values, returns).item()
            
            # Explained variance
            # EV = 1 - Var(returns - values) / Var(returns)
            var_returns = returns.var()
            var_residual = (returns - values).var()
            
            if var_returns > 1e-8:
                explained_var = 1 - (var_residual / var_returns)
            else:
                explained_var = 0.0
            
            explained_var = explained_var.item()
        
        return {
            'value_loss': value_loss,
            'explained_variance': explained_var,
            'value_mean': values.mean().item(),
            'value_std': values.std().item(),
            'returns_mean': returns.mean().item(),
            'returns_std': returns.std().item(),
        }


# ═════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════

def test_critic_accuracy(
    critic: CriticNetwork,
    global_obs: torch.Tensor,
    true_returns: torch.Tensor,
    threshold: float = 0.7
) -> bool:
    """
    Test if critic has learned reasonably well.
    
    Checks if explained variance > threshold.
    
    Args:
        critic: CriticNetwork instance
        global_obs: Global observations [batch, 554]
        true_returns: True returns [batch]
        threshold: Minimum explained variance (default: 0.7)
    
    Returns:
        True if explained variance > threshold
    """
    metrics = critic.compute_value_metrics(global_obs, true_returns)
    ev = metrics['explained_variance']
    
    return ev > threshold


def initialize_critic_for_env(
    global_obs_dim: int,
    hidden_dims: List[int] = [512, 256],
    device: str = 'cpu'
) -> CriticNetwork:
    """
    Factory function to create critic for SAR environment.
    
    Args:
        global_obs_dim: Global observation dimension (554 for SAR)
        hidden_dims: Hidden layer dimensions
        device: Device to put network on
    
    Returns:
        Initialized CriticNetwork on specified device
    """
    critic = CriticNetwork(
        global_obs_dim=global_obs_dim,
        hidden_dims=hidden_dims,
        activation='tanh',
        use_layer_norm=False
    )
    
    critic = critic.to(device)
    
    return critic