"""
training/algorithms/mappo/actor.py

Actor network for MAPPO (Gaussian policy).

Architecture:
    obs (68-dim) → MLP → mean (3-dim)
                       ↘ log_std (3-dim, learnable parameter)
    
    Action sampling: a ~ N(mean, exp(log_std))

Key Design:
    - Decentralized execution (only uses local obs)
    - Shared weights across agents (parameter sharing)
    - Gaussian distribution (continuous action space)
    - log_std as learnable parameter (state-independent)
"""

import torch
import torch.nn as nn
from typing import Tuple, List

from .networks import MLP


class ActorNetwork(nn.Module):
    """
    Actor network for continuous action space.
    
    Outputs mean and std for Gaussian policy.
    
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = 'tanh',
        use_layer_norm: bool = False,
        log_std_init: float = 0.0
    ):
        """
        Initialize Actor network.
        
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # ═════════════════════════════════════════════════════
        # MEAN NETWORK
        # ═════════════════════════════════════════════════════
        self.mean_net = MLP(
            input_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
            output_activation='none'  # No activation for mean
        )
        
        # ═════════════════════════════════════════════════════
        # LOG STD (Learnable Parameter)
        # ═════════════════════════════════════════════════════
        # Shape: (action_dim,) = (3,)
        # Will be broadcast to batch size during forward pass
        self.log_std = nn.Parameter(
            torch.ones(action_dim) * log_std_init
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: obs → (mean, std).
        
        Args:
            obs: Observation tensor
                Shape: [batch_size, obs_dim]
        
        Returns:
            mean: Action mean
                 Shape: [batch_size, action_dim]
            
            std: Action standard deviation
                Shape: [batch_size, action_dim]
                
        """
        # Compute mean from network
        mean = self.mean_net(obs)  # [batch, 3]
        
        # Compute std from log_std parameter
        std = torch.exp(self.log_std)  # [3]
        std = std.expand_as(mean)       # [3] → [batch, 3]
        
        return mean, std
    
    def get_action(
        self, 
        obs: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Used during ROLLOUT (experience collection).
        
        Args:
            obs: Observation tensor [batch, obs_dim]
            deterministic: If True, return mean (no sampling)
                          - True: Evaluation mode
                          - False: Training mode (exploration)
        
        Returns:
            action: Sampled action [batch, action_dim]
                   - If deterministic: action = mean
                   - If stochastic: action ~ N(mean, std)
            
            log_prob: Log probability of action [batch]
                     - If deterministic: log_prob = 0 (delta dist)
                     - If stochastic: log_prob = log π(a|s)
        
        """
        mean, std = self.forward(obs)
        
        if deterministic:
            # Evaluation mode: return mean (no randomness)
            action = mean
            
            # Log prob for delta distribution (undefined, set to 0)
            log_prob = torch.zeros(mean.shape[0], device=mean.device)
        else:
            # Training mode: sample from Gaussian
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            
            # Compute log probability (sum over action dimensions)
            # log π(a|s) = Σ_i log N(a_i | μ_i, σ_i)
            log_prob = dist.log_prob(action).sum(dim=-1)  # [batch, 3] → [batch]
        
        return action, log_prob
    
    def evaluate_actions(
        self, 
        obs: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of given actions.
        
        Used during PPO UPDATE (compute policy loss).
        
        This function answers:
        1. What is log π_new(a|s) for old actions a?
           → For computing policy ratio r = π_new / π_old
        
        2. What is entropy of current policy?
           → For entropy regularization
        
        Args:
            obs: Observation tensor [batch, obs_dim]
                Example: [256, 68] (minibatch from buffer)
            
            actions: Action tensor [batch, action_dim]
                    OLD actions from buffer (sampled earlier)
                    Example: [256, 3]
        
        Returns:
            log_prob: Log probability of actions under CURRENT policy
                     Shape: [batch]
                     Example: [256]
                     
                     Used in PPO ratio:
                     ratio = exp(log_prob_new - log_prob_old)
            
            entropy: Entropy of current policy distribution
                    Shape: [batch]                    
                    
                    Gaussian entropy: H = 0.5 * log(2πe * σ²)
                                       = 0.5 * (log(2πe) + 2*log(σ))
        
        """
        mean, std = self.forward(obs)
        
        # Create Gaussian distribution
        dist = torch.distributions.Normal(mean, std)
        
        # Compute log probability of given actions
        # Sum over action dimensions (independent Gaussians)
        log_prob = dist.log_prob(actions).sum(dim=-1)  # [batch, 3] → [batch]
        
        # Compute entropy
        # Sum over action dimensions (entropy is additive for independent vars)
        entropy = dist.entropy().sum(dim=-1)  # [batch, 3] → [batch]
        
        return log_prob, entropy
    
    def get_log_std(self) -> torch.Tensor:
        """
        Get current log_std parameter.
        
        Useful for monitoring exploration during training.
        
        Returns:
            log_std: Current log_std values [action_dim]
        """
        return self.log_std
    
    def set_log_std(self, log_std: float):
        """
        Set log_std to specific value.
        
        Useful for:
        - Annealing exploration (decrease std over time)
        - Switching between exploration/exploitation
        
        Args:
            log_std: New log_std value (same for all action dims)
        """
        with torch.no_grad():
            self.log_std.fill_(log_std)


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