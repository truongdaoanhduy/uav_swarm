"""
training/algorithms/mappo/networks.py

Shared neural network architectures for MAPPO.

Components:
    - orthogonal_init(): Initialization function (PPO best practice)
    - MLP: Flexible multi-layer perceptron
    
Design Philosophy:
    - Orthogonal init → stable gradient flow
    - Configurable architecture → easy ablation
    - Small output init (gain=0.01) → exploration-friendly
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List


def orthogonal_init(layer: nn.Linear, gain: float = np.sqrt(2)) -> nn.Linear:
    """
    Orthogonal initialization for linear layers.
    
    WHY orthogonal instead of Kaiming/Xavier?
    ──────────────────────────────────────────
    1. Gradient preservation: Orthogonal matrices preserve gradient magnitude
       → Prevents vanishing/exploding gradients
    
    2. PPO empirical best practice: Papers show better performance
       (Schulman et al., 2017; Engstrom et al., 2020)
    
    3. Faster convergence: Network learns useful features faster
    
    GAIN parameter:
    ───────────────
    - sqrt(2) for hidden layers: Standard for ReLU/Tanh
    - 0.01 for output layer: Small initial policy (near zero actions)
      → Encourages exploration at start
    
    Args:
        layer: Linear layer to initialize
        gain: Scaling factor
              - np.sqrt(2) for hidden layers
              - 0.01 for output layer (policy starts near 0)
    
    Returns:
        Initialized layer (in-place modification)
    
    Example:
        >>> layer = nn.Linear(256, 256)
        >>> orthogonal_init(layer, gain=np.sqrt(2))
        >>> # Weights are now orthogonal matrix scaled by sqrt(2)
    """
    # Orthogonal initialization for weights
    nn.init.orthogonal_(layer.weight, gain=gain)
    
    # Bias initialized to zero (standard practice)
    nn.init.constant_(layer.bias, 0.0)
    
    return layer


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Fully Connected Network).
    
    Flexible architecture:
    ──────────────────────
    - Configurable hidden dimensions
    - Multiple activation functions
    - Optional layer normalization
    - Orthogonal initialization
    
    
    Architecture Examples:
    ──────────────────────
    Actor (68 → 256 → 256 → 3):
        mlp = MLP(68, [256, 256], 3, activation='tanh')
    
    Critic (554 → 512 → 256 → 1):
        mlp = MLP(554, [512, 256], 1, activation='tanh')
    
    Small network (faster):
        mlp = MLP(68, [128, 128], 3)
    
    Deep network (more expressive):
        mlp = MLP(68, [256, 256, 128], 3)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = 'tanh',
        use_layer_norm: bool = False,
        output_activation: str = 'none'
    ):
        """
        Initialize MLP.
        
        Args:
            input_dim: Input feature dimension
                      - Actor: 68 (local obs)
                      - Critic: 554 (global obs)
            
            hidden_dims: List of hidden layer dimensions
                        Example: [256, 256] → 2 hidden layers
                        Example: [512, 256, 128] → 3 layers
            
            output_dim: Output dimension
                       - Actor: 3 (action mean)
                       - Critic: 1 (value)
            
            activation: Activation function for hidden layers
                       - 'tanh': Smooth, bounded [-1,1] (RECOMMENDED)
                       - 'relu': Faster, unbounded (may cause dead neurons)
                       - 'elu': Smooth ReLU variant
            
            use_layer_norm: Whether to use layer normalization
                           - False: Standard MLP (RECOMMENDED)
                           - True: Normalize activations (advanced)
                           
                           Enable if:
                           - Training is unstable
                           - Network is very deep (>4 layers)
            
            output_activation: Activation for output layer
                              - 'none': No activation (RECOMMENDED)
                              - 'tanh': Bounded output (rarely used)
        
        Raises:
            ValueError: If activation is unknown
        """
        super().__init__()
        
        # ═════════════════════════════════════════════════════
        # 1. SETUP ACTIVATION FUNCTIONS
        # ═════════════════════════════════════════════════════
        self.activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'elu': nn.ELU()
        }
        
        # Validate activation choice
        if activation not in self.activations:
            raise ValueError(
                f"Unknown activation: {activation}. "
                f"Choose from {list(self.activations.keys())}"
            )
        
        self.activation = self.activations[activation]
        
        # Output activation (default: Identity = no activation)
        if output_activation == 'none':
            self.output_activation = nn.Identity()
        else:
            self.output_activation = self.activations.get(
                output_activation,
                nn.Identity()
            )
        
        # ═════════════════════════════════════════════════════
        # 2. BUILD NETWORK LAYERS
        # ═════════════════════════════════════════════════════
        layers = []
        
        # Dimensions: [input_dim, hidden_dim1, hidden_dim2, ..., hidden_dimN]
        dims = [input_dim] + list(hidden_dims)
        
        # Build hidden layers
        for i in range(len(dims) - 1):
            # Linear layer: dims[i] → dims[i+1]
            layer = nn.Linear(dims[i], dims[i+1])
            
            # Orthogonal initialization (gain=sqrt(2) for hidden)
            layer = orthogonal_init(layer, gain=np.sqrt(2))
            
            layers.append(layer)
            
            # Layer normalization (optional)
            if use_layer_norm:
                layers.append(nn.LayerNorm(dims[i+1]))
            
            # Activation function
            layers.append(self.activation)
        
        # Output layer: last_hidden_dim → output_dim
        output_layer = nn.Linear(dims[-1], output_dim)
        
        # Small initialization for output (policy starts near 0)
        output_layer = orthogonal_init(output_layer, gain=0.01)
        
        layers.append(output_layer)
        
        # Output activation (if specified)
        if not isinstance(self.output_activation, nn.Identity):
            layers.append(self.output_activation)
        
        # ═════════════════════════════════════════════════════
        # 3. COMBINE INTO SEQUENTIAL NETWORK
        # ═════════════════════════════════════════════════════
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.network(x)


# ═════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════

def get_parameter_count(model: nn.Module) -> int:
    """
    Count trainable parameters in model.
    Example:
        >>> mlp = MLP(68, [256, 256], 3)
        >>> print(f"Parameters: {get_parameter_count(mlp):,}")
        Parameters: 197,891
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_network_summary(model: nn.Module, model_name: str = "Network"):
    """
    Print network architecture summary.
    
    """
    print(f"\n{'═'*60}")
    print(f"{model_name} Network Summary")
    print(f"{'═'*60}")
    print(f"Total Parameters: {get_parameter_count(model):,}")
    print(f"\nArchitecture:")
    for layer in model.network:
        print(f"  {layer}")
    print(f"{'═'*60}\n")