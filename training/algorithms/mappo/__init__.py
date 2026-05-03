"""
MAPPO (Multi-Agent Proximal Policy Optimization) implementation.
"""

from .networks import MLP, orthogonal_init, get_parameter_count
from .actor import ActorNetwork
from .critic import CriticNetwork
from .buffer import RolloutBuffer
from .trainer import MAPPOTrainer  # ← NEW

__all__ = [
    'MLP',
    'orthogonal_init',
    'get_parameter_count',
    'ActorNetwork',
    'CriticNetwork',
    'RolloutBuffer',
    'MAPPOTrainer',  # ← NEW
]