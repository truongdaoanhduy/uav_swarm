# config/train.py
from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainConfig:
    """
    Training configuration.
    
    Phase 2 additions:
        - n_seeds: Tăng từ 3 → 5 cho statistical power
        - seeds: Fixed list cho reproducibility
        - confidence_level: Cho confidence intervals trong paper
    
    NOTE: Algorithm hyperparameters (lr, gamma, etc.) sẽ được
    định nghĩa trong training/algorithms/*/trainer.py, KHÔNG ở đây.
    Lý do: Mỗi algorithm có hyperparameter space khác nhau.
    """
    # ══════════════════════════════════════════════════════════
    # REPRODUCIBILITY
    # ══════════════════════════════════════════════════════════
    n_seeds: int = 5
    seeds: List[int] = field(
        default_factory=lambda: [42, 123, 456, 789, 1011]
    )
    
    # ══════════════════════════════════════════════════════════
    # STATISTICAL TESTING (cho paper)
    # ══════════════════════════════════════════════════════════
    confidence_level: float = 0.95    # Wilcoxon / t-test
    
    # ══════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ══════════════════════════════════════════════════════════
    total_episodes: int = 3000
    
    # ══════════════════════════════════════════════════════════
    # LOGGING & CHECKPOINTING
    # ══════════════════════════════════════════════════════════
    eval_interval: int = 50
    save_interval: int = 100
    log_window: int = 100             # Rolling mean window
    
    # ══════════════════════════════════════════════════════════
    # DEPRECATED (kept for backward compat with curriculum_trainer.py)
    # ══════════════════════════════════════════════════════════
    phase_1a_episodes: int = 500    # DEPRECATED
    phase_1b_episodes: int = 1000   # DEPRECATED
    phase_1c_episodes: int = 2000   # DEPRECATED

    # ══════════════════════════════════════════════════════════
    # MAPPO HYPERPARAMETERS (NEW)
    # ══════════════════════════════════════════════════════════
    
    # Rollout
    mappo_rollout_length: int = 2048     # Steps per update
    
    # PPO
    mappo_n_epochs: int = 10             # Epochs per update
    mappo_batch_size: int = 256          # Minibatch size
    mappo_clip_epsilon: float = 0.2      # PPO clip range
    
    # GAE
    mappo_gamma: float = 0.99            # Discount factor
    mappo_gae_lambda: float = 0.95       # GAE lambda
    
    # Optimization
    mappo_lr_actor: float = 3e-4         # Actor LR
    mappo_lr_critic: float = 1e-3        # Critic LR (higher)
    mappo_max_grad_norm: float = 0.5     # Gradient clipping
    
    # Exploration
    mappo_entropy_coeff: float = 0.01    # Entropy bonus
    
    # Network architecture
    mappo_actor_hidden: tuple = (256, 256)      # Actor layers
    mappo_critic_hidden: tuple = (512, 256)     # Critic layers
    mappo_activation: str = 'tanh'              # Activation fn
    mappo_use_layer_norm: bool = False          # Layer norm
    
    # Logging
    mappo_log_interval: int = 10                # Console log
    mappo_viz_interval: int = 100               # 2D viz
    mappo_checkpoint_interval: int = 100        # Checkpoint save