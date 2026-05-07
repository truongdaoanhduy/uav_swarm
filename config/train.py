from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainConfig:
    """
    Training configuration for MAPPO/MASAC/MATD3.

    UPDATED v2.0 - Stabilize PPO training

    Changes vs v1.0:
        - mappo_clip_epsilon:   0.2  → 0.15  (giảm clip range)
        - mappo_entropy_coeff:  0.01 → 0.005 (giảm entropy khi đã explore đủ)
        - mappo_lr_actor:       3e-4 → 2e-4  (learning rate thấp hơn)
        - mappo_lr_critic:      1e-3 → 5e-4  (critic lr thấp hơn)
        - mappo_n_epochs:       10   → 8     (ít epochs hơn/update)
        - mappo_use_layer_norm: False → True  (normalize để stabilize critic)

    Rationale:
        clip_frac=0.4 → quá cao → giảm clip_epsilon
        c_loss=100    → reward scale lớn → normalize + giảm lr
        entropy=5.7   → vẫn explore nhiều → giảm entropy_coeff nhẹ
    """

    # ══════════════════════════════════════════════════════════════════════
    # MAPPO HYPERPARAMETERS
    # ══════════════════════════════════════════════════════════════════════

    # Rollout (auto-computed trong train_mappo.py)
    mappo_rollout_length: int   = 2048
    mappo_n_epochs:       int   = 8          # 10 → 8
    mappo_batch_size:     int   = 256

    # ✅ FIX v2.0: Giảm clip_epsilon vì clip_frac=0.4 quá cao
    mappo_clip_epsilon:   float = 0.15       # 0.2 → 0.15

    mappo_gamma:          float = 0.99
    mappo_gae_lambda:     float = 0.95

    # ✅ FIX v2.0: Giảm learning rates để stabilize
    mappo_lr_actor:       float = 2e-4       # 3e-4 → 2e-4
    mappo_lr_critic:      float = 5e-4       # 1e-3 → 5e-4

    mappo_max_grad_norm:  float = 0.5

    # ✅ FIX v2.0: Giảm entropy_coeff nhẹ
    mappo_entropy_coeff:  float = 0.005      # 0.01 → 0.005

    # Network architecture
    mappo_actor_hidden:   tuple = (256, 256)
    mappo_critic_hidden:  tuple = (512, 256)
    mappo_activation:     str   = 'tanh'

    # ✅ FIX v2.0: Enable layer norm để stabilize critic
    mappo_use_layer_norm: bool  = True       # False → True

    # ══════════════════════════════════════════════════════════════════════
    # MASAC HYPERPARAMETERS
    # ══════════════════════════════════════════════════════════════════════

    masac_buffer_capacity:  int   = 500_000
    masac_batch_size:       int   = 256
    masac_lr_actor:         float = 3e-4
    masac_lr_critic:        float = 3e-4
    masac_lr_alpha:         float = 3e-4
    masac_gamma:            float = 0.99
    masac_tau:              float = 0.005
    masac_alpha_init:       float = 0.2
    masac_auto_alpha:       bool  = True
    masac_warmup_steps:     int   = 1000
    masac_update_every:     int   = 1
    masac_updates_per_step: int   = 1

    # ✅ Actor dùng chung kiến trúc với MAPPO (cùng obs_dim=68 → action_dim=4)
    masac_actor_hidden:     tuple = (256, 256)   # = mappo_actor_hidden

    # ✅ Twin Q-critic riêng (input = global_obs[554] + action[4] → Q scalar)
    #    Khác MAPPO critic (value function, không có action input)
    masac_critic_hidden:    tuple = (400, 300)

    # ══════════════════════════════════════════════════════════════════════
    # MATD3 HYPERPARAMETERS
    # ══════════════════════════════════════════════════════════════════════

    matd3_buffer_capacity:  int   = 500_000
    matd3_batch_size:       int   = 256
    matd3_lr_actor:         float = 3e-4
    matd3_lr_critic:        float = 3e-4
    matd3_gamma:            float = 0.99
    matd3_tau:              float = 0.005
    matd3_policy_delay:     int   = 2
    matd3_explore_noise:    float = 0.1
    matd3_target_noise:     float = 0.2
    matd3_noise_clip:       float = 0.5
    matd3_warmup_steps:     int   = 1000
    matd3_update_every:     int   = 1
    matd3_updates_per_step: int   = 1

    # ✅ Actor dùng chung kiến trúc với MAPPO (cùng obs_dim=68 → action_dim=4)
    matd3_actor_hidden:     tuple = (256, 256)   # = mappo_actor_hidden

    # ✅ Twin Q-critic riêng (input = global_obs[554] + action[4] → Q scalar)
    matd3_critic_hidden:    tuple = (400, 300)

    # ══════════════════════════════════════════════════════════════════════
    # REPRODUCIBILITY
    # ══════════════════════════════════════════════════════════════════════

    n_seeds:          int   = 5
    seeds:            List  = field(
        default_factory=lambda: [42, 123, 456, 789, 1011]
    )
    confidence_level: float = 0.95