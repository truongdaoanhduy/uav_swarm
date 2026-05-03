"""
rewards package

Reward functions.
"""

from rewards.baseline_reward import BaselineReward

# LLM reward (Phase 3 - optional)
try:
    from rewards.llm_reward import LLMReward
    LLM_REWARD_AVAILABLE = True
except ImportError:
    LLMReward = None
    LLM_REWARD_AVAILABLE = False

__all__ = [
    "BaselineReward",
    "LLMReward",
    "LLM_REWARD_AVAILABLE",
]
