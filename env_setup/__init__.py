"""
env/__init__.py

Environment package for SAR UAV Swarm.

Usage:
    from env import SARBaseEnv, SARPettingZooEnv
    
    # hoặc
    from env.base_env import SARBaseEnv
"""

from env_setup.base_env import SARBaseEnv

# PettingZoo wrapper (optional import - không crash nếu pettingzoo chưa cài)
try:
    from env_setup.sar_pettingzoo_env import SARPettingZooEnv
    PETTINGZOO_AVAILABLE = True
except ImportError:
    SARPettingZooEnv = None
    PETTINGZOO_AVAILABLE = False

# Backends
from env_setup.backends.base_backend import BaseBackend
from env_setup.backends.logic_backend import LogicBackend

__all__ = [
    "SARBaseEnv",
    "SARPettingZooEnv",
    "BaseBackend",
    "LogicBackend",
    "PETTINGZOO_AVAILABLE",
]