"""
config/__init__.py
Export all config classes.
"""

from .config import AppConfig
from .env import EnvConfig
from .uav import UAVConfig
from .sensor import SensorConfig
from .entity import VictimConfig, ObstacleConfig, DangerZoneConfig  # ← FIX: DangerZoneConfig
from .reward import RewardConfig
from .obs import ObsConfig
from .train import TrainConfig
from .curriculum_config import (
    StageConfig,
    STAGE_EASY,
    STAGE_MEDIUM,
    STAGE_HARD,
    CURRICULUM_STAGES,
)
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
__all__ = [
    # Main config
    "AppConfig",

    # Sub-configs
    "EnvConfig",
    "UAVConfig",
    "SensorConfig",
    "VictimConfig",
    "ObstacleConfig",
    "DangerZoneConfig",   # ← FIX: tên đúng
    "RewardConfig",
    "ObsConfig",
    "TrainConfig",
    # "NoiseConfig"       ← XÓA: không tồn tại
    # "DangerConfig"      ← XÓA: sai tên

    # Curriculum
    "StageConfig",
    "STAGE_EASY",
    "STAGE_MEDIUM",
    "STAGE_HARD",
    "CURRICULUM_STAGES",
]