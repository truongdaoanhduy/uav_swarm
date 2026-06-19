# config/__init__.py

from .config import AppConfig
from .env import EnvConfig
from .uav import UAVConfig
from .sensor import SensorConfig
from .entity import VictimConfig, ObstacleConfig, DangerZoneConfig
from .reward import RewardConfig
from .obs import ObsConfig
from .train import TrainConfig
from .curriculum_config import (
    StageConfig,
    STAGE_HARD,
    STAGE_EXTREME,
    STAGE_TRANSFER,
)

# ✅ XÓA: sys.path hack không cần thiết và gây bug
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

__all__ = [
    "AppConfig",
    "EnvConfig",
    "UAVConfig",
    "SensorConfig",
    "VictimConfig",
    "ObstacleConfig",
    "DangerZoneConfig",
    "RewardConfig",
    "ObsConfig",
    "TrainConfig",
    "StageConfig",
    "STAGE_HARD",
    "STAGE_EXTREME",
    "STAGE_TRANSFER",
]