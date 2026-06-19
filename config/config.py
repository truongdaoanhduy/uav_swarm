"""
Config package - Master config.
Dùng: from config import AppConfig
"""
from dataclasses import dataclass, field, asdict
import json
import numpy as np

# ── FIX: Dùng relative import, XÓA sys.path hack ─────────────────────────────
from .env import EnvConfig
from .uav import UAVConfig
from .sensor import SensorConfig
from .entity import VictimConfig, ObstacleConfig, DangerZoneConfig
from .reward import RewardConfig
from .obs import ObsConfig
from .train import TrainConfig
# XÓA: import sys, os, sys.path.append(...)


@dataclass
class AppConfig:
    """
    Master config - truyền object này vào env/algorithm.

    Usage:
        cfg = AppConfig()
        env = SARBaseEnv(cfg)

    Curriculum usage:
        cfg.apply_stage(stage)  # Apply stage config in-place
    """
    env:      EnvConfig       = field(default_factory=EnvConfig)
    uav:      UAVConfig       = field(default_factory=UAVConfig)
    sensor:   SensorConfig    = field(default_factory=SensorConfig)
    victim:   VictimConfig    = field(default_factory=VictimConfig)
    obstacle: ObstacleConfig  = field(default_factory=ObstacleConfig)
    danger:   DangerZoneConfig = field(default_factory=DangerZoneConfig)
    reward:   RewardConfig    = field(default_factory=RewardConfig)
    obs:      ObsConfig       = field(default_factory=ObsConfig)
    train:    TrainConfig     = field(default_factory=TrainConfig)
    viz_mode: str = "none"   # "2d" | "3d" | "none"
    viz_3d_cfg: dict = None  # Passed to RenderConfig3D

    def __post_init__(self):
        """
        Post-init hook:
            - Auto-sync n_stations: EnvConfig → ObsConfig
            - Validate DangerZoneConfig keys
            - Validate ObsConfig dimensions
        """
        # ── Sync n_stations ───────────────────────────────────────────────────
        self.obs.n_stations = self.env.n_stations

        # ── Validate ─────────────────────────────────────────────────────────
        self.danger.validate()
        self.obs.validate()

        if self.viz_3d_cfg is None:
                    self.viz_3d_cfg = {}
    # ── Curriculum integration ────────────────────────────────────────────────

    def apply_stage(self, stage) -> None:
        self.env.map_size         = stage.map_size
        self.env.grid_size        = stage.map_size
        self.env.n_uav            = stage.n_uav
        self.env.max_steps        = stage.max_steps
        self.env.station_capacity = stage.station_capacity
        self.victim.n_victims_min = stage.n_victims_min
        self.victim.n_victims_max = stage.n_victims_max
        self.obstacle.n_debris       = stage.n_debris
        self.obstacle.n_danger_total = stage.n_danger_total
        self.__post_init__()

    @property
    def map_diagonal(self) -> float:
        return float(np.sqrt(2) * self.env.map_size)

    @property
    def grid_cell_size(self) -> float:
        return self.env.map_size / self.env.grid_size

    def save(self, path: str) -> None:
        data = asdict(self)
        data["danger"]["heights"]["radiation"] = "inf"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data["danger"]["heights"].get("radiation") == "inf":
            data["danger"]["heights"]["radiation"] = np.inf
        obs_data = data["obs"].copy()
        obs_data.pop("schema", None)
        return cls(
            env      = EnvConfig(**data["env"]),
            uav      = UAVConfig(**data["uav"]),
            sensor   = SensorConfig(**data["sensor"]),
            victim   = VictimConfig(**data["victim"]),
            obstacle = ObstacleConfig(**data["obstacle"]),
            danger   = DangerZoneConfig(**data["danger"]),
            reward   = RewardConfig(**data["reward"]),
            obs      = ObsConfig(**obs_data),
            train    = TrainConfig(**data["train"]),
        )