"""Curriculum stage definitions for training and transfer evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StageConfig:
    """Configuration for one SAR scenario stage."""

    name: str
    map_size: int
    n_uav: int
    n_victims_min: int
    n_victims_max: int
    n_debris: int
    n_danger_total: int
    station_capacity: int
    max_steps: int
    min_episodes: int
    advance_coverage: float
    advance_victims: float

    @property
    def map_area_m2(self) -> int:
        return self.map_size * self.map_size

    @property
    def coverage_pressure_m2_per_uav(self) -> float:
        return self.map_area_m2 / self.n_uav

    @property
    def victim_density_per_1000m2(self) -> float:
        avg = (self.n_victims_min + self.n_victims_max) / 2.0
        return avg / self.map_area_m2 * 1000

    @property
    def obstacle_density_per_1000m2(self) -> float:
        return (self.n_debris + self.n_danger_total) / self.map_area_m2 * 1000

    @property
    def steps_per_m2(self) -> float:
        return self.max_steps / self.map_area_m2

    def describe(self) -> str:
        return (
            f"[{self.name.upper()}] "
            f"map={self.map_size}×{self.map_size} ({self.map_area_m2:,}m²) | "
            f"UAVs={self.n_uav} "
            f"(pressure={self.coverage_pressure_m2_per_uav:,.0f}m²/UAV) | "
            f"victims={self.n_victims_min}-{self.n_victims_max} "
            f"(density={self.victim_density_per_1000m2:.2f}/1000m²) | "
            f"debris={self.n_debris} | danger={self.n_danger_total} | "
            f"cap={self.station_capacity} | steps={self.max_steps}"
        )


STAGE_HARD = StageConfig(
    name="hard",
    map_size=250,
    n_uav=4,
    n_victims_min=40,
    n_victims_max=55,
    n_debris=30,
    n_danger_total=12,
    station_capacity=1,
    max_steps=2500,
    min_episodes=500,
    advance_coverage=0.60,
    advance_victims=0.70,
)


STAGE_EXTREME = StageConfig(
    name="extreme",
    map_size=400,
    n_uav=4,
    n_victims_min=70,
    n_victims_max=90,
    n_debris=74,
    n_danger_total=24,
    station_capacity=1,
    max_steps=4200,
    min_episodes=500,
    advance_coverage=0.55,
    advance_victims=0.65,
)


STAGE_TRANSFER = StageConfig(
    name="transfer",
    map_size=350,
    n_uav=4,
    n_victims_min=55,
    n_victims_max=70,
    n_debris=60,
    n_danger_total=18,
    station_capacity=1,
    max_steps=3500,
    min_episodes=0,
    advance_coverage=0.0,
    advance_victims=0.0,
)
