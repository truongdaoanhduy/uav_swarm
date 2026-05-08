# config/curriculum_config.py
# THAY THẾ TOÀN BỘ FILE

from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class StageConfig:
    """
    Config cho 1 curriculum stage.

    Difficulty Progression:
        Difficulty đến từ coverage_pressure (m²/UAV), không phải victim density.

        Stage    Map         Area         Pressure         Steps/m²
        HARD     250×250     62,500 m²    15,625 m²/UAV   0.0064
        EXTREME  350×350     122,500 m²   30,625 m²/UAV   0.0033  (2× HARD)
        TRANSFER 300×300     90,000 m²    22,500 m²/UAV   0.0036  (eval only)
    """
    name:             str
    map_size:         int
    n_uav:            int
    n_victims_min:    int
    n_victims_max:    int
    n_debris:         int
    n_danger_total:   int
    station_capacity: int
    max_steps:        int
    min_episodes:     int
    advance_coverage: float
    advance_victims:  float

    # ══════════════════════════════════════════════════════════
    # COMPUTED PROPERTIES
    # ══════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════
# STAGE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════

STAGE_HARD = StageConfig(
    name="hard",
    map_size=250,
    n_uav=4,
    n_victims_min=30,
    n_victims_max=40,
    n_debris=30,
    n_danger_total=8,
    station_capacity=1,
    max_steps=2500,
    min_episodes=500,
    advance_coverage=0.60,
    advance_victims=0.70,
)

STAGE_EXTREME = StageConfig(
    name="extreme",

    # ══ MAP: 350×350 = 122,500 m² (gấp đôi HARD) ══════════
    map_size=350,
    n_uav=4,
    # coverage_pressure = 30,625 m²/UAV ← 2× HARD

    # ══ VICTIMS: density ~ 0.50/1000m² (consistent với HARD) ══
    # avg = (55+68)/2 = 61.5 → 61.5/122500*1000 = 0.502
    n_victims_min=55,
    n_victims_max=68,

    # ══ OBSTACLES: scale tương tự HARD density ═════════════
    # HARD: (30+8)/62500 = 0.608/1000m²
    # EXTREME: (74+17)/122500 = 0.743/1000m²  (nhích lên 1 chút)
    n_debris=74,
    n_danger_total=17,

    # ══ STATION: vẫn bottleneck ════════════════════════════
    station_capacity=1,

    # ══ TIME: 4000/122500 = 0.0033 steps/m² (hardest) ═════
    max_steps=4000,

    min_episodes=500,
    advance_coverage=0.55,
    advance_victims=0.65,
)

# TRANSFER: Dùng để test zero-shot generalization
# Map size khác, density tương tự → test spatial generalization
STAGE_TRANSFER = StageConfig(
    name="transfer",
    map_size=300,
    n_uav=4,
    # avg victims = 44.5 → 44.5/90000*1000 = 0.494/1000m²
    n_victims_min=40,
    n_victims_max=50,
    # obstacles density = (54+12)/90000*1000 = 0.733/1000m²
    n_debris=54,
    n_danger_total=12,
    station_capacity=1,
    max_steps=3200,
    min_episodes=0,       # Eval only, không train
    advance_coverage=0.0,
    advance_victims=0.0,
)


# ══════════════════════════════════════════════════════════════════════
# VERIFICATION
# ══════════════════════════════════════════════════════════════════════

def _verify_stages() -> None:
    """
    Verify density consistency cho các stage được dùng.
    
    NOTE: Chỉ verify STAGE_HARD và STAGE_EXTREME.
    STAGE_TRANSFER là eval-only, không cần verify chặt.
    """
    # Chỉ verify các stage training
    stages_to_verify = [STAGE_HARD, STAGE_EXTREME]

    for stage in stages_to_verify:
        vd = stage.victim_density_per_1000m2
        od = stage.obstacle_density_per_1000m2

        assert 0.40 <= vd <= 0.70, (
            f"[{stage.name}] victim density {vd:.3f} "
            f"out of range [0.40, 0.70]/1000m²"
        )
        assert 0.25 <= od <= 1.20, (
            f"[{stage.name}] obstacle density {od:.3f} "
            f"out of range [0.25, 1.20]/1000m²"
        )

    # Verify pressure tăng HARD → EXTREME
    assert (
        STAGE_EXTREME.coverage_pressure_m2_per_uav
        > STAGE_HARD.coverage_pressure_m2_per_uav
    ), (
        f"EXTREME pressure ({STAGE_EXTREME.coverage_pressure_m2_per_uav:.0f}) "
        f"phải lớn hơn HARD ({STAGE_HARD.coverage_pressure_m2_per_uav:.0f})"
    )

    # print(
    #     f"✅ Stage verification passed:\n"
    #     f"   HARD:    {STAGE_HARD.describe()}\n"
    #     f"   EXTREME: {STAGE_EXTREME.describe()}\n"
    #     f"   TRANSFER:{STAGE_TRANSFER.describe()}"
    # )


_verify_stages()