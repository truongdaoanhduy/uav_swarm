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
        HARD     250×250     62,500 m²    15,625 m²/UAV   0.040
        EXTREME  350×350     122,500 m²   30,625 m²/UAV   0.041  (2× pressure)
        TRANSFER 300×300     90,000 m²    22,500 m²/UAV   0.056  (eval only)
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
    n_victims_min=40,
    n_victims_max=55,
    n_debris=30,
    n_danger_total=12,
    station_capacity=1,
    max_steps=200,
    min_episodes=500,
    advance_coverage=0.60,
    advance_victims=0.70,
)

# ✅ EXTREME: TĂNG debris 30→74, danger 8→24
STAGE_EXTREME = StageConfig(
    name="extreme",

    # ══ MAP: 350×350 = 122,500 m² (gấp đôi HARD) ══════════
    map_size=400,
    n_uav=4,
    # coverage_pressure = 30,625 m²/UAV ← 2× HARD

    # ══ VICTIMS: density ~ 0.50/1000m² (consistent với HARD) ══
    n_victims_min=70,
    n_victims_max=90,

    # ══ OBSTACLES: TĂNG MẠNH ═══════════════════════════════
    # HARD: (30+8)/62500 = 0.608/1000m²
    # EXTREME: (74+24)/122500 = 0.800/1000m² ← 30% denser
    n_debris=74,          # ← CHANGED: Giữ nguyên 74
    n_danger_total=24,    # ← CHANGED: 17→24 (3× HARD)

    # ══ STATION: vẫn bottleneck ════════════════════════════
    station_capacity=1,

    # ══ TIME: 5000/122500 = 0.041 steps/m² ════════════════
    max_steps=4200,       # ← CHANGED: 4000→5000 (buffer thêm vì nhiều obstacles)

    min_episodes=500,
    advance_coverage=0.55,
    advance_victims=0.65,
)

# ✅ TRANSFER: TĂNG debris 54→60, danger 12→18
STAGE_TRANSFER = StageConfig(
    name="transfer",
    map_size=350,
    n_uav=4,
    
    # Victims giữ nguyên
    n_victims_min=55,
    n_victims_max=70,
    
    # ══ OBSTACLES: TĂNG cho eval khó hơn ═══════════════════
    # (60+18)/90000*1000 = 0.867/1000m² ← higher than HARD
    n_debris=60,          # ← CHANGED: 54→60
    n_danger_total=18,    # ← CHANGED: 12→18 (2.25× HARD)
    
    station_capacity=1,
    max_steps=3500,       # ← CHANGED: 3500→5000 (buffer cho obstacles)
    
    min_episodes=0,       # Eval only
    advance_coverage=0.0,
    advance_victims=0.0,
)


# ══════════════════════════════════════════════════════════════════════
# VERIFICATION
# ══════════════════════════════════════════════════════════════════════

# def _verify_stages() -> None:
#     """
#     Verify density consistency cho các stage được dùng.
    
#     NOTE: Chỉ verify STAGE_HARD và STAGE_EXTREME.
#     STAGE_TRANSFER là eval-only, không cần verify chặt.
#     """
#     stages_to_verify = [STAGE_HARD, STAGE_EXTREME]

#     for stage in stages_to_verify:
#         vd = stage.victim_density_per_1000m2
#         od = stage.obstacle_density_per_1000m2

        

#     print(
#         f"✅ Stage verification passed:\n"
#         f"   HARD:    {STAGE_HARD.describe()}\n"
#         f"   EXTREME: {STAGE_EXTREME.describe()}\n"
#         f"   TRANSFER:{STAGE_TRANSFER.describe()}"
#     )


# _verify_stages()
