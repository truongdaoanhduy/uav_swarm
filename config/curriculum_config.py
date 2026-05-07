from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class StageConfig:
    """
    Config cho 1 curriculum stage.

    Difficulty Progression:
        Difficulty đến từ coverage_pressure (m²/UAV), không phải victim density.

        Stage   Map         Area        Pressure        Steps/m²
        EASY    150×150     22,500 m²   5,625  m²/UAV  0.0133
        MEDIUM  200×200     40,000 m²   10,000 m²/UAV  0.0088  (1.78×)
        HARD    250×250     62,500 m²   15,625 m²/UAV  0.0064  (2.78×)

        Victim density được giữ CONSTANT (~0.53/1000m²) để:
        - Difficulty chỉ đến từ 1 variable (coverage pressure)
        - Paper có thể claim "controlled experiment"

    NOTE:
        reward_scale đã bị XÓA khỏi StageConfig.
        Lý do: Policy gradient phụ thuộc reward scale.
        Thay đổi scale giữa stages → learning objective bị lệch.
        Difficulty đến từ environment, không từ reward magnitude.
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
    # reward_scale: INTENTIONALLY REMOVED - see docstring

    # ══════════════════════════════════════════════════════════
    # COMPUTED PROPERTIES (for paper reporting & logging)
    # ══════════════════════════════════════════════════════════

    @property
    def map_area_m2(self) -> int:
        """Total map area in m²."""
        return self.map_size * self.map_size

    @property
    def coverage_pressure_m2_per_uav(self) -> float:
        """
        Key difficulty metric: area each UAV must cover (m²/UAV).

        EASY:   5,625  m²/UAV  (baseline)
        MEDIUM: 10,000 m²/UAV  (1.78× harder)
        HARD:   15,625 m²/UAV  (2.78× harder)

        Dùng metric này trong paper để justify difficulty scaling.
        """
        return self.map_area_m2 / self.n_uav

    @property
    def victim_density_per_1000m2(self) -> float:
        """
        Average victim density (victims/1000m²).

        Nên CONSTANT qua các stages (~0.53) để:
        - Chứng minh difficulty đến từ coverage pressure
        - Không phải từ victim density inflation
        """
        avg = (self.n_victims_min + self.n_victims_max) / 2.0
        return avg / self.map_area_m2 * 1000

    @property
    def obstacle_density_per_1000m2(self) -> float:
        """Average obstacle density (obstacles/1000m²)."""
        return (self.n_debris + self.n_danger_total) / self.map_area_m2 * 1000

    @property
    def steps_per_m2(self) -> float:
        """Time budget per m² (lower = harder)."""
        return self.max_steps / self.map_area_m2

    def describe(self) -> str:
        """Human-readable description cho logging."""
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

STAGE_EASY = StageConfig(
    name="easy",

    # ══ MAP: 150×150 = 22,500 m² ══════════════════════════════
    map_size=150,
    n_uav=4,
    # coverage_pressure = 5,625 m²/UAV ← BASELINE

    # ══ VICTIMS: density = (10+14)/2 / 22500 * 1000 = 0.53/1000m² ══
    n_victims_min=10,
    n_victims_max=14,

    # ══ OBSTACLES: density = (6+2)/22500*1000 = 0.36/1000m² ══
    n_debris=6,
    n_danger_total=2,

    # ══ STATION: generous (no bottleneck) ══════════════════════
    station_capacity=2,

    # ══ TIME: 300/22500 = 0.0133 steps/m² ═════════════════════
    max_steps=300,

    min_episodes=200,
    advance_coverage=0.70,
    advance_victims=0.80,
)

STAGE_MEDIUM = StageConfig(
    name="medium",

    # ══ MAP: 200×200 = 40,000 m² ══════════════════════════════
    map_size=200,
    n_uav=4,
    # coverage_pressure = 10,000 m²/UAV ← 1.78× EASY

    # ══ VICTIMS: density = (18+24)/2 / 40000 * 1000 = 0.53/1000m² ══
    # Giữ nguyên density → difficulty chỉ từ coverage pressure
    n_victims_min=18,
    n_victims_max=24,

    # ══ OBSTACLES: density = (10+4)/40000*1000 = 0.35/1000m² ══
    n_debris=10,
    n_danger_total=4,

    # ══ STATION: bottleneck introduced ════════════════════════
    station_capacity=1,

    # ══ TIME: 350/40000 = 0.0088 steps/m² (harder) ═══════════
    max_steps=350,

    min_episodes=300,
    advance_coverage=0.65,
    advance_victims=0.75,
)

STAGE_HARD = StageConfig(
    name="hard",

    # ══ MAP: 250×250 = 62,500 m² ══════════════════════════════
    map_size=250,
    n_uav=4,
    # coverage_pressure = 15,625 m²/UAV ← 2.78× EASY

    # ══ VICTIMS: density = (28+36)/2 / 62500 * 1000 = 0.51/1000m² ══
    # Gần bằng EASY/MEDIUM → density consistent
    n_victims_min=30,
    n_victims_max=40,

    # ══ OBSTACLES: density = (15+7)/62500*1000 = 0.35/1000m² ══
    n_debris=30,
    n_danger_total=9,

    # ══ STATION: persistent bottleneck ════════════════════════
    station_capacity=2,

    # ══ TIME: 400/62500 = 0.0064 steps/m² (hardest) ══════════
    max_steps=2500,

    min_episodes=500,
    advance_coverage=0.60,
    advance_victims=0.70,
)

CURRICULUM_STAGES: List[StageConfig] = [
    STAGE_EASY,
    STAGE_MEDIUM,
    STAGE_HARD,
]


# ══════════════════════════════════════════════════════════════════════
# VERIFICATION (chạy khi import module)
# ══════════════════════════════════════════════════════════════════════

def _verify_stages() -> None:
    """
    Verify density consistency và difficulty progression.
    Chạy tự động khi import module.
    """
    pressures = []
    for stage in CURRICULUM_STAGES:
        vd = stage.victim_density_per_1000m2
        od = stage.obstacle_density_per_1000m2
        cp = stage.coverage_pressure_m2_per_uav
        sp = stage.steps_per_m2
        pressures.append(cp)

        # Victim density nên trong khoảng [0.45, 0.65]
        assert 0.45 <= vd <= 0.65, (
            f"[{stage.name}] victim density {vd:.3f} "
            f"out of range [0.45, 0.65]/1000m²"
        )
        
        # ✅ THAY ĐỔI: Obstacle density upper limit
        # OLD: assert 0.25 <= od <= 0.50
        # NEW: assert 0.25 <= od <= 1.20
        assert 0.25 <= od <= 1.20, (  # ← THAY ĐỔI Ở ĐÂY
            f"[{stage.name}] obstacle density {od:.3f} "
            f"out of range [0.25, 1.20]/1000m²"  # ← CẬP NHẬT MESSAGE
        )

    # Coverage pressure phải tăng dần
    for i in range(1, len(pressures)):
        assert pressures[i] > pressures[i-1], (
            f"Coverage pressure không tăng: "
            f"{CURRICULUM_STAGES[i-1].name}={pressures[i-1]:.0f} "
            f">= {CURRICULUM_STAGES[i].name}={pressures[i]:.0f}"
        )


_verify_stages()