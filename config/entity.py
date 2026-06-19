from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class VictimConfig:
    """
    Configuration for victim generation and behavior.

    Victim Types:
        - InjuredVictim: Stationary, high urgency (4-5)
        - MobileVictim:  Random walk, lower urgency (1-3)

    Urgency scale (1-5) affects reward:
        reward = r_victim_base × (urgency / 5.0)
    """
    # ══════════════════════════════════════════════════════════
    # POPULATION PARAMETERS
    # ══════════════════════════════════════════════════════════
    n_victims_min: int = 30        # min victims per episode
    n_victims_max: int = 45        # max victims per episode

    # ══════════════════════════════════════════════════════════
    # TYPE DISTRIBUTION
    # ══════════════════════════════════════════════════════════
    injured_ratio_min: float = 0.4  # min % injured victims
    injured_ratio_max: float = 0.7  # max % injured victims

    # ══════════════════════════════════════════════════════════
    # URGENCY PARAMETERS (1-5 scale)
    # ══════════════════════════════════════════════════════════
    injured_urgency_min: float = 4.0   # critical
    injured_urgency_max: float = 5.0
    mobile_urgency_min: float = 1.0    # lower priority
    mobile_urgency_max: float = 3.0

    # ══════════════════════════════════════════════════════════
    # MOBILE VICTIM MOVEMENT
    # ══════════════════════════════════════════════════════════
    mobile_speed_min_mps: float = 0.2      # min speed (m/s)
    mobile_speed_max_mps: float = 0.4      # max speed (m/s)
    mobile_dir_change_steps: int = 20      # change direction every N steps

    # ══════════════════════════════════════════════════════════
    # BACKWARD COMPATIBILITY
    # ══════════════════════════════════════════════════════════
    @property
    def mobile_speed_min(self) -> float:
        return self.mobile_speed_min_mps

    @property
    def mobile_speed_max(self) -> float:
        return self.mobile_speed_max_mps

    @property
    def mobile_dir_change(self) -> int:
        return self.mobile_dir_change_steps


@dataclass
class ObstacleConfig:
    """
    Configuration for obstacle generation.

    Notes:
        - n_danger_total distributed across danger types
        - Actual count per type bounded by DangerZoneConfig.max_counts
    """
    # ══════════════════════════════════════════════════════════
    # DEBRIS
    # ══════════════════════════════════════════════════════════
    # n_debris: int = 6
    n_debris: int = 50  # ← CHANGED: Much denser obstacle field

    # train
    # debris_width_min_m: float = 2.0    # min XY footprint (meters)
    # debris_width_max_m: float = 5.0    # max XY footprint (meters)
    # debris_height_min_m: float = 3.0   # min 3D height (meters)
    # debris_height_max_m: float = 8.0   # max 3D height (meters)
    
    # test
    debris_width_min_m: float = 4.0    # min XY footprint (meters)
    debris_width_max_m: float = 6.0    # max XY footprint (meters)
    debris_height_min_m: float = 6.0   # min 3D height (meters)
    debris_height_max_m: float = 15.0   # max 3D height (meters)
    

    # ══════════════════════════════════════════════════════════
    # DANGER ZONES
    # ══════════════════════════════════════════════════════════
    # n_danger_total: int = 4
    n_danger_total: int = 24  # ← CHANGED: 6× more danger zones

    # ══════════════════════════════════════════════════════════
    # BACKWARD COMPATIBILITY
    # ══════════════════════════════════════════════════════════
    @property
    def debris_width_min(self) -> float:
        return self.debris_width_min_m

    @property
    def debris_width_max(self) -> float:
        return self.debris_width_max_m

    @property
    def debris_height_min(self) -> float:
        return self.debris_height_min_m

    @property
    def debris_height_max(self) -> float:
        return self.debris_height_max_m

@dataclass
class DangerZoneConfig:
    """
    REBALANCED v2:
        Old: radiation=-25/step → 1 zone × 50 steps = -1250 (catastrophic)
        New: radiation=-5/step  → 1 zone × 50 steps = -250  (significant but survivable)

    Design principle:
        Danger zone penalty nên đủ để discourage UAV đi vào
        Nhưng không nên wipe toàn bộ episode reward
    """
    # train
    # heights: Dict[str, float] = field(default_factory=lambda: {
    #     "gas":       3.0,
    #     "fire":      15.0,
    #     "smoke":     25.0,
    #     "collapse":  10.0,
    #     "radiation": np.inf,
    # })
    # test
    heights: Dict[str, float] = field(default_factory=lambda: {
        "gas":       6.0,      # ← CHANGED: 3→6m
        "fire":      22.0,     # ← CHANGED: 15→22m
        "smoke":     30.0,     # ← CHANGED: 25→30m
        "collapse":  18.0,     # ← CHANGED: 10→18m
        "radiation": np.inf,   # Still blocks all altitudes
    })

    # REBALANCED: Scale down ~5× so với cũ
    penalties: Dict[str, float] = field(default_factory=lambda: {
        "gas":       -3.0,   # Was -15.0
        "fire":      -3.0,   # Was -15.0
        "smoke":     -2.5,   # Was -8.0
        "collapse":  -1.0,   # Was -5.0
        "radiation": -5.0,   # Was -25.0 ← THỦ PHẠM CHÍNH
    })

    # Giải thích scale:
    # Old radiation: -25/step × 50 steps = -1250 per zone visit
    # New radiation: -5/step  × 50 steps = -250  per zone visit
    # → Vẫn rất painful, nhưng không catastrophic

    # max_counts: Dict[str, int] = field(default_factory=lambda: {
    #     "gas":       3,
    #     "fire":      2,
    #     "smoke":     2,
    #     "collapse":  3,
    #     "radiation": 1,
    # })
    max_counts: Dict[str, int] = field(default_factory=lambda: {
        "gas":       6,   # ← CHANGED: 3→6
        "fire":      4,   # ← CHANGED: 2→4
        "smoke":     4,   # ← CHANGED: 2→4
        "collapse":  6,   # ← CHANGED: 3→6
        "radiation": 4,   # ← CHANGED: 1→4 (nhiều radiation zones!)
    })  # Tổng max = 24 (khớp với n_danger_total)

    # train
    # widths: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
    #     "gas":       (4.0,  8.0),
    #     "fire":      (6.0,  12.0),
    #     "smoke":     (10.0, 20.0),
    #     "collapse":  (5.0,  10.0),
    #     "radiation": (15.0, 25.0),
    # })
    widths: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "gas":       (5.0,  10.0),   # ← CHANGED: (4,8) → (5,10)
        "fire":      (6.0,  12.0),   # Giữ nguyên
        "smoke":     (12.0, 24.0),   # ← CHANGED: (10,20) → (12,24)
        "collapse":  (8.0,  18.0),   # ← CHANGED: (5,10) → (8,18)
        "radiation": (20.0, 30.0),   # ← CHANGED: (15,25) → (20,30)
    })

    # test
    # widths: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
    #         "gas":       (5.0,  10.0),
    #         "fire":      (6.0,  12.0),
    #         "smoke":     (12.0, 24.0),
    #         "collapse":  (8.0,  18.0),
    #         "radiation": (20.0, 30.0),
    #     })

    def validate(self) -> None:
        keys_heights   = set(self.heights.keys())
        keys_penalties = set(self.penalties.keys())
        keys_counts    = set(self.max_counts.keys())
        keys_widths    = set(self.widths.keys())

        assert keys_heights == keys_penalties, (
            f"DangerZoneConfig key mismatch:\n"
            f"  heights:   {sorted(keys_heights)}\n"
            f"  penalties: {sorted(keys_penalties)}"
        )
        assert keys_heights == keys_counts, (
            f"DangerZoneConfig key mismatch:\n"
            f"  heights:    {sorted(keys_heights)}\n"
            f"  max_counts: {sorted(keys_counts)}"
        )
        assert keys_heights == keys_widths, (
            f"DangerZoneConfig key mismatch:\n"
            f"  heights: {sorted(keys_heights)}\n"
            f"  widths:  {sorted(keys_widths)}"
        )

    @property
    def danger_types(self) -> List[str]:
        return list(self.heights.keys())
