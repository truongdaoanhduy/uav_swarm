"""
config/obs.py
Observation configuration với validation
"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.env import EnvConfig


@dataclass
class ObsSchemaConfig:
    """
    Observation vector schema definition.
    
    Actor Observation (per UAV):
        - SELF: pos(3) + vel(3) + battery(1) + state(4) = 11
        - STATIONS: [rel_pos(2) + dist(1) + occupancy(1)] × n_stations
        - TEAMMATES: [dist(1) + bearing(1) + rel_alt(1)] × 3
        - OBSTACLES: [rel_pos(2) + dist(1)] × 4
        - VICTIMS: [rel_pos(2) + urgency(1) + dist(1) + bearing(1)] × 5
        - COVERAGE: local_small(1) + local_large(1) + time_remaining(1) = 3
    
    Critic Observation (centralized):
        - All actor observations concatenated
        - GLOBAL: total_victims(1) + found_victims(1) + coverage(1) + ...
    """
    # Self
    SELF_FEATURES: int = 11

    # Station
    STATION_FEATURES_PER: int = 4

    # Team
    TEAMMATE_FEATURES_PER: int = 3

    # Obstacle
    OBSTACLE_FEATURES_PER: int = 3

    # Victim
    VICTIM_FEATURES_PER: int = 5

    # Coverage
    COVERAGE_FEATURES: int = 3

    # Global critic
    GLOBAL_FEATURES: int = 10


@dataclass
class ObsConfig:
    """
    Observation configuration.
    
    Entity Limits:
        - n_obs_victims: Max victims in FOV to include
        - n_obs_obstacles: Max obstacles in FOV to include
        - n_tracked_teammates: Max teammates to track (comm range)
    
    Coverage:
        - local_cov_small: Radius for local coverage metric (meters)
        - local_cov_large: Radius for larger local coverage metric (meters)
    
    Critic:
        - max_uav: Max UAVs for critic input padding
    
    Auto-sync:
        - n_stations: Synced from EnvConfig in AppConfig.__post_init__()
    """
    # ══════════════════════════════════════════════════════════
    # ENTITY OBSERVATION LIMITS
    # ══════════════════════════════════════════════════════════
    n_obs_victims: int = 5         # max victims in observation
    n_obs_obstacles: int = 4       # max obstacles in observation
    n_tracked_teammates: int = 3   # max teammates to track
    
    # ══════════════════════════════════════════════════════════
    # COVERAGE METRICS
    # ══════════════════════════════════════════════════════════
    local_cov_small: int = 15      # small local coverage radius (meters)
    local_cov_large: int = 30      # large local coverage radius (meters)
    
    # ══════════════════════════════════════════════════════════
    # CRITIC PARAMETERS
    # ══════════════════════════════════════════════════════════
    max_uav: int = 8               # max UAVs for padding
    
    # ══════════════════════════════════════════════════════════
    # AUTO-SYNC FROM EnvConfig
    # ══════════════════════════════════════════════════════════
    n_stations: int = None         # set by AppConfig.__post_init__()
    
    # ══════════════════════════════════════════════════════════
    # OBSERVATION SCHEMA
    # ══════════════════════════════════════════════════════════
    schema: ObsSchemaConfig = field(default_factory=ObsSchemaConfig)

    # ══════════════════════════════════════════════════════════
    # COMPUTED DIMENSIONS
    # ══════════════════════════════════════════════════════════
    @property
    def self_dim(self) -> int:
        """Self observation dimension (11)."""
        return self.schema.SELF_FEATURES

    @property
    def station_dim(self) -> int:
        """Station observation dimension (n_stations × 4)."""
        return self.n_stations * self.schema.STATION_FEATURES_PER

    @property
    def team_dim(self) -> int:
        """Teammate observation dimension (3 × 3 = 9)."""
        return self.n_tracked_teammates * self.schema.TEAMMATE_FEATURES_PER

    @property
    def obstacle_dim(self) -> int:
        """Obstacle observation dimension (4 × 3 = 12)."""
        return self.n_obs_obstacles * self.schema.OBSTACLE_FEATURES_PER

    @property
    def victim_dim(self) -> int:
        """Victim observation dimension (5 × 5 = 25)."""
        return self.n_obs_victims * self.schema.VICTIM_FEATURES_PER

    @property
    def coverage_dim(self) -> int:
        """Coverage observation dimension (3)."""
        return self.schema.COVERAGE_FEATURES

    @property
    def actor_dim(self) -> int:
        """
        Total actor observation dimension.
        
        Default (n_stations=2): 11 + 8 + 9 + 12 + 25 + 3 = 68
        """
        return (
            self.self_dim +
            self.station_dim +
            self.team_dim +
            self.obstacle_dim +
            self.victim_dim +
            self.coverage_dim
        )

    @property
    def global_dim(self) -> int:
        """Global state dimension for critic (10)."""
        return self.schema.GLOBAL_FEATURES

    @property
    def critic_dim(self) -> int:
        """
        Total critic observation dimension.
        
        Default: 8 × 68 + 10 = 554
        """
        return self.max_uav * self.actor_dim + self.global_dim

    # ══════════════════════════════════════════════════════════
    # VALIDATION
    # ══════════════════════════════════════════════════════════
    def validate(self) -> None:
        """Validate configuration after auto-sync."""
        assert self.n_stations is not None, "n_stations must be set (auto-synced from EnvConfig)"
        assert self.n_stations >= 1, "Must have at least 1 station"
        assert self.max_uav >= 1, "max_uav must >= 1"
    
    # ══════════════════════════════════════════════════════════
    # BACKWARD COMPATIBILITY
    # ══════════════════════════════════════════════════════════
    @property
    def n_tracked_uavs(self) -> int:
        """DEPRECATED: Use n_tracked_teammates instead."""
        return self.n_tracked_teammates
    
    # Trong class DangerZone - thay thế method cũ
    def get_sensor_modifier(self) -> float:
        """
        Camera visibility multiplier khi victim đứng trong zone.
        
        Giải thích từng type:
            smoke:     Khói đặc → camera gần như mù → 0.40
                    Ví dụ: tòa nhà cháy, khói bốc cao
            
            fire:      Lửa + nhiệt → haze + glare → 0.55
                    Camera thermal vẫn hoạt động nhưng kém
            
            gas:       Khí vô hình → không ảnh hưởng visual → 0.85
                    Chỉ nguy hiểm cho UAV hardware
            
            radiation: Vô hình → không ảnh hưởng camera → 0.95
                    Ảnh hưởng sensors điện tử (nhẹ)
            
            collapse:  Bụi đất, không khí mờ → 0.70
                    Dust cloud từ tòa nhà sập
        
        Returns:
            float: [0.0, 1.0], càng nhỏ càng khó detect
        """
        _MODIFIERS = {
            "smoke":     0.40,
            "fire":      0.55,
            "collapse":  0.70,
            "gas":       0.85,
            "radiation": 0.95,
        }
        return _MODIFIERS.get(self.danger_type, 1.0)