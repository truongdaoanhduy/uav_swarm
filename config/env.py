# config/env.py
from dataclasses import dataclass


@dataclass
class EnvConfig:
    # ══════════════════════════════════════════════════════════
    # MAP PARAMETERS
    # ══════════════════════════════════════════════════════════
    map_size: int = 100
    grid_size: int = 100
    
    # ══════════════════════════════════════════════════════════
    # TIME PARAMETERS
    # ══════════════════════════════════════════════════════════
    dt_seconds: float = 1
    max_steps: int = 600
    
    # ══════════════════════════════════════════════════════════
    # FLEET PARAMETERS
    # ══════════════════════════════════════════════════════════
    n_uav: int = 4
    
    # ══════════════════════════════════════════════════════════
    # CHARGING STATION PARAMETERS
    # ══════════════════════════════════════════════════════════
    n_stations: int = 2
    charge_radius_m: float = 3.0
    station_capacity: int = 2
    min_station_spacing_m: float = 15.0
    station_min_boundary_dist_m: float = 5.0
    
    # ══════════════════════════════════════════════════════════
    # OBJECT PLACEMENT CONSTRAINTS
    # ══════════════════════════════════════════════════════════
    max_place_attempts: int = 500  # ✅ INCREASED: 200 → 500 (FIX-P1E)
    
    # ✅ FIX 3.1+3.2: UNIFIED placement params (single source of truth)
    min_object_spacing_m: float = 2.5  # ✅ CHANGED: 2.0 → 2.5 (sync với MapGenerator)
    
    # Victim placement
    victim_clearance_m: float = 1.5
    victim_near_dist_m: float = 6.0
    victim_min_dist_m: float = 1.0
    
    # UAV spawn
    uav_spawn_radius_m: float = 3.0
    
    # ✅ FIX 3.1: Progressive relaxation params (used by MapGenerator)
    placement_relax_threshold: float = 0.7   # Relax after 70% attempts
    placement_relaxed_spacing_m: float = 1.5 # Relaxed spacing (meters)
    allow_partial_obstacles: bool = True     # Skip instead of crash
    warn_on_skipped_objects: bool = False     # Log warnings
    
    # ══════════════════════════════════════════════════════════
    # REPRODUCIBILITY & ABLATION CONTROL
    # ══════════════════════════════════════════════════════════
    deterministic_eval: bool = False    # ✅ FIX 3.3: Enable fixed-seed eval
    eval_seed: int = 42                 # Fixed seed for evaluation episodes
    
    # ══════════════════════════════════════════════════════════
    # DERIVED PROPERTIES
    # ══════════════════════════════════════════════════════════
    @property
    def map_area_m2(self) -> int:
        return self.map_size * self.map_size
    
    @property
    def cell_size_m(self) -> float:
        return self.map_size / self.grid_size
    
    # ══════════════════════════════════════════════════════════
    # BACKWARD COMPATIBILITY
    # ══════════════════════════════════════════════════════════
    @property
    def dt(self) -> float:
        return self.dt_seconds
    
    @property
    def charge_radius(self) -> float:
        return self.charge_radius_m
    
    @property
    def min_station_spacing(self) -> float:
        return self.min_station_spacing_m
    
    @property
    def station_min_boundary_dist(self) -> float:
        return self.station_min_boundary_dist_m
    
    @property
    def victim_clearance(self) -> float:
        return self.victim_clearance_m
    
    @property
    def victim_near_dist(self) -> float:
        return self.victim_near_dist_m
    
    @property
    def victim_min_dist(self) -> float:
        return self.victim_min_dist_m
    
    @property
    def uav_spawn_radius(self) -> float:
        return self.uav_spawn_radius_m
    
    @property
    def min_object_spacing(self) -> float:
        return self.min_object_spacing_m