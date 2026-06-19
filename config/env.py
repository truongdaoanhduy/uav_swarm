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
    done_coverage_threshold: float = 0.90  # >= 90% coverage -> episode complete
    
    # ══════════════════════════════════════════════════════════
    # FLEET PARAMETERS
    # ══════════════════════════════════════════════════════════
    n_uav: int = 4
    
    # ══════════════════════════════════════════════════════════
    # CHARGING STATION PARAMETERS
    # ══════════════════════════════════════════════════════════
    n_stations: int = 2
    charge_radius_m: float = 3.0
    station_capacity: int = 1
    min_station_spacing_m: float = 15.0
    station_min_boundary_dist_m: float = 5.0
    
    # ══════════════════════════════════════════════════════════
    # OBJECT PLACEMENT CONSTRAINTS
    # ══════════════════════════════════════════════════════════
    max_place_attempts: int = 1000

    min_object_spacing_m: float = 2.5
    
    # Victim placement
    victim_clearance_m: float = 1.5
    victim_near_dist_m: float = 6.0
    victim_min_dist_m: float = 1.0
    
    # UAV spawn
    uav_spawn_radius_m: float = 3.0
    
    # Progressive relaxation params
    placement_relax_threshold: float = 0.7
    placement_relaxed_spacing_m: float = 1.5
    allow_partial_obstacles: bool = True
    warn_on_skipped_objects: bool = False
    
    # ══════════════════════════════════════════════════════════
    # ✅ REPRODUCIBILITY - THÊM MỚI
    # ══════════════════════════════════════════════════════════
    deterministic_eval: bool = False
    eval_seed: int = 42

    # ✅ THÊM: global_seed cho training reproducibility
    # Episode seed = global_seed + episode_id (deterministic, không dùng time.time())
    global_seed: int = 42

    # ✅ THÊM: debug flag cho obs sanity check
    debug_obs: bool = False

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
