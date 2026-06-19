from dataclasses import dataclass
import numpy as np


@dataclass
class SensorConfig:
    """
    Configuration for UAV sensors.

    Responsibility:
        SensorConfig  → HOW sensors work (physics, range, probability)
        ObsConfig     → HOW MANY entities in observation vector

    NOTE: n_obs_victims và n_obs_obstacles đã bị XÓA khỏi đây.
          Chúng thuộc ObsConfig (representation concern).
          Sensor chỉ cần biết range và detection probability.
    """
    # ══════════════════════════════════════════════════════════
    # COMMUNICATION SENSOR
    # ══════════════════════════════════════════════════════════
    comm_range_m: float = 30.0        # max comm distance (meters)

    # ══════════════════════════════════════════════════════════
    # FOV SENSOR - GEOMETRY
    # ══════════════════════════════════════════════════════════
    hfov_deg: float = 90.0            # horizontal field of view (degrees)

    # ══════════════════════════════════════════════════════════
    # FOV SENSOR - DETECTION PROBABILITY
    # Model: P(detect | in_FOV) = p_base × exp(-decay × altitude)
    # ══════════════════════════════════════════════════════════
    p_detect_base: float = 0.8       # base prob at altitude = 0m
    p_detect_decay: float = 0.04      # exponential decay per meter altitude

    enable_noise:       bool  = True
    motion_blur_coeff:  float = 0.06   # max penalty khi speed = max_speed
    base_miss_rate:     float = 0.03   # hardware miss rate bất kể điều kiện
    # ══════════════════════════════════════════════════════════
    # DERIVED PROPERTIES
    # ══════════════════════════════════════════════════════════
    @property
    def fov_tan(self) -> float:
        """
        Tangent of half-FOV angle.

        Used: fov_radius = altitude × fov_tan
        At hfov=90°: fov_tan = 1.0 → radius = altitude
        """
        return float(np.tan(np.radians(self.hfov_deg / 2.0)))

    @property
    def fov_radius_at_altitude(self) -> callable:
        """
        Closure: compute FOV radius at given altitude.

        Example:
            calc = sensor_cfg.fov_radius_at_altitude
            radius = calc(10.0)  # → 10.0m (at hfov=90°)
        """
        tan = self.fov_tan
        return lambda alt: float(alt) * tan

    # ══════════════════════════════════════════════════════════
    # BACKWARD COMPATIBILITY
    # ══════════════════════════════════════════════════════════
    @property
    def comm_range(self) -> float:
        """DEPRECATED: Use comm_range_m."""
        return self.comm_range_m

    @property
    def n_tracked_uavs(self) -> int:
        """
        DEPRECATED: Value moved to ObsConfig.n_tracked_teammates.
        Returns default 3 for backward compat.
        """
        return 3

    # n_obs_victims và n_obs_obstacles đã XÓA khỏi SensorConfig
    # → Dùng cfg.obs.n_obs_victims và cfg.obs.n_obs_obstacles