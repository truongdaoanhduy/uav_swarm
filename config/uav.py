from dataclasses import dataclass


@dataclass
class UAVConfig:
    """
    Unified UAV configuration.

    Battery drain rates are in % per SECOND (not per step).
    Actual drain per step = rate × dt_seconds.
    This decouples physics from simulation timestep.

    Example:
        drain_xy = 0.10 %/s × dt=1.0s = 0.10% per step
        drain_xy = 0.10 %/s × dt=0.5s = 0.05% per step
    """
    # ══════════════════════════════════════════════════════════
    # PHYSICS PARAMETERS
    # ══════════════════════════════════════════════════════════
    z_min_m: float = 3.0               # min altitude (meters)
    z_max_m: float = 40.0              # max altitude (meters)
    max_speed_xy_mps: float = 5.0      # max horizontal speed (m/s)
    max_speed_z_mps: float = 2.0       # max vertical speed (m/s)
    collision_radius_m: float = 0.5    # collision detection radius (meters)

    # ══════════════════════════════════════════════════════════
    # BATTERY DRAIN RATES (% per SECOND)
    # FIX: _per_second thay vì _per_step → decoupled from dt
    # Actual drain = rate × cfg.env.dt_seconds
    # ══════════════════════════════════════════════════════════
    drain_xy_pct_per_s: float = 0.10      # horizontal movement
    drain_z_up_pct_per_s: float = 0.15    # climbing
    drain_z_down_pct_per_s: float = 0.03  # descending
    drain_idle_pct_per_s: float = 0.05    # hovering

    # ══════════════════════════════════════════════════════════
    # BATTERY CHARGE RATE (% per SECOND at station)
    # ══════════════════════════════════════════════════════════
    charge_rate_pct_per_s: float = 1.5

    # ══════════════════════════════════════════════════════════
    # BATTERY STATE THRESHOLDS (%)
    # ══════════════════════════════════════════════════════════
    battery_return_pct: float = 10.0   # auto-return when ≤ this
    battery_ready_pct: float = 80.0    # ready to deploy when ≥ this
    battery_dead_pct: float = 0.0      # terminal (disabled)

    # ══════════════════════════════════════════════════════════
    # BATTERY PENALTY THRESHOLDS (for reward shaping)
    # ══════════════════════════════════════════════════════════
    battery_warning_pct: float = 20.0   # warning level ≤ 20%
    battery_critical_pct: float = 10.0  # critical level ≤ 10%
    battery_emergency_pct: float = 5.0  # emergency level ≤ 5%

    # ══════════════════════════════════════════════════════════
    # FLEET MANAGEMENT POLICY
    # ══════════════════════════════════════════════════════════
    reserve_ratio: float = 0.2          # keep 20% in reserve
    min_reserve: int = 2                # minimum reserve UAVs
    incentive_deploy: float = 2.0       # reward for deploying
    incentive_recall: float = -1.0      # penalty for recalling

    # ══════════════════════════════════════════════════════════
    # BACKWARD COMPATIBILITY - Physics
    # ══════════════════════════════════════════════════════════
    @property
    def z_min(self) -> float:
        return self.z_min_m

    @property
    def z_max(self) -> float:
        return self.z_max_m

    @property
    def max_speed_xy(self) -> float:
        return self.max_speed_xy_mps

    @property
    def max_speed_z(self) -> float:
        return self.max_speed_z_mps

    @property
    def collision_radius(self) -> float:
        return self.collision_radius_m

    # ══════════════════════════════════════════════════════════
    # BACKWARD COMPATIBILITY - Battery drain (old _per_step names)
    # NOTE: Returns _per_s values (dt=1.0 → numerically same)
    # ══════════════════════════════════════════════════════════
    @property
    def drain_xy_max(self) -> float:
        """DEPRECATED: Use drain_xy_pct_per_s."""
        return self.drain_xy_pct_per_s

    @property
    def drain_xy_pct_per_step(self) -> float:
        """DEPRECATED: Use drain_xy_pct_per_s."""
        return self.drain_xy_pct_per_s

    @property
    def drain_z_up(self) -> float:
        """DEPRECATED: Use drain_z_up_pct_per_s."""
        return self.drain_z_up_pct_per_s

    @property
    def drain_z_up_pct_per_step(self) -> float:
        """DEPRECATED: Use drain_z_up_pct_per_s."""
        return self.drain_z_up_pct_per_s

    @property
    def drain_z_down(self) -> float:
        """DEPRECATED: Use drain_z_down_pct_per_s."""
        return self.drain_z_down_pct_per_s

    @property
    def drain_z_down_pct_per_step(self) -> float:
        """DEPRECATED: Use drain_z_down_pct_per_s."""
        return self.drain_z_down_pct_per_s

    @property
    def drain_idle(self) -> float:
        """DEPRECATED: Use drain_idle_pct_per_s."""
        return self.drain_idle_pct_per_s

    @property
    def drain_idle_pct_per_step(self) -> float:
        """DEPRECATED: Use drain_idle_pct_per_s."""
        return self.drain_idle_pct_per_s

    @property
    def charge_rate(self) -> float:
        """DEPRECATED: Use charge_rate_pct_per_s."""
        return self.charge_rate_pct_per_s

    @property
    def charge_rate_pct_per_step(self) -> float:
        """DEPRECATED: Use charge_rate_pct_per_s."""
        return self.charge_rate_pct_per_s

    # ══════════════════════════════════════════════════════════
    # BACKWARD COMPATIBILITY - Battery thresholds
    # ══════════════════════════════════════════════════════════
    @property
    def battery_return_threshold(self) -> float:
        return self.battery_return_pct

    @property
    def battery_ready_threshold(self) -> float:
        return self.battery_ready_pct

    @property
    def battery_dead_threshold(self) -> float:  # ✅ FIX: ADD THIS
        """BACKWARD COMPATIBILITY: Use battery_dead_pct."""
        return self.battery_dead_pct

    @property
    def battery_dead(self) -> float:
        return self.battery_dead_pct

    @property
    def battery_penalty_low(self) -> float:
        return self.battery_warning_pct

    @property
    def battery_penalty_critical(self) -> float:
        return self.battery_critical_pct

    @property
    def battery_penalty_emergency(self) -> float:
        return self.battery_emergency_pct

    @property
    def battery_critical(self) -> float:
        return self.battery_return_pct

    @property
    def battery_ready(self) -> float:
        return self.battery_ready_pct

    @property
    def battery_low_20(self) -> float:
        return self.battery_warning_pct

    @property
    def battery_critical_10(self) -> float:
        return self.battery_critical_pct

    @property
    def battery_critical_5(self) -> float:
        return self.battery_emergency_pct