"""
Abstract backend interface for SAR UAV environment.

Separates physics simulation from RL logic.
Allows switching between different physics engines (logic/pybullet/isaac).
"""

from __future__ import annotations
from typing import Optional
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

# ✅ FIX 4.1: REMOVED sys.path hack
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class BaseBackend(ABC):
    """
    Abstract interface for environment physics backend.

    Concrete implementations:
    - LogicBackend: Pure Python simulation (CPU)
    - PyBulletBackend: PyBullet physics (CPU, realistic)
    - IsaacBackend: IsaacLab physics (GPU, parallel)
    """

    @abstractmethod
    def reset(self, map_data: dict[str, Any]) -> None:
        """
        Initialize/reset backend with map from MapGenerator.

        Builds entities (UAVs, victims, stations, obstacles) from map_data
        and resets internal state (coverage, fleet manager, etc).

        Args:
            map_data: Dictionary containing:
                - "stations":     list of station dicts
                - "debris":       list of debris dicts
                - "danger_zones": list of danger zone dicts
                - "victims":      list of victim dicts
                - "uav_spawns":   list of spawn position dicts (from MapGenerator)
        """
        pass

    @abstractmethod
    def apply_actions(self, actions: dict[int, np.ndarray]) -> None:
        """
        Apply velocity commands to UAVs.

        For ACTIVE UAVs:              apply action to set velocity.
        For RETURNING/DEPLOYING UAVs: auto-navigate towards target.
        For CHARGING/DISABLED UAVs:   no movement.

        Args:
            actions: {uav_id: action_array [vx, vy, vz] ∈ [-1, 1]³}
                     Actions are normalized, will be scaled by backend.
        """
        pass

    @abstractmethod
    def step_physics(self) -> None:
        """
        Update physics state: movement, collision, battery.

        Execution order:
            1. Battery drain (ACTIVE / RETURNING / DEPLOYING)
            2. Battery charge (CHARGING at station)
            3. Collision detection (UAV vs Debris)
            4. State transitions (battery dead → DISABLED)

        NOTE: No reward logic here. Backend is pure simulation.
        """
        pass

    @abstractmethod
    def step_world(self) -> None:
        """
        Update world state: victims, coverage, fleet management.

        Execution order (temporal correctness):
            1. Fleet manager step (UAV state transitions)
            2. Mobile victims movement
            3. Coverage map update (mark explored cells)
            4. Victim detection (FOV scan at current positions)

        NOTE: Order matters for temporal consistency.
              Detection uses current-step positions (no lag).
        """
        pass

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """
        Get current state of all entities.

        Used by base_env to compute observations and rewards.

        Returns:
            {
                "uavs":         list[UAV]
                "victims":      list[BaseVictim]
                "stations":     list[ChargingStation]
                "obstacles":    list[Debris | DangerZone]
                "coverage_map": CoverageMap
                "fleet_manager": FleetManager
            }
        """
        pass