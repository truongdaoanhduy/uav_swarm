"""
core package

Core systems: CoverageMap, MapGenerator, FleetManager.
"""

from core.coverage_map import CoverageMap
from core.map_generator import MapGenerator
from core.fleet_manager import FleetManager

__all__ = [
    "CoverageMap",
    "MapGenerator",
    "FleetManager",
]
