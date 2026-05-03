"""
env.backends package

Physics backends: Logic, PyBullet, IsaacLab.
"""

from env_setup.backends.base_backend import BaseBackend
from env_setup.backends.logic_backend import LogicBackend

# PyBullet (Phase 4 - optional)
try:
    from env_setup.backends.pybullet_backend import PyBulletBackend
    PYBULLET_AVAILABLE = True
except ImportError:
    PyBulletBackend = None
    PYBULLET_AVAILABLE = False

# IsaacLab (Phase 4 - optional)
try:
    from env_setup.backends.isaac_backend import IsaacBackend
    ISAAC_AVAILABLE = True
except ImportError:
    IsaacBackend = None
    ISAAC_AVAILABLE = False

__all__ = [
    "BaseBackend",
    "LogicBackend",
    "PyBulletBackend",
    "IsaacBackend",
    "PYBULLET_AVAILABLE",
    "ISAAC_AVAILABLE",
]
