"""
sensors package

Sensor modules: FOV, Comm.
"""

from sensors.fov_sensor import FOVSensor
from sensors.comm_sensor import CommSensor

__all__ = [
    "FOVSensor",
    "CommSensor",
]
