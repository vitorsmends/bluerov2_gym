"""Reference trajectories for BlueROV2 path-tracking experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class FigureEightTrajectory:
    """Time-varying figure-eight reference trajectory.

    The returned reference state follows the 12-state convention used by the
    BlueROV2 Gym environment:

        [x, y, z, roll, pitch, yaw, u, v, w, p, q, r]

    Positions are expressed in the inertial frame, while the linear velocity
    reference is expressed in the inertial frame and later used only to build
    the virtual tracking observation for the PPO policy or controller errors.
    """

    radius: float = 1.0
    speed: float = 0.15
    z_target: float = -0.5
    z_ramp_time: float = 10.0

    def get_reference(self, t: float) -> np.ndarray:
        ts = t * self.speed

        x = self.radius * math.sin(ts)
        y = self.radius * math.sin(ts) * math.cos(ts)

        if t < self.z_ramp_time:
            z = (self.z_target / self.z_ramp_time) * t
            vz = self.z_target / self.z_ramp_time
        else:
            z = self.z_target
            vz = 0.0

        vx = self.radius * math.cos(ts) * self.speed
        vy = self.radius * (math.cos(ts) ** 2 - math.sin(ts) ** 2) * self.speed
        yaw = math.atan2(vy, vx)

        return np.array(
            [x, y, z, 0.0, 0.0, yaw, vx, vy, vz, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )


# Alias used by the runners for clarity.
PathTrackingReference = FigureEightTrajectory
