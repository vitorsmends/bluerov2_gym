"""PID controller for BlueROV2 path tracking.

The controller computes a desired generalized wrench and maps it to individual
thruster commands using the pseudoinverse of the environment allocation matrix.
"""

from __future__ import annotations

import numpy as np

from base_controller import BaseController
from env_utils import wrap_angle


class PIDController(BaseController):
    name = "pid"

    def __init__(self, dynamics=None, dt: float = 0.1):
        self.dt = dt
        self.dynamics = dynamics
        self.integral = np.zeros(6, dtype=float)
        self.prev_error = np.zeros(6, dtype=float)

        self.kp = np.array([18.0, 18.0, 22.0, 4.0, 4.0, 10.0], dtype=float)
        self.ki = np.array([0.02, 0.02, 0.04, 0.0, 0.0, 0.01], dtype=float)
        self.kd = np.array([8.0, 8.0, 10.0, 1.5, 1.5, 4.0], dtype=float)

        self.thrust_limit = 40.0

    def reset(self):
        self.integral[:] = 0.0
        self.prev_error[:] = 0.0

    def _allocation_pinv(self):
        if self.dynamics is None:
            raise RuntimeError("PIDController requires env.unwrapped.dynamics.")
        return np.linalg.pinv(self.dynamics.allocation_matrix)

    def get_action(self, obs, state, reference, t):
        error = reference[0:6] - state[0:6]
        error[3] = wrap_angle(error[3])
        error[4] = wrap_angle(error[4])
        error[5] = wrap_angle(error[5])

        vel_error = reference[6:12] - state[6:12]

        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -2.0, 2.0)

        tau = self.kp * error + self.ki * self.integral + self.kd * vel_error

        thrust = self._allocation_pinv() @ tau
        thrust = np.clip(thrust, -self.thrust_limit, self.thrust_limit)
        return thrust.astype(np.float32)
