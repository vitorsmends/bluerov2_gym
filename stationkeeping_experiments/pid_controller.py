"""PID controller for BlueROV2 station keeping."""

from __future__ import annotations

import numpy as np

from base_controller import BaseController
from env_utils import wrap_angle


class PIDController(BaseController):
    """Cascaded PID-style station-keeping controller.

    The controller computes a desired generalized force vector
    [X, Y, Z, K, M, N] and maps it to direct thruster commands using the
    pseudo-inverse of the environment allocation matrix.
    """

    def __init__(
        self,
        allocation_matrix: np.ndarray,
        kp_pos=(20.0, 20.0, 25.0),
        kd_pos=(8.0, 8.0, 10.0),
        kp_att=(8.0, 8.0, 12.0),
        kd_att=(2.0, 2.0, 3.0),
        thrust_limit: float = 40.0,
    ) -> None:
        self.B = np.asarray(allocation_matrix, dtype=float)
        self.B_pinv = np.linalg.pinv(self.B)

        self.kp_pos = np.asarray(kp_pos, dtype=float)
        self.kd_pos = np.asarray(kd_pos, dtype=float)
        self.kp_att = np.asarray(kp_att, dtype=float)
        self.kd_att = np.asarray(kd_att, dtype=float)
        self.thrust_limit = float(thrust_limit)

    def get_action(self, state: np.ndarray, reference: np.ndarray, t: float) -> np.ndarray:
        pos = state[0:3]
        att = state[3:6]
        lin_vel = state[6:9]
        ang_vel = state[9:12]

        pos_ref = reference[0:3]
        att_ref = reference[3:6]
        lin_vel_ref = reference[6:9]
        ang_vel_ref = reference[9:12]

        pos_error = pos_ref - pos
        vel_error = lin_vel_ref - lin_vel

        att_error = att_ref - att
        att_error[2] = wrap_angle(att_ref[2] - att[2])
        ang_vel_error = ang_vel_ref - ang_vel

        force_cmd = self.kp_pos * pos_error + self.kd_pos * vel_error
        moment_cmd = self.kp_att * att_error + self.kd_att * ang_vel_error

        tau_cmd = np.concatenate((force_cmd, moment_cmd))

        thrust = self.B_pinv @ tau_cmd
        thrust = np.clip(thrust, -self.thrust_limit, self.thrust_limit)

        return thrust.astype(np.float32)
