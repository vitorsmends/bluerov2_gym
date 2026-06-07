"""Discrete LQR controller for BlueROV2 path tracking."""

from __future__ import annotations

import time
import numpy as np

from base_controller import BaseController
from env_utils import wrap_angle


class LQRController(BaseController):
    name = "lqr"

    def __init__(self, dt: float = 0.1):
        self.dt = float(dt)

        self.u_min = -40.0
        self.u_max = 40.0
        self.max_delta_u = 8.0

        self.u_eq = np.array(
            [0.000680, 0.000737, 0.000679, 0.000738, 1.566768, -1.566737],
            dtype=float,
        )

        self.K = np.array(
            [
                [23.311965, 24.044708, 0.000053, 0.524578, -0.201984, 27.950770,
                 20.598171, 28.835218, 0.000097, 2.084931, -1.378985, 23.201644],
                [23.312131, -24.044719, 0.000454, -0.523170, -0.229425, -27.949055,
                 20.595187, -28.835249, 0.000510, -2.084383, -1.480717, -23.200696],
                [23.311952, -23.400507, 0.000044, -0.355832, -0.200855, 28.535274,
                 20.598313, -27.965310, 0.000086, -0.913926, -1.376856, 23.600032],
                [23.312144, 23.400495, 0.000464, 0.357241, -0.230554, -28.533559,
                 20.595045, 27.965279, 0.000521, 0.914473, -1.482846, -23.599084],
                [0.003129, 1.386395, 41.439519, -20.319484, -0.098271, -0.156110,
                 -0.003128, 1.104590, 43.826236, -15.196167, -0.078583, -0.585363],
                [-0.002733, 1.386403, -41.439297, -20.319647, 0.079796, -0.156363,
                 0.001508, 1.104600, -43.826004, -15.196287, 0.017458, -0.585620],
            ],
            dtype=float,
        )

        self.prev_u = self.u_eq.copy()
        self.last_metrics = self._empty_metrics()

    def reset(self):
        self.prev_u = self.u_eq.copy()
        self.last_metrics = self._empty_metrics()

    @staticmethod
    def _empty_metrics():
        return {
            "controller_wall_time_s": 0.0,
            "controller_cpu_time_s": 0.0,
            "controller_frequency_hz": np.nan,
            "controller_prepare_time_s": 0.0,
            "controller_solver_time_s": 0.0,
            "controller_post_time_s": 0.0,
            "controller_success": 1,
        }

    @staticmethod
    def _state_error(state, reference):
        state = np.asarray(state, dtype=float).reshape(12)
        reference = np.asarray(reference, dtype=float).reshape(12)

        error = state - reference

        error[3] = wrap_angle(error[3])
        error[4] = wrap_angle(error[4])
        error[5] = wrap_angle(error[5])

        return error

    def get_action(self, obs, state, reference, t):
        wall_total_start = time.perf_counter()
        cpu_total_start = time.process_time()

        prepare_start = time.perf_counter()
        error = self._state_error(state, reference)
        prepare_time = time.perf_counter() - prepare_start

        solver_start = time.perf_counter()
        action_cmd = self.u_eq - self.K @ error
        solver_time = time.perf_counter() - solver_start

        post_start = time.perf_counter()

        action_cmd = np.clip(action_cmd, self.u_min, self.u_max)

        delta_u = np.clip(
            action_cmd - self.prev_u,
            -self.max_delta_u,
            self.max_delta_u,
        )

        action = self.prev_u + delta_u
        action = np.clip(action, self.u_min, self.u_max)

        self.prev_u = action.copy()

        post_time = time.perf_counter() - post_start

        wall_total = time.perf_counter() - wall_total_start
        cpu_total = time.process_time() - cpu_total_start

        self.last_metrics = {
            "controller_wall_time_s": float(wall_total),
            "controller_cpu_time_s": float(cpu_total),
            "controller_frequency_hz": float(1.0 / wall_total) if wall_total > 0.0 else np.nan,
            "controller_prepare_time_s": float(prepare_time),
            "controller_solver_time_s": float(solver_time),
            "controller_post_time_s": float(post_time),
            "controller_success": 1,
        }

        return action.astype(np.float32)