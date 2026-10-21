"""PID controller for BlueROV2 path tracking."""

from __future__ import annotations

import time
import numpy as np

from .base import BaseController
from ..env_utils import wrap_angle


class PIDController(BaseController):
    name = "pid"

    def __init__(
        self,
        dynamics=None,
        dt: float = 0.1,
        reference_provider=None,
        **kwargs,
    ):
        del reference_provider, kwargs

        self.dt = float(dt)
        self.dynamics = dynamics

        self.integral = np.zeros(6, dtype=float)
        self.prev_thrust = np.zeros(6, dtype=float)

        self.kp = np.array([4.10, 5.70, 7.60, 1.70, 2.45, 1.75], dtype=float)
        self.ki = np.array([0.12, 0.17, 0.23, 0.0, 0.0, 0.05], dtype=float)
        self.kd = np.array([2.80, 1.85, 3.20, 0.55, 1.05, 0.45], dtype=float)

        self.integral_limit = np.array(
            [1.0, 1.0, 0.8, 0.3, 0.3, 0.5],
            dtype=float,
        )

        self.thrust_limit = 40.0
        self.max_delta_thrust = 8.0

        self._pinv = None
        self.last_metrics = self._empty_metrics()

    def reset(self):
        self.integral[:] = 0.0
        self.prev_thrust[:] = 0.0
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

    def _allocation_pinv(self):
        if self.dynamics is None:
            raise RuntimeError("PIDController requires env.unwrapped.dynamics.")

        if self._pinv is None:
            self._pinv = np.linalg.pinv(self.dynamics.allocation_matrix)

        return self._pinv

    @staticmethod
    def _rotation_world_to_body(yaw: float) -> np.ndarray:
        c = np.cos(yaw)
        s = np.sin(yaw)

        return np.array(
            [
                [c, s, 0.0],
                [-s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def get_action(self, obs, state, reference, t):
        wall_total_start = time.perf_counter()
        cpu_total_start = time.process_time()

        prepare_start = time.perf_counter()

        state = np.asarray(state, dtype=float).reshape(12)
        reference = np.asarray(reference, dtype=float).reshape(12)

        eta = state[0:6]
        nu = state[6:12]

        eta_ref = reference[0:6]
        nu_ref = reference[6:12]

        yaw = eta[5]
        R_wb = self._rotation_world_to_body(yaw)

        prepare_time = time.perf_counter() - prepare_start

        solver_start = time.perf_counter()

        pos_error_world = eta_ref[0:3] - eta[0:3]
        pos_error_body = R_wb @ pos_error_world

        att_error = eta_ref[3:6] - eta[3:6]
        att_error[0] = wrap_angle(att_error[0])
        att_error[1] = wrap_angle(att_error[1])
        att_error[2] = wrap_angle(att_error[2])

        error = np.concatenate((pos_error_body, att_error))

        vel_error_world = nu_ref[0:3] - nu[0:3]
        vel_error_body = R_wb @ vel_error_world

        ang_vel_error = nu_ref[3:6] - nu[3:6]
        vel_error = np.concatenate((vel_error_body, ang_vel_error))

        self.integral += error * self.dt
        self.integral = np.clip(
            self.integral,
            -self.integral_limit,
            self.integral_limit,
        )

        tau = self.kp * error + self.ki * self.integral + self.kd * vel_error

        thrust_cmd = self._allocation_pinv() @ tau

        solver_time = time.perf_counter() - solver_start

        post_start = time.perf_counter()

        thrust_cmd = np.clip(thrust_cmd, -self.thrust_limit, self.thrust_limit)

        delta = np.clip(
            thrust_cmd - self.prev_thrust,
            -self.max_delta_thrust,
            self.max_delta_thrust,
        )

        thrust = self.prev_thrust + delta
        thrust = np.clip(thrust, -self.thrust_limit, self.thrust_limit)

        self.prev_thrust = thrust.copy()

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

        return thrust.astype(np.float32)