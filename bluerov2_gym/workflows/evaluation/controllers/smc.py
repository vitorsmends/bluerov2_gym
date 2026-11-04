"""Sliding Mode Controller for BlueROV2 path tracking."""

from __future__ import annotations

import time

import numpy as np

from .base import BaseController
from ..env_utils import wrap_angle


# Sliding surface gains
LAMBDA = [1.5, 1.5, 2.0, 1.0, 1.0, 1.0]

# Boundary layer thickness
EPS = [0.2, 0.2, 0.1, 0.1, 0.1, 0.1]

# Initial adaptive gains
K_INIT = [10.0, 10.0, 20.0, 5.0, 5.0, 5.0]

# Adaptation rates
K_BAR = [5.0, 5.0, 10.0, 2.0, 2.0, 2.0]

# Adaptation dead zone
MU = [0.05, 0.05, 0.05, 0.02, 0.02, 0.02]

# Minimum adaptive gain
ALPHA = [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]


class SMCController(BaseController):

    name = "smc"

    def __init__(
        self,
        dynamics,
        dt: float = 0.1,
        reference_provider=None,
        **kwargs,
    ):

        del reference_provider, kwargs

        self.dt = float(dt)
        self.dynamics = dynamics

        self._validate_dynamics()

        self.lam = np.asarray(LAMBDA, dtype=float)
        self.eps = np.asarray(EPS, dtype=float)
        self.alpha = np.asarray(ALPHA, dtype=float)
        self.kbar = np.asarray(K_BAR, dtype=float)
        self.mu = np.asarray(MU, dtype=float)
        self.K = np.asarray(K_INIT, dtype=float)

        self.wrench_sat = 20.0
        self.thrust_limit = 40.0
        self.max_delta_thrust = 8.0

        self.prev_thrust = np.zeros(6, dtype=float)

        self.allocation_pinv = np.linalg.pinv(
            self.dynamics.allocation_matrix
        )

        self.last_metrics = self._empty_metrics()

    def _validate_dynamics(self) -> None:

        required = ("allocation_matrix",)

        missing = [
            attr
            for attr in required
            if not hasattr(self.dynamics, attr)
        ]

        if missing:
            raise AttributeError(
                f"Dynamics object missing attributes: {', '.join(missing)}"
            )

    def reset(self) -> None:

        self.K = np.asarray(K_INIT, dtype=float)
        self.prev_thrust[:] = 0.0
        self.last_metrics = self._empty_metrics()

    def get_metrics(self) -> dict:

        return self.last_metrics.copy()

    @staticmethod
    def _empty_metrics() -> dict:

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
    def _world_to_body_xyz(vec_world, yaw):

        c = np.cos(yaw)
        s = np.sin(yaw)

        return np.array(
            [
                vec_world[0] * c + vec_world[1] * s,
                -vec_world[0] * s + vec_world[1] * c,
                vec_world[2],
            ],
            dtype=float,
        )

    def _smc_wrench(self, error_body, velocity_error_body):

        s_surface = velocity_error_body + self.lam * error_body

        wrench = np.zeros(6, dtype=float)

        for i in range(6):

            adaptation_rate = (
                self.kbar[i]
                * np.sign(np.abs(s_surface[i]) - self.mu[i])
            )

            self.K[i] += self.dt * adaptation_rate

            self.K[i] = np.clip(
                self.K[i],
                self.alpha[i],
                60.0,
            )

            if abs(s_surface[i]) > self.eps[i]:
                wrench[i] = self.K[i] * np.sign(s_surface[i])
            else:
                wrench[i] = self.K[i] * (
                    s_surface[i] / self.eps[i]
                )

        return np.clip(
            wrench,
            -self.wrench_sat,
            self.wrench_sat,
        )

    def get_action(
        self,
        obs,
        state,
        reference,
        t,
    ) -> np.ndarray:

        del obs, t

        wall_total_start = time.perf_counter()
        cpu_total_start = time.process_time()

        prepare_start = time.perf_counter()

        state_vector = np.asarray(
            state,
            dtype=float,
        ).reshape(12)

        reference_vector = np.asarray(
            reference,
            dtype=float,
        ).reshape(12)

        eta = state_vector[:6]
        nu = state_vector[6:]

        eta_ref = reference_vector[:6]
        nu_ref = reference_vector[6:]

        yaw = eta[5]

        prepare_time = (
            time.perf_counter() - prepare_start
        )

        solver_start = time.perf_counter()

        pos_error_world = eta_ref[:3] - eta[:3]
        pos_error_body = self._world_to_body_xyz(
            pos_error_world,
            yaw,
        )

        attitude_error = eta_ref[3:] - eta[3:]

        attitude_error[0] = wrap_angle(attitude_error[0])
        attitude_error[1] = wrap_angle(attitude_error[1])
        attitude_error[2] = wrap_angle(attitude_error[2])

        error_body = np.concatenate(
            (
                pos_error_body,
                attitude_error,
            )
        )

        linear_velocity_error_world = (
            nu_ref[:3] - nu[:3]
        )

        linear_velocity_error_body = (
            self._world_to_body_xyz(
                linear_velocity_error_world,
                yaw,
            )
        )

        angular_velocity_error = (
            nu_ref[3:] - nu[3:]
        )

        velocity_error_body = np.concatenate(
            (
                linear_velocity_error_body,
                angular_velocity_error,
            )
        )

        wrench = self._smc_wrench(
            error_body,
            velocity_error_body,
        )

        thrust_cmd = self.allocation_pinv @ wrench

        solver_time = (
            time.perf_counter() - solver_start
        )

        post_start = time.perf_counter()

        thrust_cmd = np.clip(
            thrust_cmd,
            -self.thrust_limit,
            self.thrust_limit,
        )

        delta = np.clip(
            thrust_cmd - self.prev_thrust,
            -self.max_delta_thrust,
            self.max_delta_thrust,
        )

        thrust = self.prev_thrust + delta

        thrust = np.clip(
            thrust,
            -self.thrust_limit,
            self.thrust_limit,
        )

        self.prev_thrust = thrust.copy()

        post_time = (
            time.perf_counter() - post_start
        )

        wall_total = (
            time.perf_counter() - wall_total_start
        )

        cpu_total = (
            time.process_time() - cpu_total_start
        )

        self.last_metrics = {
            "controller_wall_time_s": float(wall_total),
            "controller_cpu_time_s": float(cpu_total),
            "controller_frequency_hz": (
                float(1.0 / wall_total)
                if wall_total > 0.0
                else np.nan
            ),
            "controller_prepare_time_s": float(
                prepare_time
            ),
            "controller_solver_time_s": float(
                solver_time
            ),
            "controller_post_time_s": float(
                post_time
            ),
            "controller_success": 1,
        }

        return thrust.astype(np.float32)