"""Nonlinear MPC controller for BlueROV2 path tracking."""

from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize

from base_controller import BaseController
from env_utils import wrap_angle


class MPCController(BaseController):
    name = "mpc"

    def __init__(self, trajectory, dynamics, dt: float = 0.1, horizon: int = 10):
        self.trajectory = trajectory
        self.dyn = dynamics
        self.dt = float(dt)
        self.N = int(horizon)

        self.u_min = -40.0
        self.u_max = 40.0
        self.prev_u = np.zeros(6, dtype=float)

        self.Q = np.diag([
            150.0, 150.0, 200.0,
            10.0, 10.0, 100.0,
            1.0, 1.0, 1.0,
            0.1, 0.1, 0.1,
        ])

        self.R = np.eye(6) * 0.01
        self.R_delta = 0.05

        self.last_metrics = self._empty_metrics()

    def reset(self):
        self.prev_u[:] = 0.0
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
            "controller_success": 0,
        }

    def predict_next_state(self, state: np.ndarray, thrust: np.ndarray) -> np.ndarray:
        eta = state[0:6].copy()
        nu = state[6:12].copy()

        phi, theta = eta[3], eta[4]

        thrust = np.asarray(thrust, dtype=float).reshape(6)
        thrust = np.clip(thrust, self.u_min, self.u_max)

        tau, _ = self.dyn._thrusters_to_tau(thrust)

        nu_rel = nu.copy()

        damping = (
            self.dyn.D_lin * nu_rel
            + self.dyn.D_quad * nu_rel * np.abs(nu_rel)
        )

        restoring = self.dyn._restoring_forces(phi, theta)
        c_rb = self.dyn._rigid_body_coriolis_force(nu)
        c_a = self.dyn._added_mass_coriolis_force(nu_rel)

        rhs = tau - c_rb - c_a - damping - restoring
        nu_dot = np.linalg.solve(self.dyn.M, rhs)

        nu_next = nu + nu_dot * self.dt

        eta_dot = self.dyn._body_to_world_kinematics(eta, nu_next)
        eta_next = eta + eta_dot * self.dt

        eta_next[3] = wrap_angle(eta_next[3])
        eta_next[4] = wrap_angle(eta_next[4])
        eta_next[5] = wrap_angle(eta_next[5])

        return np.concatenate((eta_next, nu_next))

    def cost_function(self, u_flat, current_state, t_start):
        u_sequence = u_flat.reshape((self.N, 6))
        state = current_state.copy()

        cost = 0.0
        prev_u = self.prev_u.copy()

        for i in range(self.N):
            u_i = u_sequence[i]
            state = self.predict_next_state(state, u_i)

            ref = self.trajectory.get_reference(t_start + (i + 1) * self.dt)
            ref = np.asarray(ref, dtype=float).reshape(12)

            error = state - ref
            error[3] = wrap_angle(error[3])
            error[4] = wrap_angle(error[4])
            error[5] = wrap_angle(error[5])

            u_norm = u_i / self.u_max
            du_norm = (u_i - prev_u) / self.u_max

            cost += float(error.T @ self.Q @ error)
            cost += float(u_norm.T @ self.R @ u_norm)
            cost += self.R_delta * float(np.sum(du_norm**2))

            prev_u = u_i.copy()

        if not np.isfinite(cost):
            return 1e20

        return float(cost)

    def get_action(self, obs, state, reference, t):
        wall_total_start = time.perf_counter()
        cpu_total_start = time.process_time()

        prepare_start = time.perf_counter()

        state = np.asarray(state, dtype=float).reshape(12)
        u0 = np.tile(self.prev_u, self.N)
        bounds = [(self.u_min, self.u_max)] * (self.N * 6)

        prepare_time = time.perf_counter() - prepare_start

        solver_start = time.perf_counter()

        result = minimize(
            self.cost_function,
            u0,
            args=(state, t),
            method="SLSQP",
            bounds=bounds,
            options={
                "ftol": 1e-2,
                "maxiter": 10,
                "disp": False,
            },
        )

        solver_time = time.perf_counter() - solver_start

        post_start = time.perf_counter()

        success = int(result.success and np.all(np.isfinite(result.x)))

        if success:
            action = result.x[:6]
        else:
            action = self.prev_u.copy()

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
            "controller_success": success,
        }

        return action.astype(np.float32)