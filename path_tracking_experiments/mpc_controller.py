"""Nonlinear MPC controller for BlueROV2 path tracking.

The optimizer directly commands the six thrusters and predicts motion using the
same dynamics object used by the Gym environment.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from base_controller import BaseController
from env_utils import wrap_angle


class MPCController(BaseController):
    name = "mpc"

    def __init__(self, trajectory, dynamics, dt: float = 0.1, horizon: int = 10):
        self.trajectory = trajectory
        self.dyn = dynamics
        self.dt = dt
        self.N = horizon

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

    def reset(self):
        self.prev_u[:] = 0.0

    def predict_next_state(self, state: np.ndarray, thrust: np.ndarray) -> np.ndarray:
        eta = state[0:6].copy()
        nu = state[6:12].copy()
        phi, theta = eta[3], eta[4]

        thrust = np.asarray(thrust, dtype=float).reshape(6)
        thrust = np.clip(thrust, self.u_min, self.u_max)

        tau, _ = self.dyn._thrusters_to_tau(thrust)

        # The MPC prediction does not forecast the stochastic current.
        # The real environment still includes the disturbance.
        nu_rel = nu.copy()

        damping = self.dyn.D_lin * nu_rel + self.dyn.D_quad * nu_rel * np.abs(nu_rel)
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
            error = state - ref
            error[3] = wrap_angle(error[3])
            error[4] = wrap_angle(error[4])
            error[5] = wrap_angle(error[5])

            u_norm = u_i / self.u_max
            du_norm = (u_i - prev_u) / self.u_max

            cost += float(error.T @ self.Q @ error)
            cost += float(u_norm.T @ self.R @ u_norm)
            cost += self.R_delta * float(np.sum(du_norm ** 2))

            prev_u = u_i.copy()

        return float(cost)

    def get_action(self, obs, state, reference, t):
        u0 = np.tile(self.prev_u, self.N)
        bounds = [(self.u_min, self.u_max)] * (self.N * 6)

        result = minimize(
            self.cost_function,
            u0,
            args=(state, t),
            method="SLSQP",
            bounds=bounds,
            options={"ftol": 1e-2, "maxiter": 10, "disp": False},
        )

        if result.success:
            action = result.x[:6]
        else:
            action = self.prev_u.copy()

        action = np.clip(action, self.u_min, self.u_max)
        self.prev_u = action.copy()
        return action.astype(np.float32)
