"""MPC controller for BlueROV2 station keeping."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from base_controller import BaseController
from env_utils import wrap_angle


class MPCController(BaseController):
    """Nonlinear MPC using direct thruster commands.

    The prediction model reuses the environment Dynamics object so that the MPC
    internal model remains consistent with the numerical simulator.
    """

    def __init__(
        self,
        dynamics,
        dt: float = 0.1,
        horizon: int = 8,
        thrust_limit: float = 40.0,
    ) -> None:
        self.dyn = dynamics
        self.dt = float(dt)
        self.N = int(horizon)
        self.thrust_limit = float(thrust_limit)
        self.prev_u = np.zeros(6, dtype=float)

        self.Q = np.diag(
            [
                120.0,
                120.0,
                180.0,
                10.0,
                10.0,
                80.0,
                1.0,
                1.0,
                1.0,
                0.1,
                0.1,
                0.1,
            ]
        )
        self.R = np.eye(6) * 0.01
        self.R_delta = 0.05

    def reset(self) -> None:
        self.prev_u[:] = 0.0

    def _predict_next_state(self, state: np.ndarray, thrust: np.ndarray) -> np.ndarray:
        eta = state[0:6].copy()
        nu = state[6:12].copy()
        phi, theta = eta[3], eta[4]

        thrust = np.asarray(thrust, dtype=float).reshape(6)
        thrust = np.clip(thrust, -self.thrust_limit, self.thrust_limit)

        tau, _ = self.dyn._thrusters_to_tau(thrust)

        # The MPC prediction assumes no future current preview. The real
        # environment still applies the stochastic ocean disturbance.
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

    def _cost(self, u_flat: np.ndarray, current_state: np.ndarray, reference: np.ndarray) -> float:
        u_seq = u_flat.reshape((self.N, 6))
        state = current_state.copy()
        cost = 0.0
        prev_u = self.prev_u.copy()

        for k in range(self.N):
            u_k = u_seq[k]
            state = self._predict_next_state(state, u_k)

            error = state - reference
            error[3] = wrap_angle(error[3])
            error[4] = wrap_angle(error[4])
            error[5] = wrap_angle(error[5])

            u_norm = u_k / self.thrust_limit
            du_norm = (u_k - prev_u) / self.thrust_limit

            cost += float(error.T @ self.Q @ error)
            cost += float(u_norm.T @ self.R @ u_norm)
            cost += float(self.R_delta * np.sum(du_norm**2))

            prev_u = u_k.copy()

        return float(cost)

    def get_action(self, state: np.ndarray, reference: np.ndarray, t: float) -> np.ndarray:
        u0 = np.tile(self.prev_u, self.N)
        bounds = [(-self.thrust_limit, self.thrust_limit)] * (self.N * 6)

        result = minimize(
            self._cost,
            u0,
            args=(state, reference),
            method="SLSQP",
            bounds=bounds,
            options={"ftol": 1e-2, "maxiter": 8, "disp": False},
        )

        if result.success:
            action = result.x[:6]
        else:
            action = self.prev_u.copy()

        action = np.clip(action, -self.thrust_limit, self.thrust_limit)
        self.prev_u = action.copy()

        return action.astype(np.float32)
