"""CasADi/IPOPT NMPC controller for BlueROV2 path tracking.

Class name is kept as NMPCController.

This implementation:
    - uses CasADi symbolic prediction;
    - uses IPOPT instead of scipy SLSQP;
    - keeps move blocking: N=10, control_blocks=5;
    - predicts JONSWAP current as an exogenous parameter;
    - commands the six thrusters directly.
"""

from __future__ import annotations

import copy
import numpy as np
import casadi as ca

from base_controller import BaseController


class NMPCController(BaseController):
    name = "nmpc"

    def __init__(
        self,
        trajectory,
        dynamics,
        dt: float = 0.1,
        horizon: int = 10,
        control_blocks: int = 5,
    ):
        self.trajectory = trajectory
        self.dyn = dynamics
        self.dt = float(dt)

        self.N = int(horizon)
        self.control_blocks = int(control_blocks)

        if self.N % self.control_blocks != 0:
            raise ValueError("horizon must be divisible by control_blocks.")

        self.block_size = self.N // self.control_blocks

        self.nx = 12
        self.nu = 6
        self.nc = 3

        self.u_min = -40.0
        self.u_max = 40.0
        self.max_delta_u = 12.0

        self.prev_u = np.zeros(6, dtype=float)
        self.last_block_solution = np.zeros((self.control_blocks, 6), dtype=float)

        self.Q = np.diag([
            55.555556, 55.555556, 80.000000,
            34.195899, 34.195899, 21.885376,
            6.530612, 6.530612, 12.800000,
            1.575747, 1.575747, 1.094269,
        ])

        self.Qf = np.diag([
            444.444444, 444.444444, 640.000000,
            273.567196, 273.567196, 175.083005,
            52.244898, 52.244898, 102.400000,
            12.605976, 12.605976, 8.754150,
        ])

        self.R = np.eye(6) * 0.00005
        self.R_delta = np.eye(6) * 0.002431

        self.roll_soft_limit = np.deg2rad(25.0)
        self.pitch_soft_limit = np.deg2rad(25.0)
        self.attitude_soft_weight = 600.0

        self._build_casadi_solver()

    def reset(self):
        self.prev_u[:] = 0.0
        self.last_block_solution[:] = 0.0

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def get_action(self, obs, state, reference, t):
        x0 = self._pack_state(state)
        current_forecast = self._forecast_current_sequence()
        ref_sequence = self._build_reference_sequence(t)

        p = np.concatenate([
            x0,
            ref_sequence.reshape(-1),
            current_forecast.reshape(-1),
            self.prev_u,
        ])

        u0 = self._initial_guess_blocks().reshape(-1)

        lbx = np.ones(self.control_blocks * 6) * self.u_min
        ubx = np.ones(self.control_blocks * 6) * self.u_max

        try:
            sol = self.solver(
                x0=u0,
                lbx=lbx,
                ubx=ubx,
                p=p,
            )

            u_blocks = np.array(sol["x"]).reshape(self.control_blocks, 6)

            if np.all(np.isfinite(u_blocks)):
                self.last_block_solution = u_blocks.copy()
                action = u_blocks[0].copy()
            else:
                action = self.prev_u.copy()

        except Exception:
            action = self.prev_u.copy()

        delta_u = np.clip(
            action - self.prev_u,
            -self.max_delta_u,
            self.max_delta_u,
        )

        action = self.prev_u + delta_u
        action = np.clip(action, self.u_min, self.u_max)

        self.prev_u = action.copy()

        return action.astype(np.float32)

    # ------------------------------------------------------------
    # CasADi solver
    # ------------------------------------------------------------

    def _build_casadi_solver(self):
        U_blocks = ca.MX.sym("U_blocks", self.control_blocks * self.nu)

        # Parameter vector:
        # x0: 12
        # ref: N * 12
        # current: N * 3
        # previous input: 6
        n_params = self.nx + self.N * self.nx + self.N * self.nc + self.nu
        P = ca.MX.sym("P", n_params)

        idx = 0

        x = P[idx:idx + self.nx]
        idx += self.nx

        refs = P[idx:idx + self.N * self.nx]
        refs = ca.reshape(refs, self.nx, self.N)
        idx += self.N * self.nx

        currents = P[idx:idx + self.N * self.nc]
        currents = ca.reshape(currents, self.nc, self.N)
        idx += self.N * self.nc

        prev_u = P[idx:idx + self.nu]

        Q = ca.DM(self.Q)
        Qf = ca.DM(self.Qf)
        R = ca.DM(self.R)
        R_delta = ca.DM(self.R_delta)

        cost = 0

        for k in range(self.N):
            block_idx = k // self.block_size
            u_k = U_blocks[
                block_idx * self.nu:(block_idx + 1) * self.nu
            ]

            nu_c_k = currents[:, k]

            x = self._casadi_dynamics_step(x, u_k, nu_c_k)

            ref_k = refs[:, k]
            error = x - ref_k

            error = ca.vertcat(
                error[0],
                error[1],
                error[2],
                self._casadi_wrap_angle(error[3]),
                self._casadi_wrap_angle(error[4]),
                self._casadi_wrap_angle(error[5]),
                error[6],
                error[7],
                error[8],
                error[9],
                error[10],
                error[11],
            )

            du = u_k - prev_u

            Qk = Qf if k == self.N - 1 else Q

            cost += ca.mtimes([error.T, Qk, error])
            cost += ca.mtimes([u_k.T, R, u_k])
            cost += ca.mtimes([du.T, R_delta, du])
            cost += self._casadi_attitude_soft_penalty(x)

            prev_u = u_k

        nlp = {
            "x": U_blocks,
            "f": cost,
            "p": P,
        }

        opts = {
            "ipopt.print_level": 0,
            "print_time": False,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 20,
            "ipopt.tol": 3e-3,
            "ipopt.acceptable_tol": 1e-2,
            "ipopt.warm_start_init_point": "yes",
        }

        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    # ------------------------------------------------------------
    # Symbolic model
    # ------------------------------------------------------------

    def _casadi_dynamics_step(self, x, thrust, nu_current):
        eta = x[0:6]
        nu = x[6:12]

        phi = eta[3]
        theta = eta[4]

        tau = ca.mtimes(ca.DM(self.dyn.allocation_matrix), thrust)

        nu_rel = ca.vertcat(
            nu[0] - nu_current[0],
            nu[1] - nu_current[1],
            nu[2] - nu_current[2],
            nu[3],
            nu[4],
            nu[5],
        )

        D_lin = ca.DM(self.dyn.D_lin)
        D_quad = ca.DM(self.dyn.D_quad)

        damping = D_lin * nu_rel + D_quad * nu_rel * ca.fabs(nu_rel)

        restoring = self._casadi_restoring_forces(phi, theta)

        c_rb = self._casadi_spatial_cross_force(nu) @ (ca.DM(self.dyn.M_RB) @ nu)
        c_a = self._casadi_spatial_cross_force(nu_rel) @ (ca.DM(self.dyn.M_A) @ nu_rel)

        rhs = tau - c_rb - c_a - damping - restoring

        nu_dot = ca.solve(ca.DM(self.dyn.M), rhs)

        nu_next = nu + self.dt * nu_dot

        eta_dot = self._casadi_body_to_world_kinematics(eta, nu_next)
        eta_next = eta + self.dt * eta_dot

        eta_next = ca.vertcat(
            eta_next[0],
            eta_next[1],
            eta_next[2],
            self._casadi_wrap_angle(eta_next[3]),
            self._casadi_wrap_angle(eta_next[4]),
            self._casadi_wrap_angle(eta_next[5]),
        )

        return ca.vertcat(eta_next, nu_next)

    def _casadi_restoring_forces(self, phi, theta):
        weight_minus_buoyancy = self.dyn.W - self.dyn.B_force
        coBM = self.dyn.coBM
        W = self.dyn.W

        return ca.vertcat(
            weight_minus_buoyancy * ca.sin(theta),
            -weight_minus_buoyancy * ca.cos(theta) * ca.sin(phi),
            -weight_minus_buoyancy * ca.cos(theta) * ca.cos(phi),
            coBM * W * ca.cos(theta) * ca.sin(phi),
            coBM * W * ca.sin(theta),
            0.0,
        )

    @staticmethod
    def _casadi_skew(a):
        return ca.vertcat(
            ca.horzcat(0.0, -a[2], a[1]),
            ca.horzcat(a[2], 0.0, -a[0]),
            ca.horzcat(-a[1], a[0], 0.0),
        )

    def _casadi_spatial_cross_force(self, nu):
        v = nu[0:3]
        omega = nu[3:6]

        S_v = self._casadi_skew(v)
        S_w = self._casadi_skew(omega)

        Z = ca.DM.zeros(3, 3)

        upper = ca.horzcat(S_w, S_v)
        lower = ca.horzcat(Z, S_w)

        return ca.vertcat(upper, lower)

    def _casadi_body_to_world_kinematics(self, eta, nu):
        phi = eta[3]
        theta = eta[4]
        psi = eta[5]

        c_psi = ca.cos(psi)
        s_psi = ca.sin(psi)
        c_th = ca.cos(theta)
        s_th = ca.sin(theta)
        c_phi = ca.cos(phi)
        s_phi = ca.sin(phi)

        u = nu[0]
        v = nu[1]
        w = nu[2]
        p = nu[3]
        q = nu[4]
        r = nu[5]

        dx = (
            u * c_psi * c_th
            + v * (c_psi * s_th * s_phi - s_psi * c_phi)
            + w * (c_psi * s_th * c_phi + s_psi * s_phi)
        )

        dy = (
            u * s_psi * c_th
            + v * (s_psi * s_th * s_phi + c_psi * c_phi)
            + w * (s_psi * s_th * c_phi - c_psi * s_phi)
        )

        dz = (
            -u * s_th
            + v * c_th * s_phi
            + w * c_th * c_phi
        )

        # Avoid division instability near pitch = +-90 deg.
        c_th_safe = c_th + 1e-6

        d_phi = p + (q * s_phi + r * c_phi) * ca.tan(theta)
        d_theta = q * c_phi - r * s_phi
        d_psi = (q * s_phi + r * c_phi) / c_th_safe

        return ca.vertcat(dx, dy, dz, d_phi, d_theta, d_psi)

    def _casadi_attitude_soft_penalty(self, x):
        roll = x[3]
        pitch = x[4]

        roll_violation = ca.fmax(0.0, ca.fabs(roll) - self.roll_soft_limit)
        pitch_violation = ca.fmax(0.0, ca.fabs(pitch) - self.pitch_soft_limit)

        return self.attitude_soft_weight * (
            roll_violation**2 + pitch_violation**2
        )

    @staticmethod
    def _casadi_wrap_angle(angle):
        return ca.atan2(ca.sin(angle), ca.cos(angle))

    # ------------------------------------------------------------
    # Numeric helpers
    # ------------------------------------------------------------

    @staticmethod
    def _pack_state(state) -> np.ndarray:
        if isinstance(state, dict):
            return np.array(
                [
                    state["x"],
                    state["y"],
                    state["z"],
                    state["roll"],
                    state["pitch"],
                    state["yaw"],
                    state["u"],
                    state["v"],
                    state["w"],
                    state["p"],
                    state["q"],
                    state["r"],
                ],
                dtype=float,
            )

        return np.asarray(state, dtype=float).reshape(12).copy()

    def _forecast_current_sequence(self) -> np.ndarray:
        dyn_copy = copy.deepcopy(self.dyn)

        forecast = np.zeros((self.N, 3), dtype=float)

        for k in range(self.N):
            forecast[k] = dyn_copy._jonswap_current()

        return forecast

    def _build_reference_sequence(self, t_start: float) -> np.ndarray:
        refs = np.zeros((self.N, 12), dtype=float)

        for k in range(self.N):
            ref = self.trajectory.get_reference(t_start + (k + 1) * self.dt)
            refs[k] = np.asarray(ref, dtype=float).reshape(12)

        return refs

    def _initial_guess_blocks(self) -> np.ndarray:
        u0 = np.zeros((self.control_blocks, 6), dtype=float)

        if self.control_blocks == 1:
            u0[0] = self.last_block_solution[0]
        else:
            u0[:-1] = self.last_block_solution[1:]
            u0[-1] = self.last_block_solution[-1]

        if np.allclose(u0, 0.0):
            u0[:] = self.prev_u

        return np.clip(u0, self.u_min, self.u_max)