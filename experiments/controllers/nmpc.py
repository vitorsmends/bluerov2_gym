"""CasADi/IPOPT NMPC controller for BlueROV2 path tracking."""

from __future__ import annotations

import time

import casadi as ca
import numpy as np

from .base import BaseController

class NMPCController(BaseController):
    """Nonlinear model predictive controller implemented with CasADi/IPOPT."""

    name = "nmpc"

    def __init__(
        self,
        dynamics,
        dt: float = 0.1,
        reference_provider=None,
        horizon: int = 10,
        control_blocks: int = 5,
        **kwargs,
    ) -> None:
        del kwargs

        if dynamics is None:
            raise ValueError(
                "NMPCController requires a dynamics object."
            )

        if reference_provider is None:
            raise ValueError(
                "NMPCController requires a reference_provider."
            )

        if not hasattr(reference_provider, "get_reference"):
            raise TypeError(
                "reference_provider must implement "
                "get_reference(t)."
            )

        self.trajectory = reference_provider
        self.dyn = dynamics
        self.dt = float(dt)
        self.N = int(horizon)
        self.control_blocks = int(control_blocks)

        if self.N <= 0:
            raise ValueError(
                "horizon must be greater than zero."
            )

        if self.control_blocks <= 0:
            raise ValueError(
                "control_blocks must be greater than zero."
            )

        if self.N % self.control_blocks != 0:
            raise ValueError(
                "horizon must be divisible by control_blocks."
            )

        self.block_size = self.N // self.control_blocks

        self.nx = 12
        self.nu = 6
        self.ndw = 9

        self.u_min = -40.0
        self.u_max = 40.0
        self.max_delta_u = 12.0

        self.prev_u = np.zeros(
            self.nu,
            dtype=float,
        )

        self.last_block_solution = np.zeros(
            (self.control_blocks, self.nu),
            dtype=float,
        )

        self.last_metrics = self._empty_metrics()

        self.Q = np.diag(
            [
                55.555556,
                55.555556,
                80.000000,
                34.195899,
                34.195899,
                21.885376,
                6.530612,
                6.530612,
                12.800000,
                1.575747,
                1.575747,
                1.094269,
            ]
        )

        self.Qf = np.diag(
            [
                444.444444,
                444.444444,
                640.000000,
                273.567196,
                273.567196,
                175.083005,
                52.244898,
                52.244898,
                102.400000,
                12.605976,
                12.605976,
                8.754150,
            ]
        )

        self.R = np.eye(self.nu) * 0.00005
        self.R_delta = np.eye(self.nu) * 0.002431

        self.roll_soft_limit = np.deg2rad(25.0)
        self.pitch_soft_limit = np.deg2rad(25.0)
        self.attitude_soft_weight = 600.0

        self._validate_dynamics()
        self._build_casadi_solver()

    def reset(self) -> None:
        """Reset controller memory and performance metrics."""

        self.prev_u[:] = 0.0
        self.last_block_solution[:] = 0.0
        self.last_metrics = self._empty_metrics()

    def get_action(self, obs, state, reference, t) -> np.ndarray:
        """Compute the control action for the current simulation step.

        Parameters
        ----------
        obs
            Environment observation. It is retained for compatibility with
            the common controller interface.
        state
            Current BlueROV2 state.
        reference
            Current reference. The NMPC internally builds the complete
            prediction-horizon reference sequence from ``trajectory``.
        t
            Current simulation time.

        Returns
        -------
        np.ndarray
            Six-dimensional thruster command.
        """

        # These arguments are part of the common controller interface.
        # The NMPC uses the full state and generates the horizon reference
        # directly from self.trajectory.
        del obs, reference

        wall_total_start = time.perf_counter()
        cpu_total_start = time.process_time()

        success = 0

        prepare_start = time.perf_counter()

        x0 = self._pack_state(state)
        disturbance_forecast = self._forecast_disturbance_sequence()
        reference_sequence = self._build_reference_sequence(t)

        parameters = np.concatenate(
            [
                x0,
                reference_sequence.reshape(-1),
                disturbance_forecast.reshape(-1),
                self.prev_u,
            ]
        )

        initial_guess = self._initial_guess_blocks().reshape(-1)

        number_of_decision_variables = self.control_blocks * self.nu

        lower_bounds = (
            np.ones(number_of_decision_variables, dtype=float) * self.u_min
        )
        upper_bounds = (
            np.ones(number_of_decision_variables, dtype=float) * self.u_max
        )

        prepare_time = time.perf_counter() - prepare_start

        solver_start = time.perf_counter()

        try:
            solution = self.solver(
                x0=initial_guess,
                lbx=lower_bounds,
                ubx=upper_bounds,
                p=parameters,
            )

            solver_time = time.perf_counter() - solver_start

            control_blocks = np.asarray(
                solution["x"],
                dtype=float,
            ).reshape(self.control_blocks, self.nu)

            solver_stats = self.solver.stats()
            solver_succeeded = bool(solver_stats.get("success", False))

            if solver_succeeded and np.all(np.isfinite(control_blocks)):
                self.last_block_solution = control_blocks.copy()
                action = control_blocks[0].copy()
                success = 1
            else:
                action = self.prev_u.copy()

        except (RuntimeError, ValueError, TypeError):
            solver_time = time.perf_counter() - solver_start
            action = self.prev_u.copy()

        post_start = time.perf_counter()

        delta_u = np.clip(
            action - self.prev_u,
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
            "controller_frequency_hz": (
                float(1.0 / wall_total)
                if wall_total > 0.0
                else np.nan
            ),
            "controller_prepare_time_s": float(prepare_time),
            "controller_solver_time_s": float(solver_time),
            "controller_post_time_s": float(post_time),
            "controller_success": int(success),
        }

        return action.astype(np.float32)

    def get_metrics(self) -> dict:
        """Return metrics measured during the latest control iteration."""

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
            "controller_success": 0,
        }

    def _validate_dynamics(self) -> None:
        """Validate the attributes required from the numerical plant model."""

        required_attributes = (
            "allocation_matrix",
            "D_lin",
            "D_quad",
            "M_RB",
            "M_A",
            "M",
            "W",
            "B_force",
            "r_g",
            "r_b",
        )

        missing_attributes = [
            attribute
            for attribute in required_attributes
            if not hasattr(self.dyn, attribute)
        ]

        if missing_attributes:
            missing = ", ".join(missing_attributes)

            raise AttributeError(
                "The dynamics object does not provide all attributes "
                f"required by NMPCController. Missing: {missing}."
            )

    def _build_casadi_solver(self) -> None:
        """Build the symbolic NMPC nonlinear program."""

        decision_variables = self.control_blocks * self.nu

        control_blocks = ca.MX.sym(
            "U_blocks",
            decision_variables,
        )

        number_of_parameters = (
            self.nx
            + self.N * self.nx
            + self.N * self.ndw
            + self.nu
        )

        parameters = ca.MX.sym(
            "P",
            number_of_parameters,
        )

        parameter_index = 0

        state = parameters[
            parameter_index:parameter_index + self.nx
        ]
        parameter_index += self.nx

        references = parameters[
            parameter_index:parameter_index + self.N * self.nx
        ]
        references = ca.reshape(references, self.nx, self.N)
        parameter_index += self.N * self.nx

        disturbances = parameters[
            parameter_index:parameter_index + self.N * self.ndw
        ]
        disturbances = ca.reshape(
            disturbances,
            self.ndw,
            self.N,
        )
        parameter_index += self.N * self.ndw

        previous_control = parameters[
            parameter_index:parameter_index + self.nu
        ]

        state_weight = ca.DM(self.Q)
        terminal_state_weight = ca.DM(self.Qf)
        control_weight = ca.DM(self.R)
        control_rate_weight = ca.DM(self.R_delta)

        cost = 0.0

        for prediction_step in range(self.N):
            block_index = prediction_step // self.block_size

            control = control_blocks[
                block_index * self.nu:(block_index + 1) * self.nu
            ]

            state = self._casadi_dynamics_step(
                state,
                control,
                disturbances[:, prediction_step],
            )

            reference = references[:, prediction_step]
            # Trajectory translational rates are inertial; state rates are body-fixed.
            reference_body = ca.vertcat(reference[0:6],
                ca.mtimes(self._casadi_rotation(state[:6]).T, reference[6:9]),
                reference[9:12])
            error = state - reference_body

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

            control_variation = control - previous_control

            current_state_weight = (
                terminal_state_weight
                if prediction_step == self.N - 1
                else state_weight
            )

            cost += ca.mtimes(
                [
                    error.T,
                    current_state_weight,
                    error,
                ]
            )

            cost += ca.mtimes(
                [
                    control.T,
                    control_weight,
                    control,
                ]
            )

            cost += ca.mtimes(
                [
                    control_variation.T,
                    control_rate_weight,
                    control_variation,
                ]
            )

            cost += self._casadi_attitude_soft_penalty(state)

            previous_control = control

        nonlinear_program = {
            "x": control_blocks,
            "f": cost,
            "p": parameters,
        }

        solver_options = {
            "ipopt.print_level": 0,
            "print_time": False,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 20,
            "ipopt.tol": 3e-3,
            "ipopt.acceptable_tol": 1e-2,
        }

        self.solver = ca.nlpsol(
            "nmpc_solver",
            "ipopt",
            nonlinear_program,
            solver_options,
        )

    def _casadi_dynamics_step(self, state, thrust, disturbance):
        """Propagate the internal CasADi model by one sampling interval."""

        eta = state[0:6]
        nu = state[6:12]

        phi = eta[3]
        theta = eta[4]

        current_velocity = disturbance[0:3]
        wave_force = disturbance[3:9]

        allocation_matrix = ca.DM(
            np.asarray(
                self.dyn.allocation_matrix,
                dtype=float,
            )
        )

        thrust_force = ca.mtimes(
            allocation_matrix,
            thrust,
        )

        relative_velocity = ca.vertcat(
            nu[0] - current_velocity[0],
            nu[1] - current_velocity[1],
            nu[2] - current_velocity[2],
            nu[3],
            nu[4],
            nu[5],
        )

        linear_damping = ca.DM(
            np.asarray(
                self.dyn.D_lin,
                dtype=float,
            ).reshape(6, 1)
        )

        quadratic_damping = ca.DM(
            np.asarray(
                self.dyn.D_quad,
                dtype=float,
            ).reshape(6, 1)
        )

        damping_force = (
            linear_damping * relative_velocity
            + quadratic_damping
            * relative_velocity
            * ca.fabs(relative_velocity)
        )

        restoring_force = self._casadi_restoring_forces(
            phi,
            theta,
        )

        rigid_body_mass = ca.DM(
            np.asarray(
                self.dyn.M_RB,
                dtype=float,
            )
        )

        added_mass = ca.DM(
            np.asarray(
                self.dyn.M_A,
                dtype=float,
            )
        )

        total_mass = ca.DM(
            np.asarray(
                self.dyn.M,
                dtype=float,
            )
        )

        rigid_body_coriolis = ca.mtimes(
            self._casadi_spatial_cross_force(nu),
            ca.mtimes(rigid_body_mass, nu),
        )

        added_mass_coriolis = ca.mtimes(
            self._casadi_spatial_cross_force(
                relative_velocity
            ),
            ca.mtimes(
                added_mass,
                relative_velocity,
            ),
        )

        right_hand_side = (
            thrust_force
            + wave_force
            - rigid_body_coriolis
            - added_mass_coriolis
            - damping_force
            - restoring_force
        )

        acceleration = ca.solve(
            total_mass,
            right_hand_side,
        )

        next_velocity = (
            nu
            + self.dt * acceleration
        )

        eta_derivative = (
            self._casadi_body_to_world_kinematics(
                eta,
                next_velocity,
            )
        )

        next_eta = (
            eta
            + self.dt * eta_derivative
        )

        next_eta = ca.vertcat(
            next_eta[0],
            next_eta[1],
            next_eta[2],
            self._casadi_wrap_angle(next_eta[3]),
            self._casadi_wrap_angle(next_eta[4]),
            self._casadi_wrap_angle(next_eta[5]),
        )

        return ca.vertcat(
            next_eta,
            next_velocity,
        )

    def _casadi_restoring_forces(self, phi, theta):
        vertical_body = ca.vertcat(-ca.sin(theta),
                                  ca.cos(theta)*ca.sin(phi),
                                  ca.cos(theta)*ca.cos(phi))
        gravity = -float(self.dyn.W) * vertical_body
        buoyancy = float(self.dyn.B_force) * vertical_body
        moment = ca.cross(ca.DM(self.dyn.r_g), gravity) + ca.cross(ca.DM(self.dyn.r_b), buoyancy)
        return -ca.vertcat(gravity + buoyancy, moment)

    @staticmethod
    def _casadi_rotation(eta):
        phi, theta, psi = eta[3], eta[4], eta[5]
        c, s = ca.cos(psi), ca.sin(psi)
        ct, st = ca.cos(theta), ca.sin(theta)
        cp, sp = ca.cos(phi), ca.sin(phi)
        return ca.vertcat(ca.horzcat(c*ct, c*st*sp-s*cp, c*st*cp+s*sp),
                          ca.horzcat(s*ct, s*st*sp+c*cp, s*st*cp-c*sp),
                          ca.horzcat(-st, ct*sp, ct*cp))

    @staticmethod
    def _casadi_skew(vector):
        return ca.vertcat(
            ca.horzcat(
                0.0,
                -vector[2],
                vector[1],
            ),
            ca.horzcat(
                vector[2],
                0.0,
                -vector[0],
            ),
            ca.horzcat(
                -vector[1],
                vector[0],
                0.0,
            ),
        )

    def _casadi_spatial_cross_force(self, velocity):
        linear_skew = self._casadi_skew(velocity[0:3])
        angular_skew = self._casadi_skew(velocity[3:6])
        zero_matrix = ca.DM.zeros(3, 3)

        return ca.vertcat(
            ca.horzcat(
                angular_skew,
                zero_matrix,
            ),
            ca.horzcat(
                linear_skew,
                angular_skew,
            ),
        )

    def _casadi_body_to_world_kinematics(self, eta, nu):
        phi = eta[3]
        theta = eta[4]
        psi = eta[5]

        cosine_psi = ca.cos(psi)
        sine_psi = ca.sin(psi)

        cosine_theta = ca.cos(theta)
        sine_theta = ca.sin(theta)

        cosine_phi = ca.cos(phi)
        sine_phi = ca.sin(phi)

        u = nu[0]
        v = nu[1]
        w = nu[2]

        p = nu[3]
        q = nu[4]
        r = nu[5]

        x_dot = (
            u * cosine_psi * cosine_theta
            + v
            * (
                cosine_psi * sine_theta * sine_phi
                - sine_psi * cosine_phi
            )
            + w
            * (
                cosine_psi * sine_theta * cosine_phi
                + sine_psi * sine_phi
            )
        )

        y_dot = (
            u * sine_psi * cosine_theta
            + v
            * (
                sine_psi * sine_theta * sine_phi
                + cosine_psi * cosine_phi
            )
            + w
            * (
                sine_psi * sine_theta * cosine_phi
                - cosine_psi * sine_phi
            )
        )

        z_dot = (
            -u * sine_theta
            + v * cosine_theta * sine_phi
            + w * cosine_theta * cosine_phi
        )

        safe_cosine_theta = ca.if_else(ca.fabs(cosine_theta) < 1e-6,
                                      ca.if_else(cosine_theta < 0, -1e-6, 1e-6),
                                      cosine_theta)

        roll_rate = (
            p
            + (q * sine_phi + r * cosine_phi)
            * ca.tan(theta)
        )

        pitch_rate = (
            q * cosine_phi
            - r * sine_phi
        )

        yaw_rate = (
            q * sine_phi
            + r * cosine_phi
        ) / safe_cosine_theta

        return ca.vertcat(
            x_dot,
            y_dot,
            z_dot,
            roll_rate,
            pitch_rate,
            yaw_rate,
        )

    def _casadi_attitude_soft_penalty(self, state):
        roll_violation = ca.fmax(
            0.0,
            ca.fabs(state[3]) - self.roll_soft_limit,
        )

        pitch_violation = ca.fmax(
            0.0,
            ca.fabs(state[4]) - self.pitch_soft_limit,
        )

        return self.attitude_soft_weight * (
            roll_violation**2
            + pitch_violation**2
        )

    @staticmethod
    def _casadi_wrap_angle(angle):
        return ca.atan2(
            ca.sin(angle),
            ca.cos(angle),
        )

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

        return np.asarray(
            state,
            dtype=float,
        ).reshape(12).copy()

    def _forecast_disturbance_sequence(self) -> np.ndarray:
        """Return a zero disturbance forecast over the prediction horizon."""

        return np.zeros(
            (self.N, self.ndw),
            dtype=float,
        )

    def _build_reference_sequence(
        self,
        initial_time: float,
    ) -> np.ndarray:
        references = np.zeros(
            (self.N, self.nx),
            dtype=float,
        )

        for prediction_step in range(self.N):
            reference_time = (
                initial_time
                + (prediction_step + 1) * self.dt
            )

            references[prediction_step] = np.asarray(
                self.trajectory.get_reference(reference_time),
                dtype=float,
            ).reshape(self.nx)

        return references

    def _initial_guess_blocks(self) -> np.ndarray:
        initial_guess = np.zeros(
            (self.control_blocks, self.nu),
            dtype=float,
        )

        if self.control_blocks == 1:
            initial_guess[0] = self.last_block_solution[0]
        else:
            initial_guess[:-1] = self.last_block_solution[1:]
            initial_guess[-1] = self.last_block_solution[-1]

        if np.allclose(initial_guess, 0.0):
            initial_guess[:] = self.prev_u

        return np.clip(
            initial_guess,
            self.u_min,
            self.u_max,
        )