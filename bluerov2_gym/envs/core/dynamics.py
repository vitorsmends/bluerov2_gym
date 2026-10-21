import numpy as np

from bluerov2_gym.envs.core.config_utils import load_yaml

class Dynamics:
    """
    BlueROV2 6-DoF dynamics with direct thruster commands.

    Disturbance model:
        1. Directional JONSWAP + Linear Wave Theory:
           - wave elevation eta(x, y, t)
           - particle velocities u_c, v_c, w_c

        2. Relative velocity:
           nu_r = nu - nu_c
           where nu_c = [u_c, v_c, w_c, 0, 0, 0]^T

        3. Direct environmental wave load:
           tau_wave = [X_wave, Y_wave, Z_wave, K_wave, M_wave, N_wave]^T

    State:
        {
            'x', 'y', 'z',
            'roll', 'pitch', 'yaw',
            'u', 'v', 'w',
            'p', 'q', 'r'
        }

    Action:
        np.array([T1, T2, T3, T4, T5, T6])
    """

    MODEL_VERSION = "zup-airy-coriolis-v2"

    def __init__(
        self,
        jonswap_params: dict | None = None,
        dynamics_config: dict | None = None,
    ):
        if isinstance(dynamics_config, (str, bytes)):
            dynamics_config = load_yaml(dynamics_config)
        cfg = dynamics_config or {}

        self.dt = float(cfg.get("dt", 0.1))
        self.rho = float(cfg.get("rho", 1000.0))
        self.g = float(cfg.get("g", 9.81))

        vehicle = cfg.get("vehicle", {})
        self.base_mass = float(vehicle.get("base_mass", 14.8))
        self.prop_mass = float(vehicle.get("prop_mass", 0.07))
        self.n_thrusters = int(vehicle.get("n_thrusters", 6))
        self.m = self.base_mass + self.n_thrusters * self.prop_mass

        dimensions = vehicle.get("dimensions", {})
        self.x_size = float(dimensions.get("x", 0.40))
        self.y_size = float(dimensions.get("y", 0.25))
        self.z_size = float(dimensions.get("z", 0.10))

        buoyancy = cfg.get("buoyancy", {})
        correction_factor = float(buoyancy.get("correction_factor", 1.01))
        self.buoyant_correction = correction_factor * (
            self.m / (self.rho * self.x_size * self.y_size * self.z_size)
        ) ** (1.0 / 3.0)

        self.volume = self.x_size * self.y_size * self.z_size * self.buoyant_correction**3

        default_rg = [0.0, 0.0, -self.buoyant_correction * self.z_size]
        default_rb = [0.0, 0.0, -self.buoyant_correction * self.z_size / 4.0]
        self.r_g = np.asarray(buoyancy.get("center_of_gravity", default_rg), dtype=float)
        self.r_b = np.asarray(buoyancy.get("center_of_buoyancy", default_rb), dtype=float)

        self.z_cog = self.r_g[2]
        self.z_cob = self.r_b[2]
        self.coBM = abs(self.z_cob - self.z_cog)
        self.W = self.m * self.g
        self.B_force = self.rho * self.g * self.volume

        inertia = vehicle.get("inertia", {})
        self.Ixx = float(inertia.get("Ixx", 5.2539))
        self.Ixy = float(inertia.get("Ixy", 0.0144))
        self.Ixz = float(inertia.get("Ixz", 0.3341))
        self.Iyy = float(inertia.get("Iyy", 7.9420))
        self.Iyz = float(inertia.get("Iyz", 0.0260))
        self.Izz = float(inertia.get("Izz", 6.9123))
        self.I_g = np.array([
            [self.Ixx, self.Ixy, self.Ixz],
            [self.Ixy, self.Iyy, self.Iyz],
            [self.Ixz, self.Iyz, self.Izz],
        ], dtype=float)
        reference = inertia.get("reference", "center_of_gravity")
        shift = -self.m * self._skew(self.r_g) @ self._skew(self.r_g)
        if reference == "body_origin":
            self.I_O = self.I_g.copy()
            self.I_g = self.I_O - shift
        elif reference == "center_of_gravity":
            self.I_O = self.I_g + shift
        else:
            raise ValueError("inertia.reference must be center_of_gravity or body_origin")
        if np.min(np.linalg.eigvalsh(self.I_g)) <= 0:
            raise ValueError("Inertia at the center of gravity must be positive definite")
        self.M_RB = self._rigid_body_mass_matrix()

        hydrodynamics = cfg.get("hydrodynamics", {})
        self.added_mass = np.asarray(
            hydrodynamics.get("added_mass", [5.5, 12.7, 14.57, 0.12, 0.12, 0.12]),
            dtype=float,
        )
        self.M_A = np.diag(self.added_mass)
        self.M = self.M_RB + self.M_A
        self.D_lin = np.asarray(
            hydrodynamics.get("linear_damping", [25.15, 7.364, 17.955, 10.888, 20.761, 3.744]),
            dtype=float,
        )
        self.D_quad = np.asarray(
            hydrodynamics.get("quadratic_damping", [33.8, 54.26875, 73.37135, 40.0, 40.0, 40.0]),
            dtype=float,
        )

        thrusters = cfg.get("thrusters", {})
        self.thruster_min = float(thrusters.get("min", -40.0))
        self.thruster_max = float(thrusters.get("max", 40.0))
        self.allocation_matrix = np.asarray(
            thrusters.get("allocation_matrix", [
                [0.70710678, 0.70710678, 0.70710678, 0.70710678, 0.0, 0.0],
                [0.70710678, -0.70710678, -0.70710678, 0.70710678, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
                [0.05126524, -0.05126524, -0.05126524, 0.05126524, -0.1105, -0.1105],
                [-0.05126524, -0.05126524, -0.05126524, -0.05126524, -0.0025, 0.0025],
                [0.16652365, -0.16652365, 0.17500893, -0.17500893, 0.0, 0.0],
            ]),
            dtype=float,
        )

        default_jonswap = {
            "Hs": 2.0, "Tp": 12.0, "gamma": 3.3, "N": 64,
            "wave_dir": (0.5, 0.5), "directional_spread_deg": 25.0,
            "water_depth": 30.0, "x_eval": 0.0, "y_eval": 0.0, "z_eval": -2.0,
            "scale": 0.5, "max_current": 0.7, "alpha_wave": 0.02,
            "noise_std": 0.01, "alpha_noise": 0.3, "alpha_js": 0.0081,
            "enable_wave_force": True, "wave_force_gain": (35.0, 35.0, 50.0),
            "wave_force_application_point": (0.0, 0.0, 0.05),
            "max_wave_force": 25.0, "max_wave_moment": 8.0, "seed": 42,
        }
        self.jonswap_params = default_jonswap.copy()
        self.jonswap_params.update(cfg.get("jonswap", {}))
        if jonswap_params is not None:
            self.jonswap_params.update(jonswap_params)

        self._init_jonswap(**self.jonswap_params)

    # ============================================================
    # Matrix utilities
    # ============================================================

    @staticmethod
    def _skew(a):
        a = np.asarray(a, dtype=float).reshape(3)

        return np.array(
            [
                [0.0, -a[2], a[1]],
                [a[2], 0.0, -a[0]],
                [-a[1], a[0], 0.0],
            ],
            dtype=float,
        )

    def _rigid_body_mass_matrix(self):
        I3 = np.eye(3)
        S_rg = self._skew(self.r_g)

        upper = np.hstack((self.m * I3, -self.m * S_rg))
        lower = np.hstack((self.m * S_rg, self.I_O))

        return np.vstack((upper, lower))

    def _spatial_cross_force(self, nu):
        v = np.asarray(nu[0:3], dtype=float)
        omega = np.asarray(nu[3:6], dtype=float)

        S_v = self._skew(v)
        S_w = self._skew(omega)

        upper = np.hstack((S_w, np.zeros((3, 3))))
        lower = np.hstack((S_v, S_w))

        return np.vstack((upper, lower))

    def _rigid_body_coriolis_force(self, nu):
        return self._spatial_cross_force(nu) @ (self.M_RB @ nu)

    def _added_mass_coriolis_force(self, nu_rel):
        return self._spatial_cross_force(nu_rel) @ (self.M_A @ nu_rel)

    # ============================================================
    # Directional JONSWAP + Linear Wave Theory
    # ============================================================

    def _solve_wave_number(self, omega, depth):
        """
        Solve dispersion relation:
            omega^2 = g k tanh(k d)
        """
        k = max(omega**2 / self.g, 1e-6)

        for _ in range(30):
            kd = k * depth
            tanh_kd = np.tanh(kd)
            sech2_kd = 1.0 / np.cosh(kd) ** 2

            f = self.g * k * tanh_kd - omega**2
            df = self.g * tanh_kd + self.g * k * depth * sech2_kd

            if abs(df) < 1e-12:
                break

            k_new = k - f / df

            if k_new <= 0.0:
                k_new = 0.5 * k

            if abs(k_new - k) < 1e-10:
                k = k_new
                break

            k = k_new

        return float(k)

    def _init_jonswap(
        self,
        Hs,
        Tp,
        gamma,
        N,
        wave_dir,
        directional_spread_deg,
        water_depth,
        x_eval,
        y_eval,
        z_eval,
        scale,
        max_current,
        alpha_wave,
        noise_std,
        alpha_noise,
        alpha_js,
        enable_wave_force,
        wave_force_gain,
        wave_force_application_point,
        max_wave_force,
        max_wave_moment,
        seed,
    ):
        self.Hs = float(Hs)
        self.Tp = float(Tp)
        self.gamma = float(gamma)
        self.N = int(N)
        if not np.isfinite(self.Hs) or self.Hs < 0:
            raise ValueError("Hs must be finite and non-negative.")
        if not np.isfinite(self.Tp) or self.Tp <= 0 or self.N < 1:
            raise ValueError("Tp must be finite and positive, and N >= 1.")
        if not np.isfinite(self.gamma) or not 1 <= self.gamma <= 7:
            raise ValueError("JONSWAP gamma must be between 1 and 7.")

        self.wave_dir = np.array(wave_dir, dtype=float)

        if np.linalg.norm(self.wave_dir) < 1e-6:
            self.wave_dir = np.array([1.0, 0.0], dtype=float)

        self.wave_dir /= np.linalg.norm(self.wave_dir)
        self.mean_wave_angle = float(np.arctan2(self.wave_dir[1], self.wave_dir[0]))
        self.directional_spread = np.deg2rad(float(directional_spread_deg))

        self.water_depth = float(water_depth)
        self.x_eval = float(x_eval)
        self.y_eval = float(y_eval)
        self.z_eval = float(z_eval)

        if self.water_depth <= 0.0:
            raise ValueError("water_depth must be positive.")

        if self.z_eval > 0.0:
            raise ValueError("z_eval must be <= 0.0, with z=0 at the still water line.")

        if self.z_eval < -self.water_depth:
            raise ValueError("z_eval cannot be below seabed: z_eval >= -water_depth.")

        self.scale = float(scale)
        self.max_current = float(max_current)
        self.alpha_wave = float(alpha_wave)
        self.noise_std = float(noise_std)
        self.alpha_noise = float(alpha_noise)
        self.alpha_js = float(alpha_js)

        self.enable_wave_force = bool(enable_wave_force)
        self.wave_force_gain = np.asarray(wave_force_gain, dtype=float).reshape(3)
        self.wave_force_application_point = np.asarray(
            wave_force_application_point,
            dtype=float,
        ).reshape(3)

        self.max_wave_force = float(max_wave_force)
        self.max_wave_moment = float(max_wave_moment)

        self._rng = np.random.default_rng(seed)

        self._t = 0.0
        self._nu_c_filt = np.zeros(3, dtype=float)
        self._noise_filt = np.zeros(3, dtype=float)

        self._last_eta_wave = 0.0
        self._last_nu_wave_raw = np.zeros(3, dtype=float)
        self._last_tau_wave = np.zeros(6, dtype=float)

        wp = 2.0 * np.pi / self.Tp
        dw = 3.0 * wp / self.N
        # Fossen MSS waveSpectrum, case 7 (JONSWAP with DNV correction).
        # Hs determines spectral energy; alpha_js is retained only for legacy
        # configuration compatibility and is not an independent scale here.
        self.alpha_js = (5.0 / 16.0) * self.Hs**2 * wp**4 / self.g**2
        conversion_factor = 1.0 - 0.287 * np.log(self.gamma)

        self._waves = []

        for i in range(1, self.N + 1):
            omega = i * dw
            sigma = 0.07 if omega <= wp else 0.09

            r = np.exp(
                -((omega - wp) ** 2)
                / (2.0 * sigma**2 * wp**2)
            )

            S = (
                conversion_factor * self.alpha_js
                * self.g**2
                * omega**-5
                * np.exp(-1.25 * (wp / omega) ** 4)
                * self.gamma**r
            )

            A = np.sqrt(2.0 * S * dw)
            phase = self._rng.uniform(0.0, 2.0 * np.pi)

            if self.directional_spread > 1e-12:
                theta = self.mean_wave_angle + self._rng.uniform(
                    -self.directional_spread,
                    self.directional_spread,
                )
            else:
                theta = self.mean_wave_angle

            k = self._solve_wave_number(omega, self.water_depth)

            self._waves.append(
                {
                    "omega": float(omega),
                    "A": float(A),
                    "phase": float(phase),
                    "theta": float(theta),
                    "k": float(k),
                    "S": float(S),
                }
            )

    def set_jonswap_params(self, **kwargs):
        self.jonswap_params.update(kwargs)
        self._init_jonswap(**self.jonswap_params)

    def get_jonswap_params(self):
        return self.jonswap_params.copy()

    def _compute_wave_force(self, nu_c):
        """
        Approximate direct environmental wave load.

        This is a compact Morison/drag-inspired approximation:
            F_wave = K_wave * nu_c * |nu_c|

        Moments are generated by applying the force at a point offset from
        the vehicle reference frame:
            M_wave = r_wave x F_wave
        """
        tau_wave = np.zeros(6, dtype=float)

        if not self.enable_wave_force:
            return tau_wave

        force = self.wave_force_gain * nu_c * np.abs(nu_c)

        force_norm = np.linalg.norm(force)

        if force_norm > self.max_wave_force:
            force = force / force_norm * self.max_wave_force

        moment = np.cross(self.wave_force_application_point, force)
        moment_norm = np.linalg.norm(moment)

        if moment_norm > self.max_wave_moment:
            moment = moment / moment_norm * self.max_wave_moment

        tau_wave[0:3] = force
        tau_wave[3:6] = moment

        return tau_wave

    def _jonswap_disturbance(self, eta=None):
        """
        Computes:
            eta_wave
            nu_current = [u_c, v_c, w_c]
            tau_wave = [X_wave, Y_wave, Z_wave, K_wave, M_wave, N_wave]
        """
        self._t += self.dt

        eta_wave = 0.0
        up = 0.0
        vp = 0.0
        wp = 0.0

        x = self.x_eval
        y = self.y_eval
        z = self.z_eval
        d = self.water_depth

        for wave in self._waves:
            omega = wave["omega"]
            A = wave["A"]
            phase = wave["phase"]
            theta = wave["theta"]
            k = wave["k"]

            spatial_phase = k * (x * np.cos(theta) + y * np.sin(theta))
            arg = spatial_phase - omega * self._t + phase

            # Airy velocities for elevation amplitude A, finite water depth.
            cosh_ratio = np.cosh(k * (z + d)) / np.sinh(k * d)
            sinh_ratio = np.sinh(k * (z + d)) / np.sinh(k * d)

            eta_wave += A * np.cos(arg)

            up += A * omega * np.cos(theta) * cosh_ratio * np.cos(arg)
            vp += A * omega * np.sin(theta) * cosh_ratio * np.cos(arg)
            wp += A * omega * sinh_ratio * np.sin(arg)

        nu_wave = np.array([up, vp, wp], dtype=float)
        nu_wave *= self.scale

        mag = np.linalg.norm(nu_wave)

        if mag > self.max_current:
            nu_wave = nu_wave / mag * self.max_current

        noise = self.noise_std * self._rng.standard_normal(3)

        self._noise_filt = (
            (1.0 - self.alpha_noise) * self._noise_filt
            + self.alpha_noise * noise
        )

        nu_c_raw = nu_wave + self._noise_filt

        self._nu_c_filt = (
            (1.0 - self.alpha_wave) * self._nu_c_filt
            + self.alpha_wave * nu_c_raw
        )

        # Noise, scaling and filtering are defined in the inertial z-up frame.
        # Transform before body-frame drag and empirical body-frame wave loads.
        rotation = np.eye(3) if eta is None else self.body_to_world_rotation(eta)
        nu_c_body = rotation.T @ self._nu_c_filt
        tau_wave = self._compute_wave_force(nu_c_body)

        self._last_eta_wave = float(eta_wave)
        self._last_nu_wave_raw = nu_wave.copy()
        self._last_tau_wave = tau_wave.copy()

        return nu_c_body, tau_wave.copy()

    # ============================================================
    # Forces and kinematics
    # ============================================================

    def _thrusters_to_tau(self, action):
        thrust = np.asarray(action, dtype=float)

        if thrust.shape != (6,):
            raise ValueError(
                f"Action must have shape (6,), got {thrust.shape}. "
                "Expected [T1, T2, T3, T4, T5, T6]."
            )

        thrust = np.clip(thrust, self.thruster_min, self.thruster_max)
        tau = self.allocation_matrix @ thrust

        return tau, thrust

    def _restoring_forces(self, phi, theta):
        # g is on the left of the dynamic equation, hence minus the applied
        # gravity/buoyancy wrench. Both inertial and body frames are z-up.
        vertical_body = np.array([-np.sin(theta),
                                  np.cos(theta) * np.sin(phi),
                                  np.cos(theta) * np.cos(phi)])
        gravity = -self.W * vertical_body
        buoyancy = self.B_force * vertical_body
        moment = np.cross(self.r_g, gravity) + np.cross(self.r_b, buoyancy)
        return -np.r_[gravity + buoyancy, moment]

    @staticmethod
    def body_to_world_rotation(eta):
        """Right-handed Rz(yaw) Ry(pitch) Rx(roll); world and body z point up."""
        phi, theta, psi = np.asarray(eta)[3:6]
        c, s = np.cos(psi), np.sin(psi)
        ct, st = np.cos(theta), np.sin(theta)
        cp, sp = np.cos(phi), np.sin(phi)
        return np.array([[c*ct, c*st*sp-s*cp, c*st*cp+s*sp],
                         [s*ct, s*st*sp+c*cp, s*st*cp-c*sp],
                         [-st, ct*sp, ct*cp]])

    def _body_to_world_kinematics(self, eta, nu):
        phi, theta, psi = eta[3], eta[4], eta[5]

        c_psi, s_psi = np.cos(psi), np.sin(psi)
        c_th, s_th = np.cos(theta), np.sin(theta)
        c_phi, s_phi = np.cos(phi), np.sin(phi)

        u, v, w = nu[0], nu[1], nu[2]
        p, q, r = nu[3], nu[4], nu[5]

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

        c_th_safe = c_th

        if abs(c_th_safe) < 1e-6:
            c_th_safe = np.sign(c_th_safe) * 1e-6 if c_th_safe != 0 else 1e-6

        d_phi = p + (q * s_phi + r * c_phi) * np.tan(theta)
        d_theta = q * c_phi - r * s_phi
        d_psi = (q * s_phi + r * c_phi) / c_th_safe

        return np.array(
            [dx, dy, dz, d_phi, d_theta, d_psi],
            dtype=float,
        )

    @staticmethod
    def _wrap_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    # ============================================================
    # Step
    # ============================================================

    def step(self, state, action):
        eta = np.array(
            [
                state["x"],
                state["y"],
                state["z"],
                state["roll"],
                state["pitch"],
                state["yaw"],
            ],
            dtype=float,
        )

        nu = np.array(
            [
                state["u"],
                state["v"],
                state["w"],
                state["p"],
                state["q"],
                state["r"],
            ],
            dtype=float,
        )

        phi, theta = eta[3], eta[4]
        nu_c, tau_wave = self._jonswap_disturbance(eta)

        tau, saturated_thrust = self._thrusters_to_tau(action)

        nu_rel = nu.copy()
        nu_rel[0:3] -= nu_c

        damping = (
            self.D_lin * nu_rel
            + self.D_quad * nu_rel * np.abs(nu_rel)
        )

        g_eta = self._restoring_forces(phi, theta)

        c_rb = self._rigid_body_coriolis_force(nu)
        c_a = self._added_mass_coriolis_force(nu_rel)

        rhs = tau + tau_wave - c_rb - c_a - damping - g_eta

        nu_dot = np.linalg.solve(self.M, rhs)

        new_nu = nu + nu_dot * self.dt

        eta_dot = self._body_to_world_kinematics(eta, new_nu)
        new_eta = eta + eta_dot * self.dt

        new_eta[3] = self._wrap_angle(new_eta[3])
        new_eta[4] = self._wrap_angle(new_eta[4])
        new_eta[5] = self._wrap_angle(new_eta[5])

        keys_eta = ["x", "y", "z", "roll", "pitch", "yaw"]
        keys_nu = ["u", "v", "w", "p", "q", "r"]

        for i, key in enumerate(keys_eta):
            state[key] = float(new_eta[i])

        for i, key in enumerate(keys_nu):
            state[key] = float(new_nu[i])

        state["thrusters"] = saturated_thrust.copy()
        state["tau"] = tau.copy()
        state["tau_wave"] = tau_wave.copy()
        state["tau_total"] = (tau + tau_wave).copy()

        state["wave_elevation"] = float(self._last_eta_wave)
        state["nu_current"] = nu_c.copy()
        state["nu_current_world"] = self._nu_c_filt.copy()
        state["nu_wave_raw"] = self._last_nu_wave_raw.copy()
        state["nu_rel"] = nu_rel.copy()
        state["nu_dot"] = nu_dot.copy()

        state["coriolis_rb"] = c_rb.copy()
        state["coriolis_added"] = c_a.copy()
        state["damping"] = damping.copy()
        state["restoring"] = g_eta.copy()

        return state

    def reset(self, jonswap_params: dict | None = None):
        if jonswap_params is not None:
            self.jonswap_params.update(jonswap_params)
            self._init_jonswap(**self.jonswap_params)
            return

        self._t = 0.0
        self._nu_c_filt[:] = 0.0
        self._noise_filt[:] = 0.0
        self._last_eta_wave = 0.0
        self._last_nu_wave_raw[:] = 0.0
        self._last_tau_wave[:] = 0.0
