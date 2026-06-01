import numpy as np


class Dynamics:
    """
    BlueROV2 6-DoF dynamics with direct thruster commands.

    State:
        {
            'x', 'y', 'z',
            'roll', 'pitch', 'yaw',
            'u', 'v', 'w',
            'p', 'q', 'r'
        }

    Action:
        np.array([T1, T2, T3, T4, T5, T6])

    where each Ti is the thrust command in Newtons.
    """

    def __init__(self):
        self.dt = 0.1
        self.rho = 1000.0
        self.g = 9.81

        # ------------------------------------------------------------
        # Physical parameters from BlueROV2 Xacro
        # ------------------------------------------------------------
        self.base_mass = 14.8
        self.prop_mass = 0.07
        self.n_thrusters = 6

        self.m = self.base_mass + self.n_thrusters * self.prop_mass

        self.x_size = 0.40
        self.y_size = 0.25
        self.z_size = 0.10

        # Same buoyancy correction used in the Xacro
        self.buoyant_correction = 1.01 * (
            self.m / (self.rho * self.x_size * self.y_size * self.z_size)
        ) ** (1.0 / 3.0)

        self.volume = (
            self.x_size
            * self.y_size
            * self.z_size
            * self.buoyant_correction**3
        )

        # Center of buoyancy and gravity from Xacro
        self.z_cob = -self.buoyant_correction * self.z_size / 4.0
        self.z_cog = -self.buoyant_correction * self.z_size

        # Vertical distance between CG and CB
        self.coBM = abs(self.z_cob - self.z_cog)

        self.W = self.m * self.g
        self.B_force = self.rho * self.g * self.volume

        # ------------------------------------------------------------
        # Rigid-body inertia from Xacro + added mass from hydrodynamics.xacro
        # ------------------------------------------------------------
        self.Ixx = 5.2539
        self.Iyy = 7.9420
        self.Izz = 6.9123

        self.added_mass = np.array([
            5.5,
            12.7,
            14.57,
            0.12,
            0.12,
            0.12
        ], dtype=float)

        self.M_diag = np.array([
            self.m,
            self.m,
            self.m,
            self.Ixx,
            self.Iyy,
            self.Izz
        ], dtype=float) + self.added_mass

        # ------------------------------------------------------------
        # Hydrodynamic damping from hydrodynamics.xacro
        # Plugin values are negative, but here we store positive damping.
        # damping = D_lin * nu_rel + D_quad * nu_rel * abs(nu_rel)
        # ------------------------------------------------------------
        self.D_lin = np.array([
            25.15,
            7.364,
            17.955,
            10.888,
            20.761,
            3.744
        ], dtype=float)

        self.D_quad = np.array([
            33.8,
            54.26875,
            73.37135,
            40.0,
            40.0,
            40.0
        ], dtype=float)

        # ------------------------------------------------------------
        # Thruster allocation matrix from thrusters.xacro
        # action = [T1, T2, T3, T4, T5, T6]
        # tau = [X, Y, Z, K, M, N]
        # ------------------------------------------------------------
        self.thruster_min = -40.0
        self.thruster_max = 40.0

        self.allocation_matrix = np.array([
            [ 0.70710678,  0.70710678,  0.70710678,  0.70710678,  0.0,     0.0    ],
            [ 0.70710678, -0.70710678, -0.70710678,  0.70710678,  0.0,     0.0    ],
            [ 0.0,         0.0,         0.0,         0.0,         1.0,    -1.0    ],
            [ 0.05126524, -0.05126524, -0.05126524,  0.05126524, -0.1105, -0.1105 ],
            [-0.05126524, -0.05126524, -0.05126524, -0.05126524, -0.0025,  0.0025 ],
            [ 0.16652365, -0.16652365,  0.17500893, -0.17500893,  0.0,     0.0    ],
        ], dtype=float)

        # ------------------------------------------------------------
        # JONSWAP + filtered Gaussian noise disturbance
        # ------------------------------------------------------------
        self._init_jonswap(
            Hs=1.0,
            Tp=6.0,
            gamma=3.3,
            N=64,
            wave_dir=(0.5, 0.5),
            scale=0.5,
            max_current=0.7,
            alpha_wave=0.02,
            noise_std=0.01,
            alpha_noise=0.3,
            seed=42,
        )

    def _init_jonswap(
        self,
        Hs,
        Tp,
        gamma,
        N,
        wave_dir,
        scale,
        max_current,
        alpha_wave,
        noise_std,
        alpha_noise,
        seed,
    ):
        self.wave_dir = np.array(wave_dir, dtype=float)

        if np.linalg.norm(self.wave_dir) < 1e-6:
            self.wave_dir = np.array([1.0, 0.0])

        self.wave_dir /= np.linalg.norm(self.wave_dir)

        self.scale = scale
        self.max_current = max_current
        self.alpha_wave = alpha_wave
        self.noise_std = noise_std
        self.alpha_noise = alpha_noise
        self._rng = np.random.default_rng(seed)

        self._t = 0.0
        self._nu_c_filt = np.zeros(3)
        self._noise_filt = np.zeros(3)

        g = self.g
        wp = 2.0 * np.pi / Tp
        dw = 3.0 * wp / N
        alpha_js = 0.076 * Hs**2

        self._waves = []

        for i in range(1, N + 1):
            omega = i * dw
            sigma = 0.07 if omega <= wp else 0.09

            r = np.exp(
                -((omega - wp) ** 2)
                / (2.0 * sigma**2 * wp**2)
            )

            S = (
                alpha_js
                * g**2
                * omega**-5
                * np.exp(-1.25 * (wp / omega) ** 4)
                * gamma**r
            )

            a = np.sqrt(2.0 * S * dw)
            phase = self._rng.uniform(0.0, 2.0 * np.pi)

            self._waves.append((omega, a, phase))

    def _jonswap_current(self):
        self._t += self.dt

        u_wave = 0.0

        for omega, a, phase in self._waves:
            u_wave += a * omega * np.cos(omega * self._t + phase)

        nu_wave = np.array([
            u_wave * self.wave_dir[0],
            u_wave * self.wave_dir[1],
            0.0
        ])

        nu_wave *= self.scale

        mag = np.linalg.norm(nu_wave)

        if mag > self.max_current:
            nu_wave = nu_wave / mag * self.max_current

        noise = self.noise_std * self._rng.standard_normal(3)
        noise[2] = 0.0

        self._noise_filt = (
            (1.0 - self.alpha_noise) * self._noise_filt
            + self.alpha_noise * noise
        )

        nu_c_raw = nu_wave + self._noise_filt

        self._nu_c_filt = (
            (1.0 - self.alpha_wave) * self._nu_c_filt
            + self.alpha_wave * nu_c_raw
        )

        return self._nu_c_filt.copy()

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
        """
        Hydrostatic restoring vector.

        g_eta follows the same convention used in the dynamics equation:

            M * nu_dot = tau - damping - g_eta

        Positive buoyancy means B_force > W.
        """

        weight_minus_buoyancy = self.W - self.B_force

        g_eta = np.array([
            weight_minus_buoyancy * np.sin(theta),
            -weight_minus_buoyancy * np.cos(theta) * np.sin(phi),
            -weight_minus_buoyancy * np.cos(theta) * np.cos(phi),
            self.coBM * self.W * np.cos(theta) * np.sin(phi),
            self.coBM * self.W * np.sin(theta),
            0.0
        ], dtype=float)

        return g_eta

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

        eta_dot = np.array([
            dx,
            dy,
            dz,
            d_phi,
            d_theta,
            d_psi
        ], dtype=float)

        return eta_dot

    def step(self, state, action):
        """
        Advances the dynamics one step.

        Parameters
        ----------
        state : dict
            Vehicle state dictionary.

        action : array-like, shape (6,)
            Direct thruster commands in Newtons:
            [T1, T2, T3, T4, T5, T6]

        Returns
        -------
        state : dict
            Updated state.
        """

        nu_c = self._jonswap_current()

        eta = np.array([
            state["x"],
            state["y"],
            state["z"],
            state["roll"],
            state["pitch"],
            state["yaw"],
        ], dtype=float)

        nu = np.array([
            state["u"],
            state["v"],
            state["w"],
            state["p"],
            state["q"],
            state["r"],
        ], dtype=float)

        phi, theta = eta[3], eta[4]

        # Thruster allocation
        tau, saturated_thrust = self._thrusters_to_tau(action)

        # Current-relative velocity
        nu_rel = nu.copy()
        nu_rel[0:3] -= nu_c

        # Hydrodynamic damping
        damping = (
            self.D_lin * nu_rel
            + self.D_quad * nu_rel * np.abs(nu_rel)
        )

        # Hydrostatic restoring forces
        g_eta = self._restoring_forces(phi, theta)

        # Body acceleration
        nu_dot = (tau - damping - g_eta) / self.M_diag

        # Semi-implicit Euler
        new_nu = nu + nu_dot * self.dt
        eta_dot = self._body_to_world_kinematics(eta, new_nu)
        new_eta = eta + eta_dot * self.dt

        # Update state
        keys_eta = ["x", "y", "z", "roll", "pitch", "yaw"]
        keys_nu = ["u", "v", "w", "p", "q", "r"]

        for i, key in enumerate(keys_eta):
            state[key] = new_eta[i]

        for i, key in enumerate(keys_nu):
            state[key] = new_nu[i]

        # Optional debug fields
        state["thrusters"] = saturated_thrust.copy()
        state["tau"] = tau.copy()
        state["nu_current"] = nu_c.copy()

        return state

    def reset(self):
        self._t = 0.0
        self._nu_c_filt[:] = 0.0
        self._noise_filt[:] = 0.0