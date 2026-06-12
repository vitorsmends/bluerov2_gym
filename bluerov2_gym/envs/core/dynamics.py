import numpy as np


class Dynamics:
    def __init__(self, dynamics_config: dict | None = None, jonswap_params: dict | None = None):
        cfg = dynamics_config if dynamics_config is not None else {}

        phys = cfg.get("physics", {})
        self.dt = 0.1
        self.rho = phys.get("rho", 1000.0)
        self.g = phys.get("g", 9.81)
        self.base_mass = phys.get("base_mass", 14.8)
        self.prop_mass = phys.get("prop_mass", 0.07)
        self.n_thrusters = phys.get("n_thrusters", 6)

        self.m = self.base_mass + self.n_thrusters * self.prop_mass

        self.x_size = phys.get("x_size", 0.40)
        self.y_size = phys.get("y_size", 0.25)
        self.z_size = phys.get("z_size", 0.10)

        self.buoyant_correction = 1.01 * (
            self.m / (self.rho * self.x_size * self.y_size * self.z_size)
        ) ** (1.0 / 3.0)

        self.volume = (
            self.x_size
            * self.y_size
            * self.z_size
            * self.buoyant_correction**3
        )

        self.r_g = np.array(
            [0.0, 0.0, -self.buoyant_correction * self.z_size],
            dtype=float,
        )

        self.r_b = np.array(
            [0.0, 0.0, -self.buoyant_correction * self.z_size / 4.0],
            dtype=float,
        )

        self.z_cog = self.r_g[2]
        self.z_cob = self.r_b[2]
        self.coBM = abs(self.z_cob - self.z_cog)

        self.W = self.m * self.g
        self.B_force = self.rho * self.g * self.volume

        inert = cfg.get("inertia", {})
        self.Ixx = inert.get("Ixx", 5.2539)
        self.Ixy = inert.get("Ixy", 0.0144)
        self.Ixz = inert.get("Ixz", 0.3341)
        self.Iyy = inert.get("Iyy", 7.9420)
        self.Iyz = inert.get("Iyz", 0.0260)
        self.Izz = inert.get("Izz", 6.9123)

        self.I_g = np.array(
            [
                [self.Ixx, self.Ixy, self.Ixz],
                [self.Ixy, self.Iyy, self.Iyz],
                [self.Ixz, self.Iyz, self.Izz],
            ],
            dtype=float,
        )

        self.M_RB = self._rigid_body_mass_matrix()

        added_m = cfg.get("added_mass", [5.5, 12.7, 14.57, 0.12, 0.12, 0.12])
        self.added_mass = np.array(added_m, dtype=float)

        self.M_A = np.diag(self.added_mass)
        self.M = self.M_RB + self.M_A

        damp = cfg.get("damping", {})
        self.D_lin = np.array(damp.get("lin", [25.15, 7.364, 17.955, 10.888, 20.761, 3.744]), dtype=float)
        self.D_quad = np.array(damp.get("quad", [33.8, 54.26875, 73.37135, 40.0, 40.0, 40.0]), dtype=float)

        t_limits = cfg.get("thruster_limits", {})
        self.thruster_min = t_limits.get("min", -40.0)
        self.thruster_max = t_limits.get("max", 40.0)

        default_alloc = [
            [0.70710678, 0.70710678, 0.70710678, 0.70710678, 0.0, 0.0],
            [0.70710678, -0.70710678, -0.70710678, 0.70710678, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
            [0.05126524, -0.05126524, -0.05126524, 0.05126524, -0.1105, -0.1105],
            [-0.05126524, -0.05126524, -0.05126524, -0.05126524, -0.0025, 0.0025],
            [0.16652365, -0.16652365, 0.17500893, -0.17500893, 0.0, 0.0],
        ]
        self.allocation_matrix = np.array(cfg.get("allocation_matrix", default_alloc), dtype=float)

        self.jonswap_params = {
            "Hs": 2.0,
            "Tp": 12.0,
            "gamma": 3.3,
            "N": 64,
            "wave_dir": (0.5, 0.5),
            "scale": 0.5,
            "max_current": 0.7,
            "alpha_wave": 0.02,
            "noise_std": 0.01,
            "alpha_noise": 0.3,
            "seed": 42,
        }

        if jonswap_params is not None:
            self.jonswap_params.update(jonswap_params)

        self._init_jonswap(**self.jonswap_params)

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
        lower = np.hstack((self.m * S_rg, self.I_g))
        return np.vstack((upper, lower))

    def _spatial_cross_force(self, nu):
        v = np.asarray(nu[0:3], dtype=float)
        omega = np.asarray(nu[3:6], dtype=float)
        S_v = self._skew(v)
        S_w = self._skew(omega)
        upper = np.hstack((S_w, S_v))
        lower = np.hstack((np.zeros((3, 3)), S_w))
        return np.vstack((upper, lower))

    def _rigid_body_coriolis_force(self, nu):
        return self._spatial_cross_force(nu) @ (self.M_RB @ nu)

    def _added_mass_coriolis_force(self, nu_rel):
        return self._spatial_cross_force(nu_rel) @ (self.M_A @ nu_rel)

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
            self.wave_dir = np.array([1.0, 0.0], dtype=float)
        self.wave_dir /= np.linalg.norm(self.wave_dir)

        self.scale = float(scale)
        self.max_current = float(max_current)
        self.alpha_wave = float(alpha_wave)
        self.noise_std = float(noise_std)
        self.alpha_noise = float(alpha_noise)
        self._rng = np.random.default_rng(seed)

        self._t = 0.0
        self._nu_c_filt = np.zeros(3, dtype=float)
        self._noise_filt = np.zeros(3, dtype=float)

        g = self.g
        wp = 2.0 * np.pi / Tp
        dw = 3.0 * wp / N
        alpha_js = 0.076 * Hs**2

        self._waves = []
        for i in range(1, N + 1):
            omega = i * dw
            sigma = 0.07 if omega <= wp else 0.09
            r = np.exp(-((omega - wp) ** 2) / (2.0 * sigma**2 * wp**2))
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

    def set_jonswap_params(self, **kwargs):
        self.jonswap_params.update(kwargs)
        self._init_jonswap(**self.jonswap_params)

    def get_jonswap_params(self):
        return self.jonswap_params.copy()

    def _jonswap_current(self):
        self._t += self.dt
        u_wave = 0.0
        for omega, a, phase in self._waves:
            u_wave += a * omega * np.cos(omega * self._t + phase)

        nu_wave = np.array(
            [
                u_wave * self.wave_dir[0],
                u_wave * self.wave_dir[1],
                0.0,
            ],
            dtype=float,
        )
        nu_wave *= self.scale
        mag = np.linalg.norm(nu_wave)
        if mag > self.max_current:
            nu_wave = nu_wave / mag * self.max_current

        noise = self.noise_std * self._rng.standard_normal(3)
        noise[2] = 0.0

        self._noise_filt = (1.0 - self.alpha_noise) * self._noise_filt + self.alpha_noise * noise
        nu_c_raw = nu_wave + self._noise_filt
        self._nu_c_filt = (1.0 - self.alpha_wave) * self._nu_c_filt + self.alpha_wave * nu_c_raw
        return self._nu_c_filt.copy()

    def _thrusters_to_tau(self, action):
        thrust = np.asarray(action, dtype=float)
        if thrust.shape != (6,):
            raise ValueError(f"Action must have shape (6,), got {thrust.shape}.")
        thrust = np.clip(thrust, self.thruster_min, self.thruster_max)
        tau = self.allocation_matrix @ thrust
        return tau, thrust

    def _restoring_forces(self, phi, theta):
        weight_minus_buoyancy = self.W - self.B_force
        g_eta = np.array(
            [
                weight_minus_buoyancy * np.sin(theta),
                -weight_minus_buoyancy * np.cos(theta) * np.sin(phi),
                -weight_minus_buoyancy * np.cos(theta) * np.cos(phi),
                self.coBM * self.W * np.cos(theta) * np.sin(phi),
                self.coBM * self.W * np.sin(theta),
                0.0,
            ],
            dtype=float,
        )
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
        dz = -u * s_th + v * c_th * s_phi + w * c_th * c_phi

        c_th_safe = c_th
        if abs(c_th_safe) < 1e-6:
            c_th_safe = np.sign(c_th_safe) * 1e-6 if c_th_safe != 0 else 1e-6

        d_phi = p + (q * s_phi + r * c_phi) * np.tan(theta)
        d_theta = q * c_phi - r * s_phi
        d_psi = (q * s_phi + r * c_phi) / c_th_safe

        return np.array([dx, dy, dz, d_phi, d_theta, d_psi], dtype=float)

    @staticmethod
    def _wrap_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def step(self, state, action):
        nu_c = self._jonswap_current()
        eta = np.array([state["x"], state["y"], state["z"], state["roll"], state["pitch"], state["yaw"]], dtype=float)
        nu = np.array([state["u"], state["v"], state["w"], state["p"], state["q"], state["r"]], dtype=float)
        phi, theta = eta[3], eta[4]

        tau, saturated_thrust = self._thrusters_to_tau(action)
        nu_rel = nu.copy()
        nu_rel[0:3] -= nu_c

        damping = self.D_lin * nu_rel + self.D_quad * nu_rel * np.abs(nu_rel)
        g_eta = self._restoring_forces(phi, theta)
        c_rb = self._rigid_body_coriolis_force(nu)
        c_a = self._added_mass_coriolis_force(nu_rel)

        rhs = tau - c_rb - c_a - damping - g_eta
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
        state["nu_current"] = nu_c.copy()
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