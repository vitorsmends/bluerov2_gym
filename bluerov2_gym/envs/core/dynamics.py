import numpy as np


def _skew(v: np.ndarray) -> np.ndarray:
    """Retorna a matriz antissimétrica S(v) para v em R^3."""
    return np.array([
        [0.0, -v[2],  v[1]],
        [v[2],  0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=float)


class Dynamics:
    """
    Modelo final 6DoF do BlueROV2 Heavy com:
      - dinâmica 6DoF completa
      - 8 thrusters como ação direta do agente
      - corrente oceânica 3D
      - distúrbio de onda explícito em surge/heave/pitch (Walker-style)

    Estado esperado no dicionário `state`:
      x, y, z, roll, pitch, yaw, u, v, w, p, q, r

    Ação esperada em `step(state, action)`:
      action.shape == (8,)
      ação = comandos normalizados dos thrusters em [-1, 1]

    Convenções:
      eta = [x, y, z, phi, theta, psi]
      nu  = [u, v, w, p, q, r]
      tau = [X, Y, Z, K, M, N]

    Observações:
      - A dinâmica do veículo é 6DoF.
      - O modelo explícito de onda segue Walker e entra apenas em surge/heave/pitch.
      - Isso gera um modelo híbrido: veículo 6DoF + onda explícita em 3 DoF.
    """

    SEA_STATE = "storm"  # "calm" | "storm"

    def __init__(self):
        self.dt = 0.1
        self.rho = 1025.0
        self.g = 9.81

        # ------------------------------------------------------------------
        # Parâmetros físicos do veículo
        # ------------------------------------------------------------------
        self.m = 11.5
        self.W = 112.8
        self.B = 114.8

        # CG e CB
        self.rG = np.array([0.0, 0.0, 0.0], dtype=float)
        self.rB = np.array([0.0, 0.0, 0.028], dtype=float)

        # Tensor de inércia
        # TODO_KYLE:
        # Confirmar Ix e Iz do modelo final usado por eles.
        # Iy vem do Walker.
        self.Ix = 0.16
        self.Iy = 0.253
        self.Iz = 0.16
        self.Ig = np.diag([self.Ix, self.Iy, self.Iz])

        # ------------------------------------------------------------------
        # Massa adicional
        # ------------------------------------------------------------------
        self.Xu_dot = -6.36
        self.Yv_dot = -12.7
        self.Zw_dot = -18.68
        self.Kp_dot = -0.12
        self.Mq_dot = -0.135
        self.Nr_dot = -0.12

        # Acoplamentos planares refinados por Walker
        self.Xq_dot = -0.67
        self.Mu_dot = -0.67

        self.MA = np.array([
            [-self.Xu_dot, 0.0,           0.0,          0.0,          -self.Xq_dot, 0.0],
            [0.0,          -self.Yv_dot,  0.0,          0.0,           0.0,          0.0],
            [0.0,          0.0,          -self.Zw_dot,  0.0,           0.0,          0.0],
            [0.0,          0.0,           0.0,         -self.Kp_dot,   0.0,          0.0],
            [-self.Mu_dot, 0.0,           0.0,          0.0,          -self.Mq_dot,  0.0],
            [0.0,          0.0,           0.0,          0.0,           0.0,         -self.Nr_dot],
        ], dtype=float)

        # ------------------------------------------------------------------
        # Amortecimento linear e quadrático
        # ------------------------------------------------------------------
        self.D_lin = np.diag([
            13.7,    # Xu
            6.22,    # Yv
            33.0,    # Zw
            0.07,    # Kp
            0.80,    # Mq
            0.07,    # Nr
        ])

        self.D_quad = np.diag([
            141.0,   # Xu|u|
            21.66,   # Yv|v|
            190.0,   # Zw|w|
            1.55,    # Kp|p|
            0.47,    # Mq|q|
            1.55,    # Nr|r|
        ])

        # ------------------------------------------------------------------
        # Geometria dos 8 thrusters (BlueROV2 Heavy)
        # braços em metros
        # ------------------------------------------------------------------
        arms_mm = np.array([
            [ 156,  111,  85],
            [ 156, -111,  85],
            [-156,  111,  85],
            [-156, -111,  85],
            [ 120,  218,   0],
            [ 120, -218,   0],
            [-120,  218,   0],
            [-120, -218,   0],
        ], dtype=float)
        self.thruster_pos = arms_mm / 1000.0

        # Direções dos thrusters no frame do corpo
        self.thruster_dirs = np.array([
            [ np.cos(np.pi/4),    -np.sin(np.pi/4),     0.0],  # T1
            [ np.cos(-np.pi/4),   -np.sin(-np.pi/4),    0.0],  # T2
            [ np.cos(-3*np.pi/4), -np.sin(-3*np.pi/4),  0.0],  # T3
            [ np.cos( 3*np.pi/4), -np.sin( 3*np.pi/4),  0.0],  # T4
            [0.0, 0.0, 1.0],                                   # T5
            [0.0, 0.0, 1.0],                                   # T6
            [0.0, 0.0, 1.0],                                   # T7
            [0.0, 0.0, 1.0],                                   # T8
        ], dtype=float)

        # Thruster max force
        # Walker usa Tmax = 35 N no modelo de controle.
        self.Tmax = 35.0

        # Dinâmica de atuador
        # TODO_KYLE: confirmar tm real usado no código deles.
        self.tm = 0.15
        self._thruster_state = np.zeros(8, dtype=float)

        # Deadzone simples
        self.deadzone = 0.05

        # ------------------------------------------------------------------
        # Distúrbios oceânicos
        # ------------------------------------------------------------------
        self.current_mean = np.array([0.15, 0.05, 0.0], dtype=float)
        self.current_amp = np.array([0.10, 0.06, 0.02], dtype=float)
        self.current_freq = np.array([0.015, 0.021, 0.01], dtype=float)

        # Geometria usada na integral do momento de onda
        # TODO_KYLE: confirmar comprimento efetivo usado na integral de ME.
        self.L = 0.457

        # Ruído opcional sobre tau_E
        self.noise_std_tauE = 0.0
        self.alpha_noise = 0.2
        self._noise_filt = np.zeros(6, dtype=float)

        # Tempo interno
        self._t = 0.0

        # Configuração do mar
        self._configure_sea_state()

    # ----------------------------------------------------------------------
    # Configuração do mar / JONSWAP
    # ----------------------------------------------------------------------

    def _configure_sea_state(self):
        """
        Configura um caso representativo.
        TODO_KYLE:
        Confirmar o caso espectral exato que você quer reproduzir.
        """
            configs = {
                "calm": {
                    "Hs": 2.78,
                    "Tp": 7.1,
                    "gamma": 3.3,
                    "N": 100,
                    "depth": 54.0,
                    "seed": 42,
                },
                "storm": {
                    "Hs": 3.47,
                    "Tp": 9.5,
                    "gamma": 3.3,
                    "N": 100,
                    "depth": 54.0,
                    "seed": 42,
                },
            }

            if self.SEA_STATE not in configs:
                raise ValueError(f"SEA_STATE invalido: {self.SEA_STATE}")

            self._init_jonswap(**configs[self.SEA_STATE])

        def _init_jonswap(self, Hs, Tp, gamma, N, depth, seed):
            self._depth = depth
            self._rng = np.random.default_rng(seed)

            wp = 2.0 * np.pi / Tp
            dw = 3.0 * wp / N
            alpha_js = 0.0081

            waves_raw = []
            for i in range(1, N + 1):
                omega = i * dw
                
                f_hz = omega / (2.0 * np.pi)
                if f_hz < 0.2 or f_hz > 2.0:
                    continue

                sigma = 0.07 if omega <= wp else 0.09
                Gamma = np.exp(-((omega - wp) ** 2) / (2.0 * sigma**2 * wp**2))
                S = (
                    alpha_js
                    * self.g**2
                    * omega**-5
                    * np.exp(-1.25 * (wp / omega) ** 4)
                    * gamma**Gamma
                )

                a = np.sqrt(2.0 * S * dw)
                phase = self._rng.uniform(0.0, 2.0 * np.pi)
                kappa = self._dispersion(omega, depth)
                waves_raw.append((omega, a, phase, kappa))

            if waves_raw:
                max_a = max(a for _, a, _, _ in waves_raw)
                threshold = 0.05 * max_a
                waves_raw = [(w, a, ph, k) for (w, a, ph, k) in waves_raw if a > threshold]

            m0 = sum(0.5 * a**2 for _, a, _, _ in waves_raw)
            hs_est = 4.0 * np.sqrt(m0) if m0 > 0 else 1.0
            scale = Hs / hs_est if hs_est > 1e-12 else 1.0

            self._waves = [(w, a * scale, ph, k) for (w, a, ph, k) in waves_raw]

    def _dispersion(self, omega, depth):
        """Resolve omega^2 = g k tanh(k d) por Newton-Raphson."""
        k = omega**2 / self.g
        for _ in range(40):
            th = np.tanh(k * depth)
            f = self.g * k * th - omega**2
            df = self.g * (th + k * depth * (1.0 - th**2))
            k_new = k - f / df
            if abs(k_new - k) < 1e-12:
                return max(k_new, 1e-9)
            k = k_new
        return max(k, 1e-9)

    # ----------------------------------------------------------------------
    # Matrizes do modelo
    # ----------------------------------------------------------------------

    def _MRB(self):
        m = self.m
        Srg = _skew(self.rG)
        upper = np.hstack((m * np.eye(3), -m * Srg))
        lower = np.hstack((m * Srg, self.Ig))
        return np.vstack((upper, lower))

    def _M(self):
        return self._MRB() + self.MA

    def _CRB(self, nu):
        v = nu[:3]
        omega = nu[3:]

        m = self.m
        S_v = _skew(v)
        Iomega = self.Ig @ omega
        S_Iomega = _skew(Iomega)

        top = np.hstack((np.zeros((3, 3)), -m * S_v))
        bottom = np.hstack((-m * S_v, -S_Iomega))
        return np.vstack((top, bottom))

    def _CA(self, nu):
        a = self.MA @ nu
        a1 = a[:3]
        a2 = a[3:]
        top = np.hstack((np.zeros((3, 3)), -_skew(a1)))
        bottom = np.hstack((-_skew(a1), -_skew(a2)))
        return np.vstack((top, bottom))

    def _D_force(self, nu_rel):
        """
        Retorna D(nu_rel) * nu_rel como vetor 6D.
        """
        diag_lin = np.diag(self.D_lin)
        diag_quad = np.diag(self.D_quad)
        return (diag_lin + diag_quad * np.abs(nu_rel)) * nu_rel

    def _g_eta(self, eta):
        """
        Forças de restauração hidrostática.
        eta = [x, y, z, phi, theta, psi]
        """
        phi, theta, psi = eta[3], eta[4], eta[5]
        W, B = self.W, self.B
        xG, yG, zG = self.rG
        xB, yB, zB = self.rB

        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)

        return np.array([
            (W - B) * sth,
            -(W - B) * cth * sphi,
            -(W - B) * cth * cphi,
            -(yG * W - yB * B) * cth * cphi + (zG * W - zB * B) * cth * sphi,
            (zG * W - zB * B) * sth + (xG * W - xB * B) * cth * cphi,
            -(xG * W - xB * B) * cth * sphi - (yG * W - yB * B) * sth,
        ], dtype=float)

    # ----------------------------------------------------------------------
    # Cinemática
    # ----------------------------------------------------------------------

    def _R_bn(self, eta):
        """
        Rotação body -> navigation, convenção ZYX.
        """
        phi, theta, psi = eta[3], eta[4], eta[5]
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)

        return np.array([
            [cpsi * cth, cpsi * sth * sphi - spsi * cphi, cpsi * sth * cphi + spsi * sphi],
            [spsi * cth, spsi * sth * sphi + cpsi * cphi, spsi * sth * cphi - cpsi * sphi],
            [-sth,       cth * sphi,                      cth * cphi],
        ], dtype=float)

    def _J(self, eta):
        """
        Matriz J(eta) tal que eta_dot = J(eta) nu.
        """
        phi, theta, psi = eta[3], eta[4], eta[5]

        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        tth = np.tan(theta)

        R = self._R_bn(eta)

        T = np.array([
            [1.0, sphi * tth, cphi * tth],
            [0.0, cphi,      -sphi],
            [0.0, sphi / cth, cphi / cth],
        ], dtype=float)

        J = np.zeros((6, 6), dtype=float)
        J[:3, :3] = R
        J[3:, 3:] = T
        return J

    # ----------------------------------------------------------------------
    # Thrusters
    # ----------------------------------------------------------------------

    def _apply_deadzone(self, u):
        out = np.array(u, dtype=float, copy=True)
        out[np.abs(out) < self.deadzone] = 0.0
        return out

    def _thruster_dynamics(self, cmd):
        """
        cmd em [-1,1] -> força de cada thruster.
        """
        cmd = np.clip(np.asarray(cmd, dtype=float), -1.0, 1.0)
        cmd = self._apply_deadzone(cmd)

        alpha = 1.0 - np.exp(-self.dt / max(self.tm, 1e-6))
        self._thruster_state += alpha * (cmd - self._thruster_state)

        # Saturação simétrica simples
        return self.Tmax * self._thruster_state

    def _thrusters_to_tau(self, thruster_cmd):
        """
        Converte 8 thrusters em tau = [X,Y,Z,K,M,N].
        """
        F = self._thruster_dynamics(thruster_cmd)
        tau = np.zeros(6, dtype=float)

        for i in range(8):
            fi = self.thruster_dirs[i] * F[i]
            ri = self.thruster_pos[i]
            mi = np.cross(ri, fi)

            tau[:3] += fi
            tau[3:] += mi

        return tau

    # ----------------------------------------------------------------------
    # Corrente oceânica
    # ----------------------------------------------------------------------

    def _current_world(self, t):
        """
        Corrente lenta 3D no frame do mundo.
        """
        return self.current_mean + self.current_amp * np.array([
            np.sin(2.0 * np.pi * self.current_freq[0] * t + 0.2),
            np.sin(2.0 * np.pi * self.current_freq[1] * t + 1.1),
            np.sin(2.0 * np.pi * self.current_freq[2] * t + 2.4),
        ], dtype=float)

    def _current_body(self, eta, t):
        R = self._R_bn(eta)
        return R.T @ self._current_world(t)

    # ----------------------------------------------------------------------
    # Onda: cinemática de partícula
    # ----------------------------------------------------------------------

    def _wave_particle_kinematics(self, z, t):
        """
        Retorna:
            up, wp, dup_dt, dwp_dt
        no plano de propagação da onda.
        """
        d = self._depth
        z = np.clip(z, -d, 0.0)

        up = 0.0
        wp = 0.0
        dup_dt = 0.0
        dwp_dt = 0.0

        x_eval = 0.0

        for omega, a, phase, kappa in self._waves:
            H = 2.0 * a
            c = omega / kappa
            arg = kappa * x_eval - omega * t + phase
            arg2 = 2.0 * arg

            # 1ª ordem
            A1 = self.g * H / (2.0 * c)
            ch1 = np.cosh(kappa * (z + d)) / np.cosh(kappa * d)
            sh1 = np.sinh(kappa * (z + d)) / np.cosh(kappa * d)

            up_i = A1 * ch1 * np.cos(arg)
            wp_i = A1 * sh1 * np.sin(arg)

            dup_i = A1 * ch1 * omega * np.sin(arg)
            dwp_i = -A1 * sh1 * omega * np.cos(arg)

            # 2ª ordem
            A2 = (3.0 / 16.0) * c * kappa**2 * H**2
            denom = max(np.sinh(kappa * d) ** 4, 1e-12)
            ch2 = np.cosh(2.0 * kappa * (z + d)) / denom
            sh2 = np.sinh(2.0 * kappa * (z + d)) / denom

            up2_i = A2 * ch2 * np.cos(arg2)
            wp2_i = A2 * sh2 * np.sin(arg2)

            dup2_i = 2.0 * A2 * ch2 * omega * np.sin(arg2)
            dwp2_i = -2.0 * A2 * sh2 * omega * np.cos(arg2)

            up += up_i + up2_i
            wp += wp_i + wp2_i
            dup_dt += dup_i + dup2_i
            dwp_dt += dwp_i + dwp2_i

        return up, wp, dup_dt, dwp_dt

    def _tau_wave(self, eta):
        """
        Distúrbio explícito de onda no estilo Walker:
            tau_E = [XE, 0, ZE, 0, ME, 0]

        eta = [x,y,z,phi,theta,psi]
        """
        z = eta[2]
        theta = eta[4]

        up, wp, dup_dt, dwp_dt = self._wave_particle_kinematics(z, self._t)

        cth, sth = np.cos(theta), np.sin(theta)

        # projeção no frame do corpo
        vp_x = cth * up + sth * wp
        vp_z = -sth * up + cth * wp

        dvp_x = cth * dup_dt + sth * dwp_dt
        dvp_z = -sth * dup_dt + cth * dwp_dt

        XE = (-self.Xu_dot) * dvp_x + (13.7 + 141.0 * abs(vp_x)) * vp_x
        ZE = (-self.Zw_dot) * dvp_z + (33.0 + 190.0 * abs(vp_z)) * vp_z

        # Momento de pitch
        xs = np.linspace(-self.L / 2.0, self.L / 2.0, 25)
        ze_local = np.zeros_like(xs)

        for i, xp in enumerate(xs):
            z_local = z + xp * np.sin(theta)
            up_l, wp_l, dup_l, dwp_l = self._wave_particle_kinematics(z_local, self._t)

            vpz_l = -sth * up_l + cth * wp_l
            dvpz_l = -sth * dup_l + cth * dwp_l

            ze_local[i] = (-self.Zw_dot) * dvpz_l + (33.0 + 190.0 * abs(vpz_l)) * vpz_l

        ME = np.trapz(ze_local * xs, xs)

        tau = np.array([XE, 0.0, ZE, 0.0, ME, 0.0], dtype=float)

        if self.noise_std_tauE > 0.0:
            noise = self.noise_std_tauE * self._rng.standard_normal(6)
            self._noise_filt = (
                (1.0 - self.alpha_noise) * self._noise_filt
                + self.alpha_noise * noise
            )
            tau = tau + self._noise_filt

        return tau

    # ----------------------------------------------------------------------
    # Dinâmica contínua
    # ----------------------------------------------------------------------

    def _f(self, eta, nu, action, t_eval):
        """
        Retorna eta_dot e nu_dot para RK4.
        """
        # salva tempo interno e usa t_eval temporariamente
        t_old = self._t
        self._t = t_eval

        M = self._M()
        C = self._CRB(nu) + self._CA(nu)

        tau_thr = self._thrusters_to_tau(action)
        tau_wave = self._tau_wave(eta)

        # corrente em body
        nu_c_body = np.zeros(6, dtype=float)
        nu_c_body[:3] = self._current_body(eta, t_eval)

        # componente de onda em velocidades relativas só em surge/heave
        up, wp, _, _ = self._wave_particle_kinematics(eta[2], t_eval)
        theta = eta[4]
        vp_x = np.cos(theta) * up + np.sin(theta) * wp
        vp_z = -np.sin(theta) * up + np.cos(theta) * wp

        nu_wave_body = np.zeros(6, dtype=float)
        nu_wave_body[0] = vp_x
        nu_wave_body[2] = vp_z

        nu_rel = nu - nu_c_body # - nu_wave_body

        D = self._D_force(nu_rel)
        g_eta = self._g_eta(eta)

        rhs = tau_thr + tau_wave - (C @ nu) - D - g_eta
        nu_dot = np.linalg.solve(M, rhs)
        eta_dot = self._J(eta) @ nu

        self._t = t_old
        return eta_dot, nu_dot

    # ----------------------------------------------------------------------
    # Passo de integração
    # ----------------------------------------------------------------------

    def step(self, state, action):
        """
        Avança um passo usando RK4.

        action:
          array-like shape (8,), comandos dos thrusters em [-1,1]
        """
        action = np.asarray(action, dtype=float)
        if action.shape != (8,):
            raise ValueError("action deve ter shape (8,) com thrusters em [-1,1].")

        eta0 = np.array([
            state["x"],
            state["y"],
            state["z"],
            state["roll"],
            state["pitch"],
            state["yaw"],
        ], dtype=float)

        nu0 = np.array([
            state["u"],
            state["v"],
            state["w"],
            state["p"],
            state["q"],
            state["r"],
        ], dtype=float)

        dt = self.dt
        t0 = self._t

        k1_eta, k1_nu = self._f(eta0, nu0, action, t0)
        k2_eta, k2_nu = self._f(eta0 + 0.5 * dt * k1_eta, nu0 + 0.5 * dt * k1_nu, action, t0 + 0.5 * dt)
        k3_eta, k3_nu = self._f(eta0 + 0.5 * dt * k2_eta, nu0 + 0.5 * dt * k2_nu, action, t0 + 0.5 * dt)
        k4_eta, k4_nu = self._f(eta0 + dt * k3_eta, nu0 + dt * k3_nu, action, t0 + dt)

        eta_new = eta0 + (dt / 6.0) * (k1_eta + 2.0 * k2_eta + 2.0 * k3_eta + k4_eta)
        nu_new = nu0 + (dt / 6.0) * (k1_nu + 2.0 * k2_nu + 2.0 * k3_nu + k4_nu)

        self._t = t0 + dt

        state["x"] = eta_new[0]
        state["y"] = eta_new[1]
        state["z"] = eta_new[2]
        state["roll"] = eta_new[3]
        state["pitch"] = eta_new[4]
        state["yaw"] = eta_new[5]

        state["u"] = nu_new[0]
        state["v"] = nu_new[1]
        state["w"] = nu_new[2]
        state["p"] = nu_new[3]
        state["q"] = nu_new[4]
        state["r"] = nu_new[5]

    def reset(self):
        self._t = 0.0
        self._thruster_state[:] = 0.0
        self._noise_filt[:] = 0.0