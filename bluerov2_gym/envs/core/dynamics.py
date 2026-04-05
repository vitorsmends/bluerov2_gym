import numpy as np


class Dynamics:
    # ==========================================
    # FLAG GLOBAL DO CENÁRIO
    # ==========================================
    SEA_STATE = "calm"   # opções: "calm", "storm"

    def __init__(self):
        self.dt = 0.1
        self.rho = 1000.0  # Densidade da água (kg/m3)
        self.g = 9.81

        # Parâmetros físicos do YAML
        self.m = 11.5
        self.volume = 0.0113459
        self.coBM = 0.01  # Distância CG-CB

        self.W = self.m * self.g
        self.B = self.rho * self.g * self.volume

        # Inércia e Massa Adicional
        self.added_mass = np.array([5.5, 12.7, 14.57, 0.12, 0.12, 0.12])
        self.M_diag = np.array([self.m, self.m, self.m, 0.16, 0.16, 0.16]) + self.added_mass

        # Amortecimento
        self.D_lin = np.array([4.03, 6.22, 5.18, 0.07, 0.07, 0.07])
        self.D_quad = np.array([18.18, 21.66, 36.99, 1.55, 1.55, 1.55])

        # Inicialização do estado do mar conforme flag
        self._configure_sea_state()

    def _configure_sea_state(self):
        sea_configs = {
            "calm": {
                "Hs": 1.0,
                "Tp": 6.0,
                "gamma": 3.3,
                "N": 64,
                "wave_dir": (0.5, 0.5),
                "scale": 0.5,
                "max_current": 0.7,
                "alpha_wave": 0.02,
                "noise_std": 0.01,
                "alpha_noise": 0.3,
                "seed": 42,
            },
            "storm": {
                "Hs": 5.5,           # Altura significativa (Mar Agitado)
                "Tp": 11.0,          # Período de pico condizente com Hs=5.5m
                "gamma": 4.5,        # Espectro mais "pontudo" (mar em desenvolvimento)
                "N": 64,             # Mantido conforme sua estrutura original
                "wave_dir": (0.7, 0.7),
                "scale": 2.0,        # Escala de energia ajustada
                "max_current": 1.2,  # Corrente de tempestade (m/s)
                "alpha_wave": 0.03,
                "noise_std": 0.05,   # Ruído de cristas e espuma
                "alpha_noise": 0.5,
                "seed": 42,
            },
        }

        if self.SEA_STATE not in sea_configs:
            raise ValueError(
                f"SEA_STATE inválido: {self.SEA_STATE}. "
                f"Use 'calm' ou 'storm'."
            )

        config = sea_configs[self.SEA_STATE]
        print(f"[INFO] Inicializando cenário marítimo: {self.SEA_STATE}")

        self._init_jonswap(**config)

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
        seed
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

        # Espectro JONSWAP
        g = 9.81
        wp = 2.0 * np.pi / Tp
        dw = 3.0 * wp / N
        alpha_js = 0.076 * Hs ** 2
        self._waves = []

        for i in range(1, N + 1):
            omega = i * dw
            sigma = 0.07 if omega <= wp else 0.09
            r = np.exp(-((omega - wp) ** 2) / (2.0 * sigma ** 2 * wp ** 2))
            S = (
                alpha_js
                * g ** 2
                * omega ** -5
                * np.exp(-1.25 * (wp / omega) ** 4)
                * gamma ** r
            )
            a = np.sqrt(2.0 * S * dw)
            phase = self._rng.uniform(0.0, 2.0 * np.pi)
            self._waves.append((omega, a, phase))

    def _jonswap_current(self):
        self._t += self.dt
        u = 0.0

        for omega, a, phase in self._waves:
            u += a * omega * np.cos(omega * self._t + phase)

        nu_wave = np.array([
            u * self.wave_dir[0],
            u * self.wave_dir[1],
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

    def step(self, state, action):
        # Corrente JONSWAP
        nu_c = self._jonswap_current()

        # Unpacking do estado
        eta = np.array([
            state['x'], state['y'], state['z'],
            state['roll'], state['pitch'], state['yaw']
        ])
        nu = np.array([
            state['u'], state['v'], state['w'],
            state['p'], state['q'], state['r']
        ])

        phi, theta, psi = eta[3], eta[4], eta[5]

        # 1. Forças de restauração
        g_eta = np.array([
            (self.W - self.B) * np.sin(theta),
            -(self.W - self.B) * np.cos(theta) * np.sin(phi),
            -(self.W - self.B) * np.cos(theta) * np.cos(phi),
            self.coBM * self.W * np.cos(theta) * np.sin(phi),
            self.coBM * self.W * np.sin(theta),
            0.0
        ])

        # 2. Velocidades relativas à corrente
        nu_rel = nu.copy()
        nu_rel[0:3] -= nu_c

        # 3. Amortecimento
        damping = self.D_lin * nu_rel + self.D_quad * nu_rel * np.abs(nu_rel)

        # 4. Ação (forças/torques)
        tau = np.array(action, dtype=float)

        # 5. Aceleração no corpo
        d_nu = (tau - damping - g_eta) / self.M_diag

        # 6. Integração de velocidade
        new_nu = nu + d_nu * self.dt

        # 7. Cinemática 6DoF
        c_psi, s_psi = np.cos(psi), np.sin(psi)
        c_th, s_th = np.cos(theta), np.sin(theta)
        c_phi, s_phi = np.cos(phi), np.sin(phi)

        u, v, w = new_nu[0], new_nu[1], new_nu[2]
        p, q, r = new_nu[3], new_nu[4], new_nu[5]

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

        # Taxas de Euler
        d_phi = p + (q * s_phi + r * c_phi) * np.tan(theta)
        d_theta = q * c_phi - r * s_phi
        d_psi = (q * s_phi + r * c_phi) / c_th

        # Atualização de posição
        state['x'] += dx * self.dt
        state['y'] += dy * self.dt
        state['z'] += dz * self.dt

        # Atualização de atitude
        state['roll'] += d_phi * self.dt
        state['pitch'] += d_theta * self.dt
        state['yaw'] += d_psi * self.dt

        # Atualização de velocidades
        keys_vel = ['u', 'v', 'w', 'p', 'q', 'r']
        for i, k in enumerate(keys_vel):
            state[k] = new_nu[i]

    def reset(self):
        self._t = 0.0
        self._nu_c_filt[:] = 0.0
        self._noise_filt[:] = 0.0