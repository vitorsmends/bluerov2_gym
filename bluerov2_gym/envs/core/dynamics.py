import numpy as np

class Dynamics:
    def __init__(self):
        self.dt = 0.1
        self.rho = 1000.0  # Densidade da água (kg/m3)
        self.g = 9.81
        
        # Parametros fisicos do YAML
        self.m = 11.5
        self.volume = 0.0113459
        self.coBM = 0.01  # Distancia CG-CB
        
        self.W = self.m * self.g
        self.B = self.rho * self.g * self.volume
        
        # Inercia e Massa Adicional (YAML)
        # Assumindo momentos de inercia rigidos de 0.16 kg.m2 conforme tese
        self.added_mass = np.array([5.5, 12.7, 14.57, 0.12, 0.12, 0.12])
        self.M_diag = np.array([self.m, self.m, self.m, 0.16, 0.16, 0.16]) + self.added_mass
        
        # Amortecimento (YAML)
        self.D_lin = np.array([4.03, 6.22, 5.18, 0.07, 0.07, 0.07])
        self.D_quad = np.array([18.18, 21.66, 36.99, 1.55, 1.55, 1.55])
        
        # JONSWAP + Gaussian noise config
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

    def _init_jonswap(self, Hs, Tp, gamma, N, wave_dir, scale, max_current, alpha_wave, noise_std, alpha_noise, seed):
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
            S = (alpha_js * g ** 2 * omega ** -5 * np.exp(-1.25 * (wp / omega) ** 4) * gamma ** r)
            a = np.sqrt(2.0 * S * dw)
            phase = self._rng.uniform(0.0, 2.0 * np.pi)
            self._waves.append((omega, a, phase))

    def _jonswap_current(self):
        self._t += self.dt
        u = 0.0
        for omega, a, phase in self._waves:
            u += a * omega * np.cos(omega * self._t + phase)

        nu_wave = np.array([u * self.wave_dir[0], u * self.wave_dir[1], 0.0])
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

    def step(self, state, action):
            # Corrente JONSWAP
            nu_c = self._jonswap_current()
            
            # Unpacking estado
            eta = np.array([state['x'], state['y'], state['z'], state['roll'], state['pitch'], state['yaw']])
            nu = np.array([state['u'], state['v'], state['w'], state['p'], state['q'], state['r']])
            phi, theta, psi = eta[3], eta[4], eta[5]
            
            # 1. Forças de Restauração (Gravidade + Empuxo)
            # Nota: coBM define o braço de alavanca para estabilidade passiva
            g_eta = np.array([
                (self.W - self.B) * np.sin(theta),
                -(self.W - self.B) * np.cos(theta) * np.sin(phi),
                -(self.W - self.B) * np.cos(theta) * np.cos(phi),
                self.coBM * self.W * np.cos(theta) * np.sin(phi),
                self.coBM * self.W * np.sin(theta),
                0.0
            ])
            
            # 2. Velocidades Relativas (Corrente)
            nu_rel = nu.copy()
            nu_rel[0:3] -= nu_c
            
            # 3. Amortecimento
            damping = self.D_lin * nu_rel + self.D_quad * nu_rel * np.abs(nu_rel)
            
            # 4. Ação (Forças/Torques)
            tau = np.array(action)
            
            # 5. Aceleração no Corpo
            d_nu = (tau - damping - g_eta) / self.M_diag
            
            # 6. Integração de velocidade (Euler)
            new_nu = nu + d_nu * self.dt
            
            # 7. Cinemática CORRIGIDA (Body -> World 6 DoF)
            # Matriz de Rotação completa (R_z * R_y * R_x)
            c_psi, s_psi = np.cos(psi), np.sin(psi)
            c_th, s_th = np.cos(theta), np.sin(theta)
            c_phi, s_phi = np.cos(phi), np.sin(phi)

            u, v, w = new_nu[0], new_nu[1], new_nu[2]
            p, q, r = new_nu[3], new_nu[4], new_nu[5]

            # Velocidades lineares no mundo (dot_x, dot_y, dot_z)
            dx = u * c_psi * c_th + v * (c_psi * s_th * s_phi - s_psi * c_phi) + w * (c_psi * s_th * c_phi + s_psi * s_phi)
            dy = u * s_psi * c_th + v * (s_psi * s_th * s_phi + c_psi * c_phi) + w * (s_psi * s_th * c_phi - c_psi * s_phi)
            dz = -u * s_th + v * c_th * s_phi + w * c_th * c_phi

            # Velocidades angulares (Taxa de Euler - Transformação para ângulos de Euler)
            # Nota: Para pequenos ângulos (Roll/Pitch prox de 0), p~d_phi, q~d_theta, r~d_psi
            # Esta matriz abaixo é a exata, mas tem singularidade em Pitch +/- 90 graus
            d_phi = p + (q * s_phi + r * c_phi) * np.tan(theta)
            d_theta = q * c_phi - r * s_phi
            d_psi = (q * s_phi + r * c_phi) / c_th

            # Atualização de Posição
            state['x'] += dx * self.dt
            state['y'] += dy * self.dt
            state['z'] += dz * self.dt
            
            # Atualização de Atitude
            state['roll'] += d_phi * self.dt
            state['pitch'] += d_theta * self.dt
            state['yaw'] += d_psi * self.dt
            
            # Atualização de Velocidades
            keys_vel = ['u', 'v', 'w', 'p', 'q', 'r']
            for i, k in enumerate(keys_vel):
                state[k] = new_nu[i]

    def reset(self):
        self._t = 0.0
        self._nu_c_filt[:] = 0.0
        self._noise_filt[:] = 0.0