import numpy as np


class Reward:
    def __init__(
        self,
        use_energy_penalty=False,
        w_energy=0.01,
    ):
        # Pesos originais
        self.w_pos = 1.0
        self.w_vel = 0.1
        self.w_ang_pos = 0.5
        self.w_stab = 0.5

        # Penalização energética opcional
        self.use_energy_penalty = use_energy_penalty
        self.w_energy = w_energy

    def _to_scalar(self, value):
        return value.item() if isinstance(value, np.ndarray) else value

    def _energy_penalty(self, action):
        """
        Penalização simples baseada no esforço dos atuadores.
        Usa norma quadrática da ação.
        """
        if action is None:
            return 0.0

        action = np.asarray(action, dtype=float).reshape(-1)
        return float(np.sum(action ** 2))

    def get_reward(self, obs, action=None):
        # Estados
        x = self._to_scalar(obs["x"])
        y = self._to_scalar(obs["y"])
        z = self._to_scalar(obs["z"])

        u = self._to_scalar(obs["u"])
        v = self._to_scalar(obs["v"])
        w = self._to_scalar(obs["w"])

        roll = self._to_scalar(obs["roll"])
        pitch = self._to_scalar(obs["pitch"])
        yaw = self._to_scalar(obs["yaw"])

        # 1. Erro de posição
        position_error = np.sqrt(x**2 + y**2 + z**2)

        # 2. Penalidade de velocidade
        velocity_penalty = np.sqrt(u**2 + v**2 + w**2)

        # 3. Erro de yaw com wrap
        yaw_error = np.arctan2(np.sin(yaw), np.cos(yaw))
        yaw_error = abs(yaw_error)

        # 4. Penalidade de estabilidade
        stability_penalty = abs(roll) + abs(pitch)

        # Reward base
        reward = -(
            self.w_pos * position_error
            + self.w_vel * velocity_penalty
            + self.w_ang_pos * yaw_error
            + self.w_stab * stability_penalty
        )

        # 5. Penalização energética opcional
        if self.use_energy_penalty:
            energy_penalty = self._energy_penalty(action)
            reward -= self.w_energy * energy_penalty

        # Bônus de sucesso
        if position_error < 0.2:
            reward += 1.0

        return float(reward)