import numpy as np


class Reward:
    def __init__(
        self,
        use_energy_penalty=False,
        w_energy=0.01,
        action_dim=6,
        energy_clip=10.0,
    ):
        # Pesos principais
        self.w_pos = 1.0
        self.w_vel = 0.1
        self.w_ang_pos = 0.5
        self.w_stab = 0.5

        # Penalização energética opcional
        self.use_energy_penalty = use_energy_penalty
        self.w_energy = w_energy
        self.action_dim = action_dim
        self.energy_clip = energy_clip

    def _to_scalar(self, value):
        if isinstance(value, np.ndarray):
            return float(value.item())
        return float(value)

    def _energy_penalty(self, action):
        """
        Penalização por esforço de controle usando ||u||².
        A penalização é normalizada pelo número de atuadores
        e limitada por clipping para evitar dominar a reward.
        """
        if action is None:
            return 0.0

        action = np.asarray(action, dtype=float).reshape(-1)

        # Média quadrática da ação
        penalty = np.sum(action ** 2) / max(1, self.action_dim)

        # Limita o impacto de ações extremas
        penalty = np.clip(penalty, 0.0, self.energy_clip)

        return float(penalty)

    def get_reward(self, obs, action=None):
        """
        Espera que:
        - x, y, z sejam erro de posição
        - u, v, w sejam erro de velocidade
        - roll, pitch, yaw sejam atitude/erro angular conforme o ambiente
        """

        # Estados / erros
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

        # 3. Erro de yaw com wrap em [-pi, pi]
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

        # 6. Bônus de sucesso
        if position_error < 0.2:
            reward += 1.0

        return float(reward)