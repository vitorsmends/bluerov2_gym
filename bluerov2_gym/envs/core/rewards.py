import numpy as np

class Reward:
    def __init__(self):
        # Pesos
        self.w_pos = 1.0
        self.w_vel = 0.1
        self.w_ang_pos = 0.5
        self.w_stab = 0.5  # Aumentei levemente para punir roll/pitch severamente

    def get_reward(self, obs):
        # Extração segura dos valores (garante que estamos usando escalares e não arrays)
        # O .item() extrai o valor de dentro do array numpy de shape (1,)
        x = obs["x"].item() if isinstance(obs["x"], np.ndarray) else obs["x"]
        y = obs["y"].item() if isinstance(obs["y"], np.ndarray) else obs["y"]
        z = obs["z"].item() if isinstance(obs["z"], np.ndarray) else obs["z"]
        
        u = obs["u"].item() if isinstance(obs["u"], np.ndarray) else obs["u"]
        v = obs["v"].item() if isinstance(obs["v"], np.ndarray) else obs["v"]
        w = obs["w"].item() if isinstance(obs["w"], np.ndarray) else obs["w"]

        roll = obs["roll"].item() if isinstance(obs["roll"], np.ndarray) else obs["roll"]
        pitch = obs["pitch"].item() if isinstance(obs["pitch"], np.ndarray) else obs["pitch"]
        yaw = obs["yaw"].item() if isinstance(obs["yaw"], np.ndarray) else obs["yaw"]

        # 1. Erro de Posição
        position_error = np.sqrt(x**2 + y**2 + z**2)

        # 2. Penalidade de Velocidade
        velocity_penalty = np.sqrt(u**2 + v**2 + w**2)

        # 3. Erro de Yaw com Correção de Descontinuidade (Wrap-around)
        # Normaliza o erro para ficar entre -pi e pi
        # Ex: Se yaw é 6.28 (2pi), o erro vira 0.
        yaw_error = np.arctan2(np.sin(yaw), np.cos(yaw))
        yaw_error = abs(yaw_error)

        # 4. Penalidade de Estabilidade
        stability_penalty = abs(roll) + abs(pitch)

        # Cálculo da Recompensa
        reward = -(
            self.w_pos * position_error
            + self.w_vel * velocity_penalty
            + self.w_ang_pos * yaw_error
            + self.w_stab * stability_penalty
        )

        # Bônus de Sucesso (Target Reached)
        if position_error < 0.2:  # Aumentei margem para 20cm (mais realista na água)
            reward += 1.0

        return float(reward) # Garante retorno float puro