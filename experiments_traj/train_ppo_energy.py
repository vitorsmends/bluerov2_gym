import gymnasium as gym
import numpy as np
import math
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
import bluerov2_gym.envs.bluerov_env as original_env


# ==========================================
# 1. REGISTRO DO AMBIENTE BASE
# ==========================================
try:
    register(
        id="BlueRov-v0",
        entry_point="bluerov2_gym.envs.bluerov_env:BlueRov",
        max_episode_steps=2000,
    )
except Exception:
    pass


# ==========================================
# 2. GERADOR DE TRAJETÓRIA
# ==========================================
class TrajectoryGenerator:
    def __init__(self):
        self.radius = 1.0
        self.speed = 0.15
        self.z_target = -0.5

    def get_state_at_time(self, t):
        t_s = t * self.speed

        x = self.radius * math.sin(t_s)
        y = self.radius * math.sin(t_s) * math.cos(t_s)

        if t < 10.0:
            z = (self.z_target / 10.0) * t
        else:
            z = self.z_target

        vx = self.radius * math.cos(t_s) * self.speed
        vy = self.radius * (math.cos(t_s) ** 2 - math.sin(t_s) ** 2) * self.speed
        vz = 0.0

        yaw = math.atan2(vy, vx)

        return (
            np.array([x, y, z], dtype=float),
            np.array([0.0, 0.0, yaw], dtype=float),
            np.array([vx, vy, vz], dtype=float),
        )


# ==========================================
# 3. AMBIENTE PERSONALIZADO
# ==========================================
class TrajectoryTrackingEnv(original_env.BlueRov):
    def __init__(
        self,
        use_energy_penalty=True,
        energy_mode="action_l2",
        w_energy=0.005,
        use_smoothness_penalty=True,
        w_smooth=0.05,
        action_clip=10.0,
    ):
        super().__init__(render_mode=None)

        self.traj = TrajectoryGenerator()
        self.current_t = 0.0
        self.dt = 0.1

        self.use_energy_penalty = use_energy_penalty
        self.energy_mode = energy_mode
        self.w_energy = w_energy

        self.use_smoothness_penalty = use_smoothness_penalty
        self.w_smooth = w_smooth

        self.action_clip = action_clip
        self.prev_action = None

    def reset(self, seed=None, options=None):
        self.current_t = np.random.uniform(0.0, 50.0)
        self.prev_action = None

        target_pos, target_att, _ = self.traj.get_state_at_time(self.current_t)

        noise_pos = np.random.uniform(-0.2, 0.2, 3)
        initial_pos = target_pos + noise_pos

        self.state = {
            "x": initial_pos[0],
            "y": initial_pos[1],
            "z": initial_pos[2],
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": target_att[2],
            "u": 0.0,
            "v": 0.0,
            "w": 0.0,
            "p": 0.0,
            "q": 0.0,
            "r": 0.0,
        }

        return self._get_obs(), {}

    def _prepare_action(self, action):
        action = np.asarray(action, dtype=float).reshape(-1)
        action = np.clip(action, -self.action_clip, self.action_clip)
        return action

    def _compute_energy_penalty(self, action):
        if not self.use_energy_penalty:
            return 0.0

        if self.energy_mode == "action_l2":
            # média quadrática da ação
            return float(np.sum(action ** 2) / len(action))

        raise ValueError(f"energy_mode inválido: {self.energy_mode}")

    def _compute_smoothness_penalty(self, action):
        if not self.use_smoothness_penalty or self.prev_action is None:
            return 0.0

        delta_u = action - self.prev_action
        return float(np.sum(delta_u ** 2) / len(action))

    def step(self, action):
        self.current_t += self.dt

        action = self._prepare_action(action)

        obs, _, terminated, truncated, info = super().step(action)

        tgt_pos, tgt_att, tgt_vel = self.traj.get_state_at_time(self.current_t)

        curr_pos = np.array(
            [obs["x"][0], obs["y"][0], obs["z"][0]],
            dtype=float
        )
        curr_vel = np.array(
            [obs["u"][0], obs["v"][0], obs["w"][0]],
            dtype=float
        )

        error_pos = curr_pos - tgt_pos
        error_vel = curr_vel - tgt_vel

        psi = float(obs["yaw"][0])
        c, s = np.cos(psi), np.sin(psi)

        err_x_body = error_pos[0] * c + error_pos[1] * s
        err_y_body = -error_pos[0] * s + error_pos[1] * c
        err_z_body = error_pos[2]

        # Observação de erro para o PPO
        obs["x"] = np.array([err_x_body], dtype=np.float32)
        obs["y"] = np.array([err_y_body], dtype=np.float32)
        obs["z"] = np.array([err_z_body], dtype=np.float32)
        obs["u"] = np.array([error_vel[0]], dtype=np.float32)

        # =========================
        # Reward
        # =========================
        dist = np.linalg.norm(error_pos)
        vel_err = np.linalg.norm(error_vel)

        energy_penalty = self._compute_energy_penalty(action)
        smooth_penalty = self._compute_smoothness_penalty(action)

        reward = (
            1.0
            - (2.0 * dist)
            - (0.1 * vel_err)
        )

        if self.use_energy_penalty:
            reward -= self.w_energy * energy_penalty

        if self.use_smoothness_penalty:
            reward -= self.w_smooth * smooth_penalty

        if dist < 0.2:
            reward += 1.0

        if dist > 3.0:
            terminated = True
            reward -= 10.0

        self.prev_action = action.copy()

        return obs, float(reward), terminated, truncated, info


# ==========================================
# 4. LOOP DE TREINAMENTO
# ==========================================
def train(
    use_energy_penalty=True,
    energy_mode="action_l2",
    w_energy=0.02,          # ↑ 4x mais forte
    use_smoothness_penalty=True,
    w_smooth=0.1,          # ↑ mais suavidade
):
    print("[INFO] Iniciando treinamento PPO ENERGY...")
    print(f"[INFO] use_energy_penalty: {use_energy_penalty}")
    print(f"[INFO] energy_mode: {energy_mode}")
    print(f"[INFO] w_energy: {w_energy}")
    print(f"[INFO] use_smoothness_penalty: {use_smoothness_penalty}")
    print(f"[INFO] w_smooth: {w_smooth}")

    env = DummyVecEnv([
        lambda: TrajectoryTrackingEnv(
            use_energy_penalty=use_energy_penalty,
            energy_mode=energy_mode,
            w_energy=w_energy,
            use_smoothness_penalty=use_smoothness_penalty,
            w_smooth=w_smooth,
        )
    ])

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        tensorboard_log="./ppo_traj_energy_tensorboard/",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./logs/",
        name_prefix="ppo_traj_energy"
    )

    model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)

    model.save("ppo_trajectory_energy_final")
    env.save("vec_normalize_energy.pkl")

    print("[OK] Treino concluído!")


if __name__ == "__main__":
    train(
        use_energy_penalty=True,
        energy_mode="action_l2",
        w_energy=0.005,
        use_smoothness_penalty=True,
        w_smooth=0.05,
    )