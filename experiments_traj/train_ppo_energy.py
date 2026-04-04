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
except:
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

        return np.array([x, y, z]), np.array([0.0, 0.0, yaw]), np.array([vx, vy, vz])


# ==========================================
# 3. UTILITÁRIOS DE ENERGIA
# ==========================================
T200_MAX_THRUST_N = 50.0
T200_MAX_POWER_W = 350.0

BLUEROV2_LENGTH_M = 0.457
BLUEROV2_WIDTH_M = 0.338
HALF_LENGTH = BLUEROV2_LENGTH_M / 2.0
HALF_WIDTH = BLUEROV2_WIDTH_M / 2.0
C45 = 1.0 / np.sqrt(2.0)
YAW_ARM = C45 * (HALF_LENGTH + HALF_WIDTH)

# Matriz simplificada de alocação:
# surge, sway, heave, yaw -> 6 thrusters
B_ALLOC = np.array([
    [ C45,  C45,  C45,  C45, 0.0, 0.0],
    [-C45,  C45,  C45, -C45, 0.0, 0.0],
    [ 0.0,  0.0,  0.0,  0.0, 1.0, 1.0],
    [-YAW_ARM, YAW_ARM, -YAW_ARM, YAW_ARM, 0.0, 0.0],
], dtype=float)

B_ALLOC_PINV = np.linalg.pinv(B_ALLOC)


def estimate_thruster_forces_from_action(action_6d):
    action_6d = np.asarray(action_6d, dtype=float).reshape(-1)
    surge, sway, heave, roll, pitch, yaw = action_6d
    tau_actuated = np.array([surge, sway, heave, yaw], dtype=float)
    thruster_forces = B_ALLOC_PINV @ tau_actuated
    return thruster_forces


def estimate_thruster_power_watts(action_6d):
    thruster_forces = estimate_thruster_forces_from_action(action_6d)
    abs_force = np.abs(thruster_forces)
    force_ratio = np.clip(abs_force / T200_MAX_THRUST_N, 0.0, 1.0)

    thruster_power = T200_MAX_POWER_W * (force_ratio ** 1.5)
    total_power = float(np.sum(thruster_power))
    return total_power


# ==========================================
# 4. AMBIENTE PERSONALIZADO
# ==========================================
class TrajectoryTrackingEnv(original_env.BlueRov):
    def __init__(
        self,
        use_energy_penalty=True,
        energy_mode="thruster_power",   # "thruster_power" ou "action_l2"
        w_energy=0.1,
    ):
        super().__init__(render_mode=None)
        self.traj = TrajectoryGenerator()
        self.current_t = 0.0
        self.dt = 0.1

        self.use_energy_penalty = use_energy_penalty
        self.energy_mode = energy_mode
        self.w_energy = w_energy

    def reset(self, seed=None, options=None):
        self.current_t = np.random.uniform(0, 50.0)

        target_pos, target_att, _ = self.traj.get_state_at_time(self.current_t)

        noise_pos = np.random.uniform(-0.2, 0.2, 3)
        initial_pos = target_pos + noise_pos

        self.state = {
            "x": initial_pos[0], "y": initial_pos[1], "z": initial_pos[2],
            "roll": 0.0, "pitch": 0.0, "yaw": target_att[2],
            "u": 0.0, "v": 0.0, "w": 0.0,
            "p": 0.0, "q": 0.0, "r": 0.0
        }

        return self._get_obs(), {}

    def _compute_energy_penalty(self, action):
        action = np.asarray(action, dtype=float).reshape(-1)

        if not self.use_energy_penalty:
            return 0.0

        if self.energy_mode == "action_l2":
            return float(np.sum(action ** 2))

        if self.energy_mode == "thruster_power":
            total_power_w = estimate_thruster_power_watts(action)
            step_energy_j = total_power_w * self.dt
            return float(step_energy_j)

        raise ValueError(f"energy_mode inválido: {self.energy_mode}")

    def step(self, action):
        self.current_t += self.dt

        obs, _, terminated, truncated, info = super().step(action)

        tgt_pos, tgt_att, tgt_vel = self.traj.get_state_at_time(self.current_t)

        curr_pos = np.array([obs["x"][0], obs["y"][0], obs["z"][0]], dtype=float)
        curr_vel = np.array([obs["u"][0], obs["v"][0], obs["w"][0]], dtype=float)

        error_pos = curr_pos - tgt_pos
        error_vel = curr_vel - tgt_vel

        psi = obs["yaw"][0]
        c, s = np.cos(psi), np.sin(psi)

        err_x_body = error_pos[0] * c + error_pos[1] * s
        err_y_body = -error_pos[0] * s + error_pos[1] * c
        err_z_body = error_pos[2]

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

        reward = (
            1.0
            - (2.0 * dist)
            - (0.1 * vel_err)
        )

        if self.use_energy_penalty:
            reward -= self.w_energy * energy_penalty

        if dist > 3.0:
            terminated = True
            reward -= 10.0

        return obs, float(reward), terminated, truncated, info


# ==========================================
# 5. LOOP DE TREINAMENTO
# ==========================================
def train(
    use_energy_penalty=True,
    energy_mode="thruster_power",
    w_energy=0.1,
):
    print("[INFO] Iniciando treinamento PPO para trajetória...")
    print(f"[INFO] Penalização energética: {use_energy_penalty}")
    print(f"[INFO] Modo de energia: {energy_mode}")
    print(f"[INFO] Peso energético: {w_energy}")

    env = DummyVecEnv([
        lambda: TrajectoryTrackingEnv(
            use_energy_penalty=use_energy_penalty,
            energy_mode=energy_mode,
            w_energy=w_energy,
        )
    ])

    # Mudança importante: sem normalização da reward
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        tensorboard_log="./ppo_traj_energy_v2_tensorboard/",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./logs/",
        name_prefix="ppo_traj_energy_v2"
    )

    model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)

    model.save("ppo_trajectory_energy_v2_final")
    env.save("vec_normalize_energy_v2.pkl")
    print("[OK] Treino concluído! Modelos salvos.")


if __name__ == "__main__":
    train(
        use_energy_penalty=True,
        energy_mode="thruster_power",
        w_energy=0.1,
    )