import csv
import math
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ==========================================
# 1. REGISTRO
# ==========================================
try:
    register(
        id="BlueRov-v0",
        entry_point="bluerov2_gym.envs.bluerov_env:BlueRov",
        max_episode_steps=5000,
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

    def get_reference(self, t):
        t_s = t * self.speed

        x_d = self.radius * math.sin(t_s)
        y_d = self.radius * math.sin(t_s) * math.cos(t_s)

        if t < 20.0:
            z_d = (self.z_target / 20.0) * t
        else:
            z_d = self.z_target

        vx_d = self.radius * math.cos(t_s) * self.speed
        vy_d = self.radius * (math.cos(t_s)**2 - math.sin(t_s)**2) * self.speed

        return np.array([x_d, y_d, z_d]), np.array([vx_d, vy_d, 0.0])


# ==========================================
# 3. EXECUÇÃO PPO
# ==========================================
def run_trajectory_ppo():
    print("[INFO] Iniciando PPO...")

    env = gym.make("BlueRov-v0", render_mode=None)

    try:
        model = PPO.load("ppo_trajectory_final")
        venv = DummyVecEnv([lambda: gym.make("BlueRov-v0")])
        venv = VecNormalize.load("vec_normalize.pkl", venv)
        venv.training = False
        venv.norm_reward = False
    except FileNotFoundError as e:
        print(f"[ERRO] Modelo não encontrado: {e}")
        return

    traj = TrajectoryGenerator()
    dt = 0.1
    steps = 800

    obs, _ = env.reset()
    data = []

    print("[INFO] Simulação iniciada")

    for i in range(steps):
        t = i * dt

        pos_d, vel_d = traj.get_reference(t)

        curr_pos = np.array([
            obs["x"].item(),
            obs["y"].item(),
            obs["z"].item()
        ])

        curr_vel = np.array([
            obs["u"].item(),
            obs["v"].item(),
            obs["w"].item()
        ])

        psi = obs["yaw"].item()

        error_pos_world = curr_pos - pos_d
        error_vel_world = curr_vel - vel_d

        c, s = np.cos(psi), np.sin(psi)

        err_x_body = error_pos_world[0] * c + error_pos_world[1] * s
        err_y_body = -error_pos_world[0] * s + error_pos_world[1] * c
        err_z_body = error_pos_world[2]

        virtual_obs = {k: v.copy() for k, v in obs.items()}
        virtual_obs["x"] = np.array([err_x_body], dtype=np.float32)
        virtual_obs["y"] = np.array([err_y_body], dtype=np.float32)
        virtual_obs["z"] = np.array([err_z_body], dtype=np.float32)
        virtual_obs["u"] = np.array([error_vel_world[0]], dtype=np.float32)

        norm_obs = venv.normalize_obs(virtual_obs)
        action, _ = model.predict(norm_obs, deterministic=True)

        obs, _, terminated, _, _ = env.step(action)

        dist_error = np.linalg.norm(error_pos_world)
        data.append([t, curr_pos[0], curr_pos[1], curr_pos[2], dist_error])

        if i % 100 == 0:
            print(f"T={t:.1f}s | Erro: {dist_error:.2f}m")

        if terminated:
            break

    with open("data_ppo_traj.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "x", "y", "z", "error"])
        writer.writerows(data)

    env.close()
    print("[OK] PPO finalizado")


if __name__ == "__main__":
    run_trajectory_ppo()