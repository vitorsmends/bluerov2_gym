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
        vy_d = self.radius * (math.cos(t_s) ** 2 - math.sin(t_s) ** 2) * self.speed

        return np.array([x_d, y_d, z_d]), np.array([vx_d, vy_d, 0.0])


# ==========================================
# 3. ESTIMATIVA DE ENERGIA DOS ATUADORES
# ==========================================
T200_MAX_THRUST_N = 50.0
T200_MAX_POWER_W = 350.0

BLUEROV2_LENGTH_M = 0.457
BLUEROV2_WIDTH_M = 0.338
HALF_LENGTH = BLUEROV2_LENGTH_M / 2.0
HALF_WIDTH = BLUEROV2_WIDTH_M / 2.0
C45 = 1.0 / np.sqrt(2.0)
YAW_ARM = C45 * (HALF_LENGTH + HALF_WIDTH)

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


def estimate_thruster_power_watts(thruster_forces):
    abs_force = np.abs(thruster_forces)
    force_ratio = np.clip(abs_force / T200_MAX_THRUST_N, 0.0, 1.0)
    power = T200_MAX_POWER_W * (force_ratio ** 1.5)
    return power


def build_energy_header():
    header = ["time", "x", "y", "z", "error"]
    header += ["cmd_surge", "cmd_sway", "cmd_heave", "cmd_roll", "cmd_pitch", "cmd_yaw"]
    header += [f"thruster_{i+1}_force_N" for i in range(6)]
    header += [f"thruster_{i+1}_power_W" for i in range(6)]
    header += [f"thruster_{i+1}_step_energy_J" for i in range(6)]
    header += [f"thruster_{i+1}_cum_energy_J" for i in range(6)]
    header += ["total_power_W", "total_step_energy_J", "total_cum_energy_J"]
    return header


def build_energy_row(t, curr_pos, dist_error, action, thr_forces, thr_power, thr_step_energy, thr_cum_energy):
    row = [t, curr_pos[0], curr_pos[1], curr_pos[2], dist_error]
    row += action.tolist()
    row += thr_forces.tolist()
    row += thr_power.tolist()
    row += thr_step_energy.tolist()
    row += thr_cum_energy.tolist()
    row += [float(np.sum(thr_power)), float(np.sum(thr_step_energy)), float(np.sum(thr_cum_energy))]
    return row


# ==========================================
# 4. EXECUÇÃO PPO - NOVO AGENTE ENERGÉTICO
# ==========================================
def run_trajectory_ppo_energy():
    print("[INFO] Iniciando PPO ENERGY com estimativa de energia...")

    env = gym.make("BlueRov-v0", render_mode=None)

    try:
        model = PPO.load("ppo_trajectory_energy_final")
        venv = DummyVecEnv([lambda: gym.make("BlueRov-v0")])
        venv = VecNormalize.load("vec_normalize_energy.pkl", venv)
        venv.training = False
        venv.norm_reward = False
    except FileNotFoundError as e:
        print(f"[ERRO] Modelo do agente energético não encontrado: {e}")
        return

    traj = TrajectoryGenerator()
    dt = 0.1
    steps = 800

    obs, _ = env.reset()
    data = []
    thruster_cum_energy = np.zeros(6, dtype=float)

    print("[INFO] Simulação do agente energético iniciada")

    for i in range(steps):
        t = i * dt

        pos_d, vel_d = traj.get_reference(t)

        curr_pos = np.array([
            obs["x"].item(),
            obs["y"].item(),
            obs["z"].item()
        ], dtype=float)

        curr_vel = np.array([
            obs["u"].item(),
            obs["v"].item(),
            obs["w"].item()
        ], dtype=float)

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
        action = np.asarray(action, dtype=float).reshape(-1)

        thruster_forces = estimate_thruster_forces_from_action(action)
        thruster_power = estimate_thruster_power_watts(thruster_forces)
        thruster_step_energy = thruster_power * dt
        thruster_cum_energy += thruster_step_energy

        obs, _, terminated, _, _ = env.step(action)

        dist_error = np.linalg.norm(error_pos_world)

        data.append(
            build_energy_row(
                t=t,
                curr_pos=curr_pos,
                dist_error=dist_error,
                action=action,
                thr_forces=thruster_forces,
                thr_power=thruster_power,
                thr_step_energy=thruster_step_energy,
                thr_cum_energy=thruster_cum_energy.copy(),
            )
        )

        if i % 100 == 0:
            print(
                f"T={t:.1f}s | Erro: {dist_error:.2f}m | "
                f"Energia acumulada: {np.sum(thruster_cum_energy):.2f} J"
            )

        if terminated:
            break

    with open("data_ppo_energy_traj.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(build_energy_header())
        writer.writerows(data)

    env.close()
    print("[OK] PPO ENERGY finalizado")
    print("[OK] Arquivo salvo: data_ppo_energy_traj.csv")


if __name__ == "__main__":
    run_trajectory_ppo_energy()