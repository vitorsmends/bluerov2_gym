import csv
import math
import numpy as np
import gymnasium as gym

from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


ENV_ID = "BlueRov-v0"
MODEL_PATH = "./models/ppo_trajectory_final"
VECNORM_PATH = "./models/vec_normalize.pkl"


try:
    register(
        id=ENV_ID,
        entry_point="bluerov2_gym.envs.bluerov_env:BlueRov",
        max_episode_steps=2000,
    )
except gym.error.Error:
    pass


class TrajectoryGenerator:
    def __init__(self):
        self.radius = 1.0
        self.speed = 0.15
        self.z_target = -0.5

    def get_reference(self, t):
        ts = t * self.speed

        x_d = self.radius * math.sin(ts)
        y_d = self.radius * math.sin(ts) * math.cos(ts)

        if t < 10.0:
            z_d = (self.z_target / 10.0) * t
            vz_d = self.z_target / 10.0
        else:
            z_d = self.z_target
            vz_d = 0.0

        vx_d = self.radius * math.cos(ts) * self.speed
        vy_d = self.radius * (math.cos(ts) ** 2 - math.sin(ts) ** 2) * self.speed
        yaw_d = math.atan2(vy_d, vx_d)

        return (
            np.array([x_d, y_d, z_d], dtype=np.float32),
            np.array([vx_d, vy_d, vz_d], dtype=np.float32),
            float(yaw_d),
        )


def scalar(value):
    return float(np.asarray(value).reshape(-1)[0])


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def build_virtual_obs(obs, pos_d, vel_d, yaw_d):
    curr_pos = np.array([
        scalar(obs["x"]),
        scalar(obs["y"]),
        scalar(obs["z"]),
    ], dtype=np.float32)

    curr_vel = np.array([
        scalar(obs["u"]),
        scalar(obs["v"]),
        scalar(obs["w"]),
    ], dtype=np.float32)

    yaw = scalar(obs["yaw"])

    error_pos_world = curr_pos - pos_d
    error_vel_world = curr_vel - vel_d
    yaw_error = wrap_angle(yaw - yaw_d)

    # Igual ao treino: erro no referencial da trajetória desejada
    c = math.cos(yaw_d)
    s = math.sin(yaw_d)

    err_x_body = error_pos_world[0] * c + error_pos_world[1] * s
    err_y_body = -error_pos_world[0] * s + error_pos_world[1] * c
    err_z_body = error_pos_world[2]

    vel_x_body = error_vel_world[0] * c + error_vel_world[1] * s
    vel_y_body = -error_vel_world[0] * s + error_vel_world[1] * c
    vel_z_body = error_vel_world[2]

    virtual_obs = {
        k: np.asarray(v, dtype=np.float32).copy()
        for k, v in obs.items()
    }

    virtual_obs["x"] = np.array([err_x_body], dtype=np.float32)
    virtual_obs["y"] = np.array([err_y_body], dtype=np.float32)
    virtual_obs["z"] = np.array([err_z_body], dtype=np.float32)

    virtual_obs["u"] = np.array([vel_x_body], dtype=np.float32)
    virtual_obs["v"] = np.array([vel_y_body], dtype=np.float32)
    virtual_obs["w"] = np.array([vel_z_body], dtype=np.float32)

    virtual_obs["yaw"] = np.array([yaw_error], dtype=np.float32)

    return virtual_obs, curr_pos, curr_vel, error_pos_world, error_vel_world, yaw_error


def make_vec_env_for_normalization():
    return DummyVecEnv([
        lambda: gym.make(ENV_ID, render_mode=None)
    ])


def run_trajectory_ppo():
    print("[INFO] Iniciando PPO Trajectory Tracking...")

    env = gym.make(ENV_ID, render_mode=None)

    try:
        venv = make_vec_env_for_normalization()
        venv = VecNormalize.load(VECNORM_PATH, venv)
        venv.training = False
        venv.norm_reward = False

        model = PPO.load(MODEL_PATH)

    except FileNotFoundError as e:
        print(f"ERRO: arquivos de treino não encontrados: {e}")
        return

    traj = TrajectoryGenerator()

    dt = 0.1
    n_steps = 1000

    obs, _ = env.reset()
    data = []

    print("Simulação iniciada.")

    for i in range(n_steps):
        t = i * dt

        pos_d, vel_d, yaw_d = traj.get_reference(t)

        virtual_obs, curr_pos, curr_vel, error_pos_world, error_vel_world, yaw_error = (
            build_virtual_obs(obs, pos_d, vel_d, yaw_d)
        )

        norm_obs = venv.normalize_obs(virtual_obs)

        action, _ = model.predict(norm_obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -40.0, 40.0)

        obs, _, terminated, truncated, info = env.step(action)

        dist_error = float(np.linalg.norm(error_pos_world))
        vel_error = float(np.linalg.norm(error_vel_world))

        data.append([
            t,
            curr_pos[0], curr_pos[1], curr_pos[2],
            pos_d[0], pos_d[1], pos_d[2],
            curr_vel[0], curr_vel[1], curr_vel[2],
            vel_d[0], vel_d[1], vel_d[2],
            dist_error,
            vel_error,
            yaw_error,
            yaw_d,
            action[0], action[1], action[2],
            action[3], action[4], action[5],
        ])

        if i % 100 == 0:
            print(
                f"T={t:.1f}s | "
                f"Erro pos: {dist_error:.3f} m | "
                f"Erro vel: {vel_error:.3f} m/s | "
                f"Yaw err: {yaw_error:.3f} rad"
            )

        if terminated or truncated:
            print(f"Episódio encerrado em t={t:.1f}s")
            break

    with open("data_ppo_traj.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "x", "y", "z",
            "x_ref", "y_ref", "z_ref",
            "u", "v", "w",
            "u_ref", "v_ref", "w_ref",
            "position_error",
            "velocity_error",
            "yaw_error",
            "yaw_ref",
            "T1", "T2", "T3", "T4", "T5", "T6",
        ])
        writer.writerows(data)

    env.close()
    venv.close()

    print("Sucesso! Arquivo 'data_ppo_traj.csv' gerado.")


if __name__ == "__main__":
    run_trajectory_ppo()