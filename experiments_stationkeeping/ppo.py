import csv
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
except Exception:
    pass


# ==========================================
# 2. CENÁRIOS
# ==========================================
SCENARIOS = [
    {"id": 1, "target": np.array([0.0,  0.0, -0.5]), "init": np.array([0.0, 0.0, 0.0])},
    {"id": 2, "target": np.array([0.5,  0.0, -0.5]), "init": np.array([0.0, 0.0, 0.0])},
    {"id": 3, "target": np.array([-0.5, 0.0, -0.5]), "init": np.array([0.0, 0.0, 0.0])},
    {"id": 4, "target": np.array([0.0,  0.5, -0.5]), "init": np.array([0.0, 0.0, 0.0])},
    {"id": 5, "target": np.array([0.0, -0.5, -0.5]), "init": np.array([0.0, 0.0, 0.0])},
    {"id": 6, "target": np.array([0.5,  0.5, -0.7]), "init": np.array([0.0, 0.0, 0.0])},
]

YAW_TARGET = 0.0


# ==========================================
# 3. ENERGIA
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
    return B_ALLOC_PINV @ tau_actuated


def estimate_thruster_power_watts(thruster_forces):
    abs_force = np.abs(thruster_forces)
    force_ratio = np.clip(abs_force / T200_MAX_THRUST_N, 0.0, 1.0)
    return T200_MAX_POWER_W * (force_ratio ** 1.5)


def set_env_state(env, pos, yaw=0.0):
    base_env = env.unwrapped
    base_env.state = {
        "x": float(pos[0]),
        "y": float(pos[1]),
        "z": float(pos[2]),
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": float(yaw),
        "u": 0.0,
        "v": 0.0,
        "w": 0.0,
        "p": 0.0,
        "q": 0.0,
        "r": 0.0,
    }
    return base_env._get_obs()


def build_header():
    header = [
        "controller", "scenario_id", "time",
        "target_x", "target_y", "target_z",
        "x", "y", "z", "error"
    ]
    header += ["cmd_surge", "cmd_sway", "cmd_heave", "cmd_roll", "cmd_pitch", "cmd_yaw"]
    header += [f"thruster_{i+1}_power_W" for i in range(6)]
    header += ["total_power_W", "total_step_energy_J", "total_cum_energy_J"]
    return header


def run_ppo_stationkeeping():
    print("[INFO] Iniciando PPO Stationkeeping...")

    env = gym.make("BlueRov-v0", render_mode=None)

    try:
        model = PPO.load("ppo_trajectory_final")
        venv = DummyVecEnv([lambda: gym.make("BlueRov-v0")])
        venv = VecNormalize.load("vec_normalize.pkl", venv)
        venv.training = False
        venv.norm_reward = False
    except FileNotFoundError as e:
        print(f"[ERRO] Modelo PPO stationkeeping não encontrado: {e}")
        print("[DICA] Ajuste o nome do modelo se o seu arquivo salvo for diferente.")
        return

    dt = 0.1
    steps = 800
    data = []

    for scenario in SCENARIOS:
        scenario_id = scenario["id"]
        target = scenario["target"]
        init = scenario["init"]

        print(f"[INFO] Cenário {scenario_id} | target={target}")

        obs, _ = env.reset()
        obs = set_env_state(env, init, yaw=YAW_TARGET)

        thruster_cum_energy = np.zeros(6, dtype=float)

        for i in range(steps):
            t = i * dt

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

            error_pos_world = curr_pos - target
            error_vel_world = curr_vel - np.zeros(3, dtype=float)

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

            obs, _, terminated, truncated, _ = env.step(action)

            dist_error = float(np.linalg.norm(error_pos_world))

            row = [
                "PPO", scenario_id, t,
                target[0], target[1], target[2],
                curr_pos[0], curr_pos[1], curr_pos[2], dist_error
            ]
            row += action.tolist()
            row += thruster_power.tolist()
            row += [
                float(np.sum(thruster_power)),
                float(np.sum(thruster_step_energy)),
                float(np.sum(thruster_cum_energy))
            ]
            data.append(row)

            if terminated or truncated:
                break

    with open("data_ppo_stationkeeping.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(build_header())
        writer.writerows(data)

    env.close()
    print("[OK] data_ppo_stationkeeping.csv gerado.")


if __name__ == "__main__":
    run_ppo_stationkeeping()