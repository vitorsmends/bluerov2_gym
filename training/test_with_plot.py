import csv
import time
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ==========================================
# 1. REGISTRO DO AMBIENTE
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
# 2. CENÁRIOS DE STATION KEEPING
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
# 3. PARÂMETROS DE POTÊNCIA E ENERGIA
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
    for key in ["x", "y", "z", "roll", "pitch", "yaw", "u", "v", "w", "p", "q", "r"]:
        base_env.state[key] = 0.0
        
    base_env.state["x"] = float(pos[0])
    base_env.state["y"] = float(pos[1])
    base_env.state["z"] = float(pos[2])
    base_env.state["yaw"] = float(yaw)
    return base_env._get_obs()


def wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def build_header():
    header = [
        "controller", "scenario_id", "time",
        "target_x", "target_y", "target_z",
        "x", "y", "z", "euclidean_error", "yaw_error"
    ]
    header += ["cmd_surge", "cmd_sway", "cmd_heave", "cmd_roll", "cmd_pitch", "cmd_yaw"]
    header += [f"thruster_{i+1}_power_W" for i in range(6)]
    header += ["total_power_W", "total_step_energy_J", "total_cum_energy_J"]
    header += [
        "controller_wall_time_s",
        "controller_cpu_time_s",
        "controller_frequency_hz",
        "controller_cum_wall_time_s",
        "controller_cum_cpu_time_s",
    ]
    return header


def run_ppo_stationkeeping():
    print("[INFO] Iniciando PPO Stationkeeping...")

    env = gym.make("BlueRov-v0", render_mode=None)

    try:
        model = PPO.load("bluerov_ppo.zip")
        venv = DummyVecEnv([lambda: gym.make("BlueRov-v0", render_mode=None)])
        venv = VecNormalize.load("bluerov_vec_normalize.pkl", venv)
        venv.training = False
        venv.norm_reward = False
    except FileNotFoundError as e:
        print(f"[ERRO] Arquivos do modelo ou normalizador não encontrados: {e}")
        return

    dt = 0.1
    steps = 800
    data = []

    global_wall_time = 0.0
    global_cpu_time = 0.0
    global_steps = 0

    for scenario in SCENARIOS:
        scenario_id = scenario["id"]
        target = scenario["target"]
        init = scenario["init"]

        print(f"\n[INFO] Cenário {scenario_id} | target={target}")

        env.reset()
        obs = set_env_state(env, init, yaw=YAW_TARGET)

        thruster_cum_energy = np.zeros(6, dtype=float)
        scenario_wall_time = 0.0
        scenario_cpu_time = 0.0

        # Históricos locais para geração dos gráficos de erro por cenário
        plot_history = {"time": [], "euclidean": [], "yaw": []}

        for i in range(steps):
            t = i * dt

            curr_pos = np.array([obs["x"].item(), obs["y"].item(), obs["z"].item()], dtype=float)
            curr_vel = np.array([obs["u"].item(), obs["v"].item(), obs["w"].item()], dtype=float)
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

            wall_t0 = time.perf_counter()
            cpu_t0 = time.process_time()

            norm_obs = venv.normalize_obs(virtual_obs)
            action, _ = model.predict(norm_obs, deterministic=True)
            action_processed = np.asarray(action, dtype=np.float32).reshape(-1)
            action_processed = np.clip(action_processed, -40.0, 40.0)

            cpu_t1 = time.process_time()
            wall_t1 = time.perf_counter()

            controller_wall_time = wall_t1 - wall_t0
            controller_cpu_time = cpu_t1 - cpu_t0
            controller_frequency = 1.0 / controller_wall_time if controller_wall_time > 0.0 else np.nan

            scenario_wall_time += controller_wall_time
            scenario_cpu_time += controller_cpu_time

            global_wall_time += controller_wall_time
            global_cpu_time += controller_cpu_time
            global_steps += 1

            thruster_forces = estimate_thruster_forces_from_action(action_processed)
            thruster_power = estimate_thruster_power_watts(thruster_forces)
            thruster_step_energy = thruster_power * dt
            thruster_cum_energy += thruster_step_energy

            obs, _, terminated, truncated, _ = env.step(action_processed)

            dist_error = float(np.linalg.norm(error_pos_world))
            yaw_error = wrap_angle(psi - YAW_TARGET)

            plot_history["time"].append(t)
            plot_history["euclidean"].append(dist_error)
            plot_history["yaw"].append(yaw_error)

            row = [
                "PPO", scenario_id, t,
                target[0], target[1], target[2],
                curr_pos[0], curr_pos[1], curr_pos[2], dist_error, yaw_error
            ]
            row += action_processed.tolist()
            row += thruster_power.tolist()
            row += [
                float(np.sum(thruster_power)),
                float(np.sum(thruster_step_energy)),
                float(np.sum(thruster_cum_energy)),
                float(controller_wall_time),
                float(controller_cpu_time),
                float(controller_frequency),
                float(scenario_wall_time),
                float(scenario_cpu_time),
            ]
            data.append(row)

            if terminated or truncated:
                break

        # Geração dos plots de desempenho para o cenário concluído
        fig, axs = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle(f"Erros de Malha Fechada - Cenário {scenario_id}", fontsize=12)

        axs[0].plot(plot_history["time"], plot_history["euclidean"], color="tab:orange", linewidth=2)
        axs[0].axhline(0.0, color="black", linestyle="--", alpha=0.6)
        axs[0].set_ylabel("Erro Euclidiano (m)")
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(plot_history["time"], plot_history["yaw"], color="tab:purple", linewidth=2)
        axs[1].axhline(0.0, color="black", linestyle="--", alpha=0.6)
        axs[1].set_xlabel("Tempo (s)")
        axs[1].set_ylabel("Erro de Yaw (rad)")
        axs[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    with open("data_ppo_stationkeeping.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(build_header())
        writer.writerows(data)

    env.close()

    if global_steps > 0:
        print("\n[RESUMO COMPUTAÇÃO PPO]")
        print(f"  Steps totais executados: {global_steps}")
        print(f"  Wall time médio por ciclo: {global_wall_time / global_steps:.6e} s")
        print(f"  Frequência média efetiva : {global_steps / global_wall_time:.2f} Hz")

    print("[OK] Arquivo data_ppo_stationkeeping.csv gravado com sucesso.")


if __name__ == "__main__":
    run_ppo_stationkeeping()