import csv
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

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
# 2. CENÁRIOS DE STATIONKEEPING
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
# 3. ESTIMATIVA DE ENERGIA DOS ATUADORES
# ==========================================
T200_MAX_THRUST_N = 51.5
T200_MAX_POWER_W = 390.0

BLUEROV2_LENGTH_M = 0.457
BLUEROV2_WIDTH_M = 0.338
HALF_LENGTH = BLUEROV2_LENGTH_M / 2.0
HALF_WIDTH = BLUEROV2_WIDTH_M / 2.0
C45 = 1.0 / np.sqrt(2.0)
YAW_ARM = C45 * (HALF_LENGTH + HALF_WIDTH)

B_ALLOC = np.array([
    [ C45,  C45,  C45,  C45, 0.0, 0.0],   # surge
    [-C45,  C45,  C45, -C45, 0.0, 0.0],   # sway
    [ 0.0,  0.0,  0.0,  0.0, 1.0, 1.0],   # heave
    [-YAW_ARM, YAW_ARM, -YAW_ARM, YAW_ARM, 0.0, 0.0],  # yaw
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


# ==========================================
# 4. UTILITÁRIOS
# ==========================================
def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def world_to_body_xy(vec_xy_world, yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    x_b = vec_xy_world[0] * c + vec_xy_world[1] * s
    y_b = -vec_xy_world[0] * s + vec_xy_world[1] * c
    return np.array([x_b, y_b], dtype=float)


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


# ==========================================
# 5. PARÂMETROS DO SMC
# ==========================================
LAMBDA = [1.5, 1.5, 2.0, 1.0, 1.0, 1.0]
EPS = [0.2, 0.2, 0.1, 0.1, 0.1, 0.1]
K_INIT = [10.0, 10.0, 20.0, 5.0, 5.0, 5.0]
K_BAR = [5.0, 5.0, 10.0, 2.0, 2.0, 2.0]
MU = [0.05, 0.05, 0.05, 0.02, 0.02, 0.02]
ALPHA = [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]


# ==========================================
# 6. CONTROLADOR SMC
# ==========================================
class SlidingModeController:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.lam = np.array(LAMBDA, dtype=float)
        self.eps = np.array(EPS, dtype=float)
        self.alpha = np.array(ALPHA, dtype=float)
        self.kbar = np.array(K_BAR, dtype=float)
        self.mu = np.array(MU, dtype=float)
        self.K = np.array(K_INIT, dtype=float)

        self.u_sat = 20.0

    def step(self, pos_error_body, current_vel_body):
        vel_error = -current_vel_body
        s_surface = vel_error + self.lam * pos_error_body

        wrench = np.zeros(6, dtype=float)

        for i in range(6):
            adaptation_rate = self.kbar[i] * np.sign(np.abs(s_surface[i]) - self.mu[i])

            self.K[i] += self.dt * adaptation_rate

            if self.K[i] < self.alpha[i]:
                self.K[i] = self.alpha[i]

            if self.K[i] > 60.0:
                self.K[i] = 60.0

            if np.abs(s_surface[i]) > self.eps[i]:
                wrench[i] = self.K[i] * np.sign(s_surface[i])
            else:
                wrench[i] = self.K[i] * (s_surface[i] / self.eps[i])

        return np.clip(wrench, -self.u_sat, self.u_sat)


# ==========================================
# 7. EXECUÇÃO
# ==========================================
def run_smc_stationkeeping():
    print("[INFO] Iniciando SMC Stationkeeping...")

    env = gym.make("BlueRov-v0", render_mode=None)

    dt = 0.1
    steps = 400
    data = []

    for scenario in SCENARIOS:
        scenario_id = scenario["id"]
        target = scenario["target"]
        init = scenario["init"]

        print(f"[INFO] Cenário {scenario_id} | target={target}")

        obs, _ = env.reset()
        obs = set_env_state(env, init, yaw=YAW_TARGET)

        smc = SlidingModeController(dt=dt)
        thruster_cum_energy = np.zeros(6, dtype=float)

        for i in range(steps):
            t = i * dt

            curr_pos = np.array([
                obs["x"].item(),
                obs["y"].item(),
                obs["z"].item()
            ], dtype=float)

            curr_att = np.array([
                obs["roll"].item(),
                obs["pitch"].item(),
                obs["yaw"].item()
            ], dtype=float)

            curr_vel_lin = np.array([
                obs["u"].item(),
                obs["v"].item(),
                obs["w"].item()
            ], dtype=float)

            curr_vel_ang = np.array([
                obs["p"].item(),
                obs["q"].item(),
                obs["r"].item()
            ], dtype=float)

            if np.any(np.isnan(curr_pos)) or np.any(np.isnan(curr_vel_lin)) or np.any(np.isnan(curr_vel_ang)):
                print(f"[FALHA] NaN no cenário {scenario_id}, passo {i}")
                break

            if np.linalg.norm(curr_pos) > 50.0:
                print(f"[FALHA] Instabilidade no cenário {scenario_id}, passo {i}")
                break

            err_pos_world = target - curr_pos
            err_att_world = np.array([0.0, 0.0, wrap_angle(YAW_TARGET - curr_att[2])], dtype=float)

            psi = curr_att[2]
            err_xy_body = world_to_body_xy(err_pos_world[:2], psi)

            pos_error_body = np.concatenate((
                np.array([err_xy_body[0], err_xy_body[1], err_pos_world[2]], dtype=float),
                err_att_world
            ))

            current_vel_body = np.concatenate((curr_vel_lin, curr_vel_ang))

            action = smc.step(pos_error_body, current_vel_body)

            thruster_forces = estimate_thruster_forces_from_action(action)
            thruster_power = estimate_thruster_power_watts(thruster_forces)
            thruster_step_energy = thruster_power * dt
            thruster_cum_energy += thruster_step_energy

            obs, _, terminated, truncated, _ = env.step(action)

            dist_error = float(np.linalg.norm(err_pos_world))

            row = [
                "SMC", scenario_id, t,
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

            if i % 50 == 0:
                print(
                    f"[INFO] Cenário {scenario_id} | "
                    f"T={t:.1f}s | Erro={dist_error:.3f}m | Kx={smc.K[0]:.1f}"
                )

            if terminated or truncated:
                break

    with open("data_smc_stationkeeping.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(build_header())
        writer.writerows(data)

    env.close()
    print("[OK] data_smc_stationkeeping.csv gerado.")


if __name__ == "__main__":
    run_smc_stationkeeping()