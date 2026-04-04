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
# 3. ESTIMATIVA DE ENERGIA
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


# ==========================================
# 5. CONTROLADOR PID
# ==========================================
class SingleAxisCascadedPID:
    def __init__(self, kp_pos, kp_vel, ki_vel, dt, u_sat, vel_sat=0.5):
        self.kp_pos = kp_pos
        self.kp_vel = kp_vel
        self.ki_vel = ki_vel
        self.dt = dt
        self.u_sat = u_sat
        self.vel_sat = vel_sat
        self.integral_vel = 0.0

    def reset(self):
        self.integral_vel = 0.0

    def update(self, pos_error, current_vel, ff_vel=0.0):
        if np.isnan(pos_error) or np.isnan(current_vel):
            return 0.0

        target_vel = (self.kp_pos * pos_error) + ff_vel
        target_vel = np.clip(target_vel, -self.vel_sat, self.vel_sat)

        vel_error = target_vel - current_vel
        self.integral_vel += vel_error * self.dt
        self.integral_vel = np.clip(self.integral_vel, -2.0, 2.0)

        u = (self.kp_vel * vel_error) + (self.ki_vel * self.integral_vel)
        return float(np.clip(u, -self.u_sat, self.u_sat))


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


def run_pid_stationkeeping():
    print("[INFO] Iniciando PID Stationkeeping...")

    env = gym.make("BlueRov-v0", render_mode=None)

    dt = 0.1
    steps = 800
    u_max = 20.0

    data = []

    for scenario in SCENARIOS:
        scenario_id = scenario["id"]
        target = scenario["target"]
        init = scenario["init"]

        print(f"[INFO] Cenário {scenario_id} | target={target}")

        obs, _ = env.reset()
        obs = set_env_state(env, init, yaw=YAW_TARGET)

        pid_x = SingleAxisCascadedPID(0.5, 5.0, 0.1, dt, u_max)
        pid_y = SingleAxisCascadedPID(0.5, 5.0, 0.1, dt, u_max)
        pid_z = SingleAxisCascadedPID(0.8, 8.0, 0.2, dt, u_max)
        pid_roll = SingleAxisCascadedPID(1.0, 4.0, 0.0, dt, u_max)
        pid_pitch = SingleAxisCascadedPID(1.0, 4.0, 0.0, dt, u_max)
        pid_yaw = SingleAxisCascadedPID(1.0, 4.0, 0.0, dt, u_max)

        controllers = [pid_x, pid_y, pid_z, pid_roll, pid_pitch, pid_yaw]
        thruster_cum_energy = np.zeros(6, dtype=float)

        for i in range(steps):
            t = i * dt

            curr_pos = np.array([obs["x"].item(), obs["y"].item(), obs["z"].item()], dtype=float)
            curr_att = np.array([obs["roll"].item(), obs["pitch"].item(), obs["yaw"].item()], dtype=float)
            curr_vel = np.array([
                obs["u"].item(), obs["v"].item(), obs["w"].item(),
                obs["p"].item(), obs["q"].item(), obs["r"].item()
            ], dtype=float)

            if np.any(np.isnan(curr_pos)) or np.any(np.isnan(curr_vel)):
                print(f"[FALHA] NaN no cenário {scenario_id}, passo {i}")
                break

            if np.linalg.norm(curr_pos) > 50.0:
                print(f"[FALHA] Instabilidade no cenário {scenario_id}, passo {i}")
                break

            err_pos_world = target - curr_pos
            err_att = np.array([0.0, 0.0, wrap_angle(YAW_TARGET - curr_att[2])], dtype=float)

            yaw = curr_att[2]
            err_xy_body = world_to_body_xy(err_pos_world[:2], yaw)
            err_pos_body = np.array([err_xy_body[0], err_xy_body[1], err_pos_world[2]], dtype=float)

            errors = np.concatenate((err_pos_body, err_att))
            ff = np.zeros(6, dtype=float)

            action = np.zeros(6, dtype=float)
            for j in range(6):
                action[j] = controllers[j].update(errors[j], curr_vel[j], ff[j])

            thruster_forces = estimate_thruster_forces_from_action(action)
            thruster_power = estimate_thruster_power_watts(thruster_forces)
            thruster_step_energy = thruster_power * dt
            thruster_cum_energy += thruster_step_energy

            obs, _, terminated, truncated, _ = env.step(action)

            dist_error = float(np.linalg.norm(err_pos_world))

            row = [
                "PID", scenario_id, t,
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

    with open("data_pid_stationkeeping.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(build_header())
        writer.writerows(data)

    env.close()
    print("[OK] data_pid_stationkeeping.csv gerado.")


if __name__ == "__main__":
    run_pid_stationkeeping()