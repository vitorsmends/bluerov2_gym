import os
import json
import csv
import time
from datetime import datetime
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

# 1. REGISTRATION
try:
    register(
        id="BlueRov-v0",
        entry_point="bluerov2_gym.envs.bluerov_env:BlueRov",
        max_episode_steps=5000,
    )
except Exception:
    pass

# 2. STATIONKEEPING SCENARIOS
SCENARIOS = [
    {"id": 1, "target": np.array([0.0,  0.0, -0.5]), "init": np.array([0.0, 0.0, 0.0])},
    {"id": 2, "target": np.array([0.5,  0.0, -0.5]), "init": np.array([0.0, 0.0, 0.0])},
    {"id": 3, "target": np.array([-0.5, 0.0, -0.5]), "init": np.array([0.0, 0.0, 0.0])},
    {"id": 4, "target": np.array([0.0,  0.5, -0.5]), "init": np.array([0.0, 0.0, 0.0])},
    {"id": 5, "target": np.array([0.0, -0.5, -0.5]), "init": np.array([0.0, 0.0, 0.0])},
    {"id": 6, "target": np.array([0.5,  0.5, -0.7]), "init": np.array([0.0, 0.0, 0.0])},
]

YAW_TARGET = 0.0

# 3. ENERGY ESTIMATION & THRUST ALLOCATION
T200_MAX_THRUST_N = 35.0
T200_MAX_POWER_W = 350.0

# 8 Thrusters Geometry for BlueROV2 Heavy
ARMS_M = np.array([
    [ 0.156,  0.111,  0.085],
    [ 0.156, -0.111,  0.085],
    [-0.156,  0.111,  0.085],
    [-0.156, -0.111,  0.085],
    [ 0.120,  0.218,  0.000],
    [ 0.120, -0.218,  0.000],
    [-0.120,  0.218,  0.000],
    [-0.120, -0.218,  0.000],
], dtype=float)

DIRS = np.array([
    [ np.cos(np.pi/4),    -np.sin(np.pi/4),     0.0],
    [ np.cos(-np.pi/4),   -np.sin(-np.pi/4),    0.0],
    [ np.cos(-3*np.pi/4), -np.sin(-3*np.pi/4),  0.0],
    [ np.cos( 3*np.pi/4), -np.sin( 3*np.pi/4),  0.0],
    [ 0.0, 0.0, 1.0],
    [ 0.0, 0.0, 1.0],
    [ 0.0, 0.0, 1.0],
    [ 0.0, 0.0, 1.0],
], dtype=float)

B_ALLOC = np.zeros((6, 8), dtype=float)
for i in range(8):
    B_ALLOC[:3, i] = DIRS[i]
    B_ALLOC[3:, i] = np.cross(ARMS_M[i], DIRS[i])

B_ALLOC_PINV = np.linalg.pinv(B_ALLOC)

def calculate_thruster_commands(tau_6d):
    # Map 6D forces/torques to 8 thruster forces
    thruster_forces = B_ALLOC_PINV @ tau_6d
    # Normalize to [-1, 1] range based on max thrust
    action_8d = thruster_forces / T200_MAX_THRUST_N
    return np.clip(action_8d, -1.0, 1.0)

def estimate_thruster_power_watts(action_8d):
    abs_force = np.abs(action_8d * T200_MAX_THRUST_N)
    force_ratio = np.clip(abs_force / T200_MAX_THRUST_N, 0.0, 1.0)
    return T200_MAX_POWER_W * (force_ratio ** 1.5)

# 4. UTILITIES
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

# 5. CASCADED PID CONTROLLER
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
    header += [f"cmd_thruster_{i+1}" for i in range(8)]
    header += [f"thruster_{i+1}_power_W" for i in range(8)]
    header += ["total_power_W", "total_step_energy_J", "total_cum_energy_J"]

    header += [
        "controller_wall_time_s",
        "controller_cpu_time_s",
        "controller_frequency_hz",
        "controller_cum_wall_time_s",
        "controller_cum_cpu_time_s",
    ]
    return header

def run_pid_stationkeeping():
    print("[INFO] Starting PID Stationkeeping setup...")

    # Create run directory
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"pid_eval_{now}"
    base_dir = "trained_models"
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    dt = 0.1
    steps = 800
    u_max = 50.0  # Max force requested per axis before allocation

    # Save hyperparameters
    hyperparameters = {
        "dt": dt,
        "steps": steps,
        "u_max_pid": u_max,
        "t200_max_thrust": T200_MAX_THRUST_N,
        "t200_max_power": T200_MAX_POWER_W,
        "scenarios": SCENARIOS
    }
    
    json_path = os.path.join(run_dir, "hyperparameters.json")
    with open(json_path, "w") as f:
        json.dump(hyperparameters, f, indent=4)

    env = gym.make("BlueRov-v0", render_mode=None)
    data = []

    global_wall_time = 0.0
    global_cpu_time = 0.0
    global_steps = 0

    print(f"[INFO] Data will be saved to: {run_dir}")

    for scenario in SCENARIOS:
        scenario_id = scenario["id"]
        target = scenario["target"]
        init = scenario["init"]

        print(f"[INFO] Running Scenario {scenario_id} | target={target}")

        obs, _ = env.reset()
        obs = set_env_state(env, init, yaw=YAW_TARGET)

        pid_x = SingleAxisCascadedPID(0.5, 5.0, 0.1, dt, u_max)
        pid_y = SingleAxisCascadedPID(0.5, 5.0, 0.1, dt, u_max)
        pid_z = SingleAxisCascadedPID(0.8, 8.0, 0.2, dt, u_max)
        pid_roll = SingleAxisCascadedPID(1.0, 4.0, 0.0, dt, u_max)
        pid_pitch = SingleAxisCascadedPID(1.0, 4.0, 0.0, dt, u_max)
        pid_yaw = SingleAxisCascadedPID(1.0, 4.0, 0.0, dt, u_max)

        controllers = [pid_x, pid_y, pid_z, pid_roll, pid_pitch, pid_yaw]
        thruster_cum_energy = np.zeros(8, dtype=float)

        scenario_wall_time = 0.0
        scenario_cpu_time = 0.0
        scenario_steps = 0

        for i in range(steps):
            t = i * dt

            curr_pos = np.array([obs["x"].item(), obs["y"].item(), obs["z"].item()], dtype=float)
            curr_att = np.array([obs["roll"].item(), obs["pitch"].item(), obs["yaw"].item()], dtype=float)
            curr_vel = np.array([
                obs["u"].item(), obs["v"].item(), obs["w"].item(),
                obs["p"].item(), obs["q"].item(), obs["r"].item()
            ], dtype=float)

            if np.any(np.isnan(curr_pos)) or np.any(np.isnan(curr_vel)):
                print(f"[FAIL] NaN detected in scenario {scenario_id}, step {i}")
                break

            if np.linalg.norm(curr_pos) > 50.0:
                print(f"[FAIL] Instability detected in scenario {scenario_id}, step {i}")
                break

            err_pos_world = target - curr_pos
            err_att = np.array([0.0, 0.0, wrap_angle(YAW_TARGET - curr_att[2])], dtype=float)

            yaw = curr_att[2]
            err_xy_body = world_to_body_xy(err_pos_world[:2], yaw)
            err_pos_body = np.array([err_xy_body[0], err_xy_body[1], err_pos_world[2]], dtype=float)

            errors = np.concatenate((err_pos_body, err_att))
            ff = np.zeros(6, dtype=float)

            wall_t0 = time.perf_counter()
            cpu_t0 = time.process_time()

            tau_cmd = np.zeros(6, dtype=float)
            for j in range(6):
                tau_cmd[j] = controllers[j].update(errors[j], curr_vel[j], ff[j])

            action_8d = calculate_thruster_commands(tau_cmd)

            cpu_t1 = time.process_time()
            wall_t1 = time.perf_counter()

            controller_wall_time = wall_t1 - wall_t0
            controller_cpu_time = cpu_t1 - cpu_t0
            controller_freq = 1.0 / controller_wall_time if controller_wall_time > 0.0 else np.nan

            scenario_wall_time += controller_wall_time
            scenario_cpu_time += controller_cpu_time
            scenario_steps += 1

            global_wall_time += controller_wall_time
            global_cpu_time += controller_cpu_time
            global_steps += 1

            thruster_power = estimate_thruster_power_watts(action_8d)
            thruster_step_energy = thruster_power * dt
            thruster_cum_energy += thruster_step_energy

            obs, _, terminated, truncated, _ = env.step(action_8d)

            dist_error = float(np.linalg.norm(err_pos_world))

            row = [
                "PID", scenario_id, t,
                target[0], target[1], target[2],
                curr_pos[0], curr_pos[1], curr_pos[2], dist_error
            ]
            row += action_8d.tolist()
            row += thruster_power.tolist()
            row += [
                float(np.sum(thruster_power)),
                float(np.sum(thruster_step_energy)),
                float(np.sum(thruster_cum_energy)),
                float(controller_wall_time),
                float(controller_cpu_time),
                float(controller_freq),
                float(scenario_wall_time),
                float(scenario_cpu_time),
            ]
            data.append(row)

            if terminated or truncated:
                break

        if scenario_steps > 0:
            print(
                f"[INFO] Scenario {scenario_id} completed | "
                f"Avg Wall Time: {scenario_wall_time / scenario_steps:.6e} s | "
                f"Avg Frequency: {scenario_steps / scenario_wall_time:.2f} Hz"
            )

    csv_path = os.path.join(run_dir, "data_pid_stationkeeping.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(build_header())
        writer.writerows(data)

    env.close()

    if global_steps > 0:
        print("[SUMMARY]")
        print(f"  Total Steps: {global_steps}")
        print(f"  Overall Avg Wall Time: {global_wall_time / global_steps:.6e} s")
        print(f"  Overall Avg Frequency: {global_steps / global_wall_time:.2f} Hz")

    print(f"[OK] Data successfully saved in: {csv_path}")

if __name__ == "__main__":
    run_pid_stationkeeping()