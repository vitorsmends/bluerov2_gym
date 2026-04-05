import csv
import time
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from scipy.optimize import minimize

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

    # custo computacional
    header += [
        "controller_wall_time_s",
        "controller_cpu_time_s",
        "controller_frequency_hz",
        "controller_cum_wall_time_s",
        "controller_cum_cpu_time_s",
        "optimizer_success",
        "optimizer_iterations",
        "optimizer_final_cost",
    ]
    return header


# ==========================================
# 4. MPC
# ==========================================
class MPCController:
    def __init__(self, target, dt=0.1, N=10):
        self.dt = dt
        self.N = N
        self.target = target

        self.M_diag = np.array([17.0, 24.2, 26.0, 0.28, 0.28, 0.28])
        self.D_lin = np.array([4.0, 6.0, 5.0, 0.07, 0.07, 0.07])
        self.D_quad = np.array([10.0, 10.0, 10.0, 0.1, 0.1, 0.1])

        self.Q = np.diag([150.0, 150.0, 200.0, 10.0, 10.0, 100.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
        self.R = np.eye(6) * 0.1
        self.u_max = 15.0

        self.last_success = np.nan
        self.last_iterations = np.nan
        self.last_cost = np.nan

    def get_reference_state(self):
        return np.array([
            self.target[0], self.target[1], self.target[2],
            0.0, 0.0, YAW_TARGET,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ], dtype=float)

    def predict_next_state(self, state, action):
        eta = state[0:6]
        nu = state[6:12]

        drag = (self.D_lin * nu) + (self.D_quad * nu * np.abs(nu))
        acc = (action - drag) / self.M_diag
        nu_next = nu + acc * self.dt

        psi = eta[5]
        c_psi, s_psi = np.cos(psi), np.sin(psi)

        dx = nu_next[0] * c_psi - nu_next[1] * s_psi
        dy = nu_next[0] * s_psi + nu_next[1] * c_psi
        dz = nu_next[2]
        d_ang = nu_next[3:6]

        eta_next = eta + np.concatenate(([dx, dy, dz], d_ang)) * self.dt
        return np.concatenate((eta_next, nu_next))

    def cost_function(self, u_flat, current_state):
        u_sequence = u_flat.reshape((self.N, 6))
        cost = 0.0
        state = current_state.copy()
        ref_state = self.get_reference_state()

        for i in range(self.N):
            state = self.predict_next_state(state, u_sequence[i])

            error = state - ref_state
            error[5] = (error[5] + np.pi) % (2 * np.pi) - np.pi

            cost += error.T @ self.Q @ error
            cost += u_sequence[i].T @ self.R @ u_sequence[i]

            if i > 0:
                cost += np.sum((u_sequence[i] - u_sequence[i - 1]) ** 2) * 0.5

        return cost

    def get_action(self, current_state):
        u0 = np.zeros(self.N * 6)
        bounds = [(-self.u_max, self.u_max)] * (self.N * 6)

        res = minimize(
            self.cost_function,
            u0,
            args=(current_state,),
            method="SLSQP",
            bounds=bounds,
            options={"ftol": 1e-2, "maxiter": 5, "disp": False},
        )

        self.last_success = int(bool(res.success))
        self.last_iterations = getattr(res, "nit", np.nan)
        self.last_cost = getattr(res, "fun", np.nan)

        return res.x[:6]


def run_mpc_stationkeeping():
    print("[INFO] Iniciando MPC Stationkeeping...")
    env = gym.make("BlueRov-v0", render_mode=None)

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

        print(f"[INFO] Cenário {scenario_id} | target={target}")

        obs, _ = env.reset()
        obs = set_env_state(env, init, yaw=YAW_TARGET)

        mpc = MPCController(target=target, dt=dt, N=10)
        thruster_cum_energy = np.zeros(6, dtype=float)

        scenario_wall_time = 0.0
        scenario_cpu_time = 0.0
        scenario_steps = 0

        for i in range(steps):
            t = i * dt

            state = np.array([
                obs["x"].item(), obs["y"].item(), obs["z"].item(),
                obs["roll"].item(), obs["pitch"].item(), obs["yaw"].item(),
                obs["u"].item(), obs["v"].item(), obs["w"].item(),
                obs["p"].item(), obs["q"].item(), obs["r"].item()
            ], dtype=float)

            if np.any(np.isnan(state)) or np.linalg.norm(state[:3]) > 20.0:
                print(f"[FALHA] Instabilidade no cenário {scenario_id}, passo {i}")
                break

            # ===== medição do custo computacional do controlador =====
            wall_t0 = time.perf_counter()
            cpu_t0 = time.process_time()

            action = np.asarray(mpc.get_action(state), dtype=float).reshape(-1)

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
            # =========================================================

            thruster_forces = estimate_thruster_forces_from_action(action)
            thruster_power = estimate_thruster_power_watts(thruster_forces)
            thruster_step_energy = thruster_power * dt
            thruster_cum_energy += thruster_step_energy

            obs, _, terminated, truncated, _ = env.step(action)

            dist_error = float(np.linalg.norm(state[:3] - target))

            row = [
                "MPC", scenario_id, t,
                target[0], target[1], target[2],
                state[0], state[1], state[2], dist_error
            ]
            row += action.tolist()
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
                float(mpc.last_success),
                float(mpc.last_iterations),
                float(mpc.last_cost),
            ]
            data.append(row)

            if terminated or truncated:
                break

        if scenario_steps > 0:
            print(
                f"[MPC][Cenário {scenario_id}] "
                f"wall médio = {scenario_wall_time / scenario_steps:.6e} s | "
                f"cpu médio = {scenario_cpu_time / scenario_steps:.6e} s | "
                f"freq média = {scenario_steps / scenario_wall_time:.2f} Hz"
            )

    with open("data_mpc_stationkeeping.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(build_header())
        writer.writerows(data)

    env.close()

    if global_steps > 0:
        print("[RESUMO MPC]")
        print(f"  Steps totais: {global_steps}")
        print(f"  Wall time médio do controlador: {global_wall_time / global_steps:.6e} s")
        print(f"  CPU time médio do controlador : {global_cpu_time / global_steps:.6e} s")
        print(f"  Frequência média equivalente  : {global_steps / global_wall_time:.2f} Hz")

    print("[OK] data_mpc_stationkeeping.csv gerado.")


if __name__ == "__main__":
    run_mpc_stationkeeping()