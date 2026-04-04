import time
import csv
import math
import numpy as np
import gymnasium as gym
from scipy.optimize import minimize
from gymnasium.envs.registration import register

# --- REGISTRO DO AMBIENTE ---
try:
    register(
        id="BlueRov-v0",
        entry_point="bluerov2_gym.envs.bluerov_env:BlueRov",
        max_episode_steps=5000,
    )
except:
    pass

# --- GERADOR DE TRAJETÓRIA ---
class TrajectoryGenerator:
    def __init__(self):
        self.radius = 1.0
        self.speed = 0.15
        self.z_target = -0.5

    def get_reference(self, t):
        t_s = t * self.speed

        z_d = 0.0
        if t < 20.0:
            z_d = (self.z_target / 20.0) * t
        else:
            z_d = self.z_target

        x_d = self.radius * math.sin(t_s)
        y_d = self.radius * math.sin(t_s) * math.cos(t_s)

        vx_d = self.radius * math.cos(t_s) * self.speed
        vy_d = self.radius * (math.cos(t_s)**2 - math.sin(t_s)**2) * self.speed
        vz_d = 0.0

        yaw_d = math.atan2(vy_d, vx_d)

        return np.array([x_d, y_d, z_d, 0.0, 0.0, yaw_d, vx_d, vy_d, vz_d, 0.0, 0.0, 0.0])


# --- ESTIMATIVA DE ENERGIA DOS ATUADORES ---
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


# --- CONTROLADOR MPC ---
class MPCController:
    def __init__(self, traj_gen, dt=0.1, N=10):
        self.dt = dt
        self.N = N
        self.traj_gen = traj_gen

        self.M_diag = np.array([17.0, 24.2, 26.0, 0.28, 0.28, 0.28])
        self.D_lin = np.array([4.0, 6.0, 5.0, 0.07, 0.07, 0.07])
        self.D_quad = np.array([10.0, 10.0, 10.0, 0.1, 0.1, 0.1])

        self.Q = np.diag([150.0, 150.0, 200.0, 10.0, 10.0, 100.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
        self.R = np.eye(6) * 0.1

        self.u_max = 15.0

    def predict_next_state(self, state, action):
        eta = state[0:6]
        nu = state[6:12]

        drag = (self.D_lin * nu) + (self.D_quad * nu * np.abs(nu))
        acc = (action - drag) / self.M_diag
        nu_next = nu + acc * self.dt

        phi, theta, psi = eta[3], eta[4], eta[5]
        c_psi, s_psi = np.cos(psi), np.sin(psi)

        dx = nu_next[0] * c_psi - nu_next[1] * s_psi
        dy = nu_next[0] * s_psi + nu_next[1] * c_psi
        dz = nu_next[2]
        d_ang = nu_next[3:6]

        eta_next = eta + np.concatenate(([dx, dy, dz], d_ang)) * self.dt
        return np.concatenate((eta_next, nu_next))

    def cost_function(self, u_flat, current_state, t_start):
        u_sequence = u_flat.reshape((self.N, 6))
        cost = 0.0
        state = current_state.copy()

        for i in range(self.N):
            state = self.predict_next_state(state, u_sequence[i])
            t_future = t_start + (i + 1) * self.dt
            ref_state = self.traj_gen.get_reference(t_future)

            error = state - ref_state
            error[5] = (error[5] + np.pi) % (2 * np.pi) - np.pi

            cost += error.T @ self.Q @ error
            cost += u_sequence[i].T @ self.R @ u_sequence[i]

            if i > 0:
                cost += np.sum((u_sequence[i] - u_sequence[i - 1])**2) * 0.5

        return cost

    def get_action(self, current_state, t_now):
        u0 = np.zeros(self.N * 6)
        bounds = [(-self.u_max, self.u_max)] * (self.N * 6)

        res = minimize(
            self.cost_function,
            u0,
            args=(current_state, t_now),
            method='SLSQP',
            bounds=bounds,
            options={'ftol': 1e-2, 'maxiter': 5, 'disp': False}
        )
        return res.x[:6]


# --- EXECUÇÃO ---
def run_mpc():
    print("[INFO] Iniciando MPC com estimativa de energia...")

    env = gym.make("BlueRov-v0", render_mode=None)

    traj = TrajectoryGenerator()
    dt = 0.1
    steps = 800

    mpc = MPCController(traj, dt=dt, N=10)

    obs, _ = env.reset()
    data = []
    thruster_cum_energy = np.zeros(6, dtype=float)

    start_time = time.time()

    for i in range(steps):
        t = i * dt

        state = np.array([
            obs["x"].item(), obs["y"].item(), obs["z"].item(),
            obs["roll"].item(), obs["pitch"].item(), obs["yaw"].item(),
            obs["u"].item(), obs["v"].item(), obs["w"].item(),
            obs["p"].item(), obs["q"].item(), obs["r"].item()
        ])

        if np.any(np.isnan(state)) or np.linalg.norm(state[:3]) > 20.0:
            print("[ERRO] Instabilidade detectada.")
            break

        action = np.asarray(mpc.get_action(state, t), dtype=float).reshape(-1)

        thruster_forces = estimate_thruster_forces_from_action(action)
        thruster_power = estimate_thruster_power_watts(thruster_forces)
        thruster_step_energy = thruster_power * dt
        thruster_cum_energy += thruster_step_energy

        obs, _, terminated, _, _ = env.step(action)

        ref_now = traj.get_reference(t)
        dist_error = np.linalg.norm(state[:3] - ref_now[:3])

        data.append(
            build_energy_row(
                t=t,
                curr_pos=state[:3],
                dist_error=dist_error,
                action=action,
                thr_forces=thruster_forces,
                thr_power=thruster_power,
                thr_step_energy=thruster_step_energy,
                thr_cum_energy=thruster_cum_energy.copy(),
            )
        )

        if i % 50 == 0:
            print(f"Progresso: {i/steps*100:.0f}% | Erro: {dist_error:.2f}m")

        if terminated:
            break

    total_time = time.time() - start_time
    print(f"Simulação concluída em {total_time:.2f} segundos.")

    with open("data_mpc_traj.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(build_energy_header())
        writer.writerows(data)

    env.close()
    print("Sucesso! Arquivo 'data_mpc_traj.csv' gerado.")


if __name__ == "__main__":
    run_mpc()