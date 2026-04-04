import time
import csv
import math
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register, registry


# ============================================================
# REGISTRO DO AMBIENTE
# ============================================================
ENV_ID = "BlueRov-v0"

if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="bluerov2_gym.envs.bluerov_env:BlueRov",
        max_episode_steps=5000,
    )


# ============================================================
# UTILITÁRIOS NUMÉRICOS
# ============================================================
def is_finite_array(x):
    """Retorna True se todos os elementos forem finitos."""
    arr = np.asarray(x, dtype=float)
    return np.all(np.isfinite(arr))


def safe_scalar(value, default=0.0):
    """Converte para float e substitui NaN/Inf por default."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)

    if not np.isfinite(v):
        return float(default)
    return v


def safe_obs_to_state(obs):
    """
    Extrai o estado do dicionário de observação com proteção contra NaN/Inf.
    """
    state = {
        "x": safe_scalar(obs["x"].item()),
        "y": safe_scalar(obs["y"].item()),
        "z": safe_scalar(obs["z"].item()),
        "roll": safe_scalar(obs["roll"].item()),
        "pitch": safe_scalar(obs["pitch"].item()),
        "yaw": safe_scalar(obs["yaw"].item()),
        "u": safe_scalar(obs["u"].item()),
        "v": safe_scalar(obs["v"].item()),
        "w": safe_scalar(obs["w"].item()),
        "p": safe_scalar(obs["p"].item()),
        "q": safe_scalar(obs["q"].item()),
        "r": safe_scalar(obs["r"].item()),
    }
    return state


def wrap_angle(angle):
    """
    Normaliza um ângulo para [-pi, pi].
    """
    angle = safe_scalar(angle, default=0.0)
    return math.atan2(math.sin(angle), math.cos(angle))


def world_to_body_xy(vec_xy_world, yaw):
    """
    Rotaciona vetor XY do referencial inercial para o corpo.
    """
    vec_xy_world = np.asarray(vec_xy_world, dtype=float).reshape(2)
    yaw = safe_scalar(yaw, default=0.0)

    if not is_finite_array(vec_xy_world):
        return np.zeros(2, dtype=float)

    c = math.cos(yaw)
    s = math.sin(yaw)

    x_b = vec_xy_world[0] * c + vec_xy_world[1] * s
    y_b = -vec_xy_world[0] * s + vec_xy_world[1] * c
    return np.array([x_b, y_b], dtype=float)


# ============================================================
# GERADOR DE TRAJETÓRIA
# ============================================================
class TrajectoryGenerator:
    def __init__(self):
        self.radius = 1.0
        self.speed = 0.15
        self.z_target = -0.5

    def get_reference(self, t):
        t_s = t * self.speed

        if t < 20.0:
            z_d = (self.z_target / 20.0) * t
        else:
            z_d = self.z_target

        x_d = self.radius * math.sin(t_s)
        y_d = self.radius * math.sin(t_s) * math.cos(t_s)

        vx_d = self.radius * math.cos(t_s) * self.speed
        vy_d = self.radius * (math.cos(t_s) ** 2 - math.sin(t_s) ** 2) * self.speed
        vz_d = 0.0

        yaw_d = math.atan2(vy_d, vx_d)

        pos_d = np.array([x_d, y_d, z_d], dtype=float)
        att_d = np.array([0.0, 0.0, yaw_d], dtype=float)
        vel_d = np.array([vx_d, vy_d, vz_d], dtype=float)
        omega_d = np.array([0.0, 0.0, 0.0], dtype=float)

        return pos_d, att_d, vel_d, omega_d


# ============================================================
# ESTIMATIVA DE ENERGIA DOS ATUADORES
# ============================================================
T200_MAX_THRUST_N = 50.0
T200_MAX_POWER_W = 350.0

BLUEROV2_LENGTH_M = 0.457
BLUEROV2_WIDTH_M = 0.338
HALF_LENGTH = BLUEROV2_LENGTH_M / 2.0
HALF_WIDTH = BLUEROV2_WIDTH_M / 2.0
C45 = 1.0 / np.sqrt(2.0)
YAW_ARM = C45 * (HALF_LENGTH + HALF_WIDTH)

B_ALLOC = np.array([
    [C45,  C45,  C45,  C45, 0.0, 0.0],
    [-C45, C45,  C45, -C45, 0.0, 0.0],
    [0.0,  0.0,  0.0,  0.0, 1.0, 1.0],
    [-YAW_ARM, YAW_ARM, -YAW_ARM, YAW_ARM, 0.0, 0.0],
], dtype=float)

B_ALLOC_PINV = np.linalg.pinv(B_ALLOC)


def estimate_thruster_forces_from_action(action_6d):
    action_6d = np.asarray(action_6d, dtype=float).reshape(-1)

    if action_6d.size != 6 or not is_finite_array(action_6d):
        return np.zeros(6, dtype=float)

    surge, sway, heave, roll, pitch, yaw = action_6d
    tau_actuated = np.array([surge, sway, heave, yaw], dtype=float)
    thruster_forces = B_ALLOC_PINV @ tau_actuated

    if not is_finite_array(thruster_forces):
        return np.zeros(6, dtype=float)

    return thruster_forces


def estimate_thruster_power_watts(thruster_forces):
    thruster_forces = np.asarray(thruster_forces, dtype=float)

    if not is_finite_array(thruster_forces):
        return np.zeros(6, dtype=float)

    abs_force = np.abs(thruster_forces)
    force_ratio = np.clip(abs_force / T200_MAX_THRUST_N, 0.0, 1.0)
    power = T200_MAX_POWER_W * (force_ratio ** 1.5)

    if not is_finite_array(power):
        return np.zeros(6, dtype=float)

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


# ============================================================
# PARÂMETROS DO SMC
# ============================================================
LAMBDA = [0.8, 0.8, 1.2, 0.0, 0.0, 0.8]
EPS = [0.5, 0.5, 0.3, 1.0, 1.0, 0.3]
K_INIT = [2.0, 2.0, 4.0, 0.0, 0.0, 2.0]
K_BAR = [0.4, 0.4, 0.8, 0.0, 0.0, 0.4]
MU = [0.05, 0.05, 0.05, 1.0, 1.0, 0.03]
ALPHA = [0.3, 0.3, 0.3, 0.0, 0.0, 0.1]


class SlidingModeController:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.lam = np.array(LAMBDA, dtype=float)
        self.eps = np.array(EPS, dtype=float)
        self.alpha = np.array(ALPHA, dtype=float)
        self.kbar = np.array(K_BAR, dtype=float)
        self.mu = np.array(MU, dtype=float)
        self.K = np.array(K_INIT, dtype=float)

        self.u_sat = 15.0

        # integral só para heave (z)
        self.int_z = 0.0
        self.ki_z = 2.0
        self.int_z_limit = 1.5

    def step(self, pos_error_body, current_vel_body, ref_vel_body=None):
        if ref_vel_body is None:
            ref_vel_body = np.zeros(6, dtype=float)

        vel_error = ref_vel_body - current_vel_body
        s_surface = vel_error + self.lam * pos_error_body

        wrench = np.zeros(6, dtype=float)

        for i in range(6):
            if self.kbar[i] == 0.0:
                wrench[i] = 0.0
                continue

            adaptation_rate = self.kbar[i] * np.sign(np.abs(s_surface[i]) - self.mu[i])
            self.K[i] += self.dt * adaptation_rate
            self.K[i] = np.clip(self.K[i], self.alpha[i], 12.0)

            if np.abs(s_surface[i]) > self.eps[i]:
                wrench[i] = self.K[i] * np.sign(s_surface[i])
            else:
                wrench[i] = self.K[i] * (s_surface[i] / self.eps[i])

        # termo integral no eixo z
        self.int_z += pos_error_body[2] * self.dt
        self.int_z = np.clip(self.int_z, -self.int_z_limit, self.int_z_limit)
        wrench[2] += self.ki_z * self.int_z

        return np.clip(wrench, -self.u_sat, self.u_sat)


# ============================================================
# EXECUÇÃO
# ============================================================
def run_smc():
    print("[INFO] Iniciando SMC com estimativa de energia...")

    env = gym.make(ENV_ID, render_mode=None)

    traj = TrajectoryGenerator()
    dt = 0.1
    steps = 800
    smc = SlidingModeController(dt=dt)

    obs, _ = env.reset()
    state = safe_obs_to_state(obs)

    data = []
    thruster_cum_energy = np.zeros(6, dtype=float)

    start_time = time.time()

    for i in range(steps):
        t = i * dt

        pos_d, att_d, vel_d_world, omega_d = traj.get_reference(t)

        curr_pos = np.array([state["x"], state["y"], state["z"]], dtype=float)
        curr_att = np.array([state["roll"], state["pitch"], state["yaw"]], dtype=float)
        curr_vel_lin = np.array([state["u"], state["v"], state["w"]], dtype=float)
        curr_vel_ang = np.array([state["p"], state["q"], state["r"]], dtype=float)

        if (
            not is_finite_array(curr_pos)
            or not is_finite_array(curr_att)
            or not is_finite_array(curr_vel_lin)
            or not is_finite_array(curr_vel_ang)
        ):
            print("[ERRO] Estado inválido detectado (NaN/Inf).")
            break

        if np.linalg.norm(curr_pos) > 20.0:
            print("[ERRO] Veículo saiu da região segura.")
            break

        err_pos_world = pos_d - curr_pos
        yaw_error = wrap_angle(att_d[2] - curr_att[2])

        psi = curr_att[2]
        err_xy_body = world_to_body_xy(err_pos_world[:2], psi)
        vel_ref_xy_body = world_to_body_xy(vel_d_world[:2], psi)

        pos_error_body = np.array([
            err_xy_body[0],
            err_xy_body[1],
            err_pos_world[2],
            0.0,
            0.0,
            yaw_error,
        ], dtype=float)

        current_vel_body = np.concatenate((curr_vel_lin, curr_vel_ang))

        ref_vel_body = np.array([
            vel_ref_xy_body[0],
            vel_ref_xy_body[1],
            vel_d_world[2],
            0.0,
            0.0,
            omega_d[2],
        ], dtype=float)

        if (
            not is_finite_array(pos_error_body)
            or not is_finite_array(current_vel_body)
            or not is_finite_array(ref_vel_body)
        ):
            print("[ERRO] Sinais de controle inválidos antes do SMC.")
            break

        action = smc.step(
            pos_error_body=pos_error_body,
            current_vel_body=current_vel_body,
            ref_vel_body=ref_vel_body,
        )

        if not is_finite_array(action):
            print("[ERRO] Ação inválida gerada pelo controlador.")
            break

        thruster_forces = estimate_thruster_forces_from_action(action)
        thruster_power = estimate_thruster_power_watts(thruster_forces)
        thruster_step_energy = thruster_power * dt
        thruster_cum_energy += thruster_step_energy

        obs, _, terminated, truncated, _ = env.step(action)
        state = safe_obs_to_state(obs)

        dist_error = float(np.linalg.norm(err_pos_world))

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

        if i % 50 == 0:
            print(
                f"Progresso: {i / steps * 100:.0f}% | "
                f"Erro: {dist_error:.2f}m | "
                f"K_x: {smc.K[0]:.2f}"
            )

        if terminated or truncated:
            print("[INFO] Episódio encerrado pelo ambiente.")
            break

    total_time = time.time() - start_time
    print(f"Simulação concluída em {total_time:.2f} segundos.")

    with open("data_smc_traj.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(build_energy_header())
        writer.writerows(data)

    env.close()
    print("Sucesso! Arquivo 'data_smc_traj.csv' gerado.")


if __name__ == "__main__":
    run_smc()