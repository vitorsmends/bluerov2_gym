import csv
import math
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

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
# 2. GERADOR DE TRAJETÓRIA
# ==========================================
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

        return (
            np.array([x_d, y_d, z_d], dtype=float),
            np.array([0.0, 0.0, yaw_d], dtype=float),
            np.array([vx_d, vy_d, vz_d], dtype=float),
            np.array([0.0, 0.0, 0.0], dtype=float),
        )


# ==========================================
# 3. PID CASCATA POR EIXO
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


# ==========================================
# 5. ESTIMATIVA DE ENERGIA DOS ATUADORES
# ==========================================
# Aproximação baseada em dados públicos do BlueROV2/T200.
T200_MAX_THRUST_N = 50.0
T200_MAX_POWER_W = 350.0

BLUEROV2_LENGTH_M = 0.457
BLUEROV2_WIDTH_M = 0.338
HALF_LENGTH = BLUEROV2_LENGTH_M / 2.0
HALF_WIDTH = BLUEROV2_WIDTH_M / 2.0
C45 = 1.0 / np.sqrt(2.0)
YAW_ARM = C45 * (HALF_LENGTH + HALF_WIDTH)

# Alocação simplificada para BlueROV2 padrão:
# 4 thrusters horizontais vetorizados + 2 verticais.
# Usa surge, sway, heave, yaw.
B_ALLOC = np.array([
    [ C45,  C45,  C45,  C45, 0.0, 0.0],   # surge
    [-C45,  C45,  C45, -C45, 0.0, 0.0],   # sway
    [ 0.0,  0.0,  0.0,  0.0, 1.0, 1.0],   # heave
    [-YAW_ARM, YAW_ARM, -YAW_ARM, YAW_ARM, 0.0, 0.0],  # yaw
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

    # Lei aproximada para hélice: potência cresce mais rápido que a força.
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
# 6. EXECUÇÃO
# ==========================================
def run_pid():
    print("[INFO] Iniciando PID (10 Hz) com estimativa de energia...")

    env = gym.make("BlueRov-v0", render_mode=None)
    traj = TrajectoryGenerator()

    dt = 0.1
    steps = 800
    u_max = 20.0

    pid_x = SingleAxisCascadedPID(kp_pos=0.5, kp_vel=5.0, ki_vel=0.1, dt=dt, u_sat=u_max)
    pid_y = SingleAxisCascadedPID(kp_pos=0.5, kp_vel=5.0, ki_vel=0.1, dt=dt, u_sat=u_max)
    pid_z = SingleAxisCascadedPID(kp_pos=0.8, kp_vel=8.0, ki_vel=0.2, dt=dt, u_sat=u_max)

    pid_roll = SingleAxisCascadedPID(kp_pos=1.0, kp_vel=4.0, ki_vel=0.0, dt=dt, u_sat=u_max)
    pid_pitch = SingleAxisCascadedPID(kp_pos=1.0, kp_vel=4.0, ki_vel=0.0, dt=dt, u_sat=u_max)
    pid_yaw = SingleAxisCascadedPID(kp_pos=1.0, kp_vel=4.0, ki_vel=0.0, dt=dt, u_sat=u_max)

    controllers = [pid_x, pid_y, pid_z, pid_roll, pid_pitch, pid_yaw]

    obs, _ = env.reset()
    data = []
    thruster_cum_energy = np.zeros(6, dtype=float)

    print(f"[INFO] Processando {steps} passos...")

    for i in range(steps):
        t = i * dt

        pos_d, att_d, vel_d_world, omega_d = traj.get_reference(t)

        curr_pos = np.array([obs["x"].item(), obs["y"].item(), obs["z"].item()], dtype=float)
        curr_att = np.array([obs["roll"].item(), obs["pitch"].item(), obs["yaw"].item()], dtype=float)
        curr_vel = np.array([
            obs["u"].item(),
            obs["v"].item(),
            obs["w"].item(),
            obs["p"].item(),
            obs["q"].item(),
            obs["r"].item(),
        ], dtype=float)

        if np.any(np.isnan(curr_pos)) or np.any(np.isnan(curr_vel)):
            print(f"[FALHA] NaN detectado no passo {i}. Encerrando.")
            break

        if np.linalg.norm(curr_pos) > 50.0:
            print(f"[FALHA] Veículo saiu da região segura no passo {i}. Encerrando.")
            break

        err_pos_world = pos_d - curr_pos
        err_att = att_d - curr_att
        err_att[2] = wrap_angle(err_att[2])

        yaw = curr_att[2]
        err_xy_body = world_to_body_xy(err_pos_world[:2], yaw)
        vel_xy_body = world_to_body_xy(vel_d_world[:2], yaw)

        err_pos_body = np.array([err_xy_body[0], err_xy_body[1], err_pos_world[2]], dtype=float)
        vel_ref_body = np.array([vel_xy_body[0], vel_xy_body[1], vel_d_world[2]], dtype=float)

        errors = np.concatenate((err_pos_body, err_att))
        ff = np.concatenate((vel_ref_body, omega_d))

        action = np.zeros(6, dtype=float)
        for j in range(6):
            action[j] = controllers[j].update(
                pos_error=errors[j],
                current_vel=curr_vel[j],
                ff_vel=ff[j],
            )

        thruster_forces = estimate_thruster_forces_from_action(action)
        thruster_power = estimate_thruster_power_watts(thruster_forces)
        thruster_step_energy = thruster_power * dt
        thruster_cum_energy += thruster_step_energy

        obs, _, terminated, truncated, _ = env.step(action)

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

        if terminated or truncated:
            print(f"[INFO] Episódio encerrado no passo {i}.")
            break

    with open("data_pid_traj.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(build_energy_header())
        writer.writerows(data)

    env.close()
    print("[OK] PID concluído. Arquivo 'data_pid_traj.csv' gerado.")


if __name__ == "__main__":
    run_pid()