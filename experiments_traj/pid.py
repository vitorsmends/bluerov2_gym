import time
import csv
import math
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

# --- REGISTRO ---
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
        self.speed = 0.1    
        self.z_target = -0.5 

    def get_reference(self, t):
        t_s = t * self.speed
        
        # Z: Rampa suave
        z_d = 0.0
        if t < 20.0:
            z_d = (self.z_target / 20.0) * t
        else:
            z_d = self.z_target

        # XY: Lemniscata (Figura 8)
        x_d = self.radius * math.sin(t_s)
        y_d = self.radius * math.sin(t_s) * math.cos(t_s)

        # Feedforward
        vx_d = self.radius * math.cos(t_s) * self.speed
        vy_d = self.radius * (math.cos(t_s)**2 - math.sin(t_s)**2) * self.speed
        vz_d = 0.0

        yaw_d = math.atan2(vy_d, vx_d)

        return (np.array([x_d, y_d, z_d]), 
                np.array([0.0, 0.0, yaw_d]),
                np.array([vx_d, vy_d, vz_d]),
                np.array([0.0, 0.0, 0.0]))

# --- PID CASCADE (Blindado) ---
class SingleAxisCascadedPID:
    def __init__(self, kp_pos, kp_vel, ki_vel, dt, u_sat):
        self.kp_pos = kp_pos
        self.kp_vel = kp_vel
        self.ki_vel = ki_vel
        self.dt = dt
        self.u_sat = u_sat
        self.integral_vel = 0.0

    def update(self, pos_error, current_vel, ff_vel=0.0):
        # Proteção contra NaN
        if np.isnan(pos_error) or np.isnan(current_vel): return 0.0

        # Loop Externo (Posição)
        target_vel = (self.kp_pos * pos_error) + ff_vel
        target_vel = np.clip(target_vel, -0.5, 0.5) # Limite de velocidade de segurança

        # Loop Interno (Velocidade)
        vel_error = target_vel - current_vel
        
        self.integral_vel += vel_error * self.dt
        self.integral_vel = np.clip(self.integral_vel, -2.0, 2.0) # Anti-windup forte
        
        # Apenas PI no loop interno (Derivativo D removido pois causa ruído em 10Hz)
        u = (self.kp_vel * vel_error) + (self.ki_vel * self.integral_vel)
        
        return np.clip(u, -self.u_sat, self.u_sat)

# --- EXECUÇÃO ---
def run_pid():
    print("[INFO] Iniciando PID Seguro (10Hz)...")
    env = gym.make("BlueRov-v0", render_mode=None)
    
    traj = TrajectoryGenerator()
    
    # IMPORTANTE: dt=0.1 para casar com a física padrão do Gym
    dt = 0.1 
    steps = 800 # 80 segundos
    
    U_MAX = 20.0 
    
    # TUNING PARA 10Hz (Ganhos mais baixos para evitar oscilação)
    # Kp_pos baixo (0.5) para não tentar corrigir posição rápido demais
    # Kp_vel médio (5.0) para ter força suficiente sem explodir
    pid_xy = SingleAxisCascadedPID(kp_pos=0.5, kp_vel=5.0, ki_vel=0.1, dt=dt, u_sat=U_MAX)
    pid_z  = SingleAxisCascadedPID(kp_pos=0.8, kp_vel=8.0, ki_vel=0.2, dt=dt, u_sat=U_MAX)
    pid_att= SingleAxisCascadedPID(kp_pos=1.0, kp_vel=4.0, ki_vel=0.0, dt=dt, u_sat=U_MAX)
    
    axes_pids = [pid_xy, pid_xy, pid_z, pid_att, pid_att, pid_att]

    obs, _ = env.reset()
    data = []
    
    print(f"Processando {steps} passos...")
    
    for i in range(steps):
        t = i * dt
        pos_d, att_d, vel_d, omega_d = traj.get_reference(t)

        curr_pos = np.array([obs["x"].item(), obs["y"].item(), obs["z"].item()])
        curr_att = np.array([obs["roll"].item(), obs["pitch"].item(), obs["yaw"].item()])
        curr_vel = np.array([obs["u"].item(), obs["v"].item(), obs["w"].item(), obs["p"].item(), obs["q"].item(), obs["r"].item()])
        
        # --- TRAVA DE SEGURANÇA (CRÍTICO) ---
        # Se detectar NaN, para imediatamente e salva o que tem.
        if np.any(np.isnan(curr_pos)) or np.any(np.isnan(curr_vel)):
            print(f"[FALHA CRÍTICA] NaN detectado no passo {i}. Parando simulação.")
            break
        
        if np.linalg.norm(curr_pos) > 50.0:
            print(f"[ALERTA] Robô saiu da área (Instabilidade). Parando.")
            break

        # Erros
        err_pos_world = pos_d - curr_pos
        err_att_world = att_d - curr_att
        err_att_world[2] = (err_att_world[2] + np.pi) % (2 * np.pi) - np.pi

        # Rotação
        psi = curr_att[2]
        c, s = np.cos(psi), np.sin(psi)
        
        # Rotação manual 2D
        err_pos_body = np.array([
            err_pos_world[0]*c + err_pos_world[1]*s,
            -err_pos_world[0]*s + err_pos_world[1]*c,
            err_pos_world[2]
        ])
        vel_ref_body = np.array([
            vel_d[0]*c + vel_d[1]*s,
            -vel_d[0]*s + vel_d[1]*c,
            vel_d[2]
        ])

        # Controle
        action = np.zeros(6)
        errors = np.concatenate((err_pos_body, err_att_world))
        ff = np.concatenate((vel_ref_body, omega_d))
        
        for j in range(6):
            action[j] = axes_pids[j].update(errors[j], curr_vel[j], ff[j])

        obs, _, terminated, _, _ = env.step(action)
        
        # LOG
        dist_error = np.linalg.norm(err_pos_world)
        data.append([t, curr_pos[0], curr_pos[1], curr_pos[2], dist_error])
            
        if terminated: break

    # SALVAR CSV
    with open("data_pid_traj.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "x", "y", "z", "error"])
        writer.writerows(data)
    env.close()
    print("PID Concluído. Arquivo 'data_pid_traj.csv' gerado.")

if __name__ == "__main__":
    run_pid()