import time
import numpy as np
import math
import gymnasium as gym
import matplotlib.pyplot as plt
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

# --- GERADOR DE TRAJETÓRIA (FIGURA 8) ---
class TrajectoryGenerator:
    def __init__(self):
        self.radius = 1.0   
        self.speed = 0.1    
        self.z_target = -0.5 

    def get_reference(self, t):
        t_s = t * self.speed
        
        # Z: Desce suavemente até -0.5m
        z_d = 0.0
        if t < 20.0:
            z_d = (self.z_target / 20.0) * t
        else:
            z_d = self.z_target

        # XY: Figura 8 (Lemniscata)
        # x = r * sin(t)
        # y = r * sin(t) * cos(t)
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

# --- CONTROLADOR PID (Simplificado) ---
class SingleAxisCascadedPID:
    def __init__(self, kp_pos, kp_vel, ki_vel, dt, u_sat):
        self.kp_pos = kp_pos
        self.kp_vel = kp_vel
        self.ki_vel = ki_vel
        self.dt = dt
        self.u_sat = u_sat
        self.integral_vel = 0.0
        self.prev_vel_error = 0.0

    def update(self, pos_error, current_vel, ff_vel=0.0):
        if np.isnan(pos_error) or np.isnan(current_vel): return 0.0
        
        # Loop Externo
        target_vel = (self.kp_pos * pos_error) + ff_vel
        target_vel = np.clip(target_vel, -0.4, 0.4)

        # Loop Interno
        vel_error = target_vel - current_vel
        self.integral_vel += vel_error * self.dt
        self.integral_vel = np.clip(self.integral_vel, -1.0, 1.0)
        
        # Sem Derivada para estabilidade visual máxima
        u = (self.kp_vel * vel_error) + (self.ki_vel * self.integral_vel)
        return np.clip(u, -self.u_sat, self.u_sat)

# --- EXECUÇÃO E PLOTAGEM ---
def run_and_plot():
    print("[INFO] Rodando simulação para captura de trajetória...")
    
    # Headless para rapidez
    env = gym.make("BlueRov-v0", render_mode=None)
    
    traj = TrajectoryGenerator()
    dt = 0.02 # 50Hz
    steps = 4000 # 80 segundos (tempo suficiente para desenhar o 8 completo)
    
    # Tuning PID Estável
    U_MAX = 10.0
    pid_xy  = SingleAxisCascadedPID(kp_pos=0.8, kp_vel=8.0, ki_vel=0.1, dt=dt, u_sat=U_MAX)
    pid_z   = SingleAxisCascadedPID(kp_pos=1.0, kp_vel=10.0, ki_vel=0.5, dt=dt, u_sat=U_MAX)
    pid_att = SingleAxisCascadedPID(kp_pos=1.5, kp_vel=5.0, ki_vel=0.0, dt=dt, u_sat=U_MAX)
    
    axes_pids = [pid_xy, pid_xy, pid_z, pid_att, pid_att, pid_att]

    obs, _ = env.reset()
    
    # Listas para guardar o histórico
    hist_ref = []
    hist_act = []
    
    for i in range(steps):
        t = i * dt
        pos_d, att_d, vel_d, omega_d = traj.get_reference(t)

        curr_pos = np.array([obs["x"].item(), obs["y"].item(), obs["z"].item()])
        curr_att = np.array([obs["roll"].item(), obs["pitch"].item(), obs["yaw"].item()])
        curr_vel_lin = np.array([obs["u"].item(), obs["v"].item(), obs["w"].item()])
        curr_vel_ang = np.array([obs["p"].item(), obs["q"].item(), obs["r"].item()])
        
        # Guardar dados para o gráfico
        hist_ref.append(pos_d.copy())
        hist_act.append(curr_pos.copy())

        # Controle
        err_pos_world = pos_d - curr_pos
        err_att_world = att_d - curr_att
        err_att_world[2] = (err_att_world[2] + np.pi) % (2 * np.pi) - np.pi

        psi = curr_att[2]
        c, s = np.cos(psi), np.sin(psi)
        
        # Rotação manual
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

        errors = np.concatenate((err_pos_body, err_att_world))
        vels = np.concatenate((curr_vel_lin, curr_vel_ang))
        ff = np.concatenate((vel_ref_body, omega_d))
        
        action = np.zeros(6)
        for j in range(6):
            action[j] = axes_pids[j].update(errors[j], vels[j], ff[j])

        obs, _, terminated, _, _ = env.step(action)
        
        if i % 500 == 0:
            print(f"Progresso: {i/steps*100:.0f}%")
        
        if terminated: break

    env.close()
    
    # --- PLOTAGEM ---
    print("Gerando gráficos...")
    ref = np.array(hist_ref)
    act = np.array(hist_act)
    
    # Criar figura com 2 subplots
    fig = plt.figure(figsize=(12, 5))
    
    # 1. Gráfico 2D (XY - Vista de Topo)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(ref[:, 0], ref[:, 1], 'r--', label='Referência (Desejado)')
    ax1.plot(act[:, 0], act[:, 1], 'b-', linewidth=2, label='Real (PID)')
    ax1.set_title('Vista Superior (Plano XY)')
    ax1.set_xlabel('X (metros)')
    ax1.set_ylabel('Y (metros)')
    ax1.axis('equal') # Importante para não distorcer o círculo/8
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # 2. Gráfico 3D (Trajetória Completa)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot(ref[:, 0], ref[:, 1], ref[:, 2], 'r--', label='Ref')
    ax2.plot(act[:, 0], act[:, 1], act[:, 2], 'b-', label='Real')
    ax2.set_title('Trajetória 3D')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z (Profundidade)')
    ax2.invert_zaxis() # Z positivo para baixo na convenção NED, mas aqui Z é up, então invertemos para parecer subaquático
    
    plt.tight_layout()
    plt.savefig('trajetoria_pid.png')
    print("Gráfico salvo como 'trajetoria_pid.png'")
    plt.show()

if __name__ == "__main__":
    run_and_plot()