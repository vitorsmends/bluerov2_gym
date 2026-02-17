import time
import csv
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

# 1. Registro alinhado (2000 passos)
try:
    register(
        id="BlueRov-v0",
        entry_point="bluerov2_gym.envs.bluerov_env:BlueRov",
        max_episode_steps=2000,
    )
except:
    pass

class SingleAxisCascadedPID:
    def __init__(self, kp_pos, kp_vel, ki_vel, kd_vel, dt, u_sat):
        self.kp_pos = kp_pos  # Ganho P do Loop Externo (Posição -> Vel Alvo)
        self.kp_vel = kp_vel  # Ganho P do Loop Interno (Vel -> Força)
        self.ki_vel = ki_vel  # Ganho I do Loop Interno
        self.kd_vel = kd_vel  # Ganho D do Loop Interno
        self.dt = dt
        self.u_sat = u_sat
        
        self.integral_vel = 0.0
        self.prev_vel_error = 0.0

    def update(self, pos_error, current_vel):
        """
        pos_error: Erro de posição (já no referencial correto)
        current_vel: Velocidade atual medida (u, v, w, p, q, r)
        """
        # --- LOOP EXTERNO (Posição) ---
        # Define a velocidade alvo baseada na distância do objetivo
        # Ex: Se estou longe, quero correr (target_vel alto). Se perto, devagar.
        target_vel = self.kp_pos * pos_error
        
        # Limita a velocidade alvo para o robô não tentar ir rápido demais (ex: max 1.5 m/s)
        target_vel = np.clip(target_vel, -1.5, 1.5)

        # --- LOOP INTERNO (Velocidade) ---
        # Calcula a força necessária para atingir a velocidade alvo
        vel_error = target_vel - current_vel
        
        self.integral_vel += vel_error * self.dt
        derivative_vel = (vel_error - self.prev_vel_error) / self.dt
        self.prev_vel_error = vel_error
        
        # PID de Velocidade
        thrust = (self.kp_vel * vel_error) + (self.ki_vel * self.integral_vel) + (self.kd_vel * derivative_vel)
        
        return np.clip(thrust, -self.u_sat, self.u_sat)

def run_cascaded_pid():
    print("[INFO] Iniciando PID em Cascata 6-DoF...")
    env = gym.make("BlueRov-v0", render_mode="human")
    dt = 0.1
    
    # Setpoint (Origem)
    setpoint_pos = np.zeros(3) # x, y, z
    setpoint_att = np.zeros(3) # roll, pitch, yaw
    
    # Limite de força dos motores (Newtons)
    U_SAT = 50.0 
    
    # Inicializando 6 controladores (X, Y, Z, Roll, Pitch, Yaw)
    # Sintonia (Tuning) Estimada:
    # - Loop Posição (kp_pos): ~1.0 a 2.0 (Valores baixos, pois define m/s)
    # - Loop Velocidade (kp_vel): ~20 a 50 (Valores altos, pois define Newtons contra água)
    
    axes_pids = [
        # Eixo X (Surge) - Precisa vencer arrasto frontal
        SingleAxisCascadedPID(kp_pos=1.0, kp_vel=30.0, ki_vel=2.0, kd_vel=5.0, dt=dt, u_sat=U_SAT),
        # Eixo Y (Sway) - Precisa vencer arrasto lateral
        SingleAxisCascadedPID(kp_pos=1.0, kp_vel=30.0, ki_vel=2.0, kd_vel=5.0, dt=dt, u_sat=U_SAT),
        # Eixo Z (Heave) - Precisa vencer flutuabilidade/gravidade
        SingleAxisCascadedPID(kp_pos=1.5, kp_vel=40.0, ki_vel=5.0, kd_vel=2.0, dt=dt, u_sat=U_SAT),
        # Roll - Estabilidade passiva ajuda, ganhos menores
        SingleAxisCascadedPID(kp_pos=2.0, kp_vel=10.0, ki_vel=0.1, kd_vel=1.0, dt=dt, u_sat=U_SAT),
        # Pitch
        SingleAxisCascadedPID(kp_pos=2.0, kp_vel=10.0, ki_vel=0.1, kd_vel=1.0, dt=dt, u_sat=U_SAT),
        # Yaw - Precisa de autoridade para girar
        SingleAxisCascadedPID(kp_pos=1.5, kp_vel=20.0, ki_vel=0.5, kd_vel=2.0, dt=dt, u_sat=U_SAT),
    ]
    
    obs, _ = env.reset()
    data = []
    
    print("Iniciando simulação (2000 passos)...")
    
    for i in range(2000):
        # 1. Extração do Estado (12 Variáveis)
        # Posição (World Frame)
        curr_pos = np.array([obs["x"].item(), obs["y"].item(), obs["z"].item()])
        curr_att = np.array([obs["roll"].item(), obs["pitch"].item(), obs["yaw"].item()])
        
        # Velocidade (Body Frame) - É isso que o loop interno controla
        curr_vel_lin = np.array([obs["u"].item(), obs["v"].item(), obs["w"].item()])
        curr_vel_ang = np.array([obs["p"].item(), obs["q"].item(), obs["r"].item()])
        
        # 2. Cálculo do Erro no MUNDO
        err_pos_world = setpoint_pos - curr_pos
        err_att_world = setpoint_att - curr_att
        
        # 3. Rotação do Erro de Posição para o CORPO (Body Frame)
        # O robô empurra em "Surge" (X do corpo), não em "North" (X do mundo).
        # Precisamos saber quanto do erro global está na frente do nariz do robô.
        psi = curr_att[2] # Yaw
        c, s = np.cos(psi), np.sin(psi)
        
        # Rotação 2D simples (assumindo Roll/Pitch pequenos para navegação)
        err_x_body =  err_pos_world[0] * c + err_pos_world[1] * s
        err_y_body = -err_pos_world[0] * s + err_pos_world[1] * c
        err_z_body =  err_pos_world[2] # Z alinhado
        
        # Vetor de erros alinhado com os controladores [X, Y, Z, Roll, Pitch, Yaw]
        errors_body = np.concatenate(([err_x_body, err_y_body, err_z_body], err_att_world))
        
        # Vetor de velocidades atuais [u, v, w, p, q, r]
        velocities_body = np.concatenate((curr_vel_lin, curr_vel_ang))
        
        # Log para CSV (Distância Euclidiana Absoluta)
        dist_error = np.linalg.norm(err_pos_world)
        data.append([round(i * dt, 2), dist_error])
        
        # 4. Update dos Controladores
        action = np.zeros(6)
        for j in range(6):
            action[j] = axes_pids[j].update(errors_body[j], velocities_body[j])
            
        # 5. Aplicação no Ambiente
        obs, _, terminated, _, _ = env.step(action)
        env.unwrapped.step_sim()
        
        if i % 50 == 0:
            print(f"T={i*dt:.1f}s | Erro: {dist_error:.3f}m | Surge Cmd: {action[0]:.1f}N")

        if terminated:
            print("[ALERTA] Instabilidade detectada.")
            break

    with open("data_cascaded_pid.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "error"])
        writer.writerows(data)
    env.close()
    print("Arquivo data_cascaded_pid.csv gerado.")

if __name__ == "__main__":
    run_cascaded_pid()