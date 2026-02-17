import time
import csv
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

# 1. Registro atualizado para 2000 passos
try:
    register(
        id="BlueRov-v0",
        entry_point="bluerov2_gym.envs.bluerov_env:BlueRov",
        max_episode_steps=2000,
    )
except:
    pass

class PIDController:
    def __init__(self, kp, ki, kd, dt):
        # Converte para array numpy para operações vetoriais
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        self.dt = dt
        
        # Estado interno para 6 graus de liberdade
        self.integral = np.zeros(6)
        self.prev_error = np.zeros(6)
        
        # Limites de força dos propulsores (aprox. 50 Newtons)
        self.limit = 50.0

    def step(self, error):
        # Integração
        self.integral += error * self.dt
        
        # Derivada (d(erro)/dt)
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        
        # Cálculo do PID: u = Kp*e + Ki*int + Kd*der
        u = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        # Clip para respeitar a física dos motores
        return np.clip(u, -self.limit, self.limit)

def run_pid():
    print("[INFO] Iniciando PID Clássico 6-DoF...")
    env = gym.make("BlueRov-v0", render_mode="human")
    
    # Setpoint: [x, y, z, roll, pitch, yaw] -> Tudo zero
    setpoint = np.zeros(6)
    
    # --- TUNING DO PID (Valores estimados para física realista) ---
    # Ordem: [X, Y, Z, Roll, Pitch, Yaw]
    # Notas: 
    # - X/Y precisam de força moderada para vencer arrasto.
    # - Z precisa de força alta para profundidade.
    # - Roll/Pitch tem estabilidade passiva, ganho baixo apenas para amortecer.
    # - Yaw precisa de torque firme para manter proa.
    
    kp_gains = [40.0, 40.0, 50.0, 20.0, 20.0, 30.0]
    ki_gains = [1.5,  1.5,  2.0,  0.1,  0.1,  0.5]
    kd_gains = [15.0, 15.0, 20.0, 5.0,  5.0,  10.0]

    pid = PIDController(kp=kp_gains, ki=ki_gains, kd=kd_gains, dt=0.1)
    
    obs, _ = env.reset()
    data = []
    
    # Loop de 2000 passos (igual PPO/MPC)
    for i in range(2000):
        # 1. Extração do Estado 6-DoF
        # Usamos .item() para garantir escalares
        current_state = np.array([
            obs["x"].item(), obs["y"].item(), obs["z"].item(),
            obs["roll"].item(), obs["pitch"].item(), obs["yaw"].item()
        ])
        
        # 2. Cálculo do Erro
        error = setpoint - current_state
        
        # Para CSV, salvamos apenas erro de posição euclidiano
        pos_error_norm = np.linalg.norm(error[:3])
        data.append([round(i * 0.1, 2), pos_error_norm])
        
        # 3. Cálculo da Ação
        action = pid.step(error)
        
        # 4. Aplicação no Ambiente
        obs, _, terminated, truncated, _ = env.step(action)
        env.unwrapped.step_sim()
        
        # Feedback visual no console a cada 50 passos
        if i % 50 == 0:
            print(f"T={i*0.1:.1f}s | Erro: {pos_error_norm:.3f}m | Ação X: {action[0]:.1f}N")

        if terminated:
            print("[ALERTA] Robô instável (Terminated).")
            break

    with open("data_pid.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "error"])
        writer.writerows(data)
    
    env.close()
    print("[INFO] Simulação PID finalizada. Arquivo data_pid.csv gerado.")

if __name__ == "__main__":
    run_pid()