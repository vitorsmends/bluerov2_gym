import csv
import time
import numpy as np
import gymnasium as gym
from scipy.optimize import minimize
from gymnasium.envs.registration import register

# 1. Registo alinhado para 2000 passos (evita TimeLimit prematuro)
try:
    register(
        id="BlueRov-v0",
        entry_point="bluerov2_gym.envs.bluerov_env:BlueRov",
        max_episode_steps=2000,
    )
except:
    pass

class MPCController:
    def __init__(self, dt=0.1, N=10):
        self.dt = dt
        self.N = N 
        
        # Setpoint: [x, y, z, roll, pitch, yaw, u, v, w, p, q, r]
        self.setpoint = np.zeros(12) 
        
        # Modelo Interno (Aproximação da Física)
        self.M_diag = np.array([17.0, 24.2, 26.07, 0.28, 0.28, 0.28])
        self.D_lin = np.array([4.03, 6.22, 5.18, 0.07, 0.07, 0.07])
        
        # Matrizes de Custo (Tuning)
        # Prioridade máxima na posição (X, Y, Z) e estabilidade (Roll, Pitch)
        q_diag = [
            100.0, 100.0, 120.0, # Posição
            50.0, 50.0, 30.0,    # Atitude
            1.0, 1.0, 1.0,       # Velocidades lineares
            0.1, 0.1, 0.1        # Velocidades angulares
        ]
        self.Q = np.diag(q_diag)
        
        # Custo de ação baixo para permitir reações fortes às ondas
        self.R = np.eye(6) * 0.01

    def predict_state(self, state, action):
        eta = state[0:6]
        nu = state[6:12]
        
        # Dinâmica simplificada: F = ma (sem Coriolis para rapidez)
        acc = (action - (self.D_lin * nu)) / self.M_diag
        nu_next = nu + acc * self.dt
        
        # Cinemática simplificada
        psi = eta[5]
        c_psi, s_psi = np.cos(psi), np.sin(psi)
        
        dx = nu_next[0] * c_psi - nu_next[1] * s_psi
        dy = nu_next[0] * s_psi + nu_next[1] * c_psi
        dz = nu_next[2]
        d_ang = nu_next[3:6]
        
        eta_next = eta + np.concatenate(([dx, dy, dz], d_ang)) * self.dt
        
        return np.concatenate((eta_next, nu_next))

    def cost_function(self, u_sequence, current_state):
        u_sequence = u_sequence.reshape((self.N, 6))
        cost = 0.0
        state = current_state.copy()

        for i in range(self.N):
            state = self.predict_state(state, u_sequence[i])
            error = state - self.setpoint
            
            # Custo de Estado
            cost += np.sum(error**2 * np.diag(self.Q))
            
            # Custo de Ação
            cost += np.sum(u_sequence[i]**2 * np.diag(self.R))
            
            # Suavização da ação (derivada)
            if i > 0:
                cost += np.sum((u_sequence[i] - u_sequence[i-1])**2) * 0.5
        
        return cost

    def step(self, current_state):
        # Otimização
        u0 = np.zeros(self.N * 6)
        bounds = [(-50.0, 50.0)] * (self.N * 6)
        
        # Reduzi ftol e maxiter ligeiramente para ganhar velocidade sem perder muita precisão
        res = minimize(
            self.cost_function, 
            u0, 
            args=(current_state,), 
            method='SLSQP', 
            bounds=bounds, 
            options={'ftol': 1e-2, 'maxiter': 10, 'disp': False}
        )
        return res.x[:6]

def run_mpc():
    print("[INFO] Iniciando MPC 6-DoF...")
    print("[NOTA] O MPC é computacionalmente pesado. A simulação pode parecer lenta.")
    
    env = gym.make("BlueRov-v0", render_mode="human")
    mpc = MPCController(dt=0.1, N=10) # Horizonte de previsão de 1 segundo
    
    obs, _ = env.reset()
    data = []
    
    # 2. Aumentado para 2000 passos (igual ao PPO)
    total_steps = 2000
    
    for i in range(total_steps):
        # Extração do estado 12D
        state = np.array([
            obs["x"].item(), obs["y"].item(), obs["z"].item(),
            obs["roll"].item(), obs["pitch"].item(), obs["yaw"].item(),
            obs["u"].item(), obs["v"].item(), obs["w"].item(),
            obs["p"].item(), obs["q"].item(), obs["r"].item()
        ])
        
        pos_error_norm = np.linalg.norm(state[:3])
        data.append([round(i * 0.1, 2), pos_error_norm])
        
        # Passo do Controlador
        action = mpc.step(state)
        
        # Passo do Ambiente
        obs, _, terminated, truncated, _ = env.step(action)
        env.unwrapped.step_sim()

        # Feedback de progresso
        if i % 50 == 0:
            pct = (i / total_steps) * 100
            print(f"Progresso: {pct:.1f}% | Erro Pos: {pos_error_norm:.3f}m | Ação Z: {action[2]:.1f}")

        # Se o MPC falhar em estabilizar, o ambiente termina o episódio
        if terminated:
            print(f"[ALERTA] Episódio terminado prematuramente no passo {i}. O ROV capotou ou saiu dos limites.")
            break

    with open("data_mpc.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "error"])
        writer.writerows(data)
    
    env.close()
    print("[INFO] Simulação MPC concluída. Ficheiro 'data_mpc.csv' gerado.")

if __name__ == "__main__":
    run_mpc()