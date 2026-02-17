import csv
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

# 1. Registro alinhado com 2000 passos
try:
    register(
        id="BlueRov-v0",
        entry_point="bluerov2_gym.envs.bluerov_env:BlueRov",
        max_episode_steps=2000,
    )
except:
    pass

# --- Parâmetros do SMC (Ajustados para 6-DoF e Física Real) ---
# Lambd: Define a "largura de banda" ou quão rápido o erro de posição deve decair.
# Valores mais altos = resposta mais rápida, mas risco de overshoot.
LAMBDA = [1.5, 1.5, 2.0, 1.0, 1.0, 1.0]

# Eps (Epsilon): Largura da camada limite (Boundary Layer) para evitar Chattering.
# Define a suavidade da transição entre +K e -K.
EPS = [0.2, 0.2, 0.1, 0.1, 0.1, 0.1]

# K_init: Ganho inicial (Força). Começa baixo e a adaptação sobe se necessário.
# Z precisa de mais força inicial devido à flutuabilidade.
K_INIT = [10.0, 10.0, 20.0, 5.0, 5.0, 5.0]

# Kbar: Taxa de adaptação do ganho K.
K_BAR = [5.0, 5.0, 10.0, 2.0, 2.0, 2.0]

# Mu: Tolerância da superfície deslizante para parar de aumentar o ganho.
MU = [0.05, 0.05, 0.05, 0.02, 0.02, 0.02]

# Alpha: Valor mínimo para o ganho K.
ALPHA = [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]

class SlidingModeController:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.lam = np.array(LAMBDA)
        self.eps = np.array(EPS)
        self.alpha = np.array(ALPHA)
        self.kbar = np.array(K_BAR)
        self.mu = np.array(MU)
        self.K = np.array(K_INIT)
        
        # Limite físico dos propulsores (Newtons)
        self.u_sat = 50.0

    def step(self, pos_error_body, current_vel_body):
        """
        Calcula a força de controle baseada no erro no referencial do CORPO.
        pos_error_body: Erro de Posição rotacionado para o corpo.
        current_vel_body: Velocidade atual (u, v, w...).
        """
        # Setpoint de velocidade é 0 (Station Keeping)
        # Erro de vel = (0 - vel_atual)
        vel_error = -current_vel_body
        
        # 1. Definição da Superfície de Deslizamento (Sliding Surface)
        # s = erro_vel + lambda * erro_pos
        s_surface = vel_error + self.lam * pos_error_body
        
        wrench = np.zeros(6)

        for i in range(6):
            # 2. Lei de Adaptação do Ganho K (SMC Adaptativo)
            # Se |s| > mu, aumenta o ganho K para vencer a perturbação.
            # Se |s| < mu, o ganho pode decair ou manter-se.
            adaptation_rate = self.kbar[i] * np.sign(np.abs(s_surface[i]) - self.mu[i])
            
            # Atualiza K
            self.K[i] += self.dt * adaptation_rate
            
            # Garante que K nunca seja menor que Alpha (ganho mínimo)
            if self.K[i] < self.alpha[i]:
                self.K[i] = self.alpha[i]
            
            # Opcional: Limitar o ganho máximo para segurança
            if self.K[i] > 60.0: 
                self.K[i] = 60.0

            # 3. Lei de Controle com Saturação Suave (Sat Function)
            # Substitui a função sign() pura para reduzir chattering (tremores)
            if np.abs(s_surface[i]) > self.eps[i]:
                # Fora da camada limite: Aplica força total K
                wrench[i] = self.K[i] * np.sign(s_surface[i])
            else:
                # Dentro da camada limite: Comportamento linear (como um ganho P alto)
                wrench[i] = self.K[i] * (s_surface[i] / self.eps[i])

        # Clip final para respeitar a física do motor
        return np.clip(wrench, -self.u_sat, self.u_sat)

def run_smc():
    print("[INFO] Iniciando SMC Adaptativo 6-DoF...")
    env = gym.make("BlueRov-v0", render_mode="human")
    
    # Setpoints (Tudo zero = Origem)
    setpoint_pos = np.zeros(3)
    setpoint_att = np.zeros(3)
    
    smc = SlidingModeController(dt=0.1)
    
    obs, _ = env.reset()
    data = []
    
    print("Iniciando simulação (2000 passos)...")
    
    for i in range(2000):
        # 1. Extração do Estado Completo
        curr_pos = np.array([obs["x"].item(), obs["y"].item(), obs["z"].item()])
        curr_att = np.array([obs["roll"].item(), obs["pitch"].item(), obs["yaw"].item()])
        
        # Velocidades no referencial do CORPO (Body Frame)
        curr_vel_lin = np.array([obs["u"].item(), obs["v"].item(), obs["w"].item()])
        curr_vel_ang = np.array([obs["p"].item(), obs["q"].item(), obs["r"].item()])
        
        # 2. Cálculo do Erro no MUNDO
        err_pos_world = setpoint_pos - curr_pos
        err_att_world = setpoint_att - curr_att
        
        # 3. Rotação do Erro para o CORPO
        psi = curr_att[2] # Yaw
        c, s = np.cos(psi), np.sin(psi)
        
        err_x_body =  err_pos_world[0] * c + err_pos_world[1] * s
        err_y_body = -err_pos_world[0] * s + err_pos_world[1] * c
        err_z_body =  err_pos_world[2]
        
        # Vetores alinhados ao corpo (Posição + Atitude)
        pos_error_body = np.concatenate(([err_x_body, err_y_body, err_z_body], err_att_world))
        current_vel_body = np.concatenate((curr_vel_lin, curr_vel_ang))
        
        # Log (Distância euclidiana absoluta)
        dist_error = np.linalg.norm(err_pos_world)
        data.append([round(i * 0.1, 2), dist_error])
        
        # 4. Controle
        action = smc.step(pos_error_body, current_vel_body)
        
        obs, _, terminated, _, _ = env.step(action)
        env.unwrapped.step_sim()
        
        if i % 50 == 0:
            # Mostra o ganho K adaptativo do eixo X para ver ele evoluindo
            print(f"T={i*0.1:.1f}s | Erro: {dist_error:.3f}m | K_x Adaptado: {smc.K[0]:.1f}")

        if terminated:
            print("[ALERTA] Terminated.")
            break

    with open("data_smc.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "error"])
        writer.writerows(data)
    env.close()
    print("Arquivo data_smc.csv gerado.")

if __name__ == "__main__":
    run_smc()