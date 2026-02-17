import gymnasium as gym
import numpy as np
import math
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
import bluerov2_gym.envs.bluerov_env as original_env

# 1. REGISTRO DO AMBIENTE BASE
try:
    register(
        id="BlueRov-v0",
        entry_point="bluerov2_gym.envs.bluerov_env:BlueRov",
        max_episode_steps=2000,
    )
except:
    pass

# ==========================================
# 2. GERADOR DE TRAJETÓRIA (O MESTRE)
# ==========================================
class TrajectoryGenerator:
    def __init__(self):
        self.radius = 1.0   
        self.speed = 0.15   # Velocidade de treino
        self.z_target = -0.5 

    def get_state_at_time(self, t):
        t_s = t * self.speed
        
        # Posição (Figura 8)
        x = self.radius * math.sin(t_s)
        y = self.radius * math.sin(t_s) * math.cos(t_s)
        
        # Z (Rampa suave depois mantém)
        if t < 10.0:
            z = (self.z_target / 10.0) * t
        else:
            z = self.z_target

        # Velocidade (Derivada para Feedforward)
        vx = self.radius * math.cos(t_s) * self.speed
        vy = self.radius * (math.cos(t_s)**2 - math.sin(t_s)**2) * self.speed
        vz = 0.0

        # Yaw Desejado (Olhando para a frente)
        yaw = math.atan2(vy, vx)

        return np.array([x, y, z]), np.array([0, 0, yaw]), np.array([vx, vy, vz])

# ==========================================
# 3. AMBIENTE PERSONALIZADO (HERANÇA)
# ==========================================
# Aqui a mágica acontece. Estendemos a classe original BlueRov
class TrajectoryTrackingEnv(original_env.BlueRov):
    def __init__(self):
        super().__init__(render_mode=None) # Chama o init original sem render
        self.traj = TrajectoryGenerator()
        self.current_t = 0.0
        self.dt = 0.1 # 10Hz para treino (mais rápido)

    def reset(self, seed=None, options=None):
        # RESET ALEATÓRIO: Inicia em qualquer ponto da trajetória
        # Isso ensina o robô a se recuperar de qualquer lugar
        self.current_t = np.random.uniform(0, 50.0) 
        
        # Pega onde ele deveria estar nesse tempo
        target_pos, target_att, _ = self.traj.get_state_at_time(self.current_t)
        
        # Coloca o robô fisicamente lá (com um pouco de ruído/erro inicial)
        noise_pos = np.random.uniform(-0.2, 0.2, 3)
        initial_pos = target_pos + noise_pos
        
        # Reseta o simulador físico para essa posição
        self.state = {
            'x': initial_pos[0], 'y': initial_pos[1], 'z': initial_pos[2],
            'roll': 0.0, 'pitch': 0.0, 'yaw': target_att[2],
            'u': 0.0, 'v': 0.0, 'w': 0.0, 
            'p': 0.0, 'q': 0.0, 'r': 0.0
        }
        
        return self._get_obs(), {}

    def step(self, action):
        # 1. Avança o tempo
        self.current_t += self.dt
        
        # 2. Executa a ação no simulador original
        obs, _, terminated, truncated, info = super().step(action)
        
        # 3. Pega o Alvo Atual
        tgt_pos, tgt_att, tgt_vel = self.traj.get_state_at_time(self.current_t)
        
        # 4. CALCULA O ERRO REAL (Observação Virtual)
        # O PPO precisa ver a DIFERENÇA, não a posição absoluta.
        # "Onde estou" - "Onde deveria estar"
        curr_pos = np.array([obs['x'][0], obs['y'][0], obs['z'][0]])
        curr_vel = np.array([obs['u'][0], obs['v'][0], obs['w'][0]])
        
        error_pos = curr_pos - tgt_pos
        error_vel = curr_vel - tgt_vel # CRUCIAL: Ensina a acompanhar a velocidade
        
        # Rotação do erro para o referencial do corpo (Facilita muito o aprendizado)
        psi = obs['yaw'][0]
        c, s = np.cos(psi), np.sin(psi)
        
        err_x_body =  error_pos[0]*c + error_pos[1]*s
        err_y_body = -error_pos[0]*s + error_pos[1]*c
        err_z_body =  error_pos[2]
        
        # Substitui a observação original pela observação de ERRO
        # O PPO vai "pensar" que o objetivo é zerar esses valores
        obs['x'] = np.array([err_x_body])
        obs['y'] = np.array([err_y_body])
        obs['z'] = np.array([err_z_body])
        obs['u'] = np.array([error_vel[0]]) # Erro de velocidade surge
        # ... mantemos roll/pitch/yaw/p/q/r originais para estabilidade
        
        # 5. FUNÇÃO DE RECOMPENSA (REWARD FUNCTION)
        dist = np.linalg.norm(error_pos)
        vel_err = np.linalg.norm(error_vel)
        act_cost = np.sum(np.square(action)) # Custo de energia
        
        # Recompensa:
        # - Penalidade grande por distância
        # - Penalidade pequena por erro de velocidade
        # - Penalidade pequena por uso excessivo de motor
        # - Bonus constante por estar vivo (evita suicídio)
        reward = 1.0 - (2.0 * dist) - (0.1 * vel_err) - (0.001 * act_cost)
        
        if dist > 3.0: # Se afastar muito, encerra o episódio (falha)
            terminated = True
            reward -= 10.0 # Punição severa
            
        return obs, reward, terminated, truncated, info

# ==========================================
# 4. LOOP DE TREINAMENTO
# ==========================================
def train():
    print("[INFO] Iniciando treinamento PPO para Trajetória...")
    
    # Cria o ambiente vetorizado com normalização
    # Normalização é CRÍTICA para PPO funcionar bem com física
    env = DummyVecEnv([lambda: TrajectoryTrackingEnv()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Modelo PPO
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        tensorboard_log="./ppo_traj_tensorboard/"
    )

    # Callback para salvar checkpoints a cada 50k passos
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./logs/', name_prefix='ppo_traj')

    # Treinar (Recomendo pelo menos 500.000 passos para ficar bom)
    # Para teste rápido, coloque 100.000
    model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)

    # Salvar modelo final e estatísticas de normalização
    model.save("ppo_trajectory_final")
    env.save("vec_normalize.pkl")
    print("Treino concluído! Modelos salvos.")

if __name__ == "__main__":
    train()