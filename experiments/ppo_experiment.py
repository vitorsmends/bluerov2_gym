import time
import csv
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 1. Atualizei o limite para 2000 passos para permitir uma simulação mais longa
try:
    register(
        id="BlueRov-v0",
        entry_point="bluerov2_gym.envs.bluerov_env:BlueRov", 
        max_episode_steps=2000, 
    )
except:
    pass

def run_ppo():
    print("Iniciando ambiente...")
    env = gym.make("BlueRov-v0", render_mode="human")
    
    print("Carregando modelo e normalização...")
    model = PPO.load("bluerov_ppo")
    
    # Carrega as estatísticas de normalização
    venv = DummyVecEnv([lambda: gym.make("BlueRov-v0")])
    venv = VecNormalize.load("bluerov_vec_normalize.pkl", venv)
    venv.training = False
    venv.norm_reward = False
    
    # Setpoint (Alvo na origem)
    target_pos = np.array([0.0, 0.0, 0.0])
    
    obs, _ = env.reset()
    data = []
    
    print("Iniciando loop de avaliação prolongado (2000 passos)...")
    
    # 2. Aumentei o loop para 2000 iterações (aprox. 3.3 minutos de tempo simulado)
    for i in range(2000):
        # Extração das coordenadas
        current_pos = np.array([
            obs["x"].item(),
            obs["y"].item(),
            obs["z"].item()
        ])
        
        # Cálculo do erro
        pos_error = np.linalg.norm(target_pos - current_pos)
        
        # Registo dos dados
        data.append([round(i * 0.1, 2), pos_error])
        
        # Normalização e Predição
        obs_norm = venv.normalize_obs(obs)
        action, _ = model.predict(obs_norm, deterministic=True)
        
        # Passo no ambiente
        obs, _, _, _, _ = env.step(action)
        
        # Visualização
        env.unwrapped.step_sim()
        
        # Delay reduzido para não demorar muito a assistir (pode remover se quiser tempo real)
        time.sleep(0.005) 

    # Salvar CSV
    with open("data_ppo.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "error"])
        writer.writerows(data)
    
    env.close()
    print("Simulação finalizada. Arquivo 'data_ppo.csv' gerado.")

if __name__ == "__main__":
    run_ppo()