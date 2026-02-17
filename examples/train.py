import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import os

# ---------------------------------------------------------
# 1. REGISTRO DO AMBIENTE (Garanta que o caminho aponta para sua classe BlueRov)
# ---------------------------------------------------------
# Se você já faz isso no __init__.py do bluerov2_gym, pode remover este bloco.
try:
    register(
        id="BlueRov-v0",
        entry_point="bluerov2_gym.envs:BlueRov", # Ajuste conforme a estrutura de pastas
        max_episode_steps=1000, # Importante para evitar loops infinitos
    )
except gym.error.Error:
    pass # Já registrado

def make_env():
    # Cria o ambiente
    env = gym.make("BlueRov-v0", render_mode="rgb_array") # ou None para treino rápido
    # Monitor é essencial para logar recompensas reais (não normalizadas)
    env = Monitor(env)
    return env

def train_model():
    # Cria pastas de log se não existirem
    os.makedirs("./bluerov_tensorboard/", exist_ok=True)

    # ---------------------------------------------------------
    # 2. CONFIGURAÇÃO DE AMBIENTE VETORIZADO
    # ---------------------------------------------------------
    env = DummyVecEnv([make_env])
    
    # VecNormalize é CRÍTICO para PPO convergir bem em controle contínuo
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # ---------------------------------------------------------
    # 3. CONFIGURAÇÃO DO MODELO (PPO)
    # ---------------------------------------------------------
    model = PPO(
        "MultiInputPolicy",  # OBRIGATÓRIO: Pois seu observation_space é Dict
        env,
        verbose=1,
        tensorboard_log="./bluerov_tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        # MELHORIA: Coeficiente de entropia evita que o robô fique parado cedo demais
        ent_coef=0.01, 
        policy_kwargs=dict(net_arch=[256, 256]) # Rede um pouco maior para lidar com 6-DoF
    )

    print("Iniciando treinamento...")
    model.learn(total_timesteps=1_000_000)
    print("Treinamento finalizado.")

    model.save("bluerov_ppo")
    env.save("bluerov_vec_normalize.pkl") # Salva estatísticas de normalização

def evaluate_model():
    print("Iniciando avaliação...")
    # Recria o ambiente
    env = DummyVecEnv([make_env])
    
    # Carrega as estatísticas de normalização do treino (MUITO IMPORTANTE)
    # Sem isso, o agente vê o mundo com "óculos errados"
    env = VecNormalize.load("bluerov_vec_normalize.pkl", env)
    
    # Desliga atualização de estatísticas e normalização de recompensa para avaliação
    env.training = False
    env.norm_reward = False

    model = PPO.load("bluerov_ppo")

    episodes = 5
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # deterministic=True é padrão para avaliação
            action, _ = model.predict(obs, deterministic=True)
            
            # VecEnv retorna arrays, por isso action já está no formato certo
            obs, reward, dones, info = env.step(action)
            
            # VecEnv retorna reward como array
            total_reward += reward[0]
            
            # VecEnv retorna dones como array de booleans
            done = dones[0] 
            
            # Opcional: Renderizar se quiser ver (ficará lento)
            # env.envs[0].render()

        print(f"Episode {ep+1} reward: {total_reward:.2f}")

if __name__ == "__main__":
    # Treina
    train_model()
    
    # Avalia
    evaluate_model()