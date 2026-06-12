import os
import gymnasium as gym
import numpy as np

from gymnasium.envs.registration import register

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

ENV_ID = "BlueRov-v0"

LOG_DIR = "./bluerov_tensorboard/"
MODEL_DIR = "./models/"
MODEL_PATH = os.path.join(MODEL_DIR, "bluerov_ppo_curriculum")
VECNORM_PATH = os.path.join(MODEL_DIR, "bluerov_vec_normalize_curriculum.pkl")

try:
    register(
        id=ENV_ID,
        entry_point="bluerov2_gym.envs:BlueRov",
        max_episode_steps=1000,
    )
except gym.error.Error:
    pass


def make_env(render_mode=None):
    def _init():
        env = gym.make(ENV_ID, render_mode=render_mode)
        env = Monitor(env)
        return env

    return _init


def train_model_with_curriculum():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Inicialização do ambiente idêntica ao seu script antigo
    env = DummyVecEnv([
        make_env(render_mode=None)
    ])

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=MODEL_DIR,
        name_prefix="bluerov_ppo_curriculum_checkpoint",
        save_vecnormalize=True,
    )

    # Instanciação do PPO mantendo rigorosamente seus hiperparâmetros originais
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                vf=[256, 256],
            )
        ),
    )

    # Define a progressão do Curriculum Learning baseada na severidade do JONSWAP
    lessons = [
        {"timesteps": 200_000, "Hs": 0.0, "max_current": 0.0, "scale": 0.0, "wave_dir": [1.0, 0.0]},
        {"timesteps": 300_000, "Hs": 0.5, "max_current": 0.2, "scale": 0.2, "wave_dir": [1.0, 0.0]},
        {"timesteps": 300_000, "Hs": 1.2, "max_current": 0.4, "scale": 0.4, "wave_dir": [1.0, 0.0]},
        {"timesteps": 200_000, "Hs": 2.0, "max_current": 0.7, "scale": 0.5, "wave_dir": [1.0, 0.0]},
    ]

    print("Iniciando treinamento estruturado por currículo...")
    
    is_first_iteration = True
    for index, lesson in enumerate(lessons):
        print(f"\n[CURRICULUM] Transição de Fase | Iniciando Etapa {index + 1} de {len(lessons)}")
        print(f"[CURRICULUM] Parâmetros de distúrbio -> Hs: {lesson['Hs']}m | Corrente Máxima: {lesson['max_current']}m/s")

        # Garante a atualização dos parâmetros injetando o dicionário na propriedade correspondente do simulador
        base_env = env.envs[0].unwrapped
        base_env.jonswap_params = {
            "Hs": lesson["Hs"],
            "Tp": 12.0,
            "gamma": 3.3,
            "N": 64,
            "wave_dir": tuple(lesson["wave_dir"]),
            "scale": lesson["scale"],
            "max_current": lesson["max_current"],
            "seed": 42
        }
        
        # Sincroniza a atualização forçando o reset da instância de dinâmica associada
        if hasattr(base_env.dynamics, "set_jonswap_params"):
            base_env.dynamics.set_jonswap_params(**base_env.jonswap_params)
        else:
            base_env.dynamics.reset(jonswap_params=base_env.jonswap_params)

        # Executa o aprendizado cumulativo sem zerar as contagens globais das métricas do otimizador
        model.learn(
            total_timesteps=lesson["timesteps"],
            callback=checkpoint_callback,
            tb_log_name="PPO_BlueROV2_Curriculum",
            reset_num_timesteps=is_first_iteration
        )
        is_first_iteration = False

    print("Treinamento por currículo finalizado com sucesso.")

    model.save(MODEL_PATH)
    env.save(VECNORM_PATH)
    env.close()

    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"VecNormalize salvo em: {VECNORM_PATH}")


if __name__ == "__main__":
    train_model_with_curriculum()