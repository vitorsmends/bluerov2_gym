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
MODEL_PATH = os.path.join(MODEL_DIR, "bluerov_ppo")
VECNORM_PATH = os.path.join(MODEL_DIR, "bluerov_vec_normalize.pkl")


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


def train_model():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

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
        name_prefix="bluerov_ppo_checkpoint",
        save_vecnormalize=True,
    )

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

    print("Iniciando treinamento...")
    model.learn(
        total_timesteps=1_000_000,
        callback=checkpoint_callback,
        tb_log_name="PPO_BlueROV2_Thrusters",
    )
    print("Treinamento finalizado.")

    model.save(MODEL_PATH)
    env.save(VECNORM_PATH)
    env.close()

    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"VecNormalize salvo em: {VECNORM_PATH}")


def evaluate_model(render_mode="human", episodes=5):
    print("Iniciando avaliação...")

    env = DummyVecEnv([
        make_env(render_mode=render_mode)
    ])

    env = VecNormalize.load(VECNORM_PATH, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(MODEL_PATH, env=env)

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, dones, infos = env.step(action)

            total_reward += float(reward[0])
            done = bool(dones[0])
            step_count += 1

            # Para debug dos comandos de thruster:
            if step_count % 100 == 0:
                print(f"Ep {ep + 1} | Step {step_count} | Action: {action[0]}")

        print(f"Episode {ep + 1} | reward: {total_reward:.2f} | steps: {step_count}")

    env.close()


if __name__ == "__main__":
    train_model()
    evaluate_model(render_mode="human", episodes=5)