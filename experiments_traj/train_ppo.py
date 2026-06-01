import os
import gymnasium as gym
import numpy as np
import math

from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

import bluerov2_gym.envs.bluerov_env as original_env


try:
    register(
        id="BlueRov-v0",
        entry_point="bluerov2_gym.envs.bluerov_env:BlueRov",
        max_episode_steps=2000,
    )
except gym.error.Error:
    pass


class TrajectoryGenerator:
    def __init__(self):
        self.radius = 1.0
        self.speed = 0.15
        self.z_target = -0.5

    def get_state_at_time(self, t):
        t_s = t * self.speed

        x = self.radius * math.sin(t_s)
        y = self.radius * math.sin(t_s) * math.cos(t_s)

        if t < 10.0:
            z = (self.z_target / 10.0) * t
        else:
            z = self.z_target

        vx = self.radius * math.cos(t_s) * self.speed
        vy = self.radius * (math.cos(t_s) ** 2 - math.sin(t_s) ** 2) * self.speed
        vz = 0.0

        yaw = math.atan2(vy, vx)

        return (
            np.array([x, y, z], dtype=np.float32),
            np.array([0.0, 0.0, yaw], dtype=np.float32),
            np.array([vx, vy, vz], dtype=np.float32),
        )


class TrajectoryTrackingEnv(original_env.BlueRov):
    def __init__(self):
        super().__init__(render_mode=None)
        self.traj = TrajectoryGenerator()
        self.current_t = 0.0
        self.dt = 0.1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.current_t = np.random.uniform(0.0, 50.0)

        target_pos, target_att, _ = self.traj.get_state_at_time(self.current_t)

        noise_pos = np.random.uniform(-0.2, 0.2, 3)
        initial_pos = target_pos + noise_pos

        self.state = {
            "x": float(initial_pos[0]),
            "y": float(initial_pos[1]),
            "z": float(initial_pos[2]),
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": float(target_att[2]),
            "u": 0.0,
            "v": 0.0,
            "w": 0.0,
            "p": 0.0,
            "q": 0.0,
            "r": 0.0,
        }

        if hasattr(self, "dynamics"):
            self.dynamics.reset()

        if hasattr(self, "reward_fn"):
            self.reward_fn.reset()

        return self._get_obs(), {}

    def step(self, action):
        self.current_t += self.dt

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -40.0, 40.0)

        obs, _, terminated, truncated, info = super().step(action)

        tgt_pos, tgt_att, tgt_vel = self.traj.get_state_at_time(self.current_t)

        curr_pos = np.array(
            [obs["x"][0], obs["y"][0], obs["z"][0]],
            dtype=np.float32,
        )

        curr_vel = np.array(
            [obs["u"][0], obs["v"][0], obs["w"][0]],
            dtype=np.float32,
        )

        error_pos = curr_pos - tgt_pos
        error_vel = curr_vel - tgt_vel

        psi = obs["yaw"][0]
        c, s = np.cos(psi), np.sin(psi)

        err_x_body = error_pos[0] * c + error_pos[1] * s
        err_y_body = -error_pos[0] * s + error_pos[1] * c
        err_z_body = error_pos[2]

        obs["x"] = np.array([err_x_body], dtype=np.float32)
        obs["y"] = np.array([err_y_body], dtype=np.float32)
        obs["z"] = np.array([err_z_body], dtype=np.float32)

        # Mantido como no original: só substitui u pelo erro de velocidade em x.
        obs["u"] = np.array([error_vel[0]], dtype=np.float32)

        dist = np.linalg.norm(error_pos)
        vel_err = np.linalg.norm(error_vel)

        # Adaptado ao novo action: agora action está em Newtons [-40, 40].
        # Mantém a ideia original de custo de energia, mas normalizado.
        act_cost = np.mean((action / 40.0) ** 2)

        reward = 1.0 - (2.0 * dist) - (0.1 * vel_err) - (0.001 * act_cost)

        if dist > 3.0:
            terminated = True
            reward -= 10.0

        info["tracking_error"] = float(dist)
        info["velocity_error"] = float(vel_err)
        info["act_cost"] = float(act_cost)

        return obs, float(reward), terminated, truncated, info


def train():
    print("[INFO] Iniciando treinamento PPO para Trajetória...")

    os.makedirs("./ppo_traj_tensorboard/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)

    env = DummyVecEnv([lambda: TrajectoryTrackingEnv()])

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        tensorboard_log="./ppo_traj_tensorboard/",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./logs/",
        name_prefix="ppo_traj",
    )

    model.learn(
        total_timesteps=1_000_000,
        callback=checkpoint_callback,
    )

    model.save("ppo_trajectory_final")
    env.save("vec_normalize.pkl")

    print("Treino concluído! Modelos salvos.")


if __name__ == "__main__":
    train()