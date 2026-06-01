import os
import gymnasium as gym
import numpy as np
import math

from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

import bluerov2_gym.envs.bluerov_env as original_env


ENV_ID = "BlueRov-v0"

try:
    register(
        id=ENV_ID,
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
        ts = t * self.speed

        x = self.radius * math.sin(ts)
        y = self.radius * math.sin(ts) * math.cos(ts)

        if t < 10.0:
            z = (self.z_target / 10.0) * t
            vz = self.z_target / 10.0
        else:
            z = self.z_target
            vz = 0.0

        vx = self.radius * math.cos(ts) * self.speed
        vy = self.radius * (math.cos(ts) ** 2 - math.sin(ts) ** 2) * self.speed

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

    def _scalar(self, value):
        return float(np.asarray(value).reshape(-1)[0])

    def _wrap_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _error_obs(self, obs):
        target_pos, target_att, target_vel = self.traj.get_state_at_time(self.current_t)

        curr_pos = np.array([
            self._scalar(obs["x"]),
            self._scalar(obs["y"]),
            self._scalar(obs["z"]),
        ], dtype=np.float32)

        curr_vel = np.array([
            self._scalar(obs["u"]),
            self._scalar(obs["v"]),
            self._scalar(obs["w"]),
        ], dtype=np.float32)

        roll = self._scalar(obs["roll"])
        pitch = self._scalar(obs["pitch"])
        yaw = self._scalar(obs["yaw"])

        error_pos = curr_pos - target_pos
        error_vel = curr_vel - target_vel
        yaw_error = self._wrap_angle(yaw - target_att[2])

        psi_ref = float(target_att[2])
        c = math.cos(psi_ref)
        s = math.sin(psi_ref)

        err_x_body = error_pos[0] * c + error_pos[1] * s
        err_y_body = -error_pos[0] * s + error_pos[1] * c
        err_z_body = error_pos[2]

        vel_x_body = error_vel[0] * c + error_vel[1] * s
        vel_y_body = -error_vel[0] * s + error_vel[1] * c
        vel_z_body = error_vel[2]

        obs["x"] = np.array([err_x_body], dtype=np.float32)
        obs["y"] = np.array([err_y_body], dtype=np.float32)
        obs["z"] = np.array([err_z_body], dtype=np.float32)

        obs["u"] = np.array([vel_x_body], dtype=np.float32)
        obs["v"] = np.array([vel_y_body], dtype=np.float32)
        obs["w"] = np.array([vel_z_body], dtype=np.float32)

        obs["yaw"] = np.array([yaw_error], dtype=np.float32)

        return obs, error_pos, error_vel, yaw_error, roll, pitch

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        rng = np.random.default_rng(seed)

        self.current_t = rng.uniform(0.0, 50.0)

        target_pos, target_att, _ = self.traj.get_state_at_time(self.current_t)

        noise_pos = rng.uniform(-0.2, 0.2, 3)
        noise_yaw = rng.uniform(-0.2, 0.2)

        initial_pos = target_pos + noise_pos

        self.state = {
            "x": float(initial_pos[0]),
            "y": float(initial_pos[1]),
            "z": float(initial_pos[2]),
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": float(target_att[2] + noise_yaw),
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

        obs = self._get_obs()
        obs, *_ = self._error_obs(obs)

        return obs, {}

    def step(self, action):
        self.current_t += self.dt

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -40.0, 40.0)

        obs, _, terminated, truncated, info = super().step(action)

        obs, error_pos, error_vel, yaw_error, roll, pitch = self._error_obs(obs)

        dist = float(np.linalg.norm(error_pos))
        vel_err = float(np.linalg.norm(error_vel))
        stability_penalty = abs(roll) + abs(pitch)

        # Original reward strategy adapted to trajectory error:
        # position + velocity + yaw + roll/pitch + success bonus.
        reward = -(
            1.0 * dist
            + 0.1 * vel_err
            + 0.5 * abs(yaw_error)
            + 0.5 * stability_penalty
        )

        if dist < 0.20:
            reward += 1.0

        info["done_reason"] = "none"
        info["is_success"] = bool(dist < 0.20)

        if dist > 5.0:
            terminated = True
            reward -= 10.0
            info["done_reason"] = "distance"

        if abs(roll) > 1.2 or abs(pitch) > 1.2:
            terminated = True
            reward -= 10.0
            info["done_reason"] = "attitude"

        info["tracking_error"] = dist
        info["velocity_error"] = vel_err
        info["yaw_error"] = float(yaw_error)

        return obs, float(reward), terminated, truncated, info


def make_env():
    def _init():
        env = TrajectoryTrackingEnv()
        env = Monitor(env)
        return env

    return _init


def train():
    os.makedirs("./ppo_traj_tensorboard/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./models/", exist_ok=True)

    env = DummyVecEnv([make_env()])

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./ppo_traj_tensorboard/",
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                vf=[256, 256],
            )
        ),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./logs/",
        name_prefix="ppo_traj",
        save_vecnormalize=True,
    )

    model.learn(
        total_timesteps=1_000_000,
        callback=checkpoint_callback,
        tb_log_name="PPO_BlueROV2_Trajectory",
    )

    model.save("./models/ppo_trajectory_final")
    env.save("./models/vec_normalize.pkl")
    env.close()

    print("Treino concluído.")
    print("Modelo salvo em ./models/ppo_trajectory_final")
    print("VecNormalize salvo em ./models/vec_normalize.pkl")


if __name__ == "__main__":
    train()